from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple, Self

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import LiteralValue, Metadata, PyEmbeddings, Where
from chromadb.base_types import InclusionExclusionOperator
from chromadb.config import Settings
from loguru import logger
from pydantic import PositiveInt, validate_call

from orchestrator.models.embeddings import Embedding
from orchestrator.rag.schemas import Chunk, RetrievalResult

# Chunk fields stored alongside user metadata so a Chunk can be rebuilt from a query hit.
_DOC_ID, _INDEX, _VERSION = "_document_id", "_index", "_version"
_INTERNAL = (_DOC_ID, _INDEX, _VERSION)
# Records of the system collection; it needs a vector per record, so each gets this placeholder.
_EMBEDDING_ID, _POINTER = "embedding", "document"
_PLACEHOLDER: PyEmbeddings = [[1.0]]


class EmbeddingMismatchError(RuntimeError):
    """The store holds vectors from a different embedding model or dimension."""


class ActiveVersion(NamedTuple):
    version: str
    chunks: int


class VectorStore:
    """ChromaDB collection holding chunks; pass EphemeralClient for tests, or use `persistent()`.

    Chunks of every version are stored side by side. A companion "system" collection holds one
    pointer record per document naming its active version, plus the embedding model the vectors
    came from. Chroma has no multi-record transactions, but a single upsert is atomic, so
    switching a pointer is how a new version becomes visible all at once.
    """

    def __init__(self, client: ClientAPI, collection: str = "documents") -> None:
        self.client = client
        self._collection = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        self._system = client.get_or_create_collection(f"{collection}__system", embedding_function=None)

    @classmethod
    def persistent(cls, path: str | Path, collection: str = "documents") -> Self:
        settings = Settings(anonymized_telemetry=False)
        return cls(chromadb.PersistentClient(path=str(path), settings=settings), collection)

    def count(self) -> int:
        return self._collection.count()

    def check_embedding(self, model: str, dimension: int) -> None:
        """Record the embedding model on first use; afterwards refuse any other model or dimension."""
        stored = self._system.get(ids=[_EMBEDDING_ID], include=["metadatas"])["metadatas"]
        if stored:
            found = (stored[0].get("model"), stored[0].get("dimension"))
            if found != (model, dimension):
                raise EmbeddingMismatchError(
                    f"store holds {found[0]}/{found[1]} vectors, configured embedder is {model}/{dimension}"
                )
            return
        if self.count():
            logger.warning("existing store has no embedding record; assuming it matches", model=model)
        self._system.upsert(
            ids=[_EMBEDDING_ID], embeddings=_PLACEHOLDER, metadatas=[{"model": model, "dimension": dimension}]
        )

    def active_version(self, document_id: str) -> ActiveVersion | None:
        res = self._system.get(ids=[_pointer_id(document_id)], include=["metadatas"])["metadatas"]
        if not res:
            return None
        return ActiveVersion(str(res[0]["version"]), int(res[0]["chunks"]))  # type: ignore[arg-type]

    def active_versions(self) -> dict[str, str]:
        """Active version of every document, read in one call so a reader sees one consistent snapshot."""
        res = self._system.get(where={"kind": _POINTER}, include=["metadatas"])["metadatas"] or []
        return {str(m["document_id"]): str(m["version"]) for m in res}

    def activate(self, document_id: str, version: str, chunks: int) -> None:
        self._system.upsert(
            ids=[_pointer_id(document_id)],
            embeddings=_PLACEHOLDER,
            metadatas=[{"kind": _POINTER, "document_id": document_id, "version": version, "chunks": chunks}],
        )

    def delete_stale(self, document_id: str, keep_version: str) -> None:
        """Drop every chunk of the document outside keep_version, including unversioned legacy chunks."""
        res = self._collection.get(where={_DOC_ID: document_id}, include=["metadatas"])
        metadatas = res["metadatas"] or []
        stale = [i for i, meta in zip(res["ids"], metadatas, strict=True) if meta.get(_VERSION) != keep_version]
        if stale:
            self._collection.delete(ids=stale)

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding], version: str | None = None) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        tag = {} if version is None else {_VERSION: version}
        vectors: PyEmbeddings = [e.vector for e in embeddings]
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=vectors,
            metadatas=[{**c.metadata, _DOC_ID: c.document_id, _INDEX: c.index, **tag} for c in chunks],
        )

    def all_chunks(self, versions: Sequence[str] | None = None) -> list[Chunk]:
        """Every stored chunk, or only those of the given versions."""
        if versions is not None and not versions:
            return []
        res = self._collection.get(where=_in_versions(versions), include=["documents", "metadatas"])
        documents, metadatas = res["documents"], res["metadatas"]
        if documents is None or metadatas is None:
            raise RuntimeError("Chroma get omitted documents or metadatas")
        return [_chunk(*hit) for hit in zip(res["ids"], documents, metadatas, strict=True)]

    @validate_call(config={"arbitrary_types_allowed": True})
    def search(
        self, embedding: Embedding, k: PositiveInt, versions: Sequence[str] | None = None
    ) -> list[RetrievalResult]:
        """Nearest chunks, optionally restricted to the given versions."""
        if self.count() == 0 or (versions is not None and not versions):
            return []
        query: PyEmbeddings = [embedding.vector]
        res = self._collection.query(
            query_embeddings=query,
            n_results=min(k, self.count()),
            where=_in_versions(versions),
            include=["documents", "metadatas", "distances"],
        )
        documents, metadatas, distances = res["documents"], res["metadatas"], res["distances"]
        # Optional in Chroma's types because `include` controls them; we always request all three.
        if documents is None or metadatas is None or distances is None:
            raise RuntimeError("Chroma query omitted documents, metadatas or distances")
        results = []
        for chunk_id, text, meta, distance in zip(
            res["ids"][0], documents[0], metadatas[0], distances[0], strict=True
        ):
            # Cosine distance is in [0, 2]; map to a similarity in [0, 1], clamping float noise.
            score = min(max(1.0 - distance / 2.0, 0.0), 1.0)
            results.append(RetrievalResult(chunk=_chunk(chunk_id, text, meta), score=score))
        return results


def _pointer_id(document_id: str) -> str:
    return f"{_POINTER}:{document_id}"


def _in_versions(versions: Sequence[str] | None) -> Where | None:
    if versions is None:
        return None
    values: list[LiteralValue] = list(versions)
    members: dict[InclusionExclusionOperator, list[LiteralValue]] = {"$in": values}
    return {_VERSION: members}


def _chunk(chunk_id: str, text: str, meta: Metadata) -> Chunk:
    user_meta = {key: value for key, value in meta.items() if key not in _INTERNAL}
    # Rebuilt from stored data, so validate rather than trust the metadata types.
    return Chunk.model_validate(
        {"id": chunk_id, "document_id": meta[_DOC_ID], "index": meta[_INDEX], "text": text, "metadata": user_meta}
    )

from collections.abc import Sequence
from pathlib import Path
from typing import Self

import chromadb
from chromadb.api import ClientAPI
from chromadb.api.types import LiteralValue, Metadata, PyEmbeddings, Where
from chromadb.base_types import InclusionExclusionOperator
from chromadb.config import Settings
from loguru import logger
from pydantic import PositiveInt, validate_call

from orchestrator.models.embeddings import Embedding
from orchestrator.rag.metadata_db import ActiveVersion, MetadataDB
from orchestrator.rag.schemas import Chunk, RetrievalResult

__all__ = ["ActiveVersion", "EmbeddingMismatchError", "VectorStore"]

# Chunk fields stored alongside user metadata so a Chunk can be rebuilt from a query hit.
_DOC_ID, _INDEX, _VERSION = "_document_id", "_index", "_version"
_INTERNAL = (_DOC_ID, _INDEX, _VERSION)
# The SQLite file holding pointers and the embedding stamp, beside Chroma's files on the same volume.
METADATA_DB_NAME = "system.db"


class EmbeddingMismatchError(RuntimeError):
    """The store holds vectors from a different embedding model or dimension."""


class VectorStore:
    """Chunks in one ChromaDB collection; versions and the embedding stamp in a MetadataDB.

    Chunks of every version are stored side by side. Each document has one pointer naming its
    active version, flipped in a single SQLite transaction, so a new version becomes visible all at
    once. Pass EphemeralClient plus an in-memory MetadataDB for tests, or use `persistent()`.
    """

    def __init__(self, client: ClientAPI, collection: str = "documents", *, metadata: MetadataDB) -> None:
        self.client = client
        self.collection_name = collection
        self._collection = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )
        self._metadata = metadata

    @classmethod
    def persistent(cls, path: str | Path, collection: str = "documents") -> Self:
        settings = Settings(anonymized_telemetry=False)
        client = chromadb.PersistentClient(path=str(path), settings=settings)
        return cls(client, collection, metadata=MetadataDB.open(Path(path) / METADATA_DB_NAME))

    def count(self) -> int:
        return self._collection.count()

    def check_embedding(self, model: str, dimension: int) -> None:
        """Record the embedding model on first use; afterwards refuse any other model or dimension."""
        found = self._metadata.embedding_model()
        if found is not None:
            if found != (model, dimension):
                raise EmbeddingMismatchError(
                    f"store holds {found[0]}/{found[1]} vectors, configured embedder is {model}/{dimension}"
                )
            return
        if self.count():
            logger.warning("existing store has no embedding record; assuming it matches", model=model)
        self._metadata.record_embedding_model(model, dimension)

    def active_version(self, document_id: str) -> ActiveVersion | None:
        return self._metadata.active_version(document_id)

    def active_versions(self) -> dict[str, str]:
        """Active version of every document, read in one query so a reader sees one consistent snapshot."""
        return self._metadata.active_versions()

    def activate(self, document_id: str, version: str, chunks: int) -> None:
        self._metadata.activate(document_id, version, chunks)

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

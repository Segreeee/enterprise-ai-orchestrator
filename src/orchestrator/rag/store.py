from collections.abc import Sequence
from pathlib import Path
from typing import Self

import chromadb
from chromadb.api import ClientAPI
from pydantic import PositiveInt, validate_call

from orchestrator.models.embeddings import Embedding
from orchestrator.rag.schemas import Chunk, RetrievalResult

# Chunk fields stored alongside user metadata so a Chunk can be rebuilt from a query hit.
_DOC_ID, _INDEX = "_document_id", "_index"


class VectorStore:
    """ChromaDB collection holding chunks; pass EphemeralClient for tests, or use `persistent()`."""

    def __init__(self, client: ClientAPI, collection: str = "documents") -> None:
        self._collection = client.get_or_create_collection(
            collection, metadata={"hnsw:space": "cosine"}, embedding_function=None
        )

    @classmethod
    def persistent(cls, path: str | Path, collection: str = "documents") -> Self:
        return cls(chromadb.PersistentClient(path=str(path)), collection)

    def count(self) -> int:
        return self._collection.count()

    def delete_document(self, document_id: str) -> None:
        self._collection.delete(where={_DOC_ID: document_id})

    def add(self, chunks: Sequence[Chunk], embeddings: Sequence[Embedding]) -> None:
        if len(chunks) != len(embeddings):
            raise ValueError("chunks and embeddings must have the same length")
        if not chunks:
            return
        self._collection.upsert(
            ids=[c.id for c in chunks],
            documents=[c.text for c in chunks],
            embeddings=[e.vector for e in embeddings],
            metadatas=[{**c.metadata, _DOC_ID: c.document_id, _INDEX: c.index} for c in chunks],
        )

    @validate_call(config={"arbitrary_types_allowed": True})
    def search(self, embedding: Embedding, k: PositiveInt) -> list[RetrievalResult]:
        if self.count() == 0:
            return []
        res = self._collection.query(
            query_embeddings=[embedding.vector],
            n_results=min(k, self.count()),
            include=["documents", "metadatas", "distances"],
        )
        results = []
        for chunk_id, text, meta, distance in zip(
            res["ids"][0], res["documents"][0], res["metadatas"][0], res["distances"][0], strict=True
        ):
            user_meta = {key: value for key, value in meta.items() if key not in (_DOC_ID, _INDEX)}
            chunk = Chunk(id=chunk_id, document_id=meta[_DOC_ID], index=meta[_INDEX], text=text, metadata=user_meta)
            # Cosine distance is in [0, 2]; map to a similarity in [0, 1], clamping float noise.
            score = min(max(1.0 - distance / 2.0, 0.0), 1.0)
            results.append(RetrievalResult(chunk=chunk, score=score))
        return results

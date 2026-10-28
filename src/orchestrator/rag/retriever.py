from orchestrator.models.embeddings import Embedder
from orchestrator.rag.chunker import Chunker
from orchestrator.rag.schemas import Document, RetrievalResult
from orchestrator.rag.store import VectorStore


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, chunker: Chunker | None = None) -> None:
        self._embedder = embedder
        self._store = store
        self._chunker = chunker or Chunker()

    def ingest(self, document: Document) -> int:
        """Chunk, embed and store a document; re-ingesting the same id replaces all its chunks."""
        chunks = self._chunker.chunk(document)
        # Embed before deleting so a failing embedder leaves the old version intact.
        embeddings = [self._embedder.embed(c.text) for c in chunks]
        self._store.delete_document(document.id)
        self._store.add(chunks, embeddings)
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[RetrievalResult]:
        return self._store.search(self._embedder.embed(query), k=k)

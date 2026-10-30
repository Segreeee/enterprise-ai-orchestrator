import re

from loguru import logger
from rank_bm25 import BM25Okapi

from orchestrator.models.embeddings import Embedder
from orchestrator.rag.chunker import Chunker
from orchestrator.rag.schemas import Chunk, Document, RetrievalResult
from orchestrator.rag.store import VectorStore

# Reciprocal Rank Fusion: score = sum over rankings of 1 / (RRF_K + rank); 60 is the value from the RRF paper.
RRF_K = 60
# Each search contributes its top POOL_FACTOR * k, so a chunk ranked just below k in both lists can still win.
POOL_FACTOR = 4


class Retriever:
    def __init__(self, embedder: Embedder, store: VectorStore, chunker: Chunker | None = None) -> None:
        self._embedder = embedder
        self._store = store
        self._chunker = chunker or Chunker()

    def ingest(self, document: Document) -> int:
        """Chunk, embed and store a document; re-ingesting the same id replaces all its chunks."""
        chunks = self._chunker.chunk(document)
        embeddings = [self._embedder.embed(c.text) for c in chunks]
        # Write first: upsert overwrites chunks "{id}:0..n-1" in place. Only after it succeeds
        # drop the old version's surplus chunks, so a failed embed or write keeps the old version.
        self._store.add(chunks, embeddings)
        self._store.delete_document(document.id, from_index=len(chunks))
        return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[RetrievalResult]:
        """Hybrid search: vector and BM25 rankings fused with RRF; the score is the RRF score."""
        pool = POOL_FACTOR * k
        rankings = [
            [r.chunk for r in self._store.search(self._embedder.embed(query), k=pool)],
            _bm25_ranking(query, self._store.all_chunks())[:pool],
        ]
        logger.debug("hybrid retrieval", k=k, vector_hits=len(rankings[0]), lexical_hits=len(rankings[1]))
        fused: dict[str, float] = {}
        chunks: dict[str, Chunk] = {}
        for ranking in rankings:
            for rank, chunk in enumerate(ranking, start=1):
                fused[chunk.id] = fused.get(chunk.id, 0.0) + 1 / (RRF_K + rank)
                chunks[chunk.id] = chunk
        best = sorted(fused, key=fused.__getitem__, reverse=True)[:k]
        return [RetrievalResult(chunk=chunks[i], score=fused[i]) for i in best]


def _tokens(text: str) -> list[str]:
    return re.findall(r"\w+", text.lower())


def _bm25_ranking(query: str, corpus: list[Chunk]) -> list[Chunk]:
    """Chunks sharing at least one token with the query, best BM25 score first."""
    # ponytail: the index is rebuilt from the store on every query (O(corpus)), which keeps it
    # consistent with concurrent ingests and other processes; cache it per store version if corpora grow large.
    terms = set(_tokens(query))
    docs = [_tokens(c.text) for c in corpus]
    # BM25 idf is <= 0 for terms in half the corpus or more, so a zero score does not mean "no match":
    # matching is decided by token overlap, BM25 only orders the matches.
    matches = [i for i, doc in enumerate(docs) if terms.intersection(doc)]
    if not matches:
        return []
    scores = BM25Okapi(docs).get_scores(list(terms))
    return [corpus[i] for i in sorted(matches, key=lambda i: scores[i], reverse=True)]

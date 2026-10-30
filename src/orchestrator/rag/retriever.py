import hashlib
import json
import re
import threading
from collections import defaultdict

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
        # Fail fast: vectors from two embedding models in one collection would rank meaninglessly.
        store.check_embedding(embedder.model, embedder.dimension)
        self._embedder = embedder
        self._store = store
        self._chunker = chunker or Chunker()
        # ponytail: in-process locks, one per document id, never evicted; with several API
        # processes on one store, replace them with a shared lock (file or database).
        self._locks: defaultdict[str, threading.Lock] = defaultdict(threading.Lock)
        self._locks_guard = threading.Lock()

    def ingest(self, document: Document) -> int:
        """Chunk, embed and store a new version of the document, then make it the active one.

        Readers only see active versions, so they get the old version or the new one, never a mix.
        A failure before activation leaves the old version active; the orphaned chunks are
        invisible and removed by the next successful ingest of the same document.
        """
        with self._lock(document.id):
            version = self._version(document)
            active = self._store.active_version(document.id)
            if active is not None and active.version == version:
                return active.chunks
            chunks = [
                c.model_copy(update={"id": f"{document.id}:{version}:{c.index}"})
                for c in self._chunker.chunk(document)
            ]
            embeddings = [self._embedder.embed(c.text) for c in chunks]
            self._store.add(chunks, embeddings, version=version)
            self._store.activate(document.id, version, len(chunks))
            self._store.delete_stale(document.id, keep_version=version)
            return len(chunks)

    def retrieve(self, query: str, k: int = 4) -> list[RetrievalResult]:
        """Hybrid search: vector and BM25 rankings fused with RRF; the score is the RRF score."""
        embedding = self._embedder.embed(query)
        # One snapshot of the active versions for both searches, so they cannot disagree on a document.
        versions = list(self._store.active_versions().values())
        pool = POOL_FACTOR * k
        rankings = [
            [r.chunk for r in self._store.search(embedding, k=pool, versions=versions)],
            _bm25_ranking(query, self._store.all_chunks(versions))[:pool],
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

    def _lock(self, document_id: str) -> threading.Lock:
        with self._locks_guard:
            return self._locks[document_id]

    def _version(self, document: Document) -> str:
        """Content hash: the same text, metadata, chunking and embedding model give the same version."""
        payload = {
            "text": document.text,
            "metadata": document.metadata,
            "chunker": self._chunker.model_dump(),
            "embedding": [self._embedder.model, self._embedder.dimension],
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:16]


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

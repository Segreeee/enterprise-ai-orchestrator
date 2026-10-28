from uuid import uuid4

import chromadb
import pytest

from orchestrator.models.embeddings import Embedding
from orchestrator.rag.store import VectorStore

KEYWORDS = ("cat", "dog", "car")


class KeywordEmbedder:
    """Deterministic stub: one axis per keyword, so similarity is predictable."""

    dimension = len(KEYWORDS)

    def __init__(self) -> None:
        self.calls: list[str] = []

    def embed(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError("cannot embed blank text")
        self.calls.append(text)
        lowered = text.lower()
        # +0.01 keeps the vector non-zero so cosine distance is always defined.
        return Embedding(vector=[lowered.count(k) + 0.01 for k in KEYWORDS], dimension=self.dimension)


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def store() -> VectorStore:
    # EphemeralClient instances share in-process state, so isolate by collection name.
    return VectorStore(chromadb.EphemeralClient(), collection=f"test-{uuid4().hex}")

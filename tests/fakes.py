"""Deterministic test doubles shared across the suite (fixtures live in conftest.py)."""

from collections.abc import Sequence

from pydantic import BaseModel

from orchestrator.agents.schemas import Message
from orchestrator.models.embeddings import Embedding
from orchestrator.rag.schemas import Chunk, RetrievalResult

KEYWORDS = ("cat", "dog", "car")


class ScriptedLLM:
    """LLMClient that returns replies in order and records every prompt it receives."""

    def __init__(self, replies: Sequence[str] | None = None) -> None:
        self._replies = list(replies or [])
        self.calls: list[list[Message]] = []

    def generate(self, messages: Sequence[Message], response_format: type[BaseModel] | None = None) -> Message:
        self.calls.append(list(messages))
        n = len(self.calls)
        content = self._replies[n - 1] if n <= len(self._replies) else f"reply {n}"
        return Message(role="assistant", content=content)


class FakeRetriever:
    def __init__(self, texts: Sequence[str] = ()) -> None:
        self._results = [
            RetrievalResult(chunk=Chunk(id=f"doc:{i}", document_id="doc", index=i, text=t), score=0.9)
            for i, t in enumerate(texts)
        ]
        self.queries: list[tuple[str, int]] = []

    def retrieve(self, query: str, k: int = 4) -> list[RetrievalResult]:
        self.queries.append((query, k))
        return self._results[:k]


class KeywordEmbedder:
    """Embedder with one axis per keyword, so similarity ranking is predictable."""

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

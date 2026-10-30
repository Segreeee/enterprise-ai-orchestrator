from typing import Protocol, Self, runtime_checkable

import ollama
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, PositiveInt, model_validator

from orchestrator.models import DEFAULT_TIMEOUT


class Embedding(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    vector: list[FiniteFloat] = Field(min_length=1)
    dimension: PositiveInt

    @model_validator(mode="after")
    def _check_dimension(self) -> Self:
        if len(self.vector) != self.dimension:
            raise ValueError(f"expected dimension {self.dimension}, got {len(self.vector)}")
        return self


@runtime_checkable
class Embedder(Protocol):
    dimension: int

    def embed(self, text: str) -> Embedding: ...


class OllamaEmbedder:
    # ponytail: one request per text; switch to batched `input=[...]` if ingest gets slow.
    def __init__(
        self,
        model: str = "nomic-embed-text",
        dimension: int = 768,
        host: str = "http://localhost:11434",
        client: ollama.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self.dimension = dimension
        self._client = client or ollama.Client(host=host, timeout=timeout)

    def embed(self, text: str) -> Embedding:
        if not text.strip():
            raise ValueError("cannot embed blank text")
        response = self._client.embed(model=self._model, input=text)
        return Embedding.model_validate({"vector": response.embeddings[0], "dimension": self.dimension})

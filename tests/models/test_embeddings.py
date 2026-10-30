import math
from types import SimpleNamespace

import httpx
import pytest
from pydantic import ValidationError

from orchestrator.models import DEFAULT_TIMEOUT
from orchestrator.models.embeddings import Embedder, Embedding, OllamaEmbedder

DIM = 768


class StubOllama:
    def __init__(self, vector: list[float]):
        self.vector = vector
        self.calls: list[dict] = []

    def embed(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(embeddings=[self.vector])


def test_embedding_accepts_vector_of_declared_dimension():
    emb = Embedding(vector=[0.1] * DIM, dimension=DIM)
    assert len(emb.vector) == DIM


@pytest.mark.parametrize("size", [DIM - 1, DIM + 1])
def test_embedding_rejects_wrong_dimension(size):
    with pytest.raises(ValidationError, match="dimension"):
        Embedding(vector=[0.1] * size, dimension=DIM)


@pytest.mark.parametrize("bad", [math.nan, math.inf])
def test_embedding_rejects_non_finite_values(bad):
    with pytest.raises(ValidationError):
        Embedding(vector=[bad] * DIM, dimension=DIM)


def test_embedding_rejects_empty_vector_and_bad_dimension():
    with pytest.raises(ValidationError):
        Embedding(vector=[], dimension=0)


def test_ollama_embedder_returns_validated_embedding():
    stub = StubOllama([0.5] * DIM)
    embedder = OllamaEmbedder(model="nomic-embed-text", dimension=DIM, client=stub)

    emb = embedder.embed("hello world")

    assert isinstance(embedder, Embedder)
    assert emb.vector == [0.5] * DIM
    assert stub.calls == [{"model": "nomic-embed-text", "input": "hello world"}]


def test_ollama_embedder_rejects_dimension_mismatch_from_model():
    embedder = OllamaEmbedder(model="m", dimension=DIM, client=StubOllama([0.5] * 384))
    with pytest.raises(ValidationError, match="dimension"):
        embedder.embed("hello")


@pytest.mark.parametrize("text", ["", "   "])
def test_ollama_embedder_rejects_blank_text(text):
    stub = StubOllama([0.5] * DIM)
    with pytest.raises(ValueError, match="blank"):
        OllamaEmbedder(model="m", dimension=DIM, client=stub).embed(text)
    assert stub.calls == []


def test_ollama_embedder_times_out_on_hung_backend(silent_server):
    embedder = OllamaEmbedder(model="m", dimension=DIM, host=silent_server, timeout=0.2)
    with pytest.raises(httpx.TimeoutException):
        embedder.embed("cat")


def test_ollama_embedder_has_a_finite_default_timeout():
    timeout = OllamaEmbedder()._client._client.timeout
    assert timeout == httpx.Timeout(DEFAULT_TIMEOUT)
    assert timeout.read is not None and 0 < timeout.read <= 120

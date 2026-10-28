"""Hits a real Ollama server; skipped when it is unreachable or the model is missing."""

import pytest

from orchestrator.agents.schemas import Message
from orchestrator.config import Settings

ollama = pytest.importorskip("ollama")

pytestmark = pytest.mark.integration

EMBED_MODEL = "nomic-embed-text"
EMBED_DIM = 768


@pytest.fixture(scope="module")
def settings() -> Settings:
    return Settings.from_env()


@pytest.fixture(scope="module")
def available_models(settings) -> set[str]:
    try:
        listing = ollama.Client(host=str(settings.ollama_host)).list()
    except Exception as exc:  # connection refused, DNS, timeout...
        pytest.skip(f"Ollama not reachable at {settings.ollama_host}: {exc}")
    return {m.model.split(":")[0] for m in listing.models}


def require(model: str, available: set[str]) -> None:
    if model not in available:
        pytest.skip(f"model '{model}' not pulled (ollama pull {model})")


def test_real_chat_round_trip(settings, available_models):
    from orchestrator.models.llm import OllamaClient

    require(settings.model_name, available_models)
    client = OllamaClient(model=settings.model_name, host=str(settings.ollama_host))
    reply = client.generate([Message(role="user", content="Reply with the single word: pong")])
    assert reply.role == "assistant"


def test_real_embedding_has_expected_dimension(settings, available_models):
    from orchestrator.models.embeddings import OllamaEmbedder

    require(EMBED_MODEL, available_models)
    embedder = OllamaEmbedder(model=EMBED_MODEL, dimension=EMBED_DIM, host=str(settings.ollama_host))
    assert len(embedder.embed("hello world").vector) == EMBED_DIM

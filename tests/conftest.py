from uuid import uuid4

import chromadb
import pytest

from orchestrator.rag.store import VectorStore
from tests.fakes import KeywordEmbedder, ScriptedLLM


@pytest.fixture
def embedder() -> KeywordEmbedder:
    return KeywordEmbedder()


@pytest.fixture
def store() -> VectorStore:
    # EphemeralClient instances share in-process state, so isolate by collection name.
    return VectorStore(chromadb.EphemeralClient(), collection=f"test-{uuid4().hex}")


@pytest.fixture
def llm() -> ScriptedLLM:
    return ScriptedLLM()

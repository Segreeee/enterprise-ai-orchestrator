import socket
import threading
from collections.abc import Iterator
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


@pytest.fixture
def silent_server() -> Iterator[str]:
    """A loopback server that accepts connections but never answers: a hung backend.

    It hangs up after a few seconds, so a client without a timeout fails the test instead of hanging it.
    """
    with socket.create_server(("127.0.0.1", 0)) as server:
        hang_up = threading.Timer(3.0, server.close)
        hang_up.start()
        yield f"http://127.0.0.1:{server.getsockname()[1]}"
        hang_up.cancel()

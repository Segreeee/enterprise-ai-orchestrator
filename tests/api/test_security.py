"""HTTP-level guards: body size, Host allowlist, docs exposure and the shipped port binding."""

import re
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from orchestrator.api.app import MAX_BODY_BYTES, create_app
from orchestrator.api.dependencies import get_llm, get_retriever
from orchestrator.config import Settings
from orchestrator.rag.chunker import Chunker
from orchestrator.rag.retriever import Retriever

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def retriever(embedder, store) -> Retriever:
    return Retriever(embedder=embedder, store=store, chunker=Chunker(chunk_size=20, chunk_overlap=0))


def make_client(retriever, llm, settings: Settings | None = None, host: str = "localhost") -> TestClient:
    app = create_app(settings or Settings())
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app, base_url=f"http://{host}")


@pytest.fixture
def client(retriever, llm) -> TestClient:
    return make_client(retriever, llm)


JSON = {"content-type": "application/json"}


# --- raw body size -----------------------------------------------------------------------


def test_body_over_the_limit_is_rejected_with_413_before_parsing(client, embedder, store):
    body = b'{"id":"big","text":"' + b"a" * MAX_BODY_BYTES + b'"}'

    response = client.post("/ingest", content=body, headers=JSON)

    assert response.status_code == 413
    assert len(response.content) < 200  # the oversized input is not echoed back
    assert embedder.calls == [] and store.count() == 0


def test_streamed_body_over_the_limit_is_rejected_without_content_length(client, embedder):
    def chunks():
        yield b'{"id":"big","text":"'
        for _ in range(MAX_BODY_BYTES // 65_536 + 1):
            yield b"a" * 65_536
        yield b'"}'

    response = client.post("/ingest", content=chunks(), headers=JSON)

    assert response.status_code == 413
    assert embedder.calls == []


def test_body_exactly_at_the_limit_is_accepted(client, store):
    payload = b'{"id":"pets","text":"my cat"}'
    body = payload + b" " * (MAX_BODY_BYTES - len(payload))  # trailing JSON whitespace is valid

    response = client.post("/ingest", content=body, headers=JSON)

    assert response.status_code == 201
    assert store.count() == 1


def test_body_limit_is_one_megabyte():
    assert MAX_BODY_BYTES == 1_000_000


# --- Host allowlist ------------------------------------------------------------------------


def test_foreign_host_is_rejected_before_reaching_the_corpus(client, store):
    # DNS rebinding: a page on another name resolving to 127.0.0.1 sends this Host header.
    response = client.post("/ingest", json={"id": "pets", "text": "my cat"}, headers={"host": "rebind.attacker.example"})

    assert response.status_code == 400
    assert store.count() == 0


@pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "orchestrator"])
def test_default_allowlist_accepts_local_and_compose_hosts(retriever, llm, host):
    assert make_client(retriever, llm, host=host).get("/health").status_code == 200


def test_allowlist_comes_from_settings(retriever, llm):
    settings = Settings(allowed_hosts=["api.internal"])

    assert make_client(retriever, llm, settings, host="api.internal").get("/health").status_code == 200
    assert make_client(retriever, llm, settings, host="localhost").get("/health").status_code == 400


# --- interactive docs ----------------------------------------------------------------------


@pytest.mark.parametrize("path", ["/docs", "/redoc", "/openapi.json"])
def test_docs_are_disabled_in_production(retriever, llm, path):
    production = make_client(retriever, llm, Settings(environment="production"))
    development = make_client(retriever, llm, Settings(environment="development"))

    assert production.get(path).status_code == 404
    assert development.get(path).status_code == 200


# --- shipped deployment --------------------------------------------------------------------


def test_compose_publishes_the_api_on_loopback_only():
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert re.findall(r'^\s*-\s*"([^"]*:8000)"\s*$', compose, re.MULTILINE) == ["127.0.0.1:8000:8000"]


def test_make_docker_run_publishes_on_loopback_only():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")
    assert "-p 127.0.0.1:8000:8000" in makefile
    assert "-p 8000:8000" not in makefile


def test_image_defaults_chroma_to_its_writable_data_dir():
    # The non-root user cannot write /app, where the relative default "chroma_db" would land, so a
    # plain `docker run` (make docker-run) failed at startup before this default.
    dockerfile = (ROOT / "Dockerfile").read_text(encoding="utf-8")
    assert "CHROMA_PATH=/data/chroma" in dockerfile

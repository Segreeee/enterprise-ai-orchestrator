from collections.abc import Sequence

import pytest
from fastapi.testclient import TestClient

from orchestrator.agents.schemas import Message
from orchestrator.api.app import create_app
from orchestrator.api.dependencies import get_llm, get_retriever
from orchestrator.rag.chunker import Chunker
from orchestrator.rag.retriever import Retriever
from orchestrator.rag.schemas import MAX_DOCUMENT_CHARS, MAX_METADATA_ENTRIES

AGENTS = [
    {"name": "planner", "system_prompt": "You plan."},
    {"name": "critic", "system_prompt": "You critique."},
]


@pytest.fixture
def retriever(embedder, store) -> Retriever:
    return Retriever(embedder=embedder, store=store, chunker=Chunker(chunk_size=20, chunk_overlap=0))


@pytest.fixture
def client(retriever, llm) -> TestClient:
    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = lambda: llm
    return TestClient(app)


def ingest(client: TestClient, doc_id: str, text: str, **extra):
    return client.post("/ingest", json={"id": doc_id, "text": text, **extra})


# --- /ingest ---------------------------------------------------------------


def test_ingest_logs_document_id_and_chunk_count(client, log_records):
    ingest(client, "pets", "my cat is a lazy cat and my dog is fast")

    events = [r["extra"] for r in log_records if r["extra"].get("document_id") == "pets"]
    assert events and events[-1]["chunks"] == 2


def test_ingest_returns_created_with_chunk_count(client, store):
    response = ingest(client, "pets", "my cat is a lazy cat and my dog is fast", metadata={"source": "a.txt"})

    assert response.status_code == 201
    assert response.json() == {"document_id": "pets", "chunks": 2}
    assert store.count() == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"id": "x"},
        {"id": "", "text": "cat"},
        {"id": "x", "text": "cat", "owner": "me"},
        {"id": "x", "text": "cat", "metadata": {"tags": ["a"]}},
    ],
)
def test_ingest_rejects_invalid_documents(client, payload):
    assert client.post("/ingest", json=payload).status_code == 422


@pytest.mark.parametrize(
    "extra",
    [
        {"text": "cat " * (MAX_DOCUMENT_CHARS // 4 + 1)},
        {"text": "cat", "metadata": {f"k{i}": i for i in range(MAX_METADATA_ENTRIES + 1)}},
        {"text": "cat", "metadata": {"note": "x" * 10_000}},
    ],
)
def test_ingest_rejects_oversized_documents_before_embedding(client, embedder, store, extra):
    assert client.post("/ingest", json={"id": "big", **extra}).status_code == 422
    assert embedder.calls == []
    assert store.count() == 0


# --- /query ----------------------------------------------------------------


def test_query_returns_ranked_results(client):
    ingest(client, "pets", "my cat is a lazy cat")
    ingest(client, "cars", "a fast red car drives")

    response = client.post("/query", json={"query": "tell me about the cat", "k": 2})

    assert response.status_code == 200
    results = response.json()["results"]
    assert len(results) == 2
    assert results[0]["chunk"]["document_id"] == "pets"
    assert results[0]["score"] >= results[1]["score"]


def test_query_on_empty_store_returns_no_results(client):
    response = client.post("/query", json={"query": "cat"})
    assert response.status_code == 200
    assert response.json() == {"results": []}


@pytest.mark.parametrize(
    "payload",
    [{"query": ""}, {"query": "   "}, {"query": "cat", "k": 0}, {"query": "cat", "k": 51}, {}],
)
def test_query_rejects_invalid_requests(client, payload):
    assert client.post("/query", json=payload).status_code == 422


# --- /conversations --------------------------------------------------------


def test_conversation_runs_round_robin(client, llm):
    response = client.post("/conversations", json={"task": "Design a cache", "agents": AGENTS, "max_turns": 3})

    assert response.status_code == 200
    body = response.json()
    assert body["task"] == "Design a cache"
    assert [t["agent"] for t in body["turns"]] == ["planner", "critic", "planner"]
    assert body["stop_reason"] == "max_turns"
    assert len(llm.calls) == 3


def test_conversation_injects_rag_context(client, llm):
    ingest(client, "pets", "a cat sleeps a lot")

    client.post("/conversations", json={"task": "cat care", "agents": AGENTS[:1], "max_turns": 1})

    assert "a cat sleeps a lot" in llm.calls[0][0].content


def test_conversation_without_rag_keeps_plain_system_prompt(client, llm):
    ingest(client, "pets", "a cat sleeps a lot")

    client.post("/conversations", json={"task": "cat care", "agents": AGENTS[:1], "max_turns": 1, "use_rag": False})

    assert llm.calls[0][0] == Message(role="system", content="You plan.")


def test_conversation_honours_stop_token(client, llm):
    llm._replies = ["draft", "LGTM DONE"]
    response = client.post(
        "/conversations", json={"task": "t", "agents": AGENTS, "max_turns": 6, "stop_token": "DONE"}
    )
    assert response.json()["stop_reason"] == "stop_token"
    assert len(response.json()["turns"]) == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"task": "t", "agents": []},
        {"task": "t", "agents": AGENTS + AGENTS[:1]},
        {"task": "", "agents": AGENTS},
        {"task": "t", "agents": AGENTS, "max_turns": 0},
        {"task": "t", "agents": AGENTS, "max_turns": 21},
        {"task": "t", "agents": [{"name": "a", "system_prompt": "p", "temperature": 1}]},
    ],
)
def test_conversation_rejects_invalid_requests(client, payload, llm):
    assert client.post("/conversations", json=payload).status_code == 422
    assert llm.calls == []


def test_llm_backend_unreachable_returns_503(retriever):
    class DownLLM:
        def generate(self, messages: Sequence[Message]) -> Message:
            raise ConnectionError("Failed to connect to Ollama")

    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = DownLLM

    response = TestClient(app).post("/conversations", json={"task": "t", "agents": AGENTS})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]

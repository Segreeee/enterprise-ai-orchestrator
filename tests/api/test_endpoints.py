from collections.abc import Sequence

import httpx
import ollama
import pytest
from fastapi.testclient import TestClient

from orchestrator.agents.schemas import MAX_SYSTEM_PROMPT_CHARS, Message
from orchestrator.api.app import create_app
from orchestrator.api.dependencies import get_llm, get_retriever
from orchestrator.rag.chunker import Chunker
from orchestrator.rag.retriever import Retriever
from orchestrator.api.schemas import MAX_AGENTS, MAX_QUERY_CHARS, MAX_STOP_TOKEN_CHARS, MAX_TASK_CHARS
from orchestrator.rag.schemas import MAX_DOCUMENT_CHARS, MAX_METADATA_ENTRIES
from orchestrator.rag.store import EmbeddingMismatchError

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
    return TestClient(app, base_url="http://localhost")


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
    [
        {"query": ""},
        {"query": "   "},
        {"query": "cat", "k": 0},
        {"query": "cat", "k": 51},
        {},
        {"query": "c" * (MAX_QUERY_CHARS + 1)},
    ],
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

    assert "a cat sleeps a lot" not in llm.calls[0][0].content
    assert "a cat sleeps a lot" in llm.calls[0][1].content


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

    response = TestClient(app, base_url="http://localhost").post("/conversations", json={"task": "t", "agents": AGENTS})

    assert response.status_code == 503
    assert "unavailable" in response.json()["detail"]


def agent(name: str = "a", prompt: str = "p") -> dict:
    return {"name": name, "system_prompt": prompt}


@pytest.mark.parametrize(
    "payload",
    [
        {"task": "t" * (MAX_TASK_CHARS + 1), "agents": [agent()]},
        {"task": "t", "agents": [agent(prompt="p" * (MAX_SYSTEM_PROMPT_CHARS + 1))]},
        {"task": "t", "agents": [agent(name="n" * 65)]},
        {"task": "t", "agents": [agent()], "stop_token": "s" * (MAX_STOP_TOKEN_CHARS + 1)},
        {"task": "t", "agents": [agent(name=f"a{i}") for i in range(MAX_AGENTS + 1)]},
    ],
    ids=["task", "system_prompt", "agent_name", "stop_token", "agent_count"],
)
def test_conversation_rejects_oversized_requests_before_any_llm_call(client, llm, payload):
    assert client.post("/conversations", json=payload).status_code == 422
    assert llm.calls == []


def test_conversation_accepts_requests_at_the_limits(client, llm):
    payload = {
        "task": "t" * MAX_TASK_CHARS,
        "agents": [agent(f"a{i}", "p" * MAX_SYSTEM_PROMPT_CHARS) for i in range(MAX_AGENTS)],
        "stop_token": "s" * MAX_STOP_TOKEN_CHARS,
        "max_turns": 1,
        "use_rag": False,
    }
    assert client.post("/conversations", json=payload).status_code == 200
    assert client.post("/query", json={"query": "c" * MAX_QUERY_CHARS}).status_code == 200


# --- backend failures -------------------------------------------------------------------

SECRET = "ollama-internal.corp:11434"


@pytest.mark.parametrize(
    ("error", "status", "detail"),
    [
        (ConnectionError(f"Failed to connect to {SECRET}"), 503, "model backend unavailable"),
        (ollama.ResponseError(f"model 'llama9' not found on {SECRET}", 404), 503, "model backend error"),
        (httpx.ReadTimeout(f"read timed out from {SECRET}"), 504, "model backend timed out"),
    ],
    ids=["unreachable", "model-missing", "timeout"],
)
def test_backend_failures_map_to_generic_5xx(retriever, error, status, detail):
    class FailingLLM:
        def generate(self, messages, response_format=None):
            raise error

    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: retriever
    app.dependency_overrides[get_llm] = FailingLLM

    response = TestClient(app, base_url="http://localhost").post("/conversations", json={"task": "t", "agents": AGENTS})

    assert response.status_code == status
    assert response.json() == {"detail": detail}
    assert SECRET not in response.text


def test_embedder_timeout_on_query_maps_to_504(store):
    class SlowEmbedder:
        model = "keyword-test"
        dimension = 3

        def embed(self, text):
            raise httpx.ConnectTimeout("connect timed out")

    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: Retriever(embedder=SlowEmbedder(), store=store)

    response = TestClient(app, base_url="http://localhost").post("/query", json={"query": "cat"})

    assert response.status_code == 504


# --- startup -----------------------------------------------------------------------------


def test_startup_builds_the_retriever_so_a_store_mismatch_fails_fast():
    def mismatched():
        raise EmbeddingMismatchError("store built with another-model/384")

    app = create_app()
    app.dependency_overrides[get_retriever] = mismatched

    with pytest.raises(EmbeddingMismatchError):
        with TestClient(app, base_url="http://localhost"):
            pass


def test_startup_succeeds_with_a_compatible_store(retriever):
    built = []
    app = create_app()
    app.dependency_overrides[get_retriever] = lambda: built.append(1) or retriever

    with TestClient(app, base_url="http://localhost") as client:
        assert client.get("/health").status_code == 200
    assert built == [1]


# --- schema hardening -------------------------------------------------------------------


@pytest.mark.parametrize("text", ["   ", "\n\t "])
def test_ingest_rejects_blank_text_and_keeps_the_existing_document(client, store, text):
    ingest(client, "pets", "my cat")

    assert ingest(client, "pets", text).status_code == 422
    assert store.count() == 1


@pytest.mark.parametrize("key", ["$and", "chroma:document", "#document"])
def test_ingest_rejects_chroma_reserved_metadata_keys_before_embedding(client, embedder, store, key):
    response = ingest(client, "pets", "my cat", metadata={key: "x"})

    assert response.status_code == 422
    assert embedder.calls == [] and store.count() == 0

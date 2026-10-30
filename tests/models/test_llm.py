from collections.abc import Sequence
from types import SimpleNamespace

import httpx
import pytest
from pydantic import BaseModel, ValidationError

from orchestrator.agents.schemas import Message
from orchestrator.models.llm import DEFAULT_TIMEOUT, LLMClient, OllamaClient


class Verdict(BaseModel):
    approved: bool
    reason: str


class FakeLLMClient:
    def generate(self, messages: Sequence[Message], response_format: type[BaseModel] | None = None) -> Message:
        if response_format is not None:
            reply = response_format.model_validate({"approved": True, "reason": messages[-1].content})
            return Message(role="assistant", content=reply.model_dump_json())
        return Message(role="assistant", content=f"echo: {messages[-1].content}")


class StubOllama:
    """Stands in for ollama.Client: records the call, returns a canned reply."""

    def __init__(self, role: str = "assistant", content: str = "hello"):
        self.reply = SimpleNamespace(message=SimpleNamespace(role=role, content=content))
        self.calls: list[dict] = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        return self.reply


CONVERSATION = [
    Message(role="system", content="You are terse."),
    Message(role="user", content="Hi"),
]


def test_fake_client_honours_contract():
    client: LLMClient = FakeLLMClient()
    reply = client.generate(CONVERSATION)
    assert isinstance(client, LLMClient)
    assert reply == Message(role="assistant", content="echo: Hi")


def test_fake_client_returns_json_for_response_format():
    reply = FakeLLMClient().generate(CONVERSATION, response_format=Verdict)
    assert Verdict.model_validate_json(reply.content) == Verdict(approved=True, reason="Hi")


def test_ollama_client_satisfies_protocol():
    assert isinstance(OllamaClient(model="m", client=StubOllama()), LLMClient)


def test_ollama_client_sends_messages_and_maps_reply():
    stub = StubOllama(content="hello")
    reply = OllamaClient(model="llama3.2", client=stub).generate(CONVERSATION)

    assert reply == Message(role="assistant", content="hello")
    assert stub.calls == [
        {
            "model": "llama3.2",
            "messages": [
                {"role": "system", "content": "You are terse."},
                {"role": "user", "content": "Hi"},
            ],
            "format": None,
        }
    ]


def test_ollama_client_constrains_reply_to_response_format_schema():
    stub = StubOllama(content='{"approved": false, "reason": "missing tests"}')
    reply = OllamaClient(model="m", client=stub).generate(CONVERSATION, response_format=Verdict)

    assert stub.calls[0]["format"] == Verdict.model_json_schema()
    assert Verdict.model_validate_json(reply.content) == Verdict(approved=False, reason="missing tests")


@pytest.mark.parametrize("content", ['{"approved": "maybe", "reason": "x"}', '{"approved": true}', "not json"])
def test_ollama_client_rejects_reply_not_matching_response_format(content):
    # The schema only guides generation; a reply that still breaks it must not reach callers.
    stub = StubOllama(content=content)
    with pytest.raises(ValidationError):
        OllamaClient(model="m", client=stub).generate(CONVERSATION, response_format=Verdict)


def test_ollama_client_rejects_empty_conversation():
    with pytest.raises(ValueError, match="at least one message"):
        OllamaClient(model="m", client=StubOllama()).generate([])


def test_ollama_client_times_out_on_hung_backend(silent_server):
    client = OllamaClient(model="m", host=silent_server, timeout=0.2)
    with pytest.raises(httpx.TimeoutException):
        client.generate(CONVERSATION)


def test_ollama_client_has_a_finite_default_timeout():
    # httpx.Timeout(None) would let a hung Ollama hold a worker thread forever.
    timeout = OllamaClient(model="m")._client._client.timeout
    assert timeout == httpx.Timeout(DEFAULT_TIMEOUT)
    assert timeout.read is not None and 0 < timeout.read <= 120


@pytest.mark.parametrize(("role", "content"), [("assistant", ""), ("tool", "x")])
def test_ollama_client_rejects_invalid_reply(role, content):
    with pytest.raises(ValidationError):
        OllamaClient(model="m", client=StubOllama(role=role, content=content)).generate(CONVERSATION)

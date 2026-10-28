from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import ollama

from orchestrator.agents.schemas import Message


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, messages: Sequence[Message]) -> Message: ...


class OllamaClient:
    def __init__(self, model: str, host: str = "http://localhost:11434", client: ollama.Client | None = None) -> None:
        self._model = model
        self._client = client or ollama.Client(host=host)

    def generate(self, messages: Sequence[Message]) -> Message:
        if not messages:
            raise ValueError("generate() needs at least one message")
        response = self._client.chat(model=self._model, messages=[m.model_dump() for m in messages])
        # Message validation rejects empty replies or unexpected roles from the model.
        return Message(role=response.message.role, content=response.message.content)

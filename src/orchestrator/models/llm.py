from collections.abc import Sequence
from typing import Protocol, runtime_checkable

import ollama
from pydantic import BaseModel

from orchestrator.agents.schemas import Message
from orchestrator.models import DEFAULT_TIMEOUT


@runtime_checkable
class LLMClient(Protocol):
    def generate(self, messages: Sequence[Message], response_format: type[BaseModel] | None = None) -> Message:
        """With response_format, the reply content is a JSON document valid against that model."""
        ...


class OllamaClient:
    def __init__(
        self,
        model: str,
        host: str = "http://localhost:11434",
        client: ollama.Client | None = None,
        timeout: float = DEFAULT_TIMEOUT,
    ) -> None:
        self._model = model
        self._client = client or ollama.Client(host=host, timeout=timeout)

    def generate(self, messages: Sequence[Message], response_format: type[BaseModel] | None = None) -> Message:
        if not messages:
            raise ValueError("generate() needs at least one message")
        schema = None if response_format is None else response_format.model_json_schema()
        response = self._client.chat(model=self._model, messages=[m.model_dump() for m in messages], format=schema)
        # Untrusted model output: validation rejects empty replies or unexpected roles.
        reply = Message.model_validate({"role": response.message.role, "content": response.message.content})
        if response_format is not None:
            # The schema only guides sampling, so the JSON is checked before any caller parses it.
            response_format.model_validate_json(reply.content)
        return reply

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]

# Every agent's system prompt is resent on each of its turns, so it is capped like any other input.
MAX_SYSTEM_PROMPT_CHARS = 4_000


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, str_strip_whitespace=True)


class Message(_Schema):
    role: Role
    content: str = Field(min_length=1)


class AgentSpec(_Schema):
    name: str = Field(min_length=1, max_length=64)
    system_prompt: str = Field(min_length=1, max_length=MAX_SYSTEM_PROMPT_CHARS)


class Turn(_Schema):
    agent: str = Field(min_length=1)
    message: Message


class Conversation(_Schema):
    task: str = Field(min_length=1)
    turns: list[Turn]
    stop_reason: Literal["max_turns", "stop_token"]

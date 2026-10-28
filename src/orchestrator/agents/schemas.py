from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

Role = Literal["system", "user", "assistant"]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True, str_strip_whitespace=True)


class Message(_Schema):
    role: Role
    content: str = Field(min_length=1)


class AgentSpec(_Schema):
    name: str = Field(min_length=1)
    system_prompt: str = Field(min_length=1)


class Turn(_Schema):
    agent: str = Field(min_length=1)
    message: Message

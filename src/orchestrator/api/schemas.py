from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from orchestrator.agents.schemas import AgentSpec
from orchestrator.rag.schemas import RetrievalResult

NonBlank = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HealthResponse(_Schema):
    status: Literal["ok"]
    version: str


class IngestResponse(_Schema):
    document_id: str
    chunks: int


class QueryRequest(_Schema):
    query: NonBlank
    k: int = Field(default=4, ge=1, le=50)


class QueryResponse(_Schema):
    results: list[RetrievalResult]


class ConversationRequest(_Schema):
    task: NonBlank
    agents: list[AgentSpec] = Field(min_length=1)
    # Capped: every turn is an LLM call.
    max_turns: int = Field(default=6, ge=1, le=20)
    stop_token: str | None = Field(default=None, min_length=1)
    use_rag: bool = True
    k: int = Field(default=4, ge=1, le=50)

    @model_validator(mode="after")
    def _unique_agent_names(self) -> Self:
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate agent names: {names}")
        return self

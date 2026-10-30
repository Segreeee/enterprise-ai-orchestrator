from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from orchestrator.agents.schemas import AgentSpec
from orchestrator.rag.schemas import RetrievalResult

# Each value below ends up in every LLM prompt or embedding call of the request, so all are capped.
MAX_QUERY_CHARS = 1_000
MAX_TASK_CHARS = 4_000
MAX_STOP_TOKEN_CHARS = 64
MAX_AGENTS = 8

QueryText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)]
TaskText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_TASK_CHARS)]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class HealthResponse(_Schema):
    status: Literal["ok"]
    version: str


class IngestResponse(_Schema):
    document_id: str
    chunks: int


class QueryRequest(_Schema):
    query: QueryText
    k: int = Field(default=4, ge=1, le=50)


class QueryResponse(_Schema):
    results: list[RetrievalResult]


class ConversationRequest(_Schema):
    task: TaskText
    agents: list[AgentSpec] = Field(min_length=1, max_length=MAX_AGENTS)
    # Capped: every turn is an LLM call.
    max_turns: int = Field(default=6, ge=1, le=20)
    stop_token: str | None = Field(default=None, min_length=1, max_length=MAX_STOP_TOKEN_CHARS)
    use_rag: bool = True
    k: int = Field(default=4, ge=1, le=50)

    @model_validator(mode="after")
    def _unique_agent_names(self) -> Self:
        names = [a.name for a in self.agents]
        if len(set(names)) != len(names):
            raise ValueError(f"duplicate agent names: {names}")
        return self

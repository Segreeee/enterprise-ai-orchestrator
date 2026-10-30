import os
from pathlib import Path
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, StringConstraints

_ENV_VARS = {
    "ollama_host": "OLLAMA_HOST",
    "chroma_path": "CHROMA_PATH",
    "model_name": "MODEL_NAME",
    "environment": "ENV",
}

HostName = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ollama_host: HttpUrl = Field(default="http://localhost:11434", validate_default=True)
    chroma_path: Path = Path("chroma_db")
    model_name: str = Field(default="llama3.2", min_length=1)
    # Host headers the API answers to: loopback plus the compose service name. Anything else is
    # refused, which stops DNS-rebinding pages from driving a locally bound instance.
    allowed_hosts: list[HostName] = Field(default_factory=lambda: ["localhost", "127.0.0.1", "orchestrator"], min_length=1)
    # "production" hides the interactive /docs, /redoc and /openapi.json.
    environment: Literal["development", "production"] = "development"

    @classmethod
    def from_env(cls) -> Self:
        values: dict[str, object] = {field: os.environ[var] for field, var in _ENV_VARS.items() if var in os.environ}
        if "ALLOWED_HOSTS" in os.environ:
            values["allowed_hosts"] = [h for h in os.environ["ALLOWED_HOSTS"].split(",") if h.strip()]
        return cls.model_validate(values)

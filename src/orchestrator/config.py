import os
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

_ENV_VARS = {
    "ollama_host": "OLLAMA_HOST",
    "chroma_path": "CHROMA_PATH",
    "model_name": "MODEL_NAME",
}


class Settings(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    ollama_host: HttpUrl = Field(default="http://localhost:11434", validate_default=True)
    chroma_path: Path = Path("chroma_db")
    model_name: str = Field(default="llama3.2", min_length=1)

    @classmethod
    def from_env(cls) -> Self:
        return cls(**{field: os.environ[var] for field, var in _ENV_VARS.items() if var in os.environ})

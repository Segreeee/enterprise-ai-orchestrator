from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt

# Chroma only stores scalar metadata values.
Metadata = dict[str, str | int | float | bool]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Document(_Schema):
    id: str = Field(min_length=1)
    text: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)


class Chunk(_Schema):
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    index: NonNegativeInt
    text: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)


class RetrievalResult(_Schema):
    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0)

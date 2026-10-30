from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, NonNegativeInt, StringConstraints

# One /ingest request costs one embed call per ~450 chars and copies the metadata into
# every stored chunk, so both are capped before any work is done.
MAX_DOCUMENT_CHARS = 50_000
MAX_METADATA_ENTRIES = 16

# Chroma only stores scalar metadata values.
MetadataKey = Annotated[str, StringConstraints(min_length=1, max_length=64)]
MetadataValue = Annotated[str, StringConstraints(max_length=256)] | int | float | bool
Metadata = dict[MetadataKey, MetadataValue]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Document(_Schema):
    id: str = Field(min_length=1, max_length=256)
    text: str = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
    metadata: Metadata = Field(default_factory=dict, max_length=MAX_METADATA_ENTRIES)


class Chunk(_Schema):
    id: str = Field(min_length=1)
    document_id: str = Field(min_length=1)
    index: NonNegativeInt
    text: str = Field(min_length=1)
    metadata: Metadata = Field(default_factory=dict)


class RetrievalResult(_Schema):
    chunk: Chunk
    score: float = Field(ge=0.0, le=1.0)

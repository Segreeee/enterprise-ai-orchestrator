from typing import Annotated

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, NonNegativeInt, StringConstraints

# One /ingest request costs one embed call per ~450 chars and copies the metadata into
# every stored chunk, so both are capped before any work is done.
MAX_DOCUMENT_CHARS = 50_000
MAX_METADATA_ENTRIES = 16

# Chroma reserves keys with these prefixes and rejects them at write time, after every chunk
# has already been embedded; refusing them here turns that late 500 into an early 422.
RESERVED_KEY_PREFIXES = ("$", "#", "chroma:")


def _not_reserved(key: str) -> str:
    if key.startswith(RESERVED_KEY_PREFIXES):
        raise ValueError(f"metadata key {key!r} uses a reserved prefix {RESERVED_KEY_PREFIXES}")
    return key


def _not_blank(text: str) -> str:
    # Blank text has no chunks: ingesting it would silently erase the document's current version.
    if not text.strip():
        raise ValueError("text must contain non-whitespace characters")
    return text


# Chroma only stores scalar metadata values.
MetadataKey = Annotated[str, StringConstraints(min_length=1, max_length=64), AfterValidator(_not_reserved)]
MetadataValue = Annotated[str, StringConstraints(max_length=256)] | int | float | bool
Metadata = dict[MetadataKey, MetadataValue]


class _Schema(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", strict=True)


class Document(_Schema):
    id: str = Field(min_length=1, max_length=256)
    text: Annotated[str, AfterValidator(_not_blank)] = Field(min_length=1, max_length=MAX_DOCUMENT_CHARS)
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

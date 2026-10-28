from typing import Self

from pydantic import BaseModel, ConfigDict, NonNegativeInt, PositiveInt, model_validator

from orchestrator.rag.schemas import Chunk, Document


class Chunker(BaseModel):
    """Fixed-size character windows; consecutive chunks share `chunk_overlap` characters."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    chunk_size: PositiveInt = 500
    chunk_overlap: NonNegativeInt = 50

    @model_validator(mode="after")
    def _overlap_smaller_than_size(self) -> Self:
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")
        return self

    def split(self, text: str) -> list[str]:
        if not text.strip():
            return []
        step = self.chunk_size - self.chunk_overlap
        # Stop before a window that would contain only already-seen overlap.
        last_start = max(len(text) - self.chunk_overlap, 1)
        return [text[i : i + self.chunk_size] for i in range(0, last_start, step)]

    def chunk(self, document: Document) -> list[Chunk]:
        return [
            Chunk(id=f"{document.id}:{i}", document_id=document.id, index=i, text=text, metadata=document.metadata)
            for i, text in enumerate(self.split(document.text))
        ]

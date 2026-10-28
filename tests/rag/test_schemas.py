import pytest
from pydantic import ValidationError

from orchestrator.rag.schemas import Chunk, Document, RetrievalResult

CHUNK = Chunk(id="d1:0", document_id="d1", index=0, text="hello")


def test_document_requires_id_and_text():
    with pytest.raises(ValidationError):
        Document(id="", text="x")
    with pytest.raises(ValidationError):
        Document(id="d1", text="")


def test_document_metadata_only_scalar_values():
    assert Document(id="d1", text="x", metadata={"source": "a.pdf", "page": 3}).metadata["page"] == 3
    with pytest.raises(ValidationError):
        Document(id="d1", text="x", metadata={"tags": ["a", "b"]})


def test_chunk_rejects_negative_index_and_extra_fields():
    with pytest.raises(ValidationError):
        Chunk(id="d1:0", document_id="d1", index=-1, text="x")
    with pytest.raises(ValidationError):
        Chunk(id="d1:0", document_id="d1", index=0, text="x", embedding=[0.1])


@pytest.mark.parametrize("score", [0.0, 0.5, 1.0])
def test_retrieval_score_in_unit_interval(score):
    assert RetrievalResult(chunk=CHUNK, score=score).score == score


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_retrieval_score_out_of_range_rejected(score):
    with pytest.raises(ValidationError):
        RetrievalResult(chunk=CHUNK, score=score)


def test_schemas_are_frozen():
    with pytest.raises(ValidationError):
        CHUNK.text = "changed"

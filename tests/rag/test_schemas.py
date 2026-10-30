import pytest
from pydantic import ValidationError

from orchestrator.rag.schemas import MAX_DOCUMENT_CHARS, MAX_METADATA_ENTRIES, Chunk, Document, RetrievalResult

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


def test_document_text_length_is_capped():
    assert len(Document(id="d1", text="x" * MAX_DOCUMENT_CHARS).text) == MAX_DOCUMENT_CHARS
    with pytest.raises(ValidationError):
        Document(id="d1", text="x" * (MAX_DOCUMENT_CHARS + 1))


def test_document_id_length_is_capped():
    assert Document(id="d" * 256, text="x").id == "d" * 256
    with pytest.raises(ValidationError):
        Document(id="d" * 257, text="x")


def test_document_metadata_entry_count_is_capped():
    full = {f"k{i}": i for i in range(MAX_METADATA_ENTRIES)}
    assert Document(id="d1", text="x", metadata=full).metadata == full
    with pytest.raises(ValidationError):
        Document(id="d1", text="x", metadata={**full, "one_more": 1})


@pytest.mark.parametrize(
    ("metadata", "ok"),
    [
        ({"k" * 64: "v"}, True),
        ({"k" * 65: "v"}, False),
        ({"": "v"}, False),
        ({"k": "v" * 256}, True),
        ({"k": "v" * 257}, False),
    ],
)
def test_document_metadata_key_and_value_sizes_are_capped(metadata, ok):
    if ok:
        assert Document(id="d1", text="x", metadata=metadata).metadata == metadata
    else:
        with pytest.raises(ValidationError):
            Document(id="d1", text="x", metadata=metadata)


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



@pytest.mark.parametrize("text", ["", " ", "\n\t  "])
def test_document_rejects_empty_or_blank_text(text):
    with pytest.raises(ValidationError):
        Document(id="d", text=text)


@pytest.mark.parametrize("key", ["$and", "$in", "chroma:document", "#document"])
def test_document_rejects_chroma_reserved_metadata_keys(key):
    with pytest.raises(ValidationError, match="reserved"):
        Document(id="d", text="t", metadata={key: "x"})


@pytest.mark.parametrize("key", ["source", "a$b", "my:chroma", "page#1"])
def test_document_accepts_ordinary_metadata_keys(key):
    assert Document(id="d", text="t", metadata={key: "x"}).metadata == {key: "x"}

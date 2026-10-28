import pytest
from pydantic import ValidationError

from orchestrator.rag.chunker import Chunker
from orchestrator.rag.schemas import Document


def test_split_respects_size_and_overlap():
    assert Chunker(chunk_size=4, chunk_overlap=1).split("abcdefghij") == ["abcd", "defg", "ghij"]


def test_split_without_overlap():
    assert Chunker(chunk_size=4, chunk_overlap=0).split("abcdefgh") == ["abcd", "efgh"]


def test_last_chunk_may_be_shorter_but_never_only_overlap():
    chunks = Chunker(chunk_size=4, chunk_overlap=2).split("abcdefg")
    assert chunks == ["abcd", "cdef", "efg"]
    assert all(len(c) <= 4 for c in chunks)


@pytest.mark.parametrize("text", ["abc", "abcd"])
def test_text_not_longer_than_chunk_size_is_single_chunk(text):
    assert Chunker(chunk_size=4, chunk_overlap=1).split(text) == [text]


@pytest.mark.parametrize("text", ["", "   \n\t"])
def test_blank_text_yields_no_chunks(text):
    assert Chunker(chunk_size=4, chunk_overlap=1).split(text) == []


def test_chunks_reassemble_to_original_text():
    text = "The quick brown fox jumps over the lazy dog." * 5
    size, overlap = 30, 7
    chunks = Chunker(chunk_size=size, chunk_overlap=overlap).split(text)
    rebuilt = chunks[0] + "".join(c[overlap:] for c in chunks[1:])
    assert rebuilt == text


@pytest.mark.parametrize(("size", "overlap"), [(0, 0), (4, -1), (4, 4), (4, 5)])
def test_invalid_configuration_rejected(size, overlap):
    with pytest.raises(ValidationError):
        Chunker(chunk_size=size, chunk_overlap=overlap)


def test_chunk_document_assigns_ids_index_and_metadata():
    doc = Document(id="doc1", text="abcdefghij", metadata={"source": "a.txt"})
    chunks = Chunker(chunk_size=4, chunk_overlap=1).chunk(doc)

    assert [c.id for c in chunks] == ["doc1:0", "doc1:1", "doc1:2"]
    assert [c.index for c in chunks] == [0, 1, 2]
    assert {c.document_id for c in chunks} == {"doc1"}
    assert all(c.metadata == {"source": "a.txt"} for c in chunks)

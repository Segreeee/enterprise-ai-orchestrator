import pytest

from orchestrator.rag.chunker import Chunker
from orchestrator.rag.retriever import Retriever
from orchestrator.rag.schemas import Document, RetrievalResult


@pytest.fixture
def retriever(embedder, store) -> Retriever:
    return Retriever(embedder=embedder, store=store, chunker=Chunker(chunk_size=20, chunk_overlap=0))


def test_ingest_chunks_embeds_and_stores(retriever, embedder, store):
    n = retriever.ingest(Document(id="pets", text="the cat sat on a mat. the dog ran far away"))
    assert n == 3
    assert store.count() == 3
    assert len(embedder.calls) == 3


def test_reingest_replaces_all_previous_chunks(retriever, store):
    retriever.ingest(Document(id="pets", text="cat " * 20))
    assert retriever.ingest(Document(id="pets", text="a short dog note")) == 1
    assert store.count() == 1
    assert retriever.retrieve("cat", k=5)[0].chunk.text == "a short dog note"


def test_ingest_blank_document_text_is_noop(retriever, store):
    assert retriever.ingest(Document(id="empty", text="   ")) == 0
    assert store.count() == 0


def test_retrieve_returns_most_relevant_chunk_first(retriever):
    retriever.ingest(Document(id="pets", text="my cat is a lazy cat"))
    retriever.ingest(Document(id="cars", text="a fast red car drives"))

    results = retriever.retrieve("tell me about the cat", k=2)

    assert all(isinstance(r, RetrievalResult) for r in results)
    assert results[0].chunk.document_id == "pets"
    assert [r.score for r in results] == sorted((r.score for r in results), reverse=True)


def test_retrieve_embeds_the_query(retriever, embedder):
    retriever.retrieve("cat", k=1)
    assert embedder.calls == ["cat"]


def test_retrieve_rejects_blank_query(retriever):
    with pytest.raises(ValueError):
        retriever.retrieve("  ", k=1)

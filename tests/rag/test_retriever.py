import re
import threading
import time

import pytest
from chromadb.errors import ChromaError

from orchestrator.rag.chunker import Chunker
from orchestrator.rag.retriever import RRF_K, Retriever, _bm25_ranking
from orchestrator.rag.schemas import Chunk, Document, RetrievalResult
from orchestrator.rag.store import EmbeddingMismatchError


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


def test_failed_reingest_keeps_previous_version(retriever, store):
    retriever.ingest(Document(id="pets", text="my cat"))

    class DownEmbedder:
        model = "keyword-test"
        dimension = 3

        def embed(self, text: str):
            raise ConnectionError("Ollama down")

    broken = Retriever(embedder=DownEmbedder(), store=store, chunker=Chunker(chunk_size=20, chunk_overlap=0))
    with pytest.raises(ConnectionError):
        broken.ingest(Document(id="pets", text="a new version"))

    assert store.count() == 1
    assert retriever.retrieve("cat", k=1)[0].chunk.text == "my cat"


def test_failed_store_write_keeps_previous_version(retriever, store, monkeypatch):
    retriever.ingest(Document(id="pets", text="my cat"))

    def reject(chunks, embeddings, version=None):
        raise RuntimeError("store rejected the write")

    monkeypatch.setattr(store, "add", reject)
    with pytest.raises(RuntimeError):
        retriever.ingest(Document(id="pets", text="a new version"))
    monkeypatch.undo()

    assert store.count() == 1
    assert retriever.retrieve("cat", k=1)[0].chunk.text == "my cat"


def test_chroma_rejected_metadata_keeps_previous_version(retriever, store):
    # Real failure from the audit: Chroma refuses "$"-prefixed metadata keys at write time.
    retriever.ingest(Document(id="pets", text="my cat"))

    with pytest.raises(ChromaError):
        retriever.ingest(Document(id="pets", text="a new version", metadata={"$and": "x"}))

    assert store.count() == 1
    assert retriever.retrieve("cat", k=1)[0].chunk.text == "my cat"


def test_reingest_into_more_chunks_keeps_only_the_new_version(retriever, store):
    retriever.ingest(Document(id="pets", text="old cat"))
    assert retriever.ingest(Document(id="pets", text="dog " * 20)) == 4
    assert store.count() == 4
    assert all("cat" not in r.chunk.text for r in retriever.retrieve("cat", k=10))


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


# Hybrid search. With KeywordEmbedder, "cats" is near "cat" for the vector search but is a
# different token for BM25, while "zebra" is invisible to the embedder but an exact BM25 match.
# For the query "cat zebra": vector ranks cats=1, zebra=2, dog park=3; BM25 matches only zebra.
HYBRID_DOCS = {"x": "cats", "y": "zebra", "z": "dog park"}


def rrf(*ranks: int) -> float:
    return sum(1 / (RRF_K + r) for r in ranks)


@pytest.fixture
def hybrid(retriever) -> Retriever:
    for doc_id, text in HYBRID_DOCS.items():
        retriever.ingest(Document(id=doc_id, text=text))
    return retriever


def test_retrieve_fuses_vector_and_bm25_ranks_with_rrf(hybrid):
    results = hybrid.retrieve("cat zebra", k=3)

    assert [r.chunk.text for r in results] == ["zebra", "cats", "dog park"]
    assert [r.score for r in results] == pytest.approx([rrf(2, 1), rrf(1), rrf(3)])


def test_rrf_k_is_the_standard_constant():
    assert RRF_K == 60


def test_fusion_looks_past_the_top_k_of_each_search(hybrid):
    # "zebra" is only 2nd for the vector search: fusing just the top-1 of each list would tie it with "cats".
    assert [r.chunk.text for r in hybrid.retrieve("cat zebra", k=1)] == ["zebra"]


def test_bm25_index_covers_chunks_ingested_by_another_process(hybrid, embedder, store):
    # A restarted API process starts with an empty memory but the same persistent store.
    fresh = Retriever(embedder=embedder, store=store)
    assert fresh.retrieve("cat zebra", k=3) == hybrid.retrieve("cat zebra", k=3)


def test_bm25_matching_ignores_case_and_punctuation(hybrid):
    assert hybrid.retrieve("ZEBRA?! cat", k=1)[0].chunk.text == "zebra"


def test_retrieve_without_lexical_matches_falls_back_to_vector_ranks(hybrid):
    # "cat" is no chunk's token ("cats" is), so only the vector ranks count.
    results = hybrid.retrieve("cat", k=3)
    assert [r.chunk.text for r in results] == ["cats", "zebra", "dog park"]
    assert [r.score for r in results] == pytest.approx([rrf(1), rrf(2), rrf(3)])


def test_bm25_orders_lexical_matches_by_score():
    # "zebra" is in 2 of 5 chunks, so its idf is positive; the shorter chunk repeating it scores higher.
    # The weaker match comes first in the corpus, so keeping corpus order would fail this test.
    texts = ["a zebra among many other different words", "zebra zebra", "cat", "dog", "car"]
    corpus = [Chunk(id=f"d:{i}", document_id="d", index=i, text=t) for i, t in enumerate(texts)]
    assert [c.text for c in _bm25_ranking("zebra", corpus)] == ["zebra zebra", texts[0]]



# --- versioned ingestion ------------------------------------------------------------

OLD, NEW = "OLD " * 25, "NEW " * 10  # 5 chunks and 2 chunks of 20 characters


def visible_texts(retriever) -> list[str]:
    return [r.chunk.text for r in retriever.retrieve("old new", k=50)]


def test_chunk_ids_carry_the_document_version(retriever):
    retriever.ingest(Document(id="pets", text="my cat"))
    first = retriever.retrieve("cat", k=1)[0].chunk.id
    retriever.ingest(Document(id="pets", text="my other cat"))
    second = retriever.retrieve("cat", k=1)[0].chunk.id

    assert re.fullmatch(r"pets:[0-9a-f]{16}:0", first)
    assert re.fullmatch(r"pets:[0-9a-f]{16}:0", second)
    assert first != second


def test_identical_reingest_is_a_noop(retriever, embedder):
    doc = Document(id="pets", text="my cat is lazy", metadata={"source": "a.txt"})
    assert retriever.ingest(doc) == 1
    embedder.calls.clear()

    assert retriever.ingest(doc) == 1
    assert embedder.calls == []

    retriever.ingest(Document(id="pets", text="my cat is lazy", metadata={"source": "b.txt"}))
    assert embedder.calls == ["my cat is lazy"]


def test_readers_never_see_a_half_written_version(retriever, store, monkeypatch):
    retriever.ingest(Document(id="doc", text=OLD))
    seen_before_activation: list[list[str]] = []
    activate = store.activate

    def probe_then_activate(*args, **kwargs):
        # The new chunks are fully written now, but not yet the active version.
        seen_before_activation.append(visible_texts(retriever))
        activate(*args, **kwargs)

    monkeypatch.setattr(store, "activate", probe_then_activate)
    retriever.ingest(Document(id="doc", text=NEW))

    assert seen_before_activation == [["OLD OLD OLD OLD OLD "] * 5]
    assert visible_texts(retriever) == ["NEW NEW NEW NEW NEW "] * 2


def test_failed_activation_keeps_previous_version_and_next_ingest_cleans_up(retriever, store, monkeypatch):
    retriever.ingest(Document(id="doc", text=OLD))

    def fail(*args, **kwargs):
        raise RuntimeError("system collection unavailable")

    monkeypatch.setattr(store, "activate", fail)
    with pytest.raises(RuntimeError):
        retriever.ingest(Document(id="doc", text=NEW))
    monkeypatch.undo()

    assert visible_texts(retriever) == ["OLD OLD OLD OLD OLD "] * 5
    assert retriever.ingest(Document(id="doc", text="dog " * 5)) == 1
    assert store.count() == 1


def run_concurrently(*targets) -> None:
    errors: list[BaseException] = []

    def guarded(target):
        try:
            target()
        except BaseException as exc:  # surfaced below: a thread's exception would otherwise vanish
            errors.append(exc)

    threads = [threading.Thread(target=guarded, args=(t,)) for t in targets]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)
    assert not errors, errors


def test_concurrent_reingests_of_a_document_are_serialised(retriever, store, monkeypatch):
    events: list[str | None] = []
    add = store.add

    def slow_add(chunks, embeddings, version=None):
        events.append(version)
        time.sleep(0.05)  # widens the window in which an unserialised ingest would interleave
        add(chunks, embeddings, version=version)
        events.append(version)

    monkeypatch.setattr(store, "add", slow_add)
    start = threading.Barrier(2)

    def ingest(text: str):
        return lambda: (start.wait(), retriever.ingest(Document(id="doc", text=text)))

    run_concurrently(ingest("AAAA " * 20), ingest("BBBB " * 8))

    assert events[0] == events[1] and events[2] == events[3]
    visible = retriever.retrieve("aaaa bbbb", k=50)
    words = {r.chunk.text.split()[0] for r in visible}
    assert len(words) == 1
    assert len(visible) == store.count() == (5 if words == {"AAAA"} else 2)


def test_ingests_of_different_documents_do_not_wait_for_each_other(retriever, store, monkeypatch):
    in_slow_add, other_done = threading.Event(), threading.Event()
    waited: list[bool] = []
    add = store.add

    def blocking_add(chunks, embeddings, version=None):
        if chunks and chunks[0].document_id == "slow":
            in_slow_add.set()
            waited.append(other_done.wait(timeout=2))
        add(chunks, embeddings, version=version)

    monkeypatch.setattr(store, "add", blocking_add)
    slow = threading.Thread(target=retriever.ingest, args=(Document(id="slow", text="cat"),))
    slow.start()
    assert in_slow_add.wait(timeout=2)
    retriever.ingest(Document(id="fast", text="dog"))
    other_done.set()
    slow.join(timeout=5)

    assert waited == [True]


def test_retriever_refuses_a_store_built_with_another_embedding_model(embedder, store):
    store.check_embedding("nomic-embed-text", 768)
    with pytest.raises(EmbeddingMismatchError):
        Retriever(embedder=embedder, store=store)

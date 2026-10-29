import pytest

from orchestrator.models.embeddings import Embedding
from orchestrator.rag.schemas import Chunk
from orchestrator.rag.store import VectorStore


def make_chunk(i: int, text: str, **metadata) -> Chunk:
    return Chunk(id=f"d1:{i}", document_id="d1", index=i, text=text, metadata=metadata)


def test_search_on_empty_store_returns_nothing(store, embedder):
    assert store.search(embedder.embed("cat"), k=3) == []


def test_add_then_search_ranks_by_similarity(store, embedder):
    chunks = [make_chunk(0, "dog dog"), make_chunk(1, "cat cat", source="pets.txt"), make_chunk(2, "car")]
    store.add(chunks, [embedder.embed(c.text) for c in chunks])

    results = store.search(embedder.embed("cat"), k=2)

    assert len(results) == 2
    assert results[0].chunk == chunks[1]
    assert results[0].score > results[1].score
    assert all(0.0 <= r.score <= 1.0 for r in results)


def test_k_larger_than_store_returns_all(store, embedder):
    chunk = make_chunk(0, "cat")
    store.add([chunk], [embedder.embed(chunk.text)])
    assert len(store.search(embedder.embed("cat"), k=10)) == 1


def test_add_is_idempotent_on_chunk_id(store, embedder):
    chunk = make_chunk(0, "cat")
    store.add([chunk], [embedder.embed(chunk.text)])
    store.add([chunk], [embedder.embed(chunk.text)])
    assert store.count() == 1


def test_add_same_id_updates_stored_text(store, embedder):
    store.add([make_chunk(0, "cat")], [embedder.embed("cat")])
    store.add([make_chunk(0, "dog dog")], [embedder.embed("dog dog")])

    assert store.count() == 1
    assert store.search(embedder.embed("dog"), k=1)[0].chunk.text == "dog dog"


@pytest.mark.parametrize(
    ("query", "expected"),
    [([1.0, 0.0, 0.0], 1.0), ([0.0, 1.0, 0.0], 0.5), ([-1.0, 0.0, 0.0], 0.0)],
    ids=["identical", "orthogonal", "opposite"],
)
def test_score_maps_cosine_distance_to_unit_interval(store, query, expected):
    store.add([make_chunk(0, "x")], [Embedding(vector=[1.0, 0.0, 0.0], dimension=3)])
    score = store.search(Embedding(vector=query, dimension=3), k=1)[0].score
    assert score == pytest.approx(expected, abs=1e-6)


def test_add_rejects_length_mismatch(store):
    with pytest.raises(ValueError, match="same length"):
        store.add([make_chunk(0, "cat")], [])


def test_search_rejects_non_positive_k(store, embedder):
    with pytest.raises(ValueError):
        store.search(embedder.embed("cat"), k=0)


def test_persistent_store_survives_reopen(tmp_path, embedder):
    chunk = make_chunk(0, "cat")
    VectorStore.persistent(tmp_path).add([chunk], [embedder.embed(chunk.text)])
    assert VectorStore.persistent(tmp_path).count() == 1

import pytest

from orchestrator.models.embeddings import Embedding
from orchestrator.rag.schemas import Chunk
from orchestrator.rag.store import EmbeddingMismatchError, VectorStore


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


def test_all_chunks_rebuilds_every_stored_chunk(store, embedder):
    chunks = [make_chunk(0, "cat", source="a.txt"), make_chunk(1, "dog")]
    store.add(chunks, [embedder.embed(c.text) for c in chunks])
    assert sorted(store.all_chunks(), key=lambda c: c.index) == chunks


def test_all_chunks_on_empty_store_is_empty(store):
    assert store.all_chunks() == []


# --- versions -------------------------------------------------------------------


def add_versioned(store, embedder, version: str, *texts: str, document_id: str = "d1") -> None:
    chunks = [
        Chunk(id=f"{document_id}:{version}:{i}", document_id=document_id, index=i, text=t)
        for i, t in enumerate(texts)
    ]
    store.add(chunks, [embedder.embed(t) for t in texts], version=version)


def test_search_and_all_chunks_only_see_the_given_versions(store, embedder):
    add_versioned(store, embedder, "v1", "old cat")
    add_versioned(store, embedder, "v2", "new cat")

    assert [r.chunk.text for r in store.search(embedder.embed("cat"), k=5, versions=["v2"])] == ["new cat"]
    assert [c.text for c in store.all_chunks(versions=["v2"])] == ["new cat"]
    assert store.search(embedder.embed("cat"), k=5, versions=[]) == []
    assert store.all_chunks(versions=[]) == []


def test_activate_points_a_document_at_one_version(store):
    assert store.active_version("d1") is None
    store.activate("d1", "v1", chunks=3)
    store.activate("d2", "w1", chunks=1)
    store.activate("d1", "v2", chunks=2)

    assert store.active_version("d1") == ("v2", 2)
    assert store.active_versions() == {"d1": "v2", "d2": "w1"}


def test_delete_stale_keeps_only_the_given_version_of_that_document(store, embedder):
    add_versioned(store, embedder, "v1", "old a", "old b")
    add_versioned(store, embedder, "v2", "new a")
    store.add([make_chunk(9, "legacy chunk without version")], [embedder.embed("legacy")])
    add_versioned(store, embedder, "w1", "other doc", document_id="d2")

    store.delete_stale("d1", keep_version="v2")

    assert sorted(c.text for c in store.all_chunks()) == ["new a", "other doc"]


def test_check_embedding_stamps_an_empty_store_then_refuses_a_different_model(store):
    store.check_embedding("nomic-embed-text", 768)
    store.check_embedding("nomic-embed-text", 768)

    for model, dimension in [("nomic-embed-text", 384), ("other-model", 768)]:
        with pytest.raises(EmbeddingMismatchError, match="nomic-embed-text"):
            store.check_embedding(model, dimension)


def test_embedding_stamp_survives_reopen(tmp_path):
    VectorStore.persistent(tmp_path).check_embedding("nomic-embed-text", 768)
    with pytest.raises(EmbeddingMismatchError):
        VectorStore.persistent(tmp_path).check_embedding("other-model", 768)


def test_persistent_store_disables_chroma_telemetry(tmp_path):
    assert VectorStore.persistent(tmp_path).client.get_settings().anonymized_telemetry is False


def test_stale_chunks_closer_to_the_query_do_not_crowd_out_active_ones(store, embedder):
    add_versioned(store, embedder, "old", "cat cat", "cat cat cat")
    add_versioned(store, embedder, "new", "dog with a cat")

    results = store.search(embedder.embed("cat"), k=1, versions=["new"])

    assert [r.chunk.text for r in results] == ["dog with a cat"]

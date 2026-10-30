import sqlite3
import threading

import pytest

from orchestrator.rag.metadata_db import ActiveVersion, MetadataDB


@pytest.fixture
def db() -> MetadataDB:
    return MetadataDB(sqlite3.connect(":memory:", check_same_thread=False))


def columns(db: MetadataDB, table: str) -> list[str]:
    return [row[1] for row in db.connection.execute(f"PRAGMA table_info({table})")]


def test_schema_is_created_on_connect(db):
    assert columns(db, "document_versions") == ["doc_id", "active_version", "chunks"]
    assert columns(db, "embedding_model") == ["id", "model", "dimension"]


def test_activate_flips_one_pointer_per_document(db):
    assert db.active_version("d1") is None
    db.activate("d1", "v1", chunks=3)
    db.activate("d2", "w1", chunks=1)
    db.activate("d1", "v2", chunks=2)

    assert db.active_version("d1") == ActiveVersion("v2", 2)
    assert db.active_versions() == {"d1": "v2", "d2": "w1"}
    assert db.connection.execute("SELECT COUNT(*) FROM document_versions").fetchone() == (2,)


def test_a_rejected_flip_leaves_the_previous_pointer(db):
    db.activate("d1", "v1", chunks=3)

    with pytest.raises(sqlite3.IntegrityError):
        db.activate("d1", "v2", chunks=-1)

    assert db.active_version("d1") == ActiveVersion("v1", 3)


def test_embedding_model_is_recorded_once(db):
    assert db.embedding_model() is None
    db.record_embedding_model("nomic-embed-text", 768)

    assert db.embedding_model() == ("nomic-embed-text", 768)
    with pytest.raises(sqlite3.IntegrityError):
        db.record_embedding_model("other-model", 384)
    assert db.embedding_model() == ("nomic-embed-text", 768)


def test_state_survives_reopening_the_file(tmp_path):
    path = tmp_path / "nested" / "system.db"
    first = MetadataDB.open(path)
    first.activate("d1", "v1", chunks=2)
    first.record_embedding_model("nomic-embed-text", 768)
    first.connection.close()

    reopened = MetadataDB.open(path)
    assert reopened.active_version("d1") == ActiveVersion("v1", 2)
    assert reopened.embedding_model() == ("nomic-embed-text", 768)


def test_concurrent_flips_from_worker_threads_are_all_recorded(db):
    # The API runs sync handlers in a thread pool, all sharing one connection.
    threads = [threading.Thread(target=db.activate, args=(f"d{i}", f"v{i}", i)) for i in range(32)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=10)

    assert db.active_versions() == {f"d{i}": f"v{i}" for i in range(32)}


def test_operations_on_the_shared_connection_are_serialised(db):
    # sqlite3 asks callers to serialise writes on a connection shared between threads.
    db._lock.acquire()
    worker = threading.Thread(target=db.activate, args=("d1", "v1", 1))
    worker.start()
    worker.join(timeout=0.3)
    still_waiting = worker.is_alive()
    db._lock.release()
    worker.join(timeout=5)

    assert still_waiting
    assert db.active_version("d1") == ActiveVersion("v1", 1)

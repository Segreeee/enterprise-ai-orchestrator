"""Transactional RAG state in SQLite: which version of each document is active, and the embedding model.

This used to live in a second Chroma collection, but chromadb 1.5.9 intermittently dropped chunk
vectors when writes to two collections were followed by a read. Chroma now holds chunks only.
"""

import sqlite3
import threading
from pathlib import Path
from typing import NamedTuple, Self

_SCHEMA = """
CREATE TABLE IF NOT EXISTS document_versions (
    doc_id TEXT PRIMARY KEY,
    active_version TEXT NOT NULL,
    chunks INTEGER NOT NULL CHECK (chunks >= 0)
);
CREATE TABLE IF NOT EXISTS embedding_model (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    model TEXT NOT NULL,
    dimension INTEGER NOT NULL CHECK (dimension > 0)
);
"""


class ActiveVersion(NamedTuple):
    version: str
    chunks: int


class MetadataDB:
    """One SQLite connection shared by the API's worker threads, serialised by a lock."""

    def __init__(self, connection: sqlite3.Connection) -> None:
        self.connection = connection
        self._lock = threading.Lock()
        with self._lock, connection:
            connection.executescript(_SCHEMA)

    @classmethod
    def open(cls, path: Path) -> Self:
        path.parent.mkdir(parents=True, exist_ok=True)
        return cls(sqlite3.connect(path, check_same_thread=False))

    def activate(self, document_id: str, version: str, chunks: int) -> None:
        """Flip the document's pointer in one transaction: readers see the old version or the new one."""
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT OR REPLACE INTO document_versions (doc_id, active_version, chunks) VALUES (?, ?, ?)",
                (document_id, version, chunks),
            )

    def active_version(self, document_id: str) -> ActiveVersion | None:
        with self._lock:
            row = self.connection.execute(
                "SELECT active_version, chunks FROM document_versions WHERE doc_id = ?", (document_id,)
            ).fetchone()
        return None if row is None else ActiveVersion(row[0], row[1])

    def active_versions(self) -> dict[str, str]:
        with self._lock:
            return dict(self.connection.execute("SELECT doc_id, active_version FROM document_versions"))

    def embedding_model(self) -> tuple[str, int] | None:
        with self._lock:
            row = self.connection.execute("SELECT model, dimension FROM embedding_model WHERE id = 1").fetchone()
        return None if row is None else (row[0], row[1])

    def record_embedding_model(self, model: str, dimension: int) -> None:
        """Stamp the model once; a second stamp raises instead of silently replacing it."""
        with self._lock, self.connection:
            self.connection.execute(
                "INSERT INTO embedding_model (id, model, dimension) VALUES (1, ?, ?)", (model, dimension)
            )

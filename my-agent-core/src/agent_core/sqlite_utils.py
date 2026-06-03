from __future__ import annotations

import os
import sqlite3
from pathlib import Path
from typing import Iterable


class ClosingSQLiteConnection(sqlite3.Connection):
    """sqlite3 connection whose context manager also closes the file handle.

    The stdlib sqlite3.Connection context manager only commits/rolls back; it
    intentionally leaves the connection open.  Our stores use short-lived
    ``with self._connect() as conn`` operations, so not closing here leaks file
    descriptors quickly under WebSocket polling / session persistence traffic.
    """

    def __exit__(self, exc_type, exc, tb):  # type: ignore[override]
        try:
            return super().__exit__(exc_type, exc, tb)
        finally:
            self.close()


def connect_sqlite(
    db_path: str | Path,
    *,
    row_factory: type | None = None,
    pragmas: Iterable[str] = (),
) -> sqlite3.Connection:
    timeout = float(os.getenv("AGENT_SQLITE_TIMEOUT_SECONDS") or "30")
    conn = sqlite3.connect(str(db_path), timeout=timeout, factory=ClosingSQLiteConnection)
    if row_factory is not None:
        conn.row_factory = row_factory
    for pragma in pragmas:
        conn.execute(pragma)
    return conn

"""Durable group/job correlations; this is not an execution-job queue.

Run Store owns job state. The ledger retains submission intent across a lost
HTTP response, cancellation intent, and the caller's consumption receipt.
"""

import fcntl
import json
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from sprout.runstore.specs import canonical


class Conflict(ValueError):
    pass


class Ledger:
    def __init__(self, path: str | Path):
        self.path = Path(path).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS driver_meta (name TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS driver_groups (
                    id TEXT PRIMARY KEY,
                    request TEXT NOT NULL,
                    document TEXT NOT NULL,
                    terminal INTEGER NOT NULL DEFAULT 0,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    acknowledged_at REAL,
                    created_at REAL NOT NULL
                );
            """)

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def bind(self, backend_url: str) -> None:
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO driver_meta VALUES ('backend_url', ?)", (backend_url,))
            if (
                db.execute("SELECT value FROM driver_meta WHERE name='backend_url'").fetchone()[0]
                != backend_url
            ):
                raise Conflict("This ledger belongs to a different Run Store URL")

    @contextmanager
    def owner(self):
        """One polling controller per ledger, including across processes."""
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Conflict("Another driver already owns this ledger") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def create(self, group_id: str, request: dict, document: dict) -> dict:
        serialized = canonical(request)
        with self.connection() as db:
            db.execute(
                "INSERT OR IGNORE INTO driver_groups(id,request,document,created_at) VALUES (?,?,?,?)",
                (group_id, serialized, canonical(document), time.time()),
            )
            row = db.execute("SELECT * FROM driver_groups WHERE id=?", (group_id,)).fetchone()
            if row["request"] != serialized:
                raise Conflict("rollout_job_id already names a different request")
            return self._row(row)

    @staticmethod
    def _row(row) -> dict:
        if row is None:
            raise KeyError("Unknown rollout group")
        return {
            **dict(row),
            "request": json.loads(row["request"]),
            "document": json.loads(row["document"]),
            "cancel_requested": bool(row["cancel_requested"]),
        }

    def get(self, group_id: str) -> dict:
        with self.connection() as db:
            return self._row(db.execute("SELECT * FROM driver_groups WHERE id=?", (group_id,)).fetchone())

    def active(self) -> list[str]:
        with self.connection() as db:
            return [
                row[0]
                for row in db.execute("SELECT id FROM driver_groups WHERE terminal=0 ORDER BY created_at,id")
            ]

    def save(self, group_id: str, document: dict, *, terminal: bool = False) -> None:
        # Separate columns prevent polling from overwriting a concurrent DELETE.
        with self.connection() as db:
            db.execute(
                "UPDATE driver_groups SET document=?, terminal=? WHERE id=?",
                (canonical(document), int(terminal), group_id),
            )

    def release(self, group_id: str) -> None:
        """A terminal DELETE acknowledges consumption; an active one requests cancellation."""
        with self.connection() as db:
            row = db.execute("SELECT terminal FROM driver_groups WHERE id=?", (group_id,)).fetchone()
            if row is None:
                raise KeyError(group_id)
            if row["terminal"]:
                db.execute(
                    "UPDATE driver_groups SET acknowledged_at=COALESCE(acknowledged_at,?) WHERE id=?",
                    (time.time(), group_id),
                )
            else:
                db.execute("UPDATE driver_groups SET cancel_requested=1 WHERE id=?", (group_id,))

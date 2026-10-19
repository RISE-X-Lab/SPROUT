"""The RL driver's durable record of its groups, in one SQLite file.

For each group the ledger keeps the request, the group document, a terminal flag, a
cancellation request and the time at which the caller acknowledged the result. The record
survives a restart, so a new driver resumes the groups that are not terminal. The ledger is
not a job queue: the Run Store owns the state of each job.
"""

import fcntl
import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from sprout.runstore.specs import canonical


class Conflict(ValueError):
    """A request or a binding that disagrees with what the ledger already holds."""


class Ledger:
    """The groups of one RL driver in the SQLite file ``path``: a ``GroupStore`` for the search coordinator.

    The constructor creates the file and its tables if they do not exist. Each call commits
    before it returns.
    """

    def __init__(self, path: str | Path) -> None:
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
    def connection(self) -> Iterator[sqlite3.Connection]:
        """A connection to the ledger file. It commits if the block succeeds, and rolls back if it raises.

        The connection closes when the block exits.
        """
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def bind(self, backend_url: str) -> None:
        """Bind the ledger to the Run Store at ``backend_url`` on its first use.

        Raises ``Conflict`` if the ledger is bound to a different URL.
        """
        with self.connection() as db:
            db.execute("INSERT OR IGNORE INTO driver_meta VALUES ('backend_url', ?)", (backend_url,))
            if (
                db.execute("SELECT value FROM driver_meta WHERE name='backend_url'").fetchone()[0]
                != backend_url
            ):
                raise Conflict("This ledger belongs to a different Run Store URL")

    @contextmanager
    def owner(self) -> Iterator[None]:
        """Hold the ledger for one polling driver until the block exits, across processes too.

        Raises ``Conflict`` at once if another driver holds it. Creates ``<path>.lock`` if needed.
        """
        with self.path.with_suffix(self.path.suffix + ".lock").open("a") as handle:
            try:
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as error:
                raise Conflict("Another driver already owns this ledger") from error
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def create(self, group_id: str, request: dict[str, Any], document: dict[str, Any]) -> dict[str, Any]:
        """Record a new group, and return its row (``get``).

        A repeated identical request returns the stored row. Raises ``Conflict`` if ``group_id``
        already names a different request.
        """
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
    def _row(row: sqlite3.Row | None) -> dict[str, Any]:
        if row is None:
            raise KeyError("Unknown rollout group")
        return {
            **dict(row),
            "request": json.loads(row["request"]),
            "document": json.loads(row["document"]),
            "cancel_requested": bool(row["cancel_requested"]),
        }

    def get(self, group_id: str) -> dict[str, Any]:
        """The group's row. Raises KeyError for an unknown group.

        The row has ``request``, ``document``, ``terminal``, ``cancel_requested`` and ``acknowledged_at``.
        """
        with self.connection() as db:
            return self._row(db.execute("SELECT * FROM driver_groups WHERE id=?", (group_id,)).fetchone())

    def active(self) -> list[str]:
        """The ids of the groups that are not terminal, oldest first."""
        with self.connection() as db:
            return [
                row[0]
                for row in db.execute("SELECT id FROM driver_groups WHERE terminal=0 ORDER BY created_at,id")
            ]

    def save(self, group_id: str, document: dict[str, Any], *, terminal: bool = False) -> None:
        """Replace the group's document and its terminal flag. The other columns of the row stay."""
        # A release writes only its own columns, so a poll that saves at the same time cannot undo it.
        with self.connection() as db:
            db.execute(
                "UPDATE driver_groups SET document=?, terminal=? WHERE id=?",
                (canonical(document), int(terminal), group_id),
            )

    def release(self, group_id: str) -> None:
        """Record the caller's release of the group. Raises KeyError for an unknown group.

        For a terminal group, the call records the time of the first acknowledgement. For an
        active group, it requests cancellation.
        """
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

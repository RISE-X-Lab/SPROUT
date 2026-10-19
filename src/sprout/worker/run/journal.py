"""The append-only JSONL journal of a run, and its readers.

Each record is one line: the envelope of :mod:`sprout.worker.run.events` and the event's
payload. ``seq`` increases by one for each record, ``ts`` is RFC 3339 UTC, and each line is
flushed when it is written.

The journal is the canonical state of a run: replay, export and resume read it. A writer
never changes a record that it wrote. A correction is a new event.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from sprout.worker.run.events import JOURNAL_SCHEMA_VERSION, TOOL_STARTED


def new_run_id() -> str:
    """A new random run id of 32 hex digits."""
    return uuid.uuid4().hex


def utc_now(timespec: str = "milliseconds") -> str:
    """The current UTC time in RFC 3339 with a ``Z`` suffix, to the precision ``timespec``.

    The journal writes milliseconds and the resource ledger writes seconds. Both precisions are
    on disk, so neither can change."""
    return datetime.now(UTC).isoformat(timespec=timespec).replace("+00:00", "Z")


class JournalWriter:
    """Appends the events of one run to its journal, and notifies the subscribers of each event.

    Thread-safe, because slots emit from their reader threads. A subscriber is notified after
    its event is on disk, outside the write lock. An exception in a subscriber is discarded, so
    an observer cannot break the run that it watches."""

    def __init__(
        self,
        path: str | os.PathLike[str],
        *,
        run_id: str | None = None,
        agent_id: str = "agent",
        sandbox_id: str = "default",
    ) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id or new_run_id()
        self.agent_id = agent_id
        self.sandbox_id = sandbox_id
        self._seq = 0
        self._lock = threading.Lock()
        self._fh = self.path.open("a", encoding="utf-8")
        self._subscribers: list[Callable[[dict[str, Any]], None]] = []
        self._notifying = threading.local()
        self._tools: dict[str, dict[str, Any]] = {}

    def subscribe(self, callback: Callable[[dict[str, Any]], None]) -> Callable[[dict[str, Any]], None]:
        """Register ``callback`` for every later event, and return it."""
        self._subscribers.append(callback)
        return callback

    def emit(self, event_type: str, **payload: Any) -> dict[str, Any]:
        """Append one event, and return the full record with its envelope.

        A ``tool.started`` event with a ``call_id`` gets ``step``, the call's ordinal in the run.
        Raises ValueError for a second ``tool.started`` event with the same ``call_id``.
        """
        with self._lock:
            call_id = payload.get("call_id")
            if event_type == TOOL_STARTED and call_id:
                if call_id in self._tools:
                    raise ValueError("tool call %r was already started" % call_id)
                # A call's step is its ordinal in the run. Checkpoints and branch points name
                # the call by its step.
                payload["step"] = len(self._tools) + 1
            self._seq += 1
            record = {
                "v": JOURNAL_SCHEMA_VERSION,
                "type": event_type,
                "ts": utc_now(),
                "seq": self._seq,
                "run_id": self.run_id,
                "agent_id": self.agent_id,
                "sandbox_id": self.sandbox_id,
            }
            record.update(payload)
            if event_type == TOOL_STARTED and call_id:
                self._tools[call_id] = record
            self._fh.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
            self._fh.flush()

        self._notify(record)
        return record

    def tool_calls(self) -> list[dict[str, Any]]:
        """The ``tool.started`` records that this writer emitted, in step order."""
        with self._lock:
            return list(self._tools.values())

    def _notify(self, record: dict[str, Any]) -> None:
        """Deliver ``record``, then each event that a subscriber emits meanwhile, to every subscriber.

        A nested emit joins the queue of the fan-out that runs on this thread. So each subscriber
        gets each event once, in emit order, and no fan-out runs inside another.
        """
        if not self._subscribers:
            return

        if getattr(self._notifying, "active", False):
            # We are inside a fan-out: hand this to the loop that owns it.
            self._notifying.queue.append(record)
            return

        self._notifying.active = True
        self._notifying.queue = [record]
        try:
            while self._notifying.queue:
                current = self._notifying.queue.pop(0)
                for callback in list(self._subscribers):
                    try:
                        callback(current)
                    except Exception:  # noqa: BLE001 - observers cannot break the run
                        pass
        finally:
            self._notifying.active = False
            self._notifying.queue = []

    def close(self) -> None:
        """Close the journal file. A second call does nothing."""
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> JournalWriter:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


def iter_journal(path: str | os.PathLike[str]) -> Iterator[dict[str, Any]]:
    """Each record of the journal at ``path``, in file order. Blank lines are skipped."""
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_journal(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every record of the journal at ``path``, as :func:`iter_journal` reads them."""
    return list(iter_journal(path))


def journal_events(path: str | os.PathLike[str]) -> list[dict[str, Any]]:
    """Every complete record of the journal at ``path``, or an empty list if the file is missing.

    A torn last line is skipped, so the call is safe on a journal that another process appends to.
    """
    try:
        lines = Path(path).read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.endswith(b"\n") and line.strip()]

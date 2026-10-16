"""Append-only JSONL journal (v2 envelope).

Monotonic ``seq``, RFC3339 UTC ``ts``, flush per line; standalone so harness/*
has no import edge into benchmark code, and thread-safe because slots emit
from reader threads.

The journal is the canonical state: replay/export/resume all read this file.
Writers never rewrite history; corrections are new events.
"""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import threading
from typing import Callable, Iterator, List, Optional, Union
import uuid

from sprout.harness_core.events import JOURNAL_SCHEMA_VERSION, TOOL_STARTED


#: Directories the OS empties on reboot (and tmpwatch empties sooner). A journal
#: is the ONLY record a run leaves -- snapshot ids, every step, the grading
#: evidence -- so writing one here schedules its own destruction. Learned the
#: expensive way: a 32-instance batch's journals lived in /tmp when the host
#: rebooted mid-regrade, and several hours of agent time now exist only as prose.
VOLATILE_ROOTS = ("/tmp", "/var/tmp", "/dev/shm", "/run")


def volatile_reason(path: Union[str, Path]) -> Optional[str]:
    """Why this path will not survive a reboot, or None if it should.

    Entry points refuse volatile journal destinations; the library layer does
    not, because tests legitimately journal into pytest's /tmp fixtures. The
    check is a resolved-prefix test rather than a filesystem-type probe --
    simple, and it catches the mistake that was actually made.
    """
    resolved = Path(path).resolve()
    for root in VOLATILE_ROOTS:
        if resolved == Path(root) or str(resolved).startswith(root + "/"):
            return (
                "%s is under %s, which the OS empties on reboot -- a journal "
                "is the only record a run leaves" % (resolved, root)
            )
    return None


def new_run_id() -> str:
    return uuid.uuid4().hex


def utc_now(timespec: str = "milliseconds") -> str:
    """RFC3339 UTC now, ``Z``-suffixed. The journal records milliseconds and the
    resource ledger seconds; both precisions are on disk, so neither may change."""
    return datetime.now(timezone.utc).isoformat(timespec=timespec).replace("+00:00", "Z")


class JournalWriter:
    """Single-writer, thread-safe JSONL emitter.

    Also the in-process event bus: :meth:`subscribe` lets components react to
    events without the slots knowing they exist (checkpoint bridge, budget
    enforcement, live dashboards). Subscribers are notified *after* the line is
    durable, outside the write lock, and their failures are isolated -- an
    observer must never be able to break the run it is watching.
    """

    def __init__(
        self,
        path: Union[str, Path],
        *,
        run_id: Optional[str] = None,
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
        self._subscribers: List[Callable[[dict], None]] = []
        self._notifying = threading.local()
        self._tools: dict[str, dict] = {}

    def subscribe(self, callback: Callable[[dict], None]) -> Callable[[dict], None]:
        """Register an observer of every subsequent event. Returns ``callback``."""
        self._subscribers.append(callback)
        return callback

    def emit(self, event_type: str, **payload) -> dict:
        """Append one event; returns the full record (including seq)."""
        with self._lock:
            call_id = payload.get("call_id")
            if event_type == TOOL_STARTED and call_id:
                if call_id in self._tools:
                    raise ValueError("tool call %r was already started" % call_id)
                # A call's step is its ordinal in the run; it names the call in
                # checkpoints and branch points.
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

    def tool_calls(self) -> list[dict]:
        """Started calls, in step order."""
        with self._lock:
            return list(self._tools.values())

    def _notify(self, record: dict) -> None:
        """Deliver ``record`` to every subscriber, then anything they emitted.

        Subscribers do emit -- the checkpoint bridge turns a ``turn.completed``
        into a ``checkpoint.captured``. Naively recursing would let one subscriber
        re-enter the notification it is already inside; simply *suppressing*
        nested events is worse, and was a real defect: an event emitted from
        inside a subscriber reached no subscriber at all, so the resource ledger
        never saw the snapshots the bridge had just claimed and a killed process
        left them unreclaimable.

        So nested emissions are queued and drained after the current fan-out
        finishes. Every event is delivered to every subscriber exactly once, and
        the depth stays flat however many subscribers emit.
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
        with self._lock:
            if not self._fh.closed:
                self._fh.close()

    def __enter__(self) -> "JournalWriter":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


def iter_journal(path: Union[str, Path]) -> Iterator[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                yield json.loads(line)


def read_journal(path: Union[str, Path]) -> List[dict]:
    return list(iter_journal(path))


def journal_events(path: Union[str, Path]) -> List[dict]:
    """Every complete record; a missing file is empty and a torn last line is skipped.

    Safe on a journal another process is still appending to.
    """
    try:
        lines = Path(path).read_bytes().splitlines(keepends=True)
    except FileNotFoundError:
        return []
    return [json.loads(line) for line in lines if line.endswith(b"\n") and line.strip()]

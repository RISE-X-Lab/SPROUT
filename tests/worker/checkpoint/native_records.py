"""The records of an attempt's journal and of mini's native history, written by hand as a run writes them."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

#: The native session of the attempts here.
SESSION = "session-1"


def append(path: Path, entries: list[dict[str, Any]]) -> None:
    """Append ``entries`` to the JSON lines file ``path``."""
    with path.open("a", encoding="utf-8") as stream:
        stream.writelines(json.dumps(entry, ensure_ascii=False) + "\n" for entry in entries)


def journaled_turn(step: int) -> list[dict[str, Any]]:
    """The journal events of turn ``turn-<step>``: call ``call-<step>``, the turn's end and its checkpoint."""
    call, turn = "call-%d" % step, "turn-%d" % step
    seq = step * 10
    return [
        {"seq": seq, "type": "tool.started", "call_id": call, "turn_id": turn, "name": "shell", "args": {},
         "step": step},
        {"seq": seq + 1, "type": "tool.finished", "call_id": call, "turn_id": turn, "output": "ok"},
        {"seq": seq + 2, "type": "model.turn.completed", "turn_id": turn, "call_ids": [call], "step": step},
        {"seq": seq + 3, "type": "checkpoint.captured", "step": step, "turn_id": turn,
         "snapshot_id": "snapshot-%d" % step, "session_ckpt": SESSION, "reason": "captured"},
    ]  # fmt: skip


def native_turn(turn: str, calls: list[str]) -> list[dict[str, Any]]:
    """mini's native records of one response: its assistant message, a result for each call, its end."""
    tool_calls = [
        {"id": call, "type": "function", "function": {"name": "bash", "arguments": "{}"}} for call in calls
    ]
    results = [
        {"type": "mini.message", "message": {"role": "tool", "tool_call_id": call, "content": "ok"}}
        for call in calls
    ]
    assistant = {
        "type": "mini.message",
        "message": {"role": "assistant", "content": "", "tool_calls": tool_calls},
    }
    return [assistant, *results, {"type": "mini.turn", "turn_id": turn, "call_ids": calls}]


def attempt_files(directory: Path) -> tuple[Path, Path]:
    """An attempt's journal, which names session ``SESSION``, and that session's history, without a turn.

    They are where the worker reads them: ``trajectory.jsonl`` and ``native-home/<session>.jsonl``."""
    journal, native = directory / "trajectory.jsonl", directory / "native-home" / ("%s.jsonl" % SESSION)
    native.parent.mkdir(parents=True)
    append(journal, [{"seq": 1, "type": "session.ref", "native_session_id": SESSION}])
    append(native, [{"type": "mini.session", "format": "sprout-mini-v1", "mini_version": "2.4.6"}])
    return journal, native


def add_turn(journal: Path, native: Path, step: int) -> None:
    """Append turn ``turn-<step>``, of the one call ``call-<step>``, to the journal and the history."""
    append(journal, journaled_turn(step))
    append(native, native_turn("turn-%d" % step, ["call-%d" % step]))


def pin(native: Path) -> dict[str, Any]:
    """The reference that pins the bytes of ``native`` as they are now: the prefix that a branch inherits."""
    data = native.read_bytes()
    return {
        "slot": "mini-swe-agent",
        "path": str(native),
        "byte_length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }

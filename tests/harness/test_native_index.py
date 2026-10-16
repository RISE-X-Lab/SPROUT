"""index_native reads a journal it is not handed the way every journal reader does.

A journal is JSON lines written with ``ensure_ascii=False``, so a tool's output
keeps U+0085 (NEL) and U+2028 (LINE SEPARATOR) as the command printed them.
``str.splitlines()`` breaks a line at either and neither half is JSON; a sweep
once crashed on that, and so did ``index_native`` reading a journal itself.
"""

from __future__ import annotations

import json

from sprout.harness.agents.mini_swe.history import index_native
from sprout.harness_core.journal import JournalWriter, journal_events

SESSION = "session"
OUTPUT = "before\x85between\u2028after"


def test_a_record_holding_unicode_line_breaks_is_read_whole(tmp_path):
    journal = tmp_path / "trajectory.jsonl"
    with JournalWriter(journal) as writer:
        writer.emit("session.ref", native_session_id=SESSION)
        writer.emit("tool.started", call_id="call-1", turn_id="turn-1", name="shell", args={})
        writer.emit("tool.finished", call_id="call-1", turn_id="turn-1", status="ok", output=OUTPUT)
        writer.emit("model.turn.completed", turn_id="turn-1", call_ids=["call-1"], step=1)
        writer.emit("checkpoint.captured", step=1, turn_id="turn-1", snapshot_id="snapshot-1",
                    session_ckpt=SESSION, reason="captured")
    native = tmp_path / f"{SESSION}.jsonl"
    rows = [{"type": "mini.session", "format": "sprout-mini-v1", "mini_version": "2.4.6"},
            {"type": "mini.message", "message": {"role": "assistant", "content": "", "tool_calls": [
                {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]}},
            {"type": "mini.message", "message": {"role": "tool", "tool_call_id": "call-1", "content": OUTPUT}},
            {"type": "mini.turn", "turn_id": "turn-1", "call_ids": ["call-1"]}]
    native.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
    assert len(journal.read_text(encoding="utf-8").splitlines()) > len(journal_events(journal)), \
        "the fixture has to hold a record that str.splitlines() breaks"

    points = index_native(journal, native, "mini-swe-agent", SESSION)
    assert [(point.message_step, point.tool_depth, point.snapshot_id) for point in points] == \
        [(1, 1, "snapshot-1")]
    assert points == index_native(journal, native, "mini-swe-agent", SESSION, events=journal_events(journal))

"""The recovery points of an attempt: the closed turns of mini's native history that its journal confirms.

Each test writes the journal and the native history by hand, as a run and mini write them, because the
behavior is how the index reads the two files, also while they grow.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sprout.worker.agents.mini_swe.history import index_native
from sprout.worker.run.journal import JournalWriter, journal_events
from tests.worker.checkpoint.native_records import (
    SESSION,
    add_turn,
    append,
    attempt_files,
    journaled_turn,
    native_turn,
    pin,
)

#: U+0085 (NEL) and U+2028 (LINE SEPARATOR). A journal keeps them as a tool printed them, and
#: str.splitlines() breaks a line at either.
UNICODE_LINE_BREAKS = "before\x85between\u2028after"


def depths(journal: Path, native: Path, **options: Any) -> list[int]:
    return [point.tool_depth for point in index_native(journal, native, "mini-swe-agent", SESSION, **options)]


def test_a_journal_record_that_holds_unicode_line_breaks_is_read_whole(tmp_path):
    """The index reads the journal as ``journal_events`` does, not with ``str.splitlines()``."""
    journal, native = tmp_path / "trajectory.jsonl", tmp_path / ("%s.jsonl" % SESSION)
    with JournalWriter(journal) as writer:
        writer.emit("session.ref", native_session_id=SESSION)
        writer.emit("tool.started", call_id="call-1", turn_id="turn-1", name="shell", args={})
        writer.emit(
            "tool.finished", call_id="call-1", turn_id="turn-1", status="ok", output=UNICODE_LINE_BREAKS
        )
        writer.emit("model.turn.completed", turn_id="turn-1", call_ids=["call-1"], step=1)
        writer.emit(
            "checkpoint.captured", step=1, turn_id="turn-1", snapshot_id="snapshot-1", session_ckpt=SESSION
        )
    append(native, [{"type": "mini.session", "format": "sprout-mini-v1", "mini_version": "2.4.6"}])
    append(native, native_turn("turn-1", ["call-1"]))

    points = index_native(journal, native, "mini-swe-agent", SESSION)

    assert [(point.message_step, point.tool_depth, point.snapshot_id) for point in points] == [
        (1, 1, "snapshot-1")
    ]
    assert points == index_native(journal, native, "mini-swe-agent", SESSION, events=journal_events(journal))


def test_the_index_of_a_journal_frame_ignores_what_the_files_gained_after_it(tmp_path):
    """A caller persists one frame of a growing journal. The points must not name a turn past that frame."""
    journal, native = attempt_files(tmp_path)
    add_turn(journal, native, 1)
    frame = journal_events(journal)
    add_turn(journal, native, 2)

    assert (depths(journal, native, events=frame), depths(journal, native)) == ([1], [1, 2])


def test_a_turn_before_the_journal_is_admitted_only_through_a_pinned_inherited_prefix(tmp_path):
    """A branch's history starts with the turns of its parent, whose calls its own journal never saw."""
    journal, native = attempt_files(tmp_path)
    append(native, native_turn("turn-0", ["call-0"]))
    inherited = pin(native)
    add_turn(journal, native, 1)

    assert (depths(journal, native), depths(journal, native, inherited_native=inherited)) == ([], [1])


def test_a_response_with_a_call_that_the_journal_never_saw_is_no_recovery_point(tmp_path):
    """The pin precedes the response, so it cannot admit the response's own unjournaled call."""
    journal, native = attempt_files(tmp_path)
    inherited = pin(native)
    append(journal, journaled_turn(1))
    append(native, native_turn("turn-1", ["call-1", "call-2"]))

    assert (depths(journal, native), depths(journal, native, inherited_native=inherited)) == ([], [])


def test_index_native_reads_only_the_histories_of_mini(tmp_path):
    journal, native = attempt_files(tmp_path)

    with pytest.raises(ValueError, match="^Unsupported native history slot$"):
        index_native(journal, native, "codex", SESSION)

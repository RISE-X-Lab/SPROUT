"""Branch points: the closed turns of a journal whose checkpoint names both a snapshot and the session."""

from __future__ import annotations

from typing import Any

import pytest

from sprout.worker.checkpoint.rollback import Checkpoint, branch_checkpoints, closed_turns, load_checkpoints
from sprout.worker.run.journal import JournalWriter


def numbered(*records: dict[str, Any]) -> list[dict[str, Any]]:
    """``records`` as journal records, with ``seq`` 1, 2 and so on."""
    return [{**record, "seq": seq} for seq, record in enumerate(records, 1)]


def one_call_turn(step: int) -> list[dict[str, Any]]:
    """Turn ``turn-<step>``: one shell call ``call-<step>`` that started and finished, and the turn's end."""
    call, turn = "call-%d" % step, "turn-%d" % step
    return [
        {"type": "tool.started", "call_id": call, "turn_id": turn, "name": "shell", "args": {}, "step": step},
        {"type": "tool.finished", "call_id": call, "turn_id": turn, "status": "ok", "output": ""},
        {"type": "model.turn.completed", "turn_id": turn, "call_ids": [call], "step": step},
    ]


def capture(step: int, snapshot_id: str | None, reason: str = "captured", **fields: Any) -> dict[str, Any]:
    """The ``checkpoint.captured`` record of turn ``turn-<step>``, paired with session ``session-1``."""
    record = {
        "type": "checkpoint.captured",
        "step": step,
        "turn_id": "turn-%d" % step,
        "snapshot_id": snapshot_id,
    }
    return {**record, "session_ckpt": "session-1", "reason": reason, **fields}


THREE_TURNS = [*one_call_turn(1), *one_call_turn(2), *one_call_turn(3)]


def test_each_capture_record_is_a_checkpoint_in_journal_order():
    events = numbered(*one_call_turn(1), capture(1, "snapshot-1"), capture(1, "snapshot-final", final=True))

    checkpoints = load_checkpoints(None, events=events)

    assert checkpoints == [
        Checkpoint(step=1, seq=4, snapshot_id="snapshot-1", session_ckpt="session-1", turn_id="turn-1"),
        Checkpoint(
            step=1,
            seq=5,
            snapshot_id="snapshot-final",
            session_ckpt="session-1",
            turn_id="turn-1",
            final=True,
        ),
    ]


def test_the_checkpoints_are_read_from_the_journal_file_when_no_events_are_given(tmp_path):
    path = tmp_path / "trajectory.jsonl"
    with JournalWriter(path) as journal:
        journal.emit(
            "checkpoint.captured", step=1, turn_id="turn-1", snapshot_id="snapshot-1", session_ckpt="s"
        )

    checkpoints = load_checkpoints(path)

    assert [(checkpoint.step, checkpoint.snapshot_id) for checkpoint in checkpoints] == [(1, "snapshot-1")]


def test_the_journal_file_is_not_read_when_its_events_are_given(tmp_path):
    """A caller that froze one frame of a growing journal indexes that frame, not the file."""
    events = numbered(*one_call_turn(1), capture(1, "snapshot-1"))

    points = branch_checkpoints(tmp_path / "missing.jsonl", events=events)

    assert [point.snapshot_id for point in points.values()] == ["snapshot-1"]


def test_checkpoints_need_a_journal_or_its_events():
    with pytest.raises(TypeError, match="a journal path or its events are required"):
        load_checkpoints(None)


def turn_records(
    *,
    started: list[str],
    finished: list[str],
    call_ids: list[str],
    step: int,
    skipped: tuple[dict[str, Any], ...] = (),
) -> list[dict[str, Any]]:
    """Turn ``turn-1``: the calls ``started`` at steps 1, 2 and so on, the ``finished`` ones, and its end.

    The ``skipped`` records come after the finished calls."""
    starts = [
        {"type": "tool.started", "call_id": call, "turn_id": "turn-1", "step": ordinal}
        for ordinal, call in enumerate(started, 1)
    ]
    finishes = [{"type": "tool.finished", "call_id": call} for call in finished]
    end = {"type": "model.turn.completed", "turn_id": "turn-1", "call_ids": call_ids, "step": step}
    return numbered(*starts, *finishes, *skipped, end)


def skip(call: str, *, turn: str = "turn-1") -> dict[str, Any]:
    """The ``tool.skipped`` record of ``call`` in ``turn``: mini ran no command after the submitting call."""
    return {"type": "tool.skipped", "call_id": call, "turn_id": turn, "reason": "after_submit"}


def test_a_turn_whose_calls_all_finished_is_closed_at_its_last_calls_step():
    events = turn_records(
        started=["call-a", "call-b"], finished=["call-a", "call-b"], call_ids=["call-a", "call-b"], step=2
    )

    assert closed_turns(events) == {"turn-1": {"step": 2, "call_ids": ["call-a", "call-b"]}}


@pytest.mark.parametrize(
    "started,finished,call_ids,step",
    [
        pytest.param(["call-a"], [], ["call-a"], 1, id="a-call-did-not-finish"),
        pytest.param(["call-a"], ["call-a", "call-x"], ["call-a", "call-x"], 1, id="a-call-never-started"),
        # A snapshot at the first call's step would pair with a conversation cut inside the turn.
        pytest.param(
            ["call-a", "call-b"], ["call-a", "call-b"], ["call-a", "call-b"], 1, id="before-the-last-call"
        ),
        pytest.param([], [], [], 1, id="no-calls"),
    ],
)
def test_a_turn_is_not_closed_when_its_record_disagrees_with_its_calls(started, finished, call_ids, step):
    events = turn_records(started=started, finished=finished, call_ids=call_ids, step=step)

    assert closed_turns(events) == {}


def test_a_turn_whose_later_call_was_skipped_is_closed_at_the_step_of_its_last_call_that_ran():
    events = turn_records(
        started=["call-a"],
        finished=["call-a"],
        skipped=(skip("call-b"),),
        call_ids=["call-a", "call-b"],
        step=1,
    )

    assert closed_turns(events) == {"turn-1": {"step": 1, "call_ids": ["call-a", "call-b"]}}


@pytest.mark.parametrize(
    "started,skipped,call_ids",
    [
        # A turn with no call that ran has no tool step, so no snapshot can pair with it.
        pytest.param([], (skip("call-a"),), ["call-a"], id="no-call-ran"),
        pytest.param(
            ["call-a"], (skip("call-b", turn="turn-2"),), ["call-a", "call-b"], id="skipped-in-another-turn"
        ),
    ],
)
def test_a_skipped_call_settles_only_its_own_turn_and_only_beside_a_call_that_ran(started, skipped, call_ids):
    events = turn_records(started=started, finished=started, skipped=skipped, call_ids=call_ids, step=1)

    assert closed_turns(events) == {}


@pytest.mark.parametrize(
    "captures,points",
    [
        pytest.param(
            [capture(1, "snapshot-1"), capture(2, "snapshot-2"), capture(3, "snapshot-3")],
            {1: "snapshot-1", 2: "snapshot-2", 3: "snapshot-3"},
            id="each-captured-turn",
        ),
        # A clean step reuses the latest capture. A failed capture never is a point.
        pytest.param(
            [capture(1, "snapshot-1"), capture(2, "snapshot-1", "clean"), capture(3, "snapshot-1", "failed")],
            {1: "snapshot-1", 2: "snapshot-1"},
            id="clean-reuses-the-latest-capture",
        ),
        pytest.param(
            [capture(1, "snapshot-0", "clean"), capture(2, "snapshot-2"), capture(3, "snapshot-x", "clean")],
            {2: "snapshot-2"},
            id="clean-names-no-capture-or-a-stale-one",
        ),
        pytest.param([capture(1, "snapshot-1", session_ckpt=None)], {}, id="no-conversation-half"),
        pytest.param([capture(1, None)], {}, id="no-snapshot-half"),
        pytest.param([capture(1, "snapshot-1", turn_id="turn-9")], {}, id="turn-not-closed"),
        pytest.param([capture(2, "snapshot-1", turn_id="turn-1")], {}, id="step-not-where-the-turn-closed"),
        pytest.param([capture(3, "snapshot-final", final=True)], {3: "snapshot-final"}, id="final-snapshot"),
    ],
)
def test_a_closed_turn_is_a_branch_point_when_its_checkpoint_is_exact_and_complete(captures, points):
    events = numbered(*THREE_TURNS, *captures)

    found = branch_checkpoints(None, events=events)

    assert {step: point.snapshot_id for step, point in found.items()} == points


def test_the_final_snapshot_is_no_branch_point_after_an_execution_whose_outcome_is_unknown():
    """Nothing after a call whose outcome is unknown pairs with a known filesystem."""
    events = numbered(
        *THREE_TURNS,
        capture(1, "snapshot-1"),
        {"type": "execution.uncertain", "call_id": "call-3", "step": 3, "kind": "transport_exception"},
        capture(3, "snapshot-final", final=True),
    )

    found = branch_checkpoints(None, events=events)

    assert {step: point.snapshot_id for step, point in found.items()} == {1: "snapshot-1"}

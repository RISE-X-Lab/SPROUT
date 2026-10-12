"""ATIF export + branch points from the journal."""

from __future__ import annotations

import pytest

from sprout.harness.atif import ATIF_VERSION, journal_to_atif
from sprout.harness.core import events as E
from sprout.harness.core.journal import JournalWriter, read_journal
from sprout.harness.checkpoint.rollback import branch_checkpoints, closed_turns, uncaptured_turns


def capture(journal, step, turn_id, snapshot_id, reason="captured", session="ses_1", **extra):
    journal.emit(E.CHECKPOINT_CAPTURED, step=step, turn_id=turn_id, snapshot_id=snapshot_id,
                 session_ckpt=session, reason=reason, captured=reason == "captured", **extra)


def build_journal(path):
    with JournalWriter(path, run_id="run1", agent_id="a1") as journal:
        journal.emit(
            E.RUN_STARTED, slot="opencode", slot_version="1.18.5", model="m1", task_prompt="task"
        )
        # turn 1
        journal.emit(E.AGENT_THINKING, text="think 1")
        journal.emit(E.TOOL_STARTED, call_id="c1", turn_id="t1", name="read", args={"p": "a.txt"})
        journal.emit(E.TOOL_FINISHED, call_id="c1", turn_id="t1", status="ok", output="contents")
        journal.emit("model.turn.completed", turn_id="t1", call_ids=["c1"], step=1)
        capture(journal, 1, "t1", "snap-1")
        journal.emit(
            E.TURN_COMPLETED,
            usage={"input_tokens": 10, "output_tokens": 5, "cached_input_tokens": 2, "cost_usd": 0.1},
        )
        # turn 2
        journal.emit(E.AGENT_MESSAGE, text="all done")
        journal.emit(E.TOOL_STARTED, call_id="c2", turn_id="t2", name="shell", args={"command": "ls"})
        journal.emit(E.TOOL_FINISHED, call_id="c2", turn_id="t2", status="ok", output="a.txt")
        journal.emit("model.turn.completed", turn_id="t2", call_ids=["c2"], step=2)
        capture(journal, 2, "t2", "snap-2", final=True)
        journal.emit(
            E.TURN_COMPLETED, usage={"input_tokens": 20, "output_tokens": 7, "cost_usd": 0.2}
        )
        journal.emit(E.SESSION_REF, native_session_id="ses_1")
        journal.emit(E.RUN_FINISHED, status="completed", usage={"input_tokens": 30})
    return read_journal(path)


def test_atif_document_shape(tmp_path):
    document = journal_to_atif(build_journal(tmp_path / "j.jsonl"), continued_from="traj-parent")
    assert document["schema_version"] == ATIF_VERSION
    assert document["agent"] == {"name": "opencode", "version": "1.18.5", "model": "m1"}
    assert document["session_id"] == "ses_1"
    assert document["continued_trajectory_ref"] == "traj-parent"
    assert len(document["steps"]) == 2
    # step_id must start at 1 and be contiguous (ATIF validator requirement)
    assert [s["step_id"] for s in document["steps"]] == [1, 2]
    # each step pairs the observation with the call that started it
    first, second = document["steps"]
    assert first["reasoning_content"] == "think 1"
    assert first["tool_calls"][0]["name"] == "read"
    assert first["observation"] == "contents"
    assert first["metrics"]["prompt_tokens"] == 10
    assert first["metrics"]["cached_tokens"] == 2
    assert second["message"] == "all done"
    assert document["final_metrics"]["prompt_tokens"] == 30
    assert document["final_metrics"]["cost_usd"] == pytest.approx(0.3)
    # every checkpoint.captured, in journal order
    assert [c["snapshot_id"] for c in document["extra"]["checkpoints"]] == ["snap-1", "snap-2"]


def test_atif_never_emits_zero_steps(tmp_path):
    """ATIF requires steps >= 1 even for a run that died immediately."""
    path = tmp_path / "empty.jsonl"
    with JournalWriter(path, run_id="r") as journal:
        journal.emit(E.RUN_STARTED, slot="codex", task_prompt="do it")
        journal.emit(E.RUN_FINISHED, status="error", error="auth failed")
    document = journal_to_atif(read_journal(path))
    assert len(document["steps"]) == 1
    assert document["steps"][0]["source"] == "system"
    assert document["extra"]["status"] == "error"


# --- branch points ------------------------------------------------------------
def turns_journal(path, *captures):
    """Three one-call turns, then the given (step, snapshot, reason) captures."""
    with JournalWriter(path) as journal:
        for step in (1, 2, 3):
            journal.emit(E.TOOL_STARTED, call_id="c%d" % step, turn_id="t%d" % step, name="shell", args={})
            journal.emit(E.TOOL_FINISHED, call_id="c%d" % step, turn_id="t%d" % step, status="ok", output="")
            journal.emit("model.turn.completed", turn_id="t%d" % step, call_ids=["c%d" % step], step=step)
        for step, snapshot, reason, *session in captures:
            capture(journal, step, "t%d" % step, snapshot, reason, *session)
    return path


@pytest.mark.parametrize("captures,expected", [
    # every closed turn with a snapshot is a point
    pytest.param([(1, "s1", "captured"), (2, "s2", "captured"), (3, "s3", "captured")],
                 {1: "s1", 2: "s2", 3: "s3"}, id="every_captured_turn"),
    # a clean turn reuses the latest capture; a failed one is never a point
    pytest.param([(1, "s1", "captured"), (2, "s1", "clean"), (3, "s1", "failed")],
                 {1: "s1", 2: "s1"}, id="clean_reuses_latest_capture"),
    # a clean record that does not name the latest capture (none yet, or a stale id) is not trusted
    pytest.param([(1, "s0", "clean"), (2, "s2", "captured"), (3, "other", "clean")],
                 {2: "s2"}, id="clean_naming_a_stale_capture"),
    # neither half alone is enough: a snapshot without the conversation ...
    pytest.param([(1, "s1", "captured", None)], {}, id="no_conversation_half"),
    # ... or a conversation without the snapshot
    pytest.param([(1, None, "captured")], {}, id="no_snapshot_half"),
])
def test_which_captures_are_branch_points(tmp_path, captures, expected):
    path = turns_journal(tmp_path / "j.jsonl", *captures)
    assert {step: point.snapshot_id for step, point in branch_checkpoints(path).items()} == expected


@pytest.mark.parametrize("started,finished,call_ids,step", [
    pytest.param(["c1"], [], ["c1"], 1, id="unfinished_call"),
    pytest.param(["c1"], ["c1", "ghost"], ["c1", "ghost"], 1, id="never_started_call"),
    # the step a turn closes at is its last call's; a record closing it earlier
    # would pair the snapshot with a conversation cut mid-turn
    pytest.param(["c1", "c2"], ["c1", "c2"], ["c1", "c2"], 1, id="closed_before_last_call"),
])
def test_a_turn_is_not_closed_when_its_record_disagrees_with_its_calls(
        tmp_path, started, finished, call_ids, step):
    path = tmp_path / "j.jsonl"
    with JournalWriter(path) as journal:
        for call_id in started:
            journal.emit(E.TOOL_STARTED, call_id=call_id, turn_id="t1", name="shell", args={})
        for call_id in finished:
            journal.emit(E.TOOL_FINISHED, call_id=call_id, turn_id="t1", status="ok", output="")
        journal.emit("model.turn.completed", turn_id="t1", call_ids=call_ids, step=step)
        capture(journal, step, "t1", "s1")
    assert closed_turns(list(read_journal(path))) == {} and branch_checkpoints(path) == {}


def test_a_final_snapshot_after_uncertain_execution_is_not_a_point(tmp_path):
    path = turns_journal(tmp_path / "j.jsonl", (1, "s1", "captured"))
    with JournalWriter(path) as journal:
        journal.emit("execution.uncertain", call_id="c3", step=3, kind="transport_exception")
        capture(journal, 3, "t3", "final", final=True)
    assert set(branch_checkpoints(path)) == {1}


def test_a_closed_turn_with_no_snapshot_is_reported(tmp_path):
    path = turns_journal(tmp_path / "j.jsonl", (1, "s1", "captured"))
    assert uncaptured_turns(path) == ["t2", "t3"]


@pytest.mark.parametrize("boundary_type", [
    # on a turn.completed: a child journal's seqs restart at 1, so the prefix
    # travels as records, never as a boundary in the parent's seq numbers --
    # that would mark the child's own work as copied
    pytest.param(E.TURN_COMPLETED, id="boundary_on_turn_completed"),
    # mid-turn, after the tool result: the prefix must be flushed explicitly or
    # the child's first step merges into it and inherits copied=True
    pytest.param(E.TOOL_FINISHED, id="boundary_mid_turn"),
])
def test_inherited_prefix_ends_a_step(tmp_path, boundary_type):
    """The inherited prefix is marked copied and the child's own work is not,
    wherever the parent's boundary falls (rarely on a turn.completed)."""
    parent_records = build_journal(tmp_path / "parent.jsonl")
    boundary = next(r["seq"] for r in parent_records if r["type"] == boundary_type)
    inherited = [r for r in parent_records if r["seq"] <= boundary]

    child_path = tmp_path / "child.jsonl"
    with JournalWriter(child_path, run_id="child") as journal:
        journal.emit(E.RUN_STARTED, slot="opencode", task_prompt="branch")
        journal.emit(E.AGENT_MESSAGE, text="child work")
        journal.emit(E.TURN_COMPLETED, usage={"input_tokens": 3})

    document = journal_to_atif(read_journal(child_path), inherited=inherited)
    flags = [s["is_copied_context"] for s in document["steps"]]
    assert any(flags[:-1]), "the inherited prefix must be marked copied"
    assert flags[-1] is False, "the child's own step must not be marked copied"
    assert "child work" in (document["steps"][-1]["message"] or "")

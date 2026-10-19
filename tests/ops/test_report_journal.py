"""``ops.report.journal``: one journal read into a trajectory, and the path and text helpers."""

from __future__ import annotations

import json

import pytest

from ops.report.journal import clip, read_trajectory, relocate, step_count, summarize_command, turn_count
from tests.ops.trees import MINI_ENV, Step, Turn, actor, numbered, write_journal


def write_events(path, events):
    """A journal of exactly ``events``, numbered in order."""
    lines = [
        json.dumps({"v": 2, "seq": seq, "run_id": "r", **event}, ensure_ascii=False)
        for seq, event in enumerate(events, 1)
    ]
    path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return path


def snapshot_by_turn(trajectory):
    """Each turn's number with the snapshot id of its capture, or None when it has none."""
    return [(turn.number, turn.capture.snapshot_id if turn.capture else None) for turn in trajectory.turns]


def test_each_turn_gets_the_checkpoint_of_its_own_turn_id_not_the_next_record_in_the_file(tmp_path):
    """Turn 1's record follows turn 2's model response, and turn 2, rejected, has none."""
    journal = write_journal(
        tmp_path / "j.jsonl", turns=[actor("echo one"), Turn(()), actor("echo three")], sandbox="par"
    )

    trajectory = read_trajectory(journal)

    assert snapshot_by_turn(trajectory) == [(1, "par-0001"), (2, None), (3, "par-0003")]


def test_a_call_shows_the_models_command_without_the_env_and_bash_wrapper_of_mini(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=[actor("cd /app && grep -n 'x' f.py")])

    [call] = read_trajectory(journal).turns[0].calls

    assert (call.command, call.args["command"]) == (
        "cd /app && grep -n 'x' f.py",
        MINI_ENV + " bash -c 'cd /app && grep -n '\"'\"'x'\"'\"' f.py'",
    )


def test_a_one_word_command_also_shows_without_the_wrapper_of_mini(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=[actor("pwd")])

    [call] = read_trajectory(journal).turns[0].calls

    assert call.command == "pwd"


def test_a_failed_call_has_its_exit_code_and_its_stdout_then_its_stderr(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=[Turn((Step("make test", "FAILED 2", exit_code=2),))])

    [call] = read_trajectory(journal).turns[0].calls

    assert (call.status, call.exit_code, call.output) == ("error", 2, "FAILED 2\n[stderr]\nERR 1")


def test_a_call_finished_without_an_outcome_is_read_from_the_runtimes_envelope(tmp_path):
    envelope = {
        "stdout": "OUT",
        "stderr": "BAD",
        "exit_code": 3,
        "timed_out": True,
        "max_per_stream": 15728640,
    }
    journal = write_events(
        tmp_path / "j.jsonl",
        [
            {"type": "turn.started", "turn": 1, "source": "actor"},
            {
                "type": "tool.started",
                "call_id": "c1",
                "turn_id": "t1",
                "name": "shell",
                "args": {"command": "x"},
                "step": 1,
            },
            {
                "type": "tool.finished",
                "call_id": "c1",
                "name": "shell",
                "status": "error",
                "output": json.dumps(envelope),
            },
        ],
    )

    [call] = read_trajectory(journal).turns[0].calls

    assert (call.exit_code, call.timed_out, call.output) == (3, True, "OUT\n[stderr]\nBAD")


def test_a_torn_last_line_is_skipped_and_marks_the_trajectory_torn(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2), status=None, torn=True)

    trajectory = read_trajectory(journal)

    assert (trajectory.torn, len(trajectory.turns)) == (True, 2)


def test_a_journal_without_run_finished_has_no_status_and_sums_its_usage_updates(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2), status=None)

    trajectory = read_trajectory(journal)

    assert (trajectory.status, trajectory.usage["input_tokens"], trajectory.usage["output_tokens"]) == (
        None,
        200,
        20,
    )


def test_the_last_run_finished_gives_the_status_and_the_error(tmp_path):
    later = {"type": "run.finished", "status": "error", "error": "SandboxError: gone", "usage": {}}
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1), tail=(later,))

    trajectory = read_trajectory(journal)

    assert (trajectory.status, trajectory.error) == ("error", "SandboxError: gone")


def test_a_sweep_and_a_released_memory_snapshot_are_read_from_their_markers(tmp_path):
    swept = {"type": "snapshots.swept", "run_id": "sweep", "backend": "podman", "deleted": 2, "failed": 0}
    released = {"type": "snapshot.released", "snapshot_id": "mem-1", "kind": "memory", "reason": "resolved"}
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1), tail=(swept, released))

    trajectory = read_trajectory(journal)

    assert (trajectory.swept, trajectory.released) == (True, ("mem-1",))


def test_a_rejected_turn_keeps_the_source_of_its_rejection(tmp_path):
    journal = write_events(
        tmp_path / "j.jsonl",
        [
            {"type": "turn.started", "turn": 1, "source": "actor", "ts": "2026-10-01T00:00:00Z"},
            {
                "type": "mini.turn.rejected",
                "turn_id": "t1",
                "source": "malformed_tool_calls",
                "executed": False,
            },
            {
                "type": "turn.completed",
                "turn": 1,
                "source": "actor",
                "rejected": True,
                "ts": "2026-10-01T00:00:01Z",
            },
        ],
    )

    [turn] = read_trajectory(journal).turns

    assert (turn.rejected, turn.rejected_source) == (True, "malformed_tool_calls")


def test_a_call_that_the_runtime_rejected_is_marked_rejected_with_its_feedback(tmp_path):
    journal = write_events(
        tmp_path / "j.jsonl",
        [
            {"type": "turn.started", "turn": 1, "source": "actor"},
            {
                "type": "tool.started",
                "call_id": "c1",
                "turn_id": "t1",
                "name": "shell",
                "args": {"command": "x"},
                "step": 1,
            },
            {"type": "tool.rejected", "call_id": "c1", "feedback": "working_dir must be absolute"},
        ],
    )

    [call] = read_trajectory(journal).turns[0].calls

    assert (call.rejected, call.status, call.output) == (True, None, "working_dir must be absolute")


def test_a_call_that_mini_skipped_after_the_submitting_call_is_no_tool_step(tmp_path):
    """The skipped call never reached the sandbox, so the turn has the submitting call only."""
    journal = write_events(
        tmp_path / "j.jsonl",
        [
            {"type": "turn.started", "turn": 1, "source": "actor"},
            {
                "type": "tool.started",
                "call_id": "c1",
                "turn_id": "t1",
                "name": "shell",
                "args": {"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"},
                "step": 1,
            },
            {"type": "tool.finished", "call_id": "c1", "turn_id": "t1", "name": "shell", "status": "ok"},
            {"type": "tool.skipped", "call_id": "c2", "turn_id": "t1", "reason": "after_submit"},
            {"type": "model.turn.completed", "turn_id": "t1", "call_ids": ["c1", "c2"], "step": 1},
            {"type": "turn.completed", "turn": 1, "source": "actor"},
        ],
    )

    trajectory = read_trajectory(journal)

    assert ([call.call_id for call in trajectory.turns[0].calls], step_count(trajectory)) == (["c1"], 1)


def test_a_unicode_line_separator_inside_a_message_does_not_split_its_record(tmp_path):
    journal = write_events(
        tmp_path / "j.jsonl",
        [
            {"type": "turn.started", "turn": 1, "source": "actor"},
            {"type": "agent.message", "text": "before\u2028after\u0085tail"},
        ],
    )

    [turn] = read_trajectory(journal).turns

    assert turn.message == "before\u2028after\u0085tail"


def test_a_turn_whose_model_call_is_in_flight_counts_as_started_not_completed(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2), status=None, open_turn=True)

    counts = turn_count(read_trajectory(journal))

    assert counts == (3, 2)


def test_relocate_finds_a_moved_root_by_the_tail_of_the_recorded_path(tmp_path):
    moved = tmp_path / "now" / "exp" / "baseline" / "trials" / "t-parent" / "agent" / "trajectory.jsonl"
    moved.parent.mkdir(parents=True)
    moved.write_text("")
    anchor = tmp_path / "now" / "exp" / "sprout" / "summary.json"
    anchor.parent.mkdir(parents=True)

    found = relocate("/path/to/old/exp/baseline/trials/t-parent/agent/trajectory.jsonl", anchor)

    assert found == moved


def test_relocate_never_resolves_a_recorded_path_to_the_file_that_records_it(tmp_path):
    """A fork.origin that names its own journal's old path must not make the branch its own parent."""
    branch = tmp_path / "out" / "trials" / "t-branch" / "agent" / "trajectory.jsonl"
    branch.parent.mkdir(parents=True)
    branch.write_text("")

    found = relocate("/path/to/old/out/trials/t-branch/agent/trajectory.jsonl", branch)

    assert found is None


@pytest.mark.parametrize(
    "text,limit,expected",
    [
        ("H" * 6 + "T" * 4, 0, "H" * 6 + "T" * 4),
        ("H" * 6 + "T" * 4, 10, "H" * 6 + "T" * 4),
        ("H" * 6 + "T" * 4, 9, "HHHHHH\n… [1 chars elided] …\nTTT"),
        ("H" * 20 + "T" * 10, 9, "HHHHHH\n… [21 chars elided] …\nTTT"),
    ],
    ids=["zero-is-unlimited", "at-the-limit", "one-over", "head-two-thirds-tail-one-third"],
)
def test_clip_keeps_two_thirds_of_the_limit_from_the_head_and_one_third_from_the_tail(text, limit, expected):
    assert clip(text, limit) == expected


@pytest.mark.parametrize(
    "command,width,expected",
    [
        ("set -e; cd /app; cat f.txt", 40, "cat f.txt"),
        ("set -euo pipefail\ncd /app && make test", 40, "make test"),
        ("cd /app && CONTEXT=abs go test ./... 2>&1 | tail -8", 20, "CONTEXT=abs go test …"),
        ("cat > /tmp/x << 'EOF'\n\tbody\nEOF", 40, "cat > /tmp/x << 'EOF' body EOF"),
        ("cd /app", 40, "cd /app"),
    ],
    ids=["set-e-and-cd", "preamble-lines", "cut-at-width", "whitespace-collapsed", "a-preamble-alone-stays"],
)
def test_summarize_command_drops_the_set_and_cd_preamble_and_cuts_one_line(command, width, expected):
    assert summarize_command(command, width=width) == expected

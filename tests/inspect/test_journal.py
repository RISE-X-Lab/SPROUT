"""One journal -> a Trajectory: grouping, pairing, the live/torn/swept cases and re-rooting."""

from __future__ import annotations

import json

from ops.report.journal import (
    clip, minutes, read_trajectory, step_count, summarize_command, turn_count,
)

from tests.inspect.conftest import FIXTURES, make_journal, simple_turns, wrapped


def test_turns_group_calls_and_pair_checkpoints_by_turn_id_not_position(tmp_path):
    """Turn 2's checkpoint record is written after turn 3's model response; it still belongs to turn 2."""
    path, _ = make_journal(tmp_path / "j.jsonl", run_id="r", sandbox="par",
                           turns=[("actor", [("cd /app && ls", "files", 0), ("grep x", "", 1)]),
                                  ("actor", [("echo two", "two", 0)]),
                                  ("actor", [("echo three", "three", 0)])])
    t = read_trajectory(path)
    assert [turn.number for turn in t.turns] == [1, 2, 3]
    assert t.turns[1].capture.snapshot_id == "par-0002" and t.turns[1].capture.turn == 2
    assert t.turns[2].capture.final and t.turns[2].capture.snapshot_id == "par-0003"
    assert [c.step for turn in t.turns for c in turn.calls] == [1, 2, 3, 4]
    assert t.turns[0].calls[0].command == "cd /app && ls"            # mini's env ... bash -c wrapper stripped
    assert t.turns[0].calls[0].args["command"] == wrapped("cd /app && ls")
    assert (t.turns[0].calls[1].exit_code, t.turns[0].calls[1].status) == (1, "error")
    assert "[stderr]\nERR 2" in t.turns[0].calls[1].output                 # from outcome, not the envelope string
    assert all(c.branch_point for c in t.captures) and t.turns[0].step == 2 and t.turns[0].closed
    assert t.status == "completed" and t.model_calls == 3 and t.tool_calls == 4
    assert t.usage["input_tokens"] == 300 and step_count(t) == 4


def test_real_branch_journal_parses():
    t = read_trajectory(FIXTURES / "r1b1.trajectory.jsonl")
    assert t.run_id == "abs-module-cache-flags-r1b1-abs-absolute-fix-precomm"
    assert (t.origin.branch_step, t.origin.snapshot_id, t.origin.branch_guidance) == (
        136, "sprout-b2867d643e67-0105", "assistant-turn")
    assert t.origin.conversation_cut.startswith("bf5a28a1") and t.origin.parent_journal_sha256.startswith("e1320561")
    assert [turn.source for turn in t.turns] == ["reviewer", "actor", "actor"]
    assert len(t.turns[0].assistant_turn["tool_calls"]) == 5
    assert [c.call_id for c in t.turns[0].calls] == ["rev136-abs-test", "rev136-abs-fix", "rev136-abs-verify",
                                                      "rev136-abs-e2e", "rev136-abs-commit"]
    assert step_count(t) == 8 and [c.branch_point for c in t.captures] == [True, True, True]
    assert t.status == "completed" and t.image == t.origin.snapshot_id and t.backend == "podman"
    assert (t.model_calls, t.tool_calls) == (2, 8) and t.sandbox_id == "sprout-69b1adc970d8"
    envelope = t.turns[0].calls[4]      # a JSON envelope (stderr present): unpacked, not shown raw
    assert envelope.exit_code == 0 and envelope.status == "ok" and not envelope.output.startswith("{")


def test_a_live_journal_has_no_status_and_a_torn_line_is_skipped(tmp_path):
    path, _ = make_journal(tmp_path / "live.jsonl", run_id="r", sandbox="s", turns=simple_turns(2),
                           status=None, open_turn=True, torn=True)
    t = read_trajectory(path)
    assert t.status is None and t.torn and turn_count(t) == (3, 2)
    assert t.last_ts and t.usage["input_tokens"] == 200       # summed from usage.updated, no run.finished
    assert t.turns[1].capture is None                          # emitted only after the next model response
    assert minutes(t) is not None


def test_run_finished_twice_takes_the_last_and_marks_uncertain(tmp_path):
    uncertain = json.dumps({"v": 2, "type": "execution.uncertain", "seq": 900, "run_id": "r", "call_id": "c2",
                            "step": 2, "kind": "runtime_timeout"})
    error = json.dumps({"v": 2, "type": "run.finished", "ts": "2026-09-30T21:00:00.000Z", "seq": 901, "run_id": "r",
                        "status": "error", "error": "SandboxError: gone"})
    path, _ = make_journal(tmp_path / "j.jsonl", run_id="r", sandbox="s", turns=simple_turns(2),
                           extra_tail=(uncertain, error))
    t = read_trajectory(path)
    assert (t.status, t.error) == ("error", "SandboxError: gone") and t.uncertain


def test_swept_and_released_markers_are_read(tmp_path):
    swept = json.dumps({"v": 2, "type": "snapshots.swept", "seq": 99, "run_id": "sweep", "agent_id": "sweep",
                        "backend": "podman", "deleted": 2, "failed": 0})
    released = json.dumps({"v": 2, "type": "snapshot.released", "seq": 100, "run_id": "evaluate",
                           "snapshot_id": "mem-1", "kind": "memory", "reason": "resolved"})
    path, _ = make_journal(tmp_path / "j.jsonl", run_id="r", sandbox="s", turns=simple_turns(2),
                           extra_tail=(swept, released))
    t = read_trajectory(path)
    stamped = [json.loads(line) for line in path.read_text().splitlines() if '"ts"' in line]
    assert t.swept and t.released == ("mem-1",)
    assert t.last_ts == stamped[-1]["ts"] and stamped[-1]["type"] == "checkpoint.captured"   # the appended lines have no ts


def test_clip_keeps_head_and_tail_and_zero_means_unlimited():
    text = "H" * 20 + "T" * 10
    clipped = clip(text, 9)
    assert clipped.startswith("HHHHHH") and clipped.endswith("TTT") and "[21 chars elided]" in clipped
    assert clip(text, 0) == text and clip(text, 100) == text


def test_summarize_command_strips_cd_and_set_e():
    assert summarize_command("set -e; cd /app; cat > /tmp/x.txt << 'EOF'\n\tbody\nEOF") == "cat > /tmp/x.txt << 'EOF' body EOF"
    assert summarize_command("cd /app && CONTEXT=abs go test ./... 2>&1 | tail -8", width=20) == "CONTEXT=abs go test …"
    # a reviewer's multi-line command: the preamble lines go too
    assert summarize_command("set -e\ncd /app\ngit checkout -b improve-abs && go build ./...", width=80) == "git checkout -b improve-abs && go build ./..."
    assert summarize_command("set -euo pipefail\ncd /app && make test") == "make test"
    assert summarize_command("cd /app") == "cd /app"                 # nothing after the preamble: keep it


def test_unicode_line_separator_inside_a_message_survives(tmp_path):
    path = tmp_path / "u.jsonl"
    events = [{"v": 2, "type": "turn.started", "seq": 1, "run_id": "r", "turn": 1, "source": "actor"},
              {"v": 2, "type": "agent.message", "seq": 2, "run_id": "r", "text": "a b"}]
    path.write_text("\n".join(json.dumps(e, ensure_ascii=False) for e in events) + "\n", encoding="utf-8")
    assert read_trajectory(path).turns[0].message == "a b"

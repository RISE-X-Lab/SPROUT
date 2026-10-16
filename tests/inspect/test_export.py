"""One trajectory as markdown/json, a branch stitched onto its parent by turn."""

from __future__ import annotations

import json
import shutil

from sprout.inspect import cli
from sprout.inspect.export import lineage, render_json, render_markdown

from tests.inspect.conftest import FIXTURES, make_journal, simple_turns


def origin_at(parent, sha, step, **extra) -> dict:
    return {"parent_journal": str(parent), "branch_step": step, "snapshot_id": "par-%04d" % step,
            "conversation_cut": "t%d" % step, "branch_guidance": "assistant-turn",
            "assistant_turn_source": "reviewer", "parent_journal_sha256": sha, **extra}


def branch_turns() -> list:
    return [("reviewer", [("go test ./...", "ok", 0), ("git commit -am fix", "[fix]", 0)]),
            ("actor", [("echo done", "done", 0)])]


def test_markdown_covers_prompt_steps_results_checkpoints_message_and_usage(tmp_path):
    path, _ = make_journal(tmp_path / "parent.jsonl", run_id="parent", sandbox="par", turns=simple_turns(2),
                           prompt="PROMPT_parent")
    text = render_markdown(lineage(path), max_output_chars=0)
    assert "PROMPT_parent" in text
    assert "#### step 1 · shell" in text and "#### step 2 · shell" in text
    assert "ACTION_2" in text and "RESULT_2" in text and "MESSAGE 2" in text and "> thinking: THINKING 1" in text
    assert "snapshot `par-0002`" in text and "steps (tool calls): 2" in text and "tokens: input 200" in text
    assert text.index("ACTION_1") < text.index("MESSAGE 2") < text.index("ACTION_2")


def test_a_branch_is_stitched_onto_its_parent_up_to_and_including_the_branch_step(tmp_path):
    parent, sha = make_journal(tmp_path / "p/agent/trajectory.jsonl", run_id="t-parent", sandbox="par",
                               turns=simple_turns(5))
    branch, _ = make_journal(tmp_path / "b/agent/trajectory.jsonl", run_id="t-r1b1", sandbox="brn",
                             turns=branch_turns(), origin=origin_at(parent, sha, 3), prompt="", image="par-0003")
    text = render_markdown(lineage(branch), max_output_chars=0)
    assert "PROMPT" in text                                   # the root ancestor's prompt
    boundary = text.index("## ✂ BRANCH POINT — t-parent cut after step 3 (turn 3)")
    assert text.index("ACTION_1") < text.index("ACTION_2") < text.index("ACTION_3") < boundary
    assert "_checkpoint after step 3 → snapshot `par-0003` (captured, disk-only)_" in text
    assert "snapshot `par-0003` · conversation cut `t3` · guidance assistant-turn" in text
    assert "parent steps 4–5 (turns 4–5) are not part of this branch's history" in text
    assert "sha256 %s… matches fork.origin" % sha[:8] in text
    assert "ACTION_4" not in text and "ACTION_5" not in text
    reviewer = text.index("REVIEWER-AUTHORED assistant turn")
    assert boundary < reviewer < text.index("#### step 1 · shell · call `c1`", boundary)   # branch steps restart at 1
    assert "from snapshot `par-0003`" in text and "## Run finished" in text
    alone = render_markdown(lineage(branch, prefix=False), max_output_chars=0)
    assert "BRANCH POINT — parent" in alone and "(prefix not rendered)" in alone and "ACTION_3" not in alone
    assert "_(parent prefix not rendered: the task prompt is in the parent's run.started)_" in alone
    assert "not found" not in alone                                   # the parent exists; it was not rendered


def test_recursion_through_a_branch_of_a_branch(tmp_path):
    grand, gsha = make_journal(tmp_path / "g/agent/trajectory.jsonl", run_id="g", sandbox="par", turns=simple_turns(5))
    parent, psha = make_journal(tmp_path / "p/agent/trajectory.jsonl", run_id="p", sandbox="mid",
                                turns=branch_turns() + simple_turns(1, prefix="MID"),
                                origin=origin_at(grand, gsha, 3), prompt="")
    leaf, _ = make_journal(tmp_path / "l/agent/trajectory.jsonl", run_id="l", sandbox="leaf", turns=branch_turns(),
                           origin={**origin_at(parent, psha, 3), "snapshot_id": "mid-0002", "conversation_cut": "t2"},
                           prompt="")
    chain = lineage(leaf)
    assert [s.trajectory.run_id for s in chain.segments] == ["g", "p", "l"]
    assert [s.cut_step for s in chain.segments] == [3, 3, None] and [s.cut_turn for s in chain.segments] == [3, 2, None]
    text = render_markdown(chain, max_output_chars=0)
    assert text.count("## ✂ BRANCH POINT") == 2 and "MID_1" not in text
    document = render_json(chain, max_output_chars=0)
    assert [s.get("run_id") or "boundary" for s in document["segments"]] == ["g", "boundary", "p", "boundary", "l"]
    selfish, _ = make_journal(tmp_path / "s/agent/trajectory.jsonl", run_id="s", sandbox="s", turns=branch_turns(),
                              origin=origin_at(tmp_path / "s/agent/trajectory.jsonl", "0" * 64, 1), prompt="")
    assert "cycle" in render_markdown(lineage(selfish), max_output_chars=0)
    assert cli.main(["export", str(selfish)]) == 0


def test_a_missing_or_modified_parent_is_reported_not_refused(tmp_path, capsys):
    gone = tmp_path / "nowhere/trials/t-parent/agent/trajectory.jsonl"
    branch, _ = make_journal(tmp_path / "b/agent/trajectory.jsonl", run_id="b", sandbox="brn", turns=branch_turns(),
                             origin=origin_at(gone, "a" * 64, 3), prompt="")
    assert cli.main(["export", str(branch)]) == 0
    text = capsys.readouterr().out
    assert "parent journal not found: %s" % gone in text and "branch_step 3" in text and "cut t3" in text
    assert text.count("## ✂ BRANCH POINT") == 1 and "inherited prefix" not in text
    assert "_(parent journal not found; the task prompt is in the parent's run.started)_" in text
    assert "REVIEWER-AUTHORED" in text and "ACTION_" not in text

    parent, sha = make_journal(tmp_path / "p/agent/trajectory.jsonl", run_id="p", sandbox="par", turns=simple_turns(4))
    branch, _ = make_journal(tmp_path / "b2/agent/trajectory.jsonl", run_id="b2", sandbox="brn", turns=branch_turns(),
                             origin=origin_at(parent, sha, 2), prompt="")
    swept = json.dumps({"v": 2, "type": "snapshots.swept", "seq": 999, "run_id": "sweep", "backend": "podman",
                        "deleted": 4, "failed": 0})
    parent.write_text(parent.read_text() + swept + "\n")
    text = render_markdown(lineage(branch), max_output_chars=0)
    assert "sha256 differs from fork.origin (recorded %s…" % sha[:8] in text and "rendered anyway" in text
    assert "ACTION_1" in text and "ACTION_2" in text and "ACTION_3" not in text
    assert "snapshots swept: ids no longer resolve" in text

    moved_root = tmp_path / "x"
    real_parent, sha = make_journal(moved_root / "baseline/tasks/t/baseline/attempt-1/trials/t-parent/agent/trajectory.jsonl",
                                    run_id="t-parent", sandbox="par", turns=simple_turns(4))
    recorded = "/old/ash-runs/x/baseline/tasks/t/baseline/attempt-1/trials/t-parent/agent/trajectory.jsonl"
    branch, _ = make_journal(moved_root / "sprout/tasks/t/sprout/trials/t-r1b1/agent/trajectory.jsonl", run_id="t-r1b1",
                             sandbox="brn", turns=branch_turns(), origin=origin_at(recorded, sha, 2), prompt="")
    chain = lineage(branch)
    assert chain.segments[0].trajectory.path == real_parent and chain.segments[0].sha256_ok is True
    assert "sha256 %s… matches" % sha[:8] in render_markdown(chain, max_output_chars=0)

    # the real r1b1 journal, exported from a copy under tmp_path: "not found" relies on its recorded
    # /path/to/ash-runs/… parent not existing on this host, and on relocate() finding no moved
    # one beside the journal -- which it does from a checkout under an experiments directory
    # (~/sprout-runs/qwen/<root>/baseline/tasks/... is the recorded path's own tail).
    real = tmp_path / "r1b1.trajectory.jsonl"
    shutil.copyfile(FIXTURES / "r1b1.trajectory.jsonl", real)
    assert cli.main(["export", str(real)]) == 0
    text = capsys.readouterr().out
    assert "parent journal not found: /path/to/ash-runs/" in text
    assert "branch_step 136" in text and "sprout-b2867d643e67-0105" in text
    assert "REVIEWER-AUTHORED assistant turn" in text and "call `rev136-abs-test`" in text
    assert "## Run finished" in text and "model calls: 2" in text


def test_max_output_chars_keeps_head_and_tail_and_zero_is_full(tmp_path):
    long = "H" * 400 + "MIDDLE" + "T" * 400
    path, _ = make_journal(tmp_path / "j.jsonl", run_id="r", sandbox="s", turns=[("actor", [("cat f", long, 0)])])
    text = render_markdown(lineage(path), max_output_chars=90)
    assert "MIDDLE" not in text and "chars elided" in text and "HHHH" in text and "TTTT" in text
    assert "MIDDLE" in render_markdown(lineage(path), max_output_chars=0)


def test_tool_result_envelope_is_unpacked_with_stderr_and_exit_code(tmp_path):
    path = tmp_path / "e.jsonl"
    envelope = json.dumps({"stdout": "OUT", "stderr": "BAD", "exit_code": 2, "running": False, "timed_out": False,
                           "stdout_bytes": 3, "stderr_bytes": 3, "stdout_truncated": False,
                           "stderr_truncated": False, "max_per_stream": 15728640})
    events = [{"type": "turn.started", "turn": 1, "source": "actor"},
              {"type": "tool.started", "call_id": "c1", "turn_id": "t1", "name": "shell",
               "args": {"command": "false"}, "step": 1},
              {"type": "tool.finished", "call_id": "c1", "turn_id": "t1", "name": "shell", "status": "error",
               "output": envelope},
              {"type": "model.turn.completed", "turn_id": "t1", "call_ids": ["c1"], "step": 1, "source": "actor"},
              {"type": "turn.completed", "turn": 1, "source": "actor"}]
    path.write_text("".join(json.dumps({"v": 2, "seq": i, "run_id": "r", **e}) + "\n" for i, e in enumerate(events, 1)))
    text = render_markdown(lineage(path), max_output_chars=0)
    assert "**result (exit 2, status error):**" in text and "OUT\n[stderr]\nBAD" in text and "max_per_stream" not in text


def test_a_live_journal_ends_with_run_still_open(tmp_path):
    path, _ = make_journal(tmp_path / "live.jsonl", run_id="r", sandbox="s", turns=simple_turns(2), status=None,
                           open_turn=True, torn=True)
    text = render_markdown(lineage(path), max_output_chars=0)
    assert "## Run still open — no run.finished; last event 2026-09-30T20:00:" in text and "last line torn" in text
    assert "### turn 3 · actor · model call in flight" in text and "## Run finished" not in text


def test_user_hint_branch_prints_the_hint_as_a_user_note(tmp_path):
    parent, sha = make_journal(tmp_path / "p/agent/trajectory.jsonl", run_id="p", sandbox="par", turns=simple_turns(3))
    origin = {**origin_at(parent, sha, 2), "branch_guidance": "user-hint", "actor_hint": "HINT TEXT",
              "hint_delivery": "reviewer-direct"}
    branch, _ = make_journal(tmp_path / "b/agent/trajectory.jsonl", run_id="b", sandbox="brn",
                             turns=simple_turns(1, prefix="AFTER"), origin=origin, prompt="Rendered: HINT TEXT")
    text = render_markdown(lineage(branch), max_output_chars=0)
    assert "## Branch note delivered as a user message" in text and "Rendered: HINT TEXT" in text
    assert "guidance user-hint" in text and lineage(branch).segments[-1].trajectory.origin.hint == "HINT TEXT"


def test_json_export_shape_and_output_file(tmp_path, capsys):
    parent, sha = make_journal(tmp_path / "p/agent/trajectory.jsonl", run_id="p", sandbox="par", turns=simple_turns(4))
    branch, _ = make_journal(tmp_path / "b/agent/trajectory.jsonl", run_id="b", sandbox="brn", turns=branch_turns(),
                             origin=origin_at(parent, sha, 3), prompt="")
    document = render_json(lineage(branch), max_output_chars=50)
    assert document["run_id"] == "b" and document["prompt"] == "PROMPT" and document["prefix"] is True
    assert [row["run_id"] for row in document["lineage"]] == ["p", "b"] and document["lineage"][0]["cut_turn"] == 3
    assert document["missing_parents"] == [] and document["lineage"][0]["sha256_ok"] is True
    parent_segment, boundary, leaf = document["segments"]
    assert parent_segment["inherited"] and [t["turn"] for t in parent_segment["turns"]] == [1, 2, 3]
    assert parent_segment["turns"][2]["checkpoint"]["snapshot_id"] == "par-0003"
    assert boundary["boundary"]["branch_step"] == 3 and boundary["boundary"]["conversation_cut"] == "t3"
    assert leaf["turns"][0]["source"] == "reviewer" and leaf["turns"][0]["assistant_turn"]["role"] == "assistant"
    assert leaf["turns"][0]["calls"][0]["command"] == "go test ./..." and document["status"] == "completed"
    out = tmp_path / "out.json"
    assert cli.main(["export", str(branch), "--format", "json", "-o", str(out)]) == 0
    assert capsys.readouterr().out.startswith("wrote %s" % out)
    assert json.loads(out.read_text())["run_id"] == "b"


def test_a_rejected_turn_names_why_it_was_rejected(tmp_path):
    """mini.turn.rejected carries the source (schema_validation, malformed_tool_calls,
    no_tool_calls); the export says which, instead of assuming a schema failure."""
    path = tmp_path / "r/agent/trajectory.jsonl"
    path.parent.mkdir(parents=True)
    events = [
        {"type": "run.started", "slot": "mini-swe-agent", "slot_version": "2.4.6", "model": "m", "task_prompt": "P"},
        {"type": "turn.started", "turn": 1, "source": "actor", "ts": "2026-10-01T00:00:00Z"},
        {"type": "agent.message", "text": "I will run it"},
        {"type": "mini.turn.rejected", "turn_id": "t1", "source": "malformed_tool_calls", "executed": False,
         "problems": [{"index": 0, "id": "call_1", "error": "function.name is blank"}]},
        {"type": "turn.completed", "turn": 1, "source": "actor", "rejected": True, "ts": "2026-10-01T00:00:01Z"},
        {"type": "turn.started", "turn": 2, "source": "actor", "ts": "2026-10-01T00:00:02Z"},
        {"type": "turn.completed", "turn": 2, "source": "actor", "rejected": True, "ts": "2026-10-01T00:00:03Z"},
        {"type": "run.finished", "status": "completed", "usage": {}},
    ]
    path.write_text("".join(json.dumps(dict(e, run_id="r", seq=i)) + "\n" for i, e in enumerate(events, 1)))
    text = render_markdown(lineage(path), max_output_chars=0)
    assert "REJECTED (malformed_tool_calls): no call executed" in text
    assert "REJECTED (reason not recorded): no call executed" in text, "a rejection without a source says so"
    document = render_json(lineage(path), max_output_chars=0)
    turns = document["segments"][0]["turns"]
    assert [t["rejected_source"] for t in turns] == ["malformed_tool_calls", None]
    assert all(t["rejected"] for t in turns)

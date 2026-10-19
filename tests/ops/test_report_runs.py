"""``python -m ops.report runs``: the progress and results of an experiment, a method root or a gate root."""

from __future__ import annotations

import json
import os

from examples.qwen.report import build_report
from ops.report import cli
from tests.ops.trees import (
    numbered,
    write_attempt,
    write_group,
    write_journal,
    write_json,
    write_lock,
    write_manifest,
    write_retained,
    write_stage,
    write_summary,
    write_task_state,
)

RUNNING_ACTOR = {"ready": False, "samples": [{"actor": {"operation": "rollout", "state": "running"}}]}


def baseline_root(tmp_path, tasks, **manifest):
    """The baseline method root of an experiment at ``tmp_path/exp``, its runner alive in this process."""
    root = tmp_path / "exp" / "baseline"
    write_manifest(root, method="baseline", tasks=tasks, **manifest)
    write_lock(root, method="baseline", pid=os.getpid())
    return root


def runs(capsys, *argv):
    assert cli.main(["runs", *map(str, argv)]) == 0
    return capsys.readouterr().out


def lines_after(text, prefix, count):
    """``count`` lines of ``text`` from the first one that starts with ``prefix``, or none if no line does."""
    lines = text.splitlines()
    first = next((index for index, line in enumerate(lines) if line.startswith(prefix)), len(lines))
    return lines[first : first + count]


def row(text, label):
    """The lines of a method's report whose label column is ``label``."""
    return [line for line in text.splitlines() if line[3:].startswith(label + " ")]


def test_runs_counts_the_tasks_of_each_state(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha", "beta", "gamma", "delta"])
    write_task_state(root, "alpha", complete=True)
    write_task_state(root, "beta", error="ValueError: Missing binary verdict")
    write_stage(root / "tasks" / "gamma" / "baseline" / "attempt-1", state="running")

    text = runs(capsys, root.parent)

    assert row(text, "tasks") == ["   tasks       1 complete · 1 errored · 1 in progress · 1 not started"]


def test_runs_counts_the_finished_attempts_their_resolve_rate_and_their_duration(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha", "beta"])
    write_attempt(
        root / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False, seconds=300.0
    )
    write_attempt(root / "tasks" / "alpha", 2, artifacts=tmp_path / "artifacts", resolved=True, seconds=900.0)
    write_attempt(root / "tasks" / "beta", 1, artifacts=tmp_path / "artifacts", resolved=True, seconds=600.0)
    write_stage(root / "tasks" / "beta" / "baseline" / "attempt-2", state="errored")
    write_stage(root / "tasks" / "beta" / "baseline" / "attempt-3", state="running")

    text = runs(capsys, root.parent)

    assert row(text, "attempts") == [
        "   attempts    3 finished (2 resolved, 66.7%) · 1 running · 1 errored · mean 10.0 min · max 15.0 min"
    ]


def test_runs_gives_the_attempt_1_rate_and_the_failed_parents_retained_for_branching(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha", "beta", "gamma"])
    write_retained(
        write_attempt(root / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False)
    )
    write_attempt(root / "tasks" / "beta", 1, artifacts=tmp_path / "artifacts", resolved=False)
    write_attempt(root / "tasks" / "gamma", 1, artifacts=tmp_path / "artifacts", resolved=True)

    text = runs(capsys, root.parent)

    assert row(text, "attempt-1") == [
        "   attempt-1   3 finished, 1 resolved (33.3%) · 1 failed parents retained "
        "(retained-checkpoints.json)"
    ]


def test_runs_lists_the_tasks_that_an_attempt_resolved(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha", "beta", "gamma"])
    write_attempt(root / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False)
    write_attempt(root / "tasks" / "alpha", 2, artifacts=tmp_path / "artifacts", resolved=True)
    write_attempt(root / "tasks" / "beta", 1, artifacts=tmp_path / "artifacts", resolved=False)
    write_attempt(root / "tasks" / "gamma", 1, artifacts=tmp_path / "artifacts", resolved=True)

    text = runs(capsys, root.parent)

    assert row(text, "resolved") == ["   resolved    alpha gamma"]


def test_a_live_attempt_through_the_run_store_shows_what_its_open_group_waits_for(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha"])
    output = root / "tasks" / "alpha" / "baseline" / "attempt-1"
    write_stage(output, state="running")
    write_summary(output, attempts=[], method="baseline", complete=False)
    write_group(output, 0, RUNNING_ACTOR)

    text = runs(capsys, root.parent)

    assert row(text, "in progress") == [
        "   in progress alpha   attempt-1   no journal yet   actor running   last event ?"
    ]


def test_a_live_attempt_with_a_harbor_trial_shows_its_turn_its_steps_and_its_last_event(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha"])
    output = root / "tasks" / "alpha" / "baseline" / "attempt-1"
    write_stage(output, state="running")
    write_summary(output, attempts=[], method="baseline", complete=False)
    write_journal(
        output / "trials" / "alpha-parent" / "agent" / "trajectory.jsonl",
        turns=numbered(4),
        status=None,
        open_turn=True,
    )

    text = runs(capsys, root.parent)

    assert row(text, "in progress") == [
        "   in progress alpha   attempt-1   turn 5 (4 closed, 4 steps)   actor running   last event "
        "2026-09-30T20:00:43Z"
    ]


def test_a_dead_runner_makes_its_unfinished_stages_stale(tmp_path, capsys, monkeypatch):
    def no_such_process(pid, signal):
        raise ProcessLookupError(pid)

    root = baseline_root(tmp_path, ["alpha"])
    write_lock(root, method="baseline", pid=4242)
    write_stage(root / "tasks" / "alpha" / "baseline" / "attempt-1", state="running")
    monkeypatch.setattr(os, "kill", no_such_process)

    text = runs(capsys, root.parent)

    assert row(text, "tasks") + row(text, "attempts") == [
        "   tasks       0 complete · 0 errored · 0 in progress · 1 stale · 0 not started",
        "   attempts    0 finished (0 resolved, n/a) · 0 running · 0 errored · "
        "1 stale (runner dead; these stages will never finish)",
    ]


def test_a_receipt_without_a_log_is_reported_as_unlaunched_with_its_hand_added_keys(tmp_path, capsys):
    """The runner opens a stage's log before it launches it, so a receipt alone was written by hand."""
    root = baseline_root(tmp_path, ["alpha", "beta"])
    write_stage(
        root / "tasks" / "alpha" / "baseline" / "attempt-1", state="unlaunched", sentinel="handed over"
    )
    write_stage(
        root / "tasks" / "beta" / "baseline" / "attempt-1", state="unlaunched", sentinel="handed over"
    )

    text = runs(capsys, root.parent)

    assert row(text, "tasks") + row(text, "attempts") + lines_after(text, "   unlaunched", 3) == [
        "   tasks       0 complete · 0 errored · 0 in progress · 2 unlaunched · 0 not started",
        "   attempts    0 finished (0 resolved, n/a) · 0 running · 0 errored · 2 unlaunched",
        "   unlaunched  2 receipts without a log or output directory (this runner never launched them)",
        "                 2× sentinel: handed over",
        "               alpha/attempt-1 beta/attempt-1",
    ]


def test_errors_are_grouped_by_cause_with_the_paths_in_a_task_error_generalized(tmp_path, capsys):
    """A task error that repeats its errored stage's error is that stage's, and counts once."""
    root = baseline_root(tmp_path, ["alpha", "beta", "gamma"])
    write_stage(
        root / "tasks" / "alpha" / "baseline" / "attempt-1",
        state="errored",
        log="RuntimeError: worktree not clean\n",
    )
    write_task_state(root, "alpha", error="Stage exited 1; see attempt-1.log")
    write_stage(
        root / "tasks" / "beta" / "baseline" / "attempt-1",
        state="errored",
        log="RuntimeError: worktree not clean\n",
    )
    write_task_state(root, "beta", error="Stage exited 1; see attempt-1.log")
    retained = root / "tasks" / "gamma" / "baseline" / "attempt-1"
    write_task_state(
        root,
        "gamma",
        error="Incomplete stage retained at %s; choose a fresh experiment root to retry" % retained,
    )

    text = runs(capsys, root.parent)

    assert lines_after(text, "   errors", 2) == [
        "   errors        2× RuntimeError: worktree not clean   (attempt-1: 2)",
        "                 1× Incomplete stage retained at <path>; choose a fresh experiment root to retry"
        "   (baseline.task.json: 1)",
    ]


def trial_receipt_attempt(root):
    """A complete attempt whose Harbor trial holds the podman receipt of its agent sandbox."""
    output = root / "tasks" / "alpha" / "baseline" / "attempt-1"
    write_stage(output, state="complete")
    trial = output / "trials" / "alpha-parent"
    write_journal(trial / "agent" / "trajectory.jsonl", turns=numbered(1))
    write_json(
        trial / "alpha-parent__env.podman.json",
        {"sandbox_id": "sbx", "store": "/path/to/store", "overlay": None},
    )


def test_the_backend_comes_from_the_trial_receipts_when_the_manifest_records_none(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha"], backend=None, store=None)
    trial_receipt_attempt(root)

    text = runs(capsys, root.parent)

    assert text.splitlines()[1] == (
        "== baseline   1 tasks · deepswe · m · podman /path/to/store (from receipts) · commit ce125c7d3b7c · "
        "runner pid %d alive · metrics-baseline.json: not written yet" % os.getpid()
    )


def test_a_manifest_and_trials_that_name_no_backend_say_backend_unknown(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha"], backend=None, store=None)
    write_attempt(root / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=True)

    text = runs(capsys, root.parent)

    assert text.splitlines()[1] == (
        "== baseline   1 tasks · deepswe · m · backend unknown · commit ce125c7d3b7c · "
        "runner pid %d alive · metrics-baseline.json: not written yet" % os.getpid()
    )


def test_runs_summarizes_the_metrics_file_that_the_report_of_examples_qwen_writes(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha"])
    write_attempt(root / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False)
    write_attempt(root / "tasks" / "alpha", 2, artifacts=tmp_path / "artifacts", resolved=True)
    write_attempt(root / "tasks" / "alpha", 3, artifacts=tmp_path / "artifacts", resolved=False)
    write_attempt(root / "tasks" / "alpha", 4, artifacts=tmp_path / "artifacts", resolved=False)
    write_json(root / "metrics-baseline.json", build_report(root, "baseline"))

    text = runs(capsys, root.parent)

    assert row(text, "metrics-baseline.json") + row(text, "comparison") == [
        "   metrics-baseline.json  pass@1: completed 1/1 · Acc 0.0% · Success@1 0.0% @5 0.0% @8 0.0% · "
        "recovery 0/1 (0.0%) · graded 1 · positive 0",
        "   metrics-baseline.json  baseline: completed 1/1 · Acc 0.0% · Success@1 0.0% @5 100.0% @8 100.0% · "
        "recovery 1/1 (100.0%) · graded 4 · positive 1",
        "   comparison   pass@1: Resolve Rate 0.00 · Recovery 0.00 · Steps 4 · Tokens (M) 0.00 · complete",
        "   comparison   pass@4 (baseline): Resolve Rate 1.00 · Recovery 1.00 · Steps 16 · Tokens (M) 0.00 · "
        "complete",
    ]


def test_a_branch_method_root_finds_its_moved_baseline_and_counts_skipped_and_recovered_tasks(
    tmp_path, capsys
):
    baseline = baseline_root(tmp_path, ["alpha", "beta"])
    write_retained(
        write_attempt(baseline / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False)
    )
    write_attempt(baseline / "tasks" / "beta", 1, artifacts=tmp_path / "artifacts", resolved=True)
    sprout = tmp_path / "exp" / "sprout"
    write_manifest(
        sprout, method="sprout", tasks=["alpha", "beta"], baseline_root="/path/to/old/exp/baseline"
    )
    output = sprout / "tasks" / "alpha" / "sprout"
    write_stage(output, state="complete", method="sprout")
    parent = write_journal(tmp_path / "artifacts" / "p.jsonl", turns=numbered(1))
    rows = [
        {"name": "parent", "resolved": False, "journal": str(parent)},
        {"name": "r1b1-fix", "round": 1, "resolved": True},
    ]
    write_summary(output, attempts=rows)
    write_task_state(sprout, "alpha", complete=True, methods={"sprout": {"status": "complete"}})
    write_task_state(sprout, "beta", complete=True, methods={"sprout": {"status": "skipped_initial_success"}})

    text = runs(capsys, sprout)

    assert row(text, "attempt-1") + row(text, "branch") + row(text, "resolved") == [
        "   attempt-1   2 finished, 1 resolved (50.0%) · 1 failed parents retained "
        "(retained-checkpoints.json)",
        "   branch      1 skipped (attempt-1 resolved) · recovered 1/1 failed parents",
        "   resolved    alpha",
    ]


def test_runs_on_a_gate_root_counts_its_checks(tmp_path, capsys):
    (tmp_path / "gate.jsonl").write_text(
        '{"task": "t", "mode": "oracle", "ok": true}\n'
        '{"task": "t", "mode": "nop", "ok": false, "error": "x"}\n'
        '{"task": "t", "mo'
    )

    text = runs(capsys, tmp_path)

    assert text == "gate root %s: 2 checks in gate.jsonl, ok 1, failures 1 (see summary.json)\n" % tmp_path


def test_the_json_of_runs_has_the_counts_of_the_text(tmp_path, capsys):
    root = baseline_root(tmp_path, ["alpha", "beta"])
    write_task_state(root, "alpha", complete=True)

    document = json.loads(runs(capsys, root.parent, "--format", "json"))

    assert document["methods"][0]["task_counts"] == {
        "complete": 1,
        "errored": 0,
        "in-progress": 0,
        "stale": 0,
        "unlaunched": 0,
        "not-started": 1,
    }

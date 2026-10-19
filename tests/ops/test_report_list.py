"""``python -m ops.report list``: every run root under a directory, with its state."""

from __future__ import annotations

import json
import os

from ops.report import cli
from tests.ops.trees import (
    attempt_row,
    numbered,
    write_group,
    write_journal,
    write_json,
    write_lock,
    write_manifest,
    write_stage,
    write_summary,
    write_task_state,
)


def listing(capsys, directory):
    """The rows of ``list`` as lists of cells, the header first."""
    assert cli.main(["list", str(directory)]) == 0
    return [line.split() for line in capsys.readouterr().out.splitlines()]


def test_list_has_a_row_for_each_run_root_among_the_children_and_grandchildren(tmp_path, capsys):
    root = tmp_path / "exp" / "baseline"
    write_manifest(root, method="baseline", tasks=["alpha", "beta"])
    write_lock(root, method="baseline", pid=os.getpid())
    write_task_state(root, "alpha", complete=True)
    write_summary(
        tmp_path / "evals" / "one",
        attempts=[attempt_row("parent", journal=tmp_path / "j.jsonl", resolved=True)],
    )
    (tmp_path / "gate").mkdir()
    (tmp_path / "gate" / "gate.jsonl").write_text("{}\n")
    write_json(tmp_path / "gate" / "summary.json", {"tasks": 1, "modes": ["oracle", "nop"]})
    write_journal(tmp_path / "lone.jsonl", turns=numbered(1))
    (tmp_path / "unrelated").mkdir()
    (tmp_path / "unrelated" / "notes.txt").write_text("x")
    write_summary(tmp_path / ".hidden" / "two", attempts=[])

    rows = listing(capsys, tmp_path)

    assert rows == [
        ["kind", "methods", "tasks", "complete", "errored", "running", "stale", "runner", "path"],
        ["evaluate", "sprout", "1", "1", "0", "0", "-", "-", "evals/one"],
        ["experiment", "baseline", "2", "1", "0", "0", "0", "alive", "exp"],
        ["gate", "oracle,nop", "1", "-", "-", "-", "-", "-", "gate"],
        ["journal", "-", "-", "-", "-", "-", "-", "-", "lone.jsonl"],
    ]


def test_a_live_runner_is_alive_and_its_unfinished_task_is_running(tmp_path, capsys):
    root = tmp_path / "exp" / "baseline"
    write_manifest(root, method="baseline", tasks=["alpha"])
    write_lock(root, method="baseline", pid=os.getpid())
    write_stage(root / "tasks" / "alpha" / "baseline" / "attempt-1", state="running")

    rows = listing(capsys, tmp_path)

    assert rows[1] == ["experiment", "baseline", "1", "0", "0", "1", "0", "alive", "exp"]


def test_a_root_without_a_lock_is_done_and_its_unfinished_task_is_stale(tmp_path, capsys):
    root = tmp_path / "exp" / "baseline"
    write_manifest(root, method="baseline", tasks=["alpha"])
    write_stage(root / "tasks" / "alpha" / "baseline" / "attempt-1", state="running")

    rows = listing(capsys, tmp_path)

    assert rows[1] == ["experiment", "baseline", "1", "0", "0", "0", "1", "done", "exp"]


def test_an_evaluation_through_the_run_store_counts_its_open_groups_as_running_and_its_error(
    tmp_path, capsys
):
    output = tmp_path / "out"
    write_summary(output, attempts=[], complete=False, error="RuntimeError: the grade job lost its sandbox")
    write_group(output, 0, {"ready": True, "samples": []})
    write_group(output, 1, {"ready": False, "samples": []})

    rows = listing(capsys, tmp_path)

    assert rows[1:] == [["evaluate", "sprout", "1", "0", "1", "1", "-", "-", "out"]]


def test_a_harbor_trial_without_a_result_and_with_an_open_journal_is_running(tmp_path, capsys):
    output = tmp_path / "out"
    write_summary(output, attempts=[], complete=False)
    write_journal(
        output / "trials" / "t-parent" / "agent" / "trajectory.jsonl", turns=numbered(1), status=None
    )
    write_journal(output / "trials" / "t-done" / "agent" / "trajectory.jsonl", turns=numbered(1))
    write_json(
        output / "trials" / "t-failed" / "result.json",
        {"exception_info": {"exception_type": "AgentTimeoutError"}},
    )

    rows = listing(capsys, tmp_path)

    assert rows[1:] == [["evaluate", "sprout", "1", "0", "1", "1", "-", "-", "out"]]


def test_list_json_has_the_fields_of_each_row(tmp_path, capsys):
    write_journal(tmp_path / "lone.jsonl", turns=numbered(1))

    assert cli.main(["list", str(tmp_path), "--format", "json"]) == 0

    assert json.loads(capsys.readouterr().out) == [
        {
            "kind": "journal",
            "methods": "-",
            "tasks": "-",
            "complete": "-",
            "errored": "-",
            "running": "-",
            "stale": "-",
            "runner": "-",
            "path": "lone.jsonl",
        }
    ]

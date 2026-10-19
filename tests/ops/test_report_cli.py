"""The ``python -m ops.report`` command line: what it refuses with exit 2, what it writes, what it never does.

Each test runs ``cli.main`` in this process, except the one that runs ``python -m ops.report``.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

import pytest

from ops.report import cli
from ops.report.cli import NOT_A_ROOT
from tests.ops.trees import (
    attempt_row,
    numbered,
    write_attempt,
    write_journal,
    write_json,
    write_lock,
    write_manifest,
    write_plan,
    write_summary,
)
from tests.support.paths import REPO


@pytest.mark.parametrize("command", ["runs", "task", "export", "snapshots"])
def test_a_target_of_no_known_kind_exits_2_and_says_what_was_looked_for(tmp_path, capsys, command):
    (tmp_path / "readme.txt").write_text("x")

    code = cli.main([command, str(tmp_path)])

    assert (code, capsys.readouterr().err) == (2, NOT_A_ROOT % tmp_path + "\n")


def test_a_journal_given_to_runs_exits_2_and_names_the_command_that_reads_it(tmp_path, capsys):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1))

    code = cli.main(["runs", str(journal)])

    assert (code, capsys.readouterr().err) == (
        2,
        "runs: %s is a journal; this command reads experiment, method, gate (try `export`)\n" % journal,
    )


def test_task_on_an_experiment_without_a_task_name_exits_2(tmp_path, capsys):
    write_manifest(tmp_path / "exp" / "baseline", method="baseline", tasks=["alpha"])

    code = cli.main(["task", str(tmp_path / "exp")])

    assert (code, capsys.readouterr().err) == (
        2,
        "task: an experiment or method root needs the task name (ROOT TASK)\n",
    )


def test_task_with_a_name_in_no_manifest_of_the_root_exits_2(tmp_path, capsys):
    write_manifest(tmp_path / "exp" / "baseline", method="baseline", tasks=["alpha"])

    code = cli.main(["task", str(tmp_path / "exp"), "no-such-task"])

    assert (code, capsys.readouterr().err) == (
        2,
        "task: no-such-task is not a task of %s (not in the tasks of manifest-baseline.json)\n"
        % (tmp_path / "exp"),
    )


def test_list_of_a_path_that_is_no_directory_exits_2(tmp_path, capsys):
    code = cli.main(["list", str(tmp_path / "nope")])

    assert (code, capsys.readouterr().err) == (2, "list: %s is not a directory\n" % (tmp_path / "nope"))


def test_export_to_a_file_writes_the_document_there_and_says_where(tmp_path, capsys):
    journal = write_journal(tmp_path / "j.jsonl", run_id="t-run", turns=numbered(2))
    out = tmp_path / "out.json"

    code = cli.main(["export", str(journal), "--format", "json", "-o", str(out)])

    lines = out.read_text().count("\n")
    assert (code, capsys.readouterr().out, json.loads(out.read_text())["run_id"]) == (
        0,
        "wrote %s (%d lines)\n" % (out, lines),
        "t-run",
    )


def test_export_of_a_harbor_trial_directory_reads_its_agent_journal(tmp_path, capsys):
    trial = tmp_path / "trials" / "t-run"
    write_json(trial / "config.json", {})
    write_journal(trial / "agent" / "trajectory.jsonl", run_id="t-run", turns=numbered(1))

    code = cli.main(["export", str(trial)])

    assert (code, capsys.readouterr().out.split("\n")[0]) == (0, "# t-run")


def test_export_of_a_trial_directory_without_a_journal_exits_2(tmp_path, capsys):
    trial = tmp_path / "trials" / "t-run"
    write_json(trial / "config.json", {})
    (trial / "agent").mkdir()

    code = cli.main(["export", str(trial)])

    assert (code, capsys.readouterr().err) == (2, "export: %s has no agent/trajectory.jsonl\n" % trial)


def experiment_with_a_sprout_round(tmp_path):
    """An experiment root with a failed attempt-1 of alpha and a sprout evaluation that branched from it."""
    baseline = tmp_path / "exp" / "baseline"
    write_manifest(baseline, method="baseline", tasks=["alpha"])
    write_lock(baseline, method="baseline", pid=os.getpid())
    parent = write_attempt(baseline / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False)
    sprout = tmp_path / "exp" / "sprout"
    write_manifest(sprout, method="sprout", tasks=["alpha"], baseline_root=str(baseline))
    output = sprout / "tasks" / "alpha" / "sprout"
    rows = json.loads((parent / "summary.json").read_text())["attempts"]
    write_summary(
        output,
        attempts=rows + [attempt_row("r1b1-fix", journal=tmp_path / "artifacts" / "b.jsonl", resolved=True)],
    )
    write_plan(output, 1)
    return tmp_path / "exp"


def command_line(argv, **paths):
    """``argv`` with each ``{name}`` replaced by its path in ``paths``."""
    return [part.format(**paths) for part in argv]


def tree_state(root):
    return sorted((str(path), path.stat().st_size, path.stat().st_mtime_ns) for path in root.rglob("*"))


@pytest.mark.parametrize(
    "argv",
    [
        ["runs", "{exp}"],
        ["task", "{exp}", "alpha"],
        ["task", "{exp}", "alpha", "--format", "mermaid"],
        ["export", "{artifacts}/job-alpha-1/attempt-job-alpha-1/trajectory.jsonl"],
        ["snapshots", "{exp}"],
        ["list", "{tmp}"],
    ],
    ids=lambda argv: "-".join(argv[:1] + argv[3:4]),
)
def test_no_command_writes_under_the_run_tree_or_starts_a_process(tmp_path, capsys, monkeypatch, argv):
    def no_process(*args, **kwargs):
        raise AssertionError("a process was started: %r" % (args,))

    exp = experiment_with_a_sprout_round(tmp_path)
    before = tree_state(tmp_path)
    monkeypatch.setattr(subprocess, "Popen", no_process)
    monkeypatch.setattr(os, "system", no_process)

    code = cli.main(command_line(argv, exp=exp, artifacts=tmp_path / "artifacts", tmp=tmp_path))

    assert (code, tree_state(tmp_path)) == (0, before)


def test_python_m_ops_report_runs_the_command_line(tmp_path):
    write_journal(tmp_path / "lone.jsonl", turns=numbered(1))

    result = subprocess.run(
        [sys.executable, "-m", "ops.report", "list", str(tmp_path)],
        cwd=REPO,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (result.returncode, result.stdout.split("\n")[1:2]) == (
        0,
        ["journal  -        -      -         -        -        -      -       lone.jsonl"],
    )

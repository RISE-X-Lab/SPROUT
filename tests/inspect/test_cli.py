"""Wiring: every subcommand through argparse, exit codes, the import budget, determinism."""

from __future__ import annotations

import ast
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

from sprout.inspect import cli, layout

from tests.inspect.conftest import FIXTURES, make_experiment

REPO = Path(__file__).resolve().parents[2]
PACKAGE = REPO / "src" / "sprout" / "inspect"
ALLOWED_SPROUT = {"sprout.worker.run.journal", "sprout.worker.run.events", "sprout.worker.checkpoint.rollback"}


def test_every_subcommand_is_dispatched(experiment, tmp_path, capsys):
    journal = experiment / "sprout/tasks/alpha/sprout/trials/alpha-r1b1-fix/agent/trajectory.jsonl"
    expected = {("runs", str(experiment)): "== baseline",
                ("task", str(experiment), "alpha"): "r1b1-fix  parent@3 (turn 3)",
                ("export", str(journal)): "## ✂ BRANCH POINT",
                ("snapshots", str(journal)): "branch of alpha-parent@3 from par-0003",
                ("list", str(experiment.parent)): "experiment  baseline,sprout"}
    for argv, token in expected.items():
        assert cli.main(list(argv)) == 0, argv
        assert token in capsys.readouterr().out, argv
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-m", "sprout.inspect", "--help"], cwd=tmp_path, env=env,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    assert all(name in result.stdout for name in ("runs", "task", "export", "snapshots", "list"))
    assert "Five commands" in result.stdout and "fork tree" in result.stdout        # the package docstring, not cli's
    assert "argument parsing, dispatch" not in result.stdout
    for command, phrase in (("runs", "dataclasses.asdict"), ("task", "mermaid a graph LR"), ("export", "md is markdown"),
                            ("snapshots", "dataclasses.asdict"), ("list", "dataclasses.asdict")):
        with pytest.raises(SystemExit) as stop:
            cli.main([command, "--help"])
        assert stop.value.code == 0
        out = capsys.readouterr().out
        assert "--format {" in out and phrase in out, command                     # every --format says what it gives


def test_a_missing_or_unrecognised_target_exits_2(tmp_path, capsys):
    assert cli.main(["runs", str(tmp_path / "nope")]) == 2
    assert capsys.readouterr().err.startswith("not a run root: %s (looked for baseline/ or sprout/" % (tmp_path / "nope"))
    (tmp_path / "plain").mkdir()
    (tmp_path / "plain" / "readme.txt").write_text("x")
    for command in ("runs", "task", "export", "snapshots"):
        assert cli.main([command, str(tmp_path / "plain")]) == 2, command
        assert "not a run root" in capsys.readouterr().err
    assert cli.main(["list", str(tmp_path / "nope")]) == 2
    assert cli.main(["runs", str(FIXTURES / "r1b1.trajectory.jsonl")]) == 2        # a journal is not a run root
    assert "try `export`" in capsys.readouterr().err
    # a corrupt manifest makes no method root: exit 2 with the message, not a traceback
    (tmp_path / "corrupt" / "baseline").mkdir(parents=True)
    (tmp_path / "corrupt" / "baseline" / "manifest-baseline.json").write_text("garbage")
    for target in (tmp_path / "corrupt", tmp_path / "corrupt" / "baseline"):
        assert cli.main(["runs", str(target)]) == 2
        assert capsys.readouterr().err.startswith("not a run root: %s" % target)
    assert cli.main(["list", str(tmp_path / "corrupt")]) == 0
    assert capsys.readouterr().out.splitlines()[1:] == []                          # listed as nothing
    # a task name in no manifest of the root
    root = make_experiment(tmp_path / "exp", with_sprout=False, tasks={"alpha": {"attempts": [("complete", True)]}})
    assert cli.main(["task", str(root), "no-such-task"]) == 2
    assert capsys.readouterr().err == ("task: no-such-task is not a task of %s (not in the tasks of manifest-baseline.json)\n"
                                       % root)
    assert cli.main(["task", str(root / "baseline"), "no-such-task"]) == 2
    assert "not a task of" in capsys.readouterr().err


def test_review_kinds_are_told_apart_by_the_prompts_branching_actually_sends():
    """load_review classifies a transport record by its prompt's first words; the words are
    prompts.yaml's. A reworded prompt would turn every review 'unknown' and the task tree would
    lose its review files and timings, so the prefixes are pinned to the templates here."""
    from sprout.methods.prompts import DEFAULT_PROMPTS
    assert DEFAULT_PROMPTS.analyst_case.lstrip().startswith(layout.ANALYST_PROMPT)
    for template in (DEFAULT_PROMPTS.reviewer_assistant_turn, DEFAULT_PROMPTS.reviewer_point,
                     DEFAULT_PROMPTS.reviewer_user_hint):
        assert template.lstrip().startswith(layout.REVIEWER_PROMPT), template[:60]
    assert not DEFAULT_PROMPTS.analyst_case.startswith(layout.REVIEWER_PROMPT)
    rendered = DEFAULT_PROMPTS.reviewer_assistant_turn.format(problem="p", reports="[]", count_rule="At most 1.")
    assert layout.load_review.__doc__ and rendered.startswith(layout.REVIEWER_PROMPT)


def test_export_supersedes_the_ops_trajectory_viewer():
    assert not (REPO / "ops" / "trajectory_view.py").exists()
    tracked = subprocess.run(["git", "ls-files"], cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    assert len(tracked) > 100 and not [path for path in tracked if "trajectory_view" in path]
    # every tracked file except this test, which has to name what it checks for
    hits = subprocess.run(["git", "grep", "-l", "trajectory_view", "--", ".", ":(exclude)tests/inspect/test_cli.py"],
                          cwd=REPO, capture_output=True, text=True)
    assert hits.returncode == 1 and hits.stdout == "", hits.stdout
    assert "python -m sprout.inspect export" in (REPO / "README.md").read_text()


def test_inspect_is_stdlib_plus_harness_only():
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) >= 5
    for path in files:
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
                names = [node.module]
            else:
                continue
            for name in names:
                root = name.split(".")[0]
                assert root in sys.stdlib_module_names or root == "sprout", "%s imports %s" % (path.name, name)
                assert root != "examples"
                if root == "sprout":
                    assert name in ALLOWED_SPROUT or name.startswith("sprout.inspect"), "%s imports %s" % (path.name, name)


def test_output_is_deterministic(experiment, monkeypatch, capsys):
    def no_clock(*args, **kwargs):
        raise AssertionError("the wall clock was read")
    monkeypatch.setattr(time, "time", no_clock)
    monkeypatch.setattr(time, "time_ns", no_clock)
    source = "".join(path.read_text() for path in PACKAGE.glob("*.py"))
    assert ".now(" not in source and "utcnow" not in source and "today()" not in source and "monotonic" not in source
    outputs = []
    for _ in range(2):
        for argv in (["runs", str(experiment)], ["task", str(experiment), "alpha"],
                     ["snapshots", str(experiment / "sprout/tasks/alpha/sprout")]):
            assert cli.main(argv) == 0
            outputs.append(capsys.readouterr().out)
    assert outputs[:3] == outputs[3:]


def test_the_real_fixture_is_scrubbed():
    files = sorted(p for p in FIXTURES.iterdir() if p.is_file())
    assert len(files) >= 10
    for path in files:
        text = path.read_text()
        assert "47.100" not in text and "api_key" not in text, path.name
    assert not (FIXTURES / "config.json").exists() and not (FIXTURES / "lock.json").exists()

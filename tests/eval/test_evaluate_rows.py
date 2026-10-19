"""The records of ``sprout.eval.evaluate``: an attempt's row, the recorded parent and the hidden-test facts.

``attempt_row`` reads a sample of a settled group document, as the search coordinator leaves it. The
others read files that an earlier evaluation or the task holds.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.eval.evaluate import (
    DEFAULT_RUNSTORE_URL,
    attempt_row,
    hidden_facts,
    parent_journal,
    recorded_parent,
)


def sample(*, actor: dict[str, Any] | None = None, grade: dict[str, Any] | None = None) -> dict[str, Any]:
    """A settled sample: a succeeded rollout of 3 steps, and a grade of reward 0, each with ``changes``."""
    rollout = {
        "state": "succeeded",
        "job_id": "job-0",
        "attempt_id": "attempt-0",
        "started_at": 100.0,
        "result": {
            "status": "completed",
            "journal_path": "/path/to/artifacts/job-0/attempt-0/trajectory.jsonl",
        },
    }
    verdict = {"state": "succeeded", "job_id": "job-1", "result": {"reward": 0.0, "reason": None}}
    return {
        "sample_slot_id": "parent",
        "actor": {**rollout, **(actor or {})},
        "grade": {**verdict, **(grade or {})},
    }


def test_an_attempt_row_records_the_rollout_its_grade_and_how_long_until_the_verdict() -> None:
    row = attempt_row(sample(), name="r1b1-a", round_no=1, hint="Check the boundary.", validated_at=160.5)

    assert row == {
        "name": "r1b1-a",
        "round": 1,
        "status": "completed",
        "stop_reason": None,
        "resolved": False,
        "reward": 0.0,
        "reason": None,
        "journal": "/path/to/artifacts/job-0/attempt-0/trajectory.jsonl",
        "job_id": "job-0",
        "attempt_id": "attempt-0",
        "grade_job_id": "job-1",
        "log": None,
        "model_patch": None,
        "started_at": 100.0,
        "validated_at": 160.5,
        "seconds": 60.5,
        "hint": "Check the boundary.",
    }


def test_an_attempt_without_a_start_time_has_no_duration_and_a_row_without_a_hint_names_none() -> None:
    row = attempt_row(
        sample(actor={"started_at": None}), name="parent", round_no=0, hint=None, validated_at=160.5
    )

    assert (row["started_at"], row["seconds"], "hint" in row) == (None, None, False)


@pytest.mark.parametrize(
    "result, status, stop_reason",
    [
        (
            {"status": "truncated", "agent_status": "timeout", "stop_reason": "max_turns_reached"},
            "timeout",
            "max_turns_reached",
        ),
        ({"status": "completed"}, "completed", None),
    ],
    ids=["truncated", "completed"],
)
def test_a_row_names_the_agents_own_status(
    result: dict[str, Any], status: str, stop_reason: str | None
) -> None:
    """A run that hit its turn or time limit is truncated and graded. The agent's status says which."""
    row = attempt_row(
        sample(actor={"result": result}), name="parent", round_no=0, hint=None, validated_at=160.5
    )

    assert (row["status"], row["stop_reason"]) == (status, stop_reason)


@pytest.mark.parametrize(
    "reward, resolved", [(1, True), (1.0, True), (0.999, False), (0.5, False), (0, False)]
)
def test_only_a_reward_of_1_resolves_the_task(reward: float, resolved: bool) -> None:
    grade = {"result": {"reward": reward, "reason": None}}

    row = attempt_row(sample(grade=grade), name="parent", round_no=0, hint=None, validated_at=160.5)

    assert (row["resolved"], row["reward"]) == (resolved, float(reward))


def test_a_rollout_that_did_not_succeed_fails_the_evaluation_and_names_its_job() -> None:
    failed = {"state": "failed", "error": "mini did not submit", "result": None}

    with pytest.raises(RuntimeError, match="^parent: the rollout job job-0 failed: mini did not submit$"):
        attempt_row(sample(actor=failed), name="parent", round_no=0, hint=None, validated_at=160.5)


@pytest.mark.parametrize(
    "grade, message",
    [
        (
            {"state": "failed", "error": "verifier crashed", "result": None},
            "the grade job job-1 failed: verifier crashed",
        ),
        ({"result": {"reward": None, "reason": "no reward file"}}, "the grade job job-1 succeeded: None"),
        ({"result": {"reward": True, "reason": None}}, "the grade job job-1 succeeded: None"),
        ({"result": {"reward": "1", "reason": None}}, "the grade job job-1 succeeded: None"),
    ],
    ids=["grade-failed", "no-reward", "boolean-reward", "text-reward"],
)
def test_a_grade_without_a_numeric_reward_fails_the_evaluation_and_names_its_job(
    grade: dict[str, Any], message: str
) -> None:
    """JSON ``true`` is an int to ``isinstance``, but it is not a reward."""
    with pytest.raises(RuntimeError, match=f"^parent: {message}$"):
        attempt_row(sample(grade=grade), name="parent", round_no=0, hint=None, validated_at=160.5)


# --- the recorded parent of a branch method ------------------------------------------------


def baseline_output(tmp_path: Path, **changes: Any) -> Path:
    """The output of a completed baseline evaluation whose parent journal exists, with ``changes``."""
    journal = tmp_path / "artifacts" / "job-0" / "attempt-0" / "trajectory.jsonl"
    journal.parent.mkdir(parents=True)
    journal.write_text("{}\n")
    parent = {
        "name": "parent",
        "job_id": "job-0",
        "attempt_id": "attempt-0",
        "journal": str(journal),
        "resolved": False,
    }
    summary = {
        "task": "shell-task",
        "dataset_digest": "sha256:pinned",
        "complete": True,
        "runstore": {"url": DEFAULT_RUNSTORE_URL},
        "attempts": [parent],
        **changes,
    }
    output = tmp_path / "baseline"
    output.mkdir()
    (output / "summary.json").write_text(json.dumps(summary))
    return output


def test_a_branch_method_continues_the_parent_row_of_a_completed_baseline(tmp_path: Path) -> None:
    output = baseline_output(tmp_path)

    parent = recorded_parent(
        output, task="shell-task", dataset_digest="sha256:pinned", runstore_url=DEFAULT_RUNSTORE_URL
    )

    assert parent == {
        "name": "parent",
        "job_id": "job-0",
        "attempt_id": "attempt-0",
        "journal": str(tmp_path / "artifacts" / "job-0" / "attempt-0" / "trajectory.jsonl"),
        "resolved": False,
    }


@pytest.mark.parametrize(
    "changes, message",
    [
        ({"task": "other-task"}, "Recorded parent metadata differs from this task"),
        ({"dataset_digest": "sha256:other"}, "Recorded parent metadata differs from this task"),
        ({"complete": False}, "Recorded parent metadata differs from this task"),
        ({"attempts": []}, "it has no Run Store job to branch from"),
        (
            {
                "attempts": [
                    {
                        "name": "parent",
                        "journal": "/path/to/trials/parent/agent/trajectory.jsonl",
                        "resolved": False,
                    }
                ]
            },
            "The recorded parent ran as a local Harbor trial",
        ),
        (
            {"runstore": {"url": "http://runstore.test:18110"}},
            "The recorded parent's jobs are in another Run Store",
        ),
    ],
    ids=["other-task", "other-dataset", "incomplete", "no-parent", "local-harbor-trial", "other-run-store"],
)
def test_a_branch_method_refuses_a_parent_that_it_cannot_branch_from(
    tmp_path: Path, changes: dict[str, Any], message: str
) -> None:
    output = baseline_output(tmp_path, **changes)

    with pytest.raises(ValueError, match=message):
        recorded_parent(
            output, task="shell-task", dataset_digest="sha256:pinned", runstore_url=DEFAULT_RUNSTORE_URL
        )


def test_a_parent_whose_journal_is_gone_cannot_be_branched_from(tmp_path: Path) -> None:
    output = baseline_output(tmp_path)
    (tmp_path / "artifacts" / "job-0" / "attempt-0" / "trajectory.jsonl").unlink()

    with pytest.raises(ValueError, match="trajectory.jsonl is gone$"):
        recorded_parent(
            output, task="shell-task", dataset_digest="sha256:pinned", runstore_url=DEFAULT_RUNSTORE_URL
        )


@pytest.mark.parametrize(
    "attempts",
    [[], [{"name": "parent", "journal": None}]],
    ids=["no-attempt", "no-journal"],
)
def test_an_output_without_a_parent_journal_has_no_parent(
    tmp_path: Path, attempts: list[dict[str, Any]]
) -> None:
    output = baseline_output(tmp_path, attempts=attempts)

    with pytest.raises(ValueError, match="^No parent attempt recorded under "):
        parent_journal(output)


# --- the hidden tests that only the leak checks read ----------------------------------------


PATCH = (
    "diff --git a/tests/test_x.py b/tests/test_x.py\n+++ b/tests/test_x.py\n+    assert f('FIXTURE-9') == 1\n"
)


@pytest.fixture
def task_with_hidden_tests(tmp_path: Path) -> Path:
    """A task whose tests/ holds a test patch and the FAIL_TO_PASS ids of SWE-bench metadata."""
    tests = tmp_path / "task" / "tests"
    tests.mkdir(parents=True)
    (tests / "test.patch").write_text(PATCH)
    (tests / "config.json").write_text(json.dumps({"f2p_node_ids": ["tests/test_x.py::test_secret_case"]}))
    return tmp_path / "task"


def test_the_leak_checks_get_the_fail_to_pass_ids_the_patched_paths_and_the_patch_for_the_judge(
    tmp_path: Path, task_with_hidden_tests: Path
) -> None:
    facts = hidden_facts(task_with_hidden_tests, patch_path=tmp_path / "hidden-tests.patch")

    assert facts == {
        "fail_to_pass": ["tests/test_x.py::test_secret_case"],
        "test_patch_paths": ["tests/test_x.py"],
        "test_patch": str(tmp_path / "hidden-tests.patch"),
    }
    assert (tmp_path / "hidden-tests.patch").read_text() == PATCH


def test_without_a_judge_the_hidden_tests_stay_in_the_task(
    tmp_path: Path, task_with_hidden_tests: Path
) -> None:
    facts = hidden_facts(task_with_hidden_tests, patch_path=None)

    assert facts == {
        "fail_to_pass": ["tests/test_x.py::test_secret_case"],
        "test_patch_paths": ["tests/test_x.py"],
    }

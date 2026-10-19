"""``ops.report.layout``: what kind of run output a directory is, and the runner's and evaluator's files."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from examples.qwen import run
from ops.report.layout import (
    Grade,
    agent_receipt,
    detect,
    load_evaluation,
    load_grade,
    load_review,
    load_stage,
    stage_state,
)
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS
from tests.ops.trees import write_journal, write_json, write_manifest, write_stage
from tests.support.paths import FIXTURES

REAL_OUTPUT = FIXTURES / "inspect" / "abs-module-cache-flags"
SENTINEL = "handed to root relaunch-tail at 00:02 UTC; this runner skips it"


def journal(tmp_path):
    return write_journal(tmp_path / "x.jsonl", turns=[])


def text_file(tmp_path):
    path = tmp_path / "notes.txt"
    path.write_text("x")
    return path


def gate_root(tmp_path):
    (tmp_path / "gate.jsonl").write_text("{}\n")
    return tmp_path


def evaluate_output(tmp_path):
    write_json(tmp_path / "summary.json", {"attempts": []})
    return tmp_path


def summary_without_attempts(tmp_path):
    write_json(tmp_path / "summary.json", {"task": "t"})
    return tmp_path


def harbor_trial(tmp_path):
    write_json(tmp_path / "config.json", {})
    (tmp_path / "agent").mkdir()
    return tmp_path


def method_root(tmp_path):
    write_manifest(tmp_path, method="baseline", tasks=["alpha"])
    return tmp_path


def corrupt_manifest(tmp_path):
    (tmp_path / "manifest-baseline.json").write_text("garbage")
    return tmp_path


def experiment_root(tmp_path):
    write_manifest(tmp_path / "baseline", method="baseline", tasks=["alpha"])
    return tmp_path


def ended_task(tmp_path):
    write_json(tmp_path / "baseline.task.json", {"complete": True})
    return tmp_path


def task_in_a_method_stage(tmp_path):
    write_json(tmp_path / "sprout.stage.json", {})
    return tmp_path


def task_in_its_first_attempt(tmp_path):
    write_json(tmp_path / "baseline" / "attempt-1.stage.json", {})
    return tmp_path


def empty_directory(tmp_path):
    return tmp_path


def missing_path(tmp_path):
    return tmp_path / "nope"


@pytest.mark.parametrize(
    "make,kind",
    [
        (journal, "journal"),
        (text_file, None),
        (gate_root, "gate"),
        (evaluate_output, "evaluate"),
        (summary_without_attempts, None),
        (harbor_trial, "trial"),
        (method_root, "method"),
        (corrupt_manifest, None),
        (experiment_root, "experiment"),
        (ended_task, "task"),
        (task_in_a_method_stage, "task"),
        (task_in_its_first_attempt, "task"),
        (empty_directory, None),
        (missing_path, None),
    ],
    ids=lambda value: getattr(value, "__name__", str(value)),
)
def test_detect_names_the_kind_of_run_output_at_a_path(tmp_path, make, kind):
    assert detect(make(tmp_path)) == kind


@pytest.mark.parametrize(
    "receipt,launched,lock_alive,state",
    [
        (None, True, True, "missing"),
        ({"complete": True}, False, False, "complete"),
        ({"complete": False, "finished_at": 9.0}, False, True, "errored"),
        ({"complete": False}, False, True, "unlaunched"),
        ({"complete": False}, True, True, "running"),
        ({"complete": False}, True, False, "stale"),
        ({"complete": False}, True, None, "stale"),
    ],
    ids=["no-receipt", "complete", "finished-incomplete", "no-log", "live-runner", "dead-runner", "no-lock"],
)
def test_a_stage_state_follows_from_its_receipt_its_log_and_the_runner(receipt, launched, lock_alive, state):
    assert stage_state(receipt, launched, lock_alive) == state


@pytest.mark.parametrize(
    "log,cause",
    [
        (
            "harbor noise\n}\n" + json.dumps({"stop_reason": "reviewer failed: bad JSON"}, indent=2) + "\n",
            "stop_reason: reviewer failed: bad JSON",
        ),
        (
            json.dumps({"error": "RuntimeError: grade lost"}, indent=2) + "\n",
            "error: RuntimeError: grade lost",
        ),
        ("harbor: trial ended\n" + json.dumps({"complete": False}, indent=2) + "\n", "harbor: trial ended"),
        (
            "Traceback\n  File x\nRuntimeError: worktree not clean\nexiting\n\n\n",
            "RuntimeError: worktree not clean",
        ),
        ("only noise\nlast words\n", "last words"),
        ("", None),
    ],
    ids=[
        "stop-reason-of-the-state-dump",
        "error-of-the-state-dump",
        "the-line-before-a-dump-without-cause",
        "last-exception-line",
        "last-line",
        "empty",
    ],
)
def test_an_errored_stage_takes_its_cause_from_the_tail_of_its_log(tmp_path, log, cause):
    output = tmp_path / "sprout"
    write_stage(output, state="errored", log=log)

    stage = load_stage(output.with_name("sprout.stage.json"), lock_alive=False)

    assert (stage.state, stage.error, stage.cause) == ("errored", "Stage exited 1; see sprout.log", cause)


def test_a_cause_that_repeats_the_receipts_error_is_dropped(tmp_path):
    output = tmp_path / "attempt-2"
    write_stage(output, state="errored", log="Stage exited 1; see attempt-2.log\n")

    stage = load_stage(output.with_name("attempt-2.stage.json"), lock_alive=False)

    assert (stage.error, stage.cause) == ("Stage exited 1; see attempt-2.log", None)


def test_a_receipt_key_that_the_runner_never_writes_is_shown_as_an_extra(tmp_path):
    output = tmp_path / "attempt-1"
    write_stage(output, state="unlaunched", sentinel=SENTINEL)

    stage = load_stage(output.with_name("attempt-1.stage.json"), lock_alive=True)

    assert (stage.state, stage.extras) == ("unlaunched", {"sentinel": SENTINEL})


def runner_args(tmp_path):
    """The parsed command line of a baseline run, with what ``main`` adds before the first stage."""
    options = (
        "--dataset deepswe --method baseline --task alpha --model-endpoint http://127.0.0.1:9/v1".split()
    )
    args = run.parse_args(
        options + ["--tasks-dir", str(tmp_path / "tasks"), "--output", str(tmp_path / "out")]
    )
    args.digest = "sha256:d"
    args.source_commit = "0" * 40
    args.preparation = tmp_path / "out" / "preparation.json"
    return args


def evaluator_exiting(code):
    """A stand-in for the ``sprout.eval.evaluate`` process: it writes a graded summary and exits ``code``."""

    def call(argv, **kwargs):
        output = Path(argv[argv.index("--output") + 1])
        write_json(
            output / "summary.json",
            {"complete": True, "attempts": [{"status": "completed", "resolved": False}]},
        )
        return code

    return call


def test_a_receipt_of_a_stage_that_the_runner_completed_has_no_extras(tmp_path, monkeypatch):
    """Each key that examples/qwen/run.py writes in a receipt is in RECEIPT_KEYS, or the report shows it."""
    monkeypatch.setattr(run.subprocess, "call", evaluator_exiting(0))
    output = tmp_path / "out" / "tasks" / "alpha" / "baseline" / "attempt-1"
    run.stage(runner_args(tmp_path), "alpha", output, "baseline")

    stage = load_stage(output.with_name("attempt-1.stage.json"), lock_alive=True)

    assert (stage.state, stage.extras) == ("complete", {})


def test_a_receipt_of_a_stage_that_the_runner_saw_fail_has_no_extras(tmp_path, monkeypatch):
    monkeypatch.setattr(run.subprocess, "call", evaluator_exiting(1))
    output = tmp_path / "out" / "tasks" / "alpha" / "baseline" / "attempt-1"
    with pytest.raises(RuntimeError):
        run.stage(runner_args(tmp_path), "alpha", output, "baseline")

    stage = load_stage(output.with_name("attempt-1.stage.json"), lock_alive=True)

    assert (stage.state, stage.extras) == ("errored", {})


@pytest.mark.parametrize(
    "result,grade",
    [
        (
            {
                "verifier_result": {
                    "rewards": {
                        "reward": 0,
                        "f2p_total": 20,
                        "f2p_passed": 9,
                        "p2p_total": 3,
                        "p2p_passed": 3,
                    }
                }
            },
            Grade(reward=0.0, resolved=False, detail="f2p 9/20 p2p 3/3", exception=None),
        ),
        (
            {"verifier_result": {"rewards": {"reward": 1}}},
            Grade(reward=1.0, resolved=True, detail="", exception=None),
        ),
        (
            {"verifier_result": None, "exception_info": {"exception_type": "AgentTimeoutError"}},
            Grade(reward=None, resolved=None, detail="", exception="AgentTimeoutError"),
        ),
    ],
    ids=["failed-with-test-counts", "resolved", "harbor-raised"],
)
def test_a_harbor_result_gives_the_reward_the_verdict_the_test_counts_and_the_exception(result, grade):
    assert load_grade(result) == grade


def test_the_agent_receipt_of_a_trial_is_the_one_that_names_the_journals_sandbox(tmp_path):
    write_json(tmp_path / "t__env.podman.json", {"sandbox_id": "sprout-a", "overlay": None})
    write_json(tmp_path / "t__other.podman.json", {"sandbox_id": "sprout-b", "overlay": None})

    receipt = agent_receipt(tmp_path, "sprout-b")

    assert receipt == {"backend": "podman", "sandbox_id": "sprout-b", "overlay": None}


def test_without_a_sandbox_id_the_agent_receipt_is_the_one_without_a_verifier_overlay(tmp_path):
    write_json(tmp_path / "t__env.agentenv.json", {"sandbox_id": "01a0-vm", "overlay": None})
    write_json(
        tmp_path / "t__verifier.agentenv.json", {"sandbox_id": "01b0-vm", "overlay": {"runs": ["chmod +x t"]}}
    )

    receipt = agent_receipt(tmp_path, None)

    assert receipt == {"backend": "microvm", "sandbox_id": "01a0-vm", "overlay": None}


def test_two_receipts_without_an_overlay_name_no_agent_receipt(tmp_path):
    write_json(tmp_path / "t__a.podman.json", {"sandbox_id": "sprout-a", "overlay": None})
    write_json(tmp_path / "t__b.podman.json", {"sandbox_id": "sprout-b", "overlay": None})

    receipt = agent_receipt(tmp_path, None)

    assert receipt is None


@pytest.mark.parametrize(
    "template,kind",
    [
        ("analyst_case", "analyst"),
        ("reviewer_assistant_turn", "reviewer"),
        ("reviewer_point", "reviewer"),
        ("reviewer_user_hint", "reviewer"),
        ("shepherd_system", "unknown"),
    ],
)
def test_a_review_receipt_is_told_apart_by_the_first_words_of_the_prompt_that_the_review_sent(
    tmp_path, template, kind
):
    """The words are those of resources/review/prompts.yaml. A reworded prompt makes no review unknown."""
    receipt = write_json(
        tmp_path / "review-001.json",
        {"request": {"prompt": getattr(DEFAULT_PROMPTS, template)}, "started_at": 1.0},
    )

    review = load_review(receipt)

    assert (review["kind"], review["in_flight"]) == (kind, True)


def copy_real_output(tmp_path):
    """The real fixture laid out as the evaluate output that held it: summary, plan and one branch trial."""
    output = tmp_path / "abs-module-cache-flags"
    trial = output / "trials" / "abs-module-cache-flags-r1b1-abs-absolute-fix-precomm"
    (trial / "agent").mkdir(parents=True)
    shutil.copy(REAL_OUTPUT / "r1b1.trajectory.jsonl", trial / "agent" / "trajectory.jsonl")
    for name in ("summary.json", "plan-round1.json"):
        shutil.copy(REAL_OUTPUT / name, output / name)
    for name in ("execution.json", "snapshot.json"):
        shutil.copy(REAL_OUTPUT / name, trial / "agent" / name)
    for name in (
        "restored-state.json",
        trial.name + "__env.podman.json",
        trial.name + "__4b20425b.podman.json",
    ):
        shutil.copy(REAL_OUTPUT / name, trial / name)
    return output


def verdicts(evaluation):
    return [(attempt.name, attempt.resolved) for attempt in evaluation.attempts]


def review_statuses(plan):
    return [attempt["status"] for attempt in plan.review_attempts]


def test_the_real_output_of_a_sprout_evaluation_loads_its_attempts_plan_and_trial(tmp_path):
    """The files are a trimmed copy of what a run wrote, so this guards the stored format."""
    output = copy_real_output(tmp_path)

    evaluation = load_evaluation(output)

    [plan], [loaded] = evaluation.plans, evaluation.trials
    assert verdicts(evaluation) == [
        ("parent", False),
        ("r1b1-abs-absolute-fix-precomm", True),
        ("r1b2-abs-absolute-fix-early", True),
    ]
    assert (plan.round, plan.limit, plan.count_mode, review_statuses(plan)) == (
        1,
        4,
        "adaptive",
        ["validation_error", "validated"],
    )
    assert (loaded.phase, loaded.receipt["sandbox_id"], loaded.restored["snapshot_id"]) == (
        "grading",
        "sprout-69b1adc970d8",
        "sprout-b2867d643e67-0105",
    )


def test_the_real_receipt_and_log_of_an_errored_attempt_give_its_error_and_cause(tmp_path):
    """The log is stored as attempt-4.log.txt, because the repository ignores *.log files."""
    shutil.copy(REAL_OUTPUT / "attempt-4.stage.json", tmp_path / "attempt-4.stage.json")
    shutil.copy(REAL_OUTPUT / "attempt-4.log.txt", tmp_path / "attempt-4.log")

    stage = load_stage(tmp_path / "attempt-4.stage.json", lock_alive=False)

    assert (stage.state, stage.error, stage.cause) == (
        "errored",
        "Stage exited 1; see attempt-4.log",
        "RuntimeError: The source worktree must be clean before a real trial",
    )

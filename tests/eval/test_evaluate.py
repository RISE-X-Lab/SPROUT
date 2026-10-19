"""``Evaluation`` through a Run Store: each attempt a rollout job and a grade job, each round a review job.

The Run Store is the test's own (``run_store``, served by ``run_store_client``). The fake worker of
``tests.eval.worker`` is the evaluation's pause: it settles each job that the search coordinator
submits. Each rollout has a recovery point after each of its three steps. The task is the Harbor
fixture task, whose verifier shares the agent's environment, or a copy of it.
"""

import hashlib
import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from sprout.contract.harbor_task import load_task
from sprout.eval import evaluate
from sprout.runstore.store import Store
from sprout.search.client import RunStoreClient
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH
from tests.eval.worker import Worker, final_snapshot_id
from tests.support.http import chat_server
from tests.support.paths import HARBOR_SHELL_TASK

INSTRUCTION = "Write `ready` followed by a newline into `/app/answer.txt`.\n"
#: The evaluation's clock. Its deadlines are this, the agent's 60 s and two days in the queue.
NOW = 1_000_000.0
DEADLINE = 1_172_860.0
IMAGE_BACKEND = {
    "podman": {"image_env": True, "image_user": True},
    "microvm": {"image_env": True, "image_user": True},
}
TURN = {
    "role": "assistant",
    "content": "Run the tests first.",
    "tool_calls": [
        {
            "id": "repair-1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "pytest -q"}'},
        }
    ],
}


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """The evaluation reads the time ``NOW``."""
    monkeypatch.setattr(evaluate, "time", SimpleNamespace(time=lambda: NOW))


def prepared(
    tmp_path: Path, task: Path, *, digest: str = "sha256:pinned", toml_sha256: str | None = None
) -> Path:
    """The preparation manifest that pins ``task`` in the dataset ``digest``."""
    manifest = tmp_path / "prep.json"
    sha256 = toml_sha256 or hashlib.sha256((task / "task.toml").read_bytes()).hexdigest()
    row = {"name": load_task(task).name, "path": str(task), "task_toml_sha256": sha256}
    manifest.write_text(json.dumps({"dataset_digest": digest, "tasks": [row]}))
    return manifest


def evaluation(
    tmp_path: Path,
    client: RunStoreClient,
    pause: Callable[[], None],
    *options: str,
    task: Path = HARBOR_SHELL_TASK,
    output: str = "out",
    manifest: Path | None = None,
    endpoint: str = "http://model.test/v1",
) -> evaluate.Evaluation:
    """The evaluation that ``python -m sprout.eval.evaluate`` makes of ``task`` with ``options``.

    Without ``manifest``, a preparation manifest pins ``task`` as it is."""
    args = evaluate.parse_args(
        [
            "--task-dir",
            str(task),
            "--dataset-digest",
            "sha256:pinned",
            "--preparation-manifest",
            str(manifest or prepared(tmp_path, task)),
            "--output",
            str(tmp_path / output),
            "--volatile-ok",
            "--model",
            "policy",
            "--model-endpoint",
            endpoint,
            "--model-key-env",
            "MODEL_KEY",
            *options,
        ]
    )
    return evaluate.Evaluation(
        args=args, task=load_task(task), client=client, pause=pause, source_commit="0" * 40
    )


def worker(store: Store, tmp_path: Path, **settings: Any) -> Worker:
    """The fake worker of ``store``, whose attempt directories are under ``tmp_path/artifacts``."""
    return Worker(store, artifacts=tmp_path / "artifacts", context_key="eval", **settings)


def rewards(table: dict[str, float]) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """A grade for each sample: ``table`` maps a sample's name to its reward, and the others score 0."""
    return lambda record: {"reward": table.get(record["sample_slot_id"], 0.0), "reason": None}


def jobs(store: Store) -> list[dict[str, Any]]:
    """The jobs of ``store`` in the order of their submission."""
    return list(reversed(store.list_jobs(limit=1000)))


def attempt_directory(tmp_path: Path, job: dict[str, Any]) -> Path:
    """Where the fake worker ran the job's attempt."""
    return tmp_path / "artifacts" / job["id"] / job["active_attempt"]


def point(
    step: int, run_name: str, *, base: str = "parent", turn: dict[str, Any] | None = TURN, **extra: Any
) -> dict[str, Any]:
    """A point of a review result at ``step`` of the attempt ``base``, with one candidate turn or none."""
    candidates = (
        [{"assistant_turn": turn, "why": "check first", "multiplicity": 1, "leak_terms": None}]
        if turn
        else []
    )
    return {
        "base": base,
        "run_name": run_name,
        "step": step,
        "why": "a decision",
        "candidates": candidates,
        **extra,
    }


def recovery_point_id(client: RunStoreClient, job: dict[str, Any], *, step: int) -> str:
    """The id of the recovery point after tool step ``step`` of the rollout ``job``."""
    points = client.points(job["id"], attempt_id=job["active_attempt"])
    (found,) = [row["id"] for row in points if row["tool_depth"] == step]
    return found


def separate_verifier(tmp_path: Path) -> Path:
    """A copy of the fixture task whose verifier runs in a sandbox of its own, from tests/Dockerfile."""
    task = Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "separate-task"))
    toml = (
        (task / "task.toml")
        .read_text()
        .replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
    )
    (task / "task.toml").write_text(toml + "\n[verifier.environment]\n")
    (task / "tests" / "Dockerfile").write_text("FROM python:3.11-slim\n")
    return task


# --- the parent ---------------------------------------------------------------------------


def test_the_parent_runs_mini_with_the_actors_settings_in_the_tasks_agent_phase(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, clock: None
) -> None:
    options = [
        "--method",
        "baseline",
        "--reasoning-effort",
        "high",
        "--max-turns",
        "40",
        "--max-output-tokens",
        "2000",
    ]

    evaluation(
        tmp_path, run_store_client, worker(run_store, tmp_path), *options, "--command-timeout", "30"
    ).run()

    rollout, _ = jobs(run_store)
    assert (rollout["request"]["profile"], rollout["request"]["max_infra_retries"]) == ("mini", 0)
    assert rollout["request"]["spec"] == {
        "prompt": INSTRUCTION,
        "agent": "mini",
        "harbor_task": str(HARBOR_SHELL_TASK),
        "sandbox_image": "python:3.11-slim",
        "timeout_s": 60.0,
        "checkpoint_enabled": True,
        "keep_sandbox": True,
        "backend": IMAGE_BACKEND,
        "sandbox_resources": {"cpu": 1, "memory_mb": 512},
        "model": "policy",
        "extra": {
            "rollout_contract": {
                "message_export": True,
                "model_endpoint": "http://model.test/v1",
                "model": "policy",
                "api_key_env": "MODEL_KEY",
                "sampling_params": {"max_tokens": 2000, "reasoning_effort": "high"},
                "max_turns": 40,
                "deadline_at": DEADLINE,
            },
            "mini": {"environment": {"timeout": 30, "cwd": "/"}},
        },
    }


@pytest.mark.parametrize("shares", [True, False], ids=["shared", "separate"])
def test_the_parents_grade_runs_the_tasks_verifier_where_the_verifier_runs(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, shares: bool
) -> None:
    """A verifier that shares the agent's environment grades the kept sandbox.

    Another verifier grades the rollout's final snapshot."""
    task = HARBOR_SHELL_TASK if shares else separate_verifier(tmp_path)

    evaluation(
        tmp_path, run_store_client, worker(run_store, tmp_path), "--method", "baseline", task=task
    ).run()

    rollout, grade = jobs(run_store)
    source = {"in_place_job": rollout["id"]} if shares else {"snapshot_id": final_snapshot_id(rollout["id"])}
    assert (grade["request"]["profile"], grade["request"]["max_infra_retries"], grade["request"]["spec"]) == (
        "grade",
        0 if shares else 2,
        {"benchmark": "harbor", "task_dir": str(task), "timeout_s": 60.0, "backend": IMAGE_BACKEND, **source},
    )


def test_a_baseline_summary_holds_the_graded_parent_and_its_jobs(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, clock: None
) -> None:
    pause = worker(run_store, tmp_path, grade=rewards({"parent": 1.0}))

    state = evaluation(tmp_path, run_store_client, pause, "--method", "baseline").run()

    rollout, grade = jobs(run_store)
    (row,) = state["attempts"]
    assert {**row, "started_at": None, "seconds": None} == {
        "name": "parent",
        "round": 0,
        "status": "completed",
        "stop_reason": None,
        "resolved": True,
        "reward": 1.0,
        "reason": None,
        "journal": str(attempt_directory(tmp_path, rollout) / "trajectory.jsonl"),
        "job_id": rollout["id"],
        "attempt_id": rollout["active_attempt"],
        "grade_job_id": grade["id"],
        "log": str(attempt_directory(tmp_path, grade) / "verifier" / "test-stdout.txt"),
        "model_patch": None,
        "started_at": None,
        "validated_at": NOW,
        "seconds": None,
    }
    assert (
        state["complete"],
        state["resolved"],
        state["source_commit"],
        state["branch_schedule_complete"],
    ) == (
        True,
        True,
        "0" * 40,
        True,
    )
    assert json.loads((tmp_path / "out" / "summary.json").read_text()) == state
    assert sorted(path.name for path in (tmp_path / "out" / "groups").iterdir()) == ["parent.json"]


@pytest.mark.parametrize(
    "flag, image",
    [
        ("--no-image-env", {"image_env": False, "image_user": True}),
        ("--no-image-user", {"image_env": True, "image_user": False}),
    ],
)
def test_each_harbor_flag_turns_its_setting_off_in_the_rollout_the_grade_and_the_summary(
    tmp_path: Path,
    run_store: Store,
    run_store_client: RunStoreClient,
    flag: str,
    image: dict[str, bool],
) -> None:
    """ADR-0005: an evaluation keeps Harbor's protocol unless a flag turns a part of it off."""
    state = evaluation(
        tmp_path, run_store_client, worker(run_store, tmp_path), "--method", "baseline", flag
    ).run()

    rollout, grade = jobs(run_store)
    backend = {"podman": image, "microvm": image}
    assert rollout["request"]["spec"]["backend"] == backend
    assert grade["request"]["spec"]["backend"] == backend
    assert {"image_env": state["image_env"], "image_user": state["image_user"]} == image


def test_a_parent_without_checkpoints_takes_no_snapshot_after_each_turn(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """A baseline rollout that nothing branches from needs only its final snapshot."""
    options = ["--method", "baseline", "--checkpoint-mode", "none"]

    evaluation(tmp_path, run_store_client, worker(run_store, tmp_path), *options).run()

    rollout, _ = jobs(run_store)
    assert rollout["request"]["spec"]["checkpoint_enabled"] is False


def test_a_rollout_that_fails_ends_the_evaluation_with_an_error_that_names_its_job(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """Nothing grades the failed rollout, and the summary keeps the error."""
    pause = worker(run_store, tmp_path, failing=frozenset({"parent"}))

    with pytest.raises(
        RuntimeError, match="^parent: the rollout job [0-9a-f]{32} failed: mini did not submit$"
    ):
        evaluation(tmp_path, run_store_client, pause, "--method", "baseline").run()

    (rollout,) = jobs(run_store)
    recorded = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert (recorded["complete"], recorded["attempts"], recorded.get("error")) == (
        False,
        [],
        f"RuntimeError: parent: the rollout job {rollout['id']} failed: mini did not submit",
    )


def imageless(tmp_path: Path) -> Path:
    """A copy of the fixture task without a docker_image."""
    task = Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "imageless-task"))
    toml = (task / "task.toml").read_text().replace('docker_image = "python:3.11-slim"\n', "")
    (task / "task.toml").write_text(toml)
    return task


@pytest.mark.parametrize(
    "setup, error, message",
    [
        (lambda tmp_path: (imageless(tmp_path), {}), ValueError, "names none"),
        (
            lambda tmp_path: (HARBOR_SHELL_TASK, {"digest": "sha256:other"}),
            ValueError,
            "Prepared dataset digest differs",
        ),
        (
            lambda tmp_path: (HARBOR_SHELL_TASK, {"toml_sha256": "ab" * 32}),
            ValueError,
            "Task is absent from the pinned preparation or its definition changed",
        ),
    ],
    ids=["no-image", "other-dataset", "changed-task"],
)
def test_a_task_that_the_evaluation_refuses_makes_no_job(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, setup: Any, error: type, message: str
) -> None:
    task, pinned = setup(tmp_path)
    manifest = prepared(tmp_path, task, **pinned)

    with pytest.raises(error, match=message):
        evaluation(
            tmp_path,
            run_store_client,
            worker(run_store, tmp_path),
            "--method",
            "baseline",
            task=task,
            manifest=manifest,
        ).run()

    assert jobs(run_store) == []


def test_an_output_that_is_not_empty_makes_no_job(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """Each evaluation keeps the only record of its jobs in its own output."""
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "summary.json").write_text("{}")

    with pytest.raises(FileExistsError, match="Use a new output directory for each one-task evaluation"):
        evaluation(tmp_path, run_store_client, worker(run_store, tmp_path), "--method", "baseline").run()

    assert jobs(run_store) == []


def test_a_task_whose_definition_changes_during_the_evaluation_fails_it(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    task = Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "task"))
    settle = worker(run_store, tmp_path)

    def pause_and_edit() -> None:
        settle()
        (task / "task.toml").write_text((task / "task.toml").read_text() + "# edited\n")

    with pytest.raises(RuntimeError, match="^Official task definition changed during evaluation$"):
        evaluation(tmp_path, run_store_client, pause_and_edit, "--method", "baseline", task=task).run()

    assert json.loads((tmp_path / "out" / "summary.json").read_text())["complete"] is False


# --- sprout ---------------------------------------------------------------------------------


def test_a_sprout_round_is_a_review_job_of_the_reviewer_that_the_options_give(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """The analyses, the point review and the candidate review may each take two request timeouts."""
    reviewer = [
        "--review-model",
        "critic",
        "--review-endpoint",
        "http://critic.test/v1",
        "--review-key-env",
        "CRITIC_KEY",
        "--review-effort",
        "high",
        "--branch-count-mode",
        "fixed",
        "--structured-review",
        "--prompt-budget-chars",
        "123456",
        "--analyst-tokens",
        "5000",
        "--review-output-tokens",
        "777",
        "--reviewer-max-attempts",
        "1",
        "--review-timeout",
        "100",
    ]
    pause = worker(run_store, tmp_path, reviews=[[]])

    evaluation(tmp_path, run_store_client, pause, "--rounds", "1", "--branches", "2", *reviewer).run()

    rollout, grade, review = jobs(run_store)
    assert (
        review["request"]["kind"],
        review["request"]["profile"],
        review["request"]["max_infra_retries"],
    ) == (
        "review",
        "review",
        0,
    )
    assert review["request"]["spec"] == {
        "problem": INSTRUCTION,
        "model": "critic",
        "model_endpoint": "http://critic.test/v1",
        "api_key_env": "CRITIC_KEY",
        "parents": [
            {
                "name": "parent",
                "job_id": rollout["id"],
                "attempt_id": rollout["active_attempt"],
                "grade": {
                    "resolved": False,
                    "reason": None,
                    "log": str(attempt_directory(tmp_path, grade) / "verifier" / "test-stdout.txt"),
                },
            }
        ],
        "guidance": "assistant-turn",
        "points": 2,
        "count_mode": "fixed",
        "candidates": 1,
        "max_output_tokens": 777,
        "prompt_chars": 123456,
        "transcript_tokens": 5000,
        "reasoning_effort": "high",
        "temperature": None,
        "max_attempts": 1,
        "round_no": 1,
        "timeout_s": 600.0,
        "request_timeout_s": 100,
        "structured": True,
        "prompts": {"path": str(DEFAULT_PROMPTS_PATH), "sha256": DEFAULT_PROMPTS.sha256},
    }


def test_a_round_branches_at_each_point_that_the_review_wrote_a_turn_for(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, clock: None
) -> None:
    """A point without a turn, or without an available recovery point (step 9 of 3), makes no branch.

    The branch continues the parent from its point with the turn and has a deadline of its own. Its
    grade runs in the sandbox that the branch keeps."""
    review = [point(2, "r1b1-a"), point(3, "r1b2-b", turn=None), point(9, "r1b3-c")]
    pause = worker(run_store, tmp_path, reviews=[review])

    state = evaluation(tmp_path, run_store_client, pause, "--rounds", "1", "--branches", "3").run()

    parent, _, review_job, branch, branch_grade = jobs(run_store)
    assert (
        branch["request"]["parent_point"],
        branch["request"]["continuation"],
        branch["request"]["branch_budget"],
    ) == (
        recovery_point_id(run_store_client, parent, step=2),
        {"guidance": "assistant-turn", "assistant_turn": TURN},
        {"deadline_at": DEADLINE},
    )
    assert branch_grade["request"]["spec"]["in_place_job"] == branch["id"]
    assert [(row["name"], row["round"], row["job_id"]) for row in state["attempts"]] == [
        ("parent", 0, parent["id"]),
        ("r1b1-a", 1, branch["id"]),
    ]
    assert (state["complete"], state["branch_schedule_complete"]) == (True, True)


def test_the_round_plan_records_what_each_selected_point_is_given_and_why_a_point_made_no_branch(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, clock: None
) -> None:
    review = [point(2, "r1b1-a"), point(3, "r1b2-b", turn=None), point(9, "r1b3-c")]
    pause = worker(run_store, tmp_path, reviews=[review])

    evaluation(tmp_path, run_store_client, pause, "--rounds", "1", "--branches", "3").run()

    parent, _, review_job, branch, _ = jobs(run_store)
    plan = json.loads((tmp_path / "out" / "plan-round1.json").read_text())
    assert [
        (row["run_name"], row["branch_guidance"], row.get("assistant_turn"))
        for row in plan["selected_branches"]
    ] == [
        ("r1b1-a", "assistant-turn", TURN),
        ("r1b2-b", "assistant-turn", None),
        ("r1b3-c", "assistant-turn", TURN),
    ]
    assert plan["skipped"] == [
        {"run_name": "r1b2-b", "step": 3, "reason": "the review wrote no turn there"},
        {"run_name": "r1b3-c", "step": 9, "reason": "no available recovery point"},
    ]
    assert plan["branches"] == [
        {
            "name": "r1b1-a",
            "job_id": parent["id"],
            "point_id": recovery_point_id(run_store_client, parent, step=2),
            "continuation": {"guidance": "assistant-turn", "assistant_turn": TURN},
            "budget": {"deadline_at": DEADLINE},
        }
    ]
    assert plan["review_job"] == {
        "job_id": review_job["id"],
        "attempt_id": review_job["active_attempt"],
        "directory": str(attempt_directory(tmp_path, review_job)),
    }
    assert sorted(path.name for path in (tmp_path / "out" / "groups").iterdir()) == [
        "parent.json",
        "review-1.json",
        "round-1.json",
    ]


@pytest.mark.parametrize(
    "guidance, continuation, hint",
    [
        ("user-hint", {"guidance": "user-hint", "hint": "Check the boundary."}, "Check the boundary."),
        ("none", {"guidance": "none"}, None),
    ],
)
def test_a_branch_is_given_what_the_review_wrote_under_its_guidance(
    tmp_path: Path,
    run_store: Store,
    run_store_client: RunStoreClient,
    guidance: str,
    continuation: dict[str, str],
    hint: str | None,
) -> None:
    review = [point(2, "r1b1-a", turn=None, hint="Check the boundary.")]
    pause = worker(run_store, tmp_path, reviews=[review])

    state = evaluation(
        tmp_path, run_store_client, pause, "--rounds", "1", "--branch-guidance", guidance
    ).run()

    _, _, review_job, branch, _ = jobs(run_store)
    assert (review_job["request"]["spec"]["guidance"], branch["request"]["continuation"]) == (
        guidance,
        continuation,
    )
    assert state["attempts"][1].get("hint") == hint


def test_each_round_reviews_every_attempt_so_far_and_branches_from_any_of_them(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    reviews = [[point(2, "r1b1-a")], [point(3, "r2b1-b", base="r1b1-a")]]
    pause = worker(run_store, tmp_path, reviews=reviews)

    state = evaluation(tmp_path, run_store_client, pause, "--rounds", "2", "--branches", "1", "1").run()

    parent, _, _, first, _, second_review, second, _ = jobs(run_store)
    spec = second_review["request"]["spec"]
    assert (spec["round_no"], [(row["name"], row["job_id"]) for row in spec["parents"]]) == (
        2,
        [("parent", parent["id"]), ("r1b1-a", first["id"])],
    )
    assert second["request"]["parent_point"] == recovery_point_id(run_store_client, first, step=3)
    assert [(row["name"], row["round"]) for row in state["attempts"]] == [
        ("parent", 0),
        ("r1b1-a", 1),
        ("r2b1-b", 2),
    ]


def test_each_round_after_the_last_branch_count_reuses_the_last_count(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    reviews = [[point(2, "r1b1-a")], [point(2, "r2b1-a", base="r1b1-a")], []]
    pause = worker(run_store, tmp_path, reviews=reviews)

    evaluation(tmp_path, run_store_client, pause, "--rounds", "3", "--branches", "2", "1").run()

    review_specs = [job["request"]["spec"] for job in jobs(run_store) if job["kind"] == "review"]
    assert [(spec["round_no"], spec["points"]) for spec in review_specs] == [(1, 2), (2, 1), (3, 1)]


def test_the_rounds_end_once_an_attempt_resolves_the_task(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    reviews = [[point(2, "r1b1-a")], [point(3, "r2b1-b", base="r1b1-a")]]
    pause = worker(run_store, tmp_path, reviews=reviews, grade=rewards({"r1b1-a": 1.0}))

    state = evaluation(tmp_path, run_store_client, pause, "--rounds", "2", "--branches", "1", "1").run()

    assert [job["kind"] for job in jobs(run_store)] == ["rollout", "grade", "review", "rollout", "grade"]
    assert (state["resolved"], [row["name"] for row in state["attempts"]]) == (True, ["parent", "r1b1-a"])


@pytest.mark.parametrize(
    "answer, stop_reason",
    [
        (([], "branch 1: no eligible checkpoint at step 9"), "branch 1: no eligible checkpoint at step 9"),
        ([], "the review selected no branches"),
    ],
    ids=["the-plans-reason", "no-reason"],
)
def test_a_review_that_picks_no_point_ends_the_schedule_and_the_summary_says_why(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, answer: Any, stop_reason: str
) -> None:
    pause = worker(run_store, tmp_path, reviews=[answer])

    state = evaluation(tmp_path, run_store_client, pause, "--rounds", "2").run()

    assert [job["kind"] for job in jobs(run_store)] == ["rollout", "grade", "review"]
    assert (state["complete"], state["branch_schedule_complete"], state["stop_reason"]) == (
        True,
        False,
        stop_reason,
    )


def test_the_leak_judge_reaches_the_review_with_the_tasks_hidden_tests(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """Only the leak checks read the hidden tests: the FAIL_TO_PASS ids, the patched paths and the patch."""
    task = Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "task"))
    patch = (
        "diff --git a/tests/test_x.py b/tests/test_x.py\n"
        "+++ b/tests/test_x.py\n"
        "+    assert f('FIXTURE-9') == 1\n"
    )
    (task / "tests" / "test.patch").write_text(patch)
    (task / "tests" / "config.json").write_text(
        json.dumps({"f2p_node_ids": ["tests/test_x.py::test_secret_case"]})
    )
    judge = ["--leak-judge", "--leak-judge-resamples", "3", "--leak-judge-key-env", "TS_KEY"]

    state = evaluation(
        tmp_path,
        run_store_client,
        worker(run_store, tmp_path, reviews=[[]]),
        "--rounds",
        "1",
        *judge,
        task=task,
    ).run()

    _, _, review = jobs(run_store)
    spec = review["request"]["spec"]
    assert spec["leak_judge"] == {
        "threshold": 0.75,
        "model": "jev-1.13.0",
        "resamples": 3,
        "api_key_env": "TS_KEY",
    }
    grade = spec["parents"][0]["grade"]
    assert [grade.get("fail_to_pass"), grade.get("test_patch_paths"), grade.get("test_patch")] == [
        ["tests/test_x.py::test_secret_case"],
        ["tests/test_x.py"],
        str(tmp_path / "out" / "hidden-tests.patch"),
    ]
    assert (tmp_path / "out" / "hidden-tests.patch").read_text() == patch
    assert state["review"]["leak_judge"] == {"threshold": 0.75, "model": "jev-1.13.0", "resamples": 3}


# --- bpo and shepherd -----------------------------------------------------------------------


@pytest.fixture
def baseline(tmp_path: Path, run_store: Store, run_store_client: RunStoreClient) -> Path:
    """The output of a completed baseline evaluation of the fixture task, whose parent failed."""
    evaluation(
        tmp_path, run_store_client, worker(run_store, tmp_path), "--method", "baseline", output="baseline"
    ).run()
    return tmp_path / "baseline"


@pytest.fixture
def model_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """The key that the selectors send from this host."""
    monkeypatch.setenv("MODEL_KEY", "fixture-key")


def completion(content: str, **choice: Any) -> dict[str, Any]:
    message = {"role": "assistant", "content": content}
    return {"choices": [{"index": 0, "finish_reason": "stop", "message": message, **choice}], "usage": {}}


SHEPHERD_ANSWER = '{"checkpoint_step": 2, "reason": "The listing at step 2 missed the file."}'


def test_shepherd_branches_from_the_recorded_parent_one_branch_at_a_time_until_one_resolves(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, baseline: Path, model_key: None
) -> None:
    """Each branch restarts from the selected point without guidance, after the grade of the one before it."""
    parent = jobs(run_store)[0]
    pause = worker(run_store, tmp_path, grade=rewards({"b02-step2": 1.0}))

    with chat_server([completion(SHEPHERD_ANSWER)]) as server:
        options = ["--method", "shepherd", "--parent-output", str(baseline), "--max-rollouts", "4"]
        state = evaluation(tmp_path, run_store_client, pause, *options, endpoint=server.url + "/v1").run()

    branches = jobs(run_store)[2:]
    assert [job["kind"] for job in branches] == ["rollout", "grade", "rollout", "grade"]
    step_2 = recovery_point_id(run_store_client, parent, step=2)
    assert [(job["request"]["parent_point"], job["request"]["continuation"]) for job in branches[::2]] == [
        (step_2, {"guidance": "none"}),
        (step_2, {"guidance": "none"}),
    ]
    assert [(row["name"], row["round"], row["resolved"]) for row in state["attempts"]] == [
        ("parent", 0, False),
        ("b01-step2", 1, False),
        ("b02-step2", 1, True),
    ]


def test_the_shepherd_plan_records_the_shared_parent_and_the_selection(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, baseline: Path, model_key: None
) -> None:
    prompts = tmp_path / "prompts.yaml"
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["shepherd"]["system"] = "Pick the one checkpoint to restart from."
    prompts.write_text(yaml.safe_dump(data))
    parent = jobs(run_store)[0]
    reason = "The listing at step 2 missed the file."

    with chat_server([completion(SHEPHERD_ANSWER)]) as server:
        options = [
            "--method",
            "shepherd",
            "--parent-output",
            str(baseline),
            "--max-rollouts",
            "3",
            "--prompts",
            str(prompts),
        ]
        evaluation(
            tmp_path, run_store_client, worker(run_store, tmp_path), *options, endpoint=server.url + "/v1"
        ).run()

    assert server.requests[0]["messages"][0] == {
        "role": "system",
        "content": "Pick the one checkpoint to restart from.",
    }
    assert json.loads((tmp_path / "out" / "plan.json").read_text()) == {
        "task": "shell-task",
        "method": "shepherd",
        "shared_parent": str(attempt_directory(tmp_path, parent) / "trajectory.jsonl"),
        "max_total_rollouts": 3,
        "budget_includes_shared_initial": True,
        "eligible_steps": [1, 2, 3],
        "selected": [
            {"step": 2, "sibling_index": 2, "reason": reason},
            {"step": 2, "sibling_index": 3, "reason": reason},
        ],
        "branch_guidance": "none",
        "hint_delivery": "fixed-neutral",
    }


def test_bpo_branches_from_the_parent_at_each_point_that_it_selects(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, baseline: Path, model_key: None
) -> None:
    """Steps 1 and 2 have a request after them, so BPO scores them, with ``--bpo-top-k`` top tokens."""
    parent = jobs(run_store)[0]
    flat = {
        "token": "a",
        "logprob": -0.6931471805599453,
        "top_logprobs": [
            {"token": "a", "logprob": -0.6931471805599453},
            {"token": "b", "logprob": -0.6931471805599453},
        ],
    }
    sure = {"token": "a", "logprob": -0.01, "top_logprobs": [{"token": "a", "logprob": -0.01}]}
    replies = [completion("a", logprobs={"content": [sure]}), completion("a", logprobs={"content": [flat]})]

    with chat_server(replies) as server:
        options = [
            "--method",
            "bpo",
            "--parent-output",
            str(baseline),
            "--max-rollouts",
            "3",
            "--bpo-top-k",
            "2",
        ]
        state = evaluation(
            tmp_path,
            run_store_client,
            worker(run_store, tmp_path),
            *options,
            "--bpo-scoring-workers",
            "1",
            endpoint=server.url + "/v1",
        ).run()

    assert [request["top_logprobs"] for request in server.requests] == [2, 2]
    branches = jobs(run_store)[2::2]
    assert [job["request"]["parent_point"] for job in branches] == [
        recovery_point_id(run_store_client, parent, step=2),
        recovery_point_id(run_store_client, parent, step=1),
    ]
    assert [row["name"] for row in state["attempts"]] == ["parent", "b01-step2", "b02-step1"]


def test_a_branch_method_refuses_a_recorded_parent_whose_summary_names_no_key_before_any_job(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, baseline: Path
) -> None:
    """Without the parent's key, its review and its grades could run on a host without the parent's data."""
    summary = json.loads((baseline / "summary.json").read_text())
    del summary["affinity"]
    (baseline / "summary.json").write_text(json.dumps(summary))
    before = jobs(run_store)
    options = ["--rounds", "1", "--parent-output", str(baseline)]

    with pytest.raises(ValueError, match="^The recorded parent's summary names no affinity key$"):
        evaluation(tmp_path, run_store_client, worker(run_store, tmp_path), *options).run()

    assert jobs(run_store) == before


def test_a_branch_method_refuses_a_parent_from_another_run_store_before_any_job(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, baseline: Path
) -> None:
    summary = json.loads((baseline / "summary.json").read_text())
    (baseline / "summary.json").write_text(
        json.dumps({**summary, "runstore": {"url": "http://runstore.test:18110"}})
    )
    before = jobs(run_store)
    options = ["--method", "bpo", "--parent-output", str(baseline)]

    with pytest.raises(ValueError, match="^The recorded parent's jobs are in another Run Store$"):
        evaluation(tmp_path, run_store_client, worker(run_store, tmp_path), *options).run()

    assert jobs(run_store) == before


# --- the affinity key of a task evaluation, and its runs again --------------------------------


def test_every_job_of_an_evaluation_carries_the_one_key_that_its_summary_records(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """ADR-0007: the Run Store runs the jobs of one key on one host, where the parent's data is.

    The reviews of the two rounds are two jobs under the one key."""
    reviews = [[point(2, "r1b1-a")], [point(3, "r2b1-b", base="r1b1-a")]]
    pause = worker(run_store, tmp_path, reviews=reviews)

    state = evaluation(tmp_path, run_store_client, pause, "--rounds", "2", "--branches", "1", "1").run()

    key = state["affinity"]
    assert re.fullmatch("eval-[0-9a-f]{32}", key), key
    assert [(job["kind"], job["affinity"]) for job in jobs(run_store)] == [
        ("rollout", key),
        ("grade", key),
        ("review", key),
        ("rollout", key),
        ("grade", key),
        ("review", key),
        ("rollout", key),
        ("grade", key),
    ]


def test_two_evaluations_of_one_task_carry_two_keys(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """Every evaluation names its groups alike, so a key of the group id would bind them to one host."""
    first = evaluation(
        tmp_path, run_store_client, worker(run_store, tmp_path), "--method", "baseline", output="one"
    ).run()

    second = evaluation(
        tmp_path, run_store_client, worker(run_store, tmp_path), "--method", "baseline", output="two"
    ).run()

    assert first["affinity"] != second["affinity"]


def test_a_branch_method_carries_the_key_of_the_evaluation_whose_parent_it_branches_from(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, baseline: Path
) -> None:
    """A branch takes the key of the parent. Its review and its grade run on the same host."""
    recorded = json.loads((baseline / "summary.json").read_text())["affinity"]
    pause = worker(run_store, tmp_path, reviews=[[point(2, "r1b1-a")]])
    options = ["--rounds", "1", "--branches", "1", "--parent-output", str(baseline)]

    state = evaluation(tmp_path, run_store_client, pause, *options).run()

    assert (state["affinity"], [(job["kind"], job["affinity"]) for job in jobs(run_store)[2:]]) == (
        recorded,
        [("review", recorded), ("rollout", recorded), ("grade", recorded)],
    )


def test_a_job_that_loses_its_host_runs_the_evaluation_again_from_its_parent_under_the_next_key(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """ADR-0007: the later jobs read the data of the lost host, so the parent runs again on another one.

    The branch's grade in its kept sandbox is the lost job. The next key gives each stage a new
    idempotency key, so each stage is a new job."""
    review = [point(2, "r1b1-a")]
    pause = worker(run_store, tmp_path, reviews=[review, review], losing=["r1b1-a"])

    state = evaluation(tmp_path, run_store_client, pause, "--rounds", "1", "--branches", "1").run()

    first = state["earlier_runs"][0]["affinity"]
    again = first + "-r1"
    assert (state["affinity"], [(job["kind"], job["affinity"], job["state"]) for job in jobs(run_store)]) == (
        again,
        [
            ("rollout", first, "succeeded"),
            ("grade", first, "succeeded"),
            ("review", first, "succeeded"),
            ("rollout", first, "succeeded"),
            ("grade", first, "failed"),
            ("rollout", again, "succeeded"),
            ("grade", again, "succeeded"),
            ("review", again, "succeeded"),
            ("rollout", again, "succeeded"),
            ("grade", again, "succeeded"),
        ],
    )


def test_an_evaluation_that_runs_again_keeps_the_records_of_its_earlier_run(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """The earlier run's rows stay in the summary, and its groups and plans move to its own directory."""
    review = [point(2, "r1b1-a")]
    pause = worker(run_store, tmp_path, reviews=[review, review], losing=["r1b1-a"])

    state = evaluation(tmp_path, run_store_client, pause, "--rounds", "1", "--branches", "1").run()

    (earlier,) = state["earlier_runs"]
    parent, _, _, _, _, again, _, _, branch, _ = jobs(run_store)
    assert (
        [(row["name"], row["job_id"]) for row in earlier["attempts"]],
        earlier["error"],
        [(row["name"], row["job_id"]) for row in state["attempts"]],
    ) == (
        [("parent", parent["id"])],
        f"round-1: a host that held the jobs of the key {earlier['affinity']} was lost",
        [("parent", again["id"]), ("r1b1-a", branch["id"])],
    )
    output = tmp_path / "out"
    kept = output / "earlier-runs" / earlier["affinity"]
    assert (
        sorted(path.name for path in output.iterdir()),
        sorted(path.name for path in kept.iterdir()),
        sorted(path.name for path in (kept / "groups").iterdir()),
    ) == (
        ["earlier-runs", "groups", "plan-round1.json", "summary.json"],
        ["groups", "plan-round1.json"],
        ["parent.json", "review-1.json", "round-1.json"],
    )


def test_an_evaluation_whose_third_run_loses_its_host_fails_with_the_reason_data_lost(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """A task evaluation runs again at most 2 times (ADR-0007)."""
    pause = worker(run_store, tmp_path, losing=["parent"] * 3)

    with pytest.raises(
        evaluate.DataLost,
        match=r"^parent: a host that held the jobs of the key eval-[0-9a-f]{32}-r2 was lost\."
        r" A task evaluation runs again at most 2 times\.$",
    ):
        evaluation(tmp_path, run_store_client, pause, "--method", "baseline").run()

    recorded = json.loads((tmp_path / "out" / "summary.json").read_text())
    first = recorded["earlier_runs"][0]["affinity"]
    assert (
        recorded["complete"],
        recorded["reason"],
        recorded["affinity"],
        [run["affinity"] for run in recorded["earlier_runs"]],
        [job["affinity"] for job in jobs(run_store)],
    ) == (False, "data_lost", first + "-r2", [first, first + "-r1"], [first, first + "-r1", first + "-r2"])


def test_a_branch_method_whose_recorded_parent_loses_its_host_fails_at_once_with_the_reason_data_lost(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, baseline: Path
) -> None:
    """The parent is the recorded evaluation's, and its data went with the host, so nothing runs again."""
    before = len(jobs(run_store))
    pause = worker(run_store, tmp_path, losing=["review-1"])

    with pytest.raises(
        evaluate.DataLost,
        match=r"^review-1: a host that held the jobs of the key eval-[0-9a-f]{32} was lost\."
        r" The evaluation branches from a recorded parent, so it does not run again\.$",
    ):
        evaluation(tmp_path, run_store_client, pause, "--rounds", "1", "--parent-output", str(baseline)).run()

    recorded = json.loads((tmp_path / "out" / "summary.json").read_text())
    assert (
        recorded["reason"],
        recorded["earlier_runs"],
        [job["kind"] for job in jobs(run_store)[before:]],
    ) == (
        "data_lost",
        [],
        ["review"],
    )

"""The Run Store jobs that ``sprout.eval.jobs`` builds from a Harbor task, without a Run Store.

The task is the Harbor fixture task, which ``task`` places at ``/path/to/shell-task``: its agent
and its verifier each have 60 s, its sandbox has 1 CPU and 512 MB, and its verifier shares the
agent's environment.
"""

import shutil
from dataclasses import replace
from pathlib import Path

import pytest

from sprout.contract.harbor_task import CollectCommand, HarborTask, load_task
from sprout.eval.jobs import (
    Actor,
    ImageOptions,
    Reviewer,
    branch_deadline,
    check_environment,
    grade_files,
    grade_template,
    mini_job,
    recovery_point,
    review_continuation,
    review_job,
    review_parent,
    rollout_job,
)
from tests.support.paths import HARBOR_SHELL_TASK

INSTRUCTION = "Write `ready` followed by a newline into `/app/answer.txt`.\n"
HARBOR_IMAGE = ImageOptions(env=True, user=True)
#: Epoch seconds at which a test builds a job with a deadline.
NOW = 1_000_000.0


def task(**changes: object) -> HarborTask:
    """The fixture task at ``/path/to/shell-task``, with ``changes`` to its fields."""
    return replace(load_task(HARBOR_SHELL_TASK), task_dir=Path("/path/to/shell-task"), **changes)


def actor(**changes: object) -> Actor:
    """An actor of the model ``policy``, with ``changes`` to its settings."""
    settings: dict[str, object] = {
        "model": "policy",
        "model_endpoint": "http://model.test/v1",
        "api_key_env": "MODEL_KEY",
        "max_output_tokens": 64000,
        "max_turns": 40,
        "command_timeout_s": 600,
    }
    return Actor(**{**settings, **changes})  # type: ignore[arg-type]


def reviewer(**changes: object) -> Reviewer:
    """A reviewer of the model ``critic``, with ``changes`` to its settings."""
    settings: dict[str, object] = {
        "profile": "review",
        "model": "critic",
        "model_endpoint": "http://critic.test/v1",
        "api_key_env": "CRITIC_KEY",
        "guidance": "assistant-turn",
        "count_mode": "adaptive",
        "max_output_tokens": 777,
        "prompt_chars": 123456,
        "transcript_tokens": 5000,
        "max_attempts": 2,
        "request_timeout_s": 100.0,
        "structured": True,
    }
    return Reviewer(**{**settings, **changes})  # type: ignore[arg-type]


def task_copy(tmp_path: Path) -> Path:
    """A copy of the fixture task directory, for a test that changes its files."""
    return Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "task"))


# --- rollouts ---------------------------------------------------------------------------


def test_a_rollout_runs_the_agent_in_the_tasks_agent_phase() -> None:
    job = rollout_job(
        task(), agent="oracle", profile="actors", checkpoint=False, image=ImageOptions(env=True, user=False)
    )

    assert job == {
        "kind": "rollout",
        "profile": "actors",
        "max_infra_retries": 0,
        "spec": {
            "prompt": INSTRUCTION,
            "agent": "oracle",
            "harbor_task": "/path/to/shell-task",
            "sandbox_image": "python:3.11-slim",
            "timeout_s": 60.0,
            "checkpoint_enabled": False,
            "keep_sandbox": True,
            "backend": {
                "podman": {"image_env": True, "image_user": False},
                "microvm": {"image_env": True, "image_user": False},
            },
            "sandbox_resources": {"cpu": 1, "memory_mb": 512},
        },
    }


def test_a_task_without_an_agent_timeout_rolls_out_for_the_run_spec_default() -> None:
    job = rollout_job(
        task(agent_timeout_s=None), agent="nop", profile="mini", checkpoint=False, image=HARBOR_IMAGE
    )

    assert job["spec"]["timeout_s"] == 3600.0


@pytest.mark.parametrize(
    "cpus, memory_mb, shape",
    [
        (1, 512, {"cpu": 1, "memory_mb": 512}),
        (None, 512, {"memory_mb": 512}),
        (2, None, {"cpu": 2}),
        (None, None, None),
    ],
)
def test_the_sandbox_shape_names_only_the_resources_that_the_task_sets(
    cpus: int | None, memory_mb: int | None, shape: dict[str, int] | None
) -> None:
    """A resource that the task leaves out is the backend's default, so the job does not name it."""
    job = rollout_job(
        task(cpus=cpus, memory_mb=memory_mb),
        agent="nop",
        profile="mini",
        checkpoint=False,
        image=HARBOR_IMAGE,
    )

    assert job["spec"].get("sandbox_resources") == shape


@pytest.mark.parametrize(
    "shares, kept, grade",
    [
        (
            True,
            True,
            {
                "profile": "grade",
                "max_infra_retries": 0,
                "in_place": True,
                "spec": {
                    "benchmark": "harbor",
                    "task_dir": "/path/to/shell-task",
                    "timeout_s": 60.0,
                    "backend": {
                        "podman": {"image_env": True, "image_user": True},
                        "microvm": {"image_env": True, "image_user": True},
                    },
                },
            },
        ),
        (
            False,
            False,
            {
                "profile": "grade",
                "spec": {
                    "benchmark": "harbor",
                    "task_dir": "/path/to/shell-task",
                    "timeout_s": 60.0,
                    "backend": {
                        "podman": {"image_env": True, "image_user": True},
                        "microvm": {"image_env": True, "image_user": True},
                    },
                },
            },
        ),
    ],
)
def test_a_verifier_that_shares_the_agents_environment_grades_in_the_sandbox_the_rollout_keeps(
    shares: bool, kept: bool, grade: dict[str, object]
) -> None:
    """ADR-0004. A grade in the kept sandbox is never retried: its first attempt destroys the sandbox.

    A separate verifier grades the rollout's final snapshot, with the Run Store's default retries."""
    shaped = task(verifier_shares_environment=shares)

    rollout = rollout_job(shaped, agent="mini", profile="mini", checkpoint=True, image=HARBOR_IMAGE)
    template = grade_template(shaped, profile="grade", image=HARBOR_IMAGE)

    assert (rollout["spec"]["keep_sandbox"], template) == (kept, grade)


def test_a_grade_has_the_verifier_timeout_and_the_timeout_of_each_collect_command() -> None:
    collecting = task(
        collect=(CollectCommand("cat /app/a.log", 30.0), CollectCommand("cat /app/b.log", 15.5))
    )

    template = grade_template(collecting, profile="grade", image=HARBOR_IMAGE)

    assert template["spec"]["timeout_s"] == 105.5


def test_a_mini_job_runs_the_actors_model_under_its_rollout_contract() -> None:
    """The run is refused when it is still queued two days past its agent timeout."""
    job = mini_job(
        task(),
        profile="mini",
        actor=actor(reasoning_effort="high"),
        checkpoint=True,
        image=HARBOR_IMAGE,
        now=NOW,
    )

    spec = job["spec"]
    assert (spec["agent"], spec["model"], spec["checkpoint_enabled"]) == ("mini", "policy", True)
    assert spec["extra"] == {
        "rollout_contract": {
            "message_export": True,
            "model_endpoint": "http://model.test/v1",
            "model": "policy",
            "api_key_env": "MODEL_KEY",
            "sampling_params": {"max_tokens": 64000, "reasoning_effort": "high"},
            "max_turns": 40,
            "deadline_at": 1_172_860.0,
        },
        "mini": {"environment": {"timeout": 600, "cwd": "/"}},
    }


def test_a_mini_job_sends_no_reasoning_effort_when_the_actor_has_none() -> None:
    job = mini_job(task(), profile="mini", actor=actor(), checkpoint=True, image=HARBOR_IMAGE, now=NOW)

    assert job["spec"]["extra"]["rollout_contract"]["sampling_params"] == {"max_tokens": 64000}


@pytest.mark.parametrize(
    "workdir, environment",
    [("/app", {"timeout": 600, "cwd": "/app"}), (None, {"timeout": 600})],
)
def test_mini_works_in_the_tasks_workdir_or_else_where_it_finds_itself(
    workdir: str | None, environment: dict[str, object]
) -> None:
    job = mini_job(
        task(workdir=workdir), profile="mini", actor=actor(), checkpoint=True, image=HARBOR_IMAGE, now=NOW
    )

    assert job["spec"]["extra"]["mini"]["environment"] == environment


@pytest.mark.parametrize("agent_timeout_s, deadline", [(60.0, 1_172_860.0), (None, 1_176_400.0)])
def test_a_branch_deadline_counts_from_the_time_the_branch_is_made(
    agent_timeout_s: float | None, deadline: float
) -> None:
    """The agent timeout (3600 s by default) and two days in the queue, from ``now``."""
    assert branch_deadline(task(agent_timeout_s=agent_timeout_s), now=NOW) == deadline


# --- the task's environment ---------------------------------------------------------------


@pytest.mark.parametrize(
    "files, removed, message",
    [
        (
            {"environment/data.txt": "x\n"},
            "environment/Dockerfile",
            "Harbor uploads these files into the sandbox",
        ),
        ({"environment/Dockerfile": "FROM python:3.11-slim\nRUN touch /ready\n"}, None, "COPY and RUN steps"),
        (
            {
                "environment/Dockerfile": "FROM python:3.11-slim\nCOPY data.txt /data.txt\n",
                "environment/data.txt": "",
            },
            None,
            "COPY and RUN steps",
        ),
    ],
)
def test_a_task_whose_environment_is_more_than_its_image_is_refused(
    tmp_path: Path, files: dict[str, str], removed: str | None, message: str
) -> None:
    """Harbor uploads ``environment/`` or replays the Dockerfile's steps on the image.

    A rollout does neither."""
    directory = task_copy(tmp_path)
    for name, text in files.items():
        (directory / name).write_text(text)
    if removed:
        (directory / removed).unlink()

    with pytest.raises(ValueError, match=message):
        check_environment(load_task(directory))


def test_a_task_without_a_docker_image_is_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="a rollout starts from the task's docker_image, and it names none"):
        check_environment(replace(load_task(task_copy(tmp_path)), docker_image=None))


@pytest.mark.parametrize(
    "dockerfile",
    [
        "FROM python:3.11-slim\n",
        "FROM debian:12\nRUN apt-get install -y python3\n",
        None,
    ],
    ids=["from-the-image-alone", "the-recipe-of-the-image", "empty-environment"],
)
def test_a_task_whose_image_is_its_whole_environment_is_accepted(
    tmp_path: Path, dockerfile: str | None
) -> None:
    """A Dockerfile FROM another image is the recipe that built the task's image, which holds its result."""
    directory = task_copy(tmp_path)
    (directory / "environment" / "Dockerfile").unlink()
    if dockerfile is not None:
        (directory / "environment" / "Dockerfile").write_text(dockerfile)

    check_environment(load_task(directory))


# --- reviews --------------------------------------------------------------------------------


def parent(name: str) -> dict[str, object]:
    """What a review reads of one graded attempt (``review_parent``)."""
    return {
        "name": name,
        "job_id": "job-" + name,
        "attempt_id": "attempt-" + name,
        "grade": {"resolved": False},
    }


def test_a_review_job_gives_the_review_its_reviewer_and_its_parents() -> None:
    prompts = {"path": "/path/to/prompts.yaml", "sha256": "ab" * 32}
    judge = {"threshold": 0.75, "model": "jev-1.13.0", "resamples": 3, "api_key_env": "TS_KEY"}

    job = review_job(
        reviewer(reasoning_effort="high", prompts=prompts, leak_judge=judge),
        problem=INSTRUCTION,
        parents=[parent("parent")],
        points=3,
        round_no=2,
    )

    assert job == {
        "kind": "review",
        "profile": "review",
        "max_infra_retries": 0,
        "spec": {
            "problem": INSTRUCTION,
            "model": "critic",
            "model_endpoint": "http://critic.test/v1",
            "api_key_env": "CRITIC_KEY",
            "parents": [parent("parent")],
            "guidance": "assistant-turn",
            "points": 3,
            "count_mode": "adaptive",
            "candidates": 1,
            "max_output_tokens": 777,
            "prompt_chars": 123456,
            "transcript_tokens": 5000,
            "reasoning_effort": "high",
            "temperature": None,
            "max_attempts": 2,
            "round_no": 2,
            "timeout_s": 600.0,
            "request_timeout_s": 100.0,
            "structured": True,
            "prompts": prompts,
            "leak_judge": judge,
        },
    }


def test_a_review_job_without_a_prompts_file_or_a_judge_names_neither() -> None:
    """Without ``prompts``, the worker reads its packaged prompts file.

    Without ``leak_judge``, no judge runs."""
    job = review_job(reviewer(), problem=INSTRUCTION, parents=[parent("parent")], points=1, round_no=1)

    assert {"prompts", "leak_judge"} & set(job["spec"]) == set()


@pytest.mark.parametrize(
    "parents, timeout_s", [(1, 600.0), (8, 600.0), (9, 800.0), (16, 800.0), (17, 1000.0)]
)
def test_a_review_may_take_two_request_timeouts_for_each_of_its_stages(
    parents: int, timeout_s: float
) -> None:
    """The analyses run eight at a time, then the point review and the candidate review run.

    The transport retries each timed-out request once, so each stage may take two request timeouts."""
    job = review_job(
        reviewer(request_timeout_s=100.0),
        problem=INSTRUCTION,
        parents=[parent(f"b{index}") for index in range(parents)],
        points=1,
        round_no=1,
    )

    assert job["spec"]["timeout_s"] == timeout_s


@pytest.mark.parametrize(
    "row, expected",
    [
        (
            {
                "name": "r1b1-a",
                "job_id": "job-3",
                "attempt_id": "attempt-3",
                "resolved": False,
                "reason": None,
                "log": "/path/to/verifier/test-stdout.txt",
                "model_patch": "/path/to/artifacts/model.patch",
                "hint": "Check the boundary.",
            },
            {
                "name": "r1b1-a",
                "job_id": "job-3",
                "attempt_id": "attempt-3",
                "grade": {
                    "resolved": False,
                    "reason": None,
                    "fail_to_pass": ["tests/test_x.py::test_case"],
                    "log": "/path/to/verifier/test-stdout.txt",
                    "model_patch": "/path/to/artifacts/model.patch",
                },
                "hint": "Check the boundary.",
            },
        ),
        (
            {
                "name": "parent",
                "job_id": "job-0",
                "attempt_id": "attempt-0",
                "resolved": False,
                "reason": "sandbox_lost",
                "log": None,
                "model_patch": None,
            },
            {
                "name": "parent",
                "job_id": "job-0",
                "attempt_id": "attempt-0",
                "grade": {
                    "resolved": False,
                    "reason": "sandbox_lost",
                    "fail_to_pass": ["tests/test_x.py::test_case"],
                },
            },
        ),
    ],
    ids=["with-files-and-hint", "without"],
)
def test_a_review_reads_each_attempts_verdict_its_verifier_files_and_its_hint(
    row: dict[str, object], expected: dict[str, object]
) -> None:
    assert review_parent(row, hidden={"fail_to_pass": ["tests/test_x.py::test_case"]}) == expected


def test_the_analyst_reads_the_verifier_log_and_the_first_model_patch_of_a_grade(tmp_path: Path) -> None:
    """The layout of a Harbor trial: ``verifier/test-stdout.txt`` beside ``artifacts/``."""
    logs = tmp_path / "trial" / "verifier"
    logs.mkdir(parents=True)
    (logs / "test-stdout.txt").write_text("1 failed\n")
    for name in ("b", "a"):
        (tmp_path / "trial" / "artifacts" / name).mkdir(parents=True)
        (tmp_path / "trial" / "artifacts" / name / "model.patch").write_text("diff\n")

    files = grade_files({"reward": 0.0, "verifier": {"logs": str(logs)}})

    assert files == {
        "log": str(logs / "test-stdout.txt"),
        "model_patch": str(tmp_path / "trial" / "artifacts" / "a" / "model.patch"),
    }


@pytest.mark.parametrize("make_logs", [False, True], ids=["no-verifier-logs", "empty-verifier-logs"])
def test_a_grade_without_verifier_files_gives_the_analyst_none(tmp_path: Path, make_logs: bool) -> None:
    logs = tmp_path / "trial" / "verifier"
    logs.mkdir(parents=True)
    result = {"reward": 0.0, "reason": "sandbox_lost", "verifier": {"logs": str(logs)} if make_logs else {}}

    assert grade_files(result) == {"log": None, "model_patch": None}


TURN = {"role": "assistant", "content": "Run the tests.", "tool_calls": []}


@pytest.mark.parametrize(
    "guidance, point, continuation",
    [
        (
            "assistant-turn",
            {"candidates": [{"assistant_turn": TURN}, {"assistant_turn": {**TURN, "content": "Other."}}]},
            {"guidance": "assistant-turn", "assistant_turn": TURN},
        ),
        ("assistant-turn", {"candidates": []}, None),
        (
            "user-hint",
            {"candidates": [], "hint": "Check the boundary."},
            {"guidance": "user-hint", "hint": "Check the boundary."},
        ),
        ("none", {"candidates": []}, {"guidance": "none"}),
    ],
    ids=["first-candidate", "no-turn-written", "user-hint", "none"],
)
def test_a_branch_is_given_what_the_review_wrote_at_its_point(
    guidance: str, point: dict[str, object], continuation: dict[str, object] | None
) -> None:
    assert review_continuation(point, guidance=guidance) == continuation


AT_STEP_2 = {"id": "p2", "tool_depth": 2, "available": True}


@pytest.mark.parametrize(
    "points, found",
    [
        ([{"id": "p1", "tool_depth": 1, "available": True}, AT_STEP_2], AT_STEP_2),
        ([{"id": "p2", "tool_depth": 2, "available": False}], None),
        ([{"id": "p2", "tool_depth": 2}], None),
        ([{"id": "p1", "tool_depth": 1, "available": True}], None),
    ],
    ids=["available-at-the-step", "unavailable", "availability-unknown", "none-at-the-step"],
)
def test_a_branch_starts_only_from_an_available_recovery_point_at_its_step(
    points: list[dict[str, object]], found: dict[str, object] | None
) -> None:
    assert recovery_point(points, step=2) == found

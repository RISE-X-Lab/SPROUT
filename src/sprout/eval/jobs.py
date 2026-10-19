"""The Run Store jobs of a gate or an evaluation, built from a Harbor task (``contract.harbor_task``).

A rollout job runs one agent in the task's agent phase as Harbor gives it: the instruction,
the image, the CPUs and memory, and the agent timeout. The worker starts the rollout on the
task's agent network. A grade job runs the task's verifier: in the sandbox that the rollout
kept when the verifier shares the agent's environment (ADR-0004), else from the rollout's
final snapshot. Each rollout and grade asks the worker's backend for the image's USER and ENV
(``ImageOptions``), as Harbor runs the agent's commands, ``solve.sh`` and ``test.sh``.

The ``*_sample`` and ``group_request`` functions put jobs into the group requests that
``sprout.search.group`` checks. Nothing here talks to the Run Store. docs/EVALUATION.md
describes how a task becomes jobs.
"""

from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sprout.contract.harbor_task import HarborTask
from sprout.contract.run_spec import RunSpec
from sprout.worker.jobs.harbor.overlay import overlay_for

#: How long a rollout may wait for a worker before its rollout contract refuses it. Arbitrary:
#: a wait this long means that no worker serves the profile.
QUEUE_ALLOWANCE_S = 2 * 24 * 3600.0
#: How many model requests a review job runs in parallel: the default of
#: ``ReviewSpec.max_concurrent_requests``, which ``review_job`` does not set.
_REVIEW_REQUESTS = 8


@dataclass(frozen=True)
class Actor:
    """The model that a ``mini`` rollout runs, and the limits it runs under."""

    model: str
    model_endpoint: str
    #: The worker-side variable that holds the endpoint's bearer key.
    api_key_env: str
    max_output_tokens: int
    max_turns: int
    command_timeout_s: int
    reasoning_effort: str | None = None


@dataclass(frozen=True)
class ImageOptions:
    """Whether the agent's commands, ``solve.sh`` and ``test.sh`` get the image's ENV and run as its USER.

    They become ``image_env`` and ``image_user`` in each section of a job's ``backend`` (docs/RUNSTORE.md).
    """

    env: bool
    user: bool


def add_image_flags(parser: argparse.ArgumentParser) -> None:
    """Add ``--no-image-env`` and ``--no-image-user`` to ``parser``. ``image_options`` reads them."""
    parser.add_argument(
        "--no-image-env",
        action="store_true",
        help="do not set the image's ENV again on the commands that run as its USER (backend image_env)",
    )
    parser.add_argument(
        "--no-image-user",
        action="store_true",
        help="run the agent's commands, solve.sh and test.sh as root, not as the image's USER "
        "(backend image_user)",
    )


def image_options(args: argparse.Namespace) -> ImageOptions:
    """The ``ImageOptions`` that the flags of ``add_image_flags`` give."""
    return ImageOptions(env=not args.no_image_env, user=not args.no_image_user)


@dataclass(frozen=True)
class Reviewer:
    """The analyst and the reviewer of a sprout round, as its review job runs them (``ReviewSpec``)."""

    profile: str
    model: str
    model_endpoint: str
    #: The worker-side variable that holds the endpoint's bearer key.
    api_key_env: str
    guidance: str
    count_mode: str
    max_output_tokens: int
    prompt_chars: int
    transcript_tokens: int
    max_attempts: int
    request_timeout_s: float
    structured: bool
    reasoning_effort: str | None = None
    #: ``{"path", "sha256"}`` of the prompts file, None for the packaged one.
    prompts: dict[str, str] | None = None
    #: ``ReviewSpec.leak_judge``: the judge's settings and its key's variable. None: no judge.
    leak_judge: dict[str, Any] | None = None


def check_environment(task: HarborTask) -> None:
    """Raise ValueError unless a sandbox from the task's ``docker_image`` is the agent's whole environment.

    Harbor also uploads a task's ``environment/`` or replays a Dockerfile on its image, which no rollout does.
    """
    if task.docker_image is None:
        raise ValueError(f"{task.task_dir}: a rollout starts from the task's docker_image, and it names none")
    environment = task.task_dir / "environment"
    if task.dockerfile is None:
        if environment.is_dir() and any(environment.iterdir()):
            raise ValueError(
                f"{environment}: Harbor uploads these files into the sandbox, and a rollout does not"
            )
        return
    overlay = overlay_for(environment, task.docker_image)
    if overlay is not None and (overlay.copies or overlay.runs):
        raise ValueError(f"{task.dockerfile}: its COPY and RUN steps on the task's image are not replayed")


def rollout_job(
    task: HarborTask, *, agent: str, profile: str, checkpoint: bool, image: ImageOptions
) -> dict[str, Any]:
    """The rollout JobSpec of ``agent`` (oracle, nop, or mini through ``mini_job``) on the task.

    With ``checkpoint``, the rollout takes a snapshot after each turn. Every rollout ends with one snapshot.
    """
    spec: dict[str, Any] = {
        "prompt": task.instruction,
        "agent": agent,
        "harbor_task": str(task.task_dir),
        "sandbox_image": task.docker_image,
        "timeout_s": task.agent_timeout_s or RunSpec.timeout_s,
        "checkpoint_enabled": checkpoint,
        "keep_sandbox": task.verifier_shares_environment,
        "backend": _image_backend(image),
    }
    resources = {name: value for name, value in (("cpu", task.cpus), ("memory_mb", task.memory_mb)) if value}
    if resources:
        spec["sandbox_resources"] = resources
    # One attempt is one execution: a retry would resume from a recovery point with another journal.
    return {"kind": "rollout", "profile": profile, "max_infra_retries": 0, "spec": spec}


def mini_job(
    task: HarborTask, *, profile: str, actor: Actor, checkpoint: bool, image: ImageOptions, now: float
) -> dict[str, Any]:
    """The rollout JobSpec of mini running ``actor``'s model on the task.

    A run that is still queued ``QUEUE_ALLOWANCE_S`` past its agent timeout after ``now`` (epoch seconds) is
    refused.
    """
    job = rollout_job(task, agent="mini", profile=profile, checkpoint=checkpoint, image=image)
    spec = job["spec"]
    spec["model"] = actor.model
    spec["extra"] = {
        "rollout_contract": _contract(actor, deadline_at=now + spec["timeout_s"] + QUEUE_ALLOWANCE_S),
        "mini": {"environment": {"timeout": actor.command_timeout_s, **_workdir(task)}},
    }
    return job


def _contract(actor: Actor, *, deadline_at: float) -> dict[str, Any]:
    sampling: dict[str, int | str] = {"max_tokens": actor.max_output_tokens}
    if actor.reasoning_effort:
        sampling["reasoning_effort"] = actor.reasoning_effort
    return {
        # The export also makes a run that hits its turn or time limit a truncated, gradable result.
        "message_export": True,
        "model_endpoint": actor.model_endpoint,
        "model": actor.model,
        "api_key_env": actor.api_key_env,
        "sampling_params": sampling,
        "max_turns": actor.max_turns,
        "deadline_at": deadline_at,
    }


def _workdir(task: HarborTask) -> dict[str, str]:
    """mini's working directory: the task's ``workdir`` when it names one, else what mini discovers."""
    return {"cwd": task.workdir} if task.workdir is not None else {}


def _image_backend(image: ImageOptions) -> dict[str, dict[str, bool]]:
    """The backend settings of a job that give it the image's ENV and USER, or not."""
    options = {"image_env": image.env, "image_user": image.user}
    # A job does not know which backend its worker profile names. The worker reads only that one's section.
    return {"podman": dict(options), "microvm": dict(options)}


def grade_template(task: HarborTask, *, profile: str, image: ImageOptions) -> dict[str, Any]:
    """The grade template of a rollout of the task: the task's verifier (a ``harbor`` grade)."""
    spec = {
        "benchmark": "harbor",
        "task_dir": str(task.task_dir),
        "timeout_s": task.verifier_timeout_s + sum(collect.timeout_s for collect in task.collect),
        "backend": _image_backend(image),
    }
    if task.verifier_shares_environment:
        # The first attempt destroys the kept sandbox, so a retry would score a lost sandbox.
        return {"profile": profile, "max_infra_retries": 0, "in_place": True, "spec": spec}
    return {"profile": profile, "spec": spec}


def branch_deadline(task: HarborTask, *, now: float) -> float:
    """The contract deadline of a branch made at ``now``: its own, not the one its parent inherits."""
    return now + (task.agent_timeout_s or RunSpec.timeout_s) + QUEUE_ALLOWANCE_S


def review_job(
    reviewer: Reviewer, *, problem: str, parents: list[dict[str, Any]], points: int, round_no: int
) -> dict[str, Any]:
    """The review JobSpec of a sprout round: ``points`` branch points over ``parents``, one candidate each."""
    spec = {
        "problem": problem,
        "model": reviewer.model,
        "model_endpoint": reviewer.model_endpoint,
        "api_key_env": reviewer.api_key_env,
        "parents": parents,
        "guidance": reviewer.guidance,
        "points": points,
        "count_mode": reviewer.count_mode,
        "candidates": 1,
        "max_output_tokens": reviewer.max_output_tokens,
        "prompt_chars": reviewer.prompt_chars,
        "transcript_tokens": reviewer.transcript_tokens,
        "reasoning_effort": reviewer.reasoning_effort,
        # The endpoint's default, as for any analyst or reviewer request of a sprout round.
        "temperature": None,
        "max_attempts": reviewer.max_attempts,
        "round_no": round_no,
        "timeout_s": _review_timeout(reviewer.request_timeout_s, parents=len(parents)),
        "request_timeout_s": reviewer.request_timeout_s,
        "structured": reviewer.structured,
    }
    if reviewer.prompts is not None:
        spec["prompts"] = reviewer.prompts
    if reviewer.leak_judge is not None:
        spec["leak_judge"] = reviewer.leak_judge
    return {"kind": "review", "profile": reviewer.profile, "max_infra_retries": 0, "spec": spec}


def _review_timeout(request_timeout_s: float, *, parents: int) -> float:
    """How long a review may take: the analyses, the point review and the candidate review in turn.

    The analyses run ``_REVIEW_REQUESTS`` at a time, and the transport retries a timed-out request once.
    """
    stages = math.ceil(parents / _REVIEW_REQUESTS) + 2
    return 2 * stages * request_timeout_s


def review_parent(row: dict[str, Any], *, hidden: dict[str, Any]) -> dict[str, Any]:
    """What a review reads of one graded attempt row of ``summary.json`` (``ReviewSpec.parents``).

    ``hidden`` holds the facts about the task's hidden tests that only the leak checks read.
    """
    grade = {"resolved": row["resolved"], "reason": row["reason"], **hidden}
    for name in ("log", "model_patch"):
        if row.get(name):
            grade[name] = row[name]
    parent = {"name": row["name"], "job_id": row["job_id"], "attempt_id": row["attempt_id"], "grade": grade}
    return {**parent, "hint": row["hint"]} if row.get("hint") else parent


def grade_files(result: dict[str, Any]) -> dict[str, str | None]:
    """The verifier's output of a harbor grade result, as the analyst reads it: ``log`` and ``model_patch``.

    Each is a path on the worker host, or None when the grade left no such file.
    """
    logs = (result.get("verifier") or {}).get("logs")
    if logs is None:
        return {"log": None, "model_patch": None}
    log = Path(logs) / "test-stdout.txt"
    artifacts = Path(logs).parent / "artifacts"
    patches = sorted(artifacts.rglob("model.patch")) if artifacts.is_dir() else []
    return {"log": str(log) if log.is_file() else None, "model_patch": str(patches[0]) if patches else None}


def review_continuation(point: dict[str, Any], *, guidance: str) -> dict[str, Any] | None:
    """What a branch from a point of a review result is given, or None when the review wrote no turn there."""
    if guidance == "assistant-turn":
        candidates = point["candidates"]
        return (
            {"guidance": "assistant-turn", "assistant_turn": candidates[0]["assistant_turn"]}
            if candidates
            else None
        )
    if guidance == "user-hint":
        return {"guidance": "user-hint", "hint": point["hint"]}
    return {"guidance": "none"}


def recovery_point(points: list[dict[str, Any]], *, step: int) -> dict[str, Any] | None:
    """The available recovery point of a rollout at tool step ``step``, or None."""
    return next(
        (point for point in points if point.get("available") is True and point["tool_depth"] == step), None
    )


def run_sample(name: str, *, job: dict[str, Any], grade: dict[str, Any]) -> dict[str, Any]:
    """The sample ``name`` of a group request: a run of the JobSpec ``job``, graded by ``grade``."""
    return {"sample_slot_id": name, "run": job, "grade": grade}


def branch_sample(
    name: str,
    *,
    job_id: str,
    point_id: str,
    continuation: dict[str, Any],
    deadline_at: float,
    grade: dict[str, Any],
) -> dict[str, Any]:
    """The sample ``name`` of a group request: a branch of the job ``job_id`` from its point ``point_id``.

    The branch is given ``continuation``, has its own ``deadline_at`` and is graded by the template ``grade``.
    """
    branch = {
        "job_id": job_id,
        "point_id": point_id,
        "continuation": continuation,
        "budget": {"deadline_at": deadline_at},
    }
    return {"sample_slot_id": name, "branch": branch, "grade": grade}


def review_sample(name: str, *, job: dict[str, Any]) -> dict[str, Any]:
    """The sample ``name`` of a group request: the review JobSpec ``job``."""
    return {"sample_slot_id": name, "review": job}


def group_request(
    group_id: str, *, run_id: str, context: dict[str, Any], samples: list[dict[str, Any]]
) -> dict[str, Any]:
    """A group request of one run: ``run_id`` is what every group of the run shares."""
    return {"rollout_job_id": group_id, "prompt_group_id": run_id, "context": context, "samples": samples}

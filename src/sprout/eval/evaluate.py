"""Evaluate one Harbor task with a method: a parent rollout, then the method's branches.

    python -m sprout.eval.evaluate --task-dir <task> --dataset-digest sha256:... \\
        --preparation-manifest prep.json --output runs/<task> --method sprout \\
        --model <model> --model-endpoint <url> --model-key-env KEY [--parent-output <baseline dir>]

Each attempt is a rollout job and a grade job (``jobs``) in the Run Store at
``--runstore-url`` (ADR-0002), which the search coordinator submits and follows, so a
worker must serve the profiles. A branch is a Run Store branch job from one of an
attempt's recovery points. Methods differ only in how branch points are chosen:

    baseline   the parent rollout alone
    sprout     a review job for each of --rounds rounds picks the points (worker.jobs.review)
    bpo        first-token entropy of the policy at each checkpoint (policies.py)
    shepherd   a meta-agent picks one restart point, and each sibling samples from it

The output holds ``summary.json``, a ``groups/`` document for each group of jobs, the
plans, and in ``earlier-runs/`` the files of each run that a lost host ended.
docs/EVALUATION.md describes the files and the options.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

from sprout.contract.harbor_task import HarborTask, load_task
from sprout.contract.storage import volatile_reason
from sprout.eval import policies
from sprout.eval.groups import DEFAULT_RUNSTORE_URL, Groups
from sprout.eval.jobs import (
    Actor,
    Reviewer,
    add_image_flags,
    branch_deadline,
    branch_sample,
    check_environment,
    grade_files,
    grade_template,
    group_request,
    image_options,
    mini_job,
    recovery_point,
    review_continuation,
    review_job,
    review_parent,
    review_sample,
    run_sample,
)
from sprout.search.client import RunStoreClient
from sprout.search.group import Submitter
from sprout.search.rerun import RUNS_AGAIN, again_key, lost
from sprout.worker.agents.mini_swe.history import available_branch_points
from sprout.worker.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S
from sprout.worker.gateway.auth import bearer_key
from sprout.worker.jobs.review.branching import (
    BRANCH_COUNT_MODES,
    BRANCH_GUIDANCE_MODES,
    DEFAULT_GUIDANCE,
    MAX_ROLLOUTS,
    require_mini_parent,
)
from sprout.worker.jobs.review.leak import judge as leak_judge
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS_PATH, load_prompts
from sprout.worker.jobs.review.request_size import DEFAULT_PROMPT_CHARS

METHODS = ("baseline", "sprout", "bpo", "shepherd")
#: The directory of an output that keeps the files of each earlier run, in a directory named by its key.
_EARLIER_RUNS = "earlier-runs"


class DataLost(RuntimeError):
    """A job of the evaluation failed because a host that held the evaluation's data was lost."""


def parent_journal(output: Path) -> Path:
    """The parent attempt's journal, as the summary of the evaluation in ``output`` names it.

    Raises ValueError when the summary records no parent or its journal is gone.
    """
    summary = json.loads((Path(output) / "summary.json").read_text())
    attempts = summary.get("attempts") or []
    if not attempts or not attempts[0].get("journal"):
        raise ValueError(f"No parent attempt recorded under {output}")
    journal = Path(attempts[0]["journal"])
    if not journal.is_file():
        raise ValueError(f"The parent's journal {journal} is gone")
    return journal


def recorded_parent(output: Path, *, task: str, dataset_digest: str, runstore_url: str) -> dict[str, Any]:
    """The parent attempt row of a completed evaluation in ``output``, for a method that branches from it.

    Raises ValueError unless that evaluation ran this task of this dataset through this Run Store.
    """
    summary = json.loads((Path(output) / "summary.json").read_text())
    if (
        summary.get("task") != task
        or summary.get("dataset_digest") != dataset_digest
        or not summary.get("complete")
    ):
        raise ValueError("Recorded parent metadata differs from this task")
    rows = summary.get("attempts") or []
    if not rows or not rows[0].get("job_id"):
        raise ValueError(
            "The recorded parent ran as a local Harbor trial; it has no Run Store job to branch from"
        )
    if (summary.get("runstore") or {}).get("url") != runstore_url:
        raise ValueError("The recorded parent's jobs are in another Run Store")
    parent_journal(output)
    return dict(rows[0])


def _recorded_key(output: Path) -> str:
    """The affinity key of the evaluation in ``output``. Raises ValueError when its summary names none."""
    key = json.loads((output / "summary.json").read_text()).get("affinity")
    if not isinstance(key, str):
        raise ValueError("The recorded parent's summary names no affinity key")
    return key


def attempt_row(
    sample: dict[str, Any], *, name: str, round_no: int, hint: str | None, validated_at: float
) -> dict[str, Any]:
    """The ``summary.json`` row of a settled sample: its rollout and its grade.

    Raises RuntimeError unless the rollout succeeded and the grade gave a reward.
    """
    actor, grade = sample["actor"], sample["grade"]
    if actor["state"] != "succeeded":
        raise RuntimeError(
            f"{name}: the rollout job {actor['job_id']} {actor['state']}: {actor.get('error')}"
        )
    verdict: dict[str, Any] = (grade.get("result") if grade["state"] == "succeeded" else None) or {}
    reward = verdict.get("reward")
    # isinstance accepts a JSON true as an int, but true is not a reward.
    if isinstance(reward, bool) or not isinstance(reward, (int, float)):
        raise RuntimeError(f"{name}: the grade job {grade['job_id']} {grade['state']}: {grade.get('error')}")
    output = actor["result"]
    started = actor.get("started_at")
    row = {
        "name": name,
        "round": round_no,
        # A run that hit its turn or time limit is truncated and graded. Its own status says which.
        "status": output.get("agent_status") or output.get("status"),
        "stop_reason": output.get("stop_reason"),
        "resolved": reward == 1,
        "reward": float(reward),
        "reason": verdict.get("reason"),
        "journal": output.get("journal_path"),
        "job_id": actor["job_id"],
        "attempt_id": actor["attempt_id"],
        "grade_job_id": grade["job_id"],
        **grade_files(verdict),
        "started_at": started,
        "validated_at": validated_at,
        "seconds": validated_at - started if started is not None else None,
    }
    return {**row, "hint": hint} if hint else row


def attempt_directory(journal: str, *, job_id: str, attempt_id: str) -> Path:
    """The directory of a job's attempt on the worker host, under the artifact root that holds ``journal``."""
    return Path(journal).parents[2] / job_id / attempt_id


def hidden_facts(task_dir: Path, *, patch_path: Path | None) -> dict[str, Any]:
    """What the leak checks of a review read about the task's hidden tests (``leak_judge.task_hidden_tests``).

    With ``patch_path`` the hidden tests are also written there, for the Jev judge.
    """
    patch, fail_to_pass = leak_judge.task_hidden_tests(task_dir)
    facts: dict[str, Any] = {}
    if fail_to_pass:
        facts["fail_to_pass"] = list(fail_to_pass)
    paths = leak_judge.patch_files(patch)
    if paths:
        facts["test_patch_paths"] = paths
    if patch.strip() and patch_path is not None:
        patch_path.write_text(patch)
        facts["test_patch"] = str(patch_path)
    return facts


def round_plan(
    review: dict[str, Any], *, guidance: str, planned: list[dict[str, Any]], skipped: list[dict[str, Any]]
) -> dict[str, Any]:
    """``plan-round<N>.json``: the round's point review as the review job recorded it, and its branches.

    ``review`` is the settled review stage, ``planned`` the branch samples, and ``skipped`` the points
    that made no branch. Each selected branch carries what the round gives it.
    """
    result = review.get("result") or {}
    points = {point["run_name"]: point for point in result.get("points") or []}
    plan: dict[str, Any] = {}
    if review.get("directory") is not None:
        path = Path(review["directory"]) / "plan-points.json"
        plan = json.loads(path.read_text()) if path.is_file() else {}
    selected = []
    for branch in plan.get("selected_branches") or []:
        continuation = review_continuation(points[branch["run_name"]], guidance=guidance) or {}
        selected.append({**branch, "branch_guidance": guidance, **_given(continuation)})
    return {
        **plan,
        "branch_guidance": guidance,
        "selected_branches": selected,
        "review_job": {key: review.get(key) for key in ("job_id", "attempt_id", "directory")},
        "branches": [{"name": sample["sample_slot_id"], **sample["branch"]} for sample in planned],
        "skipped": skipped,
    }


def selection_plan(
    *, task: str, method: str, journal: Path, budget: int, eligible: list[int], selected: list[dict[str, Any]]
) -> dict[str, Any]:
    """``plan.json`` of bpo and shepherd: the points the selector chose from the parent, and why."""
    return {
        "task": task,
        "method": method,
        "shared_parent": str(journal),
        "max_total_rollouts": budget,
        "budget_includes_shared_initial": True,
        "eligible_steps": eligible,
        "selected": selected,
        "branch_guidance": "none",
        "hint_delivery": "fixed-neutral",
    }


def _keep_files(output: Path, *, directory: Path) -> None:
    """Move each entry of ``output`` but the summary and the earlier runs into ``directory``, a new one."""
    directory.mkdir(parents=True)
    for path in sorted(output.iterdir()):
        if path.name not in {"summary.json", _EARLIER_RUNS}:
            path.rename(directory / path.name)


def _given(continuation: dict[str, Any]) -> dict[str, Any]:
    return {key: continuation[key] for key in ("assistant_turn", "hint") if key in continuation}


def _source_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("The source worktree must be clean before a real evaluation")
    return head


def _verify_prepared_task(*, manifest: Path, dataset_digest: str, task: HarborTask, task_hash: str) -> None:
    preparation = json.loads(manifest.read_text())
    if preparation.get("dataset_digest") != dataset_digest:
        raise ValueError("Prepared dataset digest differs from the selected cohort")
    rows = [
        row
        for row in preparation.get("tasks", [])
        if Path(row.get("path", "")).resolve() == task.task_dir and row.get("name") == task.name
    ]
    if len(rows) != 1 or rows[0].get("task_toml_sha256") != task_hash:
        raise ValueError("Task is absent from the pinned preparation or its definition changed")


def _task_hash(task: HarborTask) -> str:
    return hashlib.sha256((task.task_dir / "task.toml").read_bytes()).hexdigest()


def initial_state(args: argparse.Namespace, task: HarborTask) -> dict[str, Any]:
    """The summary of an evaluation before its first attempt: what it ran, and with which settings."""
    sprout = args.method == "sprout"
    state = {
        "evaluation_id": uuid.uuid4().hex,
        "source_commit": None,
        # The key that each job of the current run carries, set when the evaluation starts.
        "affinity": None,
        "task": task.name,
        "task_toml_sha256": _task_hash(task),
        "dataset_digest": args.dataset_digest,
        "model": args.model,
        "method": args.method,
        "runstore": {
            "url": args.runstore_url,
            "profiles": {"rollout": args.profile, "grade": args.grade_profile, "review": args.review_profile},
        },
        "parent_output": str(args.parent_output) if args.parent_output else None,
        "branch_guidance": args.branch_guidance if sprout else "none",
        "branch_count_mode": args.branch_count_mode if sprout else None,
        "structured_review": bool(args.structured_review) if sprout else False,
        # Which prompts file the reviews and the Shepherd selector read (parse_args loaded it).
        "prompts": {"path": str(args.prompts.path), "sha256": args.prompts.sha256},
        "branch_caps": args.branches,
        "rounds": args.rounds if sprout else 1,
        "command_timeout": args.command_timeout,
        "image_env": not args.no_image_env,
        "image_user": not args.no_image_user,
        "max_rollouts": args.max_rollouts,
        "attempts": [],
        "earlier_runs": [],
        "complete": False,
    }
    if sprout:
        # Where the analyst and reviewer ran (never the key: only the endpoint's URL).
        state["review"] = {
            "model": args.review_model,
            "endpoint": args.review_endpoint,
            "effort": args.review_effort,
        }
        if args.leak_judge:
            state["review"]["leak_judge"] = _leak_judge(args)
    return state


def _leak_judge(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "threshold": args.leak_judge_threshold,
        "model": args.leak_judge_model,
        "resamples": args.leak_judge_resamples,
    }


def _actor(args: argparse.Namespace) -> Actor:
    return Actor(
        slot=getattr(args, "slot", "mini-swe-agent"),
        tool_protocol=getattr(args, "opencollab_tool_protocol", "single2"),
        model=args.model,
        model_endpoint=args.model_endpoint,
        api_key_env=args.model_key_env,
        max_output_tokens=args.max_output_tokens,
        max_turns=args.max_turns,
        command_timeout_s=args.command_timeout,
        reasoning_effort=args.reasoning_effort,
    )


def _reviewer(args: argparse.Namespace) -> Reviewer:
    return Reviewer(
        profile=args.review_profile,
        model=args.review_model,
        model_endpoint=args.review_endpoint,
        api_key_env=args.review_key_env,
        guidance=args.branch_guidance,
        count_mode=args.branch_count_mode,
        max_output_tokens=args.review_output_tokens,
        prompt_chars=args.prompt_budget_chars,
        transcript_tokens=args.analyst_tokens,
        max_attempts=args.reviewer_max_attempts,
        request_timeout_s=args.review_timeout,
        structured=bool(args.structured_review),
        reasoning_effort=args.review_effort,
        # The worker reads this file, whatever its own checkout packages: the sha256 pins it.
        prompts={"path": str(args.prompts.path), "sha256": args.prompts.sha256},
        leak_judge={**_leak_judge(args), "api_key_env": args.leak_judge_key_env} if args.leak_judge else None,
    )


class Evaluation:
    """One task evaluated with one method: its groups of jobs, its attempts and its summary.

    ``run`` blocks until the method is done, and ``pause`` runs between two looks at the Run Store. The
    summary records ``source_commit``, the commit of the code that built the jobs, and ``affinity``, their
    key. After a lost host, an evaluation that ran its own parent runs again, at most ``RUNS_AGAIN`` times."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        task: HarborTask,
        client: RunStoreClient,
        pause: Callable[[], None],
        source_commit: str,
    ) -> None:
        self._args = args
        self._task = task
        self._client = client
        self._source_commit = source_commit
        self._output = Path(args.output)
        self._attempts: list[dict[str, Any]] = []
        self._init_state()
        self._init_groups(pause=pause)
        self._init_jobs()

    def _init_state(self) -> None:
        self._state = initial_state(self._args, self._task)

    def _init_groups(self, *, pause: Callable[[], None]) -> None:
        run_id = self._state["evaluation_id"]
        self._groups = Groups(
            client=self._client,
            directory=self._output / "groups",
            submitter=Submitter(context_key="eval", key_prefix=f"eval/{run_id}/"),
            pause=pause,
        )

    def _init_jobs(self) -> None:
        self._actor = _actor(self._args)
        self._image = image_options(self._args)
        self._grade = grade_template(self._task, profile=self._args.grade_profile, image=self._image)
        self._context = {"task": self._task.name, "method": self._args.method}

    def run(self) -> dict[str, Any]:
        """The summary of the finished evaluation, also in ``summary.json`` with any error after the start.

        Raises FileExistsError for an output that is not empty, ValueError for a task or a parent that the
        evaluation refuses, and RuntimeError when a job fails: ``DataLost`` when it cannot run again.
        """
        self._start()
        try:
            self._search()
            if _task_hash(self._task) != self._state["task_toml_sha256"]:
                raise RuntimeError("Official task definition changed during evaluation")
        except Exception as error:
            self._state["error"] = f"{type(error).__name__}: {error}"
            if isinstance(error, DataLost):
                self._state["reason"] = "data_lost"
            self._save()
            raise
        self._state["complete"] = True
        self._state.setdefault("branch_schedule_complete", True)
        self._save()
        return self._state

    def _start(self) -> None:
        args = self._args
        if self._output.exists() and any(self._output.iterdir()):
            raise FileExistsError("Use a new output directory for each one-task evaluation")
        check_environment(self._task)
        _verify_prepared_task(
            manifest=Path(args.preparation_manifest),
            dataset_digest=args.dataset_digest,
            task=self._task,
            task_hash=self._state["task_toml_sha256"],
        )
        self._state["source_commit"] = self._source_commit
        # A branch carries the key of its parent, so a method from a recorded parent keeps that key.
        self._state["affinity"] = (
            _recorded_key(Path(args.parent_output)) if args.parent_output else "eval-" + uuid.uuid4().hex
        )
        self._output.mkdir(parents=True, exist_ok=True)
        self._save()

    def _search(self) -> None:
        """Run the method, and run it again from a new parent after each lost host, while it can."""
        while True:
            try:
                self._method()
                return
            except DataLost as error:
                self._run_again(error)

    def _method(self) -> None:
        """The current run: the parent, then the attempts that the method makes from it."""
        self._attempts.append(self._parent())
        self._save()
        if self._args.method != "baseline":
            require_mini_parent(Path(self._attempts[0]["journal"]))
        if self._args.method == "sprout":
            self._sprout()
        elif self._args.method in ("bpo", "shepherd"):
            self._sample()

    def _run_again(self, error: DataLost) -> None:
        """Keep the run that ``error`` ended in ``earlier_runs``, and start the next one under a new key.

        Raises DataLost for a method from a recorded parent, and after ``RUNS_AGAIN`` runs again."""
        earlier = self._state["earlier_runs"]
        if self._args.parent_output:
            raise DataLost(
                f"{error}. The evaluation branches from a recorded parent, so it does not run again."
            )
        if len(earlier) == RUNS_AGAIN:
            raise DataLost(f"{error}. A task evaluation runs again at most {RUNS_AGAIN} times.")
        key = self._state["affinity"]
        _keep_files(self._output, directory=self._output / _EARLIER_RUNS / key)
        earlier.append({"affinity": key, "attempts": self._attempts, "error": str(error)})
        self._state["affinity"] = again_key(earlier[0]["affinity"], runs=len(earlier))
        self._attempts = []
        self._save()

    def _parent(self) -> dict[str, Any]:
        args = self._args
        if args.parent_output:
            return recorded_parent(
                Path(args.parent_output),
                task=self._task.name,
                dataset_digest=args.dataset_digest,
                runstore_url=args.runstore_url,
            )
        job = mini_job(
            self._task,
            profile=args.profile,
            actor=self._actor,
            checkpoint=args.checkpoint_mode == "disk_only",
            image=self._image,
            now=time.time(),
        )
        (sample,) = self._settle("parent", [run_sample("parent", job=job, grade=self._grade)])
        return attempt_row(sample, name="parent", round_no=0, hint=None, validated_at=time.time())

    def _sprout(self) -> None:
        """Each round: a review job over every attempt so far, then a branch at each point it picks."""
        args = self._args
        hidden = hidden_facts(
            self._task.task_dir, patch_path=self._output / "hidden-tests.patch" if args.leak_judge else None
        )
        for round_no in range(1, args.rounds + 1):
            if any(row["resolved"] for row in self._attempts):
                break
            width = args.branches[min(round_no - 1, len(args.branches) - 1)]
            review = self._review(round_no=round_no, width=width, hidden=hidden)
            planned, skipped = self._branches(review.get("result") or {})
            plan = round_plan(review, guidance=args.branch_guidance, planned=planned, skipped=skipped)
            policies.write_json(self._output / f"plan-round{round_no}.json", plan)
            if not planned:
                self._state["stop_reason"] = plan.get("validation_error") or "the review selected no branches"
                self._state["branch_schedule_complete"] = False
                break
            samples = self._settle(f"round-{round_no}", planned)
            hints = {branch["name"]: branch["continuation"].get("hint") for branch in plan["branches"]}
            for sample in samples:
                name = sample["sample_slot_id"]
                self._attempts.append(
                    attempt_row(
                        sample, name=name, round_no=round_no, hint=hints[name], validated_at=time.time()
                    )
                )
            self._save()

    def _review(self, *, round_no: int, width: int, hidden: dict[str, Any]) -> dict[str, Any]:
        """The settled review stage of a round, with ``directory``: where its attempt ran, on the worker."""
        job = review_job(
            _reviewer(self._args),
            problem=self._task.instruction,
            parents=[review_parent(row, hidden=hidden) for row in self._attempts],
            points=width,
            round_no=round_no,
        )
        # The reviews of all rounds carry one key, so each needs a sample id of its own (Submitter.key).
        sample_id = f"review-{round_no}"
        (sample,) = self._settle(sample_id, [review_sample(sample_id, job=job)])
        stage = sample["actor"]
        if stage["state"] != "succeeded":
            raise RuntimeError(
                f"Round {round_no}'s review job {stage['job_id']} {stage['state']}: {stage.get('error')}"
            )
        directory = attempt_directory(
            self._attempts[0]["journal"], job_id=stage["job_id"], attempt_id=stage["attempt_id"]
        )
        return {**stage, "directory": str(directory)}

    def _branches(self, result: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """The branch samples of a round's review result, and the points that make none."""
        planned: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        for point in result.get("points") or []:
            continuation = review_continuation(point, guidance=self._args.branch_guidance)
            recovery = recovery_point(
                self._client.points(point["job_id"], attempt_id=point["attempt_id"]), step=point["step"]
            )
            if continuation is None or recovery is None:
                reason = (
                    "the review wrote no turn there"
                    if continuation is None
                    else "no available recovery point"
                )
                skipped.append({"run_name": point["run_name"], "step": point["step"], "reason": reason})
                continue
            planned.append(
                self._branch(
                    point["run_name"],
                    job_id=point["job_id"],
                    point_id=recovery["id"],
                    continuation=continuation,
                )
            )
        return planned, skipped

    def _branch(
        self, name: str, *, job_id: str, point_id: str, continuation: dict[str, Any]
    ) -> dict[str, Any]:
        return branch_sample(
            name,
            job_id=job_id,
            point_id=point_id,
            continuation=continuation,
            deadline_at=branch_deadline(self._task, now=time.time()),
            grade=self._grade,
        )

    def _sample(self) -> None:
        """bpo and shepherd: point-only branches from the parent, one after another until one resolves."""
        args = self._args
        parent = self._attempts[0]
        journal = Path(parent["journal"])
        points = available_branch_points(journal)
        if not points:
            raise RuntimeError("Shared parent exposes no exact mini checkpoint/session pairs")
        client = policies.ModelClient(args.model_endpoint, bearer_key(args.model_key_env))
        selected = policies.select_points(
            args, self._task.instruction, journal, points, client, prompts=args.prompts
        )
        policies.write_json(
            self._output / "plan.json",
            selection_plan(
                task=self._task.name,
                method=args.method,
                journal=journal,
                budget=args.max_rollouts,
                eligible=sorted(points),
                selected=selected,
            ),
        )
        recovery = self._client.points(parent["job_id"], attempt_id=parent["attempt_id"])
        for index, selection in enumerate(selected, 1):
            name = f"b{index:02d}-step{selection['step']}"
            point = recovery_point(recovery, step=selection["step"])
            if point is None:
                raise RuntimeError(f"The parent has no available recovery point at step {selection['step']}")
            branch = self._branch(
                name, job_id=parent["job_id"], point_id=point["id"], continuation={"guidance": "none"}
            )
            (sample,) = self._settle(name, [branch])
            self._attempts.append(
                attempt_row(sample, name=name, round_no=1, hint=None, validated_at=time.time())
            )
            self._save()
            if self._attempts[-1]["resolved"]:
                break

    def _settle(self, group_id: str, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """The settled samples of the group ``group_id``. Raises DataLost when a job of it lost its host."""
        request = group_request(
            group_id, run_id=self._state["evaluation_id"], context=self._context, samples=samples
        )
        key = self._state["affinity"]
        document = self._groups.settle(request, affinity=key)
        if lost(document):
            raise DataLost(f"{group_id}: a host that held the jobs of the key {key} was lost")
        settled: list[dict[str, Any]] = document["samples"]
        return settled

    def _save(self) -> None:
        self._state["attempts"] = self._attempts
        self._state["resolved"] = any(row["resolved"] for row in self._attempts)
        policies.write_json(self._output / "summary.json", self._state)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The options of an evaluation, from ``argv``, with ``prompts`` loaded and the review's defaults set.

    Exits with status 2 for a combination of options that no evaluation can run.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    _add_task_options(parser)
    _add_runstore_options(parser)
    _add_actor_options(parser)
    _add_review_options(parser)
    _add_selector_options(parser)
    args = parser.parse_args(argv)
    _check(parser, args)
    # The reviewer's settings fall back to the actor's, one by one.
    args.review_model = args.review_model or args.model
    args.review_endpoint = args.review_endpoint or args.model_endpoint
    args.review_key_env = args.review_key_env or args.model_key_env
    args.review_effort = args.review_effort or args.reasoning_effort
    return args


def _add_task_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--dataset-digest", required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="sprout")
    parser.add_argument(
        "--parent-output",
        type=Path,
        help="branch from the parent of a completed baseline evaluation through the same Run Store",
    )
    parser.add_argument(
        "--volatile-ok",
        action="store_true",
        help="allow --output under /tmp and friends; refused by default because the summary and the "
        "group documents there are the evaluation's only record of its jobs",
    )


def _add_runstore_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--runstore-url", default=DEFAULT_RUNSTORE_URL, help="the Run Store's HTTP API")
    parser.add_argument(
        "--runstore-token-env",
        default="SPROUT_RUNSTORE_TOKEN",
        help="environment variable holding the Run Store's bearer token",
    )
    parser.add_argument("--profile", default="mini", help="the worker profile of every rollout")
    parser.add_argument("--grade-profile", default="grade", help="the worker profile of every grade")
    parser.add_argument(
        "--review-profile", default="review", help="sprout: the worker profile of every review"
    )
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="seconds between two looks at the Run Store's jobs"
    )


def _add_actor_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--slot", choices=("mini-swe-agent", "opencollab-base"), default="mini-swe-agent")
    parser.add_argument("--opencollab-tool-protocol", choices=("single2", "review50-v1"), default="single2")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument(
        "--model-key-env",
        required=True,
        help="environment variable holding the bearer key for --model-endpoint: the worker's for every "
        "rollout, and also this process's for the bpo and shepherd selectors",
    )
    parser.add_argument(
        "--checkpoint-mode",
        choices=["disk_only", "none"],
        default="disk_only",
        help="per-turn disk snapshots (branch points) or none; every rollout ends with one disk snapshot",
    )
    parser.add_argument("--max-output-tokens", type=int, default=64000)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument(
        "--command-timeout",
        type=int,
        default=DEFAULT_COMMAND_TIMEOUT_S,
        help="seconds the runtime gives each agent command before killing it (the model's "
        "bash tool has no timeout argument, so every command runs under this one limit; "
        "a branch runs under its parent's)",
    )
    parser.add_argument("--reasoning-effort")
    add_image_flags(parser)


def _add_review_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--branch-guidance", choices=BRANCH_GUIDANCE_MODES, default=DEFAULT_GUIDANCE)
    parser.add_argument("--branch-count-mode", choices=BRANCH_COUNT_MODES, default="adaptive")
    parser.add_argument(
        "--prompts",
        type=Path,
        default=DEFAULT_PROMPTS_PATH,
        help="YAML with every analyst/reviewer/shepherd prompt (default: the packaged file); the review "
        "jobs read it on the worker host, and its sha256 is recorded",
    )
    parser.add_argument(
        "--structured-review",
        action="store_true",
        help="ask the analyst and reviewer for schema-constrained JSON (response_format "
        "json_schema) instead of fenced plain-text plans; needs an endpoint that "
        "supports json_schema, e.g. OpenAI or a LiteLLM proxy",
    )
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument(
        "--branches",
        type=int,
        nargs="+",
        default=[4, 3],
        help="branch count per round: a maximum in adaptive mode, exact in fixed mode",
    )
    parser.add_argument("--review-output-tokens", type=int, default=64000)
    # The analyst and reviewer may run on another model, or behind another endpoint,
    # while the actor that continues each branch keeps --model; each option defaults to the actor's.
    parser.add_argument(
        "--review-model", help="the analyst and reviewer model (default --model), served by --review-endpoint"
    )
    parser.add_argument(
        "--review-endpoint",
        help="the Chat Completions endpoint the analyst and reviewer use (default "
        "--model-endpoint), e.g. a LiteLLM proxy alias for a model the actor's "
        "endpoint does not serve",
    )
    parser.add_argument(
        "--review-key-env",
        help="the worker's environment variable holding the bearer key for --review-endpoint "
        "(default --model-key-env)",
    )
    parser.add_argument(
        "--review-effort",
        help="reasoning_effort of the analyst and reviewer requests (default --reasoning-effort)",
    )
    parser.add_argument("--reviewer-max-attempts", type=int, default=3)
    parser.add_argument(
        "--leak-judge",
        action="store_true",
        help="sprout, assistant-turn: check every turn the reviewer writes for hidden-test content "
        "(the task's tests, the verifier output) with worker.jobs.review.leak.terms's leak_terms and Jev "
        "before it is taken; a flagged turn goes back to the reviewer",
    )
    parser.add_argument(
        "--leak-judge-resamples",
        type=int,
        default=2,
        help="how often a flagged turn goes back before it is taken anyway, marked (default 2)",
    )
    parser.add_argument("--leak-judge-threshold", type=float, default=leak_judge.THRESHOLD)
    parser.add_argument("--leak-judge-model", default=leak_judge.MODEL)
    parser.add_argument(
        "--leak-judge-key-env",
        default=leak_judge.KEY_ENV,
        help="the worker's environment variable holding the TypeSafe key",
    )
    parser.add_argument("--analyst-tokens", type=int, default=100000)
    parser.add_argument(
        "--review-timeout",
        type=int,
        default=3600,
        help="seconds one analyst or reviewer request may take; a request that times out "
        "is retried once with half the output budget (default 3600: 64k output "
        "tokens take about 30 minutes at the rates seen)",
    )
    parser.add_argument(
        "--prompt-budget-chars",
        type=int,
        default=DEFAULT_PROMPT_CHARS,
        help="size of every analyst and reviewer request, in characters (about a "
        "quarter as many tokens); a reviewer request over it is reduced in a fixed "
        "order, recorded in plan-roundN.json, and one the endpoint refuses (HTTP "
        "400) is retried with the budget halved, twice at most",
    )


def _add_selector_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--max-rollouts", type=int, default=8, help="bpo/shepherd: total rollouts including the shared parent"
    )
    parser.add_argument("--bpo-top-k", type=int, default=5)
    parser.add_argument("--bpo-min-spacing", type=int, default=64)
    parser.add_argument("--bpo-max-points", type=int, default=7)
    parser.add_argument("--bpo-scoring-workers", type=int, default=2)
    parser.add_argument("--meta-max-tokens", type=int, default=64000)
    parser.add_argument("--meta-transcript-tokens", type=int, default=60000)


def _check(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    """Refuse a combination of options that no evaluation can run, before any job exists."""
    reason = volatile_reason(args.output)
    if reason and not args.volatile_ok:
        parser.error("refusing: %s (pass --volatile-ok to override)" % reason)
    if (
        args.rounds < 0
        or not args.branches
        or any(n < 1 for n in args.branches)
        or args.reviewer_max_attempts < 1
        or not 1 <= args.max_rollouts <= MAX_ROLLOUTS
        or not 1 <= args.bpo_top_k <= 20
    ):
        parser.error("Invalid branch schedule, rollout budget or reviewer attempts")
    # bpo and shepherd keep the protocol's 8. A larger budget is for a sprout round-scaling schedule.
    if args.method in ("bpo", "shepherd") and args.max_rollouts > 8:
        parser.error("bpo and shepherd allow at most 8 rollouts, the shared parent included")
    if args.prompt_budget_chars < 50_000:
        parser.error("--prompt-budget-chars below 50000 leaves no room for the task and a transcript")
    try:
        # Loaded once, here: a bad file fails before any job exists.
        args.prompts = load_prompts(args.prompts)
    except (OSError, ValueError) as error:
        parser.error("--prompts: %s" % error)
    for flag, variable in (
        ("--model-key-env", args.model_key_env),
        ("--review-key-env", args.review_key_env),
        ("--runstore-token-env", args.runstore_token_env),
        ("--leak-judge-key-env", args.leak_judge_key_env),
    ):
        if variable is not None and not variable.isidentifier():
            parser.error(f"{flag} must name one environment variable")
    if args.method in ("bpo", "shepherd") and not args.parent_output:
        parser.error("bpo and shepherd branch from a recorded baseline: pass --parent-output")
    if args.command_timeout <= 0 or args.review_timeout <= 0 or args.poll_interval <= 0:
        parser.error("--command-timeout, --review-timeout and --poll-interval must be positive")
    if args.leak_judge:
        if args.method != "sprout" or args.branch_guidance != "assistant-turn":
            parser.error(
                "--leak-judge checks reviewer-written turns: --method sprout with assistant-turn guidance"
            )
        if args.leak_judge_resamples < 0 or not 0 < args.leak_judge_threshold <= 1:
            parser.error("--leak-judge-resamples must be >= 0 and --leak-judge-threshold in (0, 1]")


def main(argv: Sequence[str] | None = None) -> int:
    """Run the evaluation that the command line ``argv`` asks for, print its summary and return the status.

    The status is 0 for a complete evaluation. Raises ValueError for a missing Run Store token or model key.
    """
    args = parse_args(argv)
    token = os.environ.get(args.runstore_token_env, "").strip()
    if not token:
        raise ValueError(f"${args.runstore_token_env} holds no Run Store token")
    if args.method in ("bpo", "shepherd"):
        bearer_key(args.model_key_env)  # the selector calls the model from here: an unset key fails now
    task = load_task(Path(args.task_dir))
    client = RunStoreClient(args.runstore_url, token)
    try:
        evaluation = Evaluation(
            args=args,
            task=task,
            client=client,
            pause=lambda: time.sleep(args.poll_interval),
            source_commit=_source_commit(),
        )
        state = evaluation.run()
    finally:
        client.close()
    print(json.dumps({key: value for key, value in state.items() if key != "attempts"}, indent=2))
    return 0 if state["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

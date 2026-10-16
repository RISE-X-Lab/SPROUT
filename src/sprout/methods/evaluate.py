"""Evaluate one Harbor task with mini-swe-agent: a parent rollout, then a method.

    python -m sprout.methods.evaluate --task-dir <task> --dataset-digest sha256:... \\
        --preparation-manifest prep.json --output runs/<task> --method sprout \\
        --model <model> --model-endpoint <url> --model-key-env KEY \\
        --runtime-bin runtime/sprout-runtime [--parent-output <baseline dir>]

Every attempt is a Harbor ``Trial``: Harbor loads the task, starts the sandbox
(``harbor_plugin``: a rootless podman container on a btrfs store, or with
``--backend microvm`` an AgentENV microVM), runs mini, collects the task's
artifacts and grades with the task's own verifier in a separate sandbox. Sprout
adds what Harbor has no concept of -- a disk snapshot after every model turn and
at the end, and a branch: a new trial whose environment is a parent's snapshot
and whose agent resumes the parent's history at that exact point. A branch runs
on the backend (and, for podman, the store) that holds its parent's snapshots.

``--final-memory-snapshot`` is the benchmark's call on memory: a grader that
runs inside the agent's VM (TerminalBench's shared verifier) may need the
processes the task started, so ``always`` adds a full snapshot after the final
disk one and ``on-failure`` keeps it only for attempts that failed -- the ones
that get re-graded or branched from. The disk snapshot is kept either way.
Podman snapshots are disk-only, so there it must stay ``never``.

Methods differ only in how branch points are chosen:

    baseline   the parent rollout alone
    sprout     analyst -> reviewer -> branches, for --rounds rounds (branching.py)
    bpo        first-token entropy of the policy at each checkpoint (policies.py)
    shepherd   a meta-agent picks one restart point; siblings sample from it
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import subprocess
import time

from harbor.models.task.task import Task
from harbor.models.trial.config import TrialConfig
from harbor.trial.trial import Trial

from sprout.harbor_plugin.agentenv import AgentENVEnvironment
from sprout.harbor_plugin.branching import PodmanSnapshotEnvironment, SnapshotEnvironment
from sprout.harbor_plugin.podman import PodmanEnvironment
from sprout.harness.agents.mini_swe.history import (
    available_branch_points, conversation_restore, training_messages,
)
from sprout.harness.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S
from sprout.harness_core.journal import read_journal, volatile_reason
from sprout.harness_gateway.auth import bearer_key
from sprout.harness_orchestrator.reap import AgentEnvClient
from sprout.harness_sandbox.backends import default_podman_store
from sprout.harness_sandbox.templates import find_regctl
from sprout.methods import leak_judge, policies
from sprout.methods.branching import (
    BRANCH_COUNT_MODES, BRANCH_GUIDANCE_MODES, DEFAULT_GUIDANCE, DEFAULT_PROMPT_CHARS, MAX_ROLLOUTS, Attempt, BranchChoice,
    Grade, analyse, outcome_from_journal, require_mini_parent, review, selected_prefix,
)
from sprout.methods.leak import hidden_terms, visible_text
from sprout.methods.prompts import DEFAULT_PROMPTS_PATH, load_prompts
from sprout.methods.review_transport import ReviewTransport

METHODS = ("baseline", "sprout", "bpo", "shepherd")
MEMORY_SNAPSHOT_POLICIES = ("never", "always", "on-failure")
AGENTS = {"mini": "sprout.harbor_plugin.agentenv_mini:AgentENVMini",
          "branch": "sprout.harbor_plugin.branching:BranchMini"}
#: The Harbor environment class per backend: for a fresh trial, and for a branch
#: (the same class with a snapshot for an image). The agents are the same on either.
ENVIRONMENTS = {"microvm": (AgentENVEnvironment, SnapshotEnvironment),
                "podman": (PodmanEnvironment, PodmanSnapshotEnvironment)}
#: Where each dataset's tasks come from, by the task-name prefix Harbor records.
DATASET_SOURCES = (("swebench-pro/", "scaleapi/SWE-bench_Pro-os/v2"),
                   ("datacurve/", "datacurve-ai/deep-swe"))
DEFAULT_SOURCE = "terminal-bench/terminal-bench-2-1"


def dataset_source(task_name: str) -> str:
    return next((source for prefix, source in DATASET_SOURCES if task_name.startswith(prefix)),
                DEFAULT_SOURCE)


def import_path(cls: type) -> str:
    """``module:Class``, the form Harbor loads a plugin class by."""
    return f"{cls.__module__}:{cls.__qualname__}"


# --- one Harbor trial ------------------------------------------------------------

def environment_kwargs(args) -> dict:
    """What the backend's Harbor environment class is constructed with."""
    kwargs = {"runtime_bin": str(Path(args.runtime_bin).resolve()),
              "checkpoint_mode": getattr(args, "checkpoint_mode", "disk_only")}
    if args.backend == "podman":
        kwargs["store"] = str(Path(args.store).expanduser().resolve())
    else:
        kwargs.update(sandbox_ttl=args.sandbox_ttl, server_url=args.server_url,
                      api_key_file=str(Path(args.api_key_file).resolve()) if args.api_key_file else None)
    return kwargs


def trial_config(args, task: Task, *, name: str, agent: str = "mini",
                 branch_context: Path | None = None) -> TrialConfig:
    """The trial Harbor runs for one attempt. ``agent`` is mini, branch, oracle or nop."""
    env_kwargs = environment_kwargs(args)
    fresh, restored = ENVIRONMENTS[args.backend]
    if (agent == "branch") != (branch_context is not None):
        raise ValueError("a branch trial needs a branch context, and only a branch trial has one")
    if agent in AGENTS:
        agent_config = {"import_path": AGENTS[agent], "model_name": args.model, "kwargs": {
            "inference_endpoint": args.model_endpoint, "api_key_env": args.model_key_env,
            "actor_timeout_s": task.config.agent.timeout_sec,
            "reasoning_effort": getattr(args, "reasoning_effort", None),
            "max_output_tokens": args.max_output_tokens, "max_turns": args.max_turns,
            "command_timeout_s": args.command_timeout,
            "final_memory_snapshot": getattr(args, "final_memory_snapshot", "never") != "never"}}
    else:
        agent_config = {"name": agent}
    environment = import_path(fresh)
    if branch_context is not None:
        agent_config["kwargs"]["branch_context"] = str(branch_context)
        env_kwargs["branch_context"] = str(branch_context)
        environment = import_path(restored)
    return TrialConfig.model_validate({
        "task": {"path": str(Path(args.task_dir).resolve()),
                 "source": dataset_source(task.name) + "@" + args.dataset_digest},
        "trial_name": f"{task.name.split('/')[-1][:36]}-{name}",
        "trials_dir": str((Path(args.output) / "trials").resolve()),
        "agent_timeout_multiplier": 1.0,
        "agent": agent_config,
        "environment": {"import_path": environment, "delete": True, "kwargs": env_kwargs},
    })


def read_grade(directory: Path) -> Grade:
    """Harbor's result.json (plus the artifacts and verifier logs it downloaded) as a Grade."""
    path = directory / "result.json"
    if not path.is_file():
        raise RuntimeError(f"Official Harbor trial has no result: {path}")
    body = json.loads(path.read_text())
    exception = body.get("exception_info")
    actor_deadline = isinstance(exception, dict) and exception.get("exception_type") == "AgentTimeoutError"
    if not body.get("finished_at") or (exception and not actor_deadline):
        raise RuntimeError(f"Official Harbor trial failed: {exception}")
    reward = ((body.get("verifier_result") or {}).get("rewards") or {}).get("reward")
    if type(reward) not in (int, float) or not 0 <= reward <= 1:
        raise RuntimeError(f"Official Harbor reward is missing or invalid: {reward!r}")
    patches = sorted((directory / "artifacts").rglob("model.patch")) if (directory / "artifacts").is_dir() else []
    patch = patches[0].read_text(errors="replace") if patches else ""
    logs = []
    verifier_dir = directory / "verifier"
    if verifier_dir.is_dir():
        for log in sorted(p for p in verifier_dir.rglob("*") if p.is_file() and p.suffix in {".txt", ".log"}):
            logs.append("== %s ==\n%s" % (log.relative_to(verifier_dir), log.read_text(errors="replace")[-8000:]))
    detail = json.dumps(body.get("verifier_result"), ensure_ascii=False)[:4000] + "\n" + "\n".join(logs)
    return Grade(resolved=reward == 1, reward=float(reward), patch=patch, detail=detail[:60000])


class TurnCheck:
    """``branching.review``'s ``turn_check`` for one round of a sprout evaluation."""

    def __init__(self, *, judge, task_statement: str, hidden, terms):
        self.judge, self.task_statement, self.hidden, self.terms = judge, task_statement, hidden, terms
        self._visible: dict = {}

    def __call__(self, choice: BranchChoice) -> dict:
        key = (choice.base.name, choice.checkpoint.step)
        if key not in self._visible:
            self._visible[key] = visible_text(training_messages(
                selected_prefix(choice.base.outcome.journal_path, choice.checkpoint)))
        return leak_judge.check_turn(choice.assistant_turn, terms=self.terms, visible=self._visible[key],
                                     judge=self.judge, task_statement=self.task_statement, hidden=self.hidden)

    def close(self) -> None:
        self.judge.close()


@dataclass
class TrialRecord:
    attempt: Attempt
    receipt: dict            # the agent environment's receipt (*.<backend type>.json)


def sandbox_receipt(directory: Path, backend: str, sandbox_id: str, checkpoint_mode: str) -> dict:
    """The receipt the trial's agent environment wrote for ``sandbox_id``.

    Each environment class writes ``<session_id>.<type>.json`` (the verifier's
    included); the one naming the actor's sandbox, on this backend and with the
    checkpoint mode this evaluation runs, is the agent environment's.
    """
    suffix = f".{ENVIRONMENTS[backend][0].type()}.json"
    receipts = [json.loads(path.read_text()) for path in directory.glob("*" + suffix)]
    owned = [row for row in receipts if row.get("sandbox_id") == sandbox_id]
    if len(owned) != 1 or owned[0].get("checkpoint_mode") != checkpoint_mode:
        raise RuntimeError(f"Cannot identify the {backend} agent environment receipt "
                           f"(*{suffix}, sandbox {sandbox_id}, checkpoint_mode {checkpoint_mode}) in {directory}")
    return owned[0]


async def run_trial(args, task: Task, *, name: str, branch_context: Path | None = None) -> TrialRecord:
    """Run (or, if its result exists, load) one mini trial and check what it left behind."""
    started_at = time.time()
    config = trial_config(args, task, name=name, agent="branch" if branch_context else "mini",
                          branch_context=branch_context)
    directory = config.trials_dir / config.trial_name
    if not (directory / "result.json").exists():
        if directory.exists():
            raise RuntimeError(f"Interrupted Harbor trial requires review before retry: {directory}")
        trial = await Trial.create(config)
        await trial.run()
    grade = read_grade(directory)
    journal = directory / "agent" / "trajectory.jsonl"
    if not journal.is_file():
        raise RuntimeError(f"Graded attempt has no Sprout journal: {journal}")
    outcome = outcome_from_journal(journal, run_id=name)
    if outcome.status not in {"completed", "timeout"}:
        raise RuntimeError(f"Graded attempt did not complete its actor: {outcome.status}")
    snapshot = json.loads((directory / "agent" / "snapshot.json").read_text())
    if not snapshot.get("final_snapshot_id") or not snapshot.get("sandbox_id"):
        raise RuntimeError("Attempt is missing its final snapshot or sandbox identity")
    receipt = sandbox_receipt(directory, args.backend, snapshot["sandbox_id"], args.checkpoint_mode)
    if branch_context is not None:
        restored = json.loads((directory / "restored-state.json").read_text())
        if restored.get("snapshot_id") != json.loads(branch_context.read_text())["snapshot_id"]:
            raise RuntimeError("Official trial restored a different snapshot")
    validated_at = time.time()
    attempt = Attempt(name, outcome, grade, started_at=started_at, validated_at=validated_at,
                      seconds=validated_at - started_at,
                      memory_snapshot=apply_memory_snapshot_policy(args, directory, grade))
    return TrialRecord(attempt, receipt)


def memory_snapshot_state(directory: Path) -> tuple[str | None, str | None]:
    """(memory snapshot id, 'kept' | 'released') for a trial, or (None, None)."""
    snapshot = json.loads((directory / "agent" / "snapshot.json").read_text())
    memory_id = snapshot.get("final_memory_snapshot_id")
    if not memory_id:
        return None, None
    released = any(r.get("type") == "snapshot.released" and r.get("snapshot_id") == memory_id
                   for r in read_journal(directory / "agent" / "trajectory.jsonl"))
    return memory_id, "released" if released else "kept"


def apply_memory_snapshot_policy(args, directory: Path, grade: Grade) -> str | None:
    """After grading: under ``on-failure`` a resolved attempt's memory snapshot is
    released, since nothing re-grades or branches from a pass. The release is
    journaled so a later reader knows the id no longer resolves. Returns the
    memory snapshot's state, or None when the trial took none."""
    memory_id, state = memory_snapshot_state(directory)
    if memory_id is None or state == "released":
        return state
    if getattr(args, "final_memory_snapshot", "never") != "on-failure" or not grade.resolved:
        return "kept"
    key = Path(args.api_key_file).read_text().strip() if getattr(args, "api_key_file", None) else None
    if not AgentEnvClient(args.server_url, key).delete_snapshot(memory_id):
        return "release_failed"
    journal = directory / "agent" / "trajectory.jsonl"
    last_seq = max((r.get("seq") or 0) for r in read_journal(journal))
    with journal.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps({"v": 2, "type": "snapshot.released", "seq": last_seq + 1,
                                 "run_id": "evaluate", "agent_id": "evaluate",
                                 "snapshot_id": memory_id, "kind": "memory",
                                 "reason": "resolved", "policy": "on-failure"}) + "\n")
    return "released"


def recorded_parent(args, task: Task) -> TrialRecord:
    """The baseline trial of an earlier evaluation, without spending another rollout.

    Its snapshots are the branch points, so it must have run on this backend:
    a podman subvolume means nothing to AgentENV and the reverse.
    """
    root = Path(args.parent_output).resolve()
    summaries = list(root.glob("summary.json"))
    results = list(root.glob("trials/*-parent/result.json"))
    if len(summaries) != 1 or len(results) != 1:
        raise ValueError("Recorded parent is incomplete or ambiguous")
    summary = json.loads(summaries[0].read_text())
    if (summary.get("task") != task.name or summary.get("dataset_digest") != args.dataset_digest
            or not summary.get("complete")):
        raise ValueError("Recorded parent metadata differs from this task")
    directory = results[0].parent
    journal = directory / "agent" / "trajectory.jsonl"
    if not journal.is_file():
        raise RuntimeError("Recorded parent has no Sprout journal")
    outcome = outcome_from_journal(journal, run_id="parent")
    if outcome.status not in {"completed", "timeout"}:
        raise RuntimeError(f"Recorded parent actor status is {outcome.status}")
    snapshot = json.loads((directory / "agent" / "snapshot.json").read_text())
    if not snapshot.get("final_snapshot_id") or not snapshot.get("sandbox_id"):
        raise RuntimeError("Recorded parent has no final snapshot identity")
    receipt = sandbox_receipt(directory, args.backend, snapshot["sandbox_id"], "disk_only")
    return TrialRecord(Attempt("parent", outcome, read_grade(directory),
                               memory_snapshot=memory_snapshot_state(directory)[1]), receipt)


def parent_journal(output: Path) -> Path:
    """The parent trial's journal in an evaluation output, as its summary names it."""
    root = Path(output).resolve()
    summary = json.loads((root / "summary.json").read_text())
    attempts = summary.get("attempts") or []
    if not attempts:
        raise ValueError(f"No parent attempt recorded under {output}")
    journal = Path(attempts[0]["journal"]).resolve()
    if not journal.is_relative_to(root) or not journal.is_file():
        raise ValueError(f"Parent journal is missing or outside {output}")
    return journal


def _task_image_config(image: str) -> dict:
    """The pinned image's OCI config from the registry, for a parent whose
    receipt predates recording it (microVM runs before the podman backend)."""
    if "@sha256:" not in image:
        raise ValueError("Official task image is not digest-pinned")
    regctl = find_regctl()
    if regctl is None:
        raise RuntimeError("regctl is required for snapshot restore")
    config = json.loads(subprocess.check_output(
        [str(regctl), "image", "config", image, "--format", "{{json .Config}}"], text=True, timeout=120))
    if not isinstance(config, dict):
        raise ValueError("Pinned task image has no OCI config")
    return config


def _source_commit() -> str:
    root = Path(__file__).resolve().parents[3]
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True).strip():
        raise RuntimeError("The source worktree must be clean before a real trial")
    return head


def _verify_prepared_task(args, task: Task, task_hash: str) -> None:
    preparation = json.loads(Path(args.preparation_manifest).read_text())
    if preparation.get("dataset_digest") != args.dataset_digest:
        raise ValueError("Prepared dataset digest differs from the selected cohort")
    rows = [row for row in preparation.get("tasks", [])
            if Path(row.get("path", "")).resolve() == Path(args.task_dir).resolve()
            and row.get("name") == task.name]
    if len(rows) != 1 or rows[0].get("task_toml_sha256") != task_hash:
        raise ValueError("Task is absent from the pinned preparation or its definition changed")


# --- the loop ----------------------------------------------------------------------

def review_settings(args) -> dict:
    """The analyst and reviewer's model, endpoint, key variable and reasoning effort.

    Each is its own option (``--review-model``, ``--review-endpoint``,
    ``--review-key-env``, ``--review-effort``) and falls back to the actor's, so
    the analyst and reviewer can run on another model, or the same model behind
    another endpoint, while the actor keeps ``--model``. ``parse_args`` fills the
    fallbacks in; a caller that builds ``args`` by hand may leave them out.
    """
    return {"model": getattr(args, "review_model", None) or args.model,
            "endpoint": getattr(args, "review_endpoint", None) or args.model_endpoint,
            "key_env": getattr(args, "review_key_env", None) or args.model_key_env,
            "effort": getattr(args, "review_effort", None) or getattr(args, "reasoning_effort", None)}


class Evaluation:
    def __init__(self, args, task: Task):
        self.args = args
        self.task = task
        self.output = Path(args.output)
        self.attempts: list[Attempt] = []
        self.cases: dict = {}
        self.base_image: str | None = None
        self.image_config: dict | None = None
        self.state = {"source_commit": None, "task": task.name, "backend": args.backend,
                      "task_toml_sha256": hashlib.sha256((Path(args.task_dir) / "task.toml").read_bytes()).hexdigest(),
                      "dataset_digest": args.dataset_digest, "model": args.model, "method": args.method,
                      "branch_guidance": args.branch_guidance if args.method == "sprout" else "none",
                      # Whether the analyst and reviewer were asked for schema-constrained JSON
                      # (--structured-review) rather than fenced plain-text plans.
                      "structured_review": (bool(getattr(args, "structured_review", False))
                                            if args.method == "sprout" else False),
                      # Which prompts file the analyst, reviewer and selector read (parse_args loaded it).
                      "prompts": {"path": str(args.prompts.path), "sha256": args.prompts.sha256},
                      "branch_caps": args.branches, "rounds": args.rounds if args.method == "sprout" else 1,
                      "final_memory_snapshot": getattr(args, "final_memory_snapshot", "never"),
                      "command_timeout": args.command_timeout,
                      "max_rollouts": args.max_rollouts, "attempts": [], "complete": False}
        if args.method == "sprout":
            # Where the analyst and reviewer ran (never the key: only the endpoint's URL).
            settings = review_settings(args)
            self.state["review"] = {"model": settings["model"], "endpoint": settings["endpoint"],
                                    "effort": settings["effort"]}
            if getattr(args, "leak_judge", False):
                self.state["review"]["leak_judge"] = {
                    "threshold": args.leak_judge_threshold, "model": args.leak_judge_model,
                    "resamples": args.leak_judge_resamples}

    def save(self) -> None:
        self.state["attempts"] = [
            {"name": a.name, "round": a.round_no, "status": a.outcome.status, "resolved": a.grade.resolved,
             "reward": a.grade.reward, "journal": str(a.outcome.journal_path),
             "started_at": a.started_at, "validated_at": a.validated_at, "seconds": a.seconds,
             "memory_snapshot": a.memory_snapshot}
            for a in self.attempts]
        self.state["resolved"] = any(a.grade.resolved for a in self.attempts)
        policies.write_json(self.output / "summary.json", self.state)

    @property
    def resolved(self) -> bool:
        return any(a.grade.resolved for a in self.attempts)

    def branch_context(self, choice: BranchChoice, *, capture_checkpoints: bool) -> Path:
        parent = choice.base.outcome.journal_path
        guidance = self.args.branch_guidance if self.args.method == "sprout" else "none"
        context = {"branch_guidance": guidance, "checkpoint_mode": self.args.checkpoint_mode,
                   "capture_checkpoints": capture_checkpoints,
                   "task_dir": str(Path(self.args.task_dir).resolve()),
                   "snapshot_id": choice.checkpoint.snapshot_id,
                   "base_image": self.base_image, "image_config": self.image_config,
                   "parent_journal": str(parent),
                   "parent_journal_sha256": hashlib.sha256(parent.read_bytes()).hexdigest(),
                   "step": choice.checkpoint.step,
                   "native_session_id": choice.checkpoint.session_ckpt,
                   "conversation_cut": choice.cut,
                   # Tasks that collect their answer from the repository (a
                   # [[verifier.collect]] hook) grade only committed work.
                   "commit_required": bool(self.task.config.verifier.collect),
                   "why": choice.why}
        if choice.assistant_turn is not None:
            context["assistant_turn"] = choice.assistant_turn
        if choice.hint:
            context["hint"] = choice.hint
        path = self.output / "contexts" / f"{choice.run_name}.json"
        policies.write_json(path, context)
        return path

    async def branch(self, choice: BranchChoice, round_no: int, *, capture_checkpoints: bool) -> Attempt:
        context = self.branch_context(choice, capture_checkpoints=capture_checkpoints)
        record = await run_trial(self.args, self.task, name=choice.run_name, branch_context=context)
        record.attempt.round_no = round_no
        record.attempt.hint = choice.hint
        record.attempt.assistant_turn = choice.assistant_turn
        self.attempts.append(record.attempt)
        self.save()
        return record.attempt

    async def sprout(self) -> None:
        args = self.args
        # One transport for the analyst and the reviewer: a bearer-key Chat Completions
        # endpoint, the actor's or their own (review_settings), with the prompts' JSON
        # directive and, under --structured-review, schema-constrained replies.
        settings = review_settings(args)
        review_model = settings["model"]
        transport = ReviewTransport(self.output / "reviews", model=review_model,
                                    model_endpoint=settings["endpoint"], api_key_env=settings["key_env"],
                                    max_output_tokens=args.review_output_tokens,
                                    reasoning_effort=settings["effort"],
                                    timeout_s=args.review_timeout,
                                    json_directive=args.prompts.transport_json_directive,
                                    supports_json_schema=bool(getattr(args, "structured_review", False)))
        for round_no in range(1, args.rounds + 1):
            if self.resolved:
                break
            width = args.branches[min(round_no - 1, len(args.branches) - 1)]
            for attempt in self.attempts:
                if attempt.name not in self.cases:
                    points = available_branch_points(attempt.outcome.journal_path)
                    self.cases[attempt.name] = analyse(transport, review_model, self.task.instruction,
                                                       attempt, points, width, args.analyst_tokens,
                                                       prompt_chars=args.prompt_budget_chars, prompts=args.prompts)
            plan_path = self.output / f"plan-round{round_no}.json"
            turn_check = self.turn_check() if getattr(args, "leak_judge", False) else None
            try:
                choices = review(transport, review_model, self.task.instruction, self.attempts, self.cases,
                                 round_no=round_no, width=width, plan_path=plan_path,
                                 guidance_mode=args.branch_guidance, count_mode=args.branch_count_mode,
                                 reviewer_max_attempts=args.reviewer_max_attempts,
                                 prompt_chars=args.prompt_budget_chars, prompts=args.prompts,
                                 turn_check=turn_check,
                                 turn_resamples=args.leak_judge_resamples if turn_check else 0)
            finally:
                if turn_check is not None:
                    turn_check.close()
            if not choices:
                plan = json.loads(plan_path.read_text())
                self.state["stop_reason"] = plan.get("validation_error") or "reviewer selected no branches"
                self.state["branch_schedule_complete"] = False
                break
            for choice in choices:
                await self.branch(choice, round_no, capture_checkpoints=True)

    def turn_check(self) -> "TurnCheck":
        """The round's leak check of each turn the reviewer writes (``leak_judge.check_turn``).

        Against what the reviewer could read and the actor could not: the task's
        hidden tests (``leak_judge.task_hidden_tests``), and the verifier output
        of every attempt graded so far (each ``Grade.detail``, as the analyst read
        it); h_s is the history the branch keeps at its point.
        """
        args = self.args
        test_patch, fail_to_pass = leak_judge.task_hidden_tests(Path(args.task_dir))
        verdicts = "\n".join(attempt.grade.detail for attempt in self.attempts if attempt.grade is not None)
        judge = leak_judge.JevLeakJudge(
            leak_judge.read_key_env(args.leak_judge_key_env), model=args.leak_judge_model,
            threshold=args.leak_judge_threshold, timeout_s=leak_judge.REVIEW_TIMEOUT_S,
            max_attempts=leak_judge.REVIEW_ATTEMPTS)
        if not test_patch.strip() and not fail_to_pass:
            # Nothing hidden to check against: every turn is unchecked (leak None), never clean.
            return TurnCheck(judge=judge, task_statement=self.task.instruction, hidden=None, terms=None)
        return TurnCheck(judge=judge, task_statement=self.task.instruction,
                         hidden=leak_judge.HiddenText(test_patch, fail_to_pass, verdicts),
                         terms=hidden_terms(fail_to_pass=list(fail_to_pass),
                                            test_patch_paths=leak_judge.patch_files(test_patch)))

    async def sample(self) -> None:
        """bpo / shepherd: one round of point-only branches from the parent."""
        args = self.args
        parent = self.attempts[0]
        journal = parent.outcome.journal_path
        points = available_branch_points(journal)
        if not points:
            raise RuntimeError("Shared parent exposes no exact mini checkpoint/session pairs")
        client = policies.ModelClient(args.model_endpoint, bearer_key(args.model_key_env))
        selected = policies.select_points(args, self.task.instruction, journal, points, client,
                                          prompts=args.prompts)
        policies.write_json(self.output / "plan.json", {
            "task": self.task.name, "method": args.method, "shared_parent": str(journal),
            "max_total_rollouts": args.max_rollouts, "budget_includes_shared_initial": True,
            "eligible_steps": sorted(points), "selected": selected,
            "branch_guidance": "none", "hint_delivery": "fixed-neutral"})
        for index, selection in enumerate(selected, 1):
            point = points[selection["step"]]
            choice = BranchChoice(f"b{index:02d}-step{point.step}", parent, point,
                                  conversation_restore(journal, point.step, point.session_ckpt), "",
                                  str(selection.get("reason") or ""))
            await self.branch(choice, 1, capture_checkpoints=False)
            if self.resolved:
                break

    async def run(self) -> dict:
        args = self.args
        if self.output.exists() and any(self.output.iterdir()):
            raise FileExistsError("Use a new output directory for each one-task evaluation")
        task_hash = self.state["task_toml_sha256"]
        _verify_prepared_task(args, self.task, task_hash)
        self.state["source_commit"] = _source_commit()
        self.output.mkdir(parents=True)
        self.save()
        if args.parent_output:
            root = recorded_parent(args, self.task)
        else:
            root = await run_trial(args, self.task, name="parent")
        if args.method != "baseline":
            require_mini_parent(root.attempt.outcome.journal_path)
        if args.backend == "podman" and root.receipt.get("store") != environment_kwargs(args)["store"]:
            raise ValueError(f"The parent's snapshots live in {root.receipt.get('store')}, not --store {args.store}")
        self.attempts.append(root.attempt)
        self.base_image = root.receipt["image"]
        # The environment records the config its sandbox ran under; a receipt
        # from before that was recorded is resolved from the registry.
        self.image_config = root.receipt.get("image_config") or _task_image_config(self.base_image)
        self.save()
        if args.method == "sprout":
            await self.sprout()
        elif args.method in ("bpo", "shepherd"):
            if self.resolved:
                raise ValueError("Policy evaluation cannot run on a resolved shared parent")
            await self.sample()
        if hashlib.sha256((Path(args.task_dir) / "task.toml").read_bytes()).hexdigest() != task_hash:
            raise RuntimeError("Official task definition changed during evaluation")
        self.state["complete"] = True
        self.state.setdefault("branch_schedule_complete", True)
        self.save()
        return self.state


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task-dir", type=Path, required=True)
    parser.add_argument("--dataset-digest", required=True)
    parser.add_argument("--preparation-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--method", choices=METHODS, default="sprout")
    parser.add_argument("--parent-output", type=Path,
                        help="reuse a completed baseline evaluation's parent trial")
    parser.add_argument("--model", required=True)
    parser.add_argument("--model-endpoint", required=True)
    parser.add_argument("--model-key-env", required=True,
                        help="environment variable holding the bearer key for --model-endpoint")
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument("--backend", choices=tuple(ENVIRONMENTS), default="podman",
                        help="where every trial's sandboxes come from: podman (rootless containers on "
                             "btrfs subvolumes of --store, the default; a branch needs the store its "
                             "parent ran on) or microvm (AgentENV)")
    parser.add_argument("--store", type=Path,
                        help="podman: the btrfs store its sandboxes and snapshots live in (sudo "
                             "ops/btrfs_store.sh makes one; default $SPROUT_PODMAN_STORE, else ~/sprout-btrfs)")
    parser.add_argument("--server-url", help="microvm: the AgentENV server")
    parser.add_argument("--api-key-file", type=Path, help="microvm: the AgentENV key")
    parser.add_argument("--sandbox-ttl", type=int, default=36000, help="microvm: the VM lease")
    parser.add_argument("--checkpoint-mode", choices=["disk_only", "none"], default="disk_only",
                        help="per-turn disk snapshots (branch points) or none; the rollout "
                             "always ends with one disk snapshot")
    parser.add_argument("--final-memory-snapshot", choices=MEMORY_SNAPSHOT_POLICIES, default="never",
                        help="also take a full (memory) snapshot at the end: never; always; or "
                             "on-failure, which releases it once the attempt is graded resolved. "
                             "For graders that run inside the agent's VM and need its processes")
    parser.add_argument("--max-output-tokens", type=int, default=64000)
    parser.add_argument("--max-turns", type=int, default=300)
    parser.add_argument("--command-timeout", type=int, default=DEFAULT_COMMAND_TIMEOUT_S,
                        help="seconds the runtime gives each agent command before killing it (the model's "
                             "bash tool has no timeout argument, so every command runs under this one limit; "
                             "a branch runs under its parent's)")
    parser.add_argument("--reasoning-effort")
    # sprout
    parser.add_argument("--branch-guidance", choices=BRANCH_GUIDANCE_MODES, default=DEFAULT_GUIDANCE)
    parser.add_argument("--branch-count-mode", choices=BRANCH_COUNT_MODES, default="adaptive")
    parser.add_argument("--prompts", type=Path, default=DEFAULT_PROMPTS_PATH,
                        help="YAML with every analyst/reviewer/shepherd prompt (default: the packaged file); "
                             "its sha256 is recorded")
    parser.add_argument("--structured-review", action="store_true",
                        help="ask the analyst and reviewer for schema-constrained JSON (response_format "
                             "json_schema) instead of fenced plain-text plans; needs an endpoint that "
                             "supports json_schema, e.g. OpenAI or a LiteLLM proxy")
    parser.add_argument("--rounds", type=int, default=2)
    parser.add_argument("--branches", type=int, nargs="+", default=[4, 3],
                        help="branch count per round: a maximum in adaptive mode, exact in fixed mode")
    parser.add_argument("--review-output-tokens", type=int, default=64000)
    # The analyst and reviewer may run on another model, or behind another endpoint,
    # while the actor that continues each branch keeps --model; each option defaults
    # to the actor's.
    parser.add_argument("--review-model",
                        help="the analyst and reviewer model (default --model), served by --review-endpoint")
    parser.add_argument("--review-endpoint",
                        help="the Chat Completions endpoint the analyst and reviewer use (default "
                             "--model-endpoint), e.g. a LiteLLM proxy alias for a model the actor's "
                             "endpoint does not serve")
    parser.add_argument("--review-key-env",
                        help="environment variable holding the bearer key for --review-endpoint "
                             "(default --model-key-env)")
    parser.add_argument("--review-effort",
                        help="reasoning_effort of the analyst and reviewer requests (default --reasoning-effort)")
    parser.add_argument("--reviewer-max-attempts", type=int, default=3)
    parser.add_argument("--leak-judge", action="store_true",
                        help="sprout, assistant-turn: check every turn the reviewer writes for hidden-test content "
                             "(the task's tests, the verifier output) with methods.leak's leak_terms and Jev before it "
                             "is taken; a flagged turn goes back to the reviewer (docs/LEAK_JUDGE.md)")
    parser.add_argument("--leak-judge-resamples", type=int, default=2,
                        help="how often a flagged turn goes back before it is taken anyway, marked (default 2)")
    parser.add_argument("--leak-judge-threshold", type=float, default=leak_judge.THRESHOLD)
    parser.add_argument("--leak-judge-model", default=leak_judge.MODEL)
    parser.add_argument("--leak-judge-key-env", default=leak_judge.KEY_ENV,
                        help="the environment variable holding the TypeSafe key")
    parser.add_argument("--analyst-tokens", type=int, default=100000)
    parser.add_argument("--review-timeout", type=int, default=3600,
                        help="seconds one analyst or reviewer request may take; a request that times out "
                             "is retried once with half the output budget (default 3600: 64k output "
                             "tokens take about 30 minutes at the rates seen)")
    parser.add_argument("--prompt-budget-chars", type=int, default=DEFAULT_PROMPT_CHARS,
                        help="size of every analyst and reviewer request, in characters (about a "
                             "quarter as many tokens); a reviewer request over it is reduced in a fixed "
                             "order, recorded in plan-roundN.json, and one the endpoint refuses (HTTP "
                             "400) is retried with the budget halved, twice at most")
    # bpo / shepherd
    parser.add_argument("--max-rollouts", type=int, default=8,
                        help="bpo/shepherd: total rollouts including the shared parent")
    parser.add_argument("--bpo-top-k", type=int, default=5)
    parser.add_argument("--bpo-min-spacing", type=int, default=64)
    parser.add_argument("--bpo-max-points", type=int, default=7)
    parser.add_argument("--bpo-scoring-workers", type=int, default=2)
    parser.add_argument("--meta-max-tokens", type=int, default=64000)
    parser.add_argument("--meta-transcript-tokens", type=int, default=60000)
    parser.add_argument("--volatile-ok", action="store_true",
                        help="allow --output under /tmp and friends; refused by default because "
                             "the trials there are the evaluation's only record")
    args = parser.parse_args(argv)
    reason = volatile_reason(args.output)
    if reason and not args.volatile_ok:
        parser.error("refusing: %s (pass --volatile-ok to override)" % reason)
    if (args.rounds < 0 or not args.branches or any(n < 1 for n in args.branches)
            or args.sandbox_ttl < 30000 or args.reviewer_max_attempts < 1
            or not 1 <= args.max_rollouts <= MAX_ROLLOUTS or not 1 <= args.bpo_top_k <= 20):
        parser.error("Invalid branch schedule, rollout budget, snapshot TTL or reviewer attempts")
    # bpo/shepherd keep the protocol's 8; a larger budget is for a sprout round-scaling schedule.
    if args.method in ("bpo", "shepherd") and args.max_rollouts > 8:
        parser.error("bpo and shepherd allow at most 8 rollouts, the shared parent included")
    if args.prompt_budget_chars < 50_000:
        parser.error("--prompt-budget-chars below 50000 leaves no room for the task and a transcript")
    try:
        # Loaded once, here: a bad file fails before any sandbox starts, and the
        # loaded Prompts (with the file's sha256) is what the loop and the summary use.
        args.prompts = load_prompts(args.prompts)
    except (OSError, ValueError) as error:
        parser.error("--prompts: %s" % error)
    if not args.model_key_env.isidentifier():
        parser.error("--model-key-env must name one environment variable")
    if args.review_key_env and not args.review_key_env.isidentifier():
        parser.error("--review-key-env must name one environment variable")
    if args.method in ("bpo", "shepherd") and not args.parent_output:
        parser.error("bpo and shepherd branch from a recorded baseline: pass --parent-output")
    if args.command_timeout <= 0:
        parser.error("--command-timeout must be a positive number of seconds")
    if args.review_timeout <= 0:
        parser.error("--review-timeout must be a positive number of seconds")
    if args.leak_judge:
        if args.method != "sprout" or args.branch_guidance != "assistant-turn":
            parser.error("--leak-judge checks reviewer-written turns: --method sprout with assistant-turn guidance")
        if args.leak_judge_resamples < 0 or not 0 < args.leak_judge_threshold <= 1:
            parser.error("--leak-judge-resamples must be >= 0 and --leak-judge-threshold in (0, 1]")
        try:
            leak_judge.read_key_env(args.leak_judge_key_env)   # what the round's judge reads; unset fails here
        except ValueError as error:
            parser.error(f"--leak-judge: {error}")
    # The reviewer's settings fall back to the actor's, one by one.
    args.review_model = args.review_model or args.model
    args.review_endpoint = args.review_endpoint or args.model_endpoint
    args.review_key_env = args.review_key_env or args.model_key_env
    args.review_effort = args.review_effort or args.reasoning_effort
    if args.backend == "podman" and not args.store:
        args.store = Path(default_podman_store())
    if args.backend == "podman" and args.final_memory_snapshot != "never":
        parser.error("--final-memory-snapshot needs a backend with memory snapshots; "
                     "podman snapshots are disk-only")
    return args


def main(argv=None) -> int:
    args = parse_args(argv)
    if not Path(args.runtime_bin).is_file():
        raise FileNotFoundError(args.runtime_bin)
    bearer_key(args.model_key_env)   # an unset key fails here, before any trial
    if args.method == "sprout":
        bearer_key(args.review_key_env)   # the analyst and reviewer's too
    state = asyncio.run(Evaluation(args, Task(args.task_dir)).run())
    print(json.dumps({key: value for key, value in state.items() if key != "attempts"}, indent=2))
    return 0 if state["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

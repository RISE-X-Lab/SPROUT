"""The filesystem model of run output: what kind of output a directory is, and what it holds.

Nothing here writes. ``config.json`` and ``lock.json`` are never read, because they
carry an endpoint and a key variable name.

The files come from ``examples/qwen/run.py`` (stage receipts, ``<method>.task.json``,
manifests, ``retained-checkpoints.json``), ``examples/qwen/report.py``
(``metrics-<method>.json``), ``sprout.eval.evaluate`` (``summary.json``, plan files,
the ``groups/`` documents) and its review jobs (``plan-points.json`` and receipts in
the worker's attempt directory). An evaluation from before the Run Store holds Harbor
``trials/`` with receipts, ``restored-state.json``, ``execution.json`` and
``snapshot.json``, and also ``contexts/`` and ``reviews/``.
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path

from ops.report.journal import Trajectory, relocate

KINDS = ("experiment", "method", "task", "evaluate", "trial", "journal", "gate")
#: The runner's methods, in report order (examples/qwen/run.py METHODS).
METHODS = ("baseline", "sprout", "bpo", "shepherd")
#: Receipt suffix -> the backend name journals, ledgers and manifests use.
RECEIPT_BACKENDS = {"podman": "podman", "agentenv": "microvm"}
#: The keys that ``stage()`` in examples/qwen/run.py writes into a stage receipt. Any other
#: key was added by hand, for example a hand-off ``sentinel``, and is shown as such.
RECEIPT_KEYS = frozenset(
    {
        "task",
        "method",
        "started_at",
        "complete",
        "output",
        "source_commit",
        "exit_code",
        "error_type",
        "error",
        "finished_at",
        "seconds",
    }
)
#: The first words of the analyst's and the reviewers' prompts in resources/review/prompts.yaml.
#: load_review() tells the kinds of review records apart by them. A test keeps the two in step.
ANALYST_PROMPT = "You are analysing ONE failed"
REVIEWER_PROMPT = "You are the REVIEWER"
_EXCEPTION_LINE = re.compile(r"^\w+(?:Error|Exception)\b")


def read_json(path: Path) -> dict | list | None:
    """The file's JSON, or None when it is missing or half-written."""
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def read_dict(path: Path) -> dict | None:
    body = read_json(path)
    return body if isinstance(body, dict) else None


def is_method_root(path: Path) -> bool:
    """A directory with a readable ``manifest-<method>.json`` (a corrupt one is no root)."""
    return path.is_dir() and any(read_dict(f) is not None for f in path.glob("manifest-*.json"))


def detect(path: Path) -> str | None:
    """The kind of run output that ``path`` is (one of ``KINDS``), or None."""
    path = Path(path)
    if path.is_file():
        return "journal" if path.suffix == ".jsonl" else None
    if not path.is_dir():
        return None
    if (path / "gate.jsonl").is_file():
        return "gate"
    summary = read_dict(path / "summary.json")
    if summary is not None and "attempts" in summary:
        return "evaluate"
    if (path / "config.json").is_file() and (path / "agent").is_dir():
        return "trial"
    if is_method_root(path):
        return "method"
    if any(is_method_root(path / name) for name in METHODS + ("all",)):
        return "experiment"
    # A task directory holds <method>.task.json once the task ends, <method>.stage.json once a
    # method stage starts, and only baseline/attempt-N.stage.json before that.
    if (
        any(path.glob("*.task.json"))
        or any(path.glob("*.stage.json"))
        or any((path / "baseline").glob("attempt-*.stage.json"))
    ):
        return "task"
    return None


# --- the runner's files ------------------------------------------------------------


@dataclass(frozen=True)
class Stage:
    """One runner stage: ``attempt-N`` of the baseline, or a method.

    ``state`` is complete, errored, running, stale, unlaunched or missing. A receipt with no
    ``<stage>.log`` beside it is ``unlaunched``, never ``running``: no runner started it.
    """

    name: str
    receipt: dict | None
    output: Path
    state: str
    error: str | None  # the receipt's error
    cause: str | None  # from the stage log: the exception or stop reason that ended it
    seconds: float | None
    extras: dict  # receipt keys the runner never writes (RECEIPT_KEYS)


@dataclass(frozen=True)
class AttemptRow:
    """A ``summary.json`` attempt row, its journal relocated."""

    name: str
    round: int
    status: str | None
    resolved: bool | None
    reward: float | None
    journal: Path | None
    started_at: float | None
    validated_at: float | None
    seconds: float | None
    memory_snapshot: str | None


@dataclass(frozen=True)
class Grade:
    reward: float | None
    resolved: bool | None
    detail: str  # "f2p 9/20 p2p 3/3" when the verifier reports them
    exception: str | None  # exception_info.exception_type


@dataclass(frozen=True)
class Trial:
    path: Path
    name: str
    journal: Path | None
    grade: Grade | None
    receipt: dict | None  # the receipt of the agent sandbox (agent_receipt)
    restored: dict | None  # restored-state.json, branch trials only
    phase: str  # running | grading | graded | errored | empty


@dataclass(frozen=True)
class Plan:
    """``plan-round<N>.json``."""

    round: int
    path: Path
    limit: int | None
    count_mode: str | None
    guidance: str | None
    reports: dict
    available_steps: dict
    review_attempts: tuple[dict, ...]
    selected: tuple[dict, ...]
    validation_error: str | None
    review_directory: Path | None  # the review job's attempt directory, for a round through the Run Store


@dataclass(frozen=True)
class Evaluation:
    """One ``sprout.eval.evaluate`` output directory."""

    path: Path
    summary: dict
    attempts: tuple[AttemptRow, ...]
    trials: tuple[Trial, ...]
    plans: tuple[Plan, ...]
    reviews: tuple[dict, ...]
    retained: dict | None
    groups: tuple[dict, ...]  # the ``groups/`` documents of an evaluation through the Run Store


@dataclass(frozen=True)
class TaskState:
    name: str
    path: Path
    task_json: dict | None
    stages: tuple[Stage, ...]
    status: str  # complete | errored | in-progress | stale | unlaunched | not-started


@dataclass(frozen=True)
class MethodRoot:
    path: Path
    method: str
    manifest: dict
    tasks: tuple[str, ...]
    lock_pid: int | None
    lock_alive: bool | None
    metrics: dict | None
    baseline_root: Path | None


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def load_method_root(path: Path) -> MethodRoot:
    path = Path(path)
    manifests = {}
    for file in sorted(path.glob("manifest-*.json")):
        body = read_dict(file)
        if body is not None:
            manifests[file.stem[len("manifest-") :]] = body
    if not manifests:
        raise ValueError(f"{path} has no readable manifest-*.json")
    # `--method all` writes manifest-baseline.json beside manifest-all.json.
    method = next((m for m in manifests if m != "baseline"), "baseline")
    manifest = manifests[method]
    lock_pid = lock_alive = None
    lock = path / f".{method}.lock"
    if lock.is_file():
        try:
            lock_pid = int(lock.read_text().strip())
        except ValueError:
            lock_pid = None
        lock_alive = pid_alive(lock_pid) if lock_pid is not None else False
    recorded_root = manifest.get("baseline_root")
    baseline_root = (
        relocate(recorded_root, path / f"manifest-{method}.json", min_parts=1) if recorded_root else None
    )
    return MethodRoot(
        path=path,
        method=method,
        manifest=manifest,
        tasks=tuple(manifest.get("tasks") or ()),
        lock_pid=lock_pid,
        lock_alive=lock_alive,
        metrics=read_dict(path / f"metrics-{method}.json"),
        baseline_root=baseline_root,
    )


def method_roots(path: Path) -> list[MethodRoot]:
    """The method roots of an experiment in ``METHODS`` order, or a method root alone."""
    path = Path(path)
    if is_method_root(path):
        return [load_method_root(path)]
    roots = [
        load_method_root(child)
        for child in sorted(path.iterdir())
        if child.is_dir() and is_method_root(child)
    ]
    order = {name: index for index, name in enumerate(METHODS)}
    return sorted(roots, key=lambda root: (order.get(root.method, len(order)), root.method))


def stage_state(receipt: dict | None, launched: bool, lock_alive: bool | None) -> str:
    """The ``Stage.state`` of a receipt.

    ``launched`` says that the stage's log exists. The runner creates the log before it starts
    the stage.
    """
    if receipt is None:
        return "missing"
    if receipt.get("complete"):
        return "complete"
    if receipt.get("finished_at") is not None:
        return "errored"
    if not launched:
        return "unlaunched"
    return "running" if lock_alive else "stale"


def log_cause(log: Path) -> str | None:
    """Why a stage ended, read from the tail of its log, or None for a missing or empty log.

    The cause is the ``stop_reason`` or ``error`` of a closing JSON state dump. Otherwise it
    is the last exception line (``SomeError: ...``), else the last non-empty line.
    """
    try:
        with log.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            size = stream.tell()
            stream.seek(max(0, size - 16384))
            tail = stream.read().decode("utf-8", errors="replace")
    except OSError:
        return None
    lines = [line.rstrip() for line in tail.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    # sprout.eval.evaluate prints its state as indented JSON last and exits 0, also when the branch
    # schedule stopped short. The runner's attempts() then refuses the evaluation as incomplete.
    if lines and lines[-1] == "}":
        start = next((i for i in range(len(lines) - 1, -1, -1) if lines[i] == "{"), None)
        if start is not None:
            try:
                state = json.loads("\n".join(lines[start:]))
            except ValueError:
                state = None
            if isinstance(state, dict):
                for key in ("stop_reason", "error"):
                    if state.get(key):
                        return "%s: %s" % (key, state[key])
                lines = lines[:start]  # the dump names no cause: what preceded it does
    stripped = [line.strip() for line in lines if line.strip()]
    exception = next((line for line in reversed(stripped) if _EXCEPTION_LINE.match(line)), None)
    return exception or (stripped[-1] if stripped else None)


def load_stage(receipt_path: Path, lock_alive: bool | None) -> Stage:
    """The stage that a receipt describes.

    The stage's output is the directory of the same name, and its log is ``<name>.log`` beside both.
    """
    output = receipt_path.with_name(receipt_path.name[: -len(".stage.json")])
    log = output.with_name(output.name + ".log")
    receipt = read_dict(receipt_path)
    state = stage_state(receipt, log.is_file(), lock_alive)
    error = cause = None
    if state == "errored":
        error = (receipt or {}).get("error")
        cause = log_cause(log)
        if cause == error:
            cause = None
    return Stage(
        name=output.name,
        receipt=receipt,
        output=output,
        state=state,
        error=error,
        cause=cause,
        seconds=(receipt or {}).get("seconds"),
        extras={k: v for k, v in (receipt or {}).items() if k not in RECEIPT_KEYS},
    )


def _attempt_number(stage: Stage) -> int:
    match = re.search(r"(\d+)$", stage.name)
    return int(match.group(1)) if match else 0


def task_stages(task_dir: Path, lock_alive: bool | None) -> tuple[Stage, ...]:
    """Baseline attempts (``baseline/attempt-N.stage.json``), then method stages (``<m>.stage.json``)."""
    baseline = sorted(
        (load_stage(p, lock_alive) for p in (task_dir / "baseline").glob("attempt-*.stage.json")),
        key=_attempt_number,
    )
    methods = sorted(
        (load_stage(p, lock_alive) for p in task_dir.glob("*.stage.json")),
        key=lambda s: METHODS.index(s.name) if s.name in METHODS else len(METHODS),
    )
    return tuple(baseline + methods)


def task_state(root: MethodRoot, name: str) -> TaskState:
    path = root.path / "tasks" / name
    task_json = read_dict(path / f"{root.method}.task.json")
    stages = task_stages(path, root.lock_alive) if path.is_dir() else ()
    states = {stage.state for stage in stages}
    if task_json is not None and task_json.get("complete"):
        status = "complete"
    elif task_json is not None and task_json.get("error"):
        status = "errored"
    elif "unlaunched" in states and states <= {"complete", "unlaunched"}:
        status = "unlaunched"  # the runner will refuse the receipt when it gets here
    elif stages:
        status = "in-progress" if root.lock_alive else "stale"
    else:
        status = "not-started"
    return TaskState(name=name, path=path, task_json=task_json, stages=stages, status=status)


# --- an evaluation and its trials --------------------------------------------------


def load_grade(result: dict) -> Grade:
    rewards = (result.get("verifier_result") or {}).get("rewards") or {}
    reward = rewards.get("reward")
    reward = float(reward) if isinstance(reward, (int, float)) else None
    detail = ""
    if all(
        isinstance(rewards.get(k), (int, float))
        for k in ("f2p_passed", "f2p_total", "p2p_passed", "p2p_total")
    ):
        detail = "f2p %d/%d p2p %d/%d" % (
            rewards["f2p_passed"],
            rewards["f2p_total"],
            rewards["p2p_passed"],
            rewards["p2p_total"],
        )
    exception = result.get("exception_info")
    return Grade(
        reward=reward,
        resolved=None if reward is None else reward == 1,
        detail=detail,
        exception=exception.get("exception_type") if isinstance(exception, dict) else None,
    )


def agent_receipt(trial_dir: Path, sandbox_id: str | None) -> dict | None:
    """The receipt of the trial's agent sandbox, or None when no single receipt matches.

    The receipt that names ``sandbox_id`` wins, else the only one without a verifier overlay.
    """
    # Receipt file names are Harbor session ids, so only the suffix and the body identify one.
    receipts = []
    for path in sorted(trial_dir.glob("*.json")):
        for suffix, backend in RECEIPT_BACKENDS.items():
            if path.name.endswith(f".{suffix}.json"):
                body = read_dict(path)
                if body is not None and "sandbox_id" in body:
                    receipts.append({"backend": backend, **body})
    if sandbox_id:
        owned = [r for r in receipts if r.get("sandbox_id") == sandbox_id]
        if len(owned) == 1:
            return owned[0]
    agents = [r for r in receipts if r.get("overlay") is None]
    return agents[0] if len(agents) == 1 else None


def load_trial(path: Path) -> Trial:
    path = Path(path)
    journal = path / "agent" / "trajectory.jsonl"
    result = read_dict(path / "result.json")
    snapshot = read_dict(path / "agent" / "snapshot.json") or {}
    execution = read_dict(path / "agent" / "execution.json") or {}
    grade = load_grade(result) if result is not None else None
    if grade is not None and grade.exception:
        phase = "errored"
    elif grade is not None:
        phase = "graded"
    elif (path / "agent" / "execution.json").is_file():
        phase = "grading"
    elif journal.is_file():
        phase = "running"
    else:
        phase = "empty"
    return Trial(
        path=path,
        name=path.name,
        journal=journal if journal.is_file() else None,
        grade=grade,
        receipt=agent_receipt(path, snapshot.get("sandbox_id") or execution.get("sandbox_id")),
        restored=read_dict(path / "restored-state.json"),
        phase=phase,
    )


def _reduce_review_attempt(attempt: dict) -> dict:
    plan = attempt.get("plan")
    branches = plan.get("branches") if isinstance(plan, dict) else None
    return {
        "attempt": attempt.get("attempt"),
        "status": attempt.get("status"),
        "validation_error": attempt.get("validation_error"),
        "request_error": attempt.get("request_error"),
        "state_error": attempt.get("state_error"),
        "branches": len(branches) if isinstance(branches, list) else None,
        "response": attempt.get("response") if isinstance(attempt.get("response"), str) else None,
    }


def load_plan(path: Path) -> Plan | None:
    body = read_dict(path)
    if body is None:
        return None
    match = re.search(r"plan-round(\d+)\.json$", path.name)
    return Plan(
        round=int(match.group(1)) if match else 0,
        path=path,
        limit=body.get("branch_limit"),
        count_mode=body.get("branch_count_mode"),
        guidance=body.get("branch_guidance"),
        reports=body.get("reports") or {},
        available_steps=body.get("available_steps") or {},
        review_attempts=tuple(
            _reduce_review_attempt(a) for a in body.get("review_attempts") or [] if isinstance(a, dict)
        ),
        selected=tuple(b for b in body.get("selected_branches") or [] if isinstance(b, dict)),
        validation_error=body.get("validation_error"),
        review_directory=Path(directory)
        if (directory := (body.get("review_job") or {}).get("directory"))
        else None,
    )


def load_review(path: Path) -> dict:
    """A review transport record, reduced to its timing, usage, kind and the text used for matching.

    ``kind`` is ``analyst``, ``reviewer`` or ``unknown``, by the first words of the prompt.
    """
    body = read_dict(path) or {}
    prompt = (body.get("request") or {}).get("prompt") or ""
    kind = (
        "analyst"
        if prompt.startswith(ANALYST_PROMPT)
        else "reviewer"
        if prompt.startswith(REVIEWER_PROMPT)
        else "unknown"
    )
    return {
        "file": path.name,
        "started_at": body.get("started_at"),
        "finished_at": body.get("finished_at"),
        "seconds": body.get("seconds"),
        "usage": (body.get("response") or {}).get("usage"),
        "kind": kind,
        "in_flight": body.get("finished_at") is None and "error" not in body,
        "error": body.get("error"),
        "text": body.get("text"),
    }


def attempt_rows(summary: dict, anchor: Path) -> tuple[AttemptRow, ...]:
    rows = []
    for row in summary.get("attempts") or []:
        if not isinstance(row, dict):
            continue
        recorded = row.get("journal")
        rows.append(
            AttemptRow(
                name=str(row.get("name")),
                round=row.get("round") or 0,
                status=row.get("status"),
                resolved=row.get("resolved") if isinstance(row.get("resolved"), bool) else None,
                reward=row.get("reward"),
                journal=relocate(recorded, anchor) if recorded else None,
                started_at=row.get("started_at"),
                validated_at=row.get("validated_at"),
                seconds=row.get("seconds"),
                memory_snapshot=row.get("memory_snapshot"),
            )
        )
    return tuple(rows)


def summary_trials(summary: dict, anchor: Path) -> tuple[Trial, ...]:
    """The attempts of an evaluation through the Run Store, as trials: each row's journal and verdict."""
    trials = []
    for row in summary.get("attempts") or []:
        if not isinstance(row, dict) or not row.get("journal"):
            continue
        journal = relocate(row["journal"], anchor)
        reward = row.get("reward")
        resolved = row.get("resolved")
        grade = Grade(
            reward=float(reward) if isinstance(reward, (int, float)) else None,
            resolved=resolved if isinstance(resolved, bool) else None,
            detail="",
            exception=None,
        )
        trials.append(
            Trial(
                path=(journal or Path(row["journal"])).parent,
                name=str(row.get("name")),
                journal=journal if journal is not None and journal.is_file() else None,
                grade=grade,
                receipt=None,
                restored=None,
                phase="graded",
            )
        )
    return tuple(trials)


def load_evaluation(path: Path) -> Evaluation:
    path = Path(path)
    summary = read_dict(path / "summary.json") or {}
    trials = (
        tuple(load_trial(p) for p in sorted((path / "trials").iterdir()) if p.is_dir())
        if (path / "trials").is_dir()
        else summary_trials(summary, path / "summary.json")
    )
    plans = tuple(
        p
        for p in (
            load_plan(f)
            for f in sorted(path.glob("plan-round*.json"), key=lambda f: int(re.findall(r"\d+", f.name)[-1]))
        )
        if p is not None
    )
    receipts = sorted((path / "reviews").glob("review-*.json")) if (path / "reviews").is_dir() else []
    # A round through the Run Store keeps its receipts in its review job's directory, one folder per call.
    for plan in plans:
        if plan.review_directory is not None:
            receipts += sorted(plan.review_directory.glob("reviews/*/review-*.json"))
    groups = tuple(
        row["document"]
        for row in (read_dict(f) for f in sorted((path / "groups").glob("*.json")))
        if row is not None and isinstance(row.get("document"), dict)
    )
    return Evaluation(
        path=path,
        summary=summary,
        attempts=attempt_rows(summary, path / "summary.json"),
        trials=trials,
        plans=plans,
        reviews=tuple(load_review(f) for f in receipts),
        retained=read_dict(path / "retained-checkpoints.json"),
        groups=groups,
    )


def baseline_dir_of(evaluation: Evaluation) -> Path | None:
    """The ``baseline/`` directory that holds the parent attempt, or None when it is not found.

    It is the directory of ``summary.json``'s ``parent_output``. An evaluation from before the
    Run Store records none. Then it is the directory that holds ``attempt-N/trials/<trial>``.
    """
    recorded = evaluation.summary.get("parent_output")
    if recorded:
        output = relocate(recorded, evaluation.path / "summary.json", min_parts=3)
        return output.parent if output is not None and output.parent.name == "baseline" else None
    if not evaluation.attempts or evaluation.attempts[0].journal is None:
        return None
    journal = evaluation.attempts[0].journal
    if len(journal.parents) < 5 or journal.parents[2].name != "trials":
        return None
    candidate = journal.parents[4]
    return candidate if candidate.is_dir() else None


def receipt_backend(root: MethodRoot) -> tuple[str | None, str | None]:
    """(backend, store) from the first agent receipt in the root's trials, or (None, None).

    It serves a manifest that records no ``backend``. The receipts of its trials still name it.
    """
    for name in root.tasks:
        task_dir = root.path / "tasks" / name
        for stage in sorted((task_dir / "baseline").glob("attempt-*")) + [task_dir / m for m in METHODS[1:]]:
            if not (stage / "trials").is_dir():
                continue
            for trial in sorted((stage / "trials").iterdir()):
                receipt = agent_receipt(trial, None) if trial.is_dir() else None
                if receipt is not None:
                    return receipt.get("backend"), receipt.get("store")
    return None, None


def receipt_identity(journal: Path) -> tuple[str | None, str | None]:
    """(backend, sandbox id) from the agent receipt of the trial that holds ``journal``.

    It serves a journal with no ``sandbox.created`` record. Outside a trial it returns (None, None).
    """
    journal = Path(journal)
    if journal.parent.name != "agent":
        return None, None
    receipt = load_trial(journal.parent.parent).receipt or {}
    return receipt.get("backend"), receipt.get("sandbox_id")


def trial_backend(trial: Trial, trajectory: Trajectory | None) -> str | None:
    """The backend in the journal's ``sandbox.created`` record, else the one in the trial's receipt."""
    if trajectory is not None and trajectory.backend:
        return trajectory.backend
    return (trial.receipt or {}).get("backend")

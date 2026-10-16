"""The ``runs``, ``task``, ``snapshots`` and ``list`` models and their text renderers.

Every model is a frozen dataclass; ``--format json`` is ``dataclasses.asdict``
of the same object the text renderer consumes, so the two cannot disagree.
Nothing reads the clock: a live attempt shows its last event's timestamp, and
the only system-dependent inputs are whether the runner's pid is alive and which
files exist (a stage's log, a store's snapshot directories).
"""

from __future__ import annotations

import json
import os
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

from ops.report.journal import (
    Capture,
    Trajectory,
    read_trajectory,
    relocate,
    step_count,
    summarize_command,
    turn_count,
)
from ops.report.journal import (
    minutes as journal_minutes,
)
from ops.report.layout import (
    METHODS,
    Evaluation,
    MethodRoot,
    Stage,
    Trial,
    baseline_dir_of,
    detect,
    is_method_root,
    load_evaluation,
    load_method_root,
    load_stage,
    load_trial,
    method_roots,
    read_dict,
    receipt_backend,
    receipt_identity,
    task_state,
    trial_backend,
)

#: Which command reads which kind of target.
SUGGESTED_COMMAND = {
    "experiment": "runs",
    "method": "runs",
    "gate": "runs",
    "task": "task",
    "evaluate": "task",
    "trial": "export",
    "journal": "export",
}
MANY_JOURNALS = 50
#: Error and cause texts are one line this wide in the text renderers (json keeps them whole).
NOTE_WIDTH = 200
UNLAUNCHED_NOTE = "receipt only, no log or output: this runner never launched it"
#: An absolute path inside an error message; the runner's own messages name the task's
#: output, so grouping them needs the path generalised.
_PATH = re.compile(r"(?<!\w)/[^\s;,)'\"]+")


# --- shared helpers ----------------------------------------------------------------


def columns(rows: list[tuple[str, ...]]) -> list[str]:
    """Each column padded to its widest cell."""
    if not rows:
        return []
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    return ["  ".join(str(cell).ljust(width) for cell, width in zip(row, widths)).rstrip() for row in rows]


def pct(n: int, d: int) -> str:
    return "%.1f%%" % (100.0 * n / d) if d else "n/a"


def mins(seconds: float | None) -> str:
    return "%.1f min" % (seconds / 60) if isinstance(seconds, (int, float)) else "? min"


def wrap(names: list[str], indent: str, width: int = 100) -> list[str]:
    lines, current = [], ""
    for name in names:
        if current and len(indent) + len(current) + 1 + len(name) > width:
            lines.append(current)
            current = name
        else:
            current = (current + " " + name).strip()
    if current:
        lines.append(current)
    return [indent + line for line in lines] if lines else [indent + "(none)"]


def one_line(text: str | None, limit: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit] + "…"


def verdict(resolved: bool | None) -> str:
    return "ungraded" if resolved is None else "PASS" if resolved else "FAIL"


def short_ts(stamp: str | None) -> str:
    return re.sub(r"\.\d+Z$", "Z", stamp) if stamp else "?"


def runner_text(lock_pid: int | None, lock_alive: bool | None) -> str:
    if lock_pid is None:
        return "no lock (runner finished)"
    return "runner pid %d %s" % (lock_pid, "alive" if lock_alive else "dead")


def extras_items(extras: dict) -> list[str]:
    """The receipt keys the runner never writes, ``key: value`` each."""
    return [
        "%s: %s" % (key, one_line(value if isinstance(value, str) else json.dumps(value), 120))
        for key, value in extras.items()
    ]


def extras_text(extras: dict) -> str:
    return "".join(" · " + item for item in extras_items(extras))


def stage_note(stage: Stage, lock_pid: int | None) -> str | None:
    """Why a stage is not complete and not running, for its line."""
    if stage.state == "errored":
        return " · cause: ".join(part for part in (stage.error, stage.cause) if part) or "stage errored"
    if stage.state == "stale":
        return runner_text(lock_pid, False) if lock_pid is not None else "no lock: the runner is gone"
    if stage.state == "unlaunched":
        return UNLAUNCHED_NOTE + extras_text(stage.extras)
    return None


def generalize_paths(text: str) -> str:
    return _PATH.sub("<path>", text)


def summary_rows(output: Path) -> list[dict]:
    summary = read_dict(output / "summary.json") or {}
    return [row for row in summary.get("attempts") or [] if isinstance(row, dict)]


# --- runs ----------------------------------------------------------------------------


@dataclass(frozen=True)
class LiveAttempt:
    task: str
    stage: str
    turns_started: int
    turns_completed: int
    steps: int
    phase: str
    last_ts: str | None


@dataclass(frozen=True)
class ErrorGroup:
    count: int
    cause: str
    stages: str


@dataclass(frozen=True)
class UnlaunchedStage:
    task: str
    stage: str
    extras: dict


@dataclass(frozen=True)
class MethodReport:
    method: str
    path: str
    tasks_total: int
    dataset: str | None
    model: str | None
    backend: str | None
    backend_source: str | None  # manifest | receipts (a manifest from before the key existed)
    store: str | None
    source_commit: str | None
    lock_pid: int | None
    lock_alive: bool | None
    metrics_present: bool
    task_counts: dict
    attempts: dict
    first_attempt: dict | None
    branch: dict | None
    resolved_tasks: tuple[str, ...]
    live: tuple[LiveAttempt, ...]
    unlaunched: tuple[UnlaunchedStage, ...]
    errors: tuple[ErrorGroup, ...]
    metrics_lines: tuple[str, ...]


@dataclass(frozen=True)
class RunsReport:
    root: str
    kind: str
    driver_banner: str | None
    methods: tuple[MethodReport, ...]
    gate: dict | None


def driver_banner(path: Path) -> str | None:
    try:
        first = path.read_text(encoding="utf-8", errors="replace").splitlines()[0]
    except (OSError, IndexError):
        return None
    match = re.match(r"^\[(.+?)\] (.+)$", first)
    return "%s %s" % (match.group(2), match.group(1)) if match else first


def gate_summary(root: Path) -> dict:
    checks = ok = failures = 0
    for line in (root / "gate.jsonl").read_text(encoding="utf-8", errors="replace").split("\n"):
        try:
            row = json.loads(line)
        except ValueError:
            continue
        if not isinstance(row, dict):
            continue
        checks += 1
        ok += row.get("ok") is True
        failures += row.get("ok") is not True
    return {"checks": checks, "ok": ok, "failures": failures}


def live_attempt(task: str, stage: Stage) -> LiveAttempt:
    output = stage.output
    if not output.is_dir():
        return LiveAttempt(task, stage.name, 0, 0, 0, "starting", None)
    evaluation = load_evaluation(output)
    open_trials = [t for t in evaluation.trials if t.phase in ("running", "grading")]
    if open_trials:
        trial = open_trials[-1]
        trajectory = read_trajectory(trial.journal) if trial.journal else None
        started, completed = turn_count(trajectory) if trajectory else (0, 0)
        phase = (
            "grading" if trial.phase == "grading" or (trajectory and trajectory.status) else "actor running"
        )
        return LiveAttempt(
            task,
            stage.name,
            started,
            completed,
            step_count(trajectory) if trajectory else 0,
            phase,
            trajectory.last_ts if trajectory else None,
        )
    if any(review["in_flight"] for review in evaluation.reviews):
        return LiveAttempt(task, stage.name, 0, 0, 0, "analyst/reviewer in flight", None)
    return LiveAttempt(
        task, stage.name, 0, 0, 0, "starting" if not evaluation.trials else "between trials", None
    )


def first_attempt_stats(baseline_root: Path | None, tasks: tuple[str, ...]) -> dict | None:
    if baseline_root is None:
        return None
    stats = {"finished": 0, "resolved": 0, "failed": 0, "retained": 0}
    for name in tasks:
        output = baseline_root / "tasks" / name / "baseline" / "attempt-1"
        receipt = read_dict(output.with_name("attempt-1.stage.json"))
        if not receipt or not receipt.get("complete"):
            continue
        stats["finished"] += 1
        rows = summary_rows(output)
        resolved = rows[0].get("resolved") if rows else None
        stats["resolved"] += resolved is True
        stats["failed"] += resolved is False
        stats["retained"] += (output / "retained-checkpoints.json").is_file()
    return stats


def metrics_lines(metrics: dict | None, method: str) -> tuple[str, ...]:
    if not metrics:
        return ()
    lines = []
    for key, m in (metrics.get("methods") or {}).items():
        failures = m.get("initial_failures_graded") or 0
        rate = m.get("initial_failure_recovery_rate")
        recovered = round(rate * failures) if isinstance(rate, (int, float)) else None
        lines.append(
            "metrics-%s.json  %s: completed %s/%s · Acc %s · Success@1 %s @5 %s @8 %s · recovery %s/%s (%s) · "
            "graded %s · positive %s"
            % (
                method,
                key,
                m.get("completed_tasks"),
                m.get("denominator_all_tasks"),
                _pct_value(m.get("Acc")),
                _pct_value(m.get("Success@1")),
                _pct_value(m.get("Success@5")),
                _pct_value(m.get("Success@8")),
                "?" if recovered is None else recovered,
                failures,
                _pct_value(rate),
                m.get("graded_trajectories"),
                m.get("successful_trajectories"),
            )
        )
    for row in metrics.get("comparison") or []:
        lines.append(
            "comparison   %s: Resolve Rate %s · Recovery %s · Steps %s · Tokens (M) %s · %s"
            % (
                row.get("method"),
                _num(row.get("Resolve Rate")),
                _num(row.get("Recovery")),
                row.get("Steps"),
                _num(row.get("Tokens (M)")),
                "complete"
                if row.get("complete")
                else "incomplete (%s/%s tasks)" % (row.get("completed_tasks"), row.get("total_tasks")),
            )
        )
    return tuple(lines)


def _pct_value(value) -> str:
    return "%.1f%%" % (100 * value) if isinstance(value, (int, float)) else "n/a"


def _num(value) -> str:
    return "%.2f" % value if isinstance(value, (int, float)) else "n/a"


def method_report(root: MethodRoot) -> MethodReport:
    states = [task_state(root, name) for name in root.tasks]
    counts = {key: 0 for key in ("complete", "errored", "in-progress", "stale", "unlaunched", "not-started")}
    for state in states:
        counts[state.status] += 1
    attempts = {
        "finished": 0,
        "resolved": 0,
        "running": 0,
        "stale": 0,
        "unlaunched": 0,
        "errored": 0,
        "mean_min": None,
        "max_min": None,
    }
    durations: list[float] = []
    live: list[LiveAttempt] = []
    unlaunched: list[UnlaunchedStage] = []
    errors: Counter = Counter()
    error_stages: dict[str, Counter] = defaultdict(Counter)
    resolved_tasks: list[str] = []
    for state in states:
        task_resolved = False
        for stage in state.stages:
            baseline_attempt = stage.name.startswith("attempt-")
            if stage.state == "complete":
                rows = summary_rows(stage.output)
                own = rows[:1] if baseline_attempt else rows[1:]
                attempts["finished"] += len(own)
                attempts["resolved"] += sum(1 for row in own if row.get("resolved") is True)
                task_resolved |= any(row.get("resolved") is True for row in own)
                if baseline_attempt:
                    durations += [stage.seconds] if isinstance(stage.seconds, (int, float)) else []
                else:
                    durations += [
                        row["seconds"] for row in own if isinstance(row.get("seconds"), (int, float))
                    ]
            elif stage.state == "running":
                attempts["running"] += 1
                live.append(live_attempt(state.name, stage))
            elif stage.state == "stale":
                attempts["stale"] += 1
            elif stage.state == "unlaunched":
                attempts["unlaunched"] += 1
                unlaunched.append(UnlaunchedStage(state.name, stage.name, stage.extras))
            elif stage.state == "errored":
                attempts["errored"] += 1
                cause = stage.cause or stage.error or "unknown error"
                errors[cause] += 1
                error_stages[cause][stage.name] += 1
        # A task can fail without any stage failing: the runner refused a retained receipt,
        # or a baseline attempt graded two rollouts. Its <method>.task.json says why.
        task_error = (state.task_json or {}).get("error")
        if task_error and not any(
            stage.state == "errored" and stage.error == task_error for stage in state.stages
        ):
            cause = generalize_paths(task_error)
            errors[cause] += 1
            error_stages[cause][f"{root.method}.task.json"] += 1
        if task_resolved:
            resolved_tasks.append(state.name)
    if durations:
        attempts["mean_min"] = sum(durations) / len(durations) / 60
        attempts["max_min"] = max(durations) / 60
    own_baseline = root.method in ("baseline", "all")
    first = first_attempt_stats(root.path if own_baseline else root.baseline_root, root.tasks)
    branch = None
    if root.method != "baseline":
        skipped = 0
        for state in states:
            methods = (state.task_json or {}).get("methods") or {}
            skipped += any(
                isinstance(m, dict) and m.get("status") == "skipped_initial_success" for m in methods.values()
            )
        branch = {
            "tasks_skipped_initial_success": skipped,
            "recovered": len(resolved_tasks),
            "failed_parents": first["failed"] if first else None,
        }
    manifest = root.manifest
    groups = tuple(
        ErrorGroup(
            count,
            cause,
            ", ".join(
                "%s: %d" % (name, n)
                for name, n in sorted(error_stages[cause].items(), key=lambda item: (len(item[0]), item[0]))
            ),
        )
        for cause, count in errors.most_common()
    )
    backend, store, backend_source = manifest.get("backend"), manifest.get("store"), "manifest"
    if backend is None:
        backend, store = receipt_backend(root)
        backend_source = "receipts" if backend else None
    return MethodReport(
        method=root.method,
        path=str(root.path),
        tasks_total=len(root.tasks),
        dataset=manifest.get("dataset"),
        model=manifest.get("model"),
        backend=backend,
        backend_source=backend_source,
        store=store,
        source_commit=manifest.get("source_commit"),
        lock_pid=root.lock_pid,
        lock_alive=root.lock_alive,
        metrics_present=root.metrics is not None,
        task_counts=counts,
        attempts=attempts,
        first_attempt=first,
        branch=branch,
        resolved_tasks=tuple(resolved_tasks),
        live=tuple(live),
        unlaunched=tuple(unlaunched),
        errors=groups,
        metrics_lines=metrics_lines(root.metrics, root.method),
    )


def runs_report(root: Path, kind: str) -> RunsReport:
    if kind == "gate":
        return RunsReport(str(root), kind, None, (), gate_summary(root))
    banner = driver_banner(root / "driver.log") if (root / "driver.log").is_file() else None
    return RunsReport(str(root), kind, banner, tuple(method_report(r) for r in method_roots(root)), None)


def render_runs(report: RunsReport) -> str:
    if report.gate is not None:
        g = report.gate
        return "gate root %s: %d checks in gate.jsonl, ok %d, failures %d (see summary.json)" % (
            report.root,
            g["checks"],
            g["ok"],
            g["failures"],
        )
    lines = []
    if report.kind == "experiment":
        lines.append(
            "experiment %s%s"
            % (report.root, "   driver.log: " + report.driver_banner if report.driver_banner else "")
        )
    for m in report.methods:
        where = ("%s %s" % (m.backend, m.store) if m.store else m.backend) or "backend unknown"
        if m.backend_source == "receipts":
            where += " (from receipts)"
        lines.append(
            "== %-10s %d tasks · %s · %s · %s · commit %s · %s · metrics-%s.json: %s"
            % (
                m.method,
                m.tasks_total,
                m.dataset,
                m.model,
                where,
                (m.source_commit or "?")[:12],
                runner_text(m.lock_pid, m.lock_alive),
                m.method,
                "written" if m.metrics_present else "not written yet",
            )
        )
        c = m.task_counts
        task_text = "%d complete · %d errored · %d in progress" % (
            c["complete"],
            c["errored"],
            c["in-progress"],
        )
        if c["stale"]:
            task_text += " · %d stale" % c["stale"]
        if c["unlaunched"]:
            task_text += " · %d unlaunched" % c["unlaunched"]
        lines.append("   %-11s %s · %d not started" % ("tasks", task_text, c["not-started"]))
        a = m.attempts
        attempt_text = "%d finished (%d resolved, %s) · %d running · %d errored" % (
            a["finished"],
            a["resolved"],
            pct(a["resolved"], a["finished"]),
            a["running"],
            a["errored"],
        )
        if a["stale"]:
            attempt_text += " · %d stale (runner dead; these stages will never finish)" % a["stale"]
        if a["unlaunched"]:
            attempt_text += " · %d unlaunched" % a["unlaunched"]
        if a["mean_min"] is not None:
            attempt_text += " · mean %.1f min · max %.1f min" % (a["mean_min"], a["max_min"])
        lines.append("   %-11s %s" % ("attempts", attempt_text))
        if m.first_attempt is not None:
            f = m.first_attempt
            lines.append(
                "   %-11s %d finished, %d resolved (%s) · %d failed parents retained (retained-checkpoints.json)"
                % (
                    "attempt-1",
                    f["finished"],
                    f["resolved"],
                    pct(f["resolved"], f["finished"]),
                    f["retained"],
                )
            )
        elif m.method != "baseline":
            lines.append("   %-11s baseline root not found (manifest.baseline_root)" % "attempt-1")
        if m.branch is not None:
            b = m.branch
            lines.append(
                "   %-11s %d skipped (attempt-1 resolved) · recovered %d/%s failed parents"
                % (
                    "branch",
                    b["tasks_skipped_initial_success"],
                    b["recovered"],
                    "?" if b["failed_parents"] is None else b["failed_parents"],
                )
            )
        if m.resolved_tasks:
            wrapped = wrap(list(m.resolved_tasks), " " * 15)
            lines.append("   %-11s %s" % ("resolved", wrapped[0].lstrip()))
            lines += wrapped[1:]
        for live in m.live:
            progress = (
                "turn %d (%d closed, %d steps)" % (live.turns_started, live.turns_completed, live.steps)
                if live.turns_started
                else "no journal yet"
            )
            lines.append(
                "   %-11s %s   %s   %s   %s   last event %s"
                % ("in progress", live.task, live.stage, progress, live.phase, short_ts(live.last_ts))
            )
        if m.unlaunched:
            lines.append(
                "   %-11s %d receipts without a log or output directory (this runner never launched them)"
                % ("unlaunched", len(m.unlaunched))
            )
            notes = Counter(
                " · ".join(extras_items(u.extras)) or "receipt carries no extra keys" for u in m.unlaunched
            )
            lines += ["   %-11s %3d× %s" % ("", count, note) for note, count in notes.most_common()]
            lines += wrap(["%s/%s" % (u.task, u.stage) for u in m.unlaunched], " " * 15)
        for index, group in enumerate(m.errors):
            lines.append(
                "   %-11s %3d× %s   (%s)"
                % (
                    "errors" if index == 0 else "",
                    group.count,
                    one_line(group.cause, NOTE_WIDTH),
                    group.stages,
                )
            )
        lines += ["   " + line for line in m.metrics_lines]
    return "\n".join(lines)


# --- task ----------------------------------------------------------------------------


@dataclass(frozen=True)
class AttemptView:
    name: str
    stage_state: str | None
    phase: str | None
    actor_status: str | None
    resolved: bool | None
    reward: float | None
    grade_detail: str
    turns: int | None
    reviewer_turns: int | None
    turns_completed: int | None
    steps: int | None
    minutes: float | None
    snapshots: int | None
    final_snapshot_id: str | None
    sandbox_id: str | None
    backend: str | None
    journal: str | None
    last_ts: str | None
    parent: bool
    retained_points: int | None
    retention_error: str | None
    error: str | None


@dataclass(frozen=True)
class BaselineView:
    path: str | None
    task_status: str | None
    task_error: str | None
    attempts: tuple[AttemptView, ...]


@dataclass(frozen=True)
class ReportView:
    failure_reason: str
    lesson: str
    salvage: str
    candidates: tuple[int, ...]
    steps: int | None
    available_steps: tuple[int, ...]
    review_file: str | None
    minutes: float | None
    in_flight: bool
    started_at: float | None


@dataclass(frozen=True)
class ReviewerAttemptView:
    attempt: int | None
    status: str | None
    validation_error: str | None
    review_file: str | None
    minutes: float | None
    in_flight: bool
    branches: int | None


@dataclass(frozen=True)
class BranchView:
    run_name: str
    base: str
    branch_step: int | None
    branch_turn: int | None
    snapshot_id: str | None
    conversation_cut: str | None
    guidance: str | None
    calls: tuple[tuple[str, str], ...]
    hint: str | None
    trial: AttemptView | None


@dataclass(frozen=True)
class RoundView:
    round: int
    limit: int | None
    count_mode: str | None
    reports: dict
    reviewer_attempts: tuple[ReviewerAttemptView, ...]
    branches: tuple[BranchView, ...]
    validation_error: str | None


@dataclass(frozen=True)
class MethodView:
    method: str
    path: str | None
    stage_state: str | None
    complete: bool | None
    resolved: bool | None
    guidance: str | None
    caps: tuple[int, ...]
    rounds_declared: int | None
    resolved_round: int | None
    stop_reason: str | None
    skipped: bool
    error: str | None
    rounds: tuple[RoundView, ...]


@dataclass(frozen=True)
class TaskTree:
    task_name: str
    dataset: str | None
    model: str | None
    backend: str | None
    store: str | None
    baseline: BaselineView
    methods: tuple[MethodView, ...]
    notes: tuple[str, ...]


def attempt_view(
    name: str,
    *,
    stage: Stage | None,
    trial: Trial | None,
    trajectory: Trajectory | None,
    row: dict | None,
    parent: bool = False,
    retained_points: int | None = None,
    retention_error: str | None = None,
    lock_pid: int | None = None,
) -> AttemptView:
    grade = trial.grade if trial else None
    resolved = (
        row.get("resolved")
        if row and isinstance(row.get("resolved"), bool)
        else (grade.resolved if grade else None)
    )
    reward = row.get("reward") if row and row.get("reward") is not None else (grade.reward if grade else None)
    error = stage_note(stage, lock_pid) if stage is not None else None
    if error is None and grade is not None and grade.exception:
        error = "Harbor raised %s" % grade.exception
    if row and isinstance(row.get("seconds"), (int, float)):
        duration = row["seconds"] / 60
    elif stage is not None and isinstance(stage.seconds, (int, float)):
        duration = stage.seconds / 60
    else:
        duration = journal_minutes(trajectory) if trajectory else None
    final = (
        next((c.snapshot_id for c in reversed(trajectory.captures) if c.final and c.snapshot_id), None)
        if trajectory
        else None
    )
    return AttemptView(
        name=name,
        stage_state=stage.state if stage else None,
        phase=trial.phase if trial else None,
        actor_status=trajectory.status if trajectory else (row or {}).get("status"),
        resolved=resolved,
        reward=reward,
        grade_detail=grade.detail if grade else "",
        turns=len(trajectory.turns) if trajectory else None,
        reviewer_turns=sum(1 for t in trajectory.turns if t.source == "reviewer") if trajectory else None,
        turns_completed=turn_count(trajectory)[1] if trajectory else None,
        steps=step_count(trajectory) if trajectory else None,
        minutes=duration,
        snapshots=sum(1 for c in trajectory.captures if c.kind == "disk" and c.captured and c.snapshot_id)
        if trajectory
        else None,
        final_snapshot_id=final,
        sandbox_id=(
            trajectory.sandbox_id
            if trajectory and trajectory.sandbox_id
            else (trial.receipt or {}).get("sandbox_id")
            if trial
            else None
        ),
        backend=trial_backend(trial, trajectory) if trial else (trajectory.backend if trajectory else None),
        journal=str(trial.journal) if trial and trial.journal else None,
        last_ts=trajectory.last_ts if trajectory else None,
        parent=parent,
        retained_points=retained_points,
        retention_error=retention_error,
        error=error,
    )


def _attempt_dirs(baseline_dir: Path) -> list[str]:
    names = {p.name for p in baseline_dir.glob("attempt-*") if p.is_dir()}
    names |= {p.name[: -len(".stage.json")] for p in baseline_dir.glob("attempt-*.stage.json")}
    return sorted(names, key=lambda n: int(re.findall(r"\d+", n)[-1]) if re.findall(r"\d+", n) else 0)


def baseline_view(
    baseline_dir: Path | None,
    task_json: dict | None,
    lock_alive: bool | None,
    lock_pid: int | None,
    trajectories: dict,
) -> BaselineView:
    status = error = None
    if task_json is not None:
        status = (
            "complete"
            if task_json.get("complete")
            else "errored"
            if task_json.get("error")
            else "in-progress"
        )
        error = task_json.get("error")
    if baseline_dir is None or not baseline_dir.is_dir():
        return BaselineView(str(baseline_dir) if baseline_dir else None, status, error, ())
    views = []
    for name in _attempt_dirs(baseline_dir):
        receipt = baseline_dir / (name + ".stage.json")
        stage = load_stage(receipt, lock_alive) if receipt.is_file() else None
        output = baseline_dir / name
        evaluation = load_evaluation(output) if (output / "summary.json").is_file() else None
        trial = evaluation.trials[0] if evaluation and evaluation.trials else None
        if trial is None and (output / "trials").is_dir():
            trials = [p for p in sorted((output / "trials").iterdir()) if p.is_dir()]
            trial = load_trial(trials[0]) if trials else None
        trajectory = read_trajectory(trial.journal) if trial and trial.journal else None
        rows = summary_rows(output)
        parent = name == "attempt-1"
        if parent and trajectory is not None:
            trajectories["parent"] = trajectory
        retained = evaluation.retained if evaluation else None
        views.append(
            attempt_view(
                name,
                stage=stage,
                trial=trial,
                trajectory=trajectory,
                row=rows[0] if rows else None,
                parent=parent,
                retained_points=len(retained.get("points") or []) if retained else None,
                retention_error=(task_json or {}).get("retention_error") if parent else None,
                lock_pid=lock_pid,
            )
        )
    return BaselineView(str(baseline_dir), status, error, tuple(views))


def _command_of_tool_call(call: dict) -> str:
    try:
        arguments = json.loads((call.get("function") or {}).get("arguments") or "{}")
    except ValueError:
        return str((call.get("function") or {}).get("arguments") or "")
    return str(arguments.get("command") or "") if isinstance(arguments, dict) else ""


def method_view(
    method: str, evaluation: Evaluation, stage: Stage | None, trajectories: dict, lock_pid: int | None
) -> MethodView:
    s = evaluation.summary
    rows = {str(r.get("name")): r for r in s.get("attempts") or [] if isinstance(r, dict)}
    analysts = [r for r in evaluation.reviews if r["kind"] == "analyst"]
    reviewers = [r for r in evaluation.reviews if r["kind"] == "reviewer"]
    assigned: dict[str, dict] = {}
    used: set[int] = set()
    rounds = []
    for plan in evaluation.plans:
        reports = {}
        for name, report in plan.reports.items():
            if not isinstance(report, dict):
                continue
            if name not in assigned and analysts:
                assigned[name] = analysts.pop(0)
            review = assigned.get(name)
            reports[name] = ReportView(
                failure_reason=str(report.get("failure_reason") or ""),
                lesson=str(report.get("lesson") or ""),
                salvage=str(report.get("salvage") or ""),
                candidates=tuple(
                    c["step"]
                    for c in report.get("branch_candidates") or []
                    if isinstance(c, dict) and isinstance(c.get("step"), int)
                ),
                steps=report.get("steps"),
                available_steps=tuple(plan.available_steps.get(name) or ()),
                review_file=review["file"] if review else None,
                minutes=review["seconds"] / 60
                if review and isinstance(review.get("seconds"), (int, float))
                else None,
                in_flight=bool(review and review["in_flight"]),
                started_at=review["started_at"] if review else None,
            )
        attempts = []
        for attempt in plan.review_attempts:
            match = None
            if attempt["response"] is not None:
                match = next(
                    (r for r in reviewers if id(r) not in used and r["text"] == attempt["response"]), None
                )
            elif attempt["status"] == "requesting":
                match = next((r for r in reviewers if id(r) not in used and r["in_flight"]), None)
            if match is not None:
                used.add(id(match))
            attempts.append(
                ReviewerAttemptView(
                    attempt=attempt["attempt"],
                    status=attempt["status"],
                    validation_error=attempt["validation_error"]
                    or attempt["request_error"]
                    or attempt["state_error"],
                    review_file=match["file"] if match else None,
                    minutes=match["seconds"] / 60
                    if match and isinstance(match.get("seconds"), (int, float))
                    else None,
                    in_flight=bool(match and match["in_flight"]),
                    branches=attempt["branches"],
                )
            )
        branches = []
        for selected in plan.selected:
            run_name = str(selected.get("run_name"))
            base = str(selected.get("base"))
            step = selected.get("branch_step")
            trial = next((t for t in evaluation.trials if t.name.endswith("-" + run_name)), None)
            trajectory = read_trajectory(trial.journal) if trial and trial.journal else None
            if trajectory is not None:
                trajectories[run_name] = trajectory
            base_trajectory = trajectories.get(base)
            branch_turn = (
                next((t.number for t in base_trajectory.turns if t.step == step), None)
                if base_trajectory and isinstance(step, int)
                else None
            )
            turn = selected.get("assistant_turn") or {}
            calls = tuple(
                (str(c.get("id")), _command_of_tool_call(c))
                for c in turn.get("tool_calls") or []
                if isinstance(c, dict)
            )
            branches.append(
                BranchView(
                    run_name=run_name,
                    base=base,
                    branch_step=step,
                    branch_turn=branch_turn,
                    snapshot_id=selected.get("snapshot_id"),
                    conversation_cut=selected.get("conversation_cut"),
                    guidance=selected.get("branch_guidance") or plan.guidance,
                    calls=calls,
                    hint=selected.get("hint"),
                    trial=attempt_view(
                        run_name,
                        stage=None,
                        trial=trial,
                        trajectory=trajectory,
                        row=rows.get(run_name),
                        lock_pid=lock_pid,
                    )
                    if trial
                    else None,
                )
            )
        rounds.append(
            RoundView(
                round=plan.round,
                limit=plan.limit,
                count_mode=plan.count_mode,
                reports=reports,
                reviewer_attempts=tuple(attempts),
                branches=tuple(branches),
                validation_error=plan.validation_error,
            )
        )
    resolved_round = min(
        (r.get("round") for r in rows.values() if r.get("resolved") is True and r.get("round")), default=None
    )
    return MethodView(
        method=method,
        path=str(evaluation.path),
        stage_state=stage.state if stage else None,
        complete=s.get("complete"),
        resolved=s.get("resolved"),
        guidance=s.get("branch_guidance"),
        caps=tuple(s.get("branch_caps") or ()),
        rounds_declared=s.get("rounds"),
        resolved_round=resolved_round,
        stop_reason=s.get("stop_reason"),
        skipped=False,
        error=stage.error if stage and stage.state == "errored" else None,
        rounds=tuple(rounds),
    )


@dataclass(frozen=True)
class _MethodSpec:
    method: str
    output: Path
    task_json: dict | None
    stage: Stage | None
    evaluation: Evaluation | None


def _locate(
    root: Path, kind: str, task: str | None
) -> tuple[Path | None, dict | None, list[_MethodSpec], dict, list[str], str, int | None, bool | None]:
    """Where a task's baseline directory and method outputs are, for every accepted ROOT kind."""
    notes: list[str] = []
    specs: list[_MethodSpec] = []
    baseline_dir = base_task_json = None
    meta: dict = {}
    lock_pid = lock_alive = None
    name = task or ""
    if kind in ("experiment", "method"):
        roots = method_roots(root)
        if not any(name in r.tasks for r in roots):
            raise LookupError(
                "%s is not a task of %s (not in the tasks of %s)"
                % (name, root, ", ".join("manifest-%s.json" % r.method for r in roots))
            )
        base_root = next((r for r in roots if r.method in ("baseline", "all")), None)
        if base_root is not None:
            baseline_dir = base_root.path / "tasks" / name / "baseline"
            base_task_json = read_dict(base_root.path / "tasks" / name / f"{base_root.method}.task.json")
            lock_pid, lock_alive = base_root.lock_pid, base_root.lock_alive
        elif roots[0].baseline_root is not None:
            baseline_dir = roots[0].baseline_root / "tasks" / name / "baseline"
            base_task_json = read_dict(roots[0].baseline_root / "tasks" / name / "baseline.task.json")
        else:
            notes.append(
                "baseline root %s not found (manifest.baseline_root; moved?)"
                % roots[0].manifest.get("baseline_root")
            )
        meta = dict(roots[0].manifest)
        for r in roots:
            for m in METHODS[1:] if r.method == "all" else [r.method] if r.method != "baseline" else []:
                output = r.path / "tasks" / name / m
                receipt = output.with_name(m + ".stage.json")
                specs.append(
                    _MethodSpec(
                        m,
                        output,
                        read_dict(r.path / "tasks" / name / f"{r.method}.task.json"),
                        load_stage(receipt, r.lock_alive) if receipt.is_file() else None,
                        load_evaluation(output) if (output / "summary.json").is_file() else None,
                    )
                )
                if lock_pid is None:
                    lock_pid, lock_alive = r.lock_pid, r.lock_alive
        if not specs and kind == "experiment":
            notes.append(
                "sprout   no sprout stage under this root (no sprout/ method root; pass --with "
                "<evaluate output> to attach one)"
            )
    elif kind == "task":
        name = root.name
        enclosing = root.parent.parent
        method_root = load_method_root(enclosing) if is_method_root(enclosing) else None
        if method_root is not None:
            meta = dict(method_root.manifest)
            lock_pid, lock_alive = method_root.lock_pid, method_root.lock_alive
        if (root / "baseline").is_dir():
            baseline_dir = root / "baseline"
            base_task_json = read_dict(root / "baseline.task.json") or read_dict(root / "all.task.json")
        elif method_root is not None and method_root.baseline_root is not None:
            baseline_dir = method_root.baseline_root / "tasks" / name / "baseline"
            base_task_json = read_dict(method_root.baseline_root / "tasks" / name / "baseline.task.json")
        for m in METHODS[1:]:
            output = root / m
            receipt = root / (m + ".stage.json")
            if receipt.is_file() or output.is_dir():
                task_json = read_dict(root / f"{method_root.method if method_root else m}.task.json")
                specs.append(
                    _MethodSpec(
                        m,
                        output,
                        task_json,
                        load_stage(receipt, lock_alive) if receipt.is_file() else None,
                        load_evaluation(output) if (output / "summary.json").is_file() else None,
                    )
                )
    elif kind == "evaluate":
        evaluation = load_evaluation(root)
        summary_task = str(evaluation.summary.get("task") or "")
        if task and summary_task and task not in (summary_task, summary_task.split("/")[-1]):
            notes.append(
                "TASK %s is not this output's task (summary.task %s); showing the output's"
                % (task, summary_task)
            )
        baseline_dir = baseline_dir_of(evaluation)
        if baseline_dir is None and root.parent.name == "baseline" and root.name.startswith("attempt-"):
            baseline_dir = root.parent  # a baseline attempt still running: no journal in summary.json yet
        meta = {"model": evaluation.summary.get("model"), "backend": evaluation.summary.get("backend")}
        # A runner stage's output sits in <method root>/tasks/<task>/<method> beside its receipt.
        enclosing = root.parents[2] if len(root.parents) > 2 else None
        method_root = load_method_root(enclosing) if enclosing and is_method_root(enclosing) else None
        if method_root is not None:
            meta = {**method_root.manifest, **meta}
            lock_pid, lock_alive = method_root.lock_pid, method_root.lock_alive
        if baseline_dir is not None:
            base_task_json = read_dict(baseline_dir.parent / "baseline.task.json")
            if is_method_root(baseline_dir.parents[2]):
                base_root = load_method_root(baseline_dir.parents[2])
                meta = {**base_root.manifest, **meta}
                lock_pid, lock_alive = base_root.lock_pid, base_root.lock_alive
        name = baseline_dir.parent.name if baseline_dir is not None else summary_task.split("/")[-1]
        method = evaluation.summary.get("method")
        if method not in (None, "baseline"):
            receipt = root.with_name(root.name + ".stage.json")
            stage = load_stage(receipt, lock_alive) if method_root is not None and receipt.is_file() else None
            task_json = (
                read_dict(root.parent / f"{method_root.method}.task.json")
                if method_root is not None
                else None
            )
            specs.append(_MethodSpec(str(method), root, task_json, stage, evaluation))
    return baseline_dir, base_task_json, specs, meta, notes, name, lock_pid, lock_alive


def task_tree(root: Path, kind: str, task: str | None, extra: list[Path]) -> TaskTree:
    baseline_dir, base_task_json, specs, meta, notes, name, lock_pid, lock_alive = _locate(root, kind, task)
    trajectories: dict[str, Trajectory] = {}
    baseline = baseline_view(baseline_dir, base_task_json, lock_alive, lock_pid, trajectories)
    parent_journal = next((Path(a.journal) for a in baseline.attempts if a.parent and a.journal), None)
    for extra_dir in extra:
        if detect(extra_dir) != "evaluate":
            notes.append("--with %s is not an evaluate output: skipped" % extra_dir)
            continue
        evaluation = load_evaluation(extra_dir)
        recorded = evaluation.attempts[0].journal if evaluation.attempts else None
        if parent_journal is None or recorded is None or recorded.resolve() != parent_journal.resolve():
            notes.append("--with %s does not branch from this task's attempt-1: skipped" % extra_dir)
            continue
        specs.append(
            _MethodSpec(str(evaluation.summary.get("method") or "?"), extra_dir, None, None, evaluation)
        )
    methods = []
    for spec in specs:
        methods_json = ((spec.task_json or {}).get("methods") or {}).get(spec.method) or {}
        if spec.evaluation is not None:
            methods.append(method_view(spec.method, spec.evaluation, spec.stage, trajectories, lock_pid))
        elif methods_json.get("status") == "skipped_initial_success" or (
            spec.stage is None and (spec.task_json or {}).get("initial_resolved") is True
        ):
            methods.append(
                MethodView(spec.method, None, None, None, None, None, (), None, None, None, True, None, ())
            )
        elif spec.stage is not None:
            methods.append(
                MethodView(
                    spec.method,
                    str(spec.output),
                    spec.stage.state,
                    None,
                    None,
                    None,
                    (),
                    None,
                    None,
                    None,
                    False,
                    stage_note(spec.stage, lock_pid),
                    (),
                )
            )
        else:
            notes.append("%s   no %s stage under this root" % (spec.method, spec.method))
    first = baseline.attempts[0] if baseline.attempts else None
    task_name = next(
        (
            str(m.evaluation.summary.get("task"))
            for m in specs
            if m.evaluation is not None and m.evaluation.summary.get("task")
        ),
        None,
    )
    if (
        task_name is None
        and baseline_dir is not None
        and (baseline_dir / "attempt-1" / "summary.json").is_file()
    ):
        task_name = str((read_dict(baseline_dir / "attempt-1" / "summary.json") or {}).get("task") or name)
    return TaskTree(
        task_name=task_name or name,
        dataset=meta.get("dataset") or (base_task_json or {}).get("dataset"),
        model=meta.get("model"),
        backend=meta.get("backend") or (first.backend if first else None),
        store=meta.get("store") or next((t.store for t in trajectories.values() if t.store), None),
        baseline=baseline,
        methods=tuple(methods),
        notes=tuple(notes),
    )


def _tree(nodes: list, prefix: str = "") -> list[str]:
    lines = []
    for index, (text, children) in enumerate(nodes):
        last = index == len(nodes) - 1
        lines.append(prefix + ("└── " if last else "├── ") + text)
        lines += _tree(children, prefix + ("    " if last else "│   "))
    return lines


def attempt_text(v: AttemptView, label: str, *, branch: bool = False) -> str:
    live = v.stage_state in ("running", "stale") or (
        v.stage_state is None and v.phase in ("running", "grading")
    )
    if v.stage_state == "unlaunched":
        return "%s  unlaunched   %s" % (label, v.error)
    if v.stage_state == "errored" and v.journal is None:
        return "%s  errored   %s · %s · no trial directory" % (
            label,
            one_line(v.error, NOTE_WIDTH),
            mins(v.minutes * 60 if v.minutes is not None else None),
        )
    if live and v.resolved is None:
        state = "stale (%s)" % v.error if v.stage_state == "stale" else "running"
        progress = (
            "turn %d (%d closed, %d steps)" % (v.turns, v.turns_completed, v.steps)
            if v.turns is not None
            else "no journal yet"
        )
        phase = "grading" if v.phase == "grading" or v.actor_status else "actor running"
        return "%s  %s   %s   %s   last event %s" % (label, state, progress, phase, short_ts(v.last_ts))
    parts = [
        label,
        (v.stage_state or v.phase or "?"),
        v.actor_status or "?",
        verdict(v.resolved),
        "reward %s" % ("?" if v.reward is None else v.reward),
    ]
    if v.grade_detail:
        parts.append(v.grade_detail)
    if v.error and v.stage_state != "complete":
        parts.append(one_line(v.error, NOTE_WIDTH))
    turns = ""
    if v.turns is not None:
        turns = (
            "%3d turns (%d reviewer + %d actor)" % (v.turns, v.reviewer_turns, v.turns - v.reviewer_turns)
            if branch
            else "%3d turns" % v.turns
        )
    text = "  ".join(parts)
    if v.turns is not None:
        snaps = v.snapshots or 0
        text += "   %s  %3d steps  %4.1f min  %3d snapshot%s  final %s" % (
            turns,
            v.steps,
            v.minutes or 0.0,
            snaps,
            "s" if snaps != 1 else " ",
            v.final_snapshot_id,
        )
    if branch and v.sandbox_id:
        text += "  sandbox %s" % v.sandbox_id
    if v.parent:
        text += "  PARENT"
        if v.retained_points is not None:
            text += " · branchable: %d points retained" % v.retained_points
        elif v.retention_error:
            text += " · not retained: %s" % v.retention_error
    return text


def method_header(m: MethodView) -> str:
    if m.skipped:
        return "%s   skipped (attempt-1 resolved)" % m.method
    if m.stage_state is not None and m.complete is None:
        return "%s %s   stage %s%s" % (
            m.method,
            m.path,
            m.stage_state,
            " · %s" % one_line(m.error, NOTE_WIDTH) if m.error else "",
        )
    state = "stage %s" % m.stage_state if m.stage_state else "complete" if m.complete else "in progress"
    text = "%s %s   %s · %s · guidance %s · caps %s" % (
        m.method,
        m.path,
        state,
        "resolved" if m.resolved else "unresolved",
        m.guidance,
        list(m.caps),
    )
    run = len(m.rounds)
    text += " · rounds %s (%d run%s)" % (
        m.rounds_declared,
        run,
        ": resolved in round %d" % m.resolved_round if m.resolved_round else "",
    )
    if m.stop_reason:
        text += " · stopped: %s" % one_line(m.stop_reason, 120)
    return text


def round_nodes(r: RoundView) -> list:
    nodes = []
    for name, report in r.reports.items():
        where = (
            "[%s in flight since %s]" % (report.review_file, short_ts(_epoch_to_text(report.started_at)))
            if report.in_flight
            else "[%s %s]" % (report.review_file, mins(report.minutes * 60 if report.minutes else None))
            if report.review_file
            else "[no review file]"
        )
        available = report.available_steps
        span = " (%d..%d)" % (available[0], available[-1]) if available else ""
        head = "analyst %s  %s  %s steps, %d available%s" % (name, where, report.steps, len(available), span)
        children = [
            ("failure: " + one_line(report.failure_reason, 150), []),
            ("lesson:  " + one_line(report.lesson, 150), []),
            ("salvage: " + one_line(report.salvage, 150), []),
            ("candidates: steps " + (", ".join(str(s) for s in report.candidates) or "none"), []),
        ]
        nodes.append((head, children))
    for a in r.reviewer_attempts:
        where = (
            "[%s in flight]" % a.review_file
            if a.in_flight
            else "[%s %s]" % (a.review_file, mins(a.minutes * 60 if a.minutes else None))
            if a.review_file
            else "[no review file]"
        )
        tail = (
            "%d branches" % a.branches
            if a.status == "validated" and a.branches is not None
            else one_line(a.validation_error, 160)
            if a.validation_error
            else ""
        )
        nodes.append(("reviewer attempt %s  %-16s  %s  %s" % (a.attempt, a.status, where, tail), []))
    for b in r.branches:
        turn = " (turn %d)" % b.branch_turn if b.branch_turn is not None else ""
        head = "%s  %s@%s%s  snapshot %s  cut %s" % (
            b.run_name,
            b.base,
            b.branch_step,
            turn,
            b.snapshot_id,
            (b.conversation_cut or "")[:8],
        )
        if b.calls:
            guidance = (
                "assistant-turn (reviewer) %d bash calls: %s"
                % (len(b.calls), " | ".join(summarize_command(c) for _, c in b.calls)),
                [],
            )
        elif b.hint:
            guidance = ("hint: " + one_line(b.hint, 120), [])
        else:
            guidance = ("no guidance (retained prefix only)", [])
        trial = (attempt_text(b.trial, "trial", branch=True), []) if b.trial else ("trial  not started", [])
        nodes.append((head, [guidance, trial]))
    if not r.branches and r.validation_error:
        nodes.append(("reviewer selected no branches: " + one_line(r.validation_error, 200), []))
    return nodes


def _epoch_to_text(epoch: float | None) -> str | None:
    if not isinstance(epoch, (int, float)):
        return None
    return datetime.fromtimestamp(epoch, UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def render_task(tree: TaskTree) -> str:
    where = ("%s %s" % (tree.backend, tree.store) if tree.store else tree.backend) or "backend unknown"
    lines = ["task %s · %s · %s · %s" % (tree.task_name, tree.dataset, tree.model, where)]
    b = tree.baseline
    status = (
        "task: %s%s" % (b.task_status, " (%s)" % b.task_error if b.task_error else "")
        if b.task_status
        else "task: no task.json"
    )
    lines.append("baseline %s   %s" % (b.path or "(not found)", status))
    lines += _tree([(attempt_text(a, a.name), []) for a in b.attempts])
    for m in tree.methods:
        lines.append(method_header(m))
        lines += _tree(
            [
                (
                    "round %d  limit %s · %s · %d branches selected"
                    % (r.round, r.limit, r.count_mode, len(r.branches)),
                    round_nodes(r),
                )
                for r in m.rounds
            ]
        )
    lines += list(tree.notes)
    return "\n".join(lines)


def _node_id(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _leaf_label(v: AttemptView | None, title: str) -> tuple[str, str | None]:
    """(label, class) of a leaf node: the verdict colours it."""
    if v is None:
        return "%s<br/>not started" % title, None
    if v.stage_state == "unlaunched":
        return "%s<br/>unlaunched: %s" % (title, one_line(v.error, 40)), "err"
    if v.stage_state == "errored" or (v.error and v.resolved is None):
        return "%s<br/>errored: %s" % (title, one_line(v.error, 40)), "err"
    if v.resolved is None:
        progress = "turn %s · %s steps" % (v.turns, v.steps) if v.turns is not None else "no journal"
        return "%s<br/>running · %s" % (title, progress), None
    stats = "%s turns · %s steps · %s" % (
        v.turns,
        v.steps,
        mins(v.minutes * 60 if v.minutes is not None else None),
    )
    return "%s<br/>%s<br/>%s reward %s" % (
        title,
        stats,
        verdict(v.resolved),
        v.reward,
    ), "pass" if v.resolved else "fail"


def render_mermaid(tree: TaskTree) -> str:
    branches = [(m, r, b) for m in tree.methods for r in m.rounds for b in r.branches]
    by_base: dict[str, list] = defaultdict(list)
    for item in branches:
        by_base[item[2].base].append(item)
    attempts = {a.name: a for a in tree.baseline.attempts}
    lines = ["graph LR"]
    classes: dict[str, list[str]] = defaultdict(list)
    bases = ["parent"] + [b.run_name for _, _, b in branches if b.run_name in by_base]
    for base in bases:
        items = by_base.get(base)
        if not items and base != "parent":
            continue
        base_id = "parent" if base == "parent" else _node_id(base)
        view = attempts.get("attempt-1") if base == "parent" else None
        previous = 'parent(["parent · attempt-1"])' if base == "parent" else base_id
        steps = sorted({b.branch_step for _, _, b in items or [] if isinstance(b.branch_step, int)})
        for step in steps:
            branch = next(b for _, _, b in items if b.branch_step == step)
            turn = " (turn %d)" % branch.branch_turn if branch.branch_turn is not None else ""
            node = "%s_s%d" % (base_id, step)
            lines.append('  %s --> %s["step %d%s<br/>%s"]' % (previous, node, step, turn, branch.snapshot_id))
            previous = node
        if base == "parent" and view is not None and view.turns is not None:
            label, cls = (
                "step %s · final<br/>%s reward %s<br/>%s"
                % (view.steps, verdict(view.resolved), view.reward, view.final_snapshot_id),
                None if view.resolved is None else "pass" if view.resolved else "fail",
            )
            lines.append('  %s --> parent_end["%s"]' % (previous, label))
            if cls:
                classes[cls].append("parent_end")
        elif base == "parent" and view is not None:  # no journal: errored, unlaunched or starting
            label, cls = _leaf_label(view, "attempt-1")
            lines.append('  %s --> parent_end["%s"]' % (previous, label))
            if cls:
                classes[cls].append("parent_end")
        for method, round_view, branch in items or []:
            node = _node_id(branch.run_name)
            title = "%s<br/>round %d · %s" % (branch.run_name, round_view.round, branch.guidance)
            label, cls = _leaf_label(branch.trial, title)
            lines.append('  %s_s%s --> %s["%s"]' % (base_id, branch.branch_step, node, label))
            if cls:
                classes[cls].append(node)
    independent = [a for a in tree.baseline.attempts if a.name != "attempt-1"]
    if independent:
        lines.append("  subgraph independent baseline attempts")
        for a in independent:
            node = _node_id(a.name)
            label, cls = _leaf_label(a, a.name)
            lines.append('    %s["%s"]' % (node, label))
            if cls:
                classes[cls].append(node)
        lines.append("  end")
    lines.append(
        "  classDef pass fill:#dff5df,stroke:#2e7d32; classDef fail fill:#fbe0e0,stroke:#c62828; "
        "classDef err fill:#eeeeee,stroke:#616161,stroke-dasharray:4 2;"
    )
    assignments = ["class %s %s;" % (",".join(nodes), cls) for cls, nodes in classes.items() if nodes]
    if assignments:
        lines.append("  " + " ".join(assignments))
    return "\n".join(lines)


# --- snapshots -----------------------------------------------------------------------


@dataclass(frozen=True)
class Selection:
    snapshot_id: str | None
    label: str
    source: str


@dataclass(frozen=True)
class SnapshotRecord:
    capture: Capture
    selected: tuple[str, ...]
    store: str


@dataclass(frozen=True)
class JournalSnapshots:
    run_id: str
    journal: str
    backend: str | None
    sandbox_id: str | None
    store: str | None
    store_note: str
    origin: dict | None
    note: str | None
    records: tuple[SnapshotRecord, ...]
    totals: dict


@dataclass(frozen=True)
class SnapshotsReport:
    store: str | None
    store_source: str | None
    selections: tuple[Selection, ...]
    journals: tuple[JournalSnapshots, ...]
    totals: dict
    warnings: tuple[str, ...]


def find_journals(target: Path, kind: str, task: str | None) -> list[tuple[Path, str | None]]:
    """(journal, note) for every journal a target names; a sprout output's recorded parent is marked."""
    if kind == "journal":
        return [(target, None)]
    if kind == "trial":
        journal = target / "agent" / "trajectory.jsonl"
        return [(journal, None)] if journal.is_file() else []
    if kind == "evaluate":
        found = [(p, None) for p in sorted(target.glob("trials/*/agent/trajectory.jsonl"))]
        inside = {p.resolve() for p, _ in found}
        for row in attempt_journals(target):
            if row.resolve() not in inside:
                found.insert(0, (row, "recorded parent, outside this output"))
                inside.add(row.resolve())
        return found
    if kind == "gate":
        return []
    roots = (
        [target]
        if kind in ("task",)
        else [target / "tasks"]
        if kind == "method"
        else [
            child / "tasks" for child in sorted(target.iterdir()) if child.is_dir() and is_method_root(child)
        ]
    )
    groups: list[tuple[Path, list[Path]]] = []
    for root in roots:
        if not root.is_dir():
            continue
        task_dirs = (
            [root]
            if kind == "task"
            else [p for p in sorted(root.iterdir()) if p.is_dir() and (task is None or p.name == task)]
        )
        groups += [
            (task_dir, sorted(task_dir.glob("**/trials/*/agent/trajectory.jsonl"))) for task_dir in task_dirs
        ]
    # A method stage's parent is a baseline attempt, in another method root when the
    # baseline was shared: its snapshots are the branch points, so it is listed too.
    inside = {p.resolve() for _, own in groups for p in own}
    found = []
    for task_dir, own in groups:
        for method in METHODS[1:]:
            if (task_dir / method / "summary.json").is_file():
                for row in attempt_journals(task_dir / method):
                    if row.resolve() not in inside:
                        found.append((row, "recorded parent, outside this output"))
                        inside.add(row.resolve())
        found += [(p, None) for p in own]
    return found


def attempt_journals(output: Path) -> list[Path]:
    rows = summary_rows(output)
    found = []
    for row in rows:
        recorded = row.get("journal")
        path = relocate(recorded, output / "summary.json") if recorded else None
        if path is not None:
            found.append(path)
    return found


def _plan_roots(target: Path, kind: str) -> list[Path]:
    if kind == "evaluate":
        return [target]
    if kind == "trial":
        return [target.parent.parent]
    if kind == "journal":  # <output>/trials/<trial>/agent/trajectory.jsonl
        return [target.parents[3]] if len(target.parents) > 3 and target.parents[2].name == "trials" else []
    return [target]


def find_selections(target: Path, kind: str) -> list[Selection]:
    selections: list[Selection] = []
    for root in _plan_roots(target, kind):
        for plan_path in sorted(root.rglob("plan-round*.json")):
            plan = read_dict(plan_path) or {}
            for b in plan.get("selected_branches") or []:
                if isinstance(b, dict) and b.get("snapshot_id"):
                    selections.append(
                        Selection(
                            str(b["snapshot_id"]),
                            "%s, %s@%s" % (b.get("run_name"), b.get("base"), b.get("branch_step")),
                            plan_path.name,
                        )
                    )
        for context in sorted(root.rglob("contexts/*.json")):
            body = read_dict(context) or {}
            if body.get("snapshot_id") and not any(
                s.snapshot_id == body["snapshot_id"] and s.label.startswith(context.stem) for s in selections
            ):
                selections.append(
                    Selection(
                        str(body["snapshot_id"]),
                        "%s, step %s" % (context.stem, body.get("step")),
                        "contexts/" + context.name,
                    )
                )
        for plan_path in sorted(root.rglob("plan.json")):
            plan = read_dict(plan_path) or {}
            parent = plan.get("shared_parent")
            journal = relocate(parent, plan_path) if parent else None
            captures = (
                {c.step: c.snapshot_id for c in read_trajectory(journal).captures if c.captured}
                if journal
                else {}
            )
            for index, chosen in enumerate(plan.get("selected") or [], 1):
                if isinstance(chosen, dict) and isinstance(chosen.get("step"), int):
                    selections.append(
                        Selection(
                            captures.get(chosen["step"]),
                            "b%02d-step%d" % (index, chosen["step"]),
                            plan_path.name,
                        )
                    )
    return selections


def store_status(store: Path | None, snapshot_id: str) -> str:
    if store is None:
        return "not checked"
    path = store / "snapshots" / snapshot_id
    if not path.is_dir():
        return "missing"
    return "present" if path.stat().st_ino == 256 else "present (not a subvolume)"


def _origin_note(trajectory: Trajectory) -> str | None:
    origin = trajectory.origin
    if origin is None:
        return None
    parent = Path(origin.parent_journal)
    parent_run = (
        parent.parents[1].name if len(parent.parents) > 1 and parent.parent.name == "agent" else parent.name
    )
    return "branch of %s@%d from %s" % (parent_run, origin.branch_step, origin.snapshot_id)


def journal_snapshots(
    path: Path, note: str | None, selections: list[Selection], store_override: str | None, warnings: list[str]
) -> JournalSnapshots:
    t = read_trajectory(path)
    backend, sandbox_id = t.backend, t.sandbox_id
    if backend is None:
        # Before commit ce125c7 a journal had no sandbox.created: the trial's receipt says.
        backend, sandbox_id = receipt_identity(path)
        sandbox_id = t.sandbox_id or sandbox_id
    store = store_note = None
    if backend == "podman":
        chosen = store_override or t.store
        if chosen is None:
            store_note = "not checked (no store: pass --store)"
        elif not (Path(chosen) / "snapshots").is_dir():
            store_note = "not checked: store %s has no snapshots/ directory" % chosen
            if store_override:
                warnings.append("store %s has no snapshots/ directory" % chosen)
        else:
            store = Path(chosen)
            store_note = "present = %s/snapshots/<id> is a directory" % chosen
    elif backend == "microvm":
        store_note = "not checked (AgentENV is not queried in this version%s)" % (
            "" if t.backend else "; backend from the receipt: the journal predates sandbox.created"
        )
    else:
        store_note = (
            "not checked (backend unknown: no sandbox.created and no receipt; AgentENV is not queried)"
        )
    by_id: dict[str, list[str]] = defaultdict(list)
    for s in selections:
        if s.snapshot_id:
            by_id[s.snapshot_id].append(s.label)
    records = []
    totals = Counter()
    for c in t.captures:
        totals["records"] += 1
        totals[c.reason] += 1
        totals["branch_points"] += c.branch_point
        totals["memory"] += c.kind == "memory"
        selected = tuple(by_id.get(c.snapshot_id or "", ()))
        if selected:
            totals["selected"] += 1
        if t.swept:
            status = "swept"
        elif c.snapshot_id and c.snapshot_id in t.released:
            status = "released"
        elif not c.snapshot_id:
            status = "none"
        elif store is not None:
            status = store_status(store, c.snapshot_id)
        else:
            status = "not checked"
        totals[status.split(" ")[0]] += 1
        records.append(SnapshotRecord(c, selected, status))
    ids = {c.snapshot_id for c in t.captures if c.captured and c.snapshot_id}
    totals["ids"] = len(ids)
    origin = None
    if t.origin is not None:
        origin = {
            "parent_journal": t.origin.parent_journal,
            "branch_step": t.origin.branch_step,
            "snapshot_id": t.origin.snapshot_id,
            "conversation_cut": t.origin.conversation_cut,
            "branch_guidance": t.origin.branch_guidance,
        }
    return JournalSnapshots(
        run_id=t.run_id,
        journal=str(path),
        backend=backend,
        sandbox_id=sandbox_id,
        store=str(store) if store else None,
        store_note=store_note,
        origin=origin,
        note=note or _origin_note(t),
        records=tuple(records),
        totals=dict(totals),
    )


def snapshots_report(target: Path, kind: str, *, store: str | None, task: str | None) -> SnapshotsReport:
    journals = find_journals(target, kind, task)
    if len(journals) > MANY_JOURNALS:
        print(
            "reading %d journals (a whole experiment takes a minute; --task narrows it)" % len(journals),
            file=sys.stderr,
        )
    selections = find_selections(target, kind)
    warnings: list[str] = []
    reports = [journal_snapshots(path, note, selections, store, warnings) for path, note in journals]
    # A branch's fork.origin names a snapshot on its parent's rows: label it when no plan did.
    forks = {r.origin["snapshot_id"]: r.run_id for r in reports if r.origin}
    for snapshot_id, run_id in forks.items():
        if not any(s.snapshot_id == snapshot_id for s in selections):
            selections.append(Selection(snapshot_id, "forked: " + run_id, "fork.origin"))
    if forks:
        reports = [journal_snapshots(path, note, selections, store, []) for path, note in journals]
    totals: Counter = Counter()
    ids = set()
    for r in reports:
        for key in ("records", "present", "missing", "swept", "released", "selected", "memory"):
            totals[key] += r.totals.get(key, 0)
        totals["not_checked"] += r.totals.get("not", 0)
        ids |= {
            rec.capture.snapshot_id for rec in r.records if rec.capture.captured and rec.capture.snapshot_id
        }
    totals["journals"] = len(reports)
    totals["ids"] = len(ids)
    totals["plan_selections"] = sum(1 for s in selections if s.source != "fork.origin")
    stores = {r.store for r in reports if r.store}
    return SnapshotsReport(
        store=store or (next(iter(stores)) if len(stores) == 1 else None),
        store_source="--store" if store else "sandbox.created" if stores else None,
        selections=tuple(selections),
        journals=tuple(reports),
        totals=dict(totals),
        warnings=tuple(dict.fromkeys(warnings)),
    )


def render_snapshots(report: SnapshotsReport) -> str:
    lines = []
    if report.store:
        lines.append(
            "store %s (from %s) · podman: present = <store>/snapshots/<id> is a directory · nothing is "
            "deleted or executed" % (report.store, report.store_source)
        )
    else:
        lines.append(
            "no podman store known · microVM ids are not checked (AgentENV is not queried) · nothing is "
            "deleted or executed"
        )
    lines += list(report.warnings)
    by_source: dict[str, list[str]] = defaultdict(list)
    for s in report.selections:
        by_source[s.source].append("%s (%s)" % (s.snapshot_id or "unresolved", s.label))
    for source, items in by_source.items():
        lines.append("selected by %s: %s" % (source, " · ".join(items)))
    for j in report.journals:
        lines.append("")
        head = "== %s   %s   sandbox %s" % (j.run_id, j.backend or "backend unknown", j.sandbox_id)
        if j.note:
            head += "   " + j.note
        lines += [head, "   " + j.journal, "   " + j.store_note]
        rows = [
            (
                "snapshot",
                "step",
                "turn",
                "kind",
                "reason",
                "disk-only",
                "final",
                "branch-point",
                "selected",
                "store",
            )
        ]
        for rec in j.records:
            c = rec.capture
            rows.append(
                (
                    c.snapshot_id or "-",
                    str(c.step),
                    "-" if c.turn is None else str(c.turn),
                    c.kind,
                    c.reason,
                    "yes" if c.disk_only else "no" if c.disk_only is False else "-",
                    "yes" if c.final else "no",
                    "yes" if c.branch_point else "no",
                    "; ".join(rec.selected) or "-",
                    rec.store,
                )
            )
        lines += ["   " + line for line in columns(rows)]
        t = j.totals
        summary = (
            "%d records · %d captured (%d ids) · %d clean · %d failed · %d branch points · %d selected"
            % (
                t.get("records", 0),
                t.get("captured", 0),
                t.get("ids", 0),
                t.get("clean", 0),
                t.get("failed", 0),
                t.get("branch_points", 0),
                t.get("selected", 0),
            )
        )
        if t.get("memory"):
            summary += " · %d memory" % t["memory"]
        if t.get("swept"):
            summary += " · %d swept" % t["swept"]
        if t.get("released"):
            summary += " · %d released" % t["released"]
        if t.get("present") or t.get("missing"):
            summary += " · store: %d present, %d missing" % (t.get("present", 0), t.get("missing", 0))
        elif t.get("not"):
            summary += " · store: not checked"
        lines.append("   " + summary)
    t = report.totals
    lines += [
        "",
        "totals  %d journals · %d records · %d snapshot ids · %d present · %d missing · %d not checked · "
        "%d selected by plans · %d swept · %d released"
        % (
            t.get("journals", 0),
            t.get("records", 0),
            t.get("ids", 0),
            t.get("present", 0),
            t.get("missing", 0),
            t.get("not_checked", 0),
            t.get("plan_selections", 0),
            t.get("swept", 0),
            t.get("released", 0),
        ),
    ]
    return "\n".join(lines)


def snapshots_json(report: SnapshotsReport) -> dict:
    return {
        "store": report.store,
        "store_source": report.store_source,
        "warnings": list(report.warnings),
        "selections": [asdict(s) for s in report.selections],
        "journals": [
            {
                "run_id": j.run_id,
                "journal": j.journal,
                "backend": j.backend,
                "sandbox_id": j.sandbox_id,
                "store": j.store,
                "store_note": j.store_note,
                "origin": j.origin,
                "note": j.note,
                "records": [
                    {**asdict(r.capture), "selected": list(r.selected), "store": r.store} for r in j.records
                ],
                "totals": j.totals,
            }
            for j in report.journals
        ],
        "totals": report.totals,
    }


# --- list ------------------------------------------------------------------------------


@dataclass(frozen=True)
class ListRow:
    kind: str
    methods: str
    tasks: str
    complete: str
    errored: str
    running: str
    stale: str  # stages of a dead runner; unlaunched receipts count as neither
    runner: str
    path: str


def journal_finished(path: Path) -> bool:
    """Whether a journal's tail holds ``run.finished`` (the final checkpoint follows it)."""
    try:
        with path.open("rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - 8192))
            return b'"type": "run.finished"' in stream.read()
    except OSError:
        return False


def list_row(path: Path, kind: str, base: Path) -> ListRow:
    rel = str(path.relative_to(base)) if path != base else "."
    dash = "-"
    if kind in ("experiment", "method"):
        roots = method_roots(path)
        states = [task_state(roots[0], name) for name in roots[0].tasks]
        counts = Counter(s.status for s in states)
        alive = [r.lock_alive for r in roots if r.lock_pid is not None]
        runner = "alive" if any(alive) else "dead" if alive else "done"
        return ListRow(
            kind,
            ",".join(r.method for r in roots),
            str(len(roots[0].tasks)),
            str(counts["complete"]),
            str(counts["errored"]),
            str(counts["in-progress"]),
            str(counts["stale"]),
            runner,
            rel,
        )
    if kind == "evaluate":
        summary = read_dict(path / "summary.json") or {}
        trials = (
            [p for p in sorted((path / "trials").iterdir()) if p.is_dir()]
            if (path / "trials").is_dir()
            else []
        )
        errored = sum(1 for t in trials if (read_dict(t / "result.json") or {}).get("exception_info"))
        running = sum(
            1
            for t in trials
            if not (t / "result.json").is_file()
            and (t / "agent" / "trajectory.jsonl").is_file()
            and not journal_finished(t / "agent" / "trajectory.jsonl")
        )
        return ListRow(
            kind,
            str(summary.get("method") or dash),
            "1",
            str(int(bool(summary.get("complete")))),
            str(errored),
            str(running),
            dash,
            dash,
            rel,
        )
    if kind == "gate":
        summary = read_dict(path / "summary.json") or {}
        return ListRow(
            kind,
            ",".join(summary.get("modes") or []) or dash,
            str(summary.get("tasks", dash)),
            dash,
            dash,
            dash,
            dash,
            dash,
            rel,
        )
    return ListRow(kind, dash, dash, dash, dash, dash, dash, dash, rel)


def list_rows(directory: Path) -> list[ListRow]:
    found: list[tuple[Path, str]] = []
    kind = detect(directory)
    if kind is not None:
        found.append((directory, kind))
    for child in sorted(directory.iterdir()):
        kind = detect(child)
        if kind is not None:
            found.append((child, kind))
        elif child.is_dir() and not child.name.startswith("."):
            for grandchild in sorted(child.iterdir()):
                kind = detect(grandchild)
                if kind is not None:
                    found.append((grandchild, kind))
    rows = [list_row(path, kind, directory) for path, kind in found]
    return sorted(rows, key=lambda row: row.path)


def render_list(rows: list[ListRow]) -> str:
    table = [("kind", "methods", "tasks", "complete", "errored", "running", "stale", "runner", "path")]
    table += [
        (r.kind, r.methods, r.tasks, r.complete, r.errored, r.running, r.stale, r.runner, r.path)
        for r in rows
    ]
    return "\n".join(columns(table))

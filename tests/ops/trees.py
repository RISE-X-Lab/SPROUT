"""Creation functions for the run output that ``ops/report`` reads: journals, evaluations and run roots.

Each function writes the files, the field names and the event order of their writer, under a
directory of the test. A journal follows the worker: a turn's ``checkpoint.captured`` comes after
the next turn's model response, and the final one after ``run.finished``. In a journal of sandbox
``sbx``, the snapshot after turn N is ``sbx-000N``, the turn id is ``tN`` and the call id of step K is
``cK``. An evaluation is the Run Store's: its journals are in the worker's artifact directory, which
``summary.json`` names. Values that a test does not check have defaults here.
"""

from __future__ import annotations

import hashlib
import json
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS

TASK = "datacurve/alpha"
IMAGE = "registry.example/task@sha256:" + "0" * 64
#: The environment that mini's sandbox puts before each command (``env K=V ... bash -c <command>``).
MINI_ENV = "env PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1"
#: The usage of each actor model call.
CALL_USAGE = {
    "input_tokens": 100,
    "output_tokens": 10,
    "cached_input_tokens": 50,
    "cache_creation_tokens": 0,
    "reasoning_output_tokens": 5,
    "cost_usd": 0.0,
}
#: The keys that every stdout/stderr envelope of the runtime carries besides the output itself.
_ENVELOPE = {"running": False, "stdout_truncated": False, "stderr_truncated": False}


@dataclass(frozen=True)
class Step:
    """One tool call of a turn: the model's command, and what it printed and returned."""

    command: str
    output: str = "ok"
    exit_code: int = 0


@dataclass(frozen=True)
class Turn:
    """One turn of a journal. A ``reviewer`` turn is a reviewer-written assistant turn of a branch."""

    steps: tuple[Step, ...]
    source: str = "actor"


def actor(*commands: str) -> Turn:
    """An actor turn with one call for each command, each printing ``ok``."""
    return Turn(tuple(Step(command) for command in commands))


def reviewer(*commands: str) -> Turn:
    """A reviewer-written turn with one call for each command, each printing ``ok``."""
    return Turn(tuple(Step(command) for command in commands), source="reviewer")


def numbered(count: int, *, prefix: str = "ACTION") -> list[Turn]:
    """``count`` actor turns of one call: turn K runs ``echo <prefix>_K`` and prints ``RESULT_K``."""
    return [Turn((Step("echo %s_%d" % (prefix, k), "RESULT_%d" % k),)) for k in range(1, count + 1)]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, value: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    return path


def fork_origin(parent: Path, *, step: int, snapshot_id: str, sha: str | None, **extra: Any) -> dict:
    """The ``fork.origin`` of a branch of ``parent`` after ``step``, with conversation cut ``t<step>``."""
    return {
        "parent_journal": str(parent),
        "branch_step": step,
        "snapshot_id": snapshot_id,
        "conversation_cut": "t%d" % step,
        "branch_guidance": "assistant-turn",
        "assistant_turn_source": "reviewer",
        "parent_journal_sha256": sha,
        **extra,
    }


class _Journal:
    def __init__(self, run_id: str) -> None:
        self.run_id = run_id
        self.lines: list[str] = []

    def emit(self, kind: str, **payload: Any) -> None:
        seq = len(self.lines) + 1
        record = {
            "v": 2,
            "type": kind,
            "ts": "2026-09-30T20:%02d:%02d.000Z" % (seq // 60, seq % 60),
            "seq": seq,
            "run_id": self.run_id,
            "agent_id": "agent",
            "sandbox_id": "default",
            **payload,
        }
        self.lines.append(json.dumps(record, ensure_ascii=False))


def _model_response(journal: _Journal, number: int, ids: list[str], turn: Turn) -> None:
    calls = [
        {
            "id": cid,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": s.command})},
        }
        for cid, s in zip(ids, turn.steps, strict=True)
    ]
    if turn.source == "reviewer":
        message = {"role": "assistant", "content": "reviewer turn %d" % number, "tool_calls": calls}
        journal.emit("branch.assistant_turn", source="reviewer", turn_id="t%d" % number, message=message)
        journal.emit("agent.message", text=message["content"])
        return
    message = {"role": "assistant", "content": "MESSAGE %d" % number, "tool_calls": calls}
    usage = {"prompt_tokens": CALL_USAGE["input_tokens"], "completion_tokens": CALL_USAGE["output_tokens"]}
    journal.emit(
        "raw.mini-swe-agent", response={"choices": [{"index": 0, "message": message}], "usage": usage}
    )
    journal.emit("usage.updated", usage=CALL_USAGE)
    journal.emit("agent.message", text="MESSAGE %d" % number)
    journal.emit("agent.thinking", text="THINKING %d" % number)


def _call(journal: _Journal, *, cid: str, turn_id: str, step: int, call: Step) -> None:
    args = {"command": "%s bash -c %s" % (MINI_ENV, shlex.quote(call.command)), "working_dir": "/app"}
    journal.emit("tool.started", call_id=cid, turn_id=turn_id, name="shell", args=args, step=step)
    stderr = "" if call.exit_code == 0 else "ERR %d" % step
    outcome = {
        "exit_code": call.exit_code,
        "stdout": call.output,
        "stderr": stderr,
        "timed_out": False,
        **_ENVELOPE,
    }
    status, output = ("ok", call.output) if call.exit_code == 0 else ("error", json.dumps(outcome))
    journal.emit(
        "tool.finished",
        call_id=cid,
        turn_id=turn_id,
        name="shell",
        status=status,
        output=output,
        outcome=outcome,
    )


def _capture(journal: _Journal, *, sandbox: str, number: int, step: int, final: bool) -> None:
    journal.emit(
        "checkpoint.captured",
        step=step,
        turn_id="t%d" % number,
        turn_closed=True,
        snapshot_id="%s-%04d" % (sandbox, number),
        session_ckpt="session-1",
        captured=True,
        reason="captured",
        disk_only=True,
        **({"final": True} if final else {}),
    )


def write_journal(
    path: Path,
    *,
    turns: list[Turn],
    run_id: str = "run",
    sandbox: str = "sbx",
    backend: str | None = "podman",
    store: str | None = None,
    image: str = IMAGE,
    origin: dict | None = None,
    prompt: str = "PROMPT",
    status: str | None = "completed",
    error: str | None = None,
    open_turn: bool = False,
    tail: tuple[dict, ...] = (),
    torn: bool = False,
) -> Path:
    """A journal of ``turns``. ``status`` None leaves it live: no ``run.finished`` and no final capture.

    ``open_turn`` starts one more turn whose model call is in flight. Each record of ``tail`` is
    appended as written, and ``torn`` ends the file with half a record."""
    journal = _Journal(run_id)
    if origin is not None:
        journal.emit("fork.origin", **origin)
    if backend is not None:
        journal.emit(
            "sandbox.created",
            sandbox_id=sandbox,
            backend=backend,
            image=image,
            **({"store": store} if store else {}),
        )
    journal.emit("run.started", slot="mini-swe-agent", slot_version="2.4.6", model="m", task_prompt=prompt)
    journal.emit("session.ref", native_session_id="session-1")
    step, pending = 0, None
    for number, turn in enumerate(turns, 1):
        journal.emit("turn.started", turn=number, source=turn.source)
        ids = ["c%d" % (step + k) for k in range(1, len(turn.steps) + 1)]
        _model_response(journal, number, ids, turn)
        if pending is not None:
            _capture(journal, sandbox=sandbox, number=pending[0], step=pending[1], final=False)
            pending = None
        if not turn.steps:
            # mini rejects a response without tool calls. The turn runs nothing and gets no capture.
            journal.emit("mini.turn.rejected", turn_id="t%d" % number, source="no_tool_calls", executed=False)
            journal.emit("turn.completed", turn=number, source=turn.source, rejected=True)
            continue
        for cid, call in zip(ids, turn.steps, strict=True):
            step += 1
            _call(journal, cid=cid, turn_id="t%d" % number, step=step, call=call)
        journal.emit(
            "model.turn.completed", turn_id="t%d" % number, call_ids=ids, step=step, source=turn.source
        )
        journal.emit("turn.completed", turn=number, source=turn.source)
        pending = (number, step)
    if open_turn:
        journal.emit("turn.started", turn=len(turns) + 1, source="actor")
    if status is not None:
        model_calls = sum(1 for turn in turns if turn.source == "actor")
        usage = {key: value * model_calls for key, value in CALL_USAGE.items()}
        journal.emit("run.result", text="DONE")
        journal.emit("run.finished", status=status, error=error, usage=usage)
        journal.emit("rollout.usage", model_calls=model_calls, tool_calls=step)
        if pending is not None:
            _capture(journal, sandbox=sandbox, number=pending[0], step=pending[1], final=True)
    text = "".join(line + "\n" for line in journal.lines + [json.dumps(record) for record in tail])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + ('{"v": 2, "type": "turn.sta' if torn else ""), encoding="utf-8")
    return path


# --- evaluations through the Run Store ----------------------------------------------


def artifact_journal(artifacts: Path, job: str, **journal: Any) -> Path:
    """A journal in the worker's attempt directory of ``job``, as a Run Store rollout leaves it."""
    return write_journal(
        artifacts / job / ("attempt-" + job) / "trajectory.jsonl", run_id="attempt-" + job, **journal
    )


def attempt_row(
    name: str, *, journal: Path, resolved: bool, round_no: int = 0, seconds: float = 120.0
) -> dict:
    """The ``summary.json`` row of a graded attempt."""
    return {
        "name": name,
        "round": round_no,
        "status": "completed",
        "stop_reason": None,
        "resolved": resolved,
        "reward": float(resolved),
        "reason": None,
        "journal": str(journal),
        "started_at": 1790800000.0,
        "validated_at": 1790800000.0 + seconds,
        "seconds": seconds,
    }


def write_summary(
    output: Path,
    *,
    attempts: list[dict],
    task: str = TASK,
    method: str = "sprout",
    complete: bool = True,
    **extra: Any,
) -> Path:
    """The ``summary.json`` of an evaluation of ``task`` with ``method``. ``extra`` adds or replaces keys."""
    summary = {
        "task": task,
        "model": "m",
        "method": method,
        "parent_output": None,
        "branch_guidance": "assistant-turn" if method == "sprout" else "none",
        "branch_caps": [4, 3],
        "rounds": 2 if method == "sprout" else 1,
        "attempts": attempts,
        "complete": complete,
        "resolved": any(row["resolved"] for row in attempts),
        **extra,
    }
    return write_json(output / "summary.json", summary)


def analyst_report(*, steps: int, candidates: list[int]) -> dict:
    """An analyst's report on a failed parent of ``steps`` steps."""
    return {
        "failure_reason": "FAILURE REASON",
        "lesson": "LESSON",
        "salvage": "SALVAGE",
        "branch_candidates": [{"step": step, "why": "late and green"} for step in candidates],
        "steps": steps,
    }


def selected_branch(run_name: str, *, step: int, snapshot_id: str, commands: tuple[str, ...]) -> dict:
    """A branch that a reviewer selected: from the parent after ``step``, with a turn of ``commands``."""
    calls = [
        {
            "id": "rev%d-%d" % (step, k),
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": c})},
        }
        for k, c in enumerate(commands, 1)
    ]
    return {
        "run_name": run_name,
        "base": "parent",
        "branch_step": step,
        "snapshot_id": snapshot_id,
        "conversation_cut": "t%d" % step,
        "branch_guidance": "assistant-turn",
        "assistant_turn": {"role": "assistant", "content": "One more check.", "tool_calls": calls},
    }


def review_attempt(
    number: int, *, status: str, response: str | None = None, error: str | None = None
) -> dict:
    """One reviewer attempt of a plan. A validated one planned one branch."""
    attempt: dict[str, Any] = {"attempt": number, "prompt": "P", "status": status}
    if response is not None:
        attempt["response"] = response
    if error is not None:
        attempt["validation_error"] = error
    if status == "validated":
        attempt["plan"] = {"branches": [{}]}
    return attempt


def write_plan(output: Path, round_no: int, **body: Any) -> Path:
    """``plan-round<N>.json`` with the defaults of an adaptive round of at most 4 branches."""
    plan = {
        "reports": {},
        "available_steps": {},
        "branch_limit": 4,
        "branch_count_mode": "adaptive",
        "branch_guidance": "assistant-turn",
        "review_attempts": [],
        "selected_branches": [],
        **body,
    }
    return write_json(output / ("plan-round%d.json" % round_no), plan)


def write_review_receipt(path: Path, *, kind: str, text: str | None, seconds: float | None) -> Path:
    """A review transport receipt of an ``analyst`` or ``reviewer`` request. ``seconds`` None is in flight."""
    prompt = DEFAULT_PROMPTS.analyst_case if kind == "analyst" else DEFAULT_PROMPTS.reviewer_point
    record: dict[str, Any] = {
        "request": {"model": "m", "prompt": prompt, "max_tokens": 1000},
        "started_at": 1790800000.0,
    }
    if seconds is not None:
        record.update(
            response={"usage": {"prompt_tokens": 10, "completion_tokens": 5}},
            text=text,
            finished_at=1790800000.0 + seconds,
            seconds=seconds,
        )
    return write_json(path, record)


def write_group(output: Path, index: int, document: dict) -> Path:
    """A ``groups/`` document of an evaluation through the Run Store."""
    row = {
        "document": document,
        "cancel_requested": False,
        "terminal": bool(document.get("ready")),
        "created_at": 0.0,
    }
    return write_json(output / "groups" / ("group-%d.json" % index), row)


# --- the runner's roots (examples/qwen/run.py) ------------------------------------------


def write_manifest(method_root: Path, *, method: str, tasks: list[str], **extra: Any) -> Path:
    """``manifest-<method>.json`` in the shape of ``manifest_for``. ``extra`` adds or replaces keys."""
    manifest = {
        "schema_version": 1,
        "dataset": "deepswe",
        "tasks": tasks,
        "model": "m",
        "baseline_rollouts": 4,
        "max_total_method_rollouts": 8,
        "backend": "podman",
        "store": "/path/to/store",
        "source_commit": "ce125c7d3b7c0000",
        "method": method,
        "baseline_root": None,
        **extra,
    }
    return write_json(method_root / ("manifest-%s.json" % method), manifest)


def write_lock(method_root: Path, *, method: str, pid: int) -> Path:
    """The runner's lock file, which holds its pid."""
    path = method_root / (".%s.lock" % method)
    path.write_text(str(pid))
    return path


def write_stage(
    output: Path,
    *,
    state: str,
    task: str = "alpha",
    method: str = "baseline",
    seconds: float = 600.0,
    log: str | None = None,
    **extra: Any,
) -> Path:
    """The receipt ``<output>.stage.json`` of a stage in ``state``, and its log if it was launched.

    ``state`` is ``complete``, ``errored``, ``running`` or ``unlaunched`` (a receipt alone). ``log``
    replaces the log's text, and ``extra`` adds receipt keys."""
    receipt: dict[str, Any] = {
        "task": task,
        "method": method,
        "started_at": 1790799500.0,
        "complete": state == "complete",
        "output": str(output),
        "source_commit": "ce125c7d3b7c0000",
        **extra,
    }
    if state in ("complete", "errored"):
        receipt.update(
            exit_code=0 if state == "complete" else 1, finished_at=1790799500.0 + seconds, seconds=seconds
        )
    if state == "errored":
        receipt.update(error_type="RuntimeError", error="Stage exited 1; see %s.log" % output.name)
    if state != "unlaunched":
        output.parent.mkdir(parents=True, exist_ok=True)
        output.with_name(output.name + ".log").write_text(log if log is not None else "harbor: started\n")
    return write_json(output.with_name(output.name + ".stage.json"), receipt)


def write_attempt(
    task_dir: Path, number: int, *, artifacts: Path, resolved: bool, turns: int = 4, seconds: float = 600.0
) -> Path:
    """A complete baseline attempt: its receipt, its log, its summary and its journal in ``artifacts``.

    The journal's sandbox is ``par`` for attempt 1 and ``sb<N>`` for the others."""
    output = task_dir / "baseline" / ("attempt-%d" % number)
    job = "job-%s-%d" % (task_dir.name, number)
    journal = artifact_journal(
        artifacts, job, turns=numbered(turns), sandbox="par" if number == 1 else "sb%d" % number
    )
    row = attempt_row("parent", journal=journal, resolved=resolved, seconds=seconds)
    write_summary(output, attempts=[row], task="datacurve/" + task_dir.name, method="baseline")
    write_stage(output, state="complete", task=task_dir.name, seconds=seconds)
    return output


def write_retained(output: Path, *, steps: tuple[int, ...] = (1, 2, 3)) -> Path:
    """The ``retained-checkpoints.json`` that the runner writes into a failed attempt-1."""
    points = [{"checkpoint": {"step": step, "snapshot_id": "par-%04d" % step}} for step in steps]
    body = {
        "schema_version": 1,
        "mode": "disk_only",
        "points": points,
        "backend": "podman",
        "store": "/path/to/store",
    }
    return write_json(output / "retained-checkpoints.json", body)


def write_task_state(method_root: Path, task: str, **body: Any) -> Path:
    """``tasks/<task>/<method>.task.json``, the state that ``run_task`` writes when the task ends."""
    method = json.loads(next(method_root.glob("manifest-*.json")).read_text())["method"]
    state = {"task": task, "dataset": "deepswe", "methods": {}, "complete": False, **body}
    return write_json(method_root / "tasks" / task / ("%s.task.json" % method), state)

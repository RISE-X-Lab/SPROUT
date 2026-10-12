"""Synthetic run trees in ``tmp_path``, with the writers' real field names and EVENT ORDER.

The order matters: a turn's ``checkpoint.captured`` lands after the NEXT turn's
model response, the final one after ``run.finished``/``rollout.usage``,
``fork.origin`` is seq 1 of a branch, ``mini.restored`` follows ``session.ref``
and a branch's first turn is the reviewer's. Snapshot ids are ``<sandbox>-NNNN``,
turn ids ``t<N>``, call ids ``c<step>``; a parent made with ``sandbox="par"`` has
its turn-3 branch point at ``par-0003`` / cut ``t3``.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sprout.inspect.layout import ANALYST_PROMPT, REVIEWER_PROMPT

FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "inspect" / "abs-module-cache-flags"
TASK = "datacurve/alpha"
WORKTREE_ERROR = "RuntimeError: The source worktree must be clean before a real trial"
SENTINEL = "handed to root relaunch-tail at 00:02 UTC to raise concurrency (user-approved); this runner skips it"
IMAGE = "public.ecr.aws/x/img@sha256:0123456789abcdef"
ENVELOPE_KEYS = {"running": False, "stdout_bytes": 0, "stderr_bytes": 0, "stdout_truncated": False,
                 "stderr_truncated": False}


def wrapped(command: str) -> str:
    """mini's wrapper around every command, as tool.started records it."""
    return "env PAGER=cat MANPAGER=cat LESS=-R PIP_PROGRESS_BAR=off TQDM_DISABLE=1 bash -c '%s'" % command


def write_json(path: Path, value) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return path


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_journal(path: Path, *, run_id: str, turns: list, sandbox: str, backend: str | None = "podman",
                 store: str | None = None, image: str = IMAGE, origin: dict | None = None,
                 status: str | None = "completed", prompt: str = "PROMPT", session: str = "ses1",
                 torn: bool = False, extra_tail: tuple = (), open_turn: bool = False, model: str = "m",
                 error: str | None = None, usage: dict | None = None) -> tuple[Path, str]:
    """``turns`` is ``[(source, [(command, output, exit_code), ...]), ...]``."""
    path.parent.mkdir(parents=True, exist_ok=True)
    seq = 0
    ts = [0]
    lines: list[str] = []

    def emit(kind: str, **payload) -> dict:
        nonlocal seq
        seq += 1
        ts[0] += 1
        record = {"v": 2, "type": kind, "ts": "2026-09-30T20:%02d:%02d.000Z" % (ts[0] // 60, ts[0] % 60),
                  "seq": seq, "run_id": run_id, "agent_id": "agent", "sandbox_id": "default"}
        record.update(payload)
        lines.append(json.dumps(record, ensure_ascii=False))
        return record

    if origin is not None:
        emit("fork.origin", **origin)
    if backend is not None:
        emit("sandbox.created", sandbox_id=sandbox, backend=backend, image=image,
             **({"store": store} if store else {}))
    emit("mini.workspace", cwd="/app", repository_dir="/app", inherited=origin is not None)
    emit("run.started", slot="mini-swe-agent", slot_version="2.4.6", model=model, task_prompt=prompt,
         cwd="/tmp/actor-workspace", prompt_primer="mini_tool_primer")
    emit("session.ref", native_session_id=session, transcript_path="/tmp/mini-native/%s.jsonl" % session)
    if origin is not None:
        emit("mini.restored", inherited_native={"slot": "mini-swe-agent", "session_id": "parentses",
                                                "path": str(Path(origin["parent_journal"]).parent / "mini-native" / "parentses.jsonl"),
                                                "byte_length": 1000, "sha256": "0" * 64,
                                                "cut": origin["conversation_cut"]})
    tools = [{"type": "function", "function": {"name": "bash", "parameters": {
        "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]
    emit("rollout.model_tools", shape="chat/completions", tools=tools)
    step = 0
    previous = None   # (turn_id, step, call_ids) awaiting its lazy checkpoint
    model_calls = 0
    per_call = {"input_tokens": 100, "output_tokens": 10, "cached_input_tokens": 50, "cache_creation_tokens": 0,
                "reasoning_output_tokens": 5, "cost_usd": 0.0}
    for index, (source, calls) in enumerate(turns, 1):
        turn_id = "t%d" % index
        emit("turn.started", turn=index, source=source)
        ids = ["c%d" % (step + i + 1) for i in range(len(calls))]
        if source == "reviewer":
            message = {"role": "assistant", "content": "reviewer turn %d" % index,
                       "tool_calls": [{"id": cid, "type": "function", "function": {
                           "name": "bash", "arguments": json.dumps({"command": c[0]})}} for cid, c in zip(ids, calls)]}
            emit("branch.assistant_turn", source="reviewer", turn_id=turn_id, message=message)
            emit("agent.message", text=message["content"])
        else:
            model_calls += 1
            emit("rollout.model_tools", shape="chat/completions", tools=tools)
            emit("gateway.request", status="ok", requested_model=model, upstream_model=model,
                 base_url="http://model.invalid", usage=per_call, spent_usd=0.0)
            emit("raw.mini-swe-agent", response={"choices": [{"finish_reason": "tool_calls", "index": 0, "message": {
                "role": "assistant", "content": "MESSAGE %d" % index, "reasoning_content": "THINKING %d" % index,
                "tool_calls": [{"id": cid, "type": "function", "function": {
                    "name": "bash", "arguments": json.dumps({"command": c[0]})}} for cid, c in zip(ids, calls)]}}],
                "usage": {"prompt_tokens": 100, "completion_tokens": 10}})
            emit("usage.updated", usage=per_call)
            emit("agent.message", text="MESSAGE %d" % index)
            emit("agent.thinking", text="THINKING %d" % index)
        if previous is not None:
            p_turn, p_step, p_ids = previous
            emit("checkpoint.captured", step=p_step, turn_id=p_turn, turn_closed=True, call_ids=p_ids,
                 snapshot_id="%s-%04d" % (sandbox, index - 1), session_ckpt=session, captured=True,
                 reason="captured", disk_only=True, delta_empty=None, reboarded=False)
            previous = None
        for cid, (command, output, code) in zip(ids, calls):
            step += 1
            emit("tool.started", call_id=cid, turn_id=turn_id, name="shell",
                 args={"command": wrapped(command), "working_dir": "/app", "timeout": 60}, step=step)
            outcome = {"exit_code": code, "stdout": output, "stderr": "", "timed_out": False, **ENVELOPE_KEYS}
            if code == 0:
                emit("tool.finished", call_id=cid, turn_id=turn_id, name="shell", status="ok", output=output,
                     outcome=outcome)
            else:
                outcome["stderr"] = "ERR %d" % step
                emit("tool.finished", call_id=cid, turn_id=turn_id, name="shell", status="error",
                     output=json.dumps({**outcome, "max_per_stream": 15728640}), outcome=outcome)
        if calls:
            emit("model.turn.completed", turn_id=turn_id, call_ids=ids, step=step, source=source)
            emit("turn.completed", turn=index, source=source)
            previous = (turn_id, step, ids)
    if open_turn:
        emit("turn.started", turn=len(turns) + 1, source="actor")
    if status is not None:
        total = {k: v * model_calls for k, v in per_call.items()}
        emit("run.result", text="")
        emit("run.finished", status=status, error=error, usage=usage if usage is not None else total)
        emit("rollout.usage", model_calls=model_calls, tool_calls=step)
        if previous is not None:
            p_turn, p_step, _ = previous
            emit("checkpoint.captured", step=p_step, turn_id=p_turn, turn_closed=True,
                 snapshot_id="%s-%04d" % (sandbox, len(turns)), session_ckpt=session, captured=True,
                 reason="captured", disk_only=True, final=True, delta_empty=None)
    text = "\n".join(lines) + "\n" + "".join(line + "\n" for line in extra_tail)
    if torn:
        text += '{"v": 2, "type": "turn.sta'
    path.write_text(text, encoding="utf-8")
    return path, sha256(path)


def simple_turns(n: int, *, prefix: str = "ACTION", exit_codes: dict | None = None) -> list:
    """``n`` actor turns of one call each: step k == turn k."""
    return [("actor", [("%s_%d" % (prefix, k), "RESULT_%d" % k, (exit_codes or {}).get(k, 0))])
            for k in range(1, n + 1)]


def trial_name(task: str, name: str) -> str:
    return "%s-%s" % (task.split("/")[-1][:36], name)


def make_evaluation(path: Path, *, task: str = TASK, method: str = "sprout", attempts: list, plans: tuple = (),
                    contexts: tuple = (), reviews: tuple = (), complete: bool = True, resolved=None,
                    stop_reason: str | None = None, backend: str = "podman", store: str | None = None) -> Path:
    """An evaluate output. Each attempt spec: ``{"name", "round", "resolved", "turns", "sandbox",
    "origin", "status", "recorded_journal", "exception", "seconds"}`` (missing keys default)."""
    path.mkdir(parents=True, exist_ok=True)
    rows = []
    for spec in attempts:
        name = spec["name"]
        status = spec.get("status", "completed")
        if spec.get("recorded_journal") is not None:
            journal = Path(spec["recorded_journal"])
        else:
            trial = path / "trials" / trial_name(task, name)
            journal, _ = make_journal(trial / "agent" / "trajectory.jsonl", run_id=trial.name,
                                      turns=spec.get("turns") or simple_turns(2), sandbox=spec.get("sandbox", "sbx"),
                                      backend=spec.get("backend", backend), store=store, origin=spec.get("origin"),
                                      status=status, image=spec.get("image", IMAGE),
                                      prompt=spec.get("prompt", "PROMPT"), open_turn=spec.get("open_turn", False),
                                      torn=spec.get("torn", False))
            write_json(trial / ("%s__env.%s.json" % (trial.name, "podman" if backend == "podman" else "agentenv")),
                       {"backend": backend, "sandbox_id": spec.get("sandbox", "sbx"),
                        "image": spec.get("image", IMAGE), "checkpoint_mode": "disk_only", "overlay": None,
                        **({"store": store} if store else {})})
            write_json(trial / ("%s__verifier__trial.%s.json" % (trial.name, "podman" if backend == "podman" else "agentenv")),
                       {"backend": backend, "sandbox_id": "verifier-" + spec.get("sandbox", "sbx"), "image": IMAGE,
                        "checkpoint_mode": "none", "overlay": {"copies": [], "runs": ["chmod +x /tests/test.sh"]}})
            if spec.get("origin"):
                write_json(trial / "restored-state.json", {"snapshot_id": spec["origin"]["snapshot_id"],
                                                          "sandbox_id": spec.get("sandbox", "sbx"), "base_image": IMAGE,
                                                          "branch_context": str(path / "contexts" / (name + ".json")),
                                                          "checkpoint_mode": "disk_only"})
            if status is not None:
                write_json(trial / "agent" / "execution.json", {"run_id": trial.name, "journal_path": str(journal),
                                                                "status": status, "sandbox_id": spec.get("sandbox", "sbx")})
                write_json(trial / "agent" / "snapshot.json", {"backend": backend, "sandbox_id": spec.get("sandbox", "sbx"),
                                                               "journal": str(journal), "actor_status": status})
            if spec.get("exception"):
                write_json(trial / "result.json", {"trial_name": trial.name, "verifier_result": None,
                                                   "exception_info": {"exception_type": spec["exception"],
                                                                      "exception_message": "boom"}})
            elif spec.get("resolved") is not None:
                reward = 1 if spec["resolved"] else 0
                write_json(trial / "result.json", {"trial_name": trial.name, "exception_info": None,
                                                   "verifier_result": {"rewards": {
                                                       "reward": reward, "f2p_total": 20, "f2p_passed": 20 if reward else 9,
                                                       "p2p_total": 3, "p2p_passed": 3}}})
        if status is None and spec.get("resolved") is None:
            continue                     # a live attempt has a trial but no summary row yet
        rows.append({"name": name, "round": spec.get("round", 0), "status": status,
                     "resolved": spec.get("resolved"), "reward": None if spec.get("resolved") is None
                     else float(bool(spec["resolved"])), "journal": str(journal),
                     "started_at": 1790800000.0, "validated_at": 1790800000.0 + spec.get("seconds", 120.0),
                     "seconds": spec.get("seconds", 120.0), "memory_snapshot": None})
    summary = {"source_commit": "abc", "task": task, "backend": backend, "model": "m", "method": method,
               "branch_guidance": "assistant-turn" if method == "sprout" else "none", "branch_caps": [4, 3],
               "rounds": 2 if method == "sprout" else 1, "final_memory_snapshot": "never", "max_rollouts": 8,
               "attempts": rows, "complete": complete,
               "resolved": any(r["resolved"] for r in rows) if resolved is None else resolved}
    if complete:
        summary["branch_schedule_complete"] = stop_reason is None
    if stop_reason:
        summary["stop_reason"] = stop_reason
        summary["branch_schedule_complete"] = False
    write_json(path / "summary.json", summary)
    for plan in plans:
        body = {"reports": plan.get("reports", {}), "available_steps": plan.get("available_steps", {}),
                "branch_limit": plan.get("branch_limit", 4), "branch_count_mode": plan.get("branch_count_mode", "adaptive"),
                "branch_guidance": plan.get("branch_guidance", "assistant-turn"), "reviewer_max_attempts": 3,
                "review_attempts": plan.get("review_attempts", []), "review": plan.get("review"),
                "selected_branches": plan.get("selected_branches", [])}
        if plan.get("validation_error"):
            body["validation_error"] = plan["validation_error"]
        write_json(path / ("plan-round%d.json" % plan.get("round", 1)), body)
    for run_name, context in contexts:
        write_json(path / "contexts" / (run_name + ".json"), context)
    for index, review in enumerate(reviews, 1):
        prompt = (ANALYST_PROMPT + " coding attempt." if review["kind"] == "analyst"
                  else REVIEWER_PROMPT + " for a failed coding task.")
        body = {"request": {"model": "m", "prompt": prompt, "max_tokens": 1000}, "started_at": review.get("started_at", 1790800000.0)}
        if review.get("finished_at", "set") is not None:
            body.update(response={"usage": {"prompt_tokens": 10, "completion_tokens": 5}}, text=review.get("text", ""),
                        finished_at=review.get("finished_at", 1790800000.0 + review.get("seconds", 60.0)),
                        seconds=review.get("seconds", 60.0))
        write_json(path / "reviews" / ("review-%03d.json" % index), body)
    return path


def analyst_report(steps: int, candidates: list) -> dict:
    return {"failure_reason": "FAILURE REASON: the resolver joined absolute targets with the base directory " * 3,
            "lesson": "LESSON: treat an absolute target as a standalone candidate",
            "salvage": "SALVAGE: the production change is present at the late checkpoints",
            "branch_candidates": [{"step": s, "why": "late and green"} for s in candidates], "steps": steps}


def assistant_turn(step: int, commands: list) -> dict:
    return {"role": "assistant", "content": "Before committing, one more check.",
            "tool_calls": [{"id": "rev%d-%d" % (step, i), "type": "function",
                            "function": {"name": "bash", "arguments": json.dumps({"command": c})}}
                           for i, c in enumerate(commands, 1)]}


def make_sprout_output(path: Path, *, parent_journal: Path, parent_sha: str, task: str = TASK,
                       branch_resolved: bool = True, branch_turns: list | None = None, live_branch: bool = False,
                       plan_state: str = "validated", step: int = 3) -> Path:
    """A sprout evaluation branching from ``parent_journal`` at ``step`` (its turn ``step``, cut ``t<step>``)."""
    origin = {"parent_journal": str(parent_journal), "branch_step": step, "snapshot_id": "par-%04d" % step,
              "conversation_cut": "t%d" % step, "branch_guidance": "assistant-turn",
              "assistant_turn_source": "reviewer", "parent_journal_sha256": parent_sha}
    turn = assistant_turn(step, ["cd /app && go test ./...", "set -e; cd /app; git commit -am fix"])
    selected = [{"run_name": "r1b1-fix", "base": "parent", "branch_step": step, "snapshot_id": "par-%04d" % step,
                 "conversation_cut": "t%d" % step, "branch_guidance": "assistant-turn", "assistant_turn": turn}]
    attempts = [{"name": "parent", "recorded_journal": parent_journal, "resolved": False}]
    if plan_state in ("validated", "requesting"):
        attempts.append({"name": "r1b1-fix", "round": 1, "resolved": None if live_branch else branch_resolved,
                         "status": None if live_branch else "completed", "sandbox": "brn", "origin": origin,
                         "image": "par-%04d" % step, "open_turn": live_branch,
                         "turns": branch_turns or [("reviewer", [("cd /app && go test ./...", "ok", 0),
                                                                 ("set -e; cd /app; git commit -am fix", "[fix]", 0)]),
                                                   ("actor", [("echo done", "done", 0)])]})
    review_attempts = [{"attempt": 1, "prompt": "P", "status": "validation_error", "response": "BAD PLAN",
                        "validation_error": "Invalid JSON inside branch-plan block: Expecting ','"}]
    reviews = [{"kind": "analyst", "text": "ANALYSIS", "seconds": 264.0},
               {"kind": "reviewer", "text": "BAD PLAN", "seconds": 702.0}]
    if plan_state == "validated":
        review_attempts.append({"attempt": 2, "prompt": "P", "status": "validated", "response": "GOOD PLAN",
                                "plan": {"synthesis": "s", "branches": [{}]}})
        reviews.append({"kind": "reviewer", "text": "GOOD PLAN", "seconds": 450.0})
    elif plan_state == "requesting":
        review_attempts.append({"attempt": 2, "prompt": "P", "status": "requesting"})
        reviews.append({"kind": "reviewer", "finished_at": None})
    plan = {"round": 1, "reports": {"parent": analyst_report(4, [step, 2])}, "available_steps": {"parent": [1, 2, 3, 4]},
            "review_attempts": review_attempts,
            "selected_branches": selected if plan_state == "validated" else []}
    if plan_state == "failed":
        plan["validation_error"] = "reviewer failed: Expecting value"
        plan["review"] = None
    return make_evaluation(path, task=task, method="sprout", attempts=attempts,
                           complete=plan_state == "validated" and not live_branch,
                           stop_reason=plan.get("validation_error") if plan_state == "failed" else None,
                           plans=(plan,), contexts=(("r1b1-fix", {"snapshot_id": "par-%04d" % step, "step": step,
                                                                  "parent_journal": str(parent_journal),
                                                                  "parent_journal_sha256": parent_sha,
                                                                  "conversation_cut": "t%d" % step}),)
                           if plan_state == "validated" else (), reviews=tuple(reviews))


def make_experiment(root: Path, *, tasks: dict, with_sprout: bool = True, lock_pid: int | None = None,
                    metrics: bool = False, commit: str = "ce125c7d3b7c0000") -> Path:
    """``tasks``: name -> {"attempts": [(state, resolved), ...], "sprout": state, "task_error": text};
    attempt states are complete | errored | live | unlaunched (a receipt written by hand to hand the
    task to another root: no log, no output), sprout states complete | skipped | errored | live |
    missing. Like the runner, every launched stage has its ``<stage>.log`` beside the receipt."""
    base = root / "baseline"
    names = list(tasks)
    manifest = {"schema_version": 1, "dataset": "deepswe", "dataset_digest": "sha256:d", "tasks": names, "model": "m",
                "reasoning_effort": "high", "agent": "mini-swe-agent==2.4.6", "branching": "assistant-turn",
                "baseline_rollouts": 4, "max_total_method_rollouts": 8, "max_turns": 300, "max_output_tokens": 64000,
                "final_memory_snapshot": "never", "backend": "podman", "store": str(root / "store"), "workers": 16,
                "source_commit": commit, "method": "baseline", "baseline_root": None}
    write_json(base / "manifest-baseline.json", manifest)
    write_json(base / "preparation.json", {"dataset": "deepswe", "dataset_digest": "sha256:d",
                                           "tasks": [{"name": "datacurve/" + n, "path": "/data/" + n, "task_toml_sha256": "x"}
                                                     for n in names]})
    (root / "driver.log").write_text("[2026-09-30T20:18:20+00:00] baseline start\n")
    if lock_pid is not None:
        (base / ".baseline.lock").write_text(str(lock_pid))
    parents: dict[str, tuple[Path, str]] = {}
    for name, spec in tasks.items():
        task_dir = base / "tasks" / name
        attempts = spec.get("attempts", [])
        if not attempts:
            continue
        rows, error = [], spec.get("task_error")
        for index, (state, resolved) in enumerate(attempts, 1):
            output = task_dir / "baseline" / ("attempt-%d" % index)
            log = output.with_name(output.name + ".log")
            receipt = {"task": name, "method": "baseline", "started_at": 1790799500.0 + index, "complete": False,
                       "output": str(output), "source_commit": commit}
            output.parent.mkdir(parents=True, exist_ok=True)
            if state == "unlaunched":
                receipt.update(source_commit=None, sentinel=SENTINEL)
            elif state == "errored":
                receipt.update(exit_code=1, error_type="RuntimeError", error="Stage exited 1; see attempt-%d.log" % index,
                               finished_at=1790799501.0 + index, seconds=0.36)
                log.write_text("Traceback (most recent call last):\n  File evaluate.py line 309\n%s\n" % WORKTREE_ERROR)
                error = receipt["error"]
            else:
                log.write_text("harbor: trial %s started\n" % name)
                evaluation = make_evaluation(output, task="datacurve/" + name, method="baseline", attempts=[{
                    "name": "parent", "resolved": resolved if state == "complete" else None, "sandbox": "par" if index == 1 else "sb%d" % index,
                    "status": "completed" if state == "complete" else None, "open_turn": state == "live",
                    "turns": simple_turns(5 if state == "complete" else 4), "seconds": 600.0 * index}],
                    complete=state == "complete", store=str(root / "store"))
                journal = output / "trials" / trial_name("datacurve/" + name, "parent") / "agent" / "trajectory.jsonl"
                if state == "complete":
                    receipt.update(exit_code=0, complete=True, finished_at=1790799500.0 + index + 600.0 * index,
                                   seconds=600.0 * index)
                    summary = json.loads((evaluation / "summary.json").read_text())
                    rows += summary["attempts"]
                    log.write_text(log.read_text() + json.dumps({k: v for k, v in summary.items() if k != "attempts"},
                                                                indent=2) + "\n")
                    if index == 1:
                        parents[name] = (journal, sha256(journal))
                        if not resolved:
                            write_json(output / "retained-checkpoints.json", {
                                "schema_version": 1, "retention": "keep-for-future-policies", "mode": "disk_only",
                                "journal": str(journal), "journal_sha256": sha256(journal),
                                "points": [{"checkpoint": {"step": s, "snapshot_id": "par-%04d" % s}} for s in (1, 2, 3)],
                                "snapshot_ids": ["par-%04d" % s for s in range(1, 6)], "backend": "podman",
                                "store": str(root / "store")})
            write_json(output.with_name(output.name + ".stage.json"), receipt)
        if error:
            write_json(task_dir / "baseline.task.json", {"task": name, "dataset": "deepswe", "methods": {}, "complete": False,
                                                        "error_type": "RuntimeError", "error": error,
                                                        "status": "infrastructure_error"})
        elif len(rows) == 4:
            write_json(task_dir / "baseline.task.json", {"task": name, "dataset": "deepswe", "methods": {}, "complete": True,
                                                        "baseline": {"output": str(task_dir / "baseline"), "attempts": rows},
                                                        "initial_resolved": rows[0]["resolved"]})
    if metrics:
        write_json(base / "metrics-baseline.json", metrics_document(manifest, "baseline"))
    if with_sprout:
        sprout = root / "sprout"
        write_json(sprout / "manifest-sprout.json", dict(manifest, method="sprout", baseline_root=str(base)))
        write_json(sprout / "preparation.json", json.loads((base / "preparation.json").read_text()))
        if lock_pid is not None:
            (sprout / ".sprout.lock").write_text(str(lock_pid))
        for name, spec in tasks.items():
            state = spec.get("sprout", "missing")
            task_dir = sprout / "tasks" / name
            output = task_dir / "sprout"
            if state == "skipped":
                write_json(task_dir / "sprout.task.json", {"task": name, "dataset": "deepswe", "complete": True,
                                                          "methods": {"sprout": {"status": "skipped_initial_success", "attempts": []}},
                                                          "initial_resolved": True})
                continue
            if state == "missing" or name not in parents:
                continue
            receipt = {"task": name, "method": "sprout", "started_at": 1790803000.0, "complete": False,
                       "output": str(output), "source_commit": commit}
            journal, digest = parents[name]
            task_dir.mkdir(parents=True, exist_ok=True)
            if state == "errored":
                receipt.update(exit_code=1, error_type="RuntimeError", error="Stage exited 1; see sprout.log",
                               finished_at=1790803001.0, seconds=1.0)
                (task_dir / "sprout.log").write_text("Traceback\n%s\n" % WORKTREE_ERROR)
            else:
                (task_dir / "sprout.log").write_text("harbor: branches of %s\n" % name)
                make_sprout_output(output, parent_journal=journal, parent_sha=digest, task="datacurve/" + name,
                                   live_branch=state == "live", plan_state="validated")
                if state == "complete":
                    receipt.update(exit_code=0, complete=True, finished_at=1790803600.0, seconds=600.0)
                    rows = json.loads((output / "summary.json").read_text())["attempts"]
                    write_json(task_dir / "sprout.task.json", {"task": name, "dataset": "deepswe", "complete": True,
                                                              "methods": {"sprout": {"status": "complete", "output": str(output),
                                                                                     "attempts": rows[1:]}},
                                                              "initial_resolved": False})
            write_json(task_dir / "sprout.stage.json", receipt)
        if metrics:
            write_json(sprout / "metrics-sprout.json", metrics_document(dict(manifest, method="sprout"), "sprout"))
    return root


def metrics_document(manifest: dict, method: str) -> dict:
    """The shape examples/qwen/report.py writes (two method keys for baseline)."""
    def block(rate):
        return {"denominator_all_tasks": 1, "completed_tasks": 1, "Acc": 0.0, "Success@1": 0.0, "Success@5": rate,
                "Success@8": rate, "initial_failure_recovery_rate": rate, "initial_failures_graded": 1,
                "successful_trajectories": int(rate), "graded_trajectories": 2, "positive_rate": rate / 2, "tasks": []}
    keys = ("pass@1", "baseline") if method == "baseline" else (method,)
    return {"schema_version": 2, "manifest": manifest, "methods": {k: block(1.0 if k != "pass@1" else 0.0) for k in keys},
            "comparison": [{"method": "SPROUT" if method == "sprout" else k, "Resolve Rate": 1.0, "Recovery": 1.0,
                            "Steps": 176, "Tokens (M)": 13.480539, "complete": True, "completed_tasks": 1,
                            "total_tasks": 1, "tokens_complete": True} for k in keys],
            "notes": ["Success@5/@8 for baseline are capped at its actual four attempts; no extrapolation."]}


@pytest.fixture
def experiment(tmp_path: Path) -> Path:
    """Four tasks: complete (attempt-1 FAIL -> retained, sprout recovered), errored (attempt-2 exit 1),
    live (attempt-3 running), not started; the runner is this process."""
    import os
    return make_experiment(tmp_path / "exp", lock_pid=os.getpid(), tasks={
        "alpha": {"attempts": [("complete", False), ("complete", True), ("complete", True), ("complete", False)],
                  "sprout": "complete"},
        "beta": {"attempts": [("complete", True), ("errored", None)], "sprout": "skipped"},
        "gamma": {"attempts": [("complete", False), ("complete", True), ("live", None)]},
        "delta": {},
    })

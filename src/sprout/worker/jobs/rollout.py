"""One isolated attempt process, with no database credentials or polling loop."""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

from sprout.contract.continuation import Continuation, prepare
from sprout.contract.mini_history import load_prefix, materialize
from sprout.contract.run_spec import RunSpec
from sprout.runstore.config import referenced_env, resolve
from sprout.runstore.payload import receive_payload
from sprout.worker.files import write_json
from sprout.worker.run.resources import ResourceLedger
from sprout.worker.run.runner import Orchestrator
from sprout.worker.sandbox.session import SandboxSession


class TrackingSession(SandboxSession):
    def __init__(self, *, ledger: ResourceLedger, **kwargs) -> None:
        super().__init__(**kwargs)
        self.ledger = ledger
        self.claim = None

    def create(self, image, resources=None):
        self.ledger._append("allocation_started", run_id="grader", image=image)
        ready = super().create(image, resources)
        if ready:
            self.ledger._append("claim", run_id="grader", kind="sandbox", id=self.sandbox_id)
        else:
            self.ledger._append("allocation_unknown", run_id="grader", image=image)
        return ready

    def destroy(self) -> None:
        identifier = self.sandbox_id
        super().destroy()
        if identifier:
            self.ledger._append("release", run_id="grader", kind="sandbox", id=identifier)


def execute(request: dict, directory: Path) -> dict:
    os.umask(0o077)
    spec = resolve(request["effective_spec"])
    profile = request["profile_config"]
    worker_env = resolve(profile.get("worker_env", {}))
    agent_env = resolve(profile.get("env", {}))
    os.environ.update(worker_env)
    os.environ.update(agent_env)
    hidden_env = (referenced_env(request) | set(worker_env)) - set(agent_env)
    cwd = directory / "cwd"
    cwd.mkdir(exist_ok=True)
    native_home = directory / "native-home"
    native_home.mkdir(exist_ok=True)
    ledger = ResourceLedger(directory / "resources.jsonl")
    if request["kind"] == "review":
        from sprout.worker.jobs.review.job import run_review

        return run_review(spec, directory)
    if request["kind"] == "grade":
        from sprout.worker.jobs.grade import grade

        return grade(
            spec, directory, session_factory=lambda **kwargs: TrackingSession(ledger=ledger, **kwargs)
        )
    slot = spec.get("slot", "mini-swe-agent")
    extra = dict(spec.get("extra", {}))
    extra["native_home"] = str(native_home)
    message_export = extra.get("rollout_contract", {}).get("message_export", False)
    if request.get("recovery"):
        recovered = request["recovery"]
        native = recovered["native"]
        if native["slot"] != slot:
            raise ValueError("Native prefix slot differs from requested slot")
        extra.update(materialize(native, cwd, directory / "restoration", native_home))
        # The same rules a Harbor branch trial follows (eval.harbor.branching).
        continuation = Continuation.from_dict(request.get("continuation"))
        prompt, options = prepare(continuation, load_prefix(extra["native_prefix"]))
        extra.update(options)
        if prompt:
            # Only a user-hint branch appends a message; the training export
            # strips what is marked, so the hint never becomes a target.
            if message_export:
                from sprout.contract.messages import mark_hint

                prompt = mark_hint(prompt)
            spec["prompt"] = prompt
        spec["sandbox_image"] = recovered["snapshot_id"]
        spec["origin"] = {
            "point_id": recovered["id"],
            "snapshot_id": recovered["snapshot_id"],
            "tool_depth": recovered["tool_depth"],
            "message_step": recovered["message_step"],
            **continuation.origin(),
        }
    spec.update(
        cwd=str(cwd),
        run_id=request["attempt_id"],
        agent_id=request["attempt_id"],
        journal_path=directory / "trajectory.jsonl",
        extra=extra,
    )
    ledger._append("allocation_started", run_id=request["attempt_id"], image=spec.get("sandbox_image"))

    def progress(kind: str, payload: dict) -> None:
        write_json(directory / "progress.json", {"phase": kind, **payload})

    class TrackedOrchestrator(Orchestrator):
        def _wire_gateway(self, run_spec, journal, task, run_id):
            task.env.update({name: "" for name in hidden_env})
            task.env.update(agent_env)
            return super()._wire_gateway(run_spec, journal, task, run_id)

        def _wire_sandbox(self, run_spec, claim):
            owned = super()._wire_sandbox(run_spec, claim)
            original_swap = owned.session.swap_sandbox

            def swap(snapshot):
                # Written ahead, as for the first sandbox: a child killed while
                # the replacement boots leaves a start that no claim answers,
                # which the worker reconciles. The orchestrator claims the
                # replacement itself once the session adopts it.
                ledger._append("allocation_started", run_id=request["attempt_id"], image=str(snapshot))
                changed = original_swap(snapshot)
                if not changed:
                    ledger._append("allocation_unknown", run_id=request["attempt_id"])
                return changed

            owned.session.swap_sandbox = swap
            return owned

    outcome = TrackedOrchestrator(out_dir=directory, ledger=ledger, on_event=progress).run(RunSpec(**spec))
    result = asdict(outcome)
    result["journal_path"] = str(outcome.journal_path)
    result["native_home"] = str(native_home)
    result["failure_kind"] = "actor" if outcome.status != "completed" else None
    if message_export:
        from sprout.worker.jobs.export import complete_message_result

        result = complete_message_result(
            result, directory, slot, request.get("recovery"), request.get("branch_point")
        )
    return result


def main() -> int:
    if len(sys.argv) != 3:
        return 2
    directory = Path(sys.argv[1]).resolve()
    try:
        request = receive_payload(sys.stdin.fileno(), sys.argv[2], directory.parent.name, directory.name)
    except (OSError, ValueError):
        print("Attempt bootstrap rejected: missing, invalid or timed-out stdin payload", file=sys.stderr)
        return 2
    started = time.monotonic()

    def terminate(signum, frame):
        raise KeyboardInterrupt("Worker requested shutdown")

    signal.signal(signal.SIGTERM, terminate)
    try:
        result = execute(request, directory)
    except BaseException as error:
        # The outcome keeps the type and message, which ``failure_kind`` reads;
        # worker.log gets the traceback and the error's notes (which call of a
        # review failed, and what else did). Nothing may keep the outcome unwritten.
        with contextlib.suppress(Exception):
            traceback.print_exception(error)
        result = {
            "status": "error",
            "failure_kind": "infrastructure",
            "error": f"{type(error).__name__}: {error}",
        }
    result.update(attempt_id=request["attempt_id"], elapsed_s=time.monotonic() - started)
    write_json(directory / "outcome.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

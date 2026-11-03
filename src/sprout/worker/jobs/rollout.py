"""The attempt process, which runs one attempt of a job and writes its ``outcome.json``.

The worker starts ``python -m sprout.worker.jobs.rollout <attempt directory> <request digest> <worker pid>``
and sends the request on stdin. The process runs the job's rollout, grade or review in the
attempt directory. It holds no database credentials and does not poll the Run Store. A SIGTERM
stops the attempt. The process also stops the attempt about 1 s after the worker process exits.
A stop after that exit gives an outcome that says ``worker_died``. If the teardown has not ended
``TEARDOWN_S`` after the exit, the process writes its outcome and exits. docs/RUNSTORE.md
("Supervision") gives the rules.
"""

from __future__ import annotations

import contextlib
import os
import signal
import sys
import threading
import time
import traceback
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sprout.contract.agents.history import load_prefix, materialize
from sprout.contract.continuation import Continuation, prepare
from sprout.contract.run_spec import KEEP_SANDBOX_S, RunSpec
from sprout.runstore.config import referenced_env, resolve
from sprout.runstore.payload import receive_payload
from sprout.worker.files import write_json
from sprout.worker.run.resources import ResourceLedger
from sprout.worker.run.runner import Runner, RunOutcome
from sprout.worker.sandbox.backends import with_sandbox_budget
from sprout.worker.sandbox.session import SandboxSession

if TYPE_CHECKING:
    from types import FrameType

    from sprout.worker.agents.slot import TaskSpec
    from sprout.worker.gateway import GatewayServer
    from sprout.worker.run.journal import JournalWriter
    from sprout.worker.run.resources import RunClaim
    from sprout.worker.run.rollout import RolloutControls
    from sprout.worker.run.runner import OwnedSandbox


class TrackingSession(SandboxSession):
    """A grading ``SandboxSession`` that records each sandbox it starts and destroys in ``ledger``."""

    def __init__(
        self,
        *,
        ledger: ResourceLedger,
        timeout: float = 300.0,
        quiet: bool = False,
        backend: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(timeout=timeout, quiet=quiet, backend=backend)
        self.ledger = ledger
        self.claim = None

    def create(self, image: str, resources: dict[str, Any] | None = None) -> bool:
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


def execute(request: dict[str, Any], directory: Path) -> dict[str, Any]:
    """The result of the attempt ``request``, run in the attempt ``directory``.

    It sets the process umask to 077 and exports the environment of the request's profile.
    The attempt records each sandbox that it starts in ``resources.jsonl`` under ``directory``.
    """
    os.umask(0o077)
    spec = resolve(request["effective_spec"])
    env = _export_profile_env(request["profile_config"], referenced=referenced_env(request))
    (directory / "cwd").mkdir(exist_ok=True)
    (directory / "native-home").mkdir(exist_ok=True)
    ledger = ResourceLedger(directory / "resources.jsonl")
    if request["kind"] == "review":
        from sprout.worker.jobs.review.job import run_review

        return run_review(spec, directory)
    if request["kind"] == "grade":
        from sprout.worker.jobs.grade import grade

        return grade(
            spec,
            directory,
            session_factory=lambda **kwargs: TrackingSession(ledger=ledger, **kwargs),
            kept_sandbox=request.get("kept_sandbox"),
        )
    return _rollout(request, spec, directory, ledger=ledger, env=env)


@dataclass(frozen=True)
class _AgentEnv:
    """The environment of a rollout's agent: each variable of ``values``, and each of ``hidden`` empty."""

    values: dict[str, str]
    hidden: frozenset[str]


def _export_profile_env(profile: dict[str, Any], *, referenced: set[str]) -> _AgentEnv:
    """Export the ``worker_env`` and the ``env`` of ``profile`` to this process, and return the agent's.

    The agent gets ``env``. Each other variable of ``worker_env`` or of ``referenced``, the names that the
    request refers to, is empty for it. Raises KeyError for a reference to an unset variable."""
    worker_env = resolve(profile.get("worker_env", {}))
    agent_env = resolve(profile.get("env", {}))
    os.environ.update(worker_env)
    os.environ.update(agent_env)
    return _AgentEnv(values=agent_env, hidden=frozenset((referenced | set(worker_env)) - set(agent_env)))


def _rollout(
    request: dict[str, Any], spec: dict[str, Any], directory: Path, *, ledger: ResourceLedger, env: _AgentEnv
) -> dict[str, Any]:
    """The result of the rollout ``request`` from its resolved effective ``spec``, run in ``directory``."""
    slot = spec.get("slot", "mini-swe-agent")
    fields = _run_fields(request, spec, directory, slot=slot)
    message_export = _exports_messages(fields["extra"])
    ledger._append("allocation_started", run_id=request["attempt_id"], image=fields.get("sandbox_image"))
    runner = _TrackedRunner(attempt_id=request["attempt_id"], directory=directory, ledger=ledger, env=env)
    outcome = runner.run(RunSpec(**_with_kept_budget(fields)))
    result = _rollout_result(
        outcome, native_home=directory / "native-home", keep_sandbox=bool(fields.get("keep_sandbox"))
    )
    if not message_export:
        return result
    from sprout.worker.jobs.export import complete_message_result

    return complete_message_result(result, directory, slot, recovery=request.get("recovery"))


def _exports_messages(extra: dict[str, Any]) -> bool:
    """Whether the rollout of the ``RunSpec.extra`` ``extra`` returns its trajectory as training messages."""
    return bool(extra.get("rollout_contract", {}).get("message_export", False))


def _run_fields(
    request: dict[str, Any], spec: dict[str, Any], directory: Path, *, slot: str
) -> dict[str, Any]:
    """The ``RunSpec`` fields of the rollout ``request`` in ``directory``, from its resolved ``spec``.

    A rollout with a recovery point starts there. Raises ValueError if the point's history is not of
    ``slot``."""
    extra = dict(spec.get("extra", {}))
    extra["native_home"] = str(directory / "native-home")
    fields = {**spec, "extra": extra}
    if request.get("recovery"):
        fields = _from_point(
            fields,
            request["recovery"],
            continuation=request.get("continuation"),
            directory=directory,
            slot=slot,
        )
    return {
        **fields,
        "cwd": str(directory / "cwd"),
        "run_id": request["attempt_id"],
        "agent_id": request["attempt_id"],
        "journal_path": directory / "trajectory.jsonl",
    }


def _from_point(
    fields: dict[str, Any], point: dict[str, Any], *, continuation: object, directory: Path, slot: str
) -> dict[str, Any]:
    """The ``RunSpec`` ``fields`` of a rollout that starts from the recovery ``point`` with ``continuation``.

    It restores the point's native history and files under ``directory``. Raises ValueError if the history
    is not of ``slot``, or if ``continuation`` is invalid."""
    native = point["native"]
    if native["slot"] != slot:
        raise ValueError("Native prefix slot differs from requested slot")
    extra = {
        **fields["extra"],
        **materialize(native, directory / "cwd", directory / "restoration", directory / "native-home"),
    }
    guidance = Continuation.from_dict(continuation)
    prompt, options = prepare(guidance, load_prefix(extra["native_prefix"]))
    restored = {
        **fields,
        "extra": {**extra, **options},
        "sandbox_image": point["snapshot_id"],
        "origin": {
            "point_id": point["id"],
            "snapshot_id": point["snapshot_id"],
            "tool_depth": point["tool_depth"],
            "message_step": point["message_step"],
            **guidance.origin(),
        },
    }
    if prompt:
        # Only a user-hint branch appends a message. The training export strips the marked
        # hint, so the hint never becomes a training target.
        from sprout.contract.messages import mark_hint

        restored["prompt"] = mark_hint(prompt) if _exports_messages(extra) else prompt
    return restored


def _with_kept_budget(fields: dict[str, Any]) -> dict[str, Any]:
    """The ``RunSpec`` ``fields``, with a backend budget that also covers the time that a sandbox is kept."""
    if not fields.get("keep_sandbox"):
        return fields
    # A microVM's lease must also cover the time that the sandbox is kept after the run.
    budget = fields.get("timeout_s", RunSpec.timeout_s) + KEEP_SANDBOX_S
    return {**fields, "backend": with_sandbox_budget(fields.get("backend") or {}, budget)}


class _TrackedRunner(Runner):
    """The ``Runner`` of the rollout attempt ``attempt_id`` in its attempt ``directory``.

    It writes each run event to ``progress.json``, and records the start of each replacement sandbox in
    ``ledger``. The agent gets the environment ``env``."""

    def __init__(self, *, attempt_id: str, directory: Path, ledger: ResourceLedger, env: _AgentEnv) -> None:
        self._attempt_id = attempt_id
        self._resources = ledger
        self._env = env
        super().__init__(out_dir=directory, ledger=ledger, on_event=self._write_progress)

    def _write_progress(self, kind: str, payload: dict[str, Any]) -> None:
        write_json(self.out_dir / "progress.json", {"phase": kind, **payload})

    def _wire_gateway(
        self,
        run_spec: RunSpec,
        journal: JournalWriter,
        task: TaskSpec,
        run_id: str,
        *,
        controls: RolloutControls | None,
    ) -> GatewayServer | None:
        task.env.update({name: "" for name in self._env.hidden})
        task.env.update(self._env.values)
        gateway: GatewayServer | None = super()._wire_gateway(
            run_spec, journal, task, run_id, controls=controls
        )
        return gateway

    def _wire_sandbox(self, run_spec: RunSpec, claim: RunClaim | None) -> OwnedSandbox:
        owned = super()._wire_sandbox(run_spec, claim)
        original_swap = owned.session.swap_sandbox

        def swap(snapshot: object) -> bool:
            # Record the start before the replacement boots, as for the first sandbox. The worker
            # warns of a start that no claim answers. ``Runner._follow_reboards`` claims the new one.
            self._resources._append("allocation_started", run_id=self._attempt_id, image=str(snapshot))
            changed: bool = original_swap(snapshot)
            if not changed:
                self._resources._append("allocation_unknown", run_id=self._attempt_id)
            return changed

        owned.session.swap_sandbox = swap
        return owned


def _rollout_result(outcome: RunOutcome, *, native_home: Path, keep_sandbox: bool) -> dict[str, Any]:
    """The job result of the run ``outcome``, with the attempt's ``native_home`` and the sandbox it kept."""
    result = asdict(outcome)
    result["journal_path"] = str(outcome.journal_path)
    result["native_home"] = str(native_home)
    result["failure_kind"] = "actor" if outcome.status != "completed" else None
    if keep_sandbox and outcome.final_snapshot_id:
        result["kept_sandbox"] = _kept_sandbox(outcome)
    return result


def _kept_sandbox(outcome: RunOutcome) -> dict[str, str | None]:
    """The ``kept_sandbox`` of a job result: the sandbox ``outcome`` kept, and until when (UTC, ISO-8601)."""
    keep_until = datetime.now(UTC) + timedelta(seconds=KEEP_SANDBOX_S)
    return {"sandbox_id": outcome.sandbox_id, "keep_until": keep_until.isoformat()}


#: How often, in seconds, the attempt process checks that its worker process still runs. Arbitrary.
WORKER_CHECK_S = 1.0
#: How long, in seconds, the teardown can take after the worker process exited. Arbitrary. A worker
#: that lives ends a hung teardown itself, with SIGKILL 10 s after its SIGTERM.
TEARDOWN_S = 60.0
#: The exit status of an attempt process whose teardown did not end in time. Arbitrary, but not 0, the
#: exit after a teardown, or 2, a rejected start.
_TEARDOWN_UNFINISHED = 3
#: What the outcome says when the worker process exited during the attempt, over what the run reported.
#: The failure is the infrastructure's, so a worker that settles the job runs it again.
_WORKER_DIED = {
    "status": "error",
    "failure_kind": "infrastructure",
    "stop_reason": "worker_died",
    "error": "Worker process exited",
}


class _Outcome:
    """The ``outcome.json`` of the attempt ``attempt_id``, which the first ``write`` writes.

    Any thread can write. Once ``worker_died`` is set, the outcome says ``_WORKER_DIED`` over its result."""

    def __init__(self, path: Path, *, attempt_id: str) -> None:
        self._path = path
        self._attempt_id = attempt_id
        self._started = time.monotonic()
        self._lock = threading.Lock()
        self._written = False
        #: Set when the worker process exited during the attempt.
        self.worker_died = threading.Event()

    def write(self, result: dict[str, Any], *, wait: bool = True) -> None:
        """Write ``result`` with the attempt id and the seconds since the start, unless an outcome is written.

        Without ``wait``, it writes nothing while another thread writes."""
        if not self._lock.acquire(blocking=wait):
            return
        try:
            if self._written:
                return
            if self.worker_died.is_set():
                result = {**result, **_WORKER_DIED}
            elapsed = time.monotonic() - self._started
            write_json(self._path, {**result, "attempt_id": self._attempt_id, "elapsed_s": elapsed})
            self._written = True
        finally:
            self._lock.release()


def _stop_after_worker(worker_pid: int, outcome: _Outcome, *, teardown_s: float, check_s: float) -> None:
    """Start a daemon thread that stops the attempt once the process ``worker_pid`` is not the parent.

    The thread checks the parent every ``check_s``. Then it sets ``outcome.worker_died`` and sends the calling
    thread one SIGTERM. After ``teardown_s``, it writes the outcome unless one is written, and ends this
    process with exit status 3. Call it from the main thread."""
    caller = threading.get_ident()

    def watch() -> None:
        # The kernel gives this process a new parent when the worker process exits. PR_SET_PDEATHSIG would
        # follow the worker thread that started this process, not the worker process.
        while os.getppid() == worker_pid:
            time.sleep(check_s)
        outcome.worker_died.set()
        # A signal to the main thread interrupts its blocking call, and Python runs the handler there.
        signal.pthread_kill(caller, signal.SIGTERM)
        time.sleep(teardown_s)
        try:
            # os._exit flushes no stream.
            print(
                "The teardown did not end %g s after the worker process exited" % teardown_s,
                file=sys.stderr,
                flush=True,
            )
            outcome.write({}, wait=False)
        finally:
            os._exit(_TEARDOWN_UNFINISHED)

    threading.Thread(target=watch, name="worker-watch", daemon=True).start()


def _interrupt_once(worker_pid: int, outcome: _Outcome) -> Callable[[int, FrameType | None], None]:
    """A SIGTERM handler that raises KeyboardInterrupt at its first call, and ignores each later call.

    A later SIGTERM then cannot cut short the teardown that the first one started. If the process
    ``worker_pid`` is no longer the parent, it sets ``outcome.worker_died``, and the message says so."""
    interrupted = False

    def interrupt(signum: int, frame: FrameType | None) -> None:
        nonlocal interrupted
        if interrupted:
            return
        interrupted = True
        if os.getppid() != worker_pid:
            # A restarted worker of the same id can send this SIGTERM before the watch sees the exit.
            outcome.worker_died.set()
            raise KeyboardInterrupt("Worker process exited")
        raise KeyboardInterrupt("Worker requested shutdown")

    return interrupt


def main(*, teardown_s: float = TEARDOWN_S, worker_check_s: float = WORKER_CHECK_S) -> int:
    """Run the attempt in ``sys.argv[1]``. Returns 0 once ``outcome.json`` is written, else 2.

    A SIGTERM stops the attempt, and so does the exit of the worker process ``sys.argv[3]``, checked every
    ``worker_check_s``. If the teardown has not ended ``teardown_s`` after that exit, the process writes
    its outcome and exits with 3."""
    if len(sys.argv) != 4 or not sys.argv[3].isdecimal():
        return 2
    directory = Path(sys.argv[1]).resolve()
    worker_pid = int(sys.argv[3])
    try:
        request = receive_payload(sys.stdin.fileno(), sys.argv[2], directory.parent.name, directory.name)
    except (OSError, ValueError):
        print("Attempt bootstrap rejected: missing, invalid or timed-out stdin payload", file=sys.stderr)
        return 2
    outcome = _Outcome(directory / "outcome.json", attempt_id=request["attempt_id"])
    signal.signal(signal.SIGTERM, _interrupt_once(worker_pid, outcome))
    try:
        # Inside the try: the SIGTERM of the watch can come at its first check, before execute starts.
        _stop_after_worker(worker_pid, outcome, teardown_s=teardown_s, check_s=worker_check_s)
        result = execute(request, directory)
    except BaseException as error:
        # The outcome keeps the type and message, which ``failure_kind`` reads. ``worker.log`` gets the
        # traceback and the error's notes, and a failure to print them must not keep the outcome unwritten.
        with contextlib.suppress(Exception):
            traceback.print_exception(error)
        result = {
            "status": "error",
            "failure_kind": "infrastructure",
            "error": f"{type(error).__name__}: {error}",
        }
    outcome.write(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

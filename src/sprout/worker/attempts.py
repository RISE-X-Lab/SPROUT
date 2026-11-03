"""The attempts of claimed Run Store jobs, each run as a child process on this host.

:class:`Worker` runs an attempt, removes its sandboxes, and publishes its outcome. A sandbox
whose removal it cannot confirm stays recorded, for :meth:`Worker.clean_up` to free later. A
job that runs again starts from its spec, after the worker deleted the data of the attempt
before. :meth:`Worker.settle` settles a job that an earlier worker process left. A
:class:`Watchdog` stops an attempt's process at its deadline, on a stop request, at shutdown
and when the worker is cut off. Failure handling: ``docs/RUNSTORE.md``.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path
from threading import Event, Lock, Thread
from typing import IO, Any, BinaryIO, cast

from sprout.contract.backend import backend_name, default_podman_store, snapshot_store
from sprout.contract.messages import failure_kind, is_truncated_result
from sprout.runstore.config import merge, referenced_env, resolve, snapshot_validator
from sprout.runstore.index import Index
from sprout.runstore.payload import PayloadUnavailable, encode_payload, send_payload
from sprout.runstore.specs import defaults_key, digest, validate_continuation
from sprout.runstore.store import Store
from sprout.worker.agents.history import index_native
from sprout.worker.cleanup import LEFTOVERS_NAME, Leftovers, forget, log_section, with_log
from sprout.worker.files import read_json, write_json
from sprout.worker.kept import LEDGER_NAME, KeptSandboxes
from sprout.worker.run.journal import journal_events
from sprout.worker.run.resources import ResourceLedger
from sprout_sandbox.podman_cli import default_podman_bin

#: The backends a queued job can run on. Their sandboxes are on this host and owned by the
#: attempt, so the worker can establish afterwards that each one is gone.
OWNED_BACKENDS = ("podman", "microvm")
#: The variables of the worker's environment that an attempt child inherits. With the podman and
#: store variables, the child makes its sandboxes on the podman that the worker later inspects.
CHILD_ENV_PASSTHROUGH = frozenset(
    {
        "PATH",
        "HOME",
        "USER",
        "LANG",
        "LC_ALL",
        "LD_LIBRARY_PATH",
        "SSL_CERT_FILE",
        "SSL_CERT_DIR",
        "SPROUT_PODMAN_BIN",
        "SPROUT_PODMAN_STORE",
    }
)
#: How often, at most, in seconds, an attempt writes its progress to the Run Store. Arbitrary: a
#: stop request waits up to this long.
PROGRESS_S = 5.0
#: Longer than ``stop_process`` can take, in seconds: 10 s after SIGTERM and 5 s after SIGKILL.
_STOP_S = 16.0
#: The stop reasons after which a job whose child has not started goes back to the queue.
_RELEASED_BEFORE_START = ("worker_shutdown", "worker_cut_off")

_log = logging.getLogger(__name__)


def local_worker_id(name: str) -> str:
    """The id of the worker ``name`` on this host: ``<host>/<name>``, with the host's node name."""
    return f"{os.uname().nodename}/{name}"


def owned_backend(kind: str, effective: dict[str, Any]) -> str:
    """The backend that a job's effective spec runs on.

    Raises ``ValueError`` if it is not in ``OWNED_BACKENDS``, or if a rollout has no ``sandbox_image``.
    """
    if kind == "rollout" and not effective.get("sandbox_image"):
        raise ValueError("A queued rollout needs a sandbox_image")
    name = backend_name(effective.get("backend") or {})
    if name not in OWNED_BACKENDS:
        raise ValueError(
            f"Queued jobs run on an owned sandbox, podman (the default) or microvm, not {name!r}"
        )
    return name


def apply_branch_budget(effective: dict[str, Any], budget: dict[str, Any] | None) -> dict[str, Any]:
    """The effective spec with a branch's own contract limits laid over the inherited ones.

    Raises ``ValueError`` if ``budget`` is not empty and the spec has no rollout contract.
    """
    if not budget:
        return effective
    contract = effective.get("extra", {}).get("rollout_contract")
    if contract is None:
        raise ValueError("A branch budget bounds a rollout contract; this run has none")
    return {**effective, "extra": {**effective["extra"], "rollout_contract": {**contract, **budget}}}


def podman_reclaimed(claims: dict[str, str | None], backend: dict[str, Any]) -> bool:
    """True if every claimed podman sandbox is gone: no container, network or rootfs remains.

    ``claims`` maps a sandbox id to the store of its rootfs, or to None for the backend's store.
    A leftover is deleted as ``python -m sprout.worker reap`` deletes it, then checked again.
    """
    from sprout.worker.run.reap import PodmanClient

    section = backend.get("podman") or {}
    default = snapshot_store(backend) or default_podman_store()
    for identifier, store in sorted(claims.items()):
        client = PodmanClient(
            podman_bin=section.get("podman_bin") or default_podman_bin(),
            stores=[path for path in (store, default) if path],
        )
        if client.leftovers(identifier) and (
            not client.delete_sandbox(identifier) or client.leftovers(identifier)
        ):
            return False
    return True


def microvm_reclaimed(claims: list[str], microvm: dict[str, Any]) -> bool:
    """True if every claimed microVM is deleted and AgentENV answers 404 for it.

    False if ``microvm`` names no ``server_url``.
    """
    import httpx

    if not microvm.get("server_url"):
        return False
    for identifier in claims:
        url = microvm["server_url"].rstrip("/") + "/sandboxes/" + identifier
        headers = {"X-API-Key": microvm.get("api_key", "")}
        try:
            response = httpx.delete(url, headers=headers, timeout=10)
            if response.status_code not in {200, 204, 404}:
                return False
            if httpx.get(url, headers=headers, timeout=5).status_code != 404:
                return False
        except httpx.HTTPError:
            return False
    return True


def process_identity(pid: int) -> dict[str, Any] | None:
    """What identifies the live process ``pid``: its pid, start time and boot id. None if it is gone.

    A zombie counts as gone. A recycled pid has another start time.
    """
    try:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(") ", 1)[1].split()
        if fields[0] == "Z":
            return None
        return {
            "pid": pid,
            "start_ticks": fields[19],
            "boot_id": Path("/proc/sys/kernel/random/boot_id").read_text().strip(),
        }
    except (OSError, IndexError):
        return None


def stop_process(identity: dict[str, Any] | None) -> bool:
    """Stop the process group that ``identity`` leads, and return True if no process of it remains.

    Returns True at once for None, an identity from another boot or a recycled pid. Returns False
    if the live pid does not lead its group. Sends SIGTERM, and SIGKILL after 10 s.
    """
    if identity is None:
        return True
    pid = identity["pid"]
    current = process_identity(pid)
    boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
    if identity["boot_id"] != boot_id or (current is not None and current != identity):
        return True
    if current is not None:
        try:
            if os.getpgid(pid) != pid:
                return False
        except ProcessLookupError:
            pass

    def group_alive() -> bool:
        for entry in Path("/proc").iterdir():
            if not entry.name.isdecimal():
                continue
            try:
                fields = (entry / "stat").read_text().rsplit(") ", 1)[1].split()
                if fields[0] != "Z" and int(fields[2]) == pid:
                    return True
            except (OSError, ValueError, IndexError):
                continue
        return False

    if not group_alive():
        return True
    try:
        os.killpg(pid, signal.SIGTERM)
    except ProcessLookupError:
        return True
    deadline = time.monotonic() + 10
    while group_alive() and time.monotonic() < deadline:
        time.sleep(0.1)
    if group_alive():
        try:
            os.killpg(pid, signal.SIGKILL)
        except ProcessLookupError:
            return True
        deadline = time.monotonic() + 5
        while group_alive() and time.monotonic() < deadline:
            time.sleep(0.05)
    return not group_alive()


def _raised_result(error: BaseException, *, reason: str | None) -> dict[str, Any]:
    """The result of an attempt that raised ``error`` before its outcome, with the watchdog's ``reason``."""
    stop_reason = reason or (str(error) if isinstance(error, AttemptStopped) else None)
    result = {
        "status": "timeout" if stop_reason == "timeout" else "error",
        "failure_kind": "infrastructure",
        "stop_reason": stop_reason,
        "error": f"{type(error).__name__}: {error}",
    }
    return {**result, "failure_kind": failure_kind(result)}


def _within_contract(kind: str, effective: dict[str, Any]) -> dict[str, Any]:
    """The effective spec of a rollout with a rollout contract, with its time bounded by the contract.

    The time in the queue belongs to the group deadline too, so the child's whole run is bounded.
    Raises ValueError if the deadline passed. Another spec is returned as it is.
    """
    contract = effective.get("extra", {}).get("rollout_contract")
    if kind != "rollout" or contract is None:
        return effective
    from sprout.worker.run.rollout import remaining_timeout

    effective = {**effective, "timeout_s": remaining_timeout(contract, effective.get("timeout_s", 3600))}
    if contract.get("message_export"):
        # The soft deadline also covers this actor's own time cap.
        deadline = min(contract["deadline_at"], time.time() + effective["timeout_s"])
        effective["extra"] = {**effective["extra"], "rollout_contract": {**contract, "deadline_at": deadline}}
    return effective


def _child_timeout(kind: str, effective: dict[str, Any]) -> float:
    """The seconds that the attempt child may run: ``timeout_s``, and more for what it does after the run.

    A grade gets 1200 s for its preparation and collection. A message-export rollout gets the time to
    write its export.
    """
    timeout = float(effective.get("timeout_s", 3600)) + (1200 if kind == "grade" else 0)
    if effective.get("extra", {}).get("rollout_contract", {}).get("message_export"):
        from sprout.contract.messages import TERMINATION_GRACE_SECONDS

        timeout += TERMINATION_GRACE_SECONDS
    return timeout


def _outcome(directory: Path, reason: str | None, started: float) -> dict[str, Any]:
    """The result of the attempt whose child exited: its ``outcome.json``, after a stop with the ``reason``.

    ``started`` is the monotonic time of the child's start. A child without an outcome failed on the
    infrastructure.
    """
    result = read_json(directory / "outcome.json") or {
        "status": "error",
        "failure_kind": "infrastructure",
        "error": "Attempt process exited without outcome",
        "elapsed_s": time.monotonic() - started,
    }
    if reason is None:
        return result
    # A requested stop is nobody's failure and is never run again.
    return {
        **result,
        "status": "timeout" if reason == "timeout" else "error",
        "failure_kind": "stopped" if reason == "stopped" else "infrastructure",
        "error": f"Attempt stopped: {reason}",
        "stop_reason": reason,
    }


def _left_outcome(directory: Path) -> dict[str, Any]:
    """The ``outcome.json`` that an attempt left in ``directory``, or a lost attempt's result without one."""
    try:
        outcome = read_json(directory / "outcome.json")
    except ValueError:
        outcome = None
    return outcome or {
        "status": "error",
        "failure_kind": "infrastructure",
        "error": "Attempt left no outcome",
    }


class Worker:
    """Runs the attempts of the Run Store jobs that this host claimed, each in a child process.

    An attempt of a job that runs again leaves no data. A sandbox whose removal the worker cannot confirm
    goes to ``leftovers``. docs/RUNSTORE.md has the rest.
    """

    def __init__(
        self,
        store: Store,
        config: dict[str, Any],
        *,
        worker_id: str | None = None,
        progress_s: float = PROGRESS_S,
        snapshot_valid: Callable[[dict[str, Any]], bool] | None = None,
        grades: tuple[tuple[str, str], ...] = (),
    ) -> None:
        """``config`` is the Run Store config. Its ``artifact_root`` holds a directory for each attempt.

        ``worker_id`` defaults to ``local_worker_id("default")``. ``snapshot_valid`` tells whether the
        snapshot of a recovery point still exists. The default asks the backend of the point's profile.
        """
        self.store = store
        self.config = config
        self.root = Path(config["artifact_root"]).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        #: The host that the worker's attempts run on.
        self.host = os.uname().nodename
        self.worker_id = worker_id or local_worker_id("default")
        #: The ``(dataset_sha256, grader_revision)`` pairs of the ``swe-rebench-v2`` grades that the
        #: worker claims (``profiles.checked_grades``).
        self.grades = grades
        self.progress_s = progress_s
        self.index = Index(store, snapshot_valid or snapshot_validator(store, config))
        #: The sandboxes that this host's rollouts keep for a grade in place.
        self.kept = KeptSandboxes(self.root / LEDGER_NAME)
        #: The sandboxes whose removal after an attempt the worker could not confirm.
        self.leftovers = Leftovers(self.root / LEFTOVERS_NAME)

    def _scope(self, request: dict[str, Any], envelope: dict[str, Any] | None = None) -> dict[str, Any]:
        recovery = (envelope or {}).get("recovery") or {}
        return {
            "context": request["context"],
            "run_spec_hash": digest(request["spec"]),
            "parent_point": request.get("parent_point"),
            "recovery_snapshot": recovery.get("snapshot_id"),
            "native_prefix_hash": recovery.get("native", {}).get("sha256"),
        }

    def _ingest(self, job: dict[str, Any], directory: Path, envelope: dict[str, Any]) -> list[dict[str, Any]]:
        events = journal_events(directory / "trajectory.jsonl")
        if not events:
            return []
        self.store.append_events(job["id"], job["lease_token"], events)
        refs = [
            event["native_session_id"]
            for event in events
            if event.get("type") == "session.ref" and event.get("native_session_id")
        ]
        points = []
        if refs:
            session_id = refs[-1]
            home = directory / "native-home"
            slot = envelope["effective_spec"].get("slot", "mini-swe-agent")
            transcripts = list(home.glob(f"**/*{session_id}.jsonl"))
            if len(transcripts) == 1:
                try:
                    points = index_native(
                        directory / "trajectory.jsonl",
                        transcripts[0],
                        slot,
                        session_id,
                        events=events,
                        inherited_native=(envelope.get("recovery") or {}).get("native"),
                    )
                except (ValueError, OSError) as error:
                    write_json(directory / "index-error.json", {"error": str(error), "slot": slot})
        self.index.project(
            job["id"], job["lease_token"], self._scope(job["request"], envelope), events, points
        )
        return events

    def _cleanup(self, directory: Path, effective: dict[str, Any]) -> bool:
        records = journal_events(directory / "resources.jsonl")
        claims = {
            row["id"]: row.get("store")
            for row in records
            if row.get("event") == "claim" and row.get("kind") == "sandbox"
        }
        started = sum(row.get("event") == "allocation_started" for row in records)
        if any(row.get("event") == "allocation_unknown" for row in records) or started > len(claims):
            return False
        # A sandbox kept for a grade in place is the reaper's to remove once its time passes.
        held = self.kept.held()
        claims = {identifier: store for identifier, store in claims.items() if identifier not in held}
        if not claims:
            return True
        backend = resolve(effective.get("backend") or {})
        if backend_name(backend) == "podman":
            return podman_reclaimed(claims, backend)
        return microvm_reclaimed(sorted(claims), backend.get("microvm", {}))

    def _publish(
        self, job: dict[str, Any], directory: Path, result: dict[str, Any], envelope: dict[str, Any]
    ) -> None:
        """End the attempt in the Run Store with its ``result``, after the removal of its sandboxes.

        A shutdown releases the job, and an infrastructure failure runs it again while attempts remain.
        docs/RUNSTORE.md ("Failure handling") gives each ending.
        """
        effective = envelope["effective_spec"]
        truncated = is_truncated_result(result)
        result = {**result, "failure_kind": None if truncated else failure_kind(result)}
        succeeds = result.get("status") == "completed" or truncated
        shutdown = result.get("stop_reason") == "worker_shutdown"
        runs_again = result["failure_kind"] == "infrastructure" and job["attempt_count"] < job["max_attempts"]
        if not succeeds and (shutdown or runs_again):
            self._end_to_run_again(job, directory, result, effective)
            return
        self._ingest(job, directory, envelope)
        result = self._keep(job, result, effective, succeeds=succeeds)
        self._remove_sandboxes(directory, effective)
        if succeeds:
            self.store.finish(job["id"], job["lease_token"], result)
            return
        salvaged = self._salvage(job, directory, result, envelope)
        if salvaged is not None:
            self.store.finish(job["id"], job["lease_token"], salvaged)
            return
        rerun = result["failure_kind"] == "infrastructure"
        self.store.finish(job["id"], job["lease_token"], result, state="failed", rerun=rerun)

    def _end_to_run_again(
        self, job: dict[str, Any], directory: Path, result: dict[str, Any], effective: dict[str, Any]
    ) -> None:
        """Release the job after a shutdown, or end the attempt as lost, once its sandboxes and data are gone.

        The Run Store keeps the tail of the attempt's log in the attempt's error.
        """
        result = self._keep(job, result, effective, succeeds=False)
        self._remove_sandboxes(directory, effective)
        log = forget(directory, resolve(effective.get("backend") or {}))
        result = {**result, "error": with_log(result.get("error"), log)}
        if result.get("stop_reason") == "worker_shutdown":
            self.store.release(job["id"], job["lease_token"], result)
        else:
            self.store.finish(job["id"], job["lease_token"], result, state="failed", rerun=True)

    def _remove_sandboxes(self, directory: Path, effective: dict[str, Any]) -> None:
        """Remove each sandbox that the attempt in ``directory`` claims, or record it in ``leftovers``.

        A sandbox that a rollout keeps for a grade stays. ``clean_up`` removes a recorded one later.
        """
        if self._cleanup(directory, effective):
            return
        _log.warning("The worker could not confirm that the sandboxes of %s are gone", directory)
        self.leftovers.leave(
            directory, backend=resolve(effective.get("backend") or {}), held=self.kept.held()
        )

    def clean_up(self) -> None:
        """Free the leftover sandboxes, and delete the local data of each lost attempt of this worker.

        A lost attempt's child is stopped first. Then the Run Store records that the attempt's data is
        deleted, with the tail of its log in its error. An attempt without a directory counts as deleted.
        """
        for attempt in self.store.lost_attempts(self.worker_id):
            directory = self.root / attempt["job_id"] / attempt["id"]
            if not directory.exists():
                # The worker deletes the directory before the attempt's job runs again, or never made it.
                self.store.note_lost_data_deleted(attempt["id"], log="")
                continue
            stop_process(attempt["process"])
            try:
                effective = self.store.payload(attempt["job_id"], attempt["id"])["effective_spec"]
            except PayloadUnavailable:
                # The payload is frozen before the child starts, so no sandbox started.
                effective = {}
            self._remove_sandboxes(directory, effective)
            log = forget(directory, resolve(effective.get("backend") or {}))
            self.store.note_lost_data_deleted(attempt["id"], log=log_section(log))
        self.leftovers.free()

    def _in_place(self, job: dict[str, Any], effective: dict[str, Any], directory: Path) -> dict[str, Any]:
        """What the envelope of a grade in a kept sandbox adds: ``kept_sandbox``, None when it is gone.

        The attempt takes the sandbox over from the kept sandboxes, so its cleanup removes it.
        """
        if job["kind"] != "grade" or effective.get("in_place_job") is None:
            return {}
        kept = (self.store.get(effective["in_place_job"])["result"] or {}).get("kept_sandbox")
        if kept is None:
            return {"kept_sandbox": None}
        with ResourceLedger(directory / "resources.jsonl").run("grader") as claim:
            taken = self.kept.take(kept["sandbox_id"], claim=claim)
        return {"kept_sandbox": kept if taken else None}

    def _give_back(self, in_place: dict[str, Any], directory: Path) -> None:
        """Give the sandbox that ``_in_place`` took over back to the kept sandboxes, if it took one.

        For an attempt whose child never started: no cleanup removes the sandbox, and no grade runs in it.
        """
        kept = in_place.get("kept_sandbox")
        if kept is None:
            return
        with ResourceLedger(directory / "resources.jsonl").run("grader") as claim:
            self.kept.give_back(kept["sandbox_id"], claim=claim)

    def _keep(
        self, job: dict[str, Any], result: dict[str, Any], effective: dict[str, Any], *, succeeds: bool
    ) -> dict[str, Any]:
        """``result`` with its kept sandbox recorded for the reaper, or dropped when the job does not succeed.

        A dropped sandbox is not graded in place, so the cleanup after the attempt removes it.
        """
        kept = result.get("kept_sandbox")
        if kept is None:
            return result
        if not succeeds:
            return {key: value for key, value in result.items() if key != "kept_sandbox"}
        # Recorded before the cleanup, which then leaves it to the reaper.
        self.kept.keep(
            sandbox_id=kept["sandbox_id"],
            keep_until=kept["keep_until"],
            attempt_id=job["active_attempt"],
            backend=resolve(effective.get("backend") or {}),
        )
        return result

    def _salvage(
        self, job: dict[str, Any], directory: Path, result: dict[str, Any], envelope: dict[str, Any]
    ) -> dict[str, Any] | None:
        """The result of a message-export rollout that will not run again, cut back to its last valid point.

        The work up to that point is the policy's own, so it is graded. None for another job, or a
        rollout without a valid point.
        """
        from sprout.worker.jobs.export import salvage_interrupted

        contract = envelope["effective_spec"].get("extra", {}).get("rollout_contract", {})
        if job["kind"] != "rollout" or not contract.get("message_export"):
            return None
        point = next(
            (
                point
                for point in reversed(self.index.points(job["active_attempt"]))
                if self.index.valid(point)
            ),
            None,
        )
        if point is None:
            return None
        return salvage_interrupted(result, directory, point, recovery=envelope.get("recovery"))

    def run_claimed(
        self, job: dict[str, Any], *, stop: Event | None = None, cut_off: Callable[[], bool] | None = None
    ) -> str:
        """Run one attempt of the claimed ``job``, end that attempt in the Run Store, and return the job's id.

        ``stop`` (a shutdown) releases the job, and so does a True ``cut_off()`` before the child starts.
        Another failure after the child started runs the job again from its spec, while attempts remain.
        """
        directory = self.root / job["id"] / job["active_attempt"]
        process: subprocess.Popen[bytes] | None = None
        envelope: dict[str, Any] = {}
        in_place: dict[str, Any] = {}
        watchdog = Watchdog(
            stop_process,
            stop if stop is not None else Event(),
            cut_off=cut_off if cut_off is not None else lambda: False,
        )
        try:
            watchdog.check()
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            effective, profile, recovery = self._effective(job)
            in_place = self._in_place(job, effective, directory)
            envelope = self._freeze(
                job, effective=effective, profile=profile, recovery=recovery, in_place=in_place
            )
            with (directory / "worker.log").open("w") as log:
                watchdog.check()
                started = time.monotonic()
                process = self._spawn(envelope, directory, log)
                self._follow(job, process, directory, envelope, watchdog, started=started)
            self._publish(job, directory, _outcome(directory, watchdog.reason, started), envelope)
        except BaseException as error:
            result = _raised_result(error, reason=watchdog.reason)
            if process is None:
                self._end_unstarted(job, directory, result, in_place)
            else:
                self._end_lost(job, directory, result, process=process, watchdog=watchdog, envelope=envelope)
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            watchdog.close()
        job_id: str = job["id"]
        return job_id

    def _effective(self, job: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        """The effective spec of the job's attempt, the profile, and the recovery point of a branch or None.

        Raises KeyError for an unknown profile or point, and ValueError for a backend that the worker does
        not own, an invalid point, or a passed group deadline.
        """
        request = job["request"]
        profile = self.config["profiles"][request["profile"]]
        effective = merge(profile.get(defaults_key(job["kind"]), {}), request["spec"])
        if job["kind"] == "review":
            # A review reads attempt directories on this host and runs nothing in a sandbox.
            effective = {**effective, "artifact_root": str(self.root)}
        else:
            owned_backend(job["kind"], effective)
        # Each attempt starts where the spec starts: a branch at its parent_point, any other job from scratch.
        recovery = self.index.get_point(request["parent_point"]) if request.get("parent_point") else None
        if recovery:
            validate_continuation(self.store.get(recovery["job_id"])["request"], request)
            if not self.index.valid(recovery):
                raise ValueError("Native prefix or snapshot invalid before dispatch")
        effective = apply_branch_budget(effective, request.get("branch_budget"))
        return _within_contract(job["kind"], effective), profile, recovery

    def _freeze(
        self,
        job: dict[str, Any],
        *,
        effective: dict[str, Any],
        profile: dict[str, Any],
        recovery: dict[str, Any] | None,
        in_place: dict[str, Any],
    ) -> dict[str, Any]:
        """The payload of the job's attempt, frozen in the Run Store, as the Run Store stores it."""
        envelope = {
            "version": 1,
            "job_id": job["id"],
            "attempt_id": job["active_attempt"],
            "kind": job["kind"],
            "effective_spec": effective,
            "profile_config": profile,
            "recovery": recovery,
            "continuation": job["request"].get("continuation") if recovery else None,
            **in_place,
        }
        return self.store.freeze_payload(job["id"], job["lease_token"], envelope)

    def _spawn(self, envelope: dict[str, Any], directory: Path, log: IO[str]) -> subprocess.Popen[bytes]:
        """Start the attempt child in its own session, with its output in ``log``. It awaits its payload.

        The child gets the pid of this process, and stops the attempt once this process exits."""
        env = {key: value for key, value in os.environ.items() if key in CHILD_ENV_PASSTHROUGH}
        for name in referenced_env(envelope):
            env[name] = os.environ[name]
        repo = Path(__file__).resolve().parents[3]
        env["PYTHONPATH"] = str(repo / "src") + ":" + str(repo)
        return subprocess.Popen(
            [
                envelope["profile_config"].get("python", sys.executable),
                "-m",
                "sprout.worker.jobs.rollout",
                str(directory),
                digest(envelope),
                str(os.getpid()),
            ],
            cwd=repo,
            env=env,
            stdin=subprocess.PIPE,
            bufsize=0,
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

    def _follow(
        self,
        job: dict[str, Any],
        process: subprocess.Popen[bytes],
        directory: Path,
        envelope: dict[str, Any],
        watchdog: Watchdog,
        *,
        started: float,
    ) -> None:
        """Record the child, send its payload, and report its progress until it exits or someone stops it.

        It writes the progress at most every ``progress_s``. A stop request makes ``watchdog`` stop the child.
        """
        identity = process_identity(process.pid)
        if identity is None:
            raise RuntimeError("Attempt child exited before process registration")
        timeout = _child_timeout(job["kind"], envelope["effective_spec"])
        watchdog.arm(identity, started + timeout, running=lambda: process.poll() is None)
        self.store.progress(
            job["id"],
            job["lease_token"],
            phase="starting",
            execution={
                "directory": str(directory),
                "process": identity,
                "request_hash": digest(envelope),
                "scope": self._scope(job["request"], envelope),
            },
        )
        send_payload(
            # stdin=PIPE makes it a binary stream, never None.
            cast(BinaryIO, process.stdin),
            encode_payload(envelope, job["id"], job["active_attempt"]),
            timeout_s=min(30, timeout - (time.monotonic() - started)),
        )
        report_at = time.monotonic()
        while process.poll() is None and watchdog.reason is None:
            if time.monotonic() >= report_at:
                report_at = time.monotonic() + self.progress_s
                if self._report(job, directory, envelope):
                    watchdog.request_stop("stopped")
            time.sleep(min(1.0, self.progress_s))
        process.wait(timeout=_STOP_S)

    def _end_unstarted(
        self, job: dict[str, Any], directory: Path, result: dict[str, Any], in_place: dict[str, Any]
    ) -> None:
        """End the attempt whose child never started: release the job after a stop, else fail it.

        A failure to reach the Run Store is logged. The job then stays running until a worker settles it.
        """
        try:
            # Before the release, so that the job's next attempt can take the sandbox over again.
            self._give_back(in_place, directory)
            if result["stop_reason"] in _RELEASED_BEFORE_START:
                forget(directory, {})
                self.store.release(job["id"], job["lease_token"], result)
            else:
                self.store.finish(job["id"], job["lease_token"], result, state="failed")
        except Exception:
            _log.exception("Could not end attempt %s of job %s", job["active_attempt"], job["id"])

    def _end_lost(
        self,
        job: dict[str, Any],
        directory: Path,
        result: dict[str, Any],
        *,
        process: subprocess.Popen[bytes],
        watchdog: Watchdog,
        envelope: dict[str, Any],
    ) -> None:
        """Stop the child of an attempt that failed after the child started, and publish the attempt as lost.

        A failure to stop the child or to reach the Run Store is logged.
        """
        try:
            if process.stdin is not None:
                process.stdin.close()
            stop_process(watchdog.identity)
            process.wait(timeout=_STOP_S)
            self._publish(job, directory, {**result, "failure_kind": "infrastructure"}, envelope)
        except Exception:
            _log.exception("Could not end attempt %s of job %s", job["active_attempt"], job["id"])

    def _report(self, job: dict[str, Any], directory: Path, envelope: dict[str, Any]) -> bool:
        """Copy the new journal events, write the progress, and return whether someone asked for a stop."""
        events = self._ingest(job, directory, envelope)
        progress = read_json(directory / "progress.json") or {}
        return self.store.progress(
            job["id"],
            job["lease_token"],
            phase=progress.get("phase", "grading" if job["kind"] == "grade" else "rollout"),
            execution={
                "last_seq": events[-1]["seq"] if events else 0,
                "tool_step": sum(event.get("type") == "tool.started" for event in events),
            },
        )

    def settle(self, job: dict[str, Any]) -> str:
        """Settle the running ``job`` of this worker that no attempt of this process runs, and return its id.

        Its child is stopped first. Its outcome is then published as an attempt's outcome is. Without one,
        the attempt is lost. Without a frozen payload, no child started, and the job is released.
        """
        directory = self.root / job["id"] / job["active_attempt"]
        try:
            envelope = self.store.payload(job["id"], job["active_attempt"])
        except PayloadUnavailable:
            result = {"status": "error", "failure_kind": "infrastructure", "error": "No child started"}
            self.store.release(job["id"], job["lease_token"], result)
            return str(job["id"])
        [attempt] = [each for each in self.store.attempts(job["id"]) if each["id"] == job["active_attempt"]]
        if not stop_process(attempt["execution"].get("process")):
            # A pid that leads no group is not the child's, which started a session of its own.
            _log.warning("Attempt %s left a pid that another process holds now", attempt["id"])
        self._publish(job, directory, _left_outcome(directory), envelope)
        return str(job["id"])


class AttemptStopped(RuntimeError):
    """The attempt was stopped before it finished. The message is the reason."""


class Watchdog:
    """Stops an attempt's process at its deadline, on a stop request, at shutdown or once it is cut off.

    A daemon thread runs until :meth:`close`. ``reason`` is set once: ``timeout``, ``worker_shutdown``,
    ``worker_cut_off`` (``cut_off`` returned True) or the reason of :meth:`request_stop`.
    """

    def __init__(
        self,
        stop_process: Callable[[dict[str, Any] | None], object],
        stop: Event,
        *,
        cut_off: Callable[[], bool],
    ) -> None:
        self.stop_process = stop_process
        self.stop = stop
        self.cut_off = cut_off
        self.deadline: float | None = None
        self.identity: dict[str, Any] | None = None
        self.running: Callable[[], bool] | None = None
        self.reason: str | None = None
        self.lock = Lock()
        self.finished = Event()
        self.thread = Thread(target=self._run, name="attempt-watchdog", daemon=True)
        self.thread.start()

    def arm(
        self, identity: dict[str, Any], deadline: float, *, running: Callable[[], bool] | None = None
    ) -> None:
        """Watch the process ``identity`` until ``deadline``.

        The deadline stops the process only while ``running`` returns True. If the attempt was
        stopped already, the process is stopped at once.
        """
        with self.lock:
            self.identity = identity
            self.deadline = deadline
            self.running = running
            stopped = self.reason is not None
        if stopped:
            self.stop_process(identity)

    def request_stop(self, reason: str) -> None:
        """Stop the attempt's process now, the way a deadline would."""
        with self.lock:
            if self.reason is not None:
                return
            self.reason = reason
            identity = self.identity
        self.stop_process(identity)

    def check(self) -> None:
        """Raise :class:`AttemptStopped` if the attempt was stopped, ``stop`` is set or it is cut off.

        Call it only before the child starts. After that, a stop ends the attempt through ``reason``.
        """
        if self.reason is not None:
            raise AttemptStopped(self.reason)
        if self.stop.is_set():
            raise AttemptStopped("worker_shutdown")
        if self.cut_off():
            raise AttemptStopped("worker_cut_off")

    def close(self) -> None:
        """Stop the watchdog's thread, and wait for it as long as ``stop_process`` can take."""
        self.finished.set()
        self.thread.join(timeout=_STOP_S)

    def _run(self) -> None:
        while not self.finished.wait(0.05):
            now = time.monotonic()
            with self.lock:
                if self.reason is not None:  # stopped on request already
                    return
                if self.stop.is_set():
                    reason = "worker_shutdown"
                elif self.cut_off():
                    reason = "worker_cut_off"
                elif (
                    self.deadline is not None
                    and now >= self.deadline
                    and (self.running is None or self.running())
                ):
                    reason = "timeout"
                else:
                    continue
                self.reason = reason
                identity = self.identity
            try:
                self.stop_process(identity)
            except Exception:
                logging.getLogger(__name__).exception("Attempt watchdog could not stop its process")
            return

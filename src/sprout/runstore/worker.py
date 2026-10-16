"""Same-host attempt supervision. Only reconciled attempts may be requeued."""

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
from uuid import uuid4

from sprout.contract.messages import failure_kind
from sprout.runstore.config import merge, referenced_env, resolve, snapshot_validator
from sprout.runstore.files import read_json, write_json
from sprout.runstore.index import Index
from sprout.runstore.payload import PayloadUnavailable, encode_payload, send_payload
from sprout.runstore.specs import defaults_key, digest, validate_continuation
from sprout.runstore.store import Store
from sprout.worker.agents.mini_swe.history import index_native
from sprout.worker.run.journal import journal_events
from sprout.worker.sandbox.backends import backend_name, default_podman_store, snapshot_store
from sprout_sandbox.podman_cli import default_podman_bin

#: Where a queued job's sandboxes may live: on this host, owned by the attempt,
#: so the worker can establish that each one is gone afterwards. podman unless
#: the profile names microvm, the default everywhere in Sprout.
OWNED_BACKENDS = ("podman", "microvm")
#: What of the worker's environment an attempt child inherits: the system basics, and the
#: podman and store every entry point resolves (sprout_sandbox.podman_cli.default_podman_bin), so
#: the child builds its sandboxes on the podman the worker later asks about them.
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


def owned_backend(kind: str, effective: dict) -> str:
    """The backend a job's effective spec runs on, refused unless it is one the worker can clean up after."""
    if kind == "rollout" and not effective.get("sandbox_image"):
        raise ValueError("A queued rollout needs a sandbox_image")
    name = backend_name(effective.get("backend") or {})
    if name not in OWNED_BACKENDS:
        raise ValueError(
            f"Queued jobs run on an owned sandbox, podman (the default) or microvm, not {name!r}"
        )
    return name


def starting_point(request: dict, previous: dict | None) -> tuple[str | None, dict | None]:
    """The recovery point an attempt starts from, and what its actor is given there.

    A job's first attempt starts where its request says: a branch at its
    ``parent_point`` with its continuation, anything else from scratch. A
    retry resumes the job's own progress from the point the failed attempt
    recorded (``previous``) and adds nothing: whatever the branch added is
    already in that history, and running an inserted turn twice would put
    two copies of it in the conversation.
    """
    if previous:
        return previous["point_id"], {"guidance": "none"}
    return request.get("parent_point"), request.get("continuation")


def apply_branch_budget(effective: dict, budget: dict | None) -> dict:
    """The effective spec with a branch's own contract limits laid over the inherited ones."""
    if not budget:
        return effective
    contract = effective.get("extra", {}).get("rollout_contract")
    if contract is None:
        raise ValueError("A branch budget bounds a rollout contract; this run has none")
    return {**effective, "extra": {**effective["extra"], "rollout_contract": {**contract, **budget}}}


def podman_reclaimed(claims: dict, backend: dict) -> bool:
    """Every claimed podman sandbox gone: no container, network or rootfs left.

    What AgentENV answering 404 establishes for a microVM. A sandbox the child
    destroyed has nothing left and is only looked at; one it leaked (a child
    killed by the watchdog, a destroy that failed half-way) is removed the way
    ``harness reap`` removes it and then looked at again. A claim names the
    store its rootfs was made in; one that does not (the grader's) is looked
    for in the store the job's backend names.
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


def microvm_reclaimed(claims: list, microvm: dict) -> bool:
    """Every claimed microVM deleted, and AgentENV answering 404 for it."""
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


def process_identity(pid: int) -> dict | None:
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


def stop_process(identity: dict | None) -> bool:
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


class Worker:
    def __init__(
        self,
        store: Store,
        config: dict,
        *,
        worker_id: str | None = None,
        lease_s: float = 60,
        snapshot_valid=None,
    ) -> None:
        self.store = store
        self.config = config
        self.root = Path(config["artifact_root"]).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=True, mode=0o700)
        self.worker_id = worker_id or f"{os.uname().nodename}-{os.getpid()}-{uuid4().hex[:8]}"
        self.lease_s = lease_s
        self.index = Index(store, snapshot_valid or snapshot_validator(store, config))

    def _scope(self, request: dict, envelope: dict | None = None) -> dict:
        recovery = (envelope or {}).get("recovery") or {}
        return {
            "context": request["context"],
            "run_spec_hash": digest(request["spec"]),
            "parent_point": request.get("parent_point"),
            "profile_hash": request.get("profile_hash"),
            "recovery_snapshot": recovery.get("snapshot_id"),
            "native_prefix_hash": recovery.get("native", {}).get("sha256"),
        }

    def _ingest(self, job: dict, directory: Path, envelope: dict) -> list[dict]:
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

    def _cleanup(self, directory: Path, effective: dict) -> bool:
        records = journal_events(directory / "resources.jsonl")
        claims = {
            row["id"]: row.get("store")
            for row in records
            if row.get("event") == "claim" and row.get("kind") == "sandbox"
        }
        started = sum(row.get("event") == "allocation_started" for row in records)
        if any(row.get("event") == "allocation_unknown" for row in records) or started > len(claims):
            return False
        if not claims:
            return True
        backend = resolve(effective.get("backend") or {})
        if backend_name(backend) == "podman":
            return podman_reclaimed(claims, backend)
        return microvm_reclaimed(sorted(claims), backend.get("microvm", {}))

    def _publish(self, job: dict, directory: Path, result: dict, envelope: dict) -> None:
        from sprout.contract.messages import is_truncated_result

        effective = envelope["effective_spec"]
        truncated = is_truncated_result(result)
        result = {**result, "failure_kind": None if truncated else failure_kind(result)}
        events = self._ingest(job, directory, envelope)
        if not self._cleanup(directory, effective):
            result = {
                **result,
                "error": "Owned allocation or cleanup needs reconciliation",
                "failure_kind": "infrastructure",
            }
            self.store.finish(job["id"], job["lease_token"], result, state="quarantined")
            return
        if result.get("status") == "completed" or truncated:
            self.store.finish(job["id"], job["lease_token"], result)
            return
        if result.get("failure_kind") == "configuration":
            self.store.finish(job["id"], job["lease_token"], result, state="quarantined")
            return
        recovery = None
        retry = result.get("failure_kind") == "infrastructure"
        if retry and job["kind"] == "rollout":
            remaining = effective.get("timeout_s", 3600) - result.get("elapsed_s", 0)
            point = next(
                (
                    point
                    for point in reversed(self.index.points(job["active_attempt"]))
                    if self.index.valid(point)
                ),
                None,
            )
            if point and remaining > 0:
                recovery = {"point_id": point["id"], "timeout_s": remaining}
                budget = effective.get("budget_usd")
                if budget is not None:
                    charged = result.get("usage", {}).get("cost_usd")
                    unpriced = any(event.get("status") == "budget_unenforceable" for event in events)
                    if type(charged) not in (float, int) or charged <= 0 or charged >= budget or unpriced:
                        retry = False
                    else:
                        recovery["budget_usd"] = budget - charged
            else:
                retry = False
        if not (retry and job["attempt_count"] < job["max_attempts"]):
            salvaged = self._salvage(job, directory, result, envelope)
            if salvaged is not None:
                self.store.finish(job["id"], job["lease_token"], salvaged)
                return
        self.store.finish(
            job["id"], job["lease_token"], result, state="failed", retry=retry, recovery=recovery
        )

    def _salvage(self, job: dict, directory: Path, result: dict, envelope: dict) -> dict | None:
        """A message-export rollout that will not run again, cut back to its last valid point.

        What it did up to there is the policy's own and is graded; a run that
        failed before closing a turn, or was misconfigured, stays failed.
        """
        from sprout.runstore.messages import salvage_interrupted

        contract = envelope["effective_spec"].get("extra", {}).get("rollout_contract", {})
        if (
            job["kind"] != "rollout"
            or not contract.get("message_export")
            or result.get("failure_kind") == "configuration"
        ):
            return None
        point = next(
            (
                point
                for point in reversed(self.index.points(job["active_attempt"]))
                if self.index.valid(point)
            ),
            None,
        )
        return salvage_interrupted(result, directory, point, envelope.get("branch_point")) if point else None

    def run_claimed(self, job: dict, *, stop: Event | None = None) -> str:
        directory = self.root / job["id"] / job["active_attempt"]
        process = None
        identity = None
        watchdog = Watchdog(stop_process, stop if stop is not None else Event(), self.lease_s)
        try:
            watchdog.check()
            directory.mkdir(parents=True, exist_ok=False, mode=0o700)
            request = job["request"]
            profile = self.config["profiles"][request["profile"]]
            if request.get("profile_hash") not in (None, digest(profile)):
                raise ValueError("Worker profile changed after submission")
            effective = merge(profile.get(defaults_key(job["kind"]), {}), request["spec"])
            if job["kind"] == "review":
                # Reads attempt directories on this host; runs nothing in a sandbox.
                effective = {**effective, "artifact_root": str(self.root)}
            else:
                owned_backend(job["kind"], effective)
            prior = self.store.attempts(job["id"])
            previous = prior[-2]["execution"].get("recovery") if len(prior) > 1 else None
            recovery_id, continuation = starting_point(request, previous)
            if previous:
                effective["timeout_s"] = previous["timeout_s"]
                if "budget_usd" in previous:
                    effective["budget_usd"] = previous["budget_usd"]
            recovery = self.index.get_point(recovery_id) if recovery_id else None
            if recovery:
                validate_continuation(self.store.get(recovery["job_id"])["request"], request)
            if recovery and not self.index.valid(recovery):
                raise ValueError("Native prefix or snapshot invalid before dispatch")
            effective = apply_branch_budget(effective, request.get("branch_budget"))
            rollout_contract = effective.get("extra", {}).get("rollout_contract")
            if job["kind"] == "rollout" and rollout_contract is not None:
                from sprout.worker.run.rollout import remaining_timeout

                # Queue time belongs to the group deadline too. Bound child
                # startup/provisioning, not only the agent's model requests.
                effective["timeout_s"] = remaining_timeout(rollout_contract, effective.get("timeout_s", 3600))
                if rollout_contract.get("message_export"):
                    # The soft deadline also covers this actor's own time cap.
                    effective["extra"] = {
                        **effective["extra"],
                        "rollout_contract": {
                            **rollout_contract,
                            "deadline_at": min(
                                rollout_contract["deadline_at"], time.time() + effective["timeout_s"]
                            ),
                        },
                    }
            envelope = {
                "version": 1,
                "job_id": job["id"],
                "attempt_id": job["active_attempt"],
                "kind": job["kind"],
                "effective_spec": effective,
                "profile_config": profile,
                "recovery": recovery,
                "continuation": continuation if recovery else None,
                # Where this job branched from, which a retry's recovery no longer says.
                "branch_point": (
                    self.index.get_point(request["parent_point"]) if request.get("parent_point") else None
                ),
            }
            envelope = self.store.freeze_payload(job["id"], job["lease_token"], envelope)
            effective = envelope["effective_spec"]
            profile = envelope["profile_config"]
            payload = encode_payload(envelope, job["id"], job["active_attempt"])
            env = {key: value for key, value in os.environ.items() if key in CHILD_ENV_PASSTHROUGH}

            for name in referenced_env(envelope):
                env[name] = os.environ[name]
            repo = Path(__file__).resolve().parents[3]
            env["PYTHONPATH"] = str(repo / "src") + ":" + str(repo)
            with (directory / "worker.log").open("w") as log:
                watchdog.check()
                started = time.monotonic()
                timeout = effective.get("timeout_s", 3600) + (1200 if job["kind"] == "grade" else 0)
                if effective.get("extra", {}).get("rollout_contract", {}).get("message_export"):
                    from sprout.contract.messages import TERMINATION_GRACE_SECONDS

                    timeout += TERMINATION_GRACE_SECONDS
                process = subprocess.Popen(
                    [
                        profile.get("python", sys.executable),
                        "-m",
                        "sprout.runstore.child",
                        str(directory),
                        digest(envelope),
                    ],
                    cwd=repo,
                    env=env,
                    stdin=subprocess.PIPE,
                    bufsize=0,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                identity = process_identity(process.pid)
                if identity is None:
                    raise RuntimeError("Attempt child exited before process registration")
                watchdog.arm(identity, started + timeout, running=lambda: process.poll() is None)
                heartbeat_started = time.monotonic()
                self.store.heartbeat(
                    job["id"],
                    job["lease_token"],
                    lease_s=self.lease_s,
                    phase="starting",
                    execution={
                        "directory": str(directory),
                        "process": identity,
                        "request_hash": digest(envelope),
                        "scope": self._scope(request, envelope),
                    },
                )
                watchdog.renew(heartbeat_started)
                send_payload(
                    process.stdin,
                    payload,
                    timeout_s=min(30, self.lease_s / 4, timeout - (time.monotonic() - started)),
                )
                while process.poll() is None:
                    events = self._ingest(job, directory, envelope)
                    progress = read_json(directory / "progress.json") or {}
                    heartbeat_started = time.monotonic()
                    stop_requested = self.store.heartbeat(
                        job["id"],
                        job["lease_token"],
                        lease_s=self.lease_s,
                        phase=progress.get("phase", "grading" if job["kind"] == "grade" else "rollout"),
                        execution={
                            "last_seq": events[-1]["seq"] if events else 0,
                            "tool_step": sum(event.get("type") == "tool.started" for event in events),
                        },
                    )
                    if stop_requested:
                        watchdog.request_stop("stopped")
                    if watchdog.reason is not None:
                        break
                    watchdog.renew(heartbeat_started)
                    time.sleep(min(1, self.lease_s / 4))
                process.wait(timeout=5)
            result = read_json(directory / "outcome.json") or {
                "status": "error",
                "failure_kind": "infrastructure",
                "error": "Attempt process exited without outcome",
                "elapsed_s": time.monotonic() - started,
            }
            if watchdog.reason is not None:
                # A requested stop is nobody's failure and is never retried.
                result = {
                    **result,
                    "status": "timeout" if watchdog.reason == "timeout" else "error",
                    "failure_kind": "stopped" if watchdog.reason == "stopped" else "infrastructure",
                    "error": f"Attempt stopped: {watchdog.reason}",
                    "stop_reason": watchdog.reason,
                }
            self._publish(job, directory, result, envelope)
        except BaseException as error:
            if process is not None:
                if process.stdin is not None:
                    process.stdin.close()
                stop_process(identity)
                process.wait(timeout=5)
            try:
                self.store.finish(
                    job["id"],
                    job["lease_token"],
                    {
                        "status": "timeout" if watchdog.reason == "timeout" else "error",
                        "failure_kind": "actor" if watchdog.reason == "timeout" else "infrastructure",
                        "stop_reason": watchdog.reason,
                        "error": f"{type(error).__name__}: {error}",
                    },
                    state="quarantined" if process is not None else "failed",
                )
            except Exception:
                pass
            if isinstance(error, (KeyboardInterrupt, SystemExit)):
                raise
        finally:
            watchdog.close()
        return job["id"]

    def reconcile(self, job_id: str, *, stop: Event | None = None) -> bool:
        original = self.store.get(job_id)
        if original["state"] != "quarantined":
            return False
        attempts = self.store.attempts(job_id)
        execution = attempts[-1]["execution"]
        directory = self.root / job_id / original["active_attempt"]
        try:
            envelope = self.store.payload(job_id, original["active_attempt"])
        except PayloadUnavailable:
            return False
        job = self.store.adopt_quarantined(job_id, self.worker_id, self.lease_s)
        if job is None:
            return False
        try:
            if stop is not None and stop.is_set():
                raise RuntimeError("Worker is shutting down")
            if not stop_process(execution.get("process")):
                raise RuntimeError("Old process remains active")
            result = read_json(directory / "outcome.json") or {
                "status": "error",
                "failure_kind": "infrastructure",
                "error": "Interrupted attempt reconciled",
                "elapsed_s": (time.time() - attempts[-1]["started_at"].timestamp()),
            }
            self._publish(job, directory, result, envelope)
            return self.store.get(job_id)["state"] != "quarantined"
        except Exception as error:
            self.store.finish(
                job_id,
                job["lease_token"],
                {"error": f"Recovery failed: {type(error).__name__}: {error}"},
                state="quarantined",
            )
            return False


class AttemptStopped(RuntimeError):
    pass


class Watchdog:
    def __init__(self, stop_process: Callable, stop: Event, lease_s: float) -> None:
        self.stop_process = stop_process
        self.stop = stop
        self.lease_s = lease_s
        self.lease_deadline = time.monotonic() + lease_s
        self.deadline: float | None = None
        self.identity: dict | None = None
        self.running: Callable | None = None
        self.reason: str | None = None
        self.lock = Lock()
        self.finished = Event()
        self.thread = Thread(target=self._run, name="attempt-watchdog", daemon=True)
        self.thread.start()

    def renew(self, started: float) -> None:
        with self.lock:
            self.lease_deadline = started + self.lease_s
        self.check()

    def arm(self, identity: dict, deadline: float, *, running: Callable | None = None) -> None:
        with self.lock:
            self.identity = identity
            self.deadline = deadline
            self.running = running
            stopped = self.reason is not None
        if stopped:
            self.stop_process(identity)
        self.check()

    def request_stop(self, reason: str) -> None:
        """Stop the attempt's process now, the way a deadline would."""
        with self.lock:
            if self.reason is not None:
                return
            self.reason = reason
            identity = self.identity
        self.stop_process(identity)

    def check(self) -> None:
        if self.reason is not None:
            raise AttemptStopped(self.reason)
        if self.stop.is_set():
            raise AttemptStopped("worker_shutdown")

    def close(self) -> None:
        self.finished.set()
        self.thread.join(timeout=16)

    def _run(self) -> None:
        while not self.finished.wait(0.05):
            now = time.monotonic()
            with self.lock:
                if self.reason is not None:  # stopped on request already
                    return
                if self.stop.is_set():
                    reason = "worker_shutdown"
                elif (
                    self.deadline is not None
                    and now >= self.deadline
                    and (self.running is None or self.running())
                ):
                    reason = "timeout"
                elif now >= self.lease_deadline:
                    reason = "lease_expired"
                else:
                    continue
                self.reason = reason
                identity = self.identity
            try:
                self.stop_process(identity)
            except Exception:
                logging.getLogger(__name__).exception("Attempt watchdog could not stop its process")
            return

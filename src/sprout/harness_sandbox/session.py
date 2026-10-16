"""One sandbox, driven synchronously: lifecycle, snapshots, and the tool seam.

This is the object the orchestrator needs in order to be an entry point rather
than a function you hand a pre-built environment to. Three things have to be held
by the same owner, and this is that owner:

- **the sandbox's lifetime** -- it outlives the agent, because grading and patch
  extraction happen after the agent stops, and its teardown must be guaranteed
  rather than left to whatever the agent did;
- **the snapshot handle** -- ``snapshot`` / ``squash_snapshot`` / ``swap_sandbox``
  is the contract ``harness.tools.access`` and the checkpointer use, so whoever
  cannot call these cannot record the environment half of a rollback pair;
- **the executor seam** -- ``(tool_name, args) -> ToolResult``, which is the only
  channel anything has into the environment.

The orchestrator builds it, so the run's owner is also the sandbox's owner.

What stays out of here, deliberately: anything that decides *what counts as the
answer*. A session runs tools and takes snapshots. Extracting a diff, grading a
test, deciding an episode succeeded -- those belong to the layer that knows what
the benchmark is.

Reporting goes through :meth:`_note` and :meth:`_warn` so a subclass can render
it however its CLI does, and so this module needs no opinion about presentation.
Both respect ``quiet``.
"""

from __future__ import annotations

import asyncio
import json
import sys
import threading
from typing import Any
from typing import Optional

from sprout_sandbox import MicroVMPool, PodmanPool, Sandbox, Snapshot
from sprout_sandbox.result import ToolResult as SdkToolResult
from sprout.harness_core.result import ToolResult
from sprout.harness_sandbox.backends import backend_name, build_pool, spawn_takes_shape
from sprout.harness_sandbox.templates import builder_from_backend

#: Identity for the owner's own traffic (patch extraction, resets, test runs).
#: Distinct from any agent's id so that bookkeeping does not appear in an agent's
#: event stream, and does not consume its cursor over the log.
OWNER_AGENT_ID = "harness"


class SandboxSession:
    """Manages one sandbox for one run.

    ``execute`` is the owner's own channel, attributed to :data:`OWNER_AGENT_ID`.
    An agent gets its own channel from :meth:`executor_for`, which binds that
    agent's identity -- the identity belongs to the channel rather than to each
    call, so a call site cannot forget it and an agent cannot act as another.
    """

    def __init__(self, timeout: float = 300.0,
                 quiet: bool = False, backend: "dict | None" = None):
        self.timeout = timeout
        self.quiet = quiet
        #: Where sandboxes come from (``harness_sandbox/backends.py``): a
        #: config's execution section, passed through. ``{"backend": "podman",
        #: "podman": {"store": ..., "runtime_bin": ...}}`` is local containers on
        #: btrfs subvolumes of the store, and empty means podman too (the
        #: default backend), whose section must still name both;
        #: ``{"backend": "microvm", ...}`` is an AgentENV microVM, its endpoint
        #: from the section or the environment.
        self.backend = backend or {}
        self._pool: "Optional[MicroVMPool | PodmanPool]" = None
        self._sandbox: Optional[Sandbox] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._loop_lock = threading.RLock()
        #: Why the last :meth:`create` returned False. Kept because ``create``
        #: reports through a bool and the reason through ``_warn`` -- and a quiet
        #: caller (the orchestrator is one) then had nothing to put in its own
        #: error, so "could not create a sandbox from <image>" pointed at the image
        #: when the actual cause was a missing setting.
        self.create_error: str = ""
        #: Why the last :meth:`upload_file` or :meth:`download_file` returned
        #: False, for the same reason: a quiet caller (the grader) raises with it.
        self.transfer_error: str = ""
        #: Called with the replacement sandbox after a successful re-board.
        #: Records that name the sandbox (a resource-ledger claim) follow the
        #: swap through here; tool calls need nothing, since they go through
        #: this session.
        self.on_swap: list = []

    # --- reporting ---------------------------------------------------------
    def _note(self, label: str, text: str) -> None:
        """One line of progress. Overridden by a subclass with a styled CLI."""
        if not self.quiet:
            sys.stderr.write("  %-9s %s\n" % (label, text))

    def _warn(self, text: str) -> None:
        """Something went wrong but the run continues."""
        if not self.quiet:
            sys.stderr.write("  ! %s\n" % text)

    def _get_loop(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def _drive(self, coroutine):
        """One owner for the session loop, even across checkpoint workers."""
        with self._loop_lock:
            unsettled = getattr(self, "_unsettled_operation", None)
            if unsettled is not None and not unsettled.done():
                if asyncio.isfuture(coroutine):
                    coroutine.cancel()
                else:
                    coroutine.close()
                raise RuntimeError("Previous sandbox operation has not settled; session is quarantined")
            if unsettled is not None:
                try:
                    unsettled.result()
                except BaseException:
                    pass
            self._unsettled_operation = None
            loop = self._get_loop()
            task = asyncio.ensure_future(coroutine, loop=loop)
            try:
                return loop.run_until_complete(task)
            except BaseException:
                if not task.done():
                    task.cancel()
                    try:
                        loop.run_until_complete(asyncio.wait(
                            {task}, timeout=getattr(self, "_cancel_grace_seconds", 5.0)))
                    except BaseException as cleanup_error:
                        self._warn("Interrupted operation cleanup failed: %s" % cleanup_error)
                if task.done():
                    try:
                        task.result()
                    except BaseException:
                        pass
                else:
                    self._unsettled_operation = task
                    self._warn("Interrupted sandbox operation did not settle; quarantining session")
                raise

    # --- lifecycle ---------------------------------------------------------
    @property
    def sandbox_id(self) -> str:
        """Stable identity of the active sandbox, or ``unknown`` before spawn.

        ``getattr``, not attribute access: this is read by progress lines, and a
        progress line must not be able to break the operation it reports. That was
        not hypothetical -- reporting used to sit behind ``if not self.quiet``, so
        the id was never computed for a quiet caller, and moving it into a helper
        made the argument eager and turned a successful re-board into an
        AttributeError. "unknown" is already this property's documented answer
        whenever the active sandbox cannot name itself.
        """
        if not self._sandbox:
            return "unknown"
        return getattr(self._sandbox, "sandbox_id", "") or "unknown"

    def create(self, image: str, resources: "Optional[dict]" = None) -> bool:
        """``resources``: e.g. ``{"cpu": 4, "memory_mb": 16384}``. A task that
        declares its needs must get them: running a build-heavy task in the
        backend's default (2 CPUs, 1 GB) OOMs it rather than failing loudly."""
        return self._drive(
            self._create_async(image, resources))

    async def _create_async(self, image: str,
                            resources: "Optional[dict]" = None) -> bool:
        self.create_error = ""
        try:
            self._pool = build_pool(self.backend)
            # A caller names its environment with an image; the microVM backend
            # starts from templates. Build one per image on demand (cached across
            # instances) when the config says how to get the runtime into it.
            builder = builder_from_backend(self.backend)
            if builder is not None:
                self._note("template", "resolving for %s" % image)
                image = builder.template_for(image, resources)
            # Two entries, and the config says which one this caller's image names
            # are for. A benchmark names its environment with an OCI image
            # reference, which only the microVM cold-start path accepts; a replay
            # or a re-board hands over a snapshot id, which only its snapshot path
            # accepts. Guessing from the string would eventually mistake a
            # template tag for an image tag. A builder-resolved name is a template
            # by construction, so it must never take the cold-start path even when
            # `from_image` is set. The podman pool has one path for a reference
            # and a snapshot id alike, and it takes the shape every time (a
            # subvolume carries none), so both go through it with the resources.
            backend_section = self.backend.get(backend_name(self.backend), {}) or {}
            shaped = builder is None and (
                spawn_takes_shape(self.backend)
                or bool(backend_section.get("from_image") or self.backend.get("from_image")))
            if shaped and self._pool.supports_cold_start():
                self._sandbox = await self._pool.spawn_from_image(
                    image, resources=resources)
            else:
                # No resources here on purpose: a microVM template or snapshot
                # has its shape baked in (only a cold start can choose one), and
                # the pool rejects the argument rather than ignoring it. The
                # task's shape reached the template above; a resumed run
                # continues in the machine it was running on.
                self._sandbox = await self._pool.spawn(image=image)
            self._note("sandbox", self.sandbox_id[:12])
            return True
        except Exception as e:  # noqa: BLE001 - create reports, it does not raise
            self.create_error = "%s: %s" % (type(e).__name__, e)
            self._warn("Failed to create sandbox: %s" % e)
            return False

    def destroy(self) -> None:
        if self._sandbox:
            try:
                self._drive(self._destroy_async())
            except Exception as e:  # noqa: BLE001
                # A sandbox the server would not release is a leak for its TTL
                # or ``harness reap`` to reclaim, not a reason to raise out of a
                # run's teardown.
                self._warn("Failed to destroy sandbox: %s" % e)
            self._sandbox = None
            self._pool = None

    async def _destroy_async(self) -> None:
        if self._pool and self._sandbox:
            sid = self.sandbox_id
            await self._pool.destroy(self._sandbox)
            self._note("cleanup", "destroyed %s" % sid[:12])
            self._sandbox = None
            self._pool = None

    async def _release_async(self) -> None:
        """Forget the sandbox without destroying it; it keeps running.

        For a sandbox that holds state nothing else has (its final snapshot
        failed): the pool drops the handle, so closing the pool afterwards
        does not destroy it, and ``harness reap`` or a person removes it later.
        """
        if self._pool and self._sandbox:
            sid = self.sandbox_id
            await self._pool.forget(self._sandbox)
            self._note("cleanup", "released %s, kept running" % sid[:12])
            self._sandbox = None
            self._pool = None

    # --- host files --------------------------------------------------------
    def supports_upload(self) -> bool:
        """Whether this session's backend can place host files in the sandbox."""
        return bool(self._pool) and self._pool.supports_upload()

    def upload_file(self, source, destination: str) -> bool:
        """Copy a host file into the sandbox. False when unavailable/failed.

        Bypasses the runtime, whose tools take commands and text rather than
        bytes: this is for binary and bulk files (grading fixtures, datasets).
        """
        self.transfer_error = ""
        if not self._sandbox or not self.supports_upload():
            self.transfer_error = "no sandbox, or its backend cannot upload"
            return False
        try:
            self._drive(
                self._pool.upload_file(self._sandbox, source, destination))
            return True
        except Exception as e:  # noqa: BLE001
            self.transfer_error = "%s: %s" % (type(e).__name__, e)
            self._warn("upload failed: %s" % e)
            return False

    def download_file(self, source: str, destination, timeout: float = 120.0) -> bool:
        """Copy a guest file to the host without runtime output truncation."""
        self.transfer_error = ""
        if not self._sandbox or not self._pool or not self._pool.supports_download():
            self.transfer_error = "no sandbox, or its backend cannot download"
            return False
        try:
            self._drive(self._pool.download_file(
                self._sandbox, source, destination, timeout=timeout))
            return True
        except Exception as error:
            self.transfer_error = "%s: %s" % (type(error).__name__, error)
            self._warn("download failed: %s" % error)
            return False

    # --- network -----------------------------------------------------------
    def set_network(self, allow: bool) -> None:
        """Give the live sandbox egress, or take it away; it keeps running.

        Harbor's per-phase policy (an agent offline, a verifier online, or the
        reverse) needs this on a sandbox that already exists. The podman pool
        connects or disconnects the container's own bridge network; the microVM
        backend has no pool operation for it (AgentENV takes a ``PUT
        /sandboxes/{id}/network``, which the Harbor environment sends itself),
        so a backend without one raises rather than pretending.
        """
        if not self._sandbox or not self._pool:
            raise RuntimeError("No active sandbox")
        toggle = getattr(self._pool, "set_network", None)
        if not callable(toggle):
            raise NotImplementedError(
                "backend %r cannot change a running sandbox's network" % backend_name(self.backend))
        self._drive(toggle(self._sandbox, bool(allow)))

    # --- checkpoints -------------------------------------------------------
    #
    # A rollout that snapshots every step needs two operations beyond
    # create/destroy, and both are pool capabilities rather than session logic:
    # publish a checkpoint, and continue on a sandbox started from one.
    # The checkpointer calls exactly the three methods below (a backend with
    # nothing to squash -- podman's subvolumes -- answers ``squash_snapshot``
    # with its input, and its snapshots carry no layer count, so it is never
    # asked to re-board).

    def supports_snapshot(self) -> bool:
        """Whether this session's backend can publish snapshots."""
        return bool(self._pool) and self._pool.supports_snapshot()

    def snapshot(self, name: "str | None" = None,
                 disk_only: bool = True) -> Optional["Snapshot"]:
        """Checkpoint the live sandbox; it keeps running.

        Defaults to ``disk_only``: a rollout replays by restoring the disk and
        re-feeding the transcript, so paying for a memory image every step buys
        nothing. Returns ``None`` when the backend cannot snapshot or the capture
        fails -- a checkpoint is an optimisation for later analysis, never a reason
        to fail the episode in progress.
        """
        if not self._sandbox or not self.supports_snapshot():
            return None
        try:
            return self._drive(
                self._pool.snapshot(self._sandbox, name=name,
                                    disk_only=disk_only))
        except Exception as e:  # noqa: BLE001
            self._warn("snapshot failed: %s" % e)
            return None

    def squash_snapshot(self, snapshot, name: "str | None" = None):
        """Flatten a snapshot's chain, returning an equivalent snapshot.

        Returns the input unchanged if the backend cannot squash or the call
        fails: a deep chain still works, it just makes its children's checkpoints
        more expensive.
        """
        if not self.supports_snapshot() or not callable(getattr(self._pool, "squash", None)):
            return snapshot
        try:
            return self._drive(
                self._pool.squash(snapshot, name=name))
        except Exception as e:  # noqa: BLE001
            self._warn("squash failed: %s" % e)
            return snapshot

    async def _reachable(self, sandbox, attempts: int = 8,
                         delay: float = 0.5) -> bool:
        """Whether a sandbox's runtime answers a trivial call.

        Retried briefly: a freshly booted sandbox may still be starting its
        runtime when the API call that created it has already returned.
        """
        for attempt in range(attempts):
            try:
                result = await sandbox.call("shell", command="true", timeout=10)
                if not result.is_error:
                    return True
            except Exception:  # noqa: BLE001
                pass
            if attempt + 1 < attempts:
                await asyncio.sleep(delay)
        return False

    def swap_sandbox(self, snapshot) -> bool:
        """Continue this session on a sandbox started from ``snapshot``.

        Called when a checkpoint shows the server compacted the layer chain: the
        running sandbox's own layer stack is never compacted, so without replacing
        it every later capture would re-compact the whole chain. The swap is
        invisible to an agent mid-run, because executors resolve the active
        sandbox per call.
        """
        if not self._sandbox or not self._pool:
            return False
        snapshot_id = getattr(snapshot, "id", snapshot)
        previous = self._sandbox
        try:
            replacement = self._drive(
                self._pool.spawn(image=snapshot_id,
                                 agent_id=previous.agent_id))
        except Exception as e:  # noqa: BLE001
            self._warn("re-board failed, keeping sandbox: %s" % e)
            return False

        # Probe before adopting. A sandbox created from a disk-only snapshot
        # cold-boots, so its runtime is only there if the template declares a
        # startup command that launches it; adopting an unreachable replacement
        # would turn every later tool call into a transport error and kill the
        # episode. Keeping the old sandbox instead costs only a deeper layer chain.
        if not self._drive(self._reachable(replacement)):
            self._warn("re-board target has no runtime; keeping sandbox (does "
                       "the template declare a startup command?)")
            try:
                self._drive(
                    self._pool.destroy(replacement))
            except Exception:  # noqa: BLE001
                pass
            return False

        self._sandbox = replacement
        # Listeners BEFORE the old sandbox dies: a server that learns of the swap
        # late serves one call into a destroyed VM; one that never learns serves
        # every later call into it.
        for listener in list(self.on_swap):
            try:
                listener(replacement)
            except Exception as e:  # noqa: BLE001
                self._warn("swap listener failed: %s" % e)
        try:
            self._drive(self._pool.destroy(previous))
        except Exception as e:  # noqa: BLE001
            # The replacement is live and serving calls; a stranded old sandbox is
            # a leak for the TTL to reap, not a run failure.
            self._note("note", "old sandbox not destroyed: %s" % e)
        self._note("re-board", self.sandbox_id[:12])
        return True

    # --- the tool seam -----------------------------------------------------
    def execute(self, tool_name: str, args: dict,
                timeout: float = 300.0) -> ToolResult:
        """Run a tool as the owner itself (resets, extraction, tests).

        """
        return self._run(tool_name, args, timeout, OWNER_AGENT_ID)

    def call(self, tool_name: str, args: dict, timeout: float = 300.0) -> ToolResult:
        """Run a tool for the agent. Unlike :meth:`execute`, a transport failure
        raises: the caller must know the difference between "the command failed"
        and "we do not know whether the command ran"."""
        if not self._sandbox:
            raise RuntimeError("No active sandbox")
        return self._drive(self._execute_async(tool_name, dict(args), timeout, OWNER_AGENT_ID))

    def _run(self, tool_name: str, args: dict, timeout: float,
             agent_id: str) -> ToolResult:
        if not self._sandbox:
            return ToolResult(success=False, output="", error="No active sandbox")
        try:
            return self._drive(
                self._execute_async(tool_name, args, timeout, agent_id)
            )
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, output="", error=str(e))

    async def _execute_async(self, tool_name: str, args: dict, timeout: float,
                             agent_id: str = OWNER_AGENT_ID) -> ToolResult:
        if "timeout" not in args and tool_name == "shell":
            args["timeout"] = int(timeout)
        # The channel's identity is the only one that counts: an agent_id that
        # arrived in the arguments (a model naming itself) is dropped rather
        # than sent on as a tool argument.
        call_args = {k: v for k, v in args.items() if k != "agent_id"}
        sdk_result: SdkToolResult = await self._sandbox.backend.call(
            tool_name, call_args, agent_id)
        return ToolResult.from_sdk(sdk_result)


def shell_outcome(session: Any, command: str, timeout: int = 120) -> tuple[str, str, int | None, bool]:
    """``(stdout, stderr, exit_code, timed_out)`` of a command that finished or ran out of time.

    For a caller to whom a timeout is an answer rather than a fault: a grader
    whose tests hang scores them, it does not retry them. A timed-out command
    reports the exit code the runtime saw (137, its process group killed), or
    ``None`` when it did not say. Raises when the outcome is unknown: the
    command still running, a stream truncated, or neither an exit code nor a
    timeout.
    """
    result = session.execute("shell", {"command": command, "timeout": timeout}, timeout=timeout + 60)
    outcome = getattr(result, "outcome", None)
    if outcome is not None:
        if outcome.running or outcome.truncated or (outcome.exit_code is None and not outcome.timed_out):
            raise RuntimeError("Sandbox command incomplete or truncated")
        return outcome.stdout, outcome.stderr, outcome.exit_code, bool(outcome.timed_out)
    try:
        payload = json.loads(result.output)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and "stdout" in payload:
        code, timed_out = payload.get("exit_code"), bool(payload.get("timed_out"))
        if (payload.get("running") or payload.get("stdout_truncated") or payload.get("stderr_truncated")
                or (code is None and not timed_out)):
            raise RuntimeError("Sandbox command incomplete or truncated")
        return (str(payload["stdout"]), str(payload.get("stderr") or ""),
                None if code is None else int(code), timed_out)
    if not result.success:
        raise RuntimeError(result.error or result.output or "Sandbox tool execution failed")
    return result.output, "", 0, False


def shell(session: Any, command: str, timeout: int = 120) -> tuple[str, str, int]:
    """``(stdout, stderr, exit_code)`` of a settled command; a timeout raises."""
    stdout, stderr, code, timed_out = shell_outcome(session, command, timeout)
    if timed_out:
        raise RuntimeError("Sandbox command timed out after %d s" % timeout)
    return stdout, stderr, code


def checked(session: Any, command: str, timeout: int = 120) -> str:
    """stdout of a command that must exit 0."""
    stdout, stderr, code = shell(session, command, timeout)
    if code != 0:
        raise RuntimeError(f"Sandbox command exited {code}: {(stderr or stdout)[-2000:]}")
    return stdout

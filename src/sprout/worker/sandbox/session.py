"""One sandbox driven synchronously: its lifecycle, its snapshots and its tool channel.

A :class:`SandboxSession` holds three things of one run, because each needs the same owner:

- the sandbox's lifetime. The sandbox outlives the agent, because grading and patch
  extraction come after the agent stops, and its teardown does not depend on the agent.
- the snapshot handle: ``snapshot``, ``squash_snapshot`` and ``swap_sandbox``. Without them,
  nothing can record the environment half of a rollback pair.
- the tool channel, ``(tool_name, args) -> ToolResult``, the only way into the sandbox.

A session runs tools and takes snapshots. It does not decide what counts as the answer:
diff extraction and grading belong to the layer that knows the benchmark.

Progress lines and warnings go through ``_note`` and ``_warn``, which a subclass can
override to render them its own way. ``quiet`` silences both.
"""

from __future__ import annotations

import asyncio
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, cast

from sprout.contract.backend import backend_name
from sprout.worker.sandbox.backends import build_pool
from sprout.worker.sandbox.session_loop import SessionLoop
from sprout.worker.sandbox.spawn import spawn_sandbox
from sprout.worker.sandbox.template_recipe import registry_config
from sprout.worker.tools.result import ToolResult
from sprout_sandbox import MicroVMPool, PodmanPool, Sandbox, Snapshot
from sprout_sandbox.result import ToolResult as SdkToolResult

#: The caller identity that each call of a session sends beside its arguments. The runtime does
#: not read it, and a model cannot set it.
OWNER_AGENT_ID = "harness"


class SandboxSession:
    """Manages one sandbox for one run.

    Every call reaches the runtime as :data:`OWNER_AGENT_ID`, and an ``agent_id`` in a call's
    arguments is dropped. :meth:`execute` returns a transport failure as a failed result, and
    :meth:`call` raises it."""

    def __init__(
        self, timeout: float = 300.0, quiet: bool = False, backend: dict[str, Any] | None = None
    ) -> None:
        self.timeout = timeout
        self.quiet = quiet
        #: The config's execution section (``sprout.worker.sandbox.backends``). Empty means
        #: podman, the default backend, whose section must still name the store and the runtime.
        self.backend = backend or {}
        self._pool: MicroVMPool | PodmanPool | None = None
        self._sandbox: Sandbox | None = None
        self._loop = SessionLoop(self._warn)
        #: Why the last :meth:`create` or :meth:`attach` returned False. A quiet caller has no
        #: warning to read, so it puts this reason in its own error.
        self.create_error: str = ""
        #: Why the last :meth:`upload_file` or :meth:`download_file` returned False. A quiet caller
        #: puts this reason in its own error.
        self.transfer_error: str = ""
        #: Called with the replacement sandbox after a successful re-board, before the old one is
        #: destroyed. A record that names the sandbox, such as a ledger claim, follows it here.
        self.on_swap: list[Callable[[Sandbox], None]] = []

    # --- reporting ---------------------------------------------------------
    def _note(self, label: str, text: str) -> None:
        """Write one progress line to stderr, unless ``quiet``. A subclass with a styled CLI overrides it."""
        if not self.quiet:
            sys.stderr.write("  %-9s %s\n" % (label, text))

    def _warn(self, text: str) -> None:
        """Report a failure that the run survives, unless ``quiet``."""
        if not self.quiet:
            sys.stderr.write("  ! %s\n" % text)

    # --- lifecycle ---------------------------------------------------------
    @property
    def sandbox_id(self) -> str:
        """The id of the active sandbox, or ``unknown`` before a spawn or for a sandbox with no id."""
        if not self._sandbox:
            return "unknown"
        # getattr: progress lines read this, and a progress line must not break the operation it
        # reports, also for a sandbox object that cannot name itself.
        return getattr(self._sandbox, "sandbox_id", "") or "unknown"

    def create(self, image: str, resources: dict[str, Any] | None = None) -> bool:
        """Start a sandbox from ``image``, and return whether it started. ``create_error`` says why not.

        ``resources`` is the task's shape, such as ``{"cpu": 4, "memory_mb": 16384}``. A task that
        declares its needs must get them: a build-heavy task in the backend's default shape runs
        out of memory instead of failing with a clear error."""
        return self._loop.drive(self._create_async(image, resources))

    async def _create_async(self, image: str, resources: dict[str, Any] | None = None) -> bool:
        self.create_error = ""
        try:
            self._pool = build_pool(self.backend)
            self._sandbox = await spawn_sandbox(self._pool, self.backend, image, resources, self._note)
            self._note("sandbox", self.sandbox_id[:12])
            return True
        except Exception as e:  # noqa: BLE001 - create reports, it does not raise
            self.create_error = "%s: %s" % (type(e).__name__, e)
            self._warn("Failed to create sandbox: %s" % e)
            return False

    def attach(self, sandbox_id: str) -> bool:
        """Drive the running sandbox ``sandbox_id`` of this session's backend, which another session kept.

        Afterwards this session owns it, and ``destroy`` removes it. Returns False, with the
        reason in ``create_error``, when the sandbox is gone or its runtime does not answer.
        """
        return self._loop.drive(self._attach_async(sandbox_id))

    async def _attach_async(self, sandbox_id: str) -> bool:
        self.create_error = ""
        try:
            pool = build_pool(self.backend)
        except Exception as e:  # noqa: BLE001 - attach reports, it does not raise
            self.create_error = "%s: %s" % (type(e).__name__, e)
            return False
        sandbox = pool.attach(sandbox_id)
        if not await self._reachable(sandbox):
            # Forgotten, not destroyed: whatever is left of it is the owner's ledger's to reclaim.
            await pool.forget(sandbox)
            self.create_error = "sandbox %s does not answer" % sandbox_id
            return False
        self._pool, self._sandbox = pool, sandbox
        self._note("sandbox", "attached %s" % sandbox_id[:12])
        return True

    def destroy(self) -> None:
        """Destroy the active sandbox, if there is one. A failure is a warning, not an exception."""
        if self._sandbox:
            try:
                self._loop.drive(self._destroy_async())
            except Exception as e:  # noqa: BLE001
                # A sandbox that the server does not release leaks until its TTL or
                # ``sprout.worker reap``. That is no reason to raise out of a run's teardown.
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
        """Forget the sandbox without destroying it. It keeps running.

        For a sandbox whose state exists nowhere else, because its final snapshot failed. Closing
        the pool afterwards does not destroy it. ``sprout.worker reap`` or a person removes it."""
        if self._pool and self._sandbox:
            sid = self.sandbox_id
            await self._pool.forget(self._sandbox)
            self._note("cleanup", "released %s, kept running" % sid[:12])
            self._sandbox = None
            self._pool = None

    # --- host files --------------------------------------------------------
    def supports_upload(self) -> bool:
        """Whether this session's backend can place host files in the sandbox."""
        return self._pool is not None and self._pool.supports_upload()

    def upload_file(self, source: str | Path, destination: str) -> bool:
        """Copy a host file into the sandbox, and return whether it arrived. ``transfer_error`` says why not.

        The copy bypasses the runtime, whose tools take commands and text, not bytes. It is for
        binary and bulk files, such as grading fixtures and datasets.
        """
        self.transfer_error = ""
        if not self._sandbox or self._pool is None or not self.supports_upload():
            self.transfer_error = "no sandbox, or its backend cannot upload"
            return False
        try:
            self._loop.drive(self._pool.upload_file(self._sandbox, source, destination))
            return True
        except Exception as e:  # noqa: BLE001
            self.transfer_error = "%s: %s" % (type(e).__name__, e)
            self._warn("upload failed: %s" % e)
            return False

    def download_file(self, source: str, destination: str | Path, timeout: float = 120.0) -> bool:
        """Copy a guest file to the host, and return whether it arrived. ``transfer_error`` says why not.

        The copy bypasses the runtime, so the runtime's output limit does not truncate it."""
        self.transfer_error = ""
        if not self._sandbox or not self._pool or not self._pool.supports_download():
            self.transfer_error = "no sandbox, or its backend cannot download"
            return False
        try:
            self._loop.drive(self._pool.download_file(self._sandbox, source, destination, timeout=timeout))
            return True
        except Exception as error:
            self.transfer_error = "%s: %s" % (type(error).__name__, error)
            self._warn("download failed: %s" % error)
            return False

    # --- the image -----------------------------------------------------------
    def image_config(self, image: str | None) -> dict[str, Any]:
        """The OCI config of the image that the active sandbox comes from: ``User``, ``Env`` and others.

        A podman sandbox keeps the config of its image, so ``image`` is not read. A microVM keeps none,
        so the config comes from the registry for ``image``, an OCI reference (``registry_config``).
        Raises RuntimeError without an active podman sandbox, TemplateError when regctl cannot read it."""
        if backend_name(self.backend) != "podman":
            return registry_config(image)
        if not isinstance(self._pool, PodmanPool) or self._sandbox is None:
            raise RuntimeError("No active sandbox")
        return self._pool.image_config(self._sandbox)

    # --- network -----------------------------------------------------------
    def set_network(self, allow: bool) -> None:
        """Give the live sandbox egress, or take it away. The sandbox keeps running.

        Raises RuntimeError without an active sandbox. Raises NotImplementedError when the pool
        cannot switch a running sandbox's network, as the microVM pool cannot."""
        if not self._sandbox or not self._pool:
            raise RuntimeError("No active sandbox")
        toggle = getattr(self._pool, "set_network", None)
        if not callable(toggle):
            raise NotImplementedError(
                "backend %r cannot change a running sandbox's network" % backend_name(self.backend)
            )
        self._loop.drive(toggle(self._sandbox, bool(allow)))

    # --- checkpoints -------------------------------------------------------
    def supports_snapshot(self) -> bool:
        """Whether this session's backend can publish snapshots."""
        return self._pool is not None and self._pool.supports_snapshot()

    def snapshot(self, name: str | None = None, disk_only: bool = True) -> Snapshot | None:
        """Checkpoint the live sandbox, which keeps running, and return the snapshot.

        Returns None when the backend cannot snapshot or the capture fails: a checkpoint never fails
        the episode. ``disk_only`` is the default, because a replay restores the disk and re-feeds
        the transcript, so a memory image at every step buys nothing."""
        if not self._sandbox or self._pool is None or not self.supports_snapshot():
            return None
        try:
            return self._loop.drive(self._pool.snapshot(self._sandbox, name=name, disk_only=disk_only))
        except Exception as e:  # noqa: BLE001
            self._warn("snapshot failed: %s" % e)
            return None

    def squash_snapshot(self, snapshot: Snapshot | str, name: str | None = None) -> Snapshot | str:
        """A snapshot equivalent to ``snapshot``, with its layer chain flattened.

        Returns the input unchanged when the call fails, or the backend cannot squash, as podman
        cannot. A deep chain still works, but the checkpoints of its children cost more."""
        # getattr: only the microVM pool has ``squash``.
        squash = getattr(self._pool, "squash", None)
        if not self.supports_snapshot() or not callable(squash):
            return snapshot
        try:
            squashed: Snapshot = self._loop.drive(squash(snapshot, name=name))
            return squashed
        except Exception as e:  # noqa: BLE001
            self._warn("squash failed: %s" % e)
            return snapshot

    async def _reachable(self, sandbox: Sandbox, attempts: int = 8, delay: float = 0.5) -> bool:
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

    def swap_sandbox(self, snapshot: Snapshot | str) -> bool:
        """Continue this session on a sandbox started from ``snapshot``, and return whether it moved.

        For a checkpoint whose chain the server compacted. The running sandbox's own chain is never
        compacted, so each later capture would compact it again. An agent does not see the swap,
        because each tool call goes to the sandbox that is active at that time."""
        if not self._sandbox or not self._pool:
            return False
        # A Snapshot's id, or the bare id itself: both are a str.
        snapshot_id = cast(str, getattr(snapshot, "id", snapshot))
        previous = self._sandbox
        try:
            replacement = self._loop.drive(self._pool.spawn(image=snapshot_id, agent_id=previous.agent_id))
        except Exception as e:  # noqa: BLE001
            self._warn("re-board failed, keeping sandbox: %s" % e)
            return False

        # Probe before adopting. A cold-booted replacement has a runtime only if its template starts
        # one, and an unreachable one fails every later call. The old sandbox costs a deeper chain.
        if not self._loop.drive(self._reachable(replacement)):
            self._warn(
                "re-board target has no runtime; keeping sandbox (does "
                "the template declare a startup command?)"
            )
            try:
                self._loop.drive(self._pool.destroy(replacement))
            except Exception:  # noqa: BLE001
                pass
            return False

        self._sandbox = replacement
        # Notify the listeners before the old sandbox is destroyed. A listener that learns of the
        # swap late serves a call into a destroyed sandbox.
        for listener in list(self.on_swap):
            try:
                listener(replacement)
            except Exception as e:  # noqa: BLE001
                self._warn("swap listener failed: %s" % e)
        try:
            self._loop.drive(self._pool.destroy(previous))
        except Exception as e:  # noqa: BLE001
            # The replacement is live and serves calls. A stranded old sandbox is a leak for its
            # TTL or the reaper, not a run failure.
            self._note("note", "old sandbox not destroyed: %s" % e)
        self._note("re-board", self.sandbox_id[:12])
        return True

    # --- the tool seam -----------------------------------------------------
    def execute(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        """Run a tool as the owner itself (resets, extraction, tests). A failure is a failed result."""
        return self._run(tool_name, args, timeout, OWNER_AGENT_ID)

    def call(self, tool_name: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        """Run a tool for the agent. Unlike :meth:`execute`, a transport failure raises.

        The caller must tell "the command failed" from "we do not know whether the command ran".
        Raises RuntimeError without an active sandbox."""
        if not self._sandbox:
            raise RuntimeError("No active sandbox")
        return self._loop.drive(self._execute_async(tool_name, dict(args), timeout, OWNER_AGENT_ID))

    def _run(self, tool_name: str, args: dict[str, Any], timeout: float, agent_id: str) -> ToolResult:
        if not self._sandbox:
            return ToolResult(success=False, output="", error="No active sandbox")
        try:
            return self._loop.drive(self._execute_async(tool_name, args, timeout, agent_id))
        except Exception as e:  # noqa: BLE001
            return ToolResult(success=False, output="", error=str(e))

    async def _execute_async(
        self, tool_name: str, args: dict[str, Any], timeout: float, agent_id: str = OWNER_AGENT_ID
    ) -> ToolResult:
        if "timeout" not in args and tool_name == "shell":
            args["timeout"] = int(timeout)
        # The channel's identity is the only one that counts. An agent_id in the arguments, a model
        # naming itself, is dropped, not sent on as a tool argument.
        call_args = {k: v for k, v in args.items() if k != "agent_id"}
        assert self._sandbox is not None, "No active sandbox"
        sdk_result: SdkToolResult = await self._sandbox.backend.call(tool_name, call_args, agent_id)
        return ToolResult.from_sdk(sdk_result)

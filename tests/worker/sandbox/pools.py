"""A stub sandbox pool, for the tests of ``SandboxSession`` that need no real backend.

``StubPool`` has the methods of ``MicroVMPool`` and ``PodmanPool`` that a session calls. It records
each operation, and an operation in ``refuse`` raises ``RuntimeError("<operation> refused")``.
``NetworkedPool`` can switch a running sandbox's network, as ``PodmanPool`` can. ``SquashingPool``
can squash a snapshot, as ``MicroVMPool`` can. The ``use_pool`` fixture makes a session build one.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from sprout.worker.sandbox.session import SandboxSession
from sprout_sandbox import Snapshot
from sprout_sandbox import ToolResult as SdkToolResult

#: A number of calls larger than any probe makes: a runtime that starts this slowly never answers.
NEVER = 10**6


@dataclass
class StubRuntime:
    """The runtime of a stub sandbox: it records each call and answers ``ran <tool>``, or from ``answers``."""

    #: The answer to a ``command``, or the exception that its call raises.
    answers: dict[str, SdkToolResult | Exception] = field(default_factory=dict)
    #: How many calls fail with ``ConnectionError`` first, as while the runtime starts.
    starting_calls: int = 0
    #: ``(tool, args, agent_id)`` of each call, in call order.
    calls: list[tuple[str, dict[str, Any], str]] = field(default_factory=list)

    async def call(self, tool_name: str, args: dict[str, Any], agent_id: str = "") -> SdkToolResult:
        self.calls.append((tool_name, dict(args), agent_id))
        if self.starting_calls:
            self.starting_calls -= 1
            raise ConnectionError("the runtime is not listening yet")
        answer = self.answers.get(
            args.get("command", ""), SdkToolResult(output="ran " + tool_name, is_error=False)
        )
        if isinstance(answer, Exception):
            raise answer
        return answer


@dataclass
class StubSandbox:
    """A handle on one sandbox, with the attributes and the ``call`` of ``sprout_sandbox.Sandbox``."""

    sandbox_id: str
    backend: StubRuntime
    agent_id: str = ""

    async def call(self, tool_name: str, agent_id: str = "", **kwargs: Any) -> SdkToolResult:
        return await self.backend.call(tool_name, kwargs, agent_id or self.agent_id)


@dataclass
class NamelessSandbox:
    """A handle that has no ``sandbox_id``, as a sandbox object of another kind can have none."""

    backend: StubRuntime
    agent_id: str = ""

    async def call(self, tool_name: str, agent_id: str = "", **kwargs: Any) -> SdkToolResult:
        return await self.backend.call(tool_name, kwargs, agent_id or self.agent_id)


class StubPool:
    """Records each operation that a session asks of it. It starts ``sandbox-1``, then ``sandbox-2``..."""

    def __init__(
        self,
        *,
        refuse: Iterable[str] = (),
        can_snapshot: bool = True,
        can_transfer: bool = True,
        runtimes: dict[str, StubRuntime] | None = None,
        nameless: bool = False,
    ) -> None:
        self.refuse = set(refuse)
        self.can_snapshot = can_snapshot
        self.can_transfer = can_transfer
        #: The runtime of each sandbox, by id. A sandbox without one gets a runtime that answers at once.
        self.runtimes = dict(runtimes or {})
        #: Whether the sandboxes that it starts are ``NamelessSandbox`` handles.
        self.nameless = nameless
        #: Each operation, as a tuple of its name and the arguments that matter, in call order.
        self.operations: list[tuple[Any, ...]] = []
        self._started = 0

    def _do(self, *operation: Any) -> None:
        self.operations.append(operation)
        if operation[0] in self.refuse:
            raise RuntimeError("%s refused" % operation[0])

    def _handle(self, sandbox_id: str) -> StubSandbox:
        return StubSandbox(sandbox_id=sandbox_id, backend=self.runtimes.setdefault(sandbox_id, StubRuntime()))

    def _start(self) -> StubSandbox | NamelessSandbox:
        self._started += 1
        sandbox_id = "sandbox-%d" % self._started
        if self.nameless:
            return NamelessSandbox(backend=self.runtimes.setdefault(sandbox_id, StubRuntime()))
        return self._handle(sandbox_id)

    def supports_snapshot(self) -> bool:
        return self.can_snapshot

    def supports_cold_start(self) -> bool:
        return True

    def supports_upload(self) -> bool:
        return self.can_transfer

    def supports_download(self) -> bool:
        return self.can_transfer

    async def spawn(
        self, image: str, env: dict[str, str] | None = None, agent_id: str = ""
    ) -> StubSandbox | NamelessSandbox:
        self._do("spawn", image)
        return self._start()

    async def spawn_from_image(
        self, image: str, agent_id: str = "", resources: dict[str, Any] | None = None
    ) -> StubSandbox | NamelessSandbox:
        self._do("spawn_from_image", image, resources)
        return self._start()

    def attach(self, sandbox_id: str, agent_id: str = "") -> StubSandbox:
        self._do("attach", sandbox_id)
        return self._handle(sandbox_id)

    async def destroy(self, *sandboxes: StubSandbox | NamelessSandbox) -> None:
        for sandbox in sandboxes:
            self._do("destroy", _name(sandbox))

    async def forget(self, *sandboxes: StubSandbox) -> None:
        for sandbox in sandboxes:
            self._do("forget", sandbox.sandbox_id)

    async def snapshot(
        self, sandbox: StubSandbox, *, name: str | None = None, disk_only: bool = False
    ) -> Snapshot:
        self._do("snapshot", sandbox.sandbox_id, name, disk_only)
        return Snapshot(id="%s-snapshot" % sandbox.sandbox_id, disk_only=disk_only)

    async def upload_file(
        self, sandbox: StubSandbox, source: str | Path, destination: str, timeout: float | None = None
    ) -> None:
        self._do("upload_file", sandbox.sandbox_id, str(source), destination)

    async def download_file(
        self, sandbox: StubSandbox, source: str, destination: str | Path, timeout: float | None = None
    ) -> None:
        self._do("download_file", sandbox.sandbox_id, source, str(destination), timeout)
        Path(destination).write_bytes(b"the bytes of " + source.encode())


class NetworkedPool(StubPool):
    """A stub pool that can switch a running sandbox's network, as ``PodmanPool`` can."""

    async def set_network(self, sandbox: StubSandbox, allow: bool) -> None:
        self._do("set_network", sandbox.sandbox_id, allow)


class SquashingPool(StubPool):
    """A stub pool that can flatten a snapshot's layer chain, as ``MicroVMPool`` can."""

    async def squash(self, snapshot: Snapshot | str, *, name: str | None = None) -> Snapshot:
        snapshot_id = snapshot.id if isinstance(snapshot, Snapshot) else snapshot
        self._do("squash", snapshot_id, name)
        return Snapshot(id=snapshot_id + "-flat", rootfs_layers=1)


def _name(sandbox: StubSandbox | NamelessSandbox) -> str:
    return sandbox.sandbox_id if isinstance(sandbox, StubSandbox) else "a nameless sandbox"


def created(**session_args: Any) -> SandboxSession:
    """A quiet session whose ``create`` succeeded on the pool of ``use_pool``."""
    session = SandboxSession(quiet=True, **session_args)
    assert session.create("docker.io/library/python:3.11-slim"), session.create_error
    return session

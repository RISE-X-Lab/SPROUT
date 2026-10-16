from __future__ import annotations

from dataclasses import dataclass, field

from .backends import GatewayBackend, HttpBackend
from .result import ToolResult


@dataclass
class Sandbox:
    """A handle on one sandbox: its identity, and the channel to its runtime."""

    #: How calls reach the runtime: through AgentENV's proxy (``GatewayBackend``)
    #: or straight at a URL or Unix socket (``HttpBackend``). Both take the same call.
    backend: GatewayBackend | HttpBackend
    #: Who this handle acts as. Sent beside the arguments of every call, so a
    #: request names its caller on the wire and an argument called ``agent_id``
    #: can never stand in for one. The runtime does not read it; the harness
    #: binds it per channel (its own traffic versus the agent's).
    agent_id: str = ""
    _container_id: str | None = field(default=None, repr=False)

    def _whoami(self, agent_id: str = "") -> str:
        """An explicit per-call identity wins over the handle's own."""
        return agent_id or self.agent_id

    @property
    def sandbox_id(self) -> str | None:
        """The pool's id for this sandbox, once it has one."""
        return self._container_id

    def require_id(self) -> str:
        """The pool's id for this sandbox; an error for a handle without one."""
        sid = self._container_id
        if not sid:
            raise RuntimeError("sandbox has no id (already destroyed?)")
        return sid

    async def call(self, tool_name: str, agent_id: str = "", **kwargs) -> ToolResult:
        """Call a tool by name; the keyword arguments are the tool's arguments."""
        return await self.backend.call(tool_name, kwargs, self._whoami(agent_id))

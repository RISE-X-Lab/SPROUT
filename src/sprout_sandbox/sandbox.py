"""A handle on one sandbox, through which a caller calls the tools of the sandbox's runtime."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .backends import GatewayBackend, HttpBackend
from .result import ToolResult


@dataclass
class Sandbox:
    """A handle on one sandbox: its identity, and the channel to its runtime."""

    #: How calls reach the runtime: through AgentENV's proxy (``GatewayBackend``)
    #: or straight at a URL or Unix socket (``HttpBackend``). Both take the same call.
    backend: GatewayBackend | HttpBackend
    #: The caller identity that a call sends beside its tool arguments when the call gives none.
    #: An empty identity is not sent. The runtime ignores it, and no tool argument can replace it.
    agent_id: str = ""
    _container_id: str | None = field(default=None, repr=False)

    def _whoami(self, agent_id: str = "") -> str:
        """An explicit per-call identity wins over the handle's own."""
        return agent_id or self.agent_id

    @property
    def sandbox_id(self) -> str | None:
        """The pool's id for this sandbox, or ``None`` if it has none, for example after a destroy."""
        return self._container_id

    def require_id(self) -> str:
        """The pool's id for this sandbox. Raises ``RuntimeError`` if the handle has none."""
        sid = self._container_id
        if not sid:
            raise RuntimeError("sandbox has no id (already destroyed?)")
        return sid

    async def call(self, tool_name: str, agent_id: str = "", **kwargs: Any) -> ToolResult:
        """Call the tool ``tool_name`` with ``kwargs`` as its JSON arguments, and return its result.

        A non-empty ``agent_id`` replaces the handle's identity for this call. A tool that fails returns
        a result with ``is_error`` set. An HTTP failure raises an ``httpx`` exception.
        """
        return await self.backend.call(tool_name, kwargs, self._whoami(agent_id))

"""The transports from the host to the runtime in a sandbox, for a JSON-RPC ``tools/call``.

``GatewayBackend`` goes through a proxy that routes on a sandbox id. ``HttpBackend`` goes straight
to a URL or a Unix socket. Both take the same call and answer with a ``ToolResult``.
"""

from __future__ import annotations

from typing import Any

import httpx

from .result import ToolResult

#: Seconds the transport waits beyond a call's own ``timeout``, for the kill, the output and the reply.
#: A reply that the client gave up on reads as a transport failure, and the harness stops the rollout.
CALL_MARGIN_S = 60.0


def call_wait(args: dict[str, Any], default: float) -> float:
    """Seconds to wait for a ``tools/call`` with ``args``: its own ``timeout`` plus ``CALL_MARGIN_S``.

    The result is never less than ``default``. A missing or non-positive ``timeout`` gives ``default``.
    """
    timeout = args.get("timeout")
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
        return max(default, float(timeout) + CALL_MARGIN_S)
    return default


def call_params(tool_name: str, args: dict[str, Any], agent_id: str = "") -> dict[str, Any]:
    """The JSON-RPC params of a ``tools/call`` request.

    ``agent_id`` travels beside the arguments, not inside them, because it names the caller. The
    runtime never passes it to a tool. An empty ``agent_id`` is left out.
    """
    params: dict[str, Any] = {"name": tool_name, "arguments": args}
    if agent_id:
        params["agent_id"] = agent_id
    return params


class GatewayBackend:
    """Calls the runtime in a sandbox through a proxy, such as the one of the AgentENV microVM host.

    The proxy reads the sandbox id and the runtime's port from two headers, whose names are parameters.
    Each backend owns one HTTP client, and ``close`` releases it."""

    def __init__(
        self,
        gateway_url: str,
        sandbox_id: str,
        sandbox_id_header: str = "X-Sandbox-ID",
        target_port: int | None = None,
        target_port_header: str = "X-Target-Port",
    ) -> None:
        self.gateway_url = gateway_url.rstrip("/")
        self.sandbox_id = sandbox_id
        self.sandbox_id_header = sandbox_id_header
        self.target_port = target_port
        self.target_port_header = target_port_header
        self.timeout = 360.0
        self._client = httpx.AsyncClient(timeout=self.timeout)

    def _routing_headers(self) -> dict[str, str]:
        headers = {self.sandbox_id_header: self.sandbox_id}
        if self.target_port is not None:
            headers[self.target_port_header] = str(self.target_port)
        return headers

    async def call(self, tool_name: str, args: dict[str, Any], agent_id: str = "") -> ToolResult:
        """The result of ``tool_name`` with ``args``, called as ``agent_id``.

        A JSON-RPC error gives a failed result with no output. Raises ``httpx.HTTPError`` if the request
        fails or the proxy answers with an error status.
        """
        # AgentENV's proxy fails a call whose response headers take over 30 s. The runtime sends its
        # headers at once and heartbeats while the command runs, so a long call survives the proxy.
        resp = await self._client.post(
            self.gateway_url,
            headers=self._routing_headers(),
            json={
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": call_params(tool_name, args, agent_id),
            },
            timeout=call_wait(args, self.timeout),
        )
        resp.raise_for_status()
        data = resp.json()
        if data.get("error"):
            return ToolResult(output="", is_error=True)
        return ToolResult.from_response(data["result"])

    async def close(self) -> None:
        await self._client.aclose()


class HttpBackend:
    """Calls the runtime at an address that reaches it directly: a URL or, with ``uds``, a Unix socket.

    The address alone names the sandbox, with no routing header. With ``uds``, the URL supplies only the
    host header. Each backend owns one HTTP client, and ``close`` releases it."""

    def __init__(self, url: str, timeout: float = 360, uds: str | None = None) -> None:
        self.url = url.rstrip("/")
        self.uds = uds
        self.timeout = timeout
        transport = httpx.AsyncHTTPTransport(uds=uds) if uds else None
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def _rpc(
        self, method: str, params: dict[str, Any] | None = None, wait: float | None = None
    ) -> dict[str, Any]:
        body: dict[str, Any] = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            body["params"] = params
        resp = await self._client.post(self.url, json=body, timeout=wait or self.timeout)
        resp.raise_for_status()
        data: dict[str, Any] = resp.json()
        return data

    async def call(self, tool_name: str, args: dict[str, Any], agent_id: str = "") -> ToolResult:
        """The result of ``tool_name`` with ``args``, called as ``agent_id``.

        A JSON-RPC error gives a failed result whose output is the runtime's message. Raises
        ``httpx.HTTPError`` if the request fails or the runtime answers with an error status.
        """
        data = await self._rpc(
            "tools/call", call_params(tool_name, args, agent_id), wait=call_wait(args, self.timeout)
        )
        error = data.get("error")
        if error:
            # A protocol-level refusal, such as an unknown tool or malformed params, has no result
            # to parse. Its message is in the runtime's own words.
            message = error.get("message", "") if isinstance(error, dict) else str(error)
            return ToolResult(output=str(message), is_error=True)
        return ToolResult.from_response(data["result"])

    async def list_tools(self) -> list[dict[str, Any]]:
        """The runtime's tool declarations, from ``tools/list``.

        A bare array and the MCP shape ``{"tools": [...]}`` read the same. Raises ``RuntimeError`` on a
        JSON-RPC error, and ``httpx.HTTPError`` as :meth:`call` does.
        """
        data = await self._rpc("tools/list")
        if data.get("error"):
            raise RuntimeError("tools/list failed: %s" % data["error"])
        result = data.get("result")
        if isinstance(result, dict):
            result = result.get("tools")
        return list(result or [])

    async def close(self) -> None:
        await self._client.aclose()

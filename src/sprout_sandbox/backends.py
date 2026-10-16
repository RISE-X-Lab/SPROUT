from __future__ import annotations

import httpx

from .result import ToolResult


#: How much longer than a call's own ``timeout`` the transport waits for the
#: answer. The runtime replies once the command has finished or been killed, and
#: the kill, the output and the trip back take a moment; a reply the HTTP client
#: gave up on reads as a transport failure, which the harness has to treat as
#: "did it run?" and stops the rollout for.
CALL_MARGIN_S = 60.0


def call_wait(args: dict, default: float) -> float:
    """Seconds to wait for a tools/call: its own ``timeout`` plus the margin,
    never less than the client's default."""
    timeout = args.get("timeout")
    if isinstance(timeout, (int, float)) and not isinstance(timeout, bool) and timeout > 0:
        return max(default, float(timeout) + CALL_MARGIN_S)
    return default


def call_params(tool_name: str, args: dict, agent_id: str = "") -> dict:
    """Build the JSON-RPC params for a tools/call request.

    agent_id travels beside the arguments rather than inside them: it names the
    caller, which is transport-level addressing rather than an argument to the
    tool, and the runtime never hands it to one. Omitted when empty, so an
    anonymous caller sends the same request it always did.
    """
    params: dict = {"name": tool_name, "arguments": args}
    if agent_id:
        params["agent_id"] = agent_id
    return params


class GatewayBackend:
    """Calls the runtime inside a sandbox through a proxy that routes to it.

    The proxy reads a sandbox id from a header and forwards the body, and it
    also needs the port the runtime listens on inside the sandbox. Both header
    names are parameters, so a proxy that spells them differently is a
    different set of names rather than a different class. Used for sandboxes
    that are only reachable that way: the AgentENV microVM host.
    """

    def __init__(
        self,
        gateway_url: str,
        sandbox_id: str,
        sandbox_id_header: str = "X-Sandbox-ID",
        target_port: int | None = None,
        target_port_header: str = "X-Target-Port",
    ):
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

    async def call(self, tool_name: str, args: dict, agent_id: str = "") -> ToolResult:
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

    async def close(self):
        await self._client.aclose()


class HttpBackend:
    """Calls the runtime at an address that reaches it directly, with nothing in between.

    Either a URL, or -- with ``uds`` -- a Unix socket on this host, which is
    how the podman pool reaches the runtime in a container: the socket's
    directory is bind-mounted into the sandbox, so no port is published and
    the container's network can be taken away without losing the runtime.
    The URL then only supplies the request's host header. Either way the
    address alone names the sandbox: no routing header, no target port, and
    nothing but that one runtime answers there. The wire format is the same
    JSON-RPC the proxy forwards. One HTTP client per backend, which is one per
    sandbox handle; ``close`` releases it.
    """

    def __init__(self, url: str, timeout: float = 360, uds: "str | None" = None):
        self.url = url.rstrip("/")
        self.uds = uds
        self.timeout = timeout
        transport = httpx.AsyncHTTPTransport(uds=uds) if uds else None
        self._client = httpx.AsyncClient(timeout=timeout, transport=transport)

    async def _rpc(self, method: str, params: dict | None = None, wait: "float | None" = None) -> dict:
        body: dict = {"jsonrpc": "2.0", "id": 1, "method": method}
        if params is not None:
            body["params"] = params
        resp = await self._client.post(self.url, json=body, timeout=wait or self.timeout)
        resp.raise_for_status()
        return resp.json()

    async def call(self, tool_name: str, args: dict, agent_id: str = "") -> ToolResult:
        data = await self._rpc(
            "tools/call", call_params(tool_name, args, agent_id), wait=call_wait(args, self.timeout)
        )
        error = data.get("error")
        if error:
            # A protocol-level refusal (unknown tool, malformed params) has no
            # result to parse; its message is the runtime's own words.
            message = error.get("message", "") if isinstance(error, dict) else str(error)
            return ToolResult(output=str(message), is_error=True)
        return ToolResult.from_response(data["result"])

    async def list_tools(self) -> list[dict]:
        """The runtime's ``tools/list``: the readiness probe, and its schema.

        The runtime answers with a bare array; a server that wraps it in
        ``{"tools": [...]}`` (the MCP shape) is read the same way.
        """
        data = await self._rpc("tools/list")
        if data.get("error"):
            raise RuntimeError("tools/list failed: %s" % data["error"])
        result = data.get("result")
        if isinstance(result, dict):
            result = result.get("tools")
        return list(result or [])

    async def close(self):
        await self._client.aclose()

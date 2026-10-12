"""The sandbox handle: identity travels beside the arguments, never inside them."""

import asyncio
import json

import pytest

from sprout_sandbox import Sandbox
from sprout_sandbox.result import ToolResult


class RecordingBackend:
    """Stands in for GatewayBackend and records what each call arrived with."""

    def __init__(self):
        self.calls = []

    async def call(self, tool_name, args, agent_id=""):
        self.calls.append((tool_name, dict(args), agent_id))
        return ToolResult(output="ok", is_error=False)

    async def close(self):
        pass


def test_a_call_carries_the_handles_identity_unless_told_otherwise():
    backend = RecordingBackend()
    sb = Sandbox(backend=backend, agent_id="main")
    asyncio.run(sb.call("shell", command="ls"))
    asyncio.run(sb.call("shell", agent_id="other", command="ls"))
    assert [identity for _, _, identity in backend.calls] == ["main", "other"]
    # Omitting it everywhere stays anonymous rather than inventing an identity.
    asyncio.run(Sandbox(backend=backend).call("shell", command="ls"))
    assert backend.calls[-1][2] == ""


@pytest.mark.parametrize("agent_id, expected", [
    # Beside the arguments, never among them: identity is transport-level
    # addressing, not something the tool should see.
    ("agent-a", {"name": "shell", "arguments": {"command": "ls"}, "agent_id": "agent-a"}),
    # Anonymous calls send exactly what they always did -- no key at all.
    ("", {"name": "shell", "arguments": {"command": "ls"}}),
])
def test_identity_travels_beside_the_arguments_on_the_wire(agent_id, expected):
    import httpx

    from sprout_sandbox.backends import HttpBackend

    bodies = []

    def answer(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1,
                                         "result": {"content": [{"type": "text", "text": "ok"}]}})

    async def call():
        # Both backends build their params the same way (call_params), so one
        # transport stands for both.
        backend = HttpBackend("http://runtime")
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(answer), timeout=backend.timeout)
        await Sandbox(backend=backend).call("shell", agent_id=agent_id, command="ls")
        await backend.close()

    asyncio.run(call())
    assert [body["params"] for body in bodies] == [expected]


def test_a_call_waits_for_its_own_timeout_plus_the_margin():
    """A command may run for its whole ``timeout``; the transport must still be
    listening when the answer comes, or a long command reads as a lost call --
    which the harness stops the rollout for. Without a ``timeout`` the client's
    default holds (the runtime's own default, 300 s, fits under it)."""
    import httpx

    from sprout_sandbox.backends import GatewayBackend, HttpBackend

    waits = []

    def answer(request):
        waits.append(request.extensions["timeout"]["read"])
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": 1,
                                         "result": {"content": [{"type": "text", "text": "ok"}]}})

    async def calls():
        for backend in (HttpBackend("http://runtime"), GatewayBackend("http://gateway", "sb-1")):
            backend._client = httpx.AsyncClient(transport=httpx.MockTransport(answer), timeout=backend.timeout)
            await backend.call("shell", {"command": "sleep 500", "timeout": 600})
            await backend.call("shell", {"command": "true"})
            await backend.close()

    asyncio.run(calls())
    assert waits == [660.0, 360.0, 660.0, 360.0]

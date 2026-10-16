"""Tests for MicroVMPool, the harness's client for AgentENV.

A stub HTTP server stands in for the AgentENV API, since Firecracker needs
/dev/kvm and a bare-metal host: what is verified here is the client contract
(endpoints, payloads, routing headers), not the hypervisor itself.
User instruction: "pool 能加一个firecracker的backend吗" + "B吧".
"""

import asyncio
import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from sprout_sandbox import MicroVMPool


class FakeAenv(BaseHTTPRequestHandler):
    """Minimal stand-in for the AgentENV HTTP API."""

    requests: list = []  # (method, path, body, headers)
    next_id = 0

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length") or 0)
        return json.loads(self.rfile.read(length) or b"{}") if length else {}

    def _record(self, method: str) -> dict:
        body = self._read_body()
        FakeAenv.requests.append((method, self.path, body, dict(self.headers)))
        return body

    def _reply(self, payload: dict | list):
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_POST(self):
        self._record("POST")
        if self.path in ("/sandboxes", "/sandboxes-cold"):
            FakeAenv.next_id += 1
            return self._reply({"sandboxID": f"vm-{FakeAenv.next_id}"})
        if self.path.rstrip("/") == "":
            # A tool call proxied into a sandbox.
            return self._reply({"result": {"content": [{"type": "text", "text": "ok"}],
                                           "isError": False}})
        return self._reply({})

    def do_DELETE(self):
        self._record("DELETE")
        self._reply({})

    def log_message(self, *args):
        pass


@pytest.fixture
def aenv():
    FakeAenv.requests = []
    FakeAenv.next_id = 0
    server = HTTPServer(("127.0.0.1", 0), FakeAenv)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()


def paths_of(method: str) -> list:
    return [p for m, p, _, _ in FakeAenv.requests if m == method]


def test_failed_delete_keeps_the_sandbox_owned():
    import httpx

    pool = MicroVMPool("http://agentenv")

    async def scenario():
        await pool._client.aclose()
        pool._client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(500, json={"error": "busy"})))
        sandbox = pool.attach("owned-vm")
        with pytest.raises(httpx.HTTPStatusError):
            await pool.destroy(sandbox)
        assert sandbox.sandbox_id == "owned-vm"
        assert pool.list() == [sandbox]
        await pool._client.aclose()

    asyncio.run(scenario())


def test_spawn_needs_a_template_or_snapshot(aenv):
    # There is no default template to fall back on: every caller names what it
    # starts from, and a request carrying `templateID: null` would fail on the
    # server with a far less useful error.
    pool = MicroVMPool(aenv)

    async def scenario():
        with pytest.raises(ValueError, match="template or snapshot"):
            await pool.spawn(image=None)
        await pool.close()

    asyncio.run(scenario())
    assert FakeAenv.requests == []


def test_calls_route_through_the_proxy_headers(aenv):
    pool = MicroVMPool(aenv, runtime_port=3000)

    async def scenario():
        # A pool-provisioned sandbox comes out named, so every call made through
        # the handle says who made it without each call site remembering to.
        sb = await pool.spawn(image="swe-base", agent_id="reviewer")
        await sb.call("shell", command="true")
        await pool.close()

    asyncio.run(scenario())
    # The tool call must carry AgentENV's routing headers: which sandbox, and
    # which port the runtime listens on inside it.
    tool_calls = [(body, h) for m, p, body, h in FakeAenv.requests
                  if m == "POST" and p.rstrip("/") == ""]
    assert tool_calls, "no proxied tool call was made"
    body, headers = tool_calls[-1]
    assert headers["x-agentenv-sandbox-id"] == "vm-1"
    assert headers["x-agentenv-target-port"] == "3000"
    params = body["params"]
    assert params["agent_id"] == "reviewer"
    # Identity is transport plumbing, not something the tool (or a model
    # reading the call) should see among the arguments.
    assert "agent_id" not in params.get("arguments", {})


def test_long_lease_reaches_create_cold_start_and_restore(aenv):
    pool = MicroVMPool(aenv, sandbox_ttl=11400)

    async def scenario():
        await pool.spawn(image="task-template")
        await pool.spawn_from_image("registry/task:tag", resources={"cpu": 2, "memory_mb": 8192, "disk_size_mb": 65536})
        await pool.spawn(image="saved-snapshot")
        await pool.close()

    asyncio.run(scenario())
    timed = [(path, body) for method, path, body, _ in FakeAenv.requests
             if method == "POST" and "timeout" in body]
    assert len(timed) == 3
    assert all(body["timeout"] == 11400 for _, body in timed)
    assert timed[0][1]["templateID"] == "task-template"
    assert timed[1][0] == "/sandboxes-cold"
    assert timed[1][1]["diskSizeMB"] == 65536
    assert timed[2][1]["templateID"] == "saved-snapshot"


def test_spawn_sends_ttl_and_pause_policy(aenv):
    # AgentENV's default TTL is 15s, after which the VM pauses — far too short
    # for an agent turn, so spawn must always pick the TTL explicitly. A paused
    # sandbox must come back on its next call, or a sandbox kept for grading
    # after the run is dead by the time the grader reaches it.
    # AgentENV validates X-API-KEY; an Authorization: Bearer header is not
    # checked (observed against a live server: garbage bearer tokens pass).
    pool = MicroVMPool(aenv, sandbox_ttl=900, api_key="secret-key")

    async def scenario():
        sb = await pool.spawn(image="swe-base")
        assert sb.sandbox_id == "vm-1"
        assert len(pool.list()) == 1
        # `agent_id` is optional: a single-agent sandbox need not name anyone.
        assert sb.agent_id == ""
        await pool.close()

    asyncio.run(scenario())
    _, path, body, headers = FakeAenv.requests[0]
    assert path == "/sandboxes"
    assert body["templateID"] == "swe-base"
    assert body["timeout"] == 900
    assert body["autoPause"] is True
    assert body["autoResume"] == {"enabled": True}
    assert headers.get("X-API-KEY") == "secret-key"
    assert "Authorization" not in headers


@pytest.mark.parametrize("allow", [True, False, None])
@pytest.mark.parametrize("cold_start", [True, False])
def test_network_policy_reaches_cold_and_restored_sandboxes(aenv, allow, cold_start):
    # A no-network benchmark is only no-network if the flag is on the request
    # AgentENV sees, on both create paths; and the default must stay absent so
    # the server's own policy applies when nobody asked.
    pool = MicroVMPool(aenv, allow_internet=allow)

    async def scenario():
        if cold_start:
            await pool.spawn_from_image("fixture/image")
        else:
            await pool.spawn(image="fixture-snapshot")
        await pool.close()

    asyncio.run(scenario())
    _, path, body, _ = FakeAenv.requests[0]
    assert path == ("/sandboxes-cold" if cold_start else "/sandboxes")
    if allow is None:
        assert "allowInternetAccess" not in body
        assert "allow_internet_access" not in body
    else:
        assert body["allowInternetAccess"] is allow
        # The flag is spelled two ways: POST /sandboxes on the deployed server
        # reads the snake_case key only, /sandboxes-cold only the camelCase one.
        if not cold_start:
            assert body["allow_internet_access"] is allow


def test_destroy_removes_only_the_named_sandbox(aenv):
    pool = MicroVMPool(aenv)

    async def scenario():
        a = await pool.spawn(image="swe-base")
        b = await pool.spawn(image="swe-base")
        await pool.destroy(a)
        return a, b

    a, b = asyncio.run(scenario())
    assert paths_of("DELETE") == ["/sandboxes/vm-1"]
    assert [sb.sandbox_id for sb in pool.list()] == [b.sandbox_id]


def test_missing_sandbox_id_in_response_is_an_error():
    from sprout_sandbox.pool import _sandbox_id

    with pytest.raises(RuntimeError, match="no sandbox id"):
        _sandbox_id({"unexpected": "shape"})
    # AgentENV's create responses spell it `sandboxID`; nothing else is guessed at.
    with pytest.raises(RuntimeError, match="no sandbox id"):
        _sandbox_id({"sandbox_id": "vm-1"})


def test_the_pool_serves_the_sessions_contract():
    """`harness_sandbox.session.SandboxSession` duck-types on exactly these."""
    pool = MicroVMPool("http://aenv:8000")
    for name in ("spawn", "spawn_from_image", "destroy", "close", "snapshot", "squash",
                 "upload_file", "download_file", "supports_snapshot",
                 "supports_cold_start", "supports_upload", "supports_download"):
        assert callable(getattr(pool, name, None)), name

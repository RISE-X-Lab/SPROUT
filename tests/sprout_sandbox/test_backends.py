"""The two transports to a sandbox's runtime, and the ``Sandbox`` handle that calls through them.

Each medium test talks JSON-RPC to the fake runtime of ``fake_runtime.py``, on a local port or on a
Unix socket, as the runtime in a microVM or in a podman container is reached.
"""

from __future__ import annotations

import json

import httpx
import pytest

from sprout_sandbox import GatewayBackend, HttpBackend, Sandbox, ToolResult
from sprout_sandbox.backends import call_wait
from tests.sprout_sandbox.fake_runtime import Reply, rpc_error, rpc_result, serving_runtime

pytestmark = pytest.mark.anyio

#: A ``shell`` reply that carries a ``CommandOutcome``: the command printed to both streams and failed.
OUTCOME = {"stdout": "out\n", "stderr": "err\n", "exit_code": 2, "stdout_bytes": 4, "stderr_bytes": 4}


def backend_to(runtime, kind: str) -> HttpBackend | GatewayBackend:
    """A backend of ``kind`` (``http`` or ``gateway``) to the fake ``runtime``, also a proxy."""
    if kind == "http":
        return HttpBackend(runtime.url)
    return GatewayBackend(runtime.url, "sb-1")


def with_wait(backend: HttpBackend | GatewayBackend, seconds: float) -> HttpBackend | GatewayBackend:
    """``backend`` with the default wait ``seconds`` for a call that gives no ``timeout``."""
    backend.timeout = seconds
    return backend


def present(headers: dict[str, str], *names: str) -> dict[str, str]:
    """The headers among ``names`` that are present."""
    return {name: headers[name] for name in names if name in headers}


@pytest.mark.parametrize(
    "args, default, expected",
    [
        pytest.param({"command": "true"}, 10.0, 10.0, id="no-timeout"),
        pytest.param({"timeout": 600}, 10.0, 660.0, id="timeout-plus-margin"),
        pytest.param({"timeout": 1.5}, 10.0, 61.5, id="fractional-timeout"),
        pytest.param({"timeout": 100}, 360.0, 360.0, id="default-is-the-floor"),
        pytest.param({"timeout": 0}, 10.0, 10.0, id="zero"),
        pytest.param({"timeout": -5}, 10.0, 10.0, id="negative"),
        pytest.param({"timeout": True}, 10.0, 10.0, id="bool"),
        pytest.param({"timeout": "600"}, 10.0, 10.0, id="string"),
    ],
)
def test_a_call_waits_for_its_own_timeout_plus_a_margin_and_never_less_than_the_default(
    args, default, expected
):
    assert call_wait(args, default) == expected


@pytest.mark.parametrize("kind", ["http", "gateway"])
async def test_a_call_with_a_timeout_outlasts_the_backends_default_wait(kind):
    """A command may run for its whole ``timeout``. A reply that the client gave up on reads as a lost
    sandbox, and the harness stops the rollout."""
    with serving_runtime() as runtime:
        runtime.reply = Reply(body=rpc_result({"content": [{"type": "text", "text": "done"}]}), delay_s=0.5)
        backend = with_wait(backend_to(runtime, kind), 0.1)

        result = await backend.call("shell", {"command": "sleep 0.5", "timeout": 1})

    assert result == ToolResult(output="done", is_error=False)


@pytest.mark.parametrize("kind", ["http", "gateway"])
async def test_a_call_without_a_timeout_gives_up_after_the_backends_default_wait(kind):
    with serving_runtime() as runtime:
        runtime.reply = Reply(body=rpc_result({"content": []}), delay_s=0.5)
        backend = with_wait(backend_to(runtime, kind), 0.1)

        with pytest.raises(httpx.ReadTimeout):
            await backend.call("shell", {"command": "sleep 0.5"})


@pytest.mark.parametrize(
    "handle_identity, call_identity, expected_params",
    [
        pytest.param(
            "main",
            "",
            {"name": "shell", "arguments": {"command": "ls"}, "agent_id": "main"},
            id="the-handles",
        ),
        pytest.param(
            "main",
            "reviewer",
            {"name": "shell", "arguments": {"command": "ls"}, "agent_id": "reviewer"},
            id="the-calls-over-the-handles",
        ),
        pytest.param(
            "",
            "reviewer",
            {"name": "shell", "arguments": {"command": "ls"}, "agent_id": "reviewer"},
            id="the-calls",
        ),
        pytest.param("", "", {"name": "shell", "arguments": {"command": "ls"}}, id="none"),
    ],
)
async def test_a_call_sends_its_identity_beside_the_arguments(
    handle_identity, call_identity, expected_params
):
    """The identity names the caller, so the runtime never passes it to a tool. An empty one is not sent."""
    with serving_runtime() as runtime:
        sandbox = Sandbox(backend=HttpBackend(runtime.url), agent_id=handle_identity)

        await sandbox.call("shell", agent_id=call_identity, command="ls")

    assert runtime.bodies() == [
        {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": expected_params}
    ]


@pytest.mark.parametrize(
    "target_port, expected_headers",
    [
        pytest.param(3000, {"x-sandbox-id": "sb-1", "x-target-port": "3000"}, id="port"),
        pytest.param(None, {"x-sandbox-id": "sb-1"}, id="no-port"),
    ],
)
async def test_a_gateway_call_names_the_sandbox_and_the_runtimes_port_in_the_proxys_headers(
    target_port, expected_headers
):
    with serving_runtime() as runtime:
        backend = GatewayBackend(
            runtime.url,
            "sb-1",
            sandbox_id_header="X-Sandbox-ID",
            target_port=target_port,
            target_port_header="X-Target-Port",
        )

        await backend.call("shell", {"command": "ls"})

    [request] = runtime.received
    assert present(request.headers, "x-sandbox-id", "x-target-port") == expected_headers


@pytest.mark.parametrize("kind", ["http", "gateway"])
async def test_a_reply_with_a_command_outcome_fills_the_outcome_fields_on_both_transports(kind):
    text = json.dumps(OUTCOME)
    with serving_runtime() as runtime:
        runtime.reply = Reply(body=rpc_result({"content": [{"type": "text", "text": text}], "isError": True}))

        result = await backend_to(runtime, kind).call("shell", {"command": "make"})

    assert result == ToolResult(
        output=text,
        is_error=True,
        exit_code=2,
        stdout="out\n",
        stderr="err\n",
        stdout_bytes=4,
        stderr_bytes=4,
    )


@pytest.mark.parametrize(
    "error, expected_output",
    [
        pytest.param({"code": -32601, "message": "unknown tool"}, "unknown tool", id="message"),
        pytest.param({"code": -32602}, "", id="no-message"),
        pytest.param("malformed params", "malformed params", id="text"),
    ],
)
async def test_a_json_rpc_error_is_a_failed_result_in_the_runtimes_words(error, expected_output):
    with serving_runtime() as runtime:
        runtime.reply = Reply(body=rpc_error(error))

        result = await HttpBackend(runtime.url).call("nope", {})

    assert result == ToolResult(output=expected_output, is_error=True)


async def test_a_json_rpc_error_through_the_gateway_is_a_failed_result_with_no_output():
    with serving_runtime() as runtime:
        runtime.reply = Reply(body=rpc_error({"code": -32601, "message": "unknown tool"}))

        result = await GatewayBackend(runtime.url, "sb-1").call("nope", {})

    assert result == ToolResult(output="", is_error=True)


@pytest.mark.parametrize("kind", ["http", "gateway"])
async def test_an_error_status_raises_on_both_transports(kind):
    with serving_runtime() as runtime:
        runtime.reply = Reply(body={"error": "no such sandbox"}, status=502)

        with pytest.raises(httpx.HTTPStatusError, match="502 Bad Gateway"):
            await backend_to(runtime, kind).call("shell", {"command": "ls"})


@pytest.mark.parametrize(
    "result",
    [
        pytest.param([{"name": "shell"}], id="bare-array"),
        pytest.param({"tools": [{"name": "shell"}]}, id="mcp-shape"),
    ],
)
async def test_the_tool_list_reads_the_same_as_a_bare_array_or_in_the_mcp_shape(result):
    with serving_runtime() as runtime:
        runtime.reply = Reply(body=rpc_result(result))

        tools = await HttpBackend(runtime.url).list_tools()

    assert tools == [{"name": "shell"}]
    assert runtime.bodies() == [{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}]


async def test_a_json_rpc_error_on_the_tool_list_raises():
    with serving_runtime() as runtime:
        runtime.reply = Reply(body=rpc_error({"code": -32603, "message": "starting"}))

        with pytest.raises(
            RuntimeError, match=r"^tools/list failed: \{'code': -32603, 'message': 'starting'\}$"
        ):
            await HttpBackend(runtime.url).list_tools()


async def test_a_backend_with_a_unix_socket_reaches_the_runtime_listening_on_it(tmp_path):
    """The URL then supplies only the host header, as for the runtime in a podman container."""
    with serving_runtime(uds=tmp_path / "runtime.sock") as runtime:
        backend = HttpBackend("http://sprout-runtime", uds=runtime.uds)

        result = await backend.call("shell", {"command": "echo ok"})

    [request] = runtime.received
    assert (result, request.headers["host"]) == (ToolResult(output="ok", is_error=False), "sprout-runtime")

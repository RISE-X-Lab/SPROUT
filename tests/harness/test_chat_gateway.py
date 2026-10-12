"""Run control reaching the transports, and shell-specific settlement.

A stop must cancel a silent HTTP post and close the gateway's upstream
connection, or a cut-off rollout leaves a model generating for nobody.
"""

import threading

import httpx
import pytest

from sprout.harness.core.control import RunAborted, RunControl
from sprout.harness.gateway.routing import ModelRoute, RoutingTable
from sprout.harness.gateway.server import GatewayServer


def test_stop_cancels_silent_http_transport(monkeypatch):
    import asyncio
    from sprout.harness.core.control import post

    started, closed = threading.Event(), threading.Event()
    control = RunControl()

    async def silent(*args, **kwargs):
        try:
            started.set()
            await asyncio.Event().wait()
        finally:
            closed.set()

    monkeypatch.setattr(httpx.AsyncClient, "post", silent)
    stopper = threading.Thread(target=lambda: (started.wait(2), control.request_stop("cutoff")))
    stopper.start()
    try:
        with pytest.raises(RunAborted, match="cutoff"):
            post("http://example.invalid", timeout_s=30, control=control)
        assert closed.is_set()
    finally:
        stopper.join(2)


def test_plain_success_metadata_is_specific_to_shell():
    from sprout.harness.core.result import ToolResult
    from sprout.harness.tools.access import _settled

    def plain():
        return ToolResult(True, "stdout\n")

    command = _settled("shell", plain())
    assert command.outcome.exit_code == 0 and command.outcome.stdout == "stdout\n"
    assert _settled("not_shell", plain()).outcome is None
    assert _settled("shell", ToolResult(False, "rejected")).outcome is None


def test_gateway_stop_closes_the_real_upstream_connection():
    import asyncio
    from concurrent.futures import ThreadPoolExecutor
    from fastapi import FastAPI, Request
    from tests.serving import serving

    app = FastAPI()
    entered, disconnected = threading.Event(), threading.Event()
    control = RunControl()

    @app.post("/v1/chat/completions")
    async def silent(request: Request):
        entered.set()
        for _ in range(200):
            if await request.is_disconnected():
                disconnected.set()
                return {}
            await asyncio.sleep(.01)
        return {}

    class Policy:
        def __init__(self):
            self.control = control
        def prepare_model_request(self, payload):
            return payload

    with serving(app) as url:
        table = RoutingTable({"default": ModelRoute(base_url=url)})
        with GatewayServer(table, request_policy=Policy()) as gateway:
            token = table.mint("fixture")
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(httpx.post, gateway.base_url + "/v1/chat/completions",
                                         json={"model": "fixture", "messages": []},
                                         headers={"Authorization": "Bearer " + token.token}, timeout=5)
                assert entered.wait(2)
                control.request_stop("rollout deadline", stop_reason="timeout")
                assert future.result(timeout=3).status_code == 502
                assert disconnected.wait(2), "upstream generation connection survived rollout stop"

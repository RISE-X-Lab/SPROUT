"""``GatewayServer`` relays an agent's Chat Completions request to its route, and meters the answer.

A local provider stands in for the model endpoint. It records each request and answers with a
scripted reply, so that a test sees both sides of the relay.
"""

from __future__ import annotations

import gzip
import json
import socket
import threading
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.worker.gateway import GatewayServer, ModelRoute, RoutingTable
from sprout.worker.gateway.routing import SlotToken
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter, read_journal
from tests.support.waiting import wait_until

CHAT = "/v1/chat/completions"
#: The fields that the journal adds to each event. A gateway record sets its own ``agent_id``.
ENVELOPE = {"v", "ts", "seq", "run_id", "sandbox_id"}

#: A provider's answer, with a field that no OpenAI client knows.
COMPLETION = {
    "id": "chatcmpl-provider",
    "object": "chat.completion",
    "model": "provider-model",
    "choices": [
        {
            "index": 0,
            "message": {"role": "assistant", "content": "pong"},
            "finish_reason": "stop",
            "provider_extra": {"kept": True},
        }
    ],
    "usage": {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "prompt_tokens_details": {"cached_tokens": 3},
        "completion_tokens_details": {"reasoning_tokens": 2},
    },
}
#: One USD for each uncached input token. COMPLETION costs 8 USD: 11 input tokens, 3 of them cached.
ONE_DOLLAR_A_TOKEN = {"input": 1_000_000.0}


@dataclass(frozen=True)
class Reply:
    """One answer of the provider. ``gzip`` encodes the body. ``hang`` sends nothing until the client goes."""

    body: dict[str, Any] = field(default_factory=lambda: COMPLETION)
    status: int = 200
    headers: dict[str, str] = field(default_factory=dict)
    gzip: bool = False
    hang: bool = False


@dataclass(frozen=True)
class Received:
    """One request that reached the provider. Header names are lower case."""

    path: str
    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class Provider:
    url: str
    requests: list[Received] = field(default_factory=list)
    #: Set when a client closes a connection that a ``hang`` reply holds open.
    disconnected: threading.Event = field(default_factory=threading.Event)


@contextmanager
def provider(*replies: Reply) -> Iterator[Provider]:
    """A provider that answers its requests with ``replies`` in order, and then with ``Reply()``."""
    script = iter(replies)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            headers = {key.lower(): value for key, value in self.headers.items()}
            server.requests.append(Received(path=self.path, headers=headers, body=body))
            reply = next(script, Reply())
            if reply.hang:
                self._hold()
                return
            data = json.dumps(reply.body).encode()
            data = gzip.compress(data) if reply.gzip else data
            self.send_response(reply.status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            if reply.gzip:
                self.send_header("Content-Encoding", "gzip")
            for key, value in reply.headers.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def _hold(self) -> None:
            # recv returns b"" once the client closes the connection, and times out otherwise.
            self.connection.settimeout(10)
            if self.connection.recv(1) == b"":
                server.disconnected.set()
            self.close_connection = True

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    server = Provider(url="http://127.0.0.1:%d" % httpd.server_port)
    # A short poll interval, because shutdown() waits for the next poll.
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


@contextmanager
def gateway(table: RoutingTable, journal_path: Path, **options: Any) -> Iterator[GatewayServer]:
    """A started ``GatewayServer`` for ``table`` that journals to ``journal_path``."""
    with JournalWriter(journal_path, run_id="gateway-run") as journal:
        with GatewayServer(table, journal=journal, **options) as server:
            yield server


def ask(
    server: GatewayServer, token: SlotToken | None, body: dict[str, Any], **headers: str
) -> httpx.Response:
    """The gateway's answer to the chat request ``body`` that the slot of ``token`` sends."""
    if token is not None:
        headers["Authorization"] = "Bearer " + token.token
    return httpx.post(server.base_url + CHAT, json=body, headers=headers, timeout=10)


def gateway_records(journal_path: Path, status: str | None = None) -> list[dict[str, Any]]:
    """The ``gateway.request`` records of the journal, with ``status`` if given, without their envelope."""
    return [
        {key: value for key, value in record.items() if key not in ENVELOPE}
        for record in read_journal(journal_path)
        if record["type"] == "gateway.request" and status in (None, record["status"])
    ]


def closed_port_url() -> str:
    """The URL of a local port where nothing listens."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return "http://127.0.0.1:%d" % listener.getsockname()[1]


def error(kind: str, message: str) -> dict[str, Any]:
    return {"error": {"type": kind, "message": message}}


class Overrides:
    """A request policy that sets ``changes`` on each request."""

    control = None

    def __init__(self, changes: dict[str, Any]) -> None:
        self.changes = changes

    def prepare_model_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, **self.changes}


class Refusal:
    """A request policy that refuses each request with ``reason``."""

    control = None

    def __init__(self, reason: str) -> None:
        self.reason = reason

    def prepare_model_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        raise ValueError(self.reason)


class Stoppable:
    """A request policy that changes nothing, for a run whose stop is ``control``."""

    def __init__(self, control: RunControl) -> None:
        self.control = control

    def prepare_model_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        return payload


# --- refusals before the provider -----------------------------------------------------------


@pytest.mark.parametrize(
    "headers", [{}, {"Authorization": "Bearer sprout-slot-never-minted"}], ids=["missing", "unknown"]
)
def test_a_request_without_a_minted_slot_token_gets_a_401_and_is_not_forwarded(tmp_path, headers):
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        with gateway(table, tmp_path / "journal.jsonl") as server:
            response = ask(server, None, {"model": "m", "messages": []}, **headers)

    assert (response.status_code, response.json(), upstream.requests) == (
        401,
        error("authentication_error", "unknown or missing slot token"),
        [],
    )


@pytest.mark.parametrize("method, path", [("POST", "/v1/messages"), ("GET", "/v1/models")])
def test_a_path_other_than_chat_completions_gets_a_404(tmp_path, method, path):
    table = RoutingTable({"default": ModelRoute(base_url=closed_port_url())})
    token = table.mint("agent-1")
    with gateway(table, tmp_path / "journal.jsonl") as server:
        response = httpx.request(
            method, server.base_url + path, json={}, headers={"Authorization": "Bearer " + token.token}
        )

    assert (response.status_code, response.json()) == (
        404,
        error("invalid_request_error", "unknown path %s" % path),
    )


def test_the_health_path_answers_ok(tmp_path):
    with gateway(RoutingTable(), tmp_path / "journal.jsonl") as server:
        response = httpx.get(server.base_url + "/health")

    assert (response.status_code, response.json()) == (200, {"status": "ok"})


def test_a_body_that_is_not_json_gets_a_400_and_is_not_forwarded(tmp_path):
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            response = httpx.post(
                server.base_url + CHAT,
                content=b"{not json",
                headers={"Authorization": "Bearer " + token.token},
            )

    assert (response.status_code, response.json(), upstream.requests) == (
        400,
        error("invalid_request_error", "body is not valid JSON"),
        [],
    )


def test_a_streamed_request_gets_a_terminal_400_and_is_not_forwarded(tmp_path):
    """The gateway cannot meter a stream before it relays it. A 429 would make the SDK retry."""
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            response = ask(server, token, {"model": "m", "messages": [], "stream": True})

    assert (response.status_code, response.json(), upstream.requests) == (
        400,
        error("invalid_request_error", "this gateway does not relay streamed responses; send stream=false"),
        [],
    )


def test_without_a_default_route_an_unmapped_model_gets_a_404_and_is_not_forwarded(tmp_path):
    with provider() as upstream:
        table = RoutingTable({"known-model": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            response = ask(server, token, {"model": "unmapped-model", "messages": []})

    assert (response.status_code, response.json(), upstream.requests) == (
        404,
        error("invalid_request_error", "no route for model 'unmapped-model' and no default route"),
        [],
    )


# --- what reaches the provider --------------------------------------------------------------


def test_a_routed_model_is_renamed_and_sent_with_the_route_key_instead_of_the_slot_token(tmp_path):
    with provider() as upstream:
        table = RoutingTable(
            {
                "default": ModelRoute(base_url=closed_port_url(), api_key="default-key"),
                "sprout-ckpt-42": ModelRoute(
                    base_url=upstream.url, api_key="rl-key", upstream_model="ckpt-42"
                ),
            }
        )
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            ask(server, token, {"model": "sprout-ckpt-42", "messages": []})

    sent = upstream.requests[0]
    assert (sent.path, sent.body, sent.headers["authorization"]) == (
        CHAT,
        {"model": "ckpt-42", "messages": []},
        "Bearer rl-key",
    )
    assert token.token not in json.dumps(sent.headers)


def test_an_unmapped_model_goes_to_the_default_route_under_its_own_name(tmp_path):
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url, api_key="default-key")})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            ask(server, token, {"model": "unmapped-model", "messages": []})

    sent = upstream.requests[0]
    assert (sent.body, sent.headers["authorization"]) == (
        {"model": "unmapped-model", "messages": []},
        "Bearer default-key",
    )


def test_a_route_without_a_key_sends_no_authorization_to_the_provider(tmp_path):
    """The slot token never travels, and no other credential takes its place."""
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            ask(server, token, {"model": "m", "messages": []})

    assert "authorization" not in upstream.requests[0].headers


def test_a_request_on_a_route_whose_key_variable_is_unset_gets_a_502_and_is_not_forwarded(
    tmp_path, monkeypatch
):
    """The provider would take a request without its key as an anonymous one."""
    monkeypatch.delenv("SPROUT_TEST_ROUTE_KEY", raising=False)
    with provider() as upstream:
        table = RoutingTable(
            {"default": ModelRoute(base_url=upstream.url, api_key_env="SPROUT_TEST_ROUTE_KEY")}
        )
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            answer = ask(server, token, {"model": "m", "messages": []})

    assert (answer.status_code, upstream.requests) == (502, [])


def test_the_agents_own_headers_reach_the_provider(tmp_path):
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            ask(server, token, {"model": "m", "messages": []}, **{"x-session-id": "session-9"})

    assert upstream.requests[0].headers["x-session-id"] == "session-9"


def test_the_policy_changes_the_request_before_it_is_forwarded(tmp_path):
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        policy = Overrides({"model": "contract-model", "temperature": 0.4})
        with gateway(table, tmp_path / "journal.jsonl", request_policy=policy) as server:
            ask(server, token, {"model": "agent-model", "messages": [], "temperature": 1.0})

    assert upstream.requests[0].body == {"model": "contract-model", "messages": [], "temperature": 0.4}


def test_a_request_that_the_policy_refuses_gets_a_400_with_its_reason_and_is_not_forwarded(tmp_path):
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        policy = Refusal("rollout turn budget exhausted")
        with gateway(table, tmp_path / "journal.jsonl", request_policy=policy) as server:
            response = ask(server, token, {"model": "m", "messages": []})

    assert (response.status_code, response.json(), upstream.requests) == (
        400,
        error("invalid_request_error", "rollout turn budget exhausted"),
        [],
    )


# --- what comes back to the agent -----------------------------------------------------------


@pytest.mark.parametrize(
    "reply",
    [
        Reply(headers={"x-request-id": "req-abc"}),
        Reply(
            status=400, body=error("invalid_request_error", "bad input"), headers={"x-request-id": "req-abc"}
        ),
    ],
    ids=["success", "provider-error"],
)
def test_the_providers_status_body_and_headers_come_back_to_the_agent(tmp_path, reply):
    with provider(reply) as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            response = ask(server, token, {"model": "m", "messages": []})

    assert (response.status_code, response.json(), response.headers["x-request-id"]) == (
        reply.status,
        reply.body,
        "req-abc",
    )


def test_a_compressed_answer_reaches_the_agent_decoded(tmp_path):
    """The gateway sends the decoded body, so it must not relay the provider's ``Content-Encoding``."""
    with provider(Reply(gzip=True)) as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            response = ask(server, token, {"model": "m", "messages": []})

    assert (response.json(), response.headers.get("content-encoding")) == (COMPLETION, None)


def test_an_unreachable_provider_gets_a_502_and_the_gateway_keeps_serving(tmp_path):
    table = RoutingTable({"default": ModelRoute(base_url=closed_port_url())})
    token = table.mint("agent-1")
    with gateway(table, tmp_path / "journal.jsonl") as server:
        response = ask(server, token, {"model": "m", "messages": []})
        health = httpx.get(server.base_url + "/health")

    assert (response.status_code, response.json()["error"]["type"], health.status_code) == (
        502,
        "api_error",
        200,
    )


def test_a_stop_of_the_run_ends_the_forwarded_request_and_closes_the_provider_connection(tmp_path):
    """A rollout that stops must not leave the provider generating for nobody."""
    control = RunControl()
    with provider(Reply(hang=True)) as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl", request_policy=Stoppable(control)) as server:
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(ask, server, token, {"model": "m", "messages": []})
                wait_until(lambda: upstream.requests, what="the forwarded request")

                control.request_stop("rollout deadline", stop_reason="timeout")

                assert pending.result(timeout=5).status_code == 502
        assert upstream.disconnected.wait(5), "the provider connection outlived the stop"


def test_the_environment_of_a_token_points_an_openai_client_at_the_gateway(tmp_path):
    """An OpenAI client appends ``/chat/completions`` to ``OPENAI_BASE_URL``."""
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            env = server.env_for(token)
            response = httpx.post(
                env["OPENAI_BASE_URL"] + "/chat/completions",
                json={"model": "m", "messages": []},
                headers={"Authorization": "Bearer " + env["OPENAI_API_KEY"]},
            )

    assert (response.status_code, len(upstream.requests)) == (200, 1)


# --- metering ---------------------------------------------------------------------------------


def test_each_relayed_request_is_journaled_with_the_usage_that_the_provider_reported(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1")
        with gateway(table, journal_path) as server:
            ask(server, token, {"model": "agent-model", "messages": []})
            wait_until(lambda: gateway_records(journal_path), what="the gateway record")

    assert gateway_records(journal_path) == [
        {
            "type": "gateway.request",
            "agent_id": "agent-1",
            "status": "ok",
            "requested_model": "agent-model",
            "upstream_model": "provider-model",
            "base_url": upstream.url,
            "usage": {
                "input_tokens": 11,
                "output_tokens": 7,
                "cached_input_tokens": 3,
                "cache_creation_tokens": 0,
                "reasoning_output_tokens": 2,
                "cost_usd": 0.0,
            },
            "spent_usd": 0.0,
        }
    ]


def test_a_provider_error_is_journaled_with_its_detail_and_without_the_route_key(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    refusal = {
        "error": {
            "code": "invalid_encrypted_content",
            "type": "invalid_request_error",
            "param": "messages",
            "message": "bad item in a request with Bearer route-secret, key route-secret",
        }
    }
    with provider(Reply(status=400, body=refusal)) as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url, api_key="route-secret")})
        token = table.mint("agent-1")
        with gateway(table, journal_path) as server:
            ask(server, token, {"model": "m", "messages": []})
            wait_until(lambda: gateway_records(journal_path), what="the gateway record")

    assert [(record["status"], record["error"]) for record in gateway_records(journal_path)] == [
        (
            "upstream_400",
            {
                "code": "invalid_encrypted_content",
                "type": "invalid_request_error",
                "param": "messages",
                "message": "bad item in a request with [redacted], key [redacted]",
            },
        )
    ]


def test_a_priced_answer_is_charged_to_the_slot_that_asked(tmp_path):
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url, pricing=ONE_DOLLAR_A_TOKEN)})
        token, other = table.mint("agent-1"), table.mint("agent-2")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            ask(server, token, {"model": "m", "messages": []})
            wait_until(lambda: token.spent_usd > 0, what="the charge")

    assert (token.spent_usd, other.spent_usd) == (pytest.approx(8.0), 0.0)


@pytest.mark.parametrize("field_name", ["cost", "cost_usd"])
def test_a_cost_that_the_provider_reports_is_charged_instead_of_the_route_price(tmp_path, field_name):
    """OpenRouter reports the amount that it charged, and that amount is authoritative."""
    body = {**COMPLETION, "usage": {**COMPLETION["usage"], field_name: 0.5}}
    with provider(Reply(body=body)) as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url, pricing=ONE_DOLLAR_A_TOKEN)})
        token = table.mint("agent-1")
        with gateway(table, tmp_path / "journal.jsonl") as server:
            ask(server, token, {"model": "m", "messages": []})
            wait_until(lambda: token.spent_usd > 0, what="the charge")

    assert token.spent_usd == pytest.approx(0.5)


def test_a_slot_that_spent_its_budget_gets_a_terminal_400_and_is_not_forwarded(tmp_path):
    journal_path = tmp_path / "journal.jsonl"
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url, pricing=ONE_DOLLAR_A_TOKEN)})
        token = table.mint("agent-1", budget_usd=8.0)
        with gateway(table, journal_path) as server:
            ask(server, token, {"model": "m", "messages": []})
            wait_until(lambda: token.spent_usd > 0, what="the charge")

            refused = ask(server, token, {"model": "m", "messages": []})

    assert (refused.status_code, refused.json(), len(upstream.requests)) == (
        400,
        error(
            "invalid_request_error",
            "run budget exhausted: spent $8.0000 of $8.0000 -- this will not succeed on retry",
        ),
        1,
    )
    assert gateway_records(journal_path, "budget_exceeded") == [
        {
            "type": "gateway.request",
            "agent_id": "agent-1",
            "status": "budget_exceeded",
            "spent_usd": 8.0,
            "budget_usd": 8.0,
        }
    ]


def test_a_budget_on_a_route_without_prices_is_reported_once(tmp_path):
    """Such a budget never binds, because the spend stays 0."""
    journal_path = tmp_path / "journal.jsonl"
    with provider() as upstream:
        table = RoutingTable({"default": ModelRoute(base_url=upstream.url)})
        token = table.mint("agent-1", budget_usd=1.0)
        with gateway(table, journal_path) as server:
            ask(server, token, {"model": "m", "messages": []})
            ask(server, token, {"model": "m", "messages": []})
            wait_until(lambda: len(gateway_records(journal_path, "ok")) == 2, what="both gateway records")

    assert gateway_records(journal_path, "budget_unenforceable") == [
        {
            "type": "gateway.request",
            "status": "budget_unenforceable",
            "agent_id": "agent-1",
            "base_url": upstream.url,
            "reason": "route has no pricing and the provider reported no cost; budget_usd cannot bind",
        }
    ]

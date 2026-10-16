"""Gateway tests against a fake Chat Completions upstream.

A stub server stands in for the provider so the whole path -- auth, routing,
model rewrite, header forwarding, the streaming refusal, usage tap, budget
refusal -- is covered without credentials or network.
"""

from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

httpx = pytest.importorskip("httpx")

from sprout.harness_core.events import Usage
from sprout.harness_core.journal import JournalWriter, read_journal
from sprout.harness_gateway.routing import ModelRoute, RoutingTable
from sprout.harness_gateway.server import GATEWAY_EVENT, GatewayServer

RESPONSE = {
    "id": "chat-1",
    "object": "chat.completion",
    "model": "upstream-model",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "pong"},
                 "finish_reason": "stop", "provider_extra": {"kept": True}}],
    "usage": {"prompt_tokens": 11, "completion_tokens": 7,
              "prompt_tokens_details": {"cached_tokens": 3},
              "completion_tokens_details": {"reasoning_tokens": 2}},
}


class _Upstream:
    """Records what it received so forwarding can be asserted."""

    def __init__(self):
        self.requests = []
        self._httpd = ThreadingHTTPServer(("127.0.0.1", 0), _make_upstream_handler(self))
        self._httpd.daemon_threads = True
        threading.Thread(target=self._httpd.serve_forever, daemon=True).start()

    @property
    def base_url(self):
        return "http://127.0.0.1:%d" % self._httpd.server_address[1]

    def stop(self):
        self._httpd.shutdown()
        self._httpd.server_close()


def _make_upstream_handler(state):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, *args):
            pass

        def do_POST(self):
            length = int(self.headers.get("Content-Length") or 0)
            body = json.loads(self.rfile.read(length) or b"{}")
            state.requests.append({"path": self.path, "headers": dict(self.headers), "body": body})

            payload = json.dumps(RESPONSE).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.send_header("x-request-id", "req_abc")
            self.end_headers()
            self.wfile.write(payload)

    return Handler


@pytest.fixture
def upstream():
    server = _Upstream()
    yield server
    server.stop()


@pytest.fixture
def gateway(upstream, tmp_path):
    table = RoutingTable(
        {
            "default": ModelRoute(base_url=upstream.base_url, api_key="real-provider-key"),
            "sprout-rl-ckpt-42": ModelRoute(
                base_url=upstream.base_url,
                api_key="rl-key",
                upstream_model="checkpoint-42",
            ),
            "dead-model": ModelRoute(base_url="http://127.0.0.1:1", api_key="x"),
        }
    )
    journal = JournalWriter(tmp_path / "gw.jsonl", run_id="gwrun")
    server = GatewayServer(table, journal=journal, port=0)
    server.start()
    yield server, table, journal, tmp_path / "gw.jsonl"
    server.stop()
    journal.close()


def post(server, token, body, **headers):
    return httpx.post(
        server.base_url + "/v1/chat/completions",
        json=body,
        headers={"Authorization": "Bearer " + token, **headers},
        timeout=30,
    )


# --- auth ------------------------------------------------------------------
@pytest.mark.parametrize("headers", [
    pytest.param({"Authorization": "Bearer not-a-real-token"}, id="unknown-token"),
    pytest.param({}, id="no-token"),
])
def test_requests_outside_the_contract_are_refused_before_forwarding(gateway, upstream, headers):
    """There is no unauthenticated mode: every request names its slot, and a
    token the table never minted is as good as none."""
    server, _table, _journal, _path = gateway
    response = httpx.post(server.base_url + "/v1/chat/completions",
                          json={"model": "m", "messages": []}, headers=headers, timeout=10)
    assert response.status_code == 401
    assert response.json()["error"]["type"] == "authentication_error"
    assert upstream.requests == []


def test_other_paths_are_not_served(gateway):
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    response = httpx.post(server.base_url + "/v1/messages", json={},
                          headers={"Authorization": "Bearer " + token.token}, timeout=10)
    assert response.status_code == 404


# --- routing / model swap --------------------------------------------------
@pytest.mark.parametrize("requested, forwarded, key", [
    pytest.param("sprout-rl-ckpt-42", "checkpoint-42", "rl-key", id="named-route"),
    pytest.param("some-unmapped-model", "some-unmapped-model", "real-provider-key", id="default-route"),
])
def test_model_swap_rewrites_name_and_targets_route(gateway, upstream, requested, forwarded, key):
    """The whole reason the gateway exists: redirect a model without touching
    the agent. The agent presents its slot token and the gateway swaps in the
    route's real key; a name no route maps goes to the default route, unrenamed."""
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    response = post(server, token.token, {"model": requested, "messages": []})
    assert response.status_code == 200

    sent = upstream.requests[-1]
    assert sent["path"] == "/v1/chat/completions"
    assert sent["body"]["model"] == forwarded                          # rewritten, or left alone
    assert sent["headers"]["Authorization"] == "Bearer " + key         # route's own credential
    assert token.token not in json.dumps(sent["headers"])              # the slot token stays here


def test_without_a_default_route_an_unknown_model_is_refused(upstream):
    table = RoutingTable({"known": ModelRoute(base_url=upstream.base_url)})
    with GatewayServer(table, port=0) as server:
        token = table.mint("agent-1")
        response = post(server, token.token, {"model": "unknown", "messages": []})
    assert response.status_code == 404 and "no default route" in response.json()["error"]["message"]
    assert upstream.requests == []


# --- header and body forwarding ---------------------------------------------
def test_the_relay_is_transparent_in_both_directions(gateway, upstream):
    """The agent's own headers reach the provider; the provider's body and
    headers come back as sent. Unknown fields must survive untouched."""
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    response = post(server, token.token, {"model": "m", "messages": []}, **{"x-session-id": "sess-9"})
    sent = {k.lower(): v for k, v in upstream.requests[-1]["headers"].items()}
    assert sent["x-session-id"] == "sess-9"
    assert response.json() == RESPONSE
    assert response.headers["x-request-id"] == "req_abc"


def test_streaming_is_refused_before_forwarding(gateway, upstream):
    """A streamed response would be relayed before its usage could be metered,
    so the gateway does not offer it -- and says so with a terminal 400, not a
    429 that would invite the SDK to retry the unwinnable."""
    server, table, _journal, _path = gateway
    token = table.mint("agent-1")
    response = post(server, token.token, {"model": "m", "messages": [], "stream": True})
    assert response.status_code == 400
    assert response.json()["error"]["type"] == "invalid_request_error"
    assert "stream" in response.json()["error"]["message"]
    assert upstream.requests == []


# --- usage tap + budget ----------------------------------------------------
def test_usage_is_tapped_into_the_journal(gateway):
    server, table, _journal, path = gateway
    token = table.mint("agent-1")
    post(server, token.token, {"model": "m", "messages": []})

    events = [r for r in read_journal(path) if r["type"] == GATEWAY_EVENT]
    assert len(events) == 1
    tap = events[0]
    assert tap["agent_id"] == "agent-1"
    assert tap["status"] == "ok"
    assert tap["usage"]["input_tokens"] == 11
    assert tap["usage"]["output_tokens"] == 7
    assert tap["usage"]["cached_input_tokens"] == 3
    assert tap["usage"]["reasoning_output_tokens"] == 2


def test_upstream_failure_becomes_502_not_a_crash(gateway):
    server, table, _journal, path = gateway
    token = table.mint("agent-1")
    response = post(server, token.token, {"model": "dead-model", "messages": []})
    assert response.status_code == 502
    assert response.json()["error"]["type"] == "api_error"
    # the gateway is still serving
    assert httpx.get(server.base_url + "/health", timeout=10).status_code == 200


# --- routing table ---------------------------------------------------------
def test_api_key_env_is_resolved_lazily(monkeypatch):
    route = ModelRoute(base_url="http://up", api_key_env="SPROUT_TEST_KEY")
    monkeypatch.delenv("SPROUT_TEST_KEY", raising=False)
    assert route.resolve_key() is None
    monkeypatch.setenv("SPROUT_TEST_KEY", "later-value")
    assert route.resolve_key() == "later-value"


def test_upstream_error_detail_redacts_the_bearer_key():
    from sprout.harness_gateway.server import _upstream_error

    request = httpx.Request("POST", "https://example.com",
                            headers={"Authorization": "Bearer provider-secret"})
    response = httpx.Response(400, request=request, json={"error": {
        "code": "bad_input", "param": "input",
        "message": "problem with provider-secret (sent as Bearer provider-secret)"}})
    detail = _upstream_error(response)
    assert detail["code"] == "bad_input" and detail["param"] == "input"
    assert "provider-secret" not in json.dumps(detail)
    assert "[redacted]" in detail["message"]


def test_provider_error_is_recorded_with_the_failed_gateway_request(gateway, monkeypatch):
    server, table, _journal, path = gateway

    def failed(url, **kwargs):
        return httpx.Response(400, request=httpx.Request("POST", url, headers=kwargs["headers"]),
                              json={"error": {"code": "invalid_encrypted_content",
                                              "message": "The reasoning item is invalid"}})

    monkeypatch.setattr("sprout.harness_core.control.post", failed)
    token = table.mint("actor")
    assert post(server, token.token, {"model": "m", "messages": []}).status_code == 400
    for _ in range(100):
        events = [r for r in read_journal(path) if r["type"] == GATEWAY_EVENT]
        if events:
            break
        time.sleep(.01)
    assert events[-1]["status"] == "upstream_400"
    assert events[-1]["error"]["code"] == "invalid_encrypted_content"


def test_tokens_are_unguessable_and_distinct():
    table = RoutingTable()
    a = table.mint("agent-a")
    b = table.mint("agent-b")
    assert a.token != b.token
    assert len(a.token) > 24
    assert table.lookup(a.token).agent_id == "agent-a"
    assert table.lookup("Bearer " + b.token).agent_id == "agent-b"
    assert table.lookup(None) is None


# --- pricing: what makes budget_usd real -------------------------------------
def test_a_route_prices_usage_and_uncached_input_is_not_double_billed():
    """The provider reports token counts but no price; without a conversion the
    budget never binds. Cache reads bill at their own rate and are a SUBSET of
    input tokens, so they must come out of the input bucket."""
    route = ModelRoute(base_url="http://up", pricing={"input": 3.0, "output": 15.0, "cache_read": 0.30})
    usage = Usage(input_tokens=1000, output_tokens=100, cached_input_tokens=400)
    # 600 uncached * 3 + 100 * 15 + 400 * 0.30 (per mtok)
    assert abs(route.price(usage) - 0.00342) < 1e-9
    assert ModelRoute(base_url="http://up").price(usage) == 0.0, "no pricing -> no invented cost"


def settled(token, spent_usd, timeout_s=5.0):
    """The usage tap runs after the response bytes are on the wire, so a client
    can read ``spent_usd`` a moment before the handler thread has charged it."""
    deadline = time.monotonic() + timeout_s
    while token.spent_usd != pytest.approx(spent_usd) and time.monotonic() < deadline:
        time.sleep(0.005)
    return token.spent_usd


def test_a_priced_response_is_charged_to_its_slot_so_the_budget_binds(upstream, tmp_path):
    """``spent_usd`` rises only through the usage the gateway taps from each
    response, priced by the route; the refusal reads that same number, so a
    tap that stopped charging would leave every budget unenforced. A slot-side
    accountant can only ask an agent to stop; this makes it."""
    # $1 per uncached input token: RESPONSE has 11 prompt tokens, 3 of them cached.
    table = RoutingTable({"default": ModelRoute(base_url=upstream.base_url,
                                                pricing={"input": 1_000_000.0})})
    journal_path = tmp_path / "j.jsonl"
    with JournalWriter(journal_path, run_id="r") as journal:
        with GatewayServer(table, journal=journal, port=0) as server:
            token = table.mint("agent-1", budget_usd=10.0)
            assert post(server, token.token, {"model": "m", "messages": []}).status_code == 200
            assert settled(token, 8.0) == pytest.approx(8.0)
            assert post(server, token.token, {"model": "m", "messages": []}).status_code == 200
            assert settled(token, 16.0) == pytest.approx(16.0)
            refused = post(server, token.token, {"model": "m", "messages": []})
    # 400, not 429: an exhausted budget is terminal, and 429 invites the SDK to
    # back off and retry a request that can never succeed (measured live).
    assert refused.status_code == 400
    assert refused.json()["error"]["type"] == "invalid_request_error"
    assert "budget exhausted" in refused.json()["error"]["message"]
    assert len(upstream.requests) == 2, "the third request never reached the provider"
    # Journaled before the 400 goes out. Selected by status, not as the last
    # record: the second response's "ok" tap is written after the charge that
    # ``settled`` waits on, so it may land after this one.
    refusals = [r for r in read_journal(journal_path)
                if r["type"] == GATEWAY_EVENT and r["status"] == "budget_exceeded"]
    assert len(refusals) == 1 and refusals[0]["spent_usd"] == pytest.approx(16.0)


def test_an_unpriced_budget_says_so_once(upstream, tmp_path):
    """A budget over a route with no pricing can never bind -- spent stays 0
    forever. The gateway must say so, once, rather than silently enforcing
    nothing."""
    table = RoutingTable({"default": ModelRoute(base_url=upstream.base_url, api_key="k")})
    journal_path = tmp_path / "j.jsonl"
    with JournalWriter(journal_path, run_id="r", agent_id="a") as journal:
        with GatewayServer(table, journal=journal, port=0) as server:
            token = table.mint("agent-1", run_id="r", budget_usd=1.0)
            for _ in range(2):       # twice: the warning must not repeat
                assert post(server, token.token, {"model": "m", "messages": []}).status_code == 200
    warnings = [r for r in read_journal(journal_path) if r.get("status") == "budget_unenforceable"]
    assert len(warnings) == 1, "say it once; per-request is noise"

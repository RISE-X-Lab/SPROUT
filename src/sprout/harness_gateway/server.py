"""The gateway HTTP server: OpenAI Chat Completions, which mini-swe-agent speaks.

Requests keep their shape except for credentials, the optional model rename and
the run's request policy (sampling, turn and deadline limits).

Implementation notes that are not obvious:

- **Headers pass through untouched** apart from hop-by-hop and auth, and response
  bytes are relayed as received.
- **Streaming is refused** with a non-retryable 400. mini-swe-agent never asks
  for it, and a streamed response would be relayed before its usage could be
  metered -- a budget that cannot see the spend cannot bind.
- **Budget is checked before forwarding**, refused with a NON-RETRYABLE 400 and
  a JSON error the agent can read. Enforcement here is real, unlike asking an
  agent to stop; and it must be terminal -- a 429 reads as "retry later" to
  every SDK, and an agent then burns wall-clock retrying the unwinnable.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Optional, Tuple

from sprout.harness_core.events import Usage
from sprout.harness_gateway.routing import RoutingTable

CHAT_PATH = "/v1/chat/completions"

#: Never forwarded upstream (hop-by-hop or replaced by us).
_STRIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-authorization",
    "authorization",
    "accept-encoding",  # we relay bytes; let httpx negotiate identity
}

_STRIP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "connection",
    "keep-alive",
    "transfer-encoding",
}

GATEWAY_EVENT = "gateway.request"   # wire-level tap event type


class GatewayServer:
    """Owns the routing table, the journal tap and the HTTP server."""

    def __init__(
        self,
        table: RoutingTable,
        *,
        journal=None,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout_s: float = 900.0,
        request_policy=None,
    ) -> None:
        self.table = table
        self.journal = journal
        self.timeout_s = timeout_s
        self.request_policy = request_policy
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self._httpd.daemon_threads = True
        self._thread: Optional[threading.Thread] = None
        self._warned_unpriced = False

    # --- lifecycle ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        host, port = self._httpd.server_address[:2]
        return "http://%s:%s" % (host, port)

    def start(self) -> "GatewayServer":
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=10)

    def __enter__(self) -> "GatewayServer":
        return self.start()

    def __exit__(self, *exc) -> None:
        self.stop()

    def env_for(self, token) -> dict:
        """Environment an agent needs to talk to this gateway."""
        return {"OPENAI_BASE_URL": self.base_url + "/v1", "OPENAI_API_KEY": token.token}

    # --- tap ---------------------------------------------------------------
    def record(self, **payload) -> None:
        if self.journal is not None:
            self.journal.emit(GATEWAY_EVENT, **payload)


def _make_handler(gateway: GatewayServer):
    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        server_version = "sprout-gateway/1"

        # --- plumbing ---------------------------------------------------
        def log_message(self, *args) -> None:  # noqa: A003 - silence stderr spam
            pass

        def _json(self, status: int, payload: dict) -> None:
            body = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def _error(self, status: int, message: str, kind: str = "invalid_request_error") -> None:
            self._json(status, {"error": {"type": kind, "message": message}})

        # --- routes -----------------------------------------------------
        def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler API
            if self.path in ("/health", "/_sprout/health"):
                self._json(200, {"status": "ok"})
                return
            self._error(404, "unknown path %s" % self.path)

        def do_POST(self) -> None:  # noqa: N802
            if self.path != CHAT_PATH:
                self._error(404, "unknown path %s" % self.path)
                return

            token = gateway.table.lookup(self.headers.get("Authorization"))
            if token is None:
                self._error(401, "unknown or missing slot token", "authentication_error")
                return

            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length) if length else b""
            try:
                payload = json.loads(raw or b"{}")
            except ValueError:
                self._error(400, "body is not valid JSON")
                return
            if payload.get("stream"):
                # Terminal, like the budget refusal below: a streamed response
                # would be relayed before its usage could be metered, so the
                # gateway does not offer it, and a retry cannot change that.
                self._error(400, "this gateway does not relay streamed responses; "
                                 "send stream=false", "invalid_request_error")
                return

            if token.over_budget():
                gateway.record(
                    agent_id=token.agent_id,
                    status="budget_exceeded",
                    spent_usd=round(token.spent_usd, 6),
                    budget_usd=token.budget_usd,
                )
                # 400/invalid_request, NOT 429/rate_limit: an exhausted budget
                # is terminal, and 429 tells every SDK "back off and retry" --
                # measured live, the agent then burned minutes retrying a
                # request that could never succeed. A non-retryable error fails
                # the run NOW, which is what a kill switch is for.
                self._error(
                    400,
                    "run budget exhausted: spent $%.4f of $%.4f -- this will "
                    "not succeed on retry"
                    % (token.spent_usd, token.budget_usd),
                    "invalid_request_error",
                )
                return

            requested_model = payload.get("model")
            try:
                route = gateway.table.route_for(requested_model)
            except KeyError as error:
                self._error(404, str(error.args[0]))
                return
            if route.upstream_model:
                payload["model"] = route.upstream_model
            if gateway.request_policy is not None:
                try:
                    payload = gateway.request_policy.prepare_model_request(payload)
                except ValueError as error:
                    self._error(400, str(error), "invalid_request_error")
                    return
            body = json.dumps(payload).encode()

            try:
                self._forward(route, body, token, requested_model)
            except Exception as exc:  # noqa: BLE001 - report, never crash the server
                gateway.record(
                    agent_id=token.agent_id,
                    status="upstream_error",
                    model=requested_model,
                    error="%s: %s" % (type(exc).__name__, exc),
                )
                try:
                    self._error(502, "upstream failure: %s" % exc, "api_error")
                except Exception:  # pragma: no cover - client already gone
                    pass

        # --- forwarding -------------------------------------------------
        def _upstream_headers(self, route) -> dict:
            headers = {
                key: value
                for key, value in self.headers.items()
                if key.lower() not in _STRIP_REQUEST_HEADERS
            }
            key = route.resolve_key()
            if key:
                headers["Authorization"] = "Bearer %s" % key
            headers["Content-Type"] = "application/json"
            return headers

        def _forward(self, route, body: bytes, token, requested_model) -> None:
            # mini's request must release inference on rollout stop, even when
            # the provider sends no bytes.
            from sprout.harness_core.control import post

            url = route.base_url.rstrip("/") + CHAT_PATH
            upstream = post(url, content=body, headers=self._upstream_headers(route),
                            timeout_s=gateway.timeout_s,
                            control=getattr(gateway.request_policy, "control", None))
            content = upstream.content
            self.send_response(upstream.status_code)
            for key, value in upstream.headers.items():
                if key.lower() in _STRIP_RESPONSE_HEADERS:
                    continue
                self.send_header(key, value)
            self.send_header("Content-Length", str(len(content)))
            self.end_headers()
            self.wfile.write(content)
            usage, model = _usage_from_response(upstream)
            self._tap(token, requested_model, model, route, upstream.status_code, usage,
                      _upstream_error(upstream) if upstream.status_code >= 400 else None)

        def _tap(self, token, requested_model, upstream_model, route, status, usage, error=None):
            cost = usage.cost_usd or route.price(usage)
            gateway.table.charge(token, cost)
            if (token.budget_usd is not None and cost == 0.0
                    and (usage.input_tokens or usage.output_tokens)
                    and not gateway._warned_unpriced):
                # A budget over a route with no pricing can never bind: the
                # provider reports tokens, nobody converts them to dollars,
                # spent stays 0 forever. Once per gateway, because once is a
                # diagnosis and per-request is noise.
                gateway._warned_unpriced = True
                gateway.record(
                    status="budget_unenforceable",
                    agent_id=token.agent_id,
                    base_url=route.base_url,
                    reason="route has no pricing and the provider reported no cost; "
                           "budget_usd cannot bind",
                )
            gateway.record(
                agent_id=token.agent_id,
                status="ok" if status < 400 else "upstream_%d" % status,
                requested_model=requested_model,
                upstream_model=upstream_model or route.upstream_model or requested_model,
                base_url=route.base_url,
                usage=dict(usage.as_dict(), cost_usd=round(cost, 6)),
                spent_usd=round(token.spent_usd, 6),
                **({"error": error} if error is not None else {}),
            )

    return Handler


def _upstream_error(response) -> dict:
    """Enough provider detail to diagnose a failed request without logging auth."""
    try:
        payload = response.json()
    except ValueError:
        return {"message": "Non-JSON upstream error"}
    if not isinstance(payload, dict):
        return {"message": "Unstructured upstream error"}
    value = payload.get("error", payload)
    if not isinstance(value, dict):
        value = {"message": str(value)}
    authorization = response.request.headers.get("authorization", "")
    secrets = [authorization]
    if authorization.lower().startswith("bearer "):
        secrets.append(authorization[7:])
    result = {}
    for key in ("code", "type", "param", "message"):
        if value.get(key) is None:
            continue
        text = str(value[key])
        for secret in secrets:
            if secret:
                text = text.replace(secret, "[redacted]")
        result[key] = text[:2000 if key == "message" else 200]
    return result


# --- usage extraction ------------------------------------------------------
def _usage_from_response(response) -> Tuple[Usage, Optional[str]]:
    usage = Usage()
    model = None
    try:
        payload = response.json()
    except Exception:  # noqa: BLE001 - non-JSON error body
        return usage, model
    if isinstance(payload, dict):
        model = payload.get("model")
        _absorb_usage(payload.get("usage"), usage)
    return usage, model


def _absorb_usage(native, usage: Usage) -> None:
    """Fold a Chat Completions usage block into ours.

    prompt_tokens / completion_tokens, with the cache and reasoning splits nested
    under *_tokens_details.
    """
    if not isinstance(native, dict):
        return
    usage.input_tokens += int(native.get("prompt_tokens") or 0)
    usage.output_tokens += int(native.get("completion_tokens") or 0)
    details = native.get("prompt_tokens_details")
    if isinstance(details, dict):
        usage.cached_input_tokens += int(details.get("cached_tokens") or 0)
    details = native.get("completion_tokens_details")
    if isinstance(details, dict):
        usage.reasoning_output_tokens += int(details.get("reasoning_tokens") or 0)
    # OpenRouter includes the charged amount in its native usage block. Keep
    # it authoritative instead of attempting to duplicate provider pricing.
    usage.cost_usd += float(native.get("cost") or native.get("cost_usd") or 0)

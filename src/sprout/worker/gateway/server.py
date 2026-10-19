"""The gateway's HTTP server, which serves OpenAI Chat Completions to an agent.

``POST /v1/chat/completions`` forwards a request to the route of its model. The
gateway changes only the credential, the model name of the route, and what the
request policy changes (sampling, turn and deadline limits). It relays the
upstream status, the end-to-end headers and the decoded body. ``GET /health``
answers ``{"status": "ok"}``.

An unknown or missing slot token gets a 401, an unknown path or model a 404, and
an upstream failure a 502. These requests get a 400 ``invalid_request_error``,
because no retry can succeed: a streamed request, a request from a slot over its
budget, and a request that the policy refuses. A 429 would make the agent's SDK
retry until the run's deadline. The gateway cannot meter a streamed response
before it relays it, so it does not stream.
"""

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import TYPE_CHECKING, Any, ClassVar, Protocol, cast

from sprout.worker.gateway.routing import ModelRoute, RoutingTable, SlotToken
from sprout.worker.run.events import Usage

if TYPE_CHECKING:
    import httpx

    from sprout.worker.run.control import RunControl
    from sprout.worker.run.journal import JournalWriter

CHAT_PATH = "/v1/chat/completions"

#: Request headers that the gateway does not forward: hop-by-hop ones and those that it replaces.
_STRIP_REQUEST_HEADERS = {
    "host",
    "content-length",
    "connection",
    "keep-alive",
    "transfer-encoding",
    "upgrade",
    "proxy-authorization",
    "authorization",
    "accept-encoding",  # httpx asks for its own encodings and decodes the body
}

#: Response headers that the gateway does not relay. It sends the decoded body with its own length.
_STRIP_RESPONSE_HEADERS = {
    "content-length",
    "content-encoding",
    "connection",
    "keep-alive",
    "transfer-encoding",
}

GATEWAY_EVENT = "gateway.request"  # wire-level tap event type


class RequestPolicy(Protocol):
    """What a run applies to each model request before the gateway forwards it.

    ``prepare_model_request`` returns the payload to send, or raises ValueError to refuse
    it. When ``control`` stops the run, the gateway cancels the upstream request in flight.
    """

    @property
    def control(self) -> RunControl | None: ...

    def prepare_model_request(self, payload: dict[str, Any], /) -> dict[str, Any]: ...


class GatewayServer:
    """An HTTP gateway on ``host`` and ``port`` (0: a free one) for the routes of ``table``.

    ``journal`` gets a ``gateway.request`` event for each request. ``timeout_s`` bounds
    each upstream request. ``request_policy`` can change or refuse each request.
    """

    def __init__(
        self,
        table: RoutingTable,
        *,
        journal: JournalWriter | None = None,
        host: str = "127.0.0.1",
        port: int = 0,
        timeout_s: float = 900.0,
        request_policy: RequestPolicy | None = None,
    ) -> None:
        self.table = table
        self.journal = journal
        self.timeout_s = timeout_s
        self.request_policy = request_policy
        self._httpd = ThreadingHTTPServer((host, port), _make_handler(self))
        self._httpd.daemon_threads = True
        self._thread: threading.Thread | None = None
        self._warned_unpriced = False

    # --- lifecycle ---------------------------------------------------------
    @property
    def base_url(self) -> str:
        """The URL of the gateway, without a path."""
        # getsockname() of an AF_INET socket gives the host as a str.
        host = cast(str, self._httpd.server_address[0])
        return "http://%s:%s" % (host, self._httpd.server_address[1])

    def start(self) -> GatewayServer:
        """Serve requests on a daemon thread, and return the gateway."""
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        """Stop serving and close the socket. Waits up to 10 s for the serving thread."""
        self._httpd.shutdown()
        self._httpd.server_close()
        if self._thread:
            self._thread.join(timeout=10)

    def __enter__(self) -> GatewayServer:
        return self.start()

    def __exit__(self, *exc: object) -> None:
        self.stop()

    def env_for(self, token: SlotToken) -> dict[str, str]:
        """The environment variables that point an agent at this gateway with ``token``."""
        return {"OPENAI_BASE_URL": self.base_url + "/v1", "OPENAI_API_KEY": token.token}

    # --- tap ---------------------------------------------------------------
    def record(self, **payload: object) -> None:
        """Write one ``gateway.request`` event with ``payload`` to the journal, if there is one."""
        if self.journal is not None:
            self.journal.emit(GATEWAY_EVENT, **payload)


def _make_handler(gateway: GatewayServer) -> type[BaseHTTPRequestHandler]:
    """A request handler class that serves the routes of ``gateway``."""

    class Handler(_GatewayHandler):
        _gateway = gateway

    return Handler


class _GatewayHandler(BaseHTTPRequestHandler):
    """The routes of a gateway. The subclass from ``_make_handler`` sets ``_gateway``."""

    protocol_version = "HTTP/1.1"
    server_version = "sprout-gateway/1"
    _gateway: ClassVar[GatewayServer]

    # --- plumbing ---------------------------------------------------
    def log_message(self, *args: Any) -> None:
        # The base class writes one stderr line for each request.
        pass

    def _json(self, status: int, payload: dict[str, Any]) -> None:
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

        token = self._gateway.table.lookup(self.headers.get("Authorization"))
        if token is None:
            self._error(401, "unknown or missing slot token", "authentication_error")
            return

        payload = self._read_payload()
        if payload is None or self._refuse_over_budget(token):
            return

        requested_model = payload.get("model")
        prepared = self._prepare_upstream_request(payload)
        if prepared is None:
            return
        route, body = prepared
        self._forward_or_report(route=route, body=body, token=token, requested_model=requested_model)

    # --- admission --------------------------------------------------
    def _read_payload(self) -> dict[str, Any] | None:
        """The JSON body of the request, or None after the gateway answers a 400."""
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length) if length else b""
        try:
            payload = json.loads(raw or b"{}")
        except ValueError:
            self._error(400, "body is not valid JSON")
            return None
        if payload.get("stream"):
            # A retry cannot change this, so the 400 is terminal, like the budget refusal.
            self._error(
                400,
                "this gateway does not relay streamed responses; send stream=false",
                "invalid_request_error",
            )
            return None
        return cast(dict[str, Any], payload)

    def _refuse_over_budget(self, token: SlotToken) -> bool:
        """Whether the slot of ``token`` is over its budget. If it is, the gateway answers a 400."""
        if not token.over_budget():
            return False
        self._gateway.record(
            agent_id=token.agent_id,
            status="budget_exceeded",
            spent_usd=round(token.spent_usd, 6),
            budget_usd=token.budget_usd,
        )
        # A 400, not a 429: each SDK retries a 429, and the agent would spend minutes on a
        # request that cannot succeed. A 400 ends the run now.
        self._error(
            400,
            "run budget exhausted: spent $%.4f of $%.4f -- this will "
            "not succeed on retry" % (token.spent_usd, token.budget_usd or 0.0),
            "invalid_request_error",
        )
        return True

    def _prepare_upstream_request(self, payload: dict[str, Any]) -> tuple[ModelRoute, bytes] | None:
        """The route and the body to forward, or None after the gateway answers an error."""
        try:
            route = self._gateway.table.route_for(payload.get("model"))
        except KeyError as error:
            self._error(404, str(error.args[0]))
            return None
        if route.upstream_model:
            payload["model"] = route.upstream_model
        if self._gateway.request_policy is not None:
            try:
                payload = self._gateway.request_policy.prepare_model_request(payload)
            except ValueError as error:
                self._error(400, str(error), "invalid_request_error")
                return None
        return route, json.dumps(payload).encode()

    # --- forwarding -------------------------------------------------
    def _forward_or_report(
        self, *, route: ModelRoute, body: bytes, token: SlotToken, requested_model: object
    ) -> None:
        """Forward ``body`` on ``route``. An upstream failure gets a tap event and a 502."""
        try:
            self._forward(route, body, token, requested_model)
        except Exception as exc:  # noqa: BLE001 - report, never crash the server
            self._gateway.record(
                agent_id=token.agent_id,
                status="upstream_error",
                model=requested_model,
                error="%s: %s" % (type(exc).__name__, exc),
            )
            try:
                self._error(502, "upstream failure: %s" % exc, "api_error")
            except Exception:  # pragma: no cover - client already gone
                pass

    def _upstream_headers(self, route: ModelRoute) -> dict[str, str]:
        headers = {
            key: value for key, value in self.headers.items() if key.lower() not in _STRIP_REQUEST_HEADERS
        }
        key = route.resolve_key()
        if key:
            headers["Authorization"] = "Bearer %s" % key
        headers["Content-Type"] = "application/json"
        return headers

    def _forward(self, route: ModelRoute, body: bytes, token: SlotToken, requested_model: object) -> None:
        from sprout.worker.run.control import post

        url = route.base_url.rstrip("/") + CHAT_PATH
        policy = self._gateway.request_policy
        # With the run's control, a stop cancels the upstream request and releases the
        # provider's inference, also while the provider sends no bytes.
        upstream = post(
            url,
            content=body,
            headers=self._upstream_headers(route),
            timeout_s=self._gateway.timeout_s,
            control=policy.control if policy is not None else None,
        )
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
        self._tap(
            token,
            requested_model,
            model,
            route,
            upstream.status_code,
            usage,
            _upstream_error(upstream) if upstream.status_code >= 400 else None,
        )

    def _tap(
        self,
        token: SlotToken,
        requested_model: object,
        upstream_model: object,
        route: ModelRoute,
        status: int,
        usage: Usage,
        error: dict[str, str] | None = None,
    ) -> None:
        gateway = self._gateway
        cost = usage.cost_usd or route.price(usage)
        gateway.table.charge(token, cost)
        if (
            token.budget_usd is not None
            and cost == 0.0
            and (usage.input_tokens or usage.output_tokens)
            and not gateway._warned_unpriced
        ):
            # A budget on a route without pricing never binds, because the spend stays 0.
            # One warning for each gateway is enough to diagnose it.
            gateway._warned_unpriced = True
            gateway.record(
                status="budget_unenforceable",
                agent_id=token.agent_id,
                base_url=route.base_url,
                reason="route has no pricing and the provider reported no cost; budget_usd cannot bind",
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


def _upstream_error(response: httpx.Response) -> dict[str, str]:
    """Enough provider detail to diagnose a failed request, with the request's bearer key redacted."""
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
        result[key] = text[: 2000 if key == "message" else 200]
    return result


# --- usage extraction ------------------------------------------------------
def _usage_from_response(response: httpx.Response) -> tuple[Usage, str | None]:
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


def _absorb_usage(native: object, usage: Usage) -> None:
    """Add the counts of the Chat Completions usage block ``native`` to ``usage``.

    A ``native`` that is not a dict adds nothing. A provider's ``cost`` or ``cost_usd``
    adds to ``usage.cost_usd``.
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
    # OpenRouter reports the amount that it charged. That amount is authoritative, and a
    # route's ``pricing`` applies only without it (``_tap``).
    usage.cost_usd += float(native.get("cost") or native.get("cost_usd") or 0)

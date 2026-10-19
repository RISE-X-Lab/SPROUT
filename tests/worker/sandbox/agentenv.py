"""A scripted AgentENV API on a local port, which records each request that it gets.

``agentenv()`` serves on a free port of 127.0.0.1 until its ``with`` block exits. A test scripts the
replies to one method and path with ``AgentEnvStub.answer``. A request without a script gets 404, as
AgentENV answers for a name that it does not know. ``scripted_build`` scripts the replies that a
template build gets from an AgentENV that knows no template yet.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import default as default_policy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit

#: A reply: the HTTP status and a body. A dict or a list is sent as JSON, a str as text, bytes as
#: they are, and None as nothing.
Reply = tuple[int, Any]


@dataclass(frozen=True)
class Request:
    """One request that the stub got. ``headers`` holds the lower-case names and is left out of ``==``."""

    method: str
    path: str
    #: The query parameters, as a dict.
    query: dict[str, str] = field(default_factory=dict)
    #: The JSON body, or ``{"name": ..., "filename": ..., "content": bytes}`` for a multipart upload.
    body: Any = None
    headers: dict[str, str] = field(default_factory=dict, compare=False)


@dataclass
class AgentEnvStub:
    """The base URL of the stub, each request in arrival order, and the scripted replies."""

    url: str
    requests: list[Request] = field(default_factory=list)
    _script: dict[tuple[str, str], list[Reply]] = field(default_factory=dict)

    def answer(self, method: str, path: str, *replies: Reply) -> None:
        """Answer ``method path`` with ``replies``, one for each request, in order. The last one repeats."""
        self._script[(method, path)] = list(replies)

    def paths(self) -> list[str]:
        """``"<method> <path>"`` of each request, in arrival order."""
        return ["%s %s" % (request.method, request.path) for request in self.requests]

    def requests_to(self, path: str) -> list[Request]:
        """Each request to ``path``, in arrival order."""
        return [request for request in self.requests if request.path == path]

    def bodies(self, *paths: str) -> list[Any]:
        """The body of each request to one of ``paths``, in arrival order."""
        return [request.body for request in self.requests if request.path in paths]

    def header_values(self, name: str) -> set[str | None]:
        """The values of the header ``name`` across all requests. ``None`` stands for a request without it."""
        return {request.headers.get(name) for request in self.requests}

    def reply(self, method: str, path: str) -> Reply:
        replies = self._script.get((method, path))
        if not replies:
            return 404, {"message": "%s not found" % path}
        return replies.pop(0) if len(replies) > 1 else replies[0]


def scripted_build(stub: AgentEnvStub) -> None:
    """Script a build that succeeds, in the staging sandbox ``staging-sb``, as the template ``tid``."""
    stub.answer("POST", "/sandboxes-cold", (201, {"sandboxID": "staging-sb"}))
    stub.answer("POST", "/files", (200, [{"path": "/usr/local/bin/sprout-runtime"}]))
    stub.answer("POST", "/sandboxes/staging-sb/snapshots", (201, {"snapshotID": "staged-snap"}))
    stub.answer("DELETE", "/sandboxes/staging-sb", (204, None))
    stub.answer("POST", "/v3/templates", (202, {"templateID": "tid", "buildID": "bid"}))
    stub.answer("POST", "/v2/templates/tid/builds/bid", (202, {}))
    stub.answer("GET", "/templates/tid/builds/bid/status", (200, {"status": "ready"}))


def _body(content_type: str, raw: bytes) -> Any:
    if not raw:
        return None
    if content_type.startswith("multipart/form-data"):
        message = BytesParser(policy=default_policy).parsebytes(
            b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + raw
        )
        [part] = list(message.iter_parts())
        return {
            "name": part.get_param("name", header="content-disposition"),
            "filename": part.get_filename(),
            "content": part.get_payload(decode=True),
        }
    return json.loads(raw)


def _encoded(body: Any) -> tuple[bytes, str]:
    if body is None:
        return b"", "application/json"
    if isinstance(body, bytes):
        return body, "application/octet-stream"
    if isinstance(body, str):
        return body.encode(), "text/plain"
    return json.dumps(body).encode(), "application/json"


@contextmanager
def agentenv() -> Iterator[AgentEnvStub]:
    """An ``AgentEnvStub`` that serves until the block exits."""
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def _serve(self) -> None:
            target = urlsplit(self.path)
            raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
            request = Request(
                method=self.command,
                path=target.path,
                query=dict(parse_qsl(target.query)),
                body=_body(self.headers.get("Content-Type", ""), raw),
                headers={name.lower(): value for name, value in self.headers.items()},
            )
            with lock:
                stub.requests.append(request)
                status, body = stub.reply(self.command, target.path)
            data, content_type = _encoded(body)
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        do_GET = do_POST = do_DELETE = _serve

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    stub = AgentEnvStub(url="http://127.0.0.1:%d" % server.server_port)
    # A short poll, so that the shutdown at the end of each test takes no half second.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)

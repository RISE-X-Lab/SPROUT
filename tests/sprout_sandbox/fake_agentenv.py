"""A stand-in for the HTTP API of an AgentENV server, which ``MicroVMPool`` is a client of.

It answers the routes that the pool uses, with the field names of AgentENV: create, cold start,
delete, snapshot, squash, the file service and the proxy of a tool call. It keeps each request. The
attributes of ``FakeAgentEnv`` set the answers that a test needs.
"""

from __future__ import annotations

import itertools
import json
import re
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from email.parser import BytesParser
from email.policy import HTTP
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from urllib.parse import parse_qsl, urlsplit


@dataclass(frozen=True)
class Request:
    """One request to the fake server."""

    method: str
    #: The path without the query.
    path: str
    params: dict[str, str]
    #: The headers, with lower-case names.
    headers: dict[str, str]
    #: The JSON body, or ``None`` if the body is not JSON.
    json: Any
    #: Each file of a multipart body: its field name, with its file name and its bytes.
    files: dict[str, tuple[str, bytes]]


@dataclass
class FakeAgentEnv:
    """The server's answers, and the requests that it got. ``url`` is its base URL."""

    url: str = ""
    requests: list[Request] = field(default_factory=list)
    #: The body of a create reply. ``None`` names a new sandbox: ``vm-1``, ``vm-2`` and so on.
    create_body: dict[str, Any] | None = None
    delete_status: int = 200
    snapshot_body: dict[str, Any] = field(
        default_factory=lambda: {"snapshotID": "snap-7", "rootfsLayerCount": 4, "memoryLayerCount": 2}
    )
    squash_body: dict[str, Any] = field(
        default_factory=lambda: {"snapshotID": "snap-flat", "rootfsLayerCount": 1, "memoryLayerCount": 1}
    )
    #: The bytes of the guest file that a download gets, and the status of that reply.
    file: bytes = b""
    file_status: int = 200
    #: When set, the server sends only this many bytes of ``file``, then closes the connection.
    file_cut_at: int | None = None

    def paths(self, method: str) -> list[str]:
        """The path of each request with ``method``, in arrival order."""
        return [request.path for request in self.requests if request.method == method]

    def posted(self) -> list[tuple[str, Any]]:
        """The path and the JSON body of each POST, in arrival order."""
        return [(request.path, request.json) for request in self.requests if request.method == "POST"]


#: The reply of the proxy to a tool call in a sandbox.
TOOL_REPLY = {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {"content": [{"type": "text", "text": "ok"}], "isError": False},
}


def _files(content_type: str, body: bytes) -> dict[str, tuple[str, bytes]]:
    if not content_type.startswith("multipart/"):
        return {}
    message = BytesParser(policy=HTTP).parsebytes(
        b"Content-Type: " + content_type.encode() + b"\r\n\r\n" + body
    )
    return {
        part.get_param("name", header="content-disposition"): (
            part.get_filename(),
            part.get_payload(decode=True),
        )
        for part in message.iter_parts()
    }


class _Handler(BaseHTTPRequestHandler):
    server: _Server

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def _record(self) -> Request:
        body = self.rfile.read(int(self.headers.get("Content-Length") or 0))
        content_type = self.headers.get("Content-Type", "")
        url = urlsplit(self.path)
        request = Request(
            method=self.command,
            path=url.path,
            params=dict(parse_qsl(url.query)),
            headers={name.lower(): value for name, value in self.headers.items()},
            json=json.loads(body) if content_type == "application/json" else None,
            files=_files(content_type, body),
        )
        self.server.fake.requests.append(request)
        return request

    def _answer(self, status: int, body: Any) -> None:
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_POST(self) -> None:
        fake, path = self.server.fake, self._record().path
        if path in ("/sandboxes", "/sandboxes-cold"):
            created = {"sandboxID": "vm-%d" % next(self.server.numbers)}
            self._answer(200, created if fake.create_body is None else fake.create_body)
        elif re.fullmatch(r"/sandboxes/[^/]+/snapshots", path):
            self._answer(200, fake.snapshot_body)
        elif re.fullmatch(r"/snapshots/[^/]+/squash", path):
            self._answer(200, fake.squash_body)
        elif path == "/files":
            self._answer(200, {})
        elif path == "/":
            self._answer(200, TOOL_REPLY)
        else:
            self._answer(404, {"message": "no route %s" % path})

    def do_DELETE(self) -> None:
        self._record()
        self._answer(self.server.fake.delete_status, {})

    def do_GET(self) -> None:
        self._record()
        fake = self.server.fake
        self.send_response(fake.file_status)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(fake.file)))
        self.end_headers()
        self.wfile.write(fake.file[: fake.file_cut_at])


class _Server(ThreadingHTTPServer):
    daemon_threads = True
    fake: FakeAgentEnv
    numbers: Iterator[int]

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that gave up on a reply closed its end first. The test expects that.
        pass


@contextmanager
def serving_agentenv() -> Iterator[FakeAgentEnv]:
    """A fake AgentENV server on a free port of 127.0.0.1, until the block exits."""
    server = _Server(("127.0.0.1", 0), _Handler)
    server.fake = FakeAgentEnv(url="http://127.0.0.1:%d" % server.server_port)
    server.numbers = itertools.count(1)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield server.fake
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)

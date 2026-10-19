"""A stub of the AgentENV REST API on a local port, for the clients that list and delete without a pool."""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any


@dataclass
class AgentEnvStub:
    """What the stub answers, and the requests that it got.

    ``GET /sandboxes`` answers ``sandboxes``. A ``DELETE`` answers the status that ``delete_statuses``
    gives for its path, such as ``/snapshots/snapshot-1``, and 200 for any other path."""

    url: str
    sandboxes: list[dict[str, Any]] = field(default_factory=list)
    delete_statuses: dict[str, int] = field(default_factory=dict)
    #: ``(method, path, X-API-Key header)`` of each request, in arrival order.
    requests: list[tuple[str, str, str | None]] = field(default_factory=list)


@contextmanager
def agentenv_stub() -> Iterator[AgentEnvStub]:
    """An ``AgentEnvStub`` that serves on 127.0.0.1 until the block exits."""

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_GET(self) -> None:
            stub.requests.append(("GET", self.path, self.headers.get("X-API-Key")))
            self._answer(200, json.dumps(stub.sandboxes).encode())

        def do_DELETE(self) -> None:
            stub.requests.append(("DELETE", self.path, self.headers.get("X-API-Key")))
            self._answer(stub.delete_statuses.get(self.path, 200), b"{}")

        def _answer(self, status: int, body: bytes) -> None:
            self.send_response(status)
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    stub = AgentEnvStub(url="http://127.0.0.1:%d" % httpd.server_port)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield stub
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)

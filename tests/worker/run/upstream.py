"""A model provider on a local port that answers in script order and keeps the headers of each request.

``tests.support.http.chat_server`` keeps only the bodies. The credential that a request carries is
the behavior of some tests here, so this server keeps the headers too.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from tests.support.http import CHAT_PATH


@dataclass(frozen=True)
class ProviderRequest:
    """One request to ``CHAT_PATH``: its headers, with lower-case names, and its JSON body."""

    headers: dict[str, str]
    body: dict[str, Any]


@dataclass
class Provider:
    """The address of the provider, and the requests that it got in arrival order."""

    #: The base URL, such as ``http://127.0.0.1:41234``, without ``/v1``.
    url: str
    requests: list[ProviderRequest] = field(default_factory=list)


@contextmanager
def provider(replies: list[dict[str, Any]]) -> Iterator[Provider]:
    """A ``Provider`` that answers the requests to ``CHAT_PATH`` with ``replies`` in order, then with 500."""
    script = iter(replies)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                served.requests.append(
                    ProviderRequest({key.lower(): value for key, value in self.headers.items()}, body)
                )
                reply = next(script, None)
            status, data = (
                (200, reply) if reply is not None else (500, {"error": {"message": "no reply left"}})
            )
            encoded = json.dumps(data).encode()
            self.send_response(status if self.path == CHAT_PATH else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(encoded)))
            self.end_headers()
            self.wfile.write(encoded)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    served = Provider(url="http://127.0.0.1:%d" % httpd.server_port)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield served
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)

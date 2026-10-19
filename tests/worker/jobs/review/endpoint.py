"""A Chat Completions endpoint on 127.0.0.1 whose answer to each request is a function of the request.

``scripted_endpoint(answer)`` serves until its ``with`` block exits. ``answer(body)`` gives a
``(status, body)`` pair, ``HOLD`` to keep the request unanswered until the block exits, or ``DROP`` to
close the connection without an answer. Each request is kept, with its path and its headers.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: An answer: keep the request open until the endpoint stops. The client times out first.
HOLD = "hold"
#: An answer: close the connection without a response.
DROP = "drop"

Answer = Callable[[dict[str, Any]], "tuple[int, dict[str, Any]] | str"]


@dataclass
class Request:
    path: str
    authorization: str | None
    body: dict[str, Any]


@dataclass
class Endpoint:
    """The base URL of the endpoint, without ``/v1``, and each request that it received, in arrival order."""

    url: str
    requests: list[Request] = field(default_factory=list)


def completion(text: str | None, *, finish_reason: str = "stop") -> dict[str, Any]:
    """A ``chat.completion`` body whose one message has the content ``text``."""
    message = {"role": "assistant", "content": text}
    return {
        "object": "chat.completion",
        "choices": [{"index": 0, "finish_reason": finish_reason, "message": message}],
    }


@contextmanager
def scripted_endpoint(answer: Answer) -> Iterator[Endpoint]:
    stopping = threading.Event()
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            with lock:
                endpoint.requests.append(Request(self.path, self.headers.get("Authorization"), body))
            reply = answer(body)
            if reply == HOLD:
                stopping.wait()
                return
            if reply == DROP:
                self.close_connection = True
                return
            status, payload = reply
            data = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    endpoint = Endpoint(url="http://127.0.0.1:%d" % server.server_port)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield endpoint
    finally:
        stopping.set()
        server.shutdown()
        server.server_close()
        thread.join(5)

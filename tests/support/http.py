"""Local HTTP servers for one test: an ASGI app on a port, a scripted OpenAI chat endpoint, and a silent one.

Each server listens on a free port of 127.0.0.1, serves from a daemon thread, and stops when its
``with`` block exits. ``serving`` needs uvicorn, from the runstore or rl extra.
"""

from __future__ import annotations

import itertools
import json
import socket
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

#: The one path that ``chat_server`` answers.
CHAT_PATH = "/v1/chat/completions"


@contextmanager
def serving(app: Any) -> Iterator[str]:
    """The base URL, such as ``http://127.0.0.1:41234``, of the ASGI ``app`` served by uvicorn.

    Raises ``AssertionError`` if the server does not start within 5 s, or does not stop within 10 s."""
    import uvicorn

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    url = "http://127.0.0.1:%d" % listener.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(app, log_level="error"))
    thread = threading.Thread(target=server.run, kwargs={"sockets": [listener]}, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 5
        while not server.started:
            if not thread.is_alive() or time.monotonic() > deadline:
                raise AssertionError("the test HTTP service did not start")
            time.sleep(0.01)
        yield url
    finally:
        server.should_exit = True
        thread.join(timeout=10)
        listener.close()
        assert not thread.is_alive(), "the test HTTP service did not stop"


@contextmanager
def silent_endpoint(connections: int) -> Iterator[tuple[str, threading.Event]]:
    """The URL of an endpoint that never answers, and an event set once ``connections`` requests are open."""
    server = socket.create_server(("127.0.0.1", 0))
    held: list[socket.socket] = []
    opened = threading.Event()

    def accept() -> None:
        while len(held) < connections:
            try:
                held.append(server.accept()[0])
            except OSError:
                return
        opened.set()

    threading.Thread(target=accept, daemon=True).start()
    try:
        yield "http://127.0.0.1:%d" % server.getsockname()[1], opened
    finally:
        server.close()
        for connection in held:
            connection.close()


@dataclass
class ChatServer:
    """A chat endpoint that answers each request with the next scripted reply, and keeps each request body."""

    #: The base URL, such as ``http://127.0.0.1:41234``. A client appends ``CHAT_PATH`` to it.
    url: str
    #: The JSON body of each request to ``CHAT_PATH``, in arrival order.
    requests: list[dict[str, Any]] = field(default_factory=list)


@contextmanager
def chat_server(replies: list[dict[str, Any] | tuple[int, dict[str, Any]]]) -> Iterator[ChatServer]:
    """A ``ChatServer`` that answers the requests to ``CHAT_PATH`` with ``replies``, one each, in order.

    A reply is a JSON body with status 200, or a ``(status, body)`` pair. A tool call with the id
    ``None`` gets ``call-1``, ``call-2`` and so on, in the order that the server sends them. After the
    last reply, it answers 500. It answers 404 to any other path, and keeps no record of the request."""
    script = iter(deepcopy(replies))
    numbering = itertools.count(1)
    lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            body = self.rfile.read(int(self.headers["Content-Length"]))
            if self.path != CHAT_PATH:
                self._answer(404, {"error": {"message": "no route %s" % self.path}})
                return
            with lock:
                server.requests.append(json.loads(body))
                status, reply = _next_reply(script, numbering)
            self._answer(status, reply)

        def _answer(self, status: int, body: dict[str, Any]) -> None:
            data = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    server = ChatServer(url="http://127.0.0.1:%d" % httpd.server_port)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def _next_reply(script: Iterator[Any], numbering: Iterator[int]) -> tuple[int, dict[str, Any]]:
    """The status and body of the next scripted reply, with each missing tool call id numbered."""
    reply = next(script, None)
    if reply is None:
        return 500, {"error": {"message": "the chat script has no reply left"}}
    status, body = reply if isinstance(reply, tuple) else (200, reply)
    for choice in body.get("choices", []):
        for call in choice.get("message", {}).get("tool_calls") or []:
            if call.get("id") is None:
                call["id"] = "call-%d" % next(numbering)
    return status, body


def chat_completion(
    *commands: str,
    content: str = "Thinking.",
    reasoning: str = "Inspect the actual file.",
    ids: tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """An OpenAI ``chat.completion`` whose message calls the ``bash`` tool once for each of ``commands``.

    ``ids`` names the tool calls. Without it, ``chat_server`` numbers them. The usage is fixed:
    11 prompt tokens, 3 of them cached, and 7 completion tokens, 5 of them reasoning."""
    calls = [
        {
            "id": ids[index] if ids else None,
            "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})},
        }
        for index, command in enumerate(commands)
    ]
    message = {"role": "assistant", "content": content, "reasoning_content": reasoning, "tool_calls": calls}
    return {
        "id": "chatcmpl-fixture",
        "object": "chat.completion",
        "model": "fixture",
        "choices": [{"index": 0, "finish_reason": "tool_calls", "message": message}],
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "prompt_tokens_details": {"cached_tokens": 3},
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    }

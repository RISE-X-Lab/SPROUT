"""A stand-in for sprout-runtime: a JSON-RPC server on a local port or on a Unix socket.

It answers ``tools/list`` with one ``shell`` tool, and ``tools/call`` with the text ``ok``, as the
runtime answers a command that has nothing to report beyond its stdout. A test can script another
reply. The server keeps the body and the headers of each request. The fake podman starts no
container, so ``SandboxRuntimes`` puts a runtime on the socket of each sandbox that a pool starts.
"""

from __future__ import annotations

import json
import socket
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, HTTPServer, ThreadingHTTPServer
from pathlib import Path
from socketserver import ThreadingMixIn
from typing import Any

#: The JSON-RPC result of ``tools/list``: sprout-runtime sends a bare array.
TOOLS = [{"name": "shell"}]
#: The JSON-RPC result of a ``tools/call`` whose command printed ``ok`` and exited 0.
PLAIN_OK = {"content": [{"type": "text", "text": "ok"}], "isError": False}


@dataclass(frozen=True)
class Received:
    """One request to the fake runtime: its JSON body, and its headers with lower-case names."""

    body: dict[str, Any]
    headers: dict[str, str]


@dataclass(frozen=True)
class Reply:
    """A scripted answer: the JSON body, the HTTP status, and the seconds the server waits first."""

    body: dict[str, Any]
    status: int = 200
    delay_s: float = 0.0


@dataclass
class FakeRuntime:
    """One fake runtime. ``url`` reaches it, through the Unix socket ``uds`` when that is set."""

    url: str = ""
    uds: str | None = None
    #: The answer to each request. ``None`` answers as the runtime does.
    reply: Reply | None = None
    received: list[Received] = field(default_factory=list)

    def bodies(self) -> list[dict[str, Any]]:
        """The JSON body of each request, in arrival order."""
        return [request.body for request in self.received]


def rpc_result(result: Any) -> dict[str, Any]:
    """A JSON-RPC reply body that carries ``result``."""
    return {"jsonrpc": "2.0", "id": 1, "result": result}


def rpc_error(error: Any) -> dict[str, Any]:
    """A JSON-RPC reply body that carries ``error``."""
    return {"jsonrpc": "2.0", "id": 1, "error": error}


def _runtime_answer(body: dict[str, Any]) -> Reply:
    return Reply(body=rpc_result(TOOLS if body.get("method") == "tools/list" else PLAIN_OK))


class _Handler(BaseHTTPRequestHandler):
    server: _TcpServer | _UnixServer

    def log_message(self, format: str, *args: Any) -> None:
        pass

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        runtime = self.server.runtime
        runtime.received.append(Received(body=body, headers={k.lower(): v for k, v in self.headers.items()}))
        reply = runtime.reply or _runtime_answer(body)
        time.sleep(reply.delay_s)
        data = json.dumps(reply.body).encode()
        self.send_response(reply.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


class _TcpServer(ThreadingHTTPServer):
    daemon_threads = True
    runtime: FakeRuntime

    def handle_error(self, request: Any, client_address: Any) -> None:
        # A client that gave up on a slow reply closed its end first. The test expects that.
        pass


class _UnixServer(ThreadingMixIn, HTTPServer):
    address_family = socket.AF_UNIX
    daemon_threads = True
    runtime: FakeRuntime

    def handle_error(self, request: Any, client_address: Any) -> None:
        pass

    def server_bind(self) -> None:
        self.socket.bind(self.server_address)
        self.server_name, self.server_port = "localhost", 0


def _start(server: _TcpServer | _UnixServer, runtime: FakeRuntime) -> threading.Thread:
    server.runtime = runtime
    # A short poll interval, so that a shutdown at the end of a test takes milliseconds.
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    return thread


def _stop(server: _TcpServer | _UnixServer, thread: threading.Thread) -> None:
    server.shutdown()
    server.server_close()
    thread.join(5)


@contextmanager
def serving_runtime(*, uds: Path | None = None) -> Iterator[FakeRuntime]:
    """A fake runtime on a free port of 127.0.0.1, or on the Unix socket ``uds``, until the block exits."""
    if uds is None:
        server: _TcpServer | _UnixServer = _TcpServer(("127.0.0.1", 0), _Handler)
        runtime = FakeRuntime(url="http://127.0.0.1:%d" % server.server_port)
    else:
        server = _UnixServer(str(uds), _Handler)
        runtime = FakeRuntime(url="http://sprout-runtime", uds=str(uds))
    thread = _start(server, runtime)
    try:
        yield runtime
    finally:
        _stop(server, thread)


class SocketWatcher:
    """Calls ``on_sandbox`` once for each sandbox directory that appears in a pool's socket directory.

    It looks about every 5 ms, from a thread of its own, until ``close``."""

    def __init__(self, directory: Path, on_sandbox: Callable[[Path], None]) -> None:
        self._directory = directory
        self._on_sandbox = on_sandbox
        self._seen: set[str] = set()
        self._stopped = threading.Event()
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()

    def _watch(self) -> None:
        while not self._stopped.wait(0.005):
            children = sorted(self._directory.iterdir()) if self._directory.is_dir() else []
            for child in children:
                if child.name in self._seen or not child.is_dir():
                    continue
                self._seen.add(child.name)
                try:
                    self._on_sandbox(child)
                except OSError:
                    # The pool removed the directory first, as it does after a failed spawn.
                    pass

    def close(self) -> None:
        self._stopped.set()
        self._thread.join(5)


class SandboxRuntimes:
    """A fake runtime on ``<socket dir>/<sandbox id>/runtime.sock`` for each sandbox that a pool starts."""

    SOCKET_FILE = "runtime.sock"

    def __init__(self, socket_dir: Path) -> None:
        self._servers: list[tuple[_UnixServer, threading.Thread]] = []
        self._runtimes: dict[str, FakeRuntime] = {}
        self._watcher = SocketWatcher(socket_dir, self._serve)

    def _serve(self, directory: Path) -> None:
        path = directory / self.SOCKET_FILE
        server = _UnixServer(str(path), _Handler)
        runtime = FakeRuntime(url="http://sprout-runtime", uds=str(path))
        self._servers.append((server, _start(server, runtime)))
        self._runtimes[directory.name] = runtime

    def of(self, sandbox_id: str) -> FakeRuntime:
        """The fake runtime of the sandbox ``sandbox_id``."""
        return self._runtimes[sandbox_id]

    def close(self) -> None:
        self._watcher.close()
        for server, thread in self._servers:
            _stop(server, thread)

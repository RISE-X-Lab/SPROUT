"""The http probe against the loopback interface: every answer is a Probe, never a raise."""

from __future__ import annotations

import socket
import threading

from sprout.services import health


def test_a_port_that_is_not_http_is_reported_not_raised():
    """A `health.url` pointing at PostgreSQL's port (one digit off) answers with
    something that is no status line; urllib raises BadStatusLine unwrapped."""
    with socket.socket() as server:
        server.bind(("127.0.0.1", 0))
        server.listen(1)
        port = server.getsockname()[1]

        def answer() -> None:
            conn, _ = server.accept()
            with conn:
                conn.recv(4096)
                conn.sendall(b"garbage\r\n\r\n")

        thread = threading.Thread(target=answer, daemon=True)
        thread.start()
        result = health.http("http://127.0.0.1:%d/health" % port)
        thread.join(5)
    assert result == health.Probe(False, "not an HTTP reply: garbage")


def test_nothing_listening_is_a_refused_connection():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    result = health.http("http://127.0.0.1:%d/health" % port)       # the socket is closed: nobody there
    assert not result.ok and "refused" in result.detail.lower(), result

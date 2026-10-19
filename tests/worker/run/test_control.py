"""The out-of-band stop of a rollout: ``RunControl`` and the HTTP ``post`` that a stop ends."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import httpx
import pytest

from sprout.worker.run.control import RunAborted, RunControl, post


def test_only_the_first_stop_request_counts():
    control = RunControl()

    control.request_stop("wall-time budget exhausted", stop_reason="timeout")
    control.request_stop("turn budget exhausted", stop_reason="max_turns_reached")

    assert (control.reason, control.stop_reason) == ("wall-time budget exhausted", "timeout")


def test_a_subscribed_callback_runs_once_when_the_rollout_stops():
    control = RunControl()
    calls: list[str] = []
    control.subscribe(lambda: calls.append("stopped"))

    control.request_stop("first")
    control.request_stop("second")

    assert calls == ["stopped"]


def test_a_callback_subscribed_after_the_stop_runs_at_once():
    control = RunControl()
    control.request_stop("stopped before the subscription")
    calls: list[str] = []

    control.subscribe(lambda: calls.append("stopped"))

    assert calls == ["stopped"]


def test_an_unsubscribed_callback_does_not_run():
    control = RunControl()
    calls: list[str] = []
    unsubscribe = control.subscribe(lambda: calls.append("stopped"))

    unsubscribe()
    control.request_stop("stopped")

    assert calls == []


def test_a_callback_that_finds_its_loop_closed_does_not_keep_the_others_from_running():
    """A driver whose event loop is already closed raises RuntimeError from its callback."""
    control = RunControl()
    calls: list[str] = []

    def closed_loop() -> None:
        raise RuntimeError("Event loop is closed")

    control.subscribe(closed_loop)
    control.subscribe(lambda: calls.append("stopped"))

    control.request_stop("stopped")

    assert (control.reason, calls) == ("stopped", ["stopped"])


def test_raise_if_stopped_raises_run_aborted_with_the_reason():
    control = RunControl()
    control.raise_if_stopped()
    control.request_stop("rollout turn budget exhausted")

    with pytest.raises(RunAborted, match="^rollout turn budget exhausted$"):
        control.raise_if_stopped()


@dataclass
class StalledServer:
    """A server that receives each POST and sends nothing until the test ends."""

    url: str
    received: threading.Event = field(default_factory=threading.Event)


@contextmanager
def stalled_server() -> Iterator[StalledServer]:
    release = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: object) -> None:
            pass

        def do_POST(self) -> None:
            server.received.set()
            release.wait(30)

    httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    httpd.daemon_threads = True
    server = StalledServer(url="http://127.0.0.1:%d/v1/chat/completions" % httpd.server_port)
    thread = threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
    thread.start()
    try:
        yield server
    finally:
        release.set()
        httpd.shutdown()
        httpd.server_close()
        thread.join(5)


def stop_once_received(server: StalledServer, control: RunControl, reason: str) -> threading.Thread:
    """A thread that stops ``control`` for ``reason`` after the server has received the request."""

    def stop() -> None:
        server.received.wait(10)
        control.request_stop(reason)

    thread = threading.Thread(target=stop, daemon=True)
    thread.start()
    return thread


def test_a_stop_ends_a_request_to_which_the_server_sends_nothing():
    control = RunControl()
    with stalled_server() as server:
        stopper = stop_once_received(server, control, "operator stop")

        with pytest.raises(RunAborted, match="^operator stop$"):
            post(server.url, json={"model": "fixture"}, timeout_s=10, control=control)

    stopper.join(5)


def test_a_request_after_the_stop_is_not_sent():
    control = RunControl()
    control.request_stop("operator stop")

    with stalled_server() as server:
        with pytest.raises(RunAborted, match="^operator stop$"):
            post(server.url, json={"model": "fixture"}, timeout_s=10, control=control)

        assert not server.received.is_set()


def test_a_request_fails_after_its_timeout():
    with stalled_server() as server:
        with pytest.raises((TimeoutError, httpx.TimeoutException)):
            post(server.url, json={"model": "fixture"}, timeout_s=0.2)

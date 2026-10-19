"""The health probes: each answer, and each failure to get one, is a ``Probe``, never an exception.

The HTTP probes talk to servers on the loopback interface. The ``pg_isready`` probes run a stub
``pg_isready`` or a stub podman, with PATH set to hide the host's own tools.
"""

from __future__ import annotations

import http.server
import socket
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import pytest

from sprout.services import health
from sprout.services.health import Probe
from sprout.services.spec import Context, Health
from sprout.services.state import Paths
from tests.services.helpers import free_port, make_stub


class TokenHandler(http.server.BaseHTTPRequestHandler):
    """200 for a request with the right bearer token, 401 for any other, as the Run Store API answers."""

    def do_GET(self) -> None:
        self.send_response(200 if self.headers.get("Authorization") == "Bearer right" else 401)
        self.end_headers()

    def log_message(self, *args: object) -> None:
        pass


@contextmanager
def token_server() -> Iterator[str]:
    """The base URL of a TokenHandler server on a free loopback port."""
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), TokenHandler)
    thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_address[1]
    finally:
        server.shutdown()
        server.server_close()


@contextmanager
def raw_server(reply: bytes | None) -> Iterator[str]:
    """A server that reads one request and sends ``reply``. It accepts nothing when ``reply`` is None."""
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)

        def answer() -> None:
            connection, _ = listener.accept()
            with connection:
                connection.recv(4096)
                connection.sendall(reply or b"")

        thread = threading.Thread(target=answer, daemon=True)
        if reply is not None:
            thread.start()
        yield "http://127.0.0.1:%d" % listener.getsockname()[1]


def context(*, environ=None, podman: str = "/path/to/podman") -> Context:
    def host(name: str) -> str:
        if not podman:
            raise ValueError("python -m sprout.worker doctor --json gave no report (exit 1): boom")
        return podman

    return Context(
        paths=Paths(config=Path("/path/to/config"), state=Path("/path/to/state"), repo=Path("/path/to/repo")),
        python="/path/to/python",
        secret=lambda name: "value-of-" + name,
        host=host,
        environ={} if environ is None else environ,
    )


# --- http ---------------------------------------------------------------------


@pytest.mark.parametrize(
    "headers, expected",
    [([("Authorization", "Bearer right")], Probe(True, "HTTP 200")), ([], Probe(False, "HTTP 401"))],
)
def test_an_http_probe_sends_its_headers_and_passes_only_on_200(headers, expected):
    with token_server() as url:
        result = health.http(url + "/v1/jobs", headers)

    assert result == expected


def test_a_port_that_answers_with_something_other_than_http_fails_the_probe():
    """A ``health.url`` one digit off can name PostgreSQL's port. urllib raises BadStatusLine unwrapped."""
    with raw_server(b"garbage\r\n\r\n") as url:
        result = health.http(url + "/health")

    assert result == Probe(False, "not an HTTP reply: garbage")


def test_a_port_that_nothing_listens_on_fails_the_probe_as_refused():
    assert health.http("http://127.0.0.1:%d/health" % free_port()) == Probe(False, "Connection refused")


def test_a_server_that_does_not_answer_within_the_timeout_fails_the_probe():
    with raw_server(None) as url:
        result = health.http(url + "/health", timeout=0.2)

    assert result == Probe(False, "timed out after 0.2s")


def test_a_url_without_a_scheme_fails_the_probe():
    """An empty ``AENV_SERVER_URL`` turns ``{env:AENV_SERVER_URL}/sandboxes`` into ``/sandboxes``."""
    assert health.http("/sandboxes") == Probe(False, "unknown url type: '/sandboxes'")


@pytest.mark.parametrize(
    "probe",
    [
        Health(kind="http", url="{env:AENV_SERVER_URL}/sandboxes"),
        Health(
            kind="http", url="http://127.0.0.1:1/sandboxes", headers=(("X-API-Key", "{env:AENV_SERVER_URL}"),)
        ),
    ],
)
def test_an_http_probe_whose_placeholder_this_run_cannot_resolve_fails_with_the_reason(probe):
    result = health.probe(probe, context(environ={}), alive=None, alive_detail="")

    assert result == Probe(False, "environment variable AENV_SERVER_URL is not set")


# --- liveness -----------------------------------------------------------------


@pytest.mark.parametrize(
    "kind, alive, detail, expected",
    [
        ("none", None, "", Probe(True, "not probed")),
        ("none", True, "pid 12", Probe(True, "pid 12")),
        ("none", False, "pid 12 is gone", Probe(False, "pid 12 is gone")),
        ("process", None, "", Probe(False, "nothing to probe")),
        ("process", True, "pid 12", Probe(True, "pid 12")),
        ("process", False, "pid 12 is gone", Probe(False, "pid 12 is gone")),
    ],
)
def test_a_probe_of_kind_none_or_process_reports_the_liveness_that_the_caller_knows(
    kind, alive, detail, expected
):
    """``alive`` is None for a service without a process of its own, such as an external one."""
    result = health.probe(Health(kind=kind), context(), alive=alive, alive_detail=detail)

    assert result == expected


# --- pg_isready ---------------------------------------------------------------


def database(*, user: str = "sprout_runstore", container: str = "sprout-services-db") -> Health:
    return Health(kind="pg_isready", host="127.0.0.1", port=15432, user=user, container=container)


@pytest.mark.parametrize(
    "user, argv",
    [
        ("sprout_runstore", ["-h", "127.0.0.1", "-p", "15432", "-U", "sprout_runstore"]),
        ("", ["-h", "127.0.0.1", "-p", "15432"]),
    ],
)
def test_a_pg_isready_probe_asks_the_hosts_pg_isready_about_the_published_port(
    tmp_path, monkeypatch, user, argv
):
    """The host's tool proves that the published port works: the Run Store connects through it."""
    tool = make_stub(tmp_path / "bin" / "pg_isready")
    tool.answer(out="127.0.0.1:15432 - accepting connections\n")
    monkeypatch.setenv("PATH", str(tool.path.parent))

    result = health.probe(database(user=user), context(), alive=None, alive_detail="")

    assert (result, tool.calls()) == (Probe(True, "127.0.0.1:15432 - accepting connections"), [argv])


def test_a_pg_isready_that_gets_no_answer_fails_the_probe_with_its_last_line(tmp_path, monkeypatch):
    tool = make_stub(tmp_path / "bin" / "pg_isready")
    tool.answer(out="127.0.0.1:15432 - no response\n", code=2)
    monkeypatch.setenv("PATH", str(tool.path.parent))

    result = health.probe(database(), context(), alive=None, alive_detail="")

    assert result == Probe(False, "127.0.0.1:15432 - no response")


def test_without_pg_isready_on_the_host_the_probe_runs_it_in_the_container_on_the_doctors_podman(
    tmp_path, monkeypatch
):
    podman = make_stub(tmp_path / "sandboxes" / "podman")
    podman.answer(out="/var/run/postgresql:5432 - accepting connections\n")
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    result = health.probe(database(), context(podman=str(podman.path)), alive=None, alive_detail="")

    assert (result, podman.calls()) == (
        Probe(True, "in sprout-services-db: /var/run/postgresql:5432 - accepting connections"),
        [["exec", "sprout-services-db", "pg_isready", "-U", "sprout_runstore"]],
    )


def test_without_pg_isready_or_a_container_the_probe_says_what_to_install(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    result = health.probe(database(container=""), context(), alive=None, alive_detail="")

    assert result == Probe(
        False, "pg_isready is not installed (postgresql-client) and there is no container to run it in"
    )


def test_without_pg_isready_or_a_doctor_report_the_probe_fails_with_the_doctors_reason(monkeypatch, tmp_path):
    monkeypatch.setenv("PATH", str(tmp_path / "empty"))

    result = health.probe(database(), context(podman=""), alive=None, alive_detail="")

    assert result == Probe(False, "python -m sprout.worker doctor --json gave no report (exit 1): boom")


# --- programs -----------------------------------------------------------------


def executable(path: Path, mode: int = 0o755) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("#!/bin/sh\n")
    path.chmod(mode)
    return path


def test_a_program_on_path_is_found_there_before_the_one_in_local_bin(tmp_path, monkeypatch):
    on_path = executable(tmp_path / "bin" / "uv")
    executable(tmp_path / "home" / ".local" / "bin" / "uv")
    monkeypatch.setenv("PATH", str(on_path.parent))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert health.which("uv") == str(on_path)


def test_a_program_only_in_local_bin_is_found_there(tmp_path, monkeypatch):
    """uv's installer puts uv there. A shell without its profile, such as cron's, lacks it on PATH."""
    local = executable(tmp_path / "home" / ".local" / "bin" / "uv")
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert health.which("uv") == str(local)


def test_a_file_in_local_bin_that_is_not_executable_is_not_a_program(tmp_path, monkeypatch):
    executable(tmp_path / "home" / ".local" / "bin" / "uv", mode=0o644)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setenv("HOME", str(tmp_path / "home"))

    assert health.which("uv") is None

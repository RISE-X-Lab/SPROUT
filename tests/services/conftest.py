"""Every test here moves the config root, the state root and HOME to ``tmp_path``,
so a wrong path in the code cannot reach the real ``~/.config/sprout``; and
every throwaway child a test starts carries ``MARKER`` in its argv, so the
fixture can kill whatever a failing test left behind. Nothing talks to podman
or leaves the loopback interface: the "services" are ``python -c`` servers on
free ports.
"""

from __future__ import annotations

import os
import signal
import socket
import sys
from pathlib import Path

import pytest
import yaml

from sprout.services.state import Paths

MARKER = "sprout-services-test"

#: A one-file HTTP server: 200 on /health, 404 elsewhere, quiet. argv[1] is the port.
SERVER = """
import http.server, sys
class Handler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == '/health' else 404)
        self.end_headers()
        self.wfile.write(b'ok')
    def log_message(self, *args):
        pass
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), Handler).serve_forever()
"""


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def server_command(port: int) -> list[str]:
    return [sys.executable, "-c", SERVER, str(port), MARKER]


def sleeper_command() -> list[str]:
    return [sys.executable, "-c", "import time; time.sleep(600)", MARKER]


def write_spec(paths: Paths, services: dict, python: str | None = None) -> Path:
    """services.yaml as given: keys in the caller's order, since the order of
    `env` is the order of podman's `-e` flags and of a start sequence."""
    paths.config.mkdir(parents=True, exist_ok=True)
    paths.spec.write_text(yaml.safe_dump({"python": python, "services": services}, sort_keys=False), encoding="utf-8")
    return paths.spec


def kill_marked() -> None:
    """SIGKILL every process whose argv carries MARKER, then reap the ones that are ours."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes().split(b"\0")
        except OSError:
            continue
        if MARKER.encode() not in argv:
            continue
        pid = int(entry.name)
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            continue
        try:
            os.waitpid(pid, 0)
        except ChildProcessError:
            pass


@pytest.fixture
def paths(tmp_path, monkeypatch) -> Paths:
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SPROUT_SERVICES_CONFIG_DIR", str(tmp_path / "config"))
    monkeypatch.setenv("SPROUT_SERVICES_STATE_DIR", str(tmp_path / "state"))
    monkeypatch.delenv("FORCE_COLOR", raising=False)
    monkeypatch.delenv("SPROUT_PODMAN_STORE", raising=False)
    monkeypatch.delenv("SPROUT_PODMAN_BIN", raising=False)
    yield Paths.from_env()
    kill_marked()

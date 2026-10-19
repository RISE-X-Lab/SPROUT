"""The programs that the services tests start, the stubs that stand in for podman and the worker's doctor.

Each program that a test starts names the test's ``tmp_path`` in its argv, so the ``paths`` fixture
can kill what a failed test left running. A stub records each call and gives the answer that the test
set. It has no state, so each test sets the answer for the state that it needs. No helper runs the
real podman, and no program listens beyond the loopback interface.
"""

from __future__ import annotations

import json
import os
import re
import signal
import socket
import sys
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from sprout.services.health import Probe
from sprout.services.state import Paths

#: A service for the tests. Each option adds one behavior, so the command in a test says what it does.
SERVICE_PROGRAM = r"""
import argparse
import http.server
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--require", help="exit 4 at once when this file does not exist")
parser.add_argument("--record", help="write the working directory, stdin and environment here")
parser.add_argument("--say", help="print this line")
parser.add_argument("--exit", type=int, help="exit with this status, after --say")
parser.add_argument("--ignore-term", action="store_true", help="ignore SIGTERM")
parser.add_argument("--child", help="start a child that sleeps, and write its pid here")
parser.add_argument("--port", type=int, help="answer GET /health with 200 on this port")
parser.add_argument("--ready", help="create this file when ready")
args = parser.parse_args()

if args.require and not Path(args.require).exists():
    sys.exit(4)
if args.record:
    record = {"cwd": os.getcwd(), "stdin": os.readlink("/proc/self/fd/0"), "env": dict(os.environ)}
    Path(args.record).write_text(json.dumps(record))
if args.say:
    print(args.say, flush=True)
if args.exit is not None:
    sys.exit(args.exit)
if args.ignore_term:
    signal.signal(signal.SIGTERM, signal.SIG_IGN)
if args.child:
    Path(args.child).write_text(str(subprocess.Popen([sys.executable, __file__]).pid))


class Health(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200 if self.path == "/health" else 404)
        self.end_headers()

    def log_message(self, *args):
        pass


server = http.server.HTTPServer(("127.0.0.1", args.port), Health) if args.port else None
if args.ready:
    Path(args.ready).touch()
if server:
    server.serve_forever()
while True:
    time.sleep(60)
"""

_STUB = """#!%(python)s
import json, os, sys
from pathlib import Path
calls, answers = Path(%(calls)r), Path(%(answers)r)
with calls.open("a") as log:
    log.write(json.dumps({"argv": sys.argv[1:], "env": dict(os.environ)}) + "\\n")
known = json.loads(answers.read_text()) if answers.exists() else {}
first = sys.argv[1] if len(sys.argv) > 1 else ""
answer = known.get(first, known.get("", {"out": "", "err": "", "code": 0}))
sys.stdout.write(answer["out"])
sys.stderr.write(answer["err"])
sys.exit(answer["code"])
"""

#: The checks of a host that passes every check of the worker's doctor, by name: (status, detail).
HEALTHY_CHECKS = {
    "podman": ("ok", "podman 9.9.9 (/path/to/podman)"),
    "store": ("ok", "/path/to/store: btrfs, user_subvol_rm_allowed"),
    "runtime": ("ok", "/path/to/sprout-runtime"),
    "keyring": ("ok", "kernel.keys.maxkeys=20000"),
}


@dataclass(frozen=True)
class StubProgram:
    """An executable that records each call and gives the answer that the test set for it."""

    path: Path

    def answer(self, *, out: str = "", err: str = "", code: int = 0, when: str = "") -> None:
        """Answer each later call whose first argument is ``when``, or each call when it is empty."""
        known = json.loads(self._answers.read_text()) if self._answers.exists() else {}
        known[when] = {"out": out, "err": err, "code": code}
        self._answers.write_text(json.dumps(known))

    def calls(self) -> list[list[str]]:
        """The arguments of each call, without the program name, in call order."""
        return [entry["argv"] for entry in self._entries()]

    def environments(self) -> list[dict[str, str]]:
        """The environment of each call, in call order."""
        return [entry["env"] for entry in self._entries()]

    @property
    def _answers(self) -> Path:
        return self.path.with_name(self.path.name + ".answers.json")

    @property
    def _calls(self) -> Path:
        return self.path.with_name(self.path.name + ".calls.jsonl")

    def _entries(self) -> list[dict[str, Any]]:
        text = self._calls.read_text() if self._calls.exists() else ""
        return [json.loads(line) for line in text.splitlines()]

    def _install(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        files = {"python": sys.executable, "calls": str(self._calls), "answers": str(self._answers)}
        self.path.write_text(_STUB % files)
        self.path.chmod(0o755)


def make_stub(path: Path) -> StubProgram:
    """A stub program at ``path``. It answers each call with no output and status 0 until told otherwise."""
    stub = StubProgram(path)
    stub._install()
    return stub


def doctor_report(
    *,
    podman_bin: str = "/path/to/podman",
    store: str = "/path/to/store",
    checks: Mapping[str, tuple[str, str]] | None = None,
) -> str:
    """What ``python -m sprout.worker doctor --json`` prints: the healthy checks with ``checks`` over them."""
    merged = {**HEALTHY_CHECKS, **(checks or {})}
    entries = {name: {"status": status, "detail": detail} for name, (status, detail) in merged.items()}
    return json.dumps({"podman_bin": podman_bin, "store": store, "checks": entries}, indent=2) + "\n"


def service_command(tmp_path: Path, *options: str) -> list[str]:
    """The argv of SERVICE_PROGRAM with ``options``. The program is a file in ``tmp_path``."""
    program = tmp_path / "service.py"
    if not program.exists():
        program.write_text(SERVICE_PROGRAM)
    return [sys.executable, str(program), *options]


def ready_probe(ready: Path) -> Callable[[], Probe]:
    """A probe that waits up to 10 s for the service to create ``ready``, so that its first answer counts."""

    def probe() -> Probe:
        deadline = time.monotonic() + 10
        while not ready.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        return Probe(ready.exists(), "ready file")

    return probe


def free_port() -> int:
    """A loopback port that nothing listens on at the time of the call."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def write_spec(paths: Paths, services: Mapping[str, Any], *, python: str | None = None) -> None:
    """Write ``services.yaml`` with ``services`` in the order given, which is the order of the file."""
    paths.config.mkdir(parents=True, exist_ok=True)
    text = yaml.safe_dump({"python": python, "services": dict(services)}, sort_keys=False)
    paths.spec.write_text(text, encoding="utf-8")


def table_rows(table: str) -> list[list[str]]:
    """The cells of each line of a ``status`` table, without the padding."""
    return [[cell.strip() for cell in line.split(" | ")] for line in table.splitlines()]


def log_lines(log: Path) -> list[str]:
    """The lines of a service log, with the timestamp of each line of the CLI's own taken out."""
    return [re.sub(r"^\[services [^\]]*\]", "[services]", line) for line in log.read_text().splitlines()]


def is_running(pid: int) -> bool:
    """Whether ``pid`` is a live process, and not a zombie that waits for its parent."""
    try:
        status = Path("/proc/%d/status" % pid).read_text()
    except OSError:
        return False
    return "\nState:\tZ" not in status


def kill_processes_naming(text: str) -> None:
    """SIGKILL each process whose argv contains ``text``. Reap each one that is a child of this process."""
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            argv = (entry / "cmdline").read_bytes()
        except OSError:
            continue
        if text.encode() not in argv:
            continue
        try:
            os.kill(int(entry.name), signal.SIGKILL)
            os.waitpid(int(entry.name), 0)
        except (ProcessLookupError, PermissionError, ChildProcessError):
            pass

"""The container lifecycle against a fake ``podman`` (and ``pg_isready``) on PATH.

The real podman is never run. The fake records every argv it is called with,
keeps "the container" as one state file and answers ``inspect`` from it, so
what is asserted is exactly what the CLI hands podman for the template's own
``db`` service: the ``{state}`` mount expanded to a real path (podman reads a
source without a leading ``/`` as a named volume, and that is how ``up db``
once failed on its first real run), the password by ``-e NAME`` only, and a
stopped container started rather than created again. ``pg_isready`` is faked
too, so the probe never opens a socket.
"""

from __future__ import annotations

import json
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest
import yaml

from sprout.services import __main__ as cli
from sprout.services import secrets
from sprout.services.spec import template_path

from tests.services.conftest import write_spec

#: `%(python)s`, `%(record)r`, `%(state)r` are filled in by the fixture.
FAKE_PODMAN = '''#!%(python)s
"""A podman that only remembers: every argv in a record, one state file as the container."""
import json, os, sys
from pathlib import Path
record, state = Path(%(record)r), Path(%(state)r)
argv = sys.argv[1:]
watched = ("POSTGRES_USER", "POSTGRES_DB", "POSTGRES_PASSWORD")
with record.open("a") as fh:
    fh.write(json.dumps({"argv": argv, "env": {k: os.environ.get(k) for k in watched}}) + "\\n")
verb = argv[0]
if verb == "inspect":
    if not state.exists():
        sys.exit("Error: no such container")
    sys.stdout.write(state.read_text())
elif verb == "run":
    if state.exists():
        sys.exit("Error: the container name is already in use")
    state.write_text("running 4242\\n")
    print("deadbeef")
elif verb == "start":
    state.write_text("running 4242\\n")
elif verb == "stop":
    state.write_text("exited 0\\n")
else:
    sys.exit("Error: unknown verb %%s" %% verb)
'''

FAKE_PG_ISREADY = '''#!%(python)s
"""A pg_isready that answers from the fake container's state, never from a socket."""
import sys
from pathlib import Path
state = Path(%(state)r)
where = "%%s:%%s" %% (sys.argv[sys.argv.index("-h") + 1], sys.argv[sys.argv.index("-p") + 1])
if state.exists() and state.read_text().startswith("running"):
    print(where + " - accepting connections")
    sys.exit(0)
print(where + " - no response")
sys.exit(2)
'''


@dataclass(frozen=True)
class FakePodman:
    record: Path
    state: Path

    def calls(self) -> list[dict]:
        if not self.record.exists():
            return []
        return [json.loads(line) for line in self.record.read_text().splitlines()]

    def verbs(self) -> list[str]:
        return [call["argv"][0] for call in self.calls()]


@pytest.fixture
def fake_podman(paths, tmp_path, monkeypatch) -> FakePodman:
    """Both fakes first on PATH; `paths` has already moved every root to tmp_path."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    fake = FakePodman(tmp_path / "podman-calls.jsonl", tmp_path / "container-state")
    values = {"python": sys.executable, "record": str(fake.record), "state": str(fake.state)}
    for name, text in (("podman", FAKE_PODMAN), ("pg_isready", FAKE_PG_ISREADY)):
        script = bin_dir / name
        script.write_text(text % values, encoding="utf-8")
        script.chmod(0o755)
    monkeypatch.setenv("PATH", str(bin_dir) + os.pathsep + os.environ.get("PATH", ""))
    return fake


def _template_db() -> dict:
    """The shipped db service, verbatim: the test is about what the template does."""
    return yaml.safe_load(template_path("services.yaml").read_text(encoding="utf-8"))["services"]["db"]


def test_up_hands_podman_the_expanded_mount_and_the_password_by_name_only(paths, fake_podman, capsys):
    write_spec(paths, {"db": _template_db()})
    assert cli.main(["up", "db"]) == 0
    out = capsys.readouterr().out
    assert "db: creating container sprout-services-db from docker.io/library/postgres:16-alpine (podman run)" in out
    assert "db: up (container sprout-services-db, 127.0.0.1:15432 - accepting connections)" in out

    run = next(call for call in fake_podman.calls() if call["argv"][0] == "run")
    argv = run["argv"]
    data = paths.state / "db" / "data"
    assert argv[:4] == ["run", "-d", "--name", "sprout-services-db"] and argv[-1] == "docker.io/library/postgres:16-alpine"
    assert argv[argv.index("-p") + 1] == "127.0.0.1:15432:5432"
    assert argv[argv.index("-v") + 1] == "%s:/var/lib/postgresql/data:Z" % data, argv
    assert data.is_dir(), "the host side of the bind mount is made before podman sees it"
    assert [argv[i + 1] for i, token in enumerate(argv) if token == "-e"] == ["POSTGRES_USER", "POSTGRES_DB", "POSTGRES_PASSWORD"]
    password = secrets.require(paths, "db-password")
    assert run["env"] == {"POSTGRES_USER": "sprout_runstore", "POSTGRES_DB": "sprout_runstore", "POSTGRES_PASSWORD": password}
    joined = " ".join(argv)
    assert password not in joined and "{" not in joined, "placeholders expanded, the value in the environment only"
    log = paths.log_file("db").read_text()
    assert str(data) in log and "{state}" not in log and password not in log and password not in out

    assert cli.main(["status"]) == 0
    table = capsys.readouterr().out
    assert re.search(r"^db\s+\| podman\s+\| 4242\s+\| ok: 127\.0\.0\.1:15432 - accepting connections\s+\| 15432$", table, re.M), table
    assert password not in table


def test_a_stopped_container_is_started_not_created_again(paths, fake_podman, capsys):
    write_spec(paths, {"db": _template_db()})
    assert cli.main(["down", "db"]) == 0
    assert "db: container sprout-services-db does not exist" in capsys.readouterr().out
    assert cli.main(["up", "db"]) == 0 and cli.main(["up", "db"]) == 0
    assert capsys.readouterr().out.count("db: already healthy (127.0.0.1:15432 - accepting connections)") == 1

    assert cli.main(["down", "db"]) == 0
    out = capsys.readouterr().out
    assert "db: stopping container sprout-services-db (podman stop)" in out and "db: stopped" in out
    assert next(call["argv"] for call in fake_podman.calls() if call["argv"][0] == "stop") == ["stop", "--time", "20", "sprout-services-db"]
    assert cli.main(["status"]) == 0
    assert re.search(r"^db\s+\| podman\s+\| -\s+\| down: 127\.0\.0\.1:15432 - no response\s+\| 15432$", capsys.readouterr().out, re.M)

    assert cli.main(["up", "db"]) == 0
    assert "db: starting container sprout-services-db (podman start)" in capsys.readouterr().out
    assert [verb for verb in fake_podman.verbs() if verb != "inspect"] == ["run", "stop", "start"]

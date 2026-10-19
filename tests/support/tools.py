"""Fixtures that put a fake ``podman``, ``btrfs`` or ``findmnt`` first on PATH, and read what it was asked.

Each fake records each argv that it gets. A test can check the exact command line, and the order of
the calls across the three tools. The modules in ``tests/support/fakes/`` say what each fake answers.
The records and the state of the fakes are files in the test's ``tmp_path``. Each fixture also sets
``SPROUT_PODMAN_BIN=podman``, so that the code runs the podman on PATH and not an installed one. The
knobs, PATH and that variable go back to their earlier values when the test ends.
"""

from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

_FAKES = Path(__file__).resolve().parent / "fakes"
#: The variable that tells each fake where its records are.
_RECORDS_ENV = "FAKE_TOOLS_DIR"
_LAUNCHER = "#!%(python)s\nimport runpy\nrunpy.run_path(%(fake)r, run_name='__main__')\n"


def _json_lines(path: Path) -> list[Any]:
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def recorded_calls(records: Path, tool: str) -> list[list[str]]:
    """Each argv that the fake ``tool`` got, without the program name, in call order."""
    return _json_lines(records / ("%s.log" % tool))


def recorded_sequence(records: Path) -> list[list[str]]:
    """Each call to any of the fakes, as the tool name followed by the argv, in call order."""
    return _json_lines(records / "tools.log")


def recorded_cwds(records: Path) -> list[tuple[str, str]]:
    """The first argv word and the working directory of each call to the fake podman, in call order."""
    return [(command, cwd) for command, cwd in _json_lines(records / "podman.cwd.log")]


def fake_networks(records: Path) -> dict[str, dict[str, Any]]:
    """The networks that the fake podman holds now, by name: their labels, creation time and options."""
    state = records / "podman.networks.json"
    return json.loads(state.read_text()) if state.exists() else {}


@dataclass(frozen=True)
class ManagedContainer:
    """A container that the fake podman lists, inspects and knows as a sandbox of ``store``."""

    name: str
    store: Path
    #: The host directory that the container mounts at /run/sprout: the sandbox's socket directory.
    sockets: Path
    #: Seconds since the epoch, which `podman ps` reports as ``Created``.
    created: int


@dataclass(frozen=True)
class ManagedNetwork:
    """A network that the fake podman holds, from the first call on, as a sandbox network of ``store``."""

    name: str
    store: Path
    #: The creation time in podman's format, such as ``2023-11-14T22:13:20.123456789Z``.
    created: str


class FakePodman:
    """The fake podman of one test: what it was asked, and knobs that put it in a given state.

    A knob holds until the test ends, or until the next call of the same knob replaces its value."""

    def __init__(self, *, records: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.records = records
        self._monkeypatch = monkeypatch

    def calls(self) -> list[list[str]]:
        """Each argv that the fake podman got, without the program name, in call order."""
        return recorded_calls(self.records, "podman")

    def cwds(self) -> list[tuple[str, str]]:
        """The first argv word and the working directory of each call, in call order."""
        return recorded_cwds(self.records)

    def networks(self) -> dict[str, dict[str, Any]]:
        """The networks that the fake holds now, by name: their labels, creation time and options."""
        return fake_networks(self.records)

    def fail(self, *commands: str) -> None:
        """Make each of ``commands``, a first argv word such as ``rm``, exit 125 with an error."""
        self._monkeypatch.setenv("FAKE_PODMAN_FAIL", ",".join(commands))

    def hang(self, *commands: str) -> None:
        """Make each of ``commands`` sleep 30 s before it answers: longer than the timeout of a test."""
        self._monkeypatch.setenv("FAKE_PODMAN_HANG", ",".join(commands))

    def unpulled(self, *references: str) -> None:
        """Make `image inspect` of each of ``references`` fail with ``image not known`` until a `pull`."""
        self._monkeypatch.setenv("FAKE_PODMAN_UNKNOWN", ",".join(references))

    def image_user(self, user: str) -> None:
        """Give every image the OCI ``User`` ``user``, such as ``appuser``."""
        self._monkeypatch.setenv("FAKE_PODMAN_USER", user)

    def container_state(self, state: str) -> None:
        """Make `inspect --format` report ``state``, such as ``exited 127``, for every container."""
        self._monkeypatch.setenv("FAKE_PODMAN_STATE", state)

    def containers_exist(self, *containers: ManagedContainer) -> None:
        """Make `ps`, `inspect --format json` and `container exists` know ``containers`` until `rm`."""
        entries = [
            {"name": c.name, "store": str(c.store), "sockets": str(c.sockets), "created": c.created}
            for c in containers
        ]
        self._monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", json.dumps(entries))

    def networks_exist(self, *networks: ManagedNetwork) -> None:
        """Make the fake hold ``networks`` at its first call. Call it before the first podman call."""
        entries = [{"name": n.name, "store": str(n.store), "created": n.created} for n in networks]
        self._monkeypatch.setenv("FAKE_PODMAN_NETWORKS", json.dumps(entries))

    def refuse_network_changes(self) -> None:
        """Make `network connect` fail as already connected, and `network disconnect` as not connected."""
        self._monkeypatch.setenv("FAKE_PODMAN_NET_REFUSE", "1")

    def lose_containers(self) -> None:
        """Make `rm` fail as podman does for a container that does not exist: ``no such container``."""
        self._monkeypatch.setenv("FAKE_PODMAN_GONE", "1")


class FakeBtrfs:
    """The fake btrfs of one test: what it was asked, and a knob that makes it fail."""

    def __init__(self, *, records: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.records = records
        self._monkeypatch = monkeypatch

    def calls(self) -> list[list[str]]:
        """Each argv that the fake btrfs got, without the program name, in call order."""
        return recorded_calls(self.records, "btrfs")

    def fail(self, *subcommands: str) -> None:
        """Make each of ``subcommands``, such as ``snapshot`` or ``delete``, exit 1 with an error."""
        self._monkeypatch.setenv("FAKE_BTRFS_FAIL", ",".join(subcommands))


class FakeFindmnt:
    """The fake findmnt of one test: what it was asked, and a knob that sets the mount that it reports."""

    def __init__(self, *, records: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        self.records = records
        self._monkeypatch = monkeypatch

    def calls(self) -> list[list[str]]:
        """Each argv that the fake findmnt got, without the program name, in call order."""
        return recorded_calls(self.records, "findmnt")

    def report(self, *, fstype: str, options: str) -> None:
        """Make it report that every path is on a mount of ``fstype`` with ``options``."""
        self._monkeypatch.setenv("FAKE_FINDMNT_FSTYPE", fstype)
        self._monkeypatch.setenv("FAKE_FINDMNT_OPTIONS", options)


def _install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, tool: str) -> None:
    """Put a launcher of the fake ``tool`` in ``tmp_path/bin``, first on PATH, recording in ``tmp_path``."""
    launcher = tmp_path / "bin" / tool
    launcher.parent.mkdir(exist_ok=True)
    launcher.write_text(_LAUNCHER % {"python": sys.executable, "fake": str(_FAKES / ("%s.py" % tool))})
    launcher.chmod(0o755)
    monkeypatch.setenv(_RECORDS_ENV, str(tmp_path))
    # default_podman_bin() prefers $SPROUT_PODMAN_BIN and an installed podman 6 to PATH.
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "podman")
    path = os.environ["PATH"]
    if not path.startswith(str(launcher.parent) + os.pathsep):
        monkeypatch.setenv("PATH", "%s%s%s" % (launcher.parent, os.pathsep, path))


@pytest.fixture
def fake_podman(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakePodman:
    """The fake podman, first on PATH, with its records in ``tmp_path``."""
    _install(tmp_path, monkeypatch, "podman")
    return FakePodman(records=tmp_path, monkeypatch=monkeypatch)


@pytest.fixture
def fake_btrfs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeBtrfs:
    """The fake btrfs, first on PATH, with its records in ``tmp_path``."""
    _install(tmp_path, monkeypatch, "btrfs")
    return FakeBtrfs(records=tmp_path, monkeypatch=monkeypatch)


@pytest.fixture
def fake_findmnt(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeFindmnt:
    """The fake findmnt, first on PATH, with its records in ``tmp_path``."""
    _install(tmp_path, monkeypatch, "findmnt")
    return FakeFindmnt(records=tmp_path, monkeypatch=monkeypatch)


@pytest.fixture
def fake_btrfs_store(tmp_path: Path, fake_btrfs: FakeBtrfs, fake_findmnt: FakeFindmnt) -> Path:
    """An empty store directory, ``tmp_path/store``, that ``BtrfsStore`` accepts with the fakes on PATH."""
    path = tmp_path / "store"
    path.mkdir()
    return path


@pytest.fixture
def fake_runtime_bin(tmp_path: Path) -> Path:
    """A file in the place of the runtime binary. It exists, and nothing can run it."""
    path = tmp_path / "sprout-runtime"
    path.write_bytes(b"\x7fELF fake runtime")
    return path

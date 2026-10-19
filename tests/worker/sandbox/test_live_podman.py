"""A ``SandboxSession`` on the real podman backend: shape, image config, branches, network and teardown.

Large tests: they need rootless podman, a btrfs store at ``$SPROUT_PODMAN_STORE`` that no run uses,
and the runtime binary. Each test skips with what is missing. The fixture destroys each session that
a test made and each snapshot that it took, and fails the test if a container, a network, a rootfs or
a socket directory is left. Run the file alone, while the store is idle::

    pytest tests/worker/sandbox/test_live_podman.py -m slow
"""

from __future__ import annotations

import asyncio
import json
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from sprout.worker.sandbox.commands import checked, shell
from sprout.worker.sandbox.session import SandboxSession
from sprout_sandbox import PodmanPool
from sprout_sandbox.podman_cli import default_podman_bin
from tests.support.live import sandbox_networks
from tests.support.paths import RUNTIME_BIN

pytestmark = pytest.mark.slow

IMAGE = "docker.io/library/python:3.11-slim"
#: 300 MiB and one CPU, as podman reports them: ``HostConfig.Memory`` in bytes, ``NanoCpus``.
SHAPE = {"cpu": 1, "memory_mb": 300}
REPORTED_SHAPE = "314572800 1000000000"


def podman(*args: str) -> str:
    return subprocess.run([default_podman_bin(), *args], capture_output=True, text=True, check=True).stdout


def entries(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()} if directory.is_dir() else set()


def shape_of(sandbox_id: str) -> str:
    return podman(
        "inspect",
        "--type",
        "container",
        "--format",
        "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}",
        sandbox_id,
    ).strip()


def networks_of(sandbox_id: str) -> list[str]:
    """The networks that the sandbox's container is connected to."""
    return sorted(
        json.loads(
            podman(
                "inspect", "--type", "container", "--format", "{{json .NetworkSettings.Networks}}", sandbox_id
            )
        )
        or {}
    )


def env_of(config: dict[str, list[str]]) -> dict[str, str]:
    """The ``Env`` of an OCI config, by name."""
    return dict(entry.split("=", 1) for entry in config["Env"])


class Sessions:
    """The sessions of one test, on the live store. ``new`` makes one."""

    def __init__(self, store: Path) -> None:
        self.store = store
        self.made: list[SandboxSession] = []

    def new(self, image: str = IMAGE, resources: dict[str, int] | None = None) -> SandboxSession:
        session = SandboxSession(
            quiet=True, backend={"podman": {"store": str(self.store), "runtime_bin": str(RUNTIME_BIN)}}
        )
        self.made.append(session)
        assert session.create(image, resources), session.create_error
        return session


@pytest.fixture
def sessions(live_podman_store: Path) -> Iterator[Sessions]:
    """Destroy each session and each snapshot of the test, then fail if anything else was left."""
    if not RUNTIME_BIN.is_file():
        pytest.skip(
            "needs the runtime binary: cd runtime && go generate ./... && go build -o sprout-runtime ."
        )
    kinds = ("sandboxes", "snapshots", "sockets")
    before = {kind: entries(live_podman_store / kind) for kind in kinds}
    containers, networks = set(podman("ps", "-a", "-q", "--no-trunc").split()), sandbox_networks()
    made = Sessions(live_podman_store)
    yield made
    for session in made.made:
        session.destroy()
    pool = PodmanPool(live_podman_store, RUNTIME_BIN)
    for name in sorted(entries(live_podman_store / "snapshots") - before["snapshots"]):
        if not name.endswith(".json"):
            asyncio.run(pool.delete_snapshot(name))
    left = {kind: entries(live_podman_store / kind) - before[kind] for kind in kinds}
    leaked = (set(podman("ps", "-a", "-q", "--no-trunc").split()) - containers, sandbox_networks() - networks)
    assert (left, leaked) == ({kind: set() for kind in kinds}, (set(), set())), (
        "the sessions left these behind"
    )


def test_a_sandbox_has_the_shape_of_its_task_and_the_env_and_config_of_its_image(sessions):
    session = sessions.new(resources=SHAPE)

    config = session.image_config(None)

    assert shape_of(session.sandbox_id) == REPORTED_SHAPE
    assert checked(session, 'printf %s "$PYTHON_VERSION"') == env_of(config)["PYTHON_VERSION"]
    assert (config["User"], config["Cmd"]) == (None, ["python3"])


def test_a_snapshot_is_disk_only_and_a_memory_snapshot_is_declined(sessions):
    """A btrfs snapshot holds files, not processes. Nothing stacks, so it reports no layer count."""
    session = sessions.new()

    snapshot = session.snapshot()
    memory = session.snapshot(disk_only=False)

    assert (snapshot.id, snapshot.disk_only, snapshot.rootfs_layers, memory) == (
        session.sandbox_id + "-0001",
        True,
        None,
        None,
    )


def test_a_branch_has_its_parents_files_at_the_snapshot_and_keeps_its_own_writes(sessions):
    """A subvolume carries no shape, so the branch is shaped again."""
    parent = sessions.new(resources=SHAPE)
    checked(parent, "echo before > /tmp/before")
    snapshot = parent.snapshot()
    checked(parent, "echo after > /tmp/after")

    branch = sessions.new(snapshot.id, SHAPE)
    checked(branch, "echo branch > /tmp/branch")

    assert shape_of(branch.sandbox_id) == REPORTED_SHAPE
    assert checked(branch, "cat /tmp/before") == "before\n"
    assert (shell(branch, "test -e /tmp/after")[2], shell(parent, "test -e /tmp/branch")[2]) == (1, 1)


def test_a_running_sandbox_goes_offline_and_back_online(sessions):
    session = sessions.new()

    session.set_network(False)
    offline = networks_of(session.sandbox_id)
    session.set_network(True)

    assert (offline, networks_of(session.sandbox_id)) == ([], [session.sandbox_id])
    assert checked(session, "echo still-answers") == "still-answers\n"

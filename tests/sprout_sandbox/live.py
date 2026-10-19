"""Helpers for the large tests on the real podman, btrfs-progs and btrfs store of this host.

``Leftovers`` takes stock of a store and of podman, and removes what appeared after the stock was
taken: the containers, networks, sandboxes and snapshots that a failed test left. ``shell`` and
``checked`` run a command through a sandbox's runtime.
"""

from __future__ import annotations

import subprocess
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from sprout.worker.run.reap import PodmanClient
from sprout_sandbox import BtrfsStore, Sandbox
from sprout_sandbox.podman_cli import default_podman_bin
from sprout_sandbox.podman_images import image_info
from tests.support.live import sandbox_networks

#: The image of most large tests. The first run pulls it and extracts its base, which stays in the store.
LIVE_IMAGE = "docker.io/library/python:3.11-slim"
#: How a large test that needs the runtime binary skips when it is not built.
NO_RUNTIME = "needs runtime/sprout-runtime (cd runtime && go generate ./... && go build -o sprout-runtime .)"


def podman(*args: str, stdin: str | None = None) -> str:
    """The stdout of the podman that the pool runs, with ``args``. Raises if it exits non-zero."""
    completed = subprocess.run(
        [default_podman_bin(), *args], input=stdin, capture_output=True, text=True, check=True, timeout=1800
    )
    return completed.stdout


def all_containers() -> set[str]:
    """The id of every container of this user, sprout-managed or not, so that an unlabeled leak counts too."""
    return set(podman("ps", "-a", "--no-trunc", "-q").split())


def entries(directory: Path) -> set[str]:
    """The names in ``directory``, or none if it does not exist."""
    return {path.name for path in directory.iterdir()} if directory.is_dir() else set()


@dataclass(frozen=True)
class Leftovers:
    """What exists now: the containers, the sandbox networks, and the sandboxes, snapshots and sockets."""

    containers: frozenset[str]
    networks: frozenset[str]
    sandboxes: frozenset[str]
    snapshots: frozenset[str]
    sockets: frozenset[str]

    @classmethod
    def now(cls, store: Path) -> Leftovers:
        return cls(
            containers=frozenset(all_containers()),
            networks=frozenset(sandbox_networks()),
            sandboxes=frozenset(entries(store / "sandboxes")),
            snapshots=frozenset(entries(store / "snapshots")),
            sockets=frozenset(entries(store / "sockets")),
        )

    def appeared_since(self, earlier: Leftovers) -> Leftovers:
        """What exists now and did not exist at ``earlier``."""
        return Leftovers(
            containers=self.containers - earlier.containers,
            networks=self.networks - earlier.networks,
            sandboxes=self.sandboxes - earlier.sandboxes,
            snapshots=self.snapshots - earlier.snapshots,
            sockets=self.sockets - earlier.sockets,
        )


def nothing() -> Leftovers:
    """No container, network, sandbox, snapshot or socket directory."""
    empty: frozenset[str] = frozenset()
    return Leftovers(containers=empty, networks=empty, sandboxes=empty, snapshots=empty, sockets=empty)


async def remove_leftovers(store: Path, earlier: Leftovers) -> None:
    """Remove each container, network, sandbox and snapshot that appeared in ``store`` after ``earlier``."""
    appeared = Leftovers.now(store).appeared_since(earlier)
    for container in appeared.containers:
        subprocess.run([default_podman_bin(), "rm", "-f", container], capture_output=True, timeout=120)
    for sandbox_id in appeared.sandboxes | appeared.networks:
        PodmanClient(stores=[str(store)]).delete_sandbox(sandbox_id.removesuffix(".json"))
    btrfs_store = BtrfsStore(store)
    for snapshot_id in appeared.snapshots:
        if not snapshot_id.endswith(".json") and (store / "snapshots" / snapshot_id).is_dir():
            await btrfs_store.delete_snapshot(snapshot_id)


async def shell(sandbox: Sandbox, command: str) -> tuple[str, int]:
    """The stdout and the exit code of ``command``, run through the sandbox's runtime."""
    result = await sandbox.call("shell", command=command, timeout=30)
    # A plain success carries bare stdout and no outcome.
    if result.exit_code is None and not result.running and not result.is_error:
        return result.output, 0
    return result.stdout or "", -1 if result.exit_code is None else result.exit_code


async def checked(sandbox: Sandbox, command: str) -> str:
    """The stdout of ``command``, stripped. Raises ``AssertionError`` with the stderr unless it exits 0."""
    result = await sandbox.call("shell", command=command, timeout=30)
    if result.exit_code is None and not result.running and not result.is_error:
        return result.output.strip()
    assert result.exit_code == 0, (command, result.exit_code, result.stdout, result.stderr)
    return (result.stdout or "").strip()


async def image_id(image: str) -> str:
    """The id of the local image ``image``, the name of its base in a store."""
    record = await image_info(image, podman_bin=default_podman_bin())
    return str(record["Id"])


@asynccontextmanager
async def built_image(tag: str, dockerfile: str, *, context: Path, store: Path) -> AsyncIterator[str]:
    """The local image ``tag``, built from ``dockerfile``. Its base and the image go after the block."""
    podman("build", "-q", "-t", tag, "-f", "-", str(context), stdin=dockerfile)
    try:
        yield tag
    finally:
        base_store = BtrfsStore(store)
        await base_store.discard(base_store.bases / await image_id(tag))
        subprocess.run([default_podman_bin(), "rmi", "-f", tag], capture_output=True, timeout=300)

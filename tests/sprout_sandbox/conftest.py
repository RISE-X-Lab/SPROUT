"""Fixtures for the pool tests: a store at a short path and a fake runtime in each sandbox, or the real tools.

The fixtures of the large tests skip with a reason when the real store, podman or runtime binary
cannot be used. ``live_pools`` closes each pool it made, and removes what a failed test left behind.
"""

from __future__ import annotations

import shutil
import tempfile
from collections.abc import AsyncIterator, Callable, Iterator
from pathlib import Path
from typing import Any

import pytest

from sprout_sandbox import PodmanPool
from tests.sprout_sandbox.fake_runtime import SandboxRuntimes
from tests.sprout_sandbox.live import NO_RUNTIME, Leftovers, remove_leftovers
from tests.support.paths import RUNTIME_BIN
from tests.support.tools import FakeBtrfs, FakeFindmnt, FakePodman


@pytest.fixture
def podman_store(fake_btrfs: FakeBtrfs, fake_findmnt: FakeFindmnt) -> Iterator[Path]:
    """An empty store that ``BtrfsStore`` accepts with the fakes on PATH, removed after the test."""
    # A Unix socket path has at most 107 bytes. Under tmp_path, <store>/sockets/<id>/runtime.sock can
    # be longer, and the fake runtime of a sandbox could not bind its socket.
    path = Path(tempfile.mkdtemp(prefix="store-")).resolve()
    yield path
    shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def runtimes(podman_store: Path) -> Iterator[SandboxRuntimes]:
    """A fake runtime that answers on the socket of each sandbox that a pool on ``podman_store`` starts."""
    started = SandboxRuntimes(podman_store / "sockets")
    yield started
    started.close()


@pytest.fixture
def pool(
    fake_podman: FakePodman, podman_store: Path, fake_runtime_bin: Path, runtimes: SandboxRuntimes
) -> PodmanPool:
    """An online pool on ``podman_store``, whose sandboxes have fake runtimes that answer."""
    return PodmanPool(podman_store, fake_runtime_bin)


@pytest.fixture
def live_runtime_bin() -> Path:
    """The runtime binary of the repository, which a large test mounts into each container."""
    if not RUNTIME_BIN.is_file():
        pytest.skip(NO_RUNTIME)
    return RUNTIME_BIN


@pytest.fixture
async def live_pools(
    live_podman_store: Path, live_runtime_bin: Path
) -> AsyncIterator[Callable[..., PodmanPool]]:
    """A maker of pools on the real store. After the test, it closes them and removes what is left."""
    before = Leftovers.now(live_podman_store)
    made: list[PodmanPool] = []

    def make(**options: Any) -> PodmanPool:
        made.append(PodmanPool(live_podman_store, live_runtime_bin, **options))
        return made[-1]

    yield make
    try:
        for pool in made:
            await pool.close()
    finally:
        await remove_leftovers(live_podman_store, before)

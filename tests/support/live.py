"""Checks for the large tests that use the real podman, btrfs-progs and the btrfs store of this host.

The fixtures of a large podman test remove each container, network and subvolume that appears while
the test runs. Beside an experiment on the same store, they would remove the experiment's sandboxes.
So a large test skips unless ``live_podman_problem`` finds nothing. The ``live_podman_store`` fixture
does the check and the skip. ``podman_store_in_use`` is the check that a store is busy.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sprout.contract.backend import default_podman_store
from sprout_sandbox.btrfs_store import store_problem
from sprout_sandbox.podman_cli import default_podman_bin


def live_store() -> Path:
    """The store that the large tests use: ``$SPROUT_PODMAN_STORE``, else ``~/sprout-btrfs``."""
    return Path(default_podman_store()).expanduser()


def podman_store_in_use(store: Path) -> str | None:
    """Why ``store`` is busy, or ``None``: it has a live sandbox, or a sprout-managed container exists.

    The podman of ``default_podman_bin()`` answers. A podman that cannot run counts as no container."""
    sandboxes = store / "sandboxes"
    live = sorted(p.name for p in sandboxes.iterdir() if p.is_dir()) if sandboxes.is_dir() else []
    if live:
        return "the store has live sandboxes (%s...)" % ", ".join(live[:3])
    try:
        managed = subprocess.run(
            [default_podman_bin(), "ps", "-a", "-q", "--filter", "label=sprout.managed=1"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        managed = []
    if managed:
        return "%d sprout-managed containers exist" % len(managed)
    return None


def sandbox_networks() -> set[str]:
    """The names of the networks that carry the pool's label ``sprout.managed=1``, for a leak check."""
    try:
        listed = subprocess.run(
            [default_podman_bin(), "network", "ls", "-q", "--filter", "label=sprout.managed=1"],
            capture_output=True,
            text=True,
            timeout=60,
        ).stdout.split()
    except (OSError, subprocess.TimeoutExpired):
        listed = []
    return set(listed)


def live_podman_problem(store: Path) -> str | None:
    """Why a large test cannot use the real podman, btrfs and ``store`` now, and what to do, or ``None``."""
    if shutil.which(default_podman_bin()) is None:
        return "needs podman on PATH or in $SPROUT_PODMAN_BIN (ops/setup_podman.sh installs podman 6)"
    problem = store_problem(store)
    if problem:
        return "needs a btrfs store at $SPROUT_PODMAN_STORE: %s" % problem
    busy = podman_store_in_use(store)
    if busy:
        return "the store is in use: %s. A large test removes what appears in it while the test runs" % busy
    return None


@pytest.fixture
def live_podman_store() -> Path:
    """The store of the large tests, if ``live_podman_problem`` finds nothing. Else the test skips with it."""
    store = live_store()
    problem = live_podman_problem(store)
    if problem:
        pytest.skip(problem)
    return store

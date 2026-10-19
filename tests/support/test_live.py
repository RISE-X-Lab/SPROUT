"""A large test skips while an experiment uses the store, or while there is no store."""

from __future__ import annotations

from pathlib import Path

from tests.support.live import live_podman_problem, podman_store_in_use
from tests.support.tools import ManagedContainer


def sandbox_container(name: str, store: Path) -> ManagedContainer:
    return ManagedContainer(name=name, store=store, sockets=store / "sockets" / name, created=1700000000)


def test_a_store_with_a_sandbox_directory_is_in_use(fake_podman, fake_btrfs_store):
    (fake_btrfs_store / "sandboxes" / "sprout-000000000001").mkdir(parents=True)

    assert podman_store_in_use(fake_btrfs_store) == "the store has live sandboxes (sprout-000000000001...)"


def test_a_store_beside_sprout_managed_containers_is_in_use(fake_podman, fake_btrfs_store):
    (fake_btrfs_store / "sandboxes").mkdir()
    fake_podman.containers_exist(
        sandbox_container("sprout-000000000001", fake_btrfs_store),
        sandbox_container("sprout-000000000002", fake_btrfs_store),
    )

    assert podman_store_in_use(fake_btrfs_store) == "2 sprout-managed containers exist"


def test_a_store_without_sandboxes_or_managed_containers_is_not_in_use(fake_podman, fake_btrfs_store):
    (fake_btrfs_store / "sandboxes").mkdir()

    assert podman_store_in_use(fake_btrfs_store) is None


def test_a_busy_store_stops_a_large_test(fake_podman, fake_btrfs_store):
    (fake_btrfs_store / "sandboxes" / "sprout-000000000001").mkdir(parents=True)

    assert live_podman_problem(fake_btrfs_store) == (
        "the store is in use: the store has live sandboxes (sprout-000000000001...). "
        "A large test removes what appears in it while the test runs"
    )


def test_a_missing_store_stops_a_large_test_with_how_to_make_one(fake_podman, fake_btrfs_store):
    absent = fake_btrfs_store / "absent"

    assert live_podman_problem(absent) == (
        "needs a btrfs store at $SPROUT_PODMAN_STORE: store does not exist: %s "
        "(create one with sudo ops/btrfs_store.sh <mount point> <image file> <size>)" % absent
    )

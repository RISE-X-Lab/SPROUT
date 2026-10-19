"""How a worker establishes that the sandboxes an attempt claimed are gone, before it ends the job.

``podman_reclaimed`` runs against the fake podman, btrfs and findmnt of ``tests/support``.
``microvm_reclaimed`` runs against a stand-in AgentENV on a local port.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sprout.worker.attempts import microvm_reclaimed, podman_reclaimed
from tests.support.fakes.podman import NETWORK_CREATED
from tests.support.http import serving
from tests.support.tools import ManagedContainer, ManagedNetwork
from tests.worker.agentenv import AgentEnv

SANDBOX = "sprout-0000000000a1"


def podman_backend(store: Path) -> dict[str, Any]:
    return {"backend": "podman", "podman": {"store": str(store)}}


def leftover_rootfs(store: Path, sandbox: str) -> Path:
    """The rootfs subvolume of ``sandbox`` in ``store``, as the fake btrfs keeps it: a directory."""
    rootfs = store / "sandboxes" / sandbox
    rootfs.mkdir(parents=True)
    return rootfs


def test_a_claimed_sandbox_that_is_already_gone_counts_as_reclaimed_without_a_removal(
    fake_podman, fake_btrfs_store
):
    reclaimed = podman_reclaimed({SANDBOX: None}, podman_backend(fake_btrfs_store))

    assert (reclaimed, ["rm", "-f", SANDBOX] in fake_podman.calls()) == (True, False)


def test_a_leftover_sandbox_is_removed_with_its_network_rootfs_and_socket_directory(
    fake_podman, fake_btrfs_store
):
    rootfs = leftover_rootfs(fake_btrfs_store, SANDBOX)
    sockets = fake_btrfs_store / "sockets" / SANDBOX
    sockets.mkdir(parents=True)
    fake_podman.containers_exist(
        ManagedContainer(name=SANDBOX, store=fake_btrfs_store, sockets=sockets, created=1700000000)
    )
    fake_podman.networks_exist(ManagedNetwork(name=SANDBOX, store=fake_btrfs_store, created=NETWORK_CREATED))

    reclaimed = podman_reclaimed({SANDBOX: str(fake_btrfs_store)}, podman_backend(fake_btrfs_store))

    assert (reclaimed, rootfs.exists(), sockets.exists(), fake_podman.networks()) == (True, False, False, {})
    assert ["rm", "-f", SANDBOX] in fake_podman.calls()


def test_a_leftover_sandbox_that_podman_cannot_remove_is_not_reclaimed(fake_podman, fake_btrfs_store):
    sockets = fake_btrfs_store / "sockets" / SANDBOX
    fake_podman.containers_exist(
        ManagedContainer(name=SANDBOX, store=fake_btrfs_store, sockets=sockets, created=1700000000)
    )
    fake_podman.fail("rm")

    reclaimed = podman_reclaimed({SANDBOX: str(fake_btrfs_store)}, podman_backend(fake_btrfs_store))

    assert reclaimed is False


def test_a_rootfs_of_a_claim_without_a_store_is_removed_from_the_store_of_the_backend(
    fake_podman, fake_btrfs_store
):
    """A grader claims its sandbox without a store. Its rootfs is in the store of the job's backend."""
    rootfs = leftover_rootfs(fake_btrfs_store, SANDBOX)

    reclaimed = podman_reclaimed({SANDBOX: None}, podman_backend(fake_btrfs_store))

    assert (reclaimed, rootfs.exists()) == (True, False)


def test_a_rootfs_is_removed_from_the_store_that_its_claim_names(fake_podman, fake_btrfs_store, tmp_path):
    rootfs = leftover_rootfs(fake_btrfs_store, SANDBOX)
    other_store = tmp_path / "other-store"
    other_store.mkdir()

    reclaimed = podman_reclaimed({SANDBOX: str(fake_btrfs_store)}, podman_backend(other_store))

    assert (reclaimed, rootfs.exists()) == (True, False)


@pytest.mark.parametrize(
    "delete, get, reclaimed",
    [(204, 404, True), (404, 404, True), (500, 404, False), (204, 200, False)],
    ids=["deleted", "already gone", "delete refused", "still there after the delete"],
)
def test_a_microvm_is_reclaimed_only_when_agentenv_deleted_it_and_then_no_longer_knows_it(
    delete, get, reclaimed
):
    agentenv = AgentEnv(delete=delete, get=get)

    with serving(agentenv) as url:
        answer = microvm_reclaimed(["vm-1"], {"server_url": url, "api_key": "aenv-key"})

    assert answer is reclaimed


def test_the_microvm_check_deletes_each_claimed_sandbox_with_the_api_key_and_then_asks_for_it():
    agentenv = AgentEnv(delete=204, get=404)

    with serving(agentenv) as url:
        microvm_reclaimed(["vm-1", "vm-2"], {"server_url": url + "/", "api_key": "aenv-key"})

    assert agentenv.requests == [
        ("DELETE", "/sandboxes/vm-1", "aenv-key"),
        ("GET", "/sandboxes/vm-1", "aenv-key"),
        ("DELETE", "/sandboxes/vm-2", "aenv-key"),
        ("GET", "/sandboxes/vm-2", "aenv-key"),
    ]


@pytest.mark.parametrize(
    "microvm", [{}, {"server_url": "http://127.0.0.1:1"}], ids=["no server", "unreachable"]
)
def test_a_microvm_cannot_be_reclaimed_without_an_agentenv_that_answers(microvm):
    assert microvm_reclaimed(["vm-1"], microvm) is False

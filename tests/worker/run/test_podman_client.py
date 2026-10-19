"""The podman client of the reaper: what of a sandbox it finds and how it removes it, on the fake tools.

A podman sandbox is a container, a network, a rootfs subvolume in a store and a socket directory.
The client must find each of them also when the others are gone, and remove the container first,
because its rootfs is mounted while it exists. ``tests/support/test_fakes.py`` checks the fakes
against the real podman and btrfs.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from sprout.worker.run.reap import PodmanClient, Reaper
from sprout.worker.run.resources import ResourceLedger
from tests.support.tools import FakeBtrfs, FakePodman, ManagedContainer, ManagedNetwork, recorded_sequence

NAME = "sprout-000000000001"
#: 2023-11-14T22:13:20 UTC.
CREATED = 1_700_000_000
NETWORK_CREATED = "2023-11-14T22:13:20.123456789Z"


def leaked_sandbox(store: Path, name: str = NAME) -> ManagedContainer:
    """The traces of sandbox ``name`` in ``store``: its rootfs, the rootfs's config and its sockets."""
    (store / "sandboxes" / name).mkdir(parents=True)
    (store / "sandboxes" / (name + ".json")).write_text("{}")
    sockets = store / "sockets" / name
    sockets.mkdir(parents=True)
    (sockets / "runtime.sock").write_text("")
    return ManagedContainer(name=name, store=store, sockets=sockets, created=CREATED)


def traces(store: Path, name: str = NAME) -> list[bool]:
    """Whether the rootfs, its config and the socket directory of sandbox ``name`` exist, in that order."""
    return [
        (store / "sandboxes" / name).exists(),
        (store / "sandboxes" / (name + ".json")).exists(),
        (store / "sockets" / name).exists(),
    ]


def test_the_client_lists_each_managed_container_dated_by_its_creation(
    fake_podman: FakePodman, fake_btrfs_store
):
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))

    sandboxes = PodmanClient().list_sandboxes()

    assert sandboxes == [{"id": NAME, "created_at": "2023-11-14T22:13:20+00:00", "names": []}]


def test_the_client_asks_podman_for_the_managed_containers_and_networks_as_json(fake_podman: FakePodman):
    PodmanClient().list_sandboxes()

    assert fake_podman.calls() == [
        ["ps", "-a", "--filter", "label=sprout.managed=1", "--format", "json"],
        ["network", "ls", "--filter", "label=sprout.managed=1", "--format", "json"],
    ]


def test_a_rootfs_without_a_container_is_listed_from_a_known_store_dated_by_its_directory(
    fake_podman: FakePodman, fake_btrfs_store
):
    """A destroy that failed after ``podman rm -f`` leaves the rootfs, and no label finds it."""
    leaked_sandbox(fake_btrfs_store)
    os.utime(fake_btrfs_store / "sandboxes" / NAME, (CREATED, CREATED))

    with_store = PodmanClient(stores=[str(fake_btrfs_store)]).list_sandboxes()
    without_store = PodmanClient().list_sandboxes()

    assert (with_store, without_store) == (
        [{"id": NAME, "created_at": "2023-11-14T22:13:20+00:00", "names": []}],
        [],
    )


def test_a_network_without_a_container_is_listed_dated_by_the_network(
    fake_podman: FakePodman, fake_btrfs_store
):
    """A run killed between ``network create`` and ``run`` leaves only the network, which has labels."""
    fake_podman.networks_exist(ManagedNetwork(name=NAME, store=fake_btrfs_store, created=NETWORK_CREATED))

    sandboxes = PodmanClient().list_sandboxes()

    assert sandboxes == [{"id": NAME, "created_at": "2023-11-14T22:13:20.123456+00:00", "names": []}]


def test_a_sandbox_is_listed_once_when_its_container_rootfs_and_network_all_exist(
    fake_podman: FakePodman, fake_btrfs_store
):
    fake_podman.networks_exist(ManagedNetwork(name=NAME, store=fake_btrfs_store, created=NETWORK_CREATED))
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))

    sandboxes = PodmanClient(stores=[str(fake_btrfs_store)]).list_sandboxes()

    assert [sandbox["id"] for sandbox in sandboxes] == [NAME]


def test_a_host_without_podman_lists_nothing_and_has_nothing_to_delete(tmp_path):
    client = PodmanClient(podman_bin=str(tmp_path / "no-such-podman"))

    assert (client.list_sandboxes(), client.delete_sandbox(NAME)) == ([], True)


def test_deleting_a_sandbox_removes_its_container_network_rootfs_and_socket_directory(
    fake_podman: FakePodman, fake_btrfs_store
):
    fake_podman.networks_exist(ManagedNetwork(name=NAME, store=fake_btrfs_store, created=NETWORK_CREATED))
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))
    client = PodmanClient()

    deleted = client.delete_sandbox(NAME)

    assert (deleted, client.leftovers(NAME), traces(fake_btrfs_store)) == (True, [], [False, False, False])


def test_deleting_a_sandbox_removes_the_container_before_its_network_and_its_rootfs(
    tmp_path, fake_podman: FakePodman, fake_btrfs_store
):
    """The container's labels and mounts locate the rest of the sandbox, and it mounts the rootfs."""
    fake_podman.networks_exist(ManagedNetwork(name=NAME, store=fake_btrfs_store, created=NETWORK_CREATED))
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))
    rootfs = fake_btrfs_store.resolve() / "sandboxes" / NAME

    PodmanClient().delete_sandbox(NAME)

    sequence = recorded_sequence(tmp_path)
    assert fake_podman.calls()[:4] == [
        ["inspect", "--type", "container", "--format", "json", NAME],
        ["rm", "-f", NAME],
        ["network", "exists", NAME],
        ["network", "rm", NAME],
    ]
    assert sequence.index(["podman", "rm", "-f", NAME]) < sequence.index(
        ["btrfs", "subvolume", "delete", str(rootfs)]
    )


def test_a_rootfs_whose_container_is_gone_is_deleted_from_a_known_store_with_a_subvolume_made_inside_it(
    fake_podman: FakePodman, fake_btrfs_store
):
    leaked_sandbox(fake_btrfs_store)
    nested = fake_btrfs_store / "sandboxes" / NAME / "adv" / "nested"
    nested.parent.mkdir()
    subprocess.run(["btrfs", "subvolume", "create", str(nested)], check=True, capture_output=True)

    deleted = PodmanClient(stores=[str(fake_btrfs_store)]).delete_sandbox(NAME)

    assert (deleted, traces(fake_btrfs_store)) == (True, [False, False, False])
    assert ["rm", "-f", NAME] not in fake_podman.calls()


def no_traces(store: Path) -> None:
    """What a run killed between ``network create`` and ``run`` leaves in the store: nothing."""


@pytest.mark.parametrize(
    "leave", [pytest.param(leaked_sandbox, id="rm-by-hand"), pytest.param(no_traces, id="network-alone")]
)
def test_a_sandbox_whose_container_was_removed_by_hand_is_found_through_the_store_label_of_its_network(
    fake_podman: FakePodman, fake_btrfs_store, leave
):
    """No store is given: the network's ``sprout.store`` label says where the rest of the sandbox is."""
    fake_podman.networks_exist(ManagedNetwork(name=NAME, store=fake_btrfs_store, created=NETWORK_CREATED))
    leave(fake_btrfs_store)
    client = PodmanClient()

    deleted = client.delete_sandbox(NAME)

    assert (deleted, fake_podman.networks(), traces(fake_btrfs_store)) == (True, {}, [False, False, False])


def test_a_sandbox_gone_between_the_plan_and_the_deletion_counts_as_deleted(fake_podman: FakePodman):
    fake_podman.fail("inspect")

    assert PodmanClient().delete_sandbox(NAME) is True


def test_a_container_that_is_listed_but_cannot_be_inspected_is_not_counted_as_deleted(
    fake_podman: FakePodman, fake_btrfs_store
):
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))
    fake_podman.fail("inspect")

    assert PodmanClient().delete_sandbox(NAME) is False


def test_a_failed_container_removal_keeps_the_rootfs_and_is_reported(
    capsys, fake_podman: FakePodman, fake_btrfs: FakeBtrfs, fake_btrfs_store
):
    """A subvolume cannot be deleted while a container mounts it."""
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))
    fake_podman.fail("rm")

    deleted = PodmanClient().delete_sandbox(NAME)

    assert (deleted, traces(fake_btrfs_store), fake_btrfs.calls()) == (False, [True, True, True], [])
    assert capsys.readouterr().err == "podman rm: Error: simulated rm failure\n"


def test_a_rootfs_that_cannot_be_deleted_is_reported_and_not_counted_as_deleted(
    capsys, fake_podman: FakePodman, fake_btrfs: FakeBtrfs, fake_btrfs_store
):
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))
    fake_btrfs.fail("delete")

    deleted = PodmanClient().delete_sandbox(NAME)

    assert deleted is False
    assert capsys.readouterr().err.startswith(
        "%s: container removed but its rootfs in %s was not: " % (NAME, fake_btrfs_store.resolve())
    )


def test_leftovers_name_the_container_the_network_and_the_rootfs_that_still_exist(
    fake_podman: FakePodman, fake_btrfs_store
):
    fake_podman.networks_exist(ManagedNetwork(name=NAME, store=fake_btrfs_store, created=NETWORK_CREATED))
    fake_podman.containers_exist(leaked_sandbox(fake_btrfs_store))

    found = PodmanClient(stores=[str(fake_btrfs_store)]).leftovers(NAME)

    assert found == ["container", "network", "rootfs in %s" % fake_btrfs_store.resolve()]


def test_leftovers_of_a_podman_that_cannot_run_are_unknown_and_not_empty(tmp_path):
    """An empty answer means that nothing is left, so an unknown answer must not look like one."""
    podman = tmp_path / "no-such-podman"

    found = PodmanClient(podman_bin=str(podman)).leftovers(NAME)

    assert found == [
        "unknown (container exists: [Errno 2] No such file or directory: '%s')" % podman,
        "unknown (network exists: [Errno 2] No such file or directory: '%s')" % podman,
    ]


def test_a_podman_orphan_without_a_container_is_planned_from_a_known_store_and_reaped(
    tmp_path, fake_podman: FakePodman, fake_btrfs_store
):
    """Without the rootfs in the listing, the reaper would take the orphan for one that is already gone."""
    leaked_sandbox(fake_btrfs_store)
    claim = {"event": "claim", "run_id": "run-1", "kind": "sandbox", "id": NAME, "pid": 999_999_999,
             "backend": "podman", "store": str(fake_btrfs_store)}  # fmt: skip
    (tmp_path / "resources.jsonl").write_text(json.dumps(claim) + "\n")
    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    reaper = Reaper({"podman": PodmanClient(stores=[str(fake_btrfs_store)])}, ledger)

    plan = reaper.plan()
    done = reaper.apply(plan)

    assert (plan.reasons, done, traces(fake_btrfs_store), reaper.plan().sandboxes) == (
        {NAME: "orphan of run run-1 (pid 999999999 gone)"},
        {"sandboxes": [NAME], "failed": []},
        [False, False, False],
        [],
    )

"""``BtrfsStore`` and the check of its mount, against the fake ``btrfs``, ``findmnt`` and ``podman``.

The worker's doctor shows the words of ``store_problem``, and the pool refuses the same stores with them.
"""

from __future__ import annotations

import json

import pytest

from sprout_sandbox import BtrfsStore
from sprout_sandbox.btrfs_store import config_path, store_problem
from sprout_sandbox.podman_cli import PodmanError

pytestmark = pytest.mark.anyio

#: What ``store_problem`` says, after the path, about a mount that cannot hold a store.
NEEDS_BTRFS = (
    "the podman backend needs btrfs mounted with user_subvol_rm_allowed "
    "(sudo ops/btrfs_store.sh <mount point> <image file> <size>)"
)


@pytest.mark.parametrize(
    "fstype, options",
    [
        pytest.param("ext4", "rw,relatime,user_subvol_rm_allowed", id="ext4"),
        pytest.param("btrfs", "rw,noatime,subvolid=5,subvol=/", id="btrfs-without-user-subvol-rm"),
    ],
)
def test_a_store_must_be_on_btrfs_mounted_with_user_subvol_rm_allowed(
    fake_btrfs_store, fake_findmnt, fstype, options
):
    """Without the option, every spawn works, and the first destroy fails after the container is gone."""
    fake_findmnt.report(fstype=fstype, options=options)

    problem = store_problem(fake_btrfs_store)

    assert problem == "store %s is on %s mounted with %s; %s" % (
        fake_btrfs_store,
        fstype,
        options,
        NEEDS_BTRFS,
    )


def test_a_store_needs_btrfs_progs_on_path(fake_findmnt, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))

    problem = store_problem(tmp_path)

    assert problem == (
        "btrfs-progs is not installed (no `btrfs` on PATH); "
        "the podman backend keeps its sandboxes in btrfs subvolumes"
    )


def test_a_store_needs_findmnt_to_check_its_mount(fake_btrfs, tmp_path, monkeypatch):
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))

    problem = store_problem(tmp_path)

    assert problem == "findmnt (util-linux) is not installed; it is how the store's mount is checked"


def test_a_refused_store_is_refused_at_construction_before_anything_is_made_in_it(
    fake_btrfs_store, fake_findmnt, fake_btrfs
):
    fake_findmnt.report(fstype="ext4", options="rw,relatime")

    with pytest.raises(ValueError) as raised:
        BtrfsStore(fake_btrfs_store)

    assert str(raised.value) == "store %s is on ext4 mounted with rw,relatime; %s" % (
        fake_btrfs_store,
        NEEDS_BTRFS,
    )
    assert (sorted(fake_btrfs_store.iterdir()), fake_btrfs.calls()) == ([], [])


async def test_a_subvolume_is_made_writable_before_it_is_deleted(fake_btrfs_store, fake_btrfs):
    """The unprivileged owner of the store can delete only a writable subvolume."""
    store = BtrfsStore(fake_btrfs_store)
    snapshot = store.snapshots / "sprout-000000000001-0001"
    await store.create(store.bases / "base")
    await store.snapshot(store.bases / "base", snapshot, read_only=True)
    made = len(fake_btrfs.calls())

    await store.delete(snapshot)

    assert fake_btrfs.calls()[made:] == [
        ["property", "set", str(snapshot), "ro", "false"],
        ["subvolume", "delete", str(snapshot)],
    ]
    assert not snapshot.exists()


async def test_deleting_a_subvolume_that_is_gone_runs_no_command(fake_btrfs_store, fake_btrfs):
    store = BtrfsStore(fake_btrfs_store)

    await store.delete(store.sandboxes / "sprout-000000000001")

    assert fake_btrfs.calls() == []


async def test_a_deleted_snapshot_takes_its_image_config_with_it(fake_btrfs_store, fake_btrfs):
    store = BtrfsStore(fake_btrfs_store)
    snapshot = store.snapshots / "sprout-000000000001-0001"
    await store.create(snapshot)
    config_path(snapshot).write_text(json.dumps({"config": {"User": ""}}))

    await store.delete_snapshot("sprout-000000000001-0001")

    assert sorted(store.snapshots.iterdir()) == []


@pytest.mark.parametrize("snapshot_id", ["", ".", "..", "../sandboxes/sprout-000000000001"])
async def test_a_snapshot_id_must_be_one_path_component(fake_btrfs_store, fake_btrfs, snapshot_id):
    store = BtrfsStore(fake_btrfs_store)

    with pytest.raises(ValueError, match="^not a snapshot id: "):
        await store.delete_snapshot(snapshot_id)

    assert fake_btrfs.calls() == []


async def test_deleting_a_snapshot_that_the_store_does_not_hold_is_an_error(fake_btrfs_store, fake_btrfs):
    store = BtrfsStore(fake_btrfs_store)

    with pytest.raises(PodmanError) as raised:
        await store.delete_snapshot("sprout-000000000001-0001")

    assert str(raised.value) == "no snapshot 'sprout-000000000001-0001' in %s/snapshots" % fake_btrfs_store


async def test_a_failed_btrfs_command_raises_with_its_command_words_and_stderr(fake_btrfs_store, fake_btrfs):
    store = BtrfsStore(fake_btrfs_store)
    fake_btrfs.fail("create")

    with pytest.raises(PodmanError) as raised:
        await store.create(store.bases / "base")

    assert str(raised.value) == "btrfs subvolume create failed (exit 1): ERROR: simulated create failure"

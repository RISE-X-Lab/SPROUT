"""The fake podman, btrfs and findmnt answer the code the way the real tools answer it.

Each test runs on the fakes, as a medium test, and on the real tools, as a large test marked ``slow``.
The real side works in a scratch directory on the store of the large tests. It skips when
``live_podman_problem`` finds a reason, and it removes each container, network and subvolume that it
makes. Run it alone, as each large test: ``pytest tests/support/test_fakes.py -m slow``.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import pytest

from sprout_sandbox.btrfs_store import BtrfsStore, config_path, mount_problem
from sprout_sandbox.podman_cli import PodmanError, default_podman_bin, run_command
from sprout_sandbox.podman_containers import MANAGED_LABEL, NETWORK_OPTIONS, SOCKET_MOUNT, STORE_LABEL
from sprout_sandbox.podman_images import EXTRACTION_FORMAT, digest_reference, ensure_base, image_info
from tests.support.tools import FakePodman, ManagedContainer

IMAGE = "docker.io/library/python:3.11-slim"
#: How podman writes the creation time of a network: UTC, to the nanosecond, without the trailing zeros.
NETWORK_TIME = re.compile(r"\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d(\.\d{0,8}[1-9])?Z")
#: The root directory of a btrfs subvolume always has this inode number.
SUBVOLUME_ROOT_INODE = 256


class FakeSide:
    """The fakes on PATH and a store in ``tmp_path``. A knob of the fake podman arranges each state."""

    def __init__(self, *, podman: FakePodman, store: Path, name: str) -> None:
        self.podman_bin = "podman"
        self.store = store
        self.name = name
        self._podman = podman

    def forget_image(self, reference: str) -> None:
        self._podman.unpulled(reference)

    def add_container(self, sockets: Path) -> None:
        self._podman.containers_exist(
            ManagedContainer(name=self.name, store=self.store, sockets=sockets, created=1700000000)
        )


class RealSide:
    """The real tools and a scratch store on btrfs. ``close`` removes what the test made."""

    def __init__(self, *, store: Path, name: str) -> None:
        self.podman_bin = default_podman_bin()
        self.store = store
        self.name = name

    def forget_image(self, reference: str) -> None:
        """Nothing to do: no image store and no registry knows the unique ``reference``."""

    def add_container(self, sockets: Path) -> None:
        labels = ["--label", MANAGED_LABEL, "--label", "%s=%s" % (STORE_LABEL, self.store)]
        mount = ["-v", "%s:%s" % (sockets, SOCKET_MOUNT)]
        argv = [self.podman_bin, "create", "--name", self.name, *labels, *mount, IMAGE, "true"]
        subprocess.run(argv, cwd=self.store, check=True, capture_output=True, timeout=600)

    def close(self) -> None:
        for argv in (["rm", "-f", self.name], ["network", "rm", self.name]):
            subprocess.run([self.podman_bin, *argv], cwd=self.store, capture_output=True, timeout=120)
        for subvolume in subvolumes_under(self.store):
            subprocess.run(["btrfs", "property", "set", str(subvolume), "ro", "false"], check=True)
            subprocess.run(["btrfs", "subvolume", "delete", str(subvolume)], check=True, capture_output=True)
        shutil.rmtree(self.store)


def subvolumes_under(root: Path) -> list[Path]:
    """The subvolumes below ``root``, deepest first."""
    found = [
        Path(directory) / name
        for directory, names, _ in os.walk(root)
        for name in names
        if os.lstat(Path(directory) / name).st_ino == SUBVOLUME_ROOT_INODE
    ]
    return sorted(found, key=lambda path: len(path.parts), reverse=True)


@pytest.fixture(params=["fakes", pytest.param("real", marks=pytest.mark.slow)])
def side(request: pytest.FixtureRequest) -> Iterator[FakeSide | RealSide]:
    """The fakes, or the real tools, with a store and a name of their own for one test."""
    name = "sprout-fidelity-" + uuid4().hex[:12]
    if request.param == "fakes":
        podman = request.getfixturevalue("fake_podman")
        yield FakeSide(podman=podman, store=request.getfixturevalue("fake_btrfs_store"), name=name)
        return
    store = request.getfixturevalue("live_podman_store") / name
    store.mkdir()
    real = RealSide(store=store, name=name)
    try:
        yield real
    finally:
        real.close()


def sources_mounted_at(info: dict, destination: str) -> list[str]:
    """The host paths that the inspect record ``info`` of a container mounts at ``destination``."""
    return [mount["Source"] for mount in info["Mounts"] if mount["Destination"] == destination]


def podman(side: FakeSide | RealSide, *args: str) -> str:
    """The stdout of the side's podman with ``args``. Raises ``PodmanError`` if the command fails."""
    return asyncio.run(run_command([side.podman_bin, *args], timeout=600, cwd=side.store))


def exit_code(side: FakeSide | RealSide, *args: str) -> int:
    """The exit code of the side's podman with ``args``."""
    argv = [side.podman_bin, *args]
    return subprocess.run(argv, cwd=side.store, capture_output=True, timeout=120).returncode


def of_the_store(side: FakeSide | RealSide) -> str:
    """The podman filter that selects what carries the label of ``side.store``."""
    return "label=%s=%s" % (STORE_LABEL, side.store)


def test_the_mount_check_accepts_the_store(side):
    assert mount_problem(side.store) is None


def test_a_snapshot_has_the_files_of_its_source_and_keeps_its_own_writes(side):
    store = BtrfsStore(side.store)
    source, copy = store.bases / "source", store.snapshots / "copy"
    asyncio.run(store.create(source))
    (source / "answer.txt").write_text("from the source\n")

    asyncio.run(store.snapshot(source, copy))
    (copy / "answer.txt").write_text("from the copy\n")

    assert (source / "answer.txt").read_text() == "from the source\n"
    assert (copy / "answer.txt").read_text() == "from the copy\n"


def test_a_read_only_snapshot_can_be_deleted(side):
    store = BtrfsStore(side.store)
    source, copy = store.bases / "source", store.snapshots / "copy"
    asyncio.run(store.create(source))
    asyncio.run(store.snapshot(source, copy, read_only=True))

    asyncio.run(store.delete(copy))

    assert not copy.exists()


def test_a_subvolume_made_inside_a_rootfs_is_found_as_nested(side):
    store = BtrfsStore(side.store)
    rootfs = store.sandboxes / "sandbox"
    asyncio.run(store.create(rootfs))
    asyncio.run(store.create(rootfs / "inner"))
    (rootfs / "plain").mkdir()

    nested = asyncio.run(store.nested_subvolumes(rootfs, podman_bin=side.podman_bin))

    assert nested == [rootfs / "inner"]


def test_discarding_a_rootfs_deletes_the_subvolumes_inside_it(side):
    store = BtrfsStore(side.store)
    rootfs = store.sandboxes / "sandbox"
    asyncio.run(store.create(rootfs))
    asyncio.run(store.create(rootfs / "inner"))

    asyncio.run(store.discard_rootfs(rootfs, podman_bin=side.podman_bin))

    assert not rootfs.exists()


def test_an_image_record_has_an_id_a_pinned_digest_and_the_image_config(side):
    info = asyncio.run(image_info(IMAGE, podman_bin=side.podman_bin, cwd=side.store))

    assert re.fullmatch(r"[0-9a-f]{64}", info["Id"]), info["Id"]
    assert re.fullmatch(r"docker\.io/library/python@sha256:[0-9a-f]{64}", digest_reference(IMAGE, info))
    assert info["Config"]["Cmd"] == ["python3"]
    assert "LANG=C.UTF-8" in info["Config"]["Env"]


def test_inspecting_an_image_that_is_not_local_fails_with_image_not_known(side):
    reference = "docker.io/library/%s:absent" % side.name
    side.forget_image(reference)

    with pytest.raises(PodmanError) as raised:
        podman(side, "image", "inspect", "--format", "json", reference)

    assert str(raised.value) == (
        "podman image inspect failed (exit 125): Error: %s: image not known" % reference
    )


def test_a_new_sandbox_network_is_listed_by_its_labels_with_its_creation_time(side):
    labels = ["--label", MANAGED_LABEL, "--label", "%s=%s" % (STORE_LABEL, side.store)]
    podman(side, "network", "create", *NETWORK_OPTIONS, *labels, side.name)

    raw = podman(side, "network", "ls", "--filter", of_the_store(side), "--format", "json")

    [listed] = json.loads(raw)
    assert listed["name"] == side.name
    assert listed["labels"] == {"sprout.managed": "1", "sprout.store": str(side.store)}
    assert NETWORK_TIME.fullmatch(listed["created"]), listed["created"]


def test_network_exists_answers_0_for_a_network_and_1_after_its_removal(side):
    podman(side, "network", "create", *NETWORK_OPTIONS, side.name)
    before = exit_code(side, "network", "exists", side.name)

    podman(side, "network", "rm", side.name)

    assert (before, exit_code(side, "network", "exists", side.name)) == (0, 1)


def test_removing_a_network_that_does_not_exist_fails_with_network_not_found(side):
    with pytest.raises(PodmanError) as raised:
        podman(side, "network", "rm", side.name)

    assert str(raised.value) == (
        "podman network rm failed (exit 1): Error: unable to find network with name or ID %s: "
        "network not found" % side.name
    )


def test_ps_lists_a_sandbox_container_by_its_labels_with_its_name_and_creation_time(side, tmp_path):
    side.add_container(tmp_path)

    raw = podman(side, "ps", "-a", "--filter", of_the_store(side), "--format", "json")

    [listed] = json.loads(raw)
    assert (listed["Names"], listed["Labels"][STORE_LABEL]) == ([side.name], str(side.store))
    assert type(listed["Created"]) is int


def test_ps_quiet_prints_the_short_id_of_each_sandbox_container(side, tmp_path):
    side.add_container(tmp_path)

    raw = podman(side, "ps", "-a", "-q", "--filter", of_the_store(side))

    assert re.fullmatch(r"[0-9a-f]{12}\n", raw), raw


def test_inspect_gives_the_store_label_and_the_socket_mount_of_a_sandbox_container(side, tmp_path):
    side.add_container(tmp_path)

    [info] = json.loads(podman(side, "inspect", "--type", "container", "--format", "json", side.name))

    assert info["Config"]["Labels"][STORE_LABEL] == str(side.store)
    assert sources_mounted_at(info, SOCKET_MOUNT) == [str(tmp_path)]


def test_a_removed_container_no_longer_exists(side, tmp_path):
    side.add_container(tmp_path)
    before = exit_code(side, "container", "exists", side.name)

    podman(side, "rm", "-f", side.name)

    assert (before, exit_code(side, "container", "exists", side.name)) == (0, 1)


def test_an_image_is_extracted_into_a_base_named_after_its_id_with_its_config(side):
    store = BtrfsStore(side.store)
    info = asyncio.run(image_info(IMAGE, podman_bin=side.podman_bin, cwd=side.store))

    base = asyncio.run(ensure_base(store, IMAGE, podman_bin=side.podman_bin, timeout=120, bulk_timeout=1800))

    saved = json.loads(config_path(base).read_text())
    assert (base.name, (base / "etc").is_dir()) == (info["Id"], True)
    assert (saved["extract"], saved["config"]["Cmd"]) == (EXTRACTION_FORMAT, ["python3"])

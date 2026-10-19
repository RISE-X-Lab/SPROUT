"""``PodmanPool`` against the real podman, btrfs-progs and btrfs store of this host.

These large tests check what the fakes cannot: that a container answers over its socket with the
image's environment, that a branch starts from the files of its snapshot, that the network toggle
cuts egress, that each sandbox is alone on its network, and that an extraction and a branch keep each
attribute of a file. They skip with a reason when ``live_podman_problem`` finds one, or when the
runtime binary is not built. The first run pulls ``LIVE_IMAGE`` and extracts its base, which stays
in the store. Run the file alone, while no experiment uses the store:

    pytest tests/sprout_sandbox/test_podman_live.py -m slow
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from collections.abc import AsyncIterator
from pathlib import Path

import pytest

from sprout_sandbox import Snapshot
from sprout_sandbox.btrfs_store import config_path
from sprout_sandbox.podman_cli import PodmanError, default_podman_bin
from sprout_sandbox.podman_images import EXTRACTION_FORMAT
from tests.sprout_sandbox.live import (
    LIVE_IMAGE,
    Leftovers,
    built_image,
    checked,
    entries,
    image_id,
    nothing,
    podman,
    shell,
)
from tests.support.live import sandbox_networks

pytestmark = [pytest.mark.slow, pytest.mark.anyio]

#: Prints ``online`` when the container has egress, and ``offline`` at once when it has none.
EGRESS = (
    "python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5)\" 2>/dev/null "
    "&& echo online || echo offline"
)
#: BTRFS_IOC_SUBVOL_CREATE on a directory: _IOW(0x94, 14, struct btrfs_ioctl_vol_args), a 64-bit fd and
#: a 4088-byte name. Any btrfs user can make a subvolume, so a process in the container can too.
MAKE_SUBVOLUME = (
    "python3 -c \"import fcntl, os, struct; fd = os.open('%s', os.O_RDONLY | os.O_DIRECTORY); "
    "fcntl.ioctl(fd, 0x5000940e, bytearray(struct.pack('q4088s', 0, b'%s')), True)\""
)
#: The base image as the user ``nobody``, in ``/tmp``.
USER_DOCKERFILE = "FROM %s\nUSER nobody\nWORKDIR /tmp\n" % LIVE_IMAGE
#: Each kind of entry that an extraction or a snapshot could lose, from owners in the user namespace and
#: file capabilities to a sparse file, a FIFO, a non-ASCII name and a VOLUME.
ADVERSARIAL_DOCKERFILE = (
    """FROM %s
RUN mkdir -p /data /adv /adv/dir1000 "/adv/odd name" && \\
    echo seed > /data/seed.txt && \\
    echo shared > /adv/h1 && ln /adv/h1 /adv/h2 && \\
    ln -s h1 /adv/rel-link && ln -s /adv/h1 /adv/abs-link && ln -s /nowhere /adv/dangling && \\
    cp /bin/true /adv/suid && chmod 4755 /adv/suid && \\
    cp /bin/true /adv/sgid && chmod 2755 /adv/sgid && \\
    cp /bin/true /adv/cap && \\
    python3 -c "import os; os.setxattr('/adv/cap', 'security.capability', \\
        bytes.fromhex('0100000200200000' + '0' * 24))" && \\
    echo xattr > /adv/xattr && \\
    python3 -c "import os; os.setxattr('/adv/xattr', 'user.sprout', b'review')" && \\
    truncate -s 64M /adv/sparse && \\
    printf mid | dd of=/adv/sparse bs=1 seek=33554432 conv=notrunc status=none && \\
    echo owned > /adv/owned && chown 1000:1000 /adv/owned && chmod 600 /adv/owned && \\
    chown 1000:1000 /adv/dir1000 && chmod 700 /adv/dir1000 && echo private > /adv/dir1000/secret && \\
    chown 1000:1000 /adv/dir1000/secret && \\
    echo nobody > /adv/nobody && chown 65534:65534 /adv/nobody && \\
    mkfifo /adv/fifo && echo "unicode" > "/adv/odd name/f\u00efl\u00e9 with spaces" && mkdir /adv/emptydir
VOLUME /data
RUN echo after-volume > /data/after.txt
"""
    % LIVE_IMAGE
)
#: Prints one line per entry of a tree: type, mode, owner, size, hardlink group (the first path with the
#: inode), symlink target, sparseness, each xattr and a content hash. It runs under ``podman unshare``.
FINGERPRINT = r"""
import hashlib, os, stat, sys
root = sys.argv[1]
groups, lines = {}, []
for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
    dirnames.sort()
    for name in sorted(dirnames + filenames):
        full = os.path.join(dirpath, name)
        rel = os.path.relpath(full, root)
        st = os.lstat(full)
        kind = {stat.S_IFDIR: "d", stat.S_IFREG: "f", stat.S_IFLNK: "l", stat.S_IFIFO: "p",
                stat.S_IFSOCK: "s", stat.S_IFCHR: "c", stat.S_IFBLK: "b"}.get(stat.S_IFMT(st.st_mode), "?")
        group = groups.setdefault((st.st_dev, st.st_ino), rel) if kind == "f" and st.st_nlink > 1 else "-"
        sparse = ("sparse" if st.st_blocks * 512 < st.st_size - 4096 else "dense") if kind == "f" else "-"
        xattrs = ",".join("%s=%s" % (k, os.getxattr(full, k, follow_symlinks=False).hex())
                          for k in sorted(os.listxattr(full, follow_symlinks=False))) or "-"
        digest = "-"
        if kind == "f":
            h = hashlib.sha256()
            with open(full, "rb") as fh:
                for chunk in iter(lambda: fh.read(1 << 20), b""):
                    h.update(chunk)
            digest = h.hexdigest()[:16]
        lines.append("%s\t%s %o %d %d %s %s %s %s %s %s" % (
            rel, kind, stat.S_IMODE(st.st_mode), st.st_uid, st.st_gid, st.st_size if kind in "fl" else "-",
            group, os.readlink(full) if kind == "l" else "-", sparse, xattrs, digest))
print("\n".join(lines))
"""


def container_state(sandbox_id: str) -> str:
    """The status, the memory limit in bytes and the CPU limit in nano-CPUs of a container."""
    return podman(
        "inspect", "--type", "container", "--format",
        "{{.State.Status}} {{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}", sandbox_id,
    ).strip()  # fmt: skip


def read_only(path: Path) -> str:
    """What btrfs-progs says of the ``ro`` property of the subvolume ``path``, such as ``ro=true``."""
    completed = subprocess.run(
        ["btrfs", "property", "get", str(path), "ro"], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def networks_of(sandbox_id: str) -> list[str]:
    """The names of the networks that the container ``sandbox_id`` is on."""
    template = "{{range $name, $_ := .NetworkSettings.Networks}}{{$name}} {{end}}"
    return podman("inspect", "--type", "container", "--format", template, sandbox_id).split()


def address_of(sandbox_id: str) -> str:
    """The IP address of the container ``sandbox_id`` on its network."""
    template = "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}"
    return podman("inspect", "--type", "container", "--format", template, sandbox_id).strip()


def connect_probe(address: str) -> str:
    """A command that prints how a TCP connect to port 9 of ``address`` ends, such as ``TimeoutError``."""
    return (
        'python3 -c "import socket\ns = socket.socket(); s.settimeout(3)\ntry:\n'
        "    s.connect(('%s', 9)); print('connected')\nexcept OSError as e:\n"
        '    print(type(e).__name__)"' % address
    )


def fingerprint(script: Path, root: str) -> dict[str, str]:
    """The attributes of each entry of the tree at ``root``, a shell word evaluated in the user namespace."""
    out = podman("unshare", "sh", "-c", "python3 %s %s" % (script, root))
    return dict(line.split("\t", 1) for line in out.splitlines())


@pytest.fixture
async def user_image(live_podman_store: Path, tmp_path: Path) -> AsyncIterator[str]:
    """An image whose User is ``nobody`` and whose WorkingDir is ``/tmp``."""
    async with built_image(
        "localhost/sprout-test-user:latest", USER_DOCKERFILE, context=tmp_path, store=live_podman_store
    ) as tag:
        yield tag


@pytest.fixture
async def adversarial_image(live_podman_store: Path, tmp_path: Path) -> AsyncIterator[str]:
    """An image with each kind of entry of ``ADVERSARIAL_DOCKERFILE``."""
    async with built_image(
        "localhost/sprout-test-fidelity:latest",
        ADVERSARIAL_DOCKERFILE,
        context=tmp_path,
        store=live_podman_store,
    ) as tag:
        yield tag


@pytest.fixture
def fingerprint_script(tmp_path: Path) -> Path:
    script = tmp_path / "fingerprint.py"
    script.write_text(FINGERPRINT)
    return script


async def test_a_sandbox_runs_the_runtime_as_root_with_the_images_environment(live_pools):
    pool = live_pools()

    sandbox = await pool.spawn(LIVE_IMAGE)

    assert await checked(sandbox, "printenv LANG && id -u && pwd && command -v python3") == (
        "C.UTF-8\n0\n/\n/usr/local/bin/python3"
    )


async def test_the_shape_of_a_sandbox_reaches_its_container(live_pools):
    pool = live_pools()

    sandbox = await pool.spawn(LIVE_IMAGE, resources={"cpu": 1, "memory_mb": 300})

    assert container_state(sandbox.sandbox_id) == "running 314572800 1000000000"


async def test_the_base_of_an_image_is_a_read_only_subvolume_named_after_the_image_id(live_pools):
    pool = live_pools()

    base = await pool.prepare(LIVE_IMAGE)

    assert base == pool.store.bases / await image_id(LIVE_IMAGE)
    assert (read_only(base), json.loads(config_path(base).read_text())["extract"]) == (
        "ro=true",
        EXTRACTION_FORMAT,
    )


async def test_a_snapshot_is_read_only_and_leaves_its_container_running(live_pools):
    pool = live_pools()
    sandbox = await pool.spawn(LIVE_IMAGE)

    snapshot = await pool.snapshot(sandbox)

    assert snapshot == Snapshot(id=sandbox.sandbox_id + "-0001", disk_only=True)
    assert read_only(pool.store.snapshots / snapshot.id) == "ro=true"
    assert container_state(sandbox.sandbox_id).startswith("running ")


async def test_a_branch_has_the_files_of_its_parent_as_of_the_snapshot(live_pools):
    pool = live_pools()
    parent = await pool.spawn(LIVE_IMAGE)
    await checked(parent, "echo before > /tmp/before")
    snapshot = await pool.snapshot(parent)
    await checked(parent, "echo after > /tmp/after")

    branch = await pool.spawn(snapshot.id)

    assert await checked(branch, "cat /tmp/before && test ! -e /tmp/after && echo no-later-write") == (
        "before\nno-later-write"
    )


async def test_the_writes_of_a_branch_stay_its_own(live_pools):
    pool = live_pools()
    parent = await pool.spawn(LIVE_IMAGE)
    branch = await pool.spawn((await pool.snapshot(parent)).id)

    await checked(branch, "echo branch > /tmp/branch")

    assert await shell(parent, "test -e /tmp/branch") == ("", 1)


async def test_host_files_go_into_and_out_of_a_sandbox(live_pools, tmp_path):
    payload = bytes(range(256)) * 16
    (tmp_path / "fixture.bin").write_bytes(payload)
    pool = live_pools()
    sandbox = await pool.spawn(LIVE_IMAGE)
    await checked(sandbox, "echo answer > /tmp/answer.txt")

    await pool.upload_file(sandbox, tmp_path / "fixture.bin", "/tmp/fixture.bin")
    await pool.download_file(sandbox, "/tmp/answer.txt", tmp_path / "answer.txt")

    assert (
        await checked(sandbox, "sha256sum /tmp/fixture.bin | cut -d' ' -f1")
        == hashlib.sha256(payload).hexdigest()
    )
    assert (tmp_path / "answer.txt").read_text() == "answer\n"


async def test_a_teardown_leaves_no_container_network_sandbox_snapshot_or_socket_directory(
    live_pools, live_podman_store
):
    before = Leftovers.now(live_podman_store)
    pool = live_pools()
    parent = await pool.spawn(LIVE_IMAGE)
    snapshot = await pool.snapshot(parent)
    branch = await pool.spawn(snapshot.id)

    await pool.destroy(branch, parent)
    await pool.delete_snapshot(snapshot.id)

    assert Leftovers.now(live_podman_store).appeared_since(before) == nothing()


async def test_setting_the_network_of_a_sandbox_cuts_and_restores_its_egress_and_its_runtime_keeps_answering(
    live_pools,
):
    pool = live_pools()
    sandbox = await pool.spawn(LIVE_IMAGE)
    online = await checked(sandbox, EGRESS)

    await pool.set_network(sandbox, False)
    offline, answering = await checked(sandbox, EGRESS), await checked(sandbox, "echo answering")
    await pool.set_network(sandbox, True)

    assert (online, offline, answering, await checked(sandbox, EGRESS)) == (
        "online",
        "offline",
        "answering",
        "online",
    )


async def test_the_first_command_of_an_offline_sandbox_already_runs_offline(live_pools):
    pool = live_pools(allow_internet=False)

    sandbox = await pool.spawn(LIVE_IMAGE)

    assert await checked(sandbox, EGRESS) == "offline"


async def test_each_sandbox_is_alone_on_a_labeled_network_without_podman_dns(live_pools):
    pool = live_pools()
    first, second = await pool.spawn(LIVE_IMAGE), await pool.spawn(LIVE_IMAGE)

    [network] = json.loads(podman("network", "inspect", first.sandbox_id))

    assert (networks_of(first.sandbox_id), networks_of(second.sandbox_id)) == (
        [first.sandbox_id],
        [second.sandbox_id],
    )
    assert (network["dns_enabled"], network["options"]["isolate"]) == (False, "true")
    assert network["labels"] == {"sprout.managed": "1", "sprout.store": str(pool.store.path)}


async def test_a_sandbox_resolves_names_through_the_resolvers_of_the_host(live_pools):
    pool = live_pools()
    sandbox = await pool.spawn(LIVE_IMAGE)
    [network] = json.loads(podman("network", "inspect", sandbox.sandbox_id))

    resolv = await checked(sandbox, "cat /etc/resolv.conf")
    resolved = await checked(
        sandbox, "python3 -c \"import socket; print(bool(socket.getaddrinfo('pypi.org', 443)))\""
    )

    assert (network["subnets"][0]["gateway"] in resolv, resolved) == (False, "True")


async def test_a_sandbox_cannot_reach_another_sandbox(live_pools):
    """The packets are dropped between the networks, not refused by the other sandbox."""
    pool = live_pools()
    first, second = await pool.spawn(LIVE_IMAGE), await pool.spawn(LIVE_IMAGE)

    ending = await checked(first, connect_probe(address_of(second.sandbox_id)))

    assert ending == "TimeoutError"


async def test_destroying_a_sandbox_removes_its_network_and_keeps_the_others(live_pools):
    pool = live_pools()
    first, second = await pool.spawn(LIVE_IMAGE), await pool.spawn(LIVE_IMAGE)
    first_id = first.sandbox_id

    await pool.destroy(first)

    assert (first_id in sandbox_networks(), second.sandbox_id in sandbox_networks()) == (False, True)


async def test_the_runtime_is_root_in_the_workdir_of_an_image_whose_user_is_not_root(live_pools, user_image):
    """The root steps of a Harbor trial, such as ``mkdir /logs``, need a runtime that is root."""
    pool = live_pools()

    sandbox = await pool.spawn(user_image)

    assert await checked(sandbox, "id -u && pwd && runuser -u nobody -- id -u") == "0\n/tmp\n65534"
    assert (
        json.loads(config_path(pool.store.sandboxes / sandbox.sandbox_id).read_text())["config"]["User"]
        == "nobody"
    )


async def test_the_runtime_socket_belongs_to_the_user_of_the_pool_on_an_image_whose_user_is_not_root(
    live_pools, user_image
):
    """The host connects to the socket with no change of owner from inside the user namespace."""
    pool = live_pools()

    sandbox = await pool.spawn(user_image)

    socket_stat = (pool.socket_dir / sandbox.sandbox_id / "runtime.sock").stat()
    assert (socket_stat.st_uid, oct(socket_stat.st_mode & 0o777)) == (os.getuid(), oct(0o660))


async def test_an_extracted_base_has_each_attribute_of_each_entry_of_its_image(
    live_pools, adversarial_image, fingerprint_script
):
    """Each setcap binary of the image keeps its file capabilities in each sandbox."""
    pool = live_pools()
    mount = "$(%s image mount %s)" % (shlex.quote(default_podman_bin()), shlex.quote(adversarial_image))

    base = await pool.prepare(adversarial_image)

    image = fingerprint(fingerprint_script, mount)
    podman("unshare", default_podman_bin(), "image", "unmount", adversarial_image)
    extracted = fingerprint(fingerprint_script, str(base))
    assert len(image) > 5000, "the fingerprint of the image mount is too small to be the image"
    assert extracted == image
    assert "security.capability=0100000200200000" in extracted["adv/cap"]


async def test_a_branch_has_each_attribute_of_each_entry_of_its_snapshot(
    live_pools, adversarial_image, fingerprint_script
):
    pool = live_pools()
    parent = await pool.spawn(adversarial_image)
    await checked(parent, "python3 -c \"import os; os.setxattr('/adv/xattr', 'user.second', b'2')\"")
    snapshot = await pool.snapshot(parent)

    branch = await pool.spawn(snapshot.id)

    snapshot_entries = fingerprint(fingerprint_script, str(pool.store.snapshots / snapshot.id))
    assert fingerprint(fingerprint_script, str(pool.store.sandboxes / branch.sandbox_id)) == snapshot_entries
    assert "user.second=32" in snapshot_entries["adv/xattr"]


async def test_a_subvolume_made_inside_a_sandbox_refuses_its_snapshot(live_pools, live_podman_store):
    """One subvolume is in a 0700 directory of another user, which only the namespace's root can read."""
    pool = live_pools()
    sandbox = await pool.spawn(LIVE_IMAGE)
    await checked(sandbox, MAKE_SUBVOLUME % ("/tmp", "nested"))
    await checked(sandbox, "mkdir -m 700 /tmp/private && chown 1000:1000 /tmp/private")
    await checked(sandbox, MAKE_SUBVOLUME % ("/tmp/private", "hidden"))
    before = Leftovers.now(live_podman_store)

    with pytest.raises(PodmanError) as raised:
        await pool.snapshot(sandbox)

    assert str(raised.value) == (
        "rootfs of %s holds nested subvolumes (/tmp/private/hidden, /tmp/nested); "
        "a snapshot would omit their contents" % sandbox.sandbox_id
    )
    assert Leftovers.now(live_podman_store).appeared_since(before) == nothing()


async def test_a_destroy_deletes_a_subvolume_made_inside_the_sandbox(live_pools):
    pool = live_pools()
    sandbox = await pool.spawn(LIVE_IMAGE)
    sid = sandbox.sandbox_id
    await checked(sandbox, MAKE_SUBVOLUME % ("/tmp", "nested") + " && echo kept > /tmp/nested/kept.txt")

    await pool.destroy(sandbox)

    assert (sid in entries(pool.store.sandboxes), pool.list()) == (False, [])

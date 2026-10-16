"""The btrfs podman backend against podman, btrfs-progs and a real store.

Everything else about the backend is covered with the fakes in
``tests/conftest.py``. This answers what they cannot: does a container start on
an extracted base and answer over the socket with the image's environment; is a
snapshot taken while it runs, and does a branch spawned from it see the files
and keep its writes to itself; does the network toggle cut egress while the
socket keeps answering, and does an offline spawn's first command already run
offline; does the runtime run as root on an image whose user is not, with the
image's working directory; does an extracted base carry every attribute of
the image (ownership across the user namespace, setuid bits, hardlinks,
capabilities and other xattrs, a FIFO, a sparse file), and a branch every
attribute of its snapshot; is a subvolume made inside a container refused as a
snapshot and deleted by destroy; is each sandbox alone on a network of its
own, resolving names through the host's resolvers and unable to reach another
sandbox; do upload and download work on a ``--rootfs`` container; and does a
teardown leave no container, no network, no subvolume and no socket directory
behind.

Skipped unless ``podman`` and ``btrfs`` are on PATH, the runtime binary is
built, and ``SPROUT_PODMAN_STORE`` (default ``~/sprout-btrfs``) is a directory
on btrfs mounted with ``user_subvol_rm_allowed`` -- the store's own check,
``mount_problem``, decides. The first run pulls python:3.11-slim and extracts
its base, which stays in the store; everything else a test makes is removed
and asserted gone. Run this file alone: the ``cleanup`` fixture diffs the whole
host's containers and the store.

    pytest tests/sdk/test_podman_live.py -v -m slow -s      # -s prints the timings
"""

from __future__ import annotations

import asyncio
import json
import os
import shlex
import shutil
import subprocess
import time
from pathlib import Path

import pytest

from tests.conftest import default_podman_bin, podman_store_in_use, sandbox_networks

from sprout_sandbox import PodmanPool
from sprout_sandbox.podman import PodmanError, _config_path, image_info, mount_problem

REPO = Path(__file__).resolve().parents[2]
RUNTIME_BIN = REPO / "runtime" / "sprout-runtime"
STORE = Path(os.environ.get("SPROUT_PODMAN_STORE", "~/sprout-btrfs")).expanduser()
#: The podman the pool runs: $SPROUT_PODMAN_BIN (the one ops/setup_podman.sh installed), else PATH.
PODMAN = default_podman_bin()
IMAGE = "docker.io/library/python:3.11-slim"
#: Exits 0 with "online" when the container has egress; ENETUNREACH at once without.
EGRESS_PROBE = ("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 443), timeout=5); "
                "print('online')\"")

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which(default_podman_bin()) is None,
                       reason="needs podman on PATH or in $SPROUT_PODMAN_BIN"),
    pytest.mark.skipif(shutil.which("btrfs") is None, reason="needs btrfs-progs on PATH"),
    pytest.mark.skipif(not RUNTIME_BIN.exists(),
                       reason="needs runtime/sprout-runtime "
                              "(cd runtime && go generate ./... && go build -o sprout-runtime .)"),
    pytest.mark.skipif(STORE.is_dir() and podman_store_in_use(STORE) is not None,
                       reason="the store is in use by a run (%s); the fixtures here remove every container "
                              "and snapshot that appears during a test" % podman_store_in_use(STORE)),
    pytest.mark.skipif(not STORE.is_dir() or mount_problem(STORE) is not None,
                       reason="needs a btrfs store with user_subvol_rm_allowed at SPROUT_PODMAN_STORE "
                              "(sudo ops/btrfs_store.sh <mount point> <image file> <size>)"),
]


def _podman(*args: str) -> str:
    return subprocess.run([PODMAN, *args], capture_output=True, text=True, check=True).stdout


def _containers() -> set[str]:
    """Every container, labelled or not, so an unlabelled leftover counts too."""
    return set(_podman("ps", "-a", "--no-trunc", "-q").split())


def _entries(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()} if directory.is_dir() else set()


def _state(container: str) -> str:
    return _podman("inspect", "--type", "container", "--format",
                   "{{.State.Status}} {{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}", container).strip()


def _ro(path: Path) -> str:
    return subprocess.run(["btrfs", "property", "get", str(path), "ro"],
                          capture_output=True, text=True, check=True).stdout.strip()


def sh(sandbox, command: str) -> tuple[str, int]:
    """``(stdout, exit code)`` of a command run through the runtime.

    A plain success is bare stdout (no outcome fields); anything else is a
    CommandOutcome with the exit code.
    """
    result = asyncio.run(sandbox.call("shell", command=command, timeout=30))
    if result.exit_code is None:
        assert not result.is_error and not result.running, result.output
        return result.output, 0
    return result.stdout or "", result.exit_code


def checked(sandbox, command: str) -> str:
    result = asyncio.run(sandbox.call("shell", command=command, timeout=30))
    if result.exit_code is None:
        assert not result.is_error and not result.running, result.output
        return result.output.strip()
    # A failure says why: the reason is usually on stderr.
    assert result.exit_code == 0, (command, result.exit_code, result.stdout, result.stderr)
    return (result.stdout or "").strip()


@pytest.fixture
def cleanup():
    """Close every pool a test made; assert no container, network, subvolume or socket dir is left."""
    before = {kind: _entries(STORE / kind) for kind in ("sandboxes", "snapshots", "sockets")}
    containers_before, networks_before = _containers(), sandbox_networks()
    pools: list[PodmanPool] = []
    yield pools
    for pool in pools:
        asyncio.run(pool.close())
    leaked = _containers() - containers_before
    if leaked:
        subprocess.run([PODMAN, "rm", "-f", *leaked], capture_output=True)
    leaked_networks = sandbox_networks() - networks_before
    if leaked_networks:
        subprocess.run([PODMAN, "network", "rm", *leaked_networks], capture_output=True)
    left = {kind: _entries(STORE / kind) - before[kind] for kind in before}
    for name in sorted(left["snapshots"]):
        if not name.endswith(".json"):
            asyncio.run(PodmanPool(STORE, RUNTIME_BIN).delete_snapshot(name))
    assert not leaked, "containers left behind: %s" % sorted(leaked)
    assert not leaked_networks, "networks left behind: %s" % sorted(leaked_networks)
    assert not left["sandboxes"] and not left["sockets"], left
    assert not left["snapshots"], "snapshots left behind (a test must delete what it takes): %s" % left


def test_spawn_snapshot_branch_files_and_teardown(cleanup):
    pool = PodmanPool(STORE, RUNTIME_BIN)
    cleanup.append(pool)
    started = time.perf_counter()
    parent = asyncio.run(pool.spawn_from_image(IMAGE, resources={"cpu": 1, "memory_mb": 300}))
    spawn_s = time.perf_counter() - started
    sid = parent.sandbox_id
    assert sid.startswith("sprout-") and (STORE / "sandboxes" / sid).is_dir()
    assert _state(sid) == "running 314572800 1000000000", "the shape reaches the container"
    base = STORE / "bases" / _digest(pool, IMAGE)
    assert base.is_dir() and _config_path(base).is_file() and _ro(base) == "ro=true"

    assert checked(parent, "echo hi > /tmp/a && cat /tmp/a && id -u && pwd") == "hi\n0\n/"
    env = checked(parent, "env")
    assert "PYTHON_VERSION=" in env and "LANG=C.UTF-8" in env, "the image's ENV reached the runtime"
    assert checked(parent, "python3 -c 'import sys; print(sys.version_info[0])'") == "3", "the image's PATH"

    started = time.perf_counter()
    snap = asyncio.run(pool.snapshot(parent))
    snapshot_s = time.perf_counter() - started
    assert snap.id.startswith(sid + "-0001") and snap.disk_only and snap.rootfs_layers is None
    assert _ro(STORE / "snapshots" / snap.id) == "ro=true"
    assert _state(sid).startswith("running"), "the container was not touched"
    checked(parent, "echo later > /tmp/after")
    with pytest.raises(NotImplementedError):
        asyncio.run(pool.snapshot(parent, disk_only=False))

    started = time.perf_counter()
    branch = asyncio.run(pool.spawn(snap.id, resources={"cpu": 1, "memory_mb": 300}))
    branch_s = time.perf_counter() - started
    assert branch.sandbox_id != sid
    assert checked(branch, "cat /tmp/a") == "hi", "the branch sees the parent's files as of the snapshot"
    assert sh(branch, "test -e /tmp/after")[1] != 0, "a write after the snapshot is not in it"
    checked(branch, "echo branch > /tmp/b")
    assert sh(parent, "test -e /tmp/b")[1] != 0, "the branch's write reached the parent: no isolation"
    assert _state(branch.sandbox_id) == "running 314572800 1000000000"

    fixture = STORE / "sockets" / "fixture.bin"
    fixture.write_bytes(bytes(range(256)) * 16)
    try:
        asyncio.run(pool.upload_file(parent, fixture, "/tmp/fixture.bin"))
        assert checked(parent, "wc -c < /tmp/fixture.bin") == "4096"
        asyncio.run(pool.download_file(parent, "/tmp/a", STORE / "sockets" / "a.txt"))
        assert (STORE / "sockets" / "a.txt").read_text() == "hi\n"
    finally:
        fixture.unlink(missing_ok=True)
        (STORE / "sockets" / "a.txt").unlink(missing_ok=True)

    asyncio.run(pool.destroy(branch, parent))
    assert pool.list() == [] and parent.sandbox_id is None
    assert not (STORE / "sandboxes" / sid).exists() and not (STORE / "sockets" / sid).exists()
    assert sid not in _podman("ps", "-a", "--format", "{{.Names}}")
    assert (STORE / "snapshots" / snap.id).is_dir(), "destroying the sandbox keeps its snapshots"
    asyncio.run(pool.delete_snapshot(snap.id))
    assert not (STORE / "snapshots" / snap.id).exists() and not _config_path(STORE / "snapshots" / snap.id).exists()
    print("\nspawn %.2f s (base cached: %s), snapshot %.3f s, branch %.2f s"
          % (spawn_s, base.is_dir(), snapshot_s, branch_s))


def test_network_toggle_and_offline_spawn(cleanup):
    online_pool = PodmanPool(STORE, RUNTIME_BIN)
    offline_pool = PodmanPool(STORE, RUNTIME_BIN, allow_internet=False)
    cleanup.extend([online_pool, offline_pool])

    sb = asyncio.run(online_pool.spawn(IMAGE))
    assert sh(sb, EGRESS_PROBE) == ("online\n", 0)
    started = time.perf_counter()
    asyncio.run(online_pool.set_network(sb, False))
    off_s = time.perf_counter() - started
    out, code = sh(sb, EGRESS_PROBE)
    assert code != 0 and "online" not in out, "still online after disconnect"
    assert checked(sb, "echo still-answering") == "still-answering", "the socket outlives the network"
    started = time.perf_counter()
    asyncio.run(online_pool.set_network(sb, True))
    on_s = time.perf_counter() - started
    assert sh(sb, EGRESS_PROBE) == ("online\n", 0)

    offline = asyncio.run(offline_pool.spawn(IMAGE))
    out, code = sh(offline, EGRESS_PROBE)
    assert code != 0 and "online" not in out, "an offline spawn's first command ran online"
    asyncio.run(offline_pool.set_network(offline, True))
    assert sh(offline, EGRESS_PROBE) == ("online\n", 0)
    print("\nnetwork off %.3f s, on %.3f s" % (off_s, on_s))


def test_each_sandbox_is_alone_on_a_network_of_its_own(cleanup):
    """On one bridge shared by every sandbox, each container that joined or
    left moved the gateway's MAC under the others and their lookups timed out
    for seconds. Each sandbox now has its own network, made without podman's
    DNS server and isolated from every other sandbox's."""
    pool = PodmanPool(STORE, RUNTIME_BIN)
    cleanup.append(pool)
    a, b = (asyncio.run(pool.spawn(IMAGE)) for _ in range(2))
    for sb in (a, b):
        joined = _podman("inspect", "--type", "container", "--format",
                         "{{range $name, $_ := .NetworkSettings.Networks}}{{$name}} {{end}}", sb.sandbox_id)
        assert joined.split() == [sb.sandbox_id]
    [network] = json.loads(_podman("network", "inspect", a.sandbox_id))
    assert network["dns_enabled"] is False and network["options"]["isolate"] == "true"
    assert network["labels"] == {"sprout.managed": "1", "sprout.store": str(STORE.resolve())}
    assert network["subnets"][0]["gateway"] not in checked(a, "cat /etc/resolv.conf"), \
        "the host's resolvers, not a podman DNS server on the gateway"
    assert checked(a, "python3 -c \"import socket; print(bool(socket.getaddrinfo('pypi.org', 443)))\"") == "True"
    address = _podman("inspect", "--type", "container", "--format",
                      "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", b.sandbox_id).strip()
    reach = ("python3 -c \"import socket\ns = socket.socket(); s.settimeout(3)\ntry:\n"
             "    s.connect(('%s', 9)); print('connected')\nexcept OSError as e:\n"
             "    print(type(e).__name__)\"" % address)
    assert checked(a, reach) == "TimeoutError", "dropped between networks, not refused by the other sandbox"
    first = a.sandbox_id
    asyncio.run(pool.destroy(a))
    assert first not in sandbox_networks() and b.sandbox_id in sandbox_networks()


def _built_image(tag: str, dockerfile: str, context: Path):
    """Build a local image; yield it; remove it and the base extracted from it."""
    subprocess.run([PODMAN, "build", "-q", "-t", tag, "-f", "-", str(context)],
                   input=dockerfile, text=True, check=True, capture_output=True)
    pool = PodmanPool(STORE, RUNTIME_BIN)
    try:
        yield tag
    finally:
        base = STORE / "bases" / _digest(pool, tag)
        asyncio.run(pool.store.discard(base))
        subprocess.run([PODMAN, "rmi", "-f", tag], capture_output=True)


USER_IMAGE = "localhost/sprout-test-user:latest"


@pytest.fixture
def user_image(tmp_path: Path):
    """The base image running as ``nobody`` in ``/tmp``."""
    yield from _built_image(USER_IMAGE, "FROM %s\nUSER nobody\nWORKDIR /tmp\n" % IMAGE, tmp_path)


def test_the_runtime_runs_as_root_in_the_images_working_directory_whatever_its_user(user_image, cleanup):
    """As in a microVM: the runtime is root (this user, under rootless podman)
    and the harness drops to the image's user per command. Run as the image's
    user, the runtime could not do Harbor's root steps (`mkdir /logs`, an
    overlay's RUN, `runuser`), and its socket belonged to a subordinate uid
    the host had to chown from inside the namespace."""
    pool = PodmanPool(STORE, RUNTIME_BIN)
    cleanup.append(pool)
    sb = asyncio.run(pool.spawn(user_image))
    assert checked(sb, "id -u && pwd") == "0\n/tmp", "root, in the image's WORKDIR"
    assert checked(sb, "runuser -u nobody -- id -u") == "65534", "and it can become the image's user"
    socket_path = STORE / "sockets" / sb.sandbox_id / "runtime.sock"
    assert socket_path.stat().st_uid == os.getuid() and socket_path.stat().st_mode & 0o777 == 0o660
    saved = _config_path(STORE / "sandboxes" / sb.sandbox_id)
    assert '"User": "nobody"' in saved.read_text(), "the image's user is kept for the harness"


ADVERSARIAL_IMAGE = "localhost/sprout-test-fidelity:latest"
#: Everything an extraction or a snapshot could lose: ownership across the
#: user namespace, a 0700 directory of another user, setuid/setgid bits, a
#: hardlink pair, relative/absolute/dangling symlinks, a file capability and a
#: user xattr, a sparse file, a FIFO, a name with spaces and non-ASCII, an empty
#: directory, a VOLUME path written to after the VOLUME.
ADVERSARIAL_DOCKERFILE = """FROM %s
RUN mkdir -p /data /adv /adv/dir1000 "/adv/odd name" && \\
    echo seed > /data/seed.txt && \\
    echo shared > /adv/h1 && ln /adv/h1 /adv/h2 && \\
    ln -s h1 /adv/rel-link && ln -s /adv/h1 /adv/abs-link && ln -s /nowhere /adv/dangling && \\
    cp /bin/true /adv/suid && chmod 4755 /adv/suid && \\
    cp /bin/true /adv/sgid && chmod 2755 /adv/sgid && \\
    cp /bin/true /adv/cap && \\
    python3 -c "import os; os.setxattr('/adv/cap', 'security.capability', bytes.fromhex('0100000200200000' + '0' * 24))" && \\
    echo xattr > /adv/xattr && python3 -c "import os; os.setxattr('/adv/xattr', 'user.sprout', b'review')" && \\
    truncate -s 64M /adv/sparse && printf mid | dd of=/adv/sparse bs=1 seek=33554432 conv=notrunc status=none && \\
    echo owned > /adv/owned && chown 1000:1000 /adv/owned && chmod 600 /adv/owned && \\
    chown 1000:1000 /adv/dir1000 && chmod 700 /adv/dir1000 && echo private > /adv/dir1000/secret && \\
    chown 1000:1000 /adv/dir1000/secret && \\
    echo nobody > /adv/nobody && chown 65534:65534 /adv/nobody && \\
    mkfifo /adv/fifo && echo "unicode" > "/adv/odd name/fïlé with spaces" && mkdir /adv/emptydir
VOLUME /data
RUN echo after-volume > /data/after.txt
""" % IMAGE

#: One line per entry of a tree: type, mode, uid, gid, size, hardlink group
#: (the first path seen with that inode, so two trees with different inode
#: numbers but the same link structure compare equal), symlink target,
#: sparse or dense, every xattr, and a content hash. Run under `podman unshare`
#: so files of mapped uids can be read.
FINGERPRINT = r'''
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
'''


def _fingerprint(script: Path, root: str) -> dict:
    """``{relative path: attributes}`` of a tree, read from inside the user namespace.

    ``root`` is a shell word evaluated in there, so it may be ``$(podman image
    mount ...)``, whose mount is visible only inside the namespace.
    """
    out = subprocess.run([PODMAN, "unshare", "sh", "-c", "python3 %s %s" % (script, root)],
                         capture_output=True, text=True, check=True).stdout
    return dict(line.split("\t", 1) for line in out.splitlines())


@pytest.fixture
def adversarial_image(tmp_path: Path):
    yield from _built_image(ADVERSARIAL_IMAGE, ADVERSARIAL_DOCKERFILE, tmp_path)


def test_an_extracted_base_and_a_branch_carry_every_attribute_of_their_source(adversarial_image, cleanup, tmp_path):
    """The base is compared with podman's own mount of the image, entry by
    entry; a branch's rootfs with the read-only snapshot it was spawned from.
    Extended attributes are the case that once shipped green: tar restores
    them only when asked, and without them every setcap'd binary in every
    sandbox of the image had lost its capabilities."""
    script = tmp_path / "fingerprint.py"
    script.write_text(FINGERPRINT)
    pool = PodmanPool(STORE, RUNTIME_BIN)
    cleanup.append(pool)
    started = time.perf_counter()
    parent = asyncio.run(pool.spawn(adversarial_image))
    extraction_s = time.perf_counter() - started
    base = STORE / "bases" / _digest(pool, adversarial_image)
    image = _fingerprint(script, "$(%s image mount %s)" % (shlex.quote(PODMAN), shlex.quote(adversarial_image)))
    subprocess.run([PODMAN, "unshare", PODMAN, "image", "unmount", adversarial_image], capture_output=True)
    extracted = _fingerprint(script, str(base))
    assert len(image) > 5000 and set(image) == set(extracted), "the same entries, none missing or added"
    differing = [(image[k], extracted[k]) for k in image if image[k] != extracted[k]]
    assert differing == [], "%d of %d entries differ, e.g.:\n%s" % (
        len(differing), len(image), "\n".join("image: %s\nbase:  %s" % pair for pair in differing[:5]))
    assert "security.capability=0100000200200000" in extracted["adv/cap"], "the file capability survived tar"
    assert "user.sprout=" in extracted["adv/xattr"]
    assert extracted["adv/suid"].split()[1] == "4755" and extracted["adv/h2"].split()[5] == "adv/h1"
    assert extracted["adv/owned"].split()[2:4] == ["1000", "1000"], "owned by uid 1000 inside the namespace"
    [odd] = [key for key in extracted if key.startswith("adv/odd name/")]
    assert extracted["adv/fifo"].split()[0] == "p" and extracted[odd].split()[0] == "f" and "with spaces" in odd

    assert checked(parent, "getcap /adv/cap 2>/dev/null || python3 -c \"import os; print(os.getxattr('/adv/cap', "
                           "'security.capability').hex())\"") != "", "the capability is visible in the container"
    checked(parent, "python3 -c \"import os; os.setxattr('/adv/xattr', 'user.second', b'2')\" && echo more > /adv/new")
    snap = asyncio.run(pool.snapshot(parent))
    branch = asyncio.run(pool.spawn(snap.id))
    snapshot_fp = _fingerprint(script, str(STORE / "snapshots" / snap.id))
    branch_fp = _fingerprint(script, str(STORE / "sandboxes" / branch.sandbox_id))
    assert snapshot_fp == branch_fp, "a branch's rootfs is its snapshot, attribute for attribute"
    assert "user.second=32" in branch_fp["adv/xattr"] and "adv/new" in branch_fp, \
        "what the parent wrote after the base, xattrs included, reached the branch"
    asyncio.run(pool.destroy(branch, parent))
    asyncio.run(pool.delete_snapshot(snap.id))
    print("\nadversarial image: extraction %.1f s, %d entries compared, odd name %r" % (extraction_s, len(image), odd))


#: BTRFS_IOC_SUBVOL_CREATE on a directory fd: _IOW(0x94, 14, struct btrfs_ioctl_vol_args) -- 4096 bytes,
#: a 64-bit fd and a 4088-byte name. Unprivileged inside the container, as it is for any btrfs user.
MAKE_NESTED = ("python3 -c \"import fcntl, os, struct; fd = os.open('%s', os.O_RDONLY | os.O_DIRECTORY); "
               "fcntl.ioctl(fd, 0x5000940e, bytearray(struct.pack('q4088s', 0, b'%s')), True)\"")


def test_a_subvolume_made_inside_the_container_refuses_the_snapshot_and_is_destroyed(cleanup):
    """`btrfs subvolume snapshot` is not recursive: a nested subvolume would be
    an empty directory in the snapshot, and a branch from it would silently
    lack those files. One is made in a 0700 directory of another user, which
    only the namespace's root can see -- the case a host-side `find` misses."""
    pool = PodmanPool(STORE, RUNTIME_BIN)
    cleanup.append(pool)
    sb = asyncio.run(pool.spawn(IMAGE))
    sid = sb.sandbox_id
    started = time.perf_counter()
    plain = asyncio.run(pool.snapshot(sb))
    plain_s = time.perf_counter() - started
    checked(sb, MAKE_NESTED % ("/tmp", "nested") + " && echo important > /tmp/nested/important.txt")
    checked(sb, "mkdir -m 700 /tmp/private && chown 1000:1000 /tmp/private && " + MAKE_NESTED % ("/tmp/private", "hidden"))
    assert checked(sb, "stat -c %i /tmp/nested /tmp/private/hidden") == "256\n256", "subvolume roots"
    # The store may hold any number of other snapshots (a finished run's): what
    # counts is that the refused one added nothing.
    before = _entries(STORE / "snapshots")
    with pytest.raises(PodmanError, match=r"holds nested subvolumes \(/tmp/private/hidden, /tmp/nested\); "
                                          r"a snapshot would omit their contents"):
        asyncio.run(pool.snapshot(sb))
    assert _entries(STORE / "snapshots") == before, "no snapshot was taken"
    assert checked(sb, "cat /tmp/nested/important.txt") == "important", "the sandbox itself is untouched"
    asyncio.run(pool.destroy(sb))
    assert pool.list() == [] and not (STORE / "sandboxes" / sid).exists(), "nested subvolumes deleted first"
    assert sid not in _podman("ps", "-a", "--format", "{{.Names}}")
    asyncio.run(pool.delete_snapshot(plain.id))
    print("\nsnapshot with the nested-subvolume check: %.3f s" % plain_s)


def _digest(pool: PodmanPool, image: str) -> str:
    return asyncio.run(image_info(image, podman_bin=pool.podman_bin))["Id"]

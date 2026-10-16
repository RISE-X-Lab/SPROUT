"""PodmanPool, BtrfsStore and HttpBackend: the container backend's client contract.

The fake ``podman`` and ``btrfs`` from ``tests/conftest.py`` stand in for the
real ones and record every argv. What is verified here is the exact command
line -- the rootfs, network, mounts, entrypoint, image env and user of a
``podman run``; the network each sandbox gets of its own; the
create/disconnect/start of an offline spawn; the read-only
flag and name of a snapshot; the order of a teardown -- and how each tool's
output is read. ``tests/sdk/test_podman_live.py`` runs the same code against
podman and a btrfs store.
"""

from __future__ import annotations

import asyncio
import os
import hashlib
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

from sprout_sandbox import HttpBackend, PodmanPool, Snapshot
from sprout_sandbox.btrfs_store import config_path, mount_problem
from sprout_sandbox.podman_cli import PodmanError, default_podman_bin
from sprout_sandbox.podman_containers import RUNTIME_PATH, PodmanContainers
from sprout_sandbox.podman_images import EXTRACTION_FORMAT
from tests.conftest import (FAKE_EXPORT_BYTES, FAKE_EXTRACTION_CONTAINER, fake_networks, recorded_calls,
                            recorded_cwds, recorded_sequence)

IMAGE = "docker.io/library/python:3.11-slim"


def image_id(reference: str = IMAGE) -> str:
    """The id the fake podman reports for a reference."""
    return hashlib.sha256(reference.encode()).hexdigest()


def podman_calls(tmp_path: Path) -> list[list[str]]:
    return recorded_calls(tmp_path, "podman")


def btrfs_calls(tmp_path: Path) -> list[list[str]]:
    return recorded_calls(tmp_path, "btrfs")


def pool(store: Path, runtime_bin: Path, **kwargs) -> PodmanPool:
    return PodmanPool(store, runtime_bin, **kwargs)


@pytest.fixture
def ready(monkeypatch):
    """The readiness probe, made to succeed: no runtime listens behind the fake."""
    probed = []

    async def _ready(self, sandbox):
        probed.append(sandbox)

    monkeypatch.setattr(PodmanContainers, "wait_for_runtime", _ready)
    return probed


@pytest.fixture
def tools(fake_podman, fake_btrfs):
    """Both fakes on PATH."""


# --------------------------------------------------------------------------- #
#  Construction
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("missing, message", [
    ("runtime_bin", "runtime_bin does not exist"),
    ("store", "store does not exist"),
])
def test_a_missing_runtime_binary_or_store_is_refused(store, runtime_bin, tmp_path, tools, missing, message):
    given = {"store": store, "runtime_bin": runtime_bin, missing: tmp_path / "absent"}
    with pytest.raises(ValueError, match=message):
        PodmanPool(given["store"], given["runtime_bin"])


def test_a_store_off_btrfs_or_without_user_subvol_rm_allowed_is_refused(store, runtime_bin, tools, monkeypatch):
    """The store is a plain directory until findmnt says otherwise. On ext4 a
    pool would create directories and fail at the first `subvolume snapshot`
    with an error about the image; on btrfs mounted without
    user_subvol_rm_allowed every spawn and snapshot would work and the first
    destroy fail after `podman rm -f`, leaving the rootfs. Both are refused at
    construction, with the same words the live tests skip on."""
    monkeypatch.setenv("FAKE_FINDMNT_FSTYPE", "ext4")
    monkeypatch.setenv("FAKE_FINDMNT_OPTIONS", "rw,relatime")
    with pytest.raises(ValueError, match=r"store .* is on ext4 mounted with rw,relatime; the podman backend "
                                          r"needs btrfs mounted with user_subvol_rm_allowed .*ops/btrfs_store.sh"):
        pool(store, runtime_bin)
    monkeypatch.setenv("FAKE_FINDMNT_FSTYPE", "btrfs")
    monkeypatch.setenv("FAKE_FINDMNT_OPTIONS", "rw,noatime,subvolid=5,subvol=/")
    with pytest.raises(ValueError, match=r"is on btrfs mounted with rw,noatime,subvolid=5,subvol=/; .*"
                                          r"needs btrfs mounted with user_subvol_rm_allowed") as refused:
        pool(store, runtime_bin)
    assert mount_problem(store) == str(refused.value), "the pool refuses exactly what the live tests skip on"
    assert recorded_calls(store.parent, "findmnt") == [["-no", "FSTYPE,OPTIONS", "--target", str(store)]] * 3
    assert btrfs_calls(store.parent) == [] and not (store / "bases").exists(), \
        "nothing was asked of btrfs or created in a store that was refused"
    monkeypatch.delenv("FAKE_FINDMNT_OPTIONS")
    assert mount_problem(store) is None and pool(store, runtime_bin).store.path == store


def test_capabilities(store, runtime_bin, tools):
    p = pool(store, runtime_bin)
    assert p.supports_snapshot() and p.supports_cold_start()
    assert p.supports_upload() and p.supports_download()


# --------------------------------------------------------------------------- #
#  Base extraction
# --------------------------------------------------------------------------- #

def test_the_first_spawn_extracts_the_base_once_per_image(store, runtime_bin, tools, ready):
    p = pool(store, runtime_bin)

    async def scenario():
        a = await p.spawn_from_image(IMAGE)
        b = await p.spawn_from_image(IMAGE)
        c = await p.spawn_from_image("docker.io/library/alpine:3.19")
        return a, b, c

    a, b, c = asyncio.run(scenario())
    base = store / "bases" / image_id()
    assert base.is_dir() and (base / "etc" / "extracted").read_text() == str(FAKE_EXPORT_BYTES), \
        "the whole export reached tar"
    assert not list((store / "bases").glob("*.partial-*")), "the partial name is gone once published"
    saved = json.loads(config_path(base).read_text())
    assert saved == {
        "image": IMAGE, "digest": image_id(), "extract": EXTRACTION_FORMAT,
        "repo_digests": ["docker.io/library/python@sha256:" + image_id()],
        "config": {"Env": ["PATH=/usr/local/bin:/usr/bin", "LANG=C.UTF-8", "FOO=a=b c"],
                   "WorkingDir": "/testbed", "User": "", "Entrypoint": None, "Cmd": ["python3"]},
    }, "the OCI config a --rootfs container needs is saved beside the base, with the extraction format"

    seen = podman_calls(store.parent)
    extractions = [c for c in seen if c[0] == "create"]
    assert extractions == [
        ["create", "--entrypoint", "/bin/true", IMAGE],
        ["create", "--entrypoint", "/bin/true", "docker.io/library/alpine:3.19"],
    ], "one extraction per image id; the second spawn of the same image reused the base"
    exports = [c for c in seen if c[0] == "export"]
    assert exports == [["export", FAKE_EXTRACTION_CONTAINER]] * 2
    first_tar = [c for c in seen if c[:2] == ["unshare", "tar"]][0]
    assert first_tar[:8] == ["unshare", "tar", "-x", "--numeric-owner", "--xattrs", "--xattrs-include=*",
                             "--acls", "-C"], \
        "tar applies the archive's xattrs (file capabilities among them) and ACLs only when asked"
    assert first_tar[8].startswith(str(base) + ".partial-"), \
        "tar runs inside the user namespace, into the partial subvolume that is then renamed"
    assert ["rm", "-f", FAKE_EXTRACTION_CONTAINER] in seen, "the export container is removed"

    btrfs = btrfs_calls(store.parent)
    creates = [c for c in btrfs if c[:2] == ["subvolume", "create"]]
    assert len(creates) == 2 and all(".partial-" in c[2] for c in creates)
    assert ["property", "set", str(base), "ro", "true"] in btrfs, "a published base is read-only"
    sequence = recorded_sequence(store.parent)
    exported = sequence.index(["podman", "export", FAKE_EXTRACTION_CONTAINER])
    removed = sequence.index(["podman", "rm", "-f", FAKE_EXTRACTION_CONTAINER])
    published = sequence.index(["btrfs", "property", "set", str(base), "ro", "true"])
    assert exported < removed < published, "export, then the container goes, then the base is published read-only"
    snapshots = [c for c in btrfs if c[:2] == ["subvolume", "snapshot"]]
    assert [c[2] for c in snapshots] == [str(base), str(base),
                                         str(store / "bases" / image_id("docker.io/library/alpine:3.19"))]
    assert [c[3] for c in snapshots] == [str(store / "sandboxes" / sb.sandbox_id) for sb in (a, b, c)], \
        "each sandbox's rootfs is a writable snapshot of its base"
    assert len({a.sandbox_id, b.sandbox_id, c.sandbox_id}) == 3


def test_a_docker_short_name_reaches_podman_qualified(store, runtime_bin, tools, ready):
    """Harbor tasks name images the Docker way (TerminalBench: alexgshaw/<task>:<date>);
    podman without a search registry refuses that outright, so every podman
    command sees the docker.io reference Docker itself would pull."""
    asyncio.run(pool(store, runtime_bin).spawn_from_image("alexgshaw/bn-fit-modify:20251031"))
    named = [c for c in podman_calls(store.parent) if c[0] in ("pull", "create") or c[:2] == ["image", "inspect"]]
    assert named and all(c[-1] == "docker.io/alexgshaw/bn-fit-modify:20251031" for c in named), named


def test_a_base_root_is_0755_whatever_the_umask(store, runtime_bin, tools, ready):
    """The export carries no entry for "/", so the base's root mode was the
    extracting process's umask: 0700 under 077, where an image's non-root user
    could not reach a single file of its own rootfs."""
    previous = os.umask(0o077)
    try:
        asyncio.run(pool(store, runtime_bin).spawn_from_image(IMAGE))
    finally:
        os.umask(previous)
    assert (store / "bases" / image_id()).stat().st_mode & 0o777 == 0o755


def test_prepare_pulls_and_extracts_the_base_and_starts_nothing(store, runtime_bin, tools, ready, monkeypatch):
    """A batch pays for every image before its agents start: the same base the first spawn would make."""
    monkeypatch.setenv("FAKE_PODMAN_UNKNOWN", IMAGE)
    p = pool(store, runtime_bin)

    async def scenario():
        first = await p.prepare(IMAGE)
        second = await p.prepare(IMAGE)
        sandbox = await p.spawn_from_image(IMAGE)
        return first, second, sandbox

    first, second, sandbox = asyncio.run(scenario())
    base = store / "bases" / image_id()
    assert first == second == base and base.is_dir()
    assert json.loads(config_path(base).read_text())["extract"] == EXTRACTION_FORMAT
    seen = podman_calls(store.parent)
    assert ["pull", IMAGE] in seen, "an image that is not local is pulled"
    assert [c for c in seen if c[0] == "create"] == [["create", "--entrypoint", "/bin/true", IMAGE]], \
        "one extraction; the second prepare and the spawn both found the base"
    runs = [c for c in seen if c[0] in ("run", "start")]
    assert len(runs) == 1 and str(store / "sandboxes" / sandbox.sandbox_id) in runs[0], \
        "prepare started no container; only the spawn did"
    assert not ready[:-1], "the readiness probe ran for the spawn alone"


def test_a_base_extracted_by_an_older_format_is_extracted_again(store, runtime_bin, tools, ready):
    """Format 1 lost every extended attribute. A base is keyed by image id
    alone, so an old one would serve every sandbox of that image for ever; its
    record says which format made it, and one that says less is replaced. The
    sandboxes and snapshots taken from it are their own subvolumes."""
    base = store / "bases" / image_id()
    (store / "bases").mkdir()
    base.mkdir()
    (base / "etc").mkdir()
    (base / "etc" / "old").write_text("format 1")
    config_path(base).write_text(json.dumps({"image": IMAGE, "digest": image_id(),
                                              "config": {"Env": [], "WorkingDir": "/", "User": ""}}))
    p = pool(store, runtime_bin)
    asyncio.run(p.spawn(IMAGE))
    assert not (base / "etc" / "old").exists() and (base / "etc" / "extracted").exists(), "extracted anew"
    assert json.loads(config_path(base).read_text())["extract"] == EXTRACTION_FORMAT
    sequence = recorded_sequence(store.parent)
    deleted = sequence.index(["btrfs", "subvolume", "delete", str(base)])
    created = next(i for i, c in enumerate(sequence) if c[:3] == ["btrfs", "subvolume", "create"])
    assert deleted < created, "the stale base goes before the new one is extracted"
    before = len(podman_calls(store.parent))
    asyncio.run(p.spawn(IMAGE))
    assert not [c for c in podman_calls(store.parent)[before:] if c[0] in ("create", "export")], \
        "a base of the current format is reused"


def test_an_image_that_is_not_local_is_pulled_first(store, runtime_bin, tools, ready, monkeypatch):
    monkeypatch.setenv("FAKE_PODMAN_UNKNOWN", IMAGE)
    p = pool(store, runtime_bin)
    asyncio.run(p.spawn(IMAGE))
    seen = podman_calls(store.parent)
    inspect = ["image", "inspect", "--format", "json", IMAGE]
    assert seen[:3] == [inspect, ["pull", IMAGE], inspect]


def test_a_failed_extraction_leaves_no_partial_base(store, runtime_bin, tools, ready, monkeypatch):
    monkeypatch.setenv("FAKE_PODMAN_FAIL", "export")
    p = pool(store, runtime_bin)
    with pytest.raises(PodmanError, match=r"podman export failed \(exit 125\): Error: simulated"):
        asyncio.run(p.spawn(IMAGE))
    assert list((store / "bases").iterdir()) == [], "the partial subvolume and its config are gone"
    assert ["rm", "-f", FAKE_EXTRACTION_CONTAINER] in podman_calls(store.parent)
    assert p.list() == []


# --------------------------------------------------------------------------- #
#  Spawning
# --------------------------------------------------------------------------- #

def test_spawn_builds_the_exact_run_argv_and_attaches_over_the_socket(
        store, runtime_bin, tools, ready, monkeypatch):
    """The image's User is saved for the harness and NOT passed: the runtime
    runs as the container's root (this user), as it does in a microVM, and
    shell_context drops to the image's user per command. Passed, it made
    every root step of a Harbor trial (`mkdir /logs`, an overlay's RUN,
    `runuser`) fail on a non-root image."""
    monkeypatch.setenv("FAKE_PODMAN_USER", "appuser")
    p = pool(store, runtime_bin)
    sb = asyncio.run(p.spawn(IMAGE, env={"A": "1", "B": "x y"}, agent_id="agent",
                             resources={"cpu": 2, "memory_mb": 4096, "disk_size_mb": 65536}))
    sid = sb.sandbox_id
    assert sid.startswith("sprout-") and len(sid) == len("sprout-") + 12
    assert sb.agent_id == "agent"
    assert isinstance(sb.backend, HttpBackend)
    assert sb.backend.uds == str(store / "sockets" / sid / "runtime.sock")
    assert p.list() == [sb]
    assert ready == [sb], "adopted only after the runtime answered"
    assert (store / "sockets" / sid).is_dir() and not (store / "sockets" / sid).stat().st_mode & 0o002, \
        "the runtime binds the socket as the container's root, which owns the directory: no world-writable hand-over"
    saved = json.loads(config_path(store / "sandboxes" / sid).read_text())
    assert saved["digest"] == image_id() and saved["config"]["User"] == "appuser", \
        "the sandbox carries its image config, User included, for the snapshots taken of it and for the harness"

    calls = podman_calls(store.parent)
    [network] = [c for c in calls if c[:2] == ["network", "create"]]
    assert network == ["network", "create", "--disable-dns", "--opt", "isolate=true",
                       "--label", "sprout.managed=1", "--label", "sprout.store=%s" % store, sid], \
        "a network of its own, named after it and labelled like it: no podman DNS, no route to other sandboxes"
    [run] = [c for c in calls if c[0] == "run"]
    assert calls.index(network) < calls.index(run)
    assert run == [
        "run", "-d", "--name", sid, "--network", sid,
        "--label", "sprout.managed=1", "--label", "sprout.store=%s" % store,
        "-v", "%s:/run/sprout" % (store / "sockets" / sid),
        "-v", "%s:%s:ro" % (runtime_bin, RUNTIME_PATH),
        "--entrypoint", RUNTIME_PATH,
        "-e", "PATH=/usr/local/bin:/usr/bin", "-e", "LANG=C.UTF-8", "-e", "FOO=a=b c",
        "-e", "A=1", "-e", "B=x y",
        "--workdir", "/testbed",
        "--cpus", "2", "--memory", "4096m",
        "--rootfs", str(store / "sandboxes" / sid), "--listen", "unix:/run/sprout/runtime.sock",
    ], ("options before --rootfs, whose path takes the image's positional slot; the image's "
        "env and workdir are passed since podman reads no config for a rootfs, its user is not; "
        "disk_size_mb has no counterpart")
    assert not [c for c in podman_calls(store.parent) if c[:2] == ["unshare", "chown"]], \
        "no socket hand-over: the runtime's socket is this user's from the start"


def test_no_shape_means_unlimited_and_an_empty_name_is_refused(store, runtime_bin, tools, ready):
    p = pool(store, runtime_bin)
    sb = asyncio.run(p.spawn_from_image(IMAGE))
    assert ready == [sb]
    [run] = [c for c in podman_calls(store.parent) if c[0] == "run"]
    assert "--user" not in run and "--cpus" not in run and "--memory" not in run
    with pytest.raises(ValueError, match="image reference or a snapshot id"):
        asyncio.run(p.spawn(""))


def test_an_offline_spawn_is_create_disconnect_start(store, runtime_bin, tools, ready):
    """`run` would give the container its network for an instant; created,
    disconnected and then started, its first command already runs offline."""
    p = pool(store, runtime_bin, allow_internet=False)
    sb = asyncio.run(p.spawn(IMAGE))
    sid = sb.sandbox_id
    seen = podman_calls(store.parent)
    [create] = [c for c in seen if c[0] == "create" and "--name" in c]
    assert create[:4] == ["create", "--name", sid, "--network"] and "-d" not in create
    assert create[-4:] == ["--rootfs", str(store / "sandboxes" / sid), "--listen", "unix:/run/sprout/runtime.sock"]
    tail = seen[seen.index(create) + 1:]
    assert tail[:2] == [["network", "disconnect", sid, sid], ["start", sid]]
    assert not [c for c in seen if c[0] == "run"]

    asyncio.run(p.set_network(sb, False))
    assert len(podman_calls(store.parent)) == len(seen), "already offline: nothing to do"
    asyncio.run(p.set_network(sb, True))
    assert podman_calls(store.parent)[-1] == ["network", "connect", sid, sid]
    assert sid in fake_networks(store.parent), "disconnected from its network, which stays its own"


def test_each_sandbox_has_a_network_of_its_own_that_goes_with_it(store, runtime_bin, tools, ready):
    """One bridge shared by every sandbox moved its MAC under all of them
    whenever one joined or left, and their packets went nowhere for seconds
    (DNS lookups failed in 42 of 176 gate trials at 32 workers). A bridge of
    one's own changes only with its own sandbox."""
    p = pool(store, runtime_bin)
    a, b = spawned(p), spawned(p)
    first, second = a.sandbox_id, b.sandbox_id
    assert set(fake_networks(store.parent)) == {first, second}
    asyncio.run(p.destroy(a))
    assert set(fake_networks(store.parent)) == {second}, "the other sandbox keeps its network"
    sequence = recorded_sequence(store.parent)
    assert sequence.index(["podman", "rm", "-f", first]) < sequence.index(
        ["podman", "network", "rm", first]), "a network is not removed while a container is on it"
    assert not [c for c in podman_calls(store.parent) if c[:2] == ["network", "rm"] and "-f" in c], \
        "never `network rm -f`, which removes the containers on it"
    asyncio.run(p.close())
    assert fake_networks(store.parent) == {}


def test_podman_runs_from_the_store_and_a_container_from_its_own_directory(store, runtime_bin, tools, ready,
                                                                             tmp_path, monkeypatch):
    """conmon inherits podman's working directory and writes an `oom` marker there when a
    process in the container is OOM-killed; run from the caller's cwd, that once dirtied a
    run's source checkout. Every podman call runs from the store, and the container's
    run/create/start from its socket directory, where the marker is read and cleaned up."""
    elsewhere = tmp_path / "callers-cwd"
    elsewhere.mkdir()
    monkeypatch.chdir(elsewhere)
    p = pool(store, runtime_bin, allow_internet=False)
    sb = asyncio.run(p.spawn(IMAGE))
    cwds = recorded_cwds(store.parent)
    assert cwds, "the fake recorded working directories"
    assert all(Path(cwd).resolve() != elsewhere.resolve() for _, cwd in cwds), \
        "no podman command ran from the caller's directory"
    socket_dir = (store / "sockets" / sb.sandbox_id).resolve()
    container_cmds = {cmd: Path(cwd).resolve() for cmd, cwd in cwds if cmd in ("create", "start", "run")}
    assert container_cmds == {"create": socket_dir, "start": socket_dir}, \
        "the container is created and started from its own socket directory"
    assert {Path(cwd).resolve() for cmd, cwd in cwds if cmd not in ("create", "start", "run")} == {store.resolve()}, \
        "everything else runs from the store"
    assert not p._containers.oom_killed(sb.sandbox_id)
    (socket_dir / "oom").touch()                      # what conmon does on an OOM kill
    assert p._containers.oom_killed(sb.sandbox_id)
    diagnosis = asyncio.run(p._containers.diagnosis(sb.sandbox_id))
    assert "OOM-killed" in diagnosis and "oom marker" in diagnosis
    asyncio.run(p.destroy(sb))
    assert not socket_dir.exists(), "the marker goes with the socket directory"


def test_a_container_whose_runtime_never_answers_is_removed_with_its_rootfs(
        store, runtime_bin, tools, monkeypatch):
    """No runtime listens behind the fake, so the probe fails until the deadline;
    the caller gets the error and neither the container, nor its subvolume, nor
    its socket directory stays behind."""
    monkeypatch.setattr(PodmanContainers, "READY_TIMEOUT_S", 0.3)
    monkeypatch.setattr(PodmanContainers, "READY_POLL_S", 0.05)
    p = pool(store, runtime_bin)
    with pytest.raises(PodmanError, match=r"did not answer within 0 s.*fake logs"):
        asyncio.run(p.spawn(IMAGE))
    assert p.list() == []
    seen = podman_calls(store.parent)
    [rm] = [c for c in seen if c[0] == "rm" and c[-1].startswith("sprout-")]
    sid = rm[-1]
    assert not (store / "sandboxes" / sid).exists() and not config_path(store / "sandboxes" / sid).exists()
    assert not (store / "sockets" / sid).exists()
    assert fake_networks(store.parent) == {}, "nor its network"
    deletes = [c for c in btrfs_calls(store.parent) if c[:2] == ["subvolume", "delete"]]
    assert deletes == [["subvolume", "delete", str(store / "sandboxes" / sid)]]


def test_a_container_that_exits_before_answering_names_the_cause(store, runtime_bin, tools, monkeypatch):
    """The runtime failed to exec (seen live: the glibc-linked binary in alpine).
    Reported at once, with the state and the logs, not after the deadline."""
    monkeypatch.setenv("FAKE_PODMAN_STATE", "exited 127")
    monkeypatch.setattr(PodmanContainers, "READY_TIMEOUT_S", 30.0)
    p = pool(store, runtime_bin)
    with pytest.raises(PodmanError, match=r"exited before answering.*exited \(exit 127\).*fake logs"):
        asyncio.run(p.spawn("docker.io/library/alpine:3.19"))
    assert p.list() == [] and list((store / "sandboxes").iterdir()) == []


# --------------------------------------------------------------------------- #
#  Snapshots
# --------------------------------------------------------------------------- #

def spawned(p: PodmanPool):
    return asyncio.run(p.spawn(IMAGE, agent_id="agent"))


def test_a_snapshot_is_a_read_only_subvolume_named_after_the_sandbox(store, runtime_bin, tools, ready):
    p = pool(store, runtime_bin)
    sb = spawned(p)
    sid = sb.sandbox_id
    first = asyncio.run(p.snapshot(sb))
    assert first == Snapshot(id="%s-0001" % sid, rootfs_layers=None, disk_only=True, delta_empty=None), \
        "nothing stacks, so no layer count: the checkpointer never re-boards"
    second = asyncio.run(p.snapshot(sb))
    named = asyncio.run(p.snapshot(sb, name="Run_01:step 7/final"))
    assert second.id == "%s-0002" % sid and named.id == "%s-0003-Run_01-step-7-final" % sid
    snapshots = [c for c in btrfs_calls(store.parent) if c[:2] == ["subvolume", "snapshot"] and "-r" in c]
    assert snapshots == [
        ["subvolume", "snapshot", "-r", str(store / "sandboxes" / sid), str(store / "snapshots" / snap.id)]
        for snap in (first, second, named)]
    for snap in (first, second, named):
        assert (store / "snapshots" / snap.id).is_dir()
        assert json.loads(config_path(store / "snapshots" / snap.id).read_text())["digest"] == image_id(), \
            "a snapshot carries the image config, for the sandboxes spawned from it"
    # A memory snapshot is refused: the session catches this and returns None, and
    # ToolAccess.finalize then journals captured=False for it -- the honest record.
    with pytest.raises(NotImplementedError, match="disk-only"):
        asyncio.run(p.snapshot(sb, disk_only=False))
    taken = {snap.id for snap in (first, second, named)}
    assert {d.name for d in (store / "snapshots").iterdir()} == taken | {i + ".json" for i in taken}, \
        "the refusal made neither a subvolume nor a config"
    assert sb.sandbox_id == sid and p.list() == [sb], "the container is untouched"


def test_spawn_from_a_snapshot_id_snapshots_the_snapshot(store, runtime_bin, tools, ready):
    """A branch: the same operation as a spawn from an image, sourced from the
    read-only snapshot instead of the base, with no image lookup at all."""
    p = pool(store, runtime_bin)
    parent = spawned(p)
    snap = asyncio.run(p.snapshot(parent))
    before = len(podman_calls(store.parent))
    branch = asyncio.run(p.spawn(snap.id, resources={"cpu": 1}))
    assert branch.sandbox_id != parent.sandbox_id
    assert not [c for c in podman_calls(store.parent)[before:] if c[:2] == ["image", "inspect"]]
    [creation] = [c for c in btrfs_calls(store.parent)
                  if c[:2] == ["subvolume", "snapshot"] and c[-1] == str(store / "sandboxes" / branch.sandbox_id)]
    assert creation == ["subvolume", "snapshot", str(store / "snapshots" / snap.id),
                        str(store / "sandboxes" / branch.sandbox_id)]
    run = [c for c in podman_calls(store.parent) if c[0] == "run"][-1]
    assert run[run.index("--cpus") + 1] == "1", "a subvolume carries no shape; the caller passes one"
    assert json.loads(config_path(store / "sandboxes" / branch.sandbox_id).read_text())["digest"] == image_id()


def test_a_snapshot_id_the_store_lacks_is_an_error_not_an_image_to_pull(store, runtime_bin, tools, ready):
    """A branch after `harness sweep`, or on the wrong store: the id names
    nothing here, and handed to podman it would fail with a registry
    short-name error that never mentions snapshots."""
    p = pool(store, runtime_bin)
    for missing in ("sprout-0123456789ab-0007", "sprout-0123456789ab-0007-Run_01-step-7-final"):
        with pytest.raises(PodmanError, match=r"no snapshot 'sprout-0123456789ab-0007.*' in .*snapshots "
                                              r"\(deleted, swept, or taken on another store\)"):
            asyncio.run(p.spawn(missing))
    assert not [c for c in podman_calls(store.parent) if c[0] in ("pull", "image", "create", "run")], \
        "nothing was asked of podman about an image"
    asyncio.run(p.spawn("alpine"))
    assert ["image", "inspect", "--format", "json", "docker.io/library/alpine"] in podman_calls(store.parent), \
        "a bare name that is not one of this pool's ids is still an image reference (Docker's)"


def test_a_rootfs_holding_a_nested_subvolume_is_not_snapshotted_and_is_destroyed_deepest_first(
        store, runtime_bin, tools, ready, monkeypatch):
    """`btrfs subvolume snapshot` is not recursive: a subvolume the container
    made (an unprivileged ioctl) would be an empty directory in the snapshot,
    and a branch from it would silently lack those files. Refused instead, so
    the checkpointer records the step as not captured; destroy then has to
    delete the nested ones first, since a subvolume holding another cannot go."""
    p = pool(store, runtime_bin)
    sb = spawned(p)
    sid = sb.sandbox_id
    rootfs = store / "sandboxes" / sid
    (rootfs / "adv" / "nested" / "deeper").mkdir(parents=True)
    (rootfs / "adv" / "nested" / "important.txt").write_text("kept")
    monkeypatch.setenv("FAKE_PODMAN_NESTED", "adv/nested,adv/nested/deeper")
    with pytest.raises(PodmanError, match=r"rootfs of %s holds nested subvolumes \(/adv/nested/deeper, /adv/nested\); "
                                          r"a snapshot would omit their contents" % sid):
        asyncio.run(p.snapshot(sb))
    assert list((store / "snapshots").iterdir()) == [], "no snapshot, and no config for one"
    assert not [c for c in btrfs_calls(store.parent) if c[:2] == ["subvolume", "snapshot"] and "-r" in c]
    assert podman_calls(store.parent)[-1] == ["unshare", "find", str(rootfs), "-mindepth", "1", "-inum", "256"], \
        "looked for from inside the namespace: a 0700 directory of the container's user is readable only there"
    assert p.list() == [sb], "the sandbox is untouched"

    asyncio.run(p.destroy(sb))
    sequence = recorded_sequence(store.parent)
    order = [sequence.index(step) for step in (
        ["podman", "rm", "-f", sid],
        ["podman", "unshare", "btrfs", "property", "set", str(rootfs / "adv" / "nested" / "deeper"), "ro", "false"],
        ["podman", "unshare", "btrfs", "subvolume", "delete", str(rootfs / "adv" / "nested" / "deeper")],
        ["podman", "unshare", "btrfs", "subvolume", "delete", str(rootfs / "adv" / "nested")],
        ["btrfs", "property", "set", str(rootfs), "ro", "false"],
        ["btrfs", "subvolume", "delete", str(rootfs)])]
    assert order == sorted(order), "container, then the nested subvolumes deepest first, then the rootfs"
    assert not rootfs.exists() and not config_path(rootfs).exists() and p.list() == []


def test_a_destroy_that_fails_after_the_container_is_gone_drops_the_handle_and_names_the_leftover(
        store, runtime_bin, tools, ready, monkeypatch):
    """Half a teardown must not leave the pool holding a sandbox it cannot
    destroy (close would raise again and again) nor hide what remains: the
    error says which rootfs is left and that `harness reap --store` reclaims it."""
    p = pool(store, runtime_bin)
    sb, other = spawned(p), spawned(p)
    sid = sb.sandbox_id
    rootfs = store / "sandboxes" / sid
    monkeypatch.setenv("FAKE_BTRFS_FAIL", "delete")
    with pytest.raises(PodmanError, match=r"container %s is removed but its rootfs %s is not \(btrfs subvolume "
                                          r"delete failed .*\); `harness reap --store %s` reclaims it$"
                                          % (sid, rootfs, store)):
        asyncio.run(p.destroy(sb))
    assert ["rm", "-f", sid] in podman_calls(store.parent) and rootfs.is_dir(), "the container went, the rootfs stayed"
    assert sid not in fake_networks(store.parent), "a rootfs that will not go does not keep the network"
    assert p.list() == [other] and sb.sandbox_id is None, "the handle is dropped all the same"
    monkeypatch.delenv("FAKE_BTRFS_FAIL")
    other_rootfs = store / "sandboxes" / other.sandbox_id
    asyncio.run(p.close())
    assert p.list() == [] and not other_rootfs.exists()
    assert rootfs.is_dir(), "close does not retry what destroy could not do; the reaper finds it in the store"
    monkeypatch.setenv("FAKE_BTRFS_FAIL", "delete")
    p = pool(store, runtime_bin)
    a, b = spawned(p), spawned(p)
    with pytest.raises(PodmanError, match="%s.*%s|%s.*%s" % (a.sandbox_id, b.sandbox_id, b.sandbox_id, a.sandbox_id)):
        asyncio.run(p.close())
    assert p.list() == [], "close tries every sandbox before it raises, and names each leftover"


def test_forget_leaves_the_container_running_and_close_no_longer_reaches_it(store, runtime_bin, tools, ready):
    """A retained sandbox (its final snapshot failed) holds the only copy of
    the rollout: Harbor's stop releases it from the pool, and closing the pool
    -- which destroys everything it still holds -- leaves it alone."""
    p = pool(store, runtime_bin)
    kept, other = spawned(p), spawned(p)
    kept_id, kept_rootfs = kept.sandbox_id, store / "sandboxes" / kept.sandbox_id
    before = len(podman_calls(store.parent))
    asyncio.run(p.forget(kept))
    assert podman_calls(store.parent)[before:] == [], "nothing was asked of podman: the container runs on"
    assert p.list() == [other] and kept.sandbox_id is None, "the handle no longer names it"
    other_id = other.sandbox_id
    asyncio.run(p.forget(kept))
    foreign = p._handle("sprout-000000000000", "")      # another pool's sandbox on the same store
    asyncio.run(p.destroy(foreign))
    assert foreign.sandbox_id == "sprout-000000000000" and podman_calls(store.parent)[before:] == [], \
        "destroy skips a sandbox this pool does not hold, as it skips the forgotten one"
    asyncio.run(p.close())
    assert p.list() == []
    assert [c for c in podman_calls(store.parent) if c[0] == "rm" and c[-1].startswith("sprout-")] == \
        [["rm", "-f", other_id]], "close removed only what the pool still held"
    assert kept_rootfs.is_dir() and config_path(kept_rootfs).is_file() and (store / "sockets" / kept_id).is_dir(), \
        "its rootfs, config and socket directory are all still there for the run that kept it"


def test_delete_snapshot_makes_it_writable_first(store, runtime_bin, tools, ready):
    p = pool(store, runtime_bin)
    sb = spawned(p)
    snap = asyncio.run(p.snapshot(sb))
    path = store / "snapshots" / snap.id
    asyncio.run(p.delete_snapshot(snap.id))
    tail = btrfs_calls(store.parent)[-2:]
    assert tail == [["property", "set", str(path), "ro", "false"], ["subvolume", "delete", str(path)]], \
        "the unprivileged owner may delete a subvolume only while it is writable"
    assert not path.exists() and not config_path(path).exists()
    with pytest.raises(PodmanError, match="no snapshot"):
        asyncio.run(p.delete_snapshot(snap.id))
    for bad in ("", "../x", "a/b", ".."):
        with pytest.raises(ValueError, match="not a snapshot id"):
            asyncio.run(p.delete_snapshot(bad))
    assert sb.sandbox_id and p.list() == [sb], "the sandbox spawned from it is untouched"


def test_a_network_that_will_not_go_does_not_keep_the_rootfs_and_is_named(
        store, runtime_bin, tools, ready, monkeypatch):
    p = pool(store, runtime_bin)
    sb = spawned(p)
    sid = sb.sandbox_id
    monkeypatch.setenv("FAKE_PODMAN_FAIL", "network")
    with pytest.raises(PodmanError, match=r"container %s is removed but its network %s is not \(podman network rm "
                                          r"failed .*\); `harness reap --store %s` reclaims it$" % (sid, sid, store)):
        asyncio.run(p.destroy(sb))
    assert not (store / "sandboxes" / sid).exists() and not (store / "sockets" / sid).exists()
    assert sid in fake_networks(store.parent) and p.list() == [] and sb.sandbox_id is None
    monkeypatch.setenv("FAKE_BTRFS_FAIL", "delete")
    monkeypatch.delenv("FAKE_PODMAN_FAIL")
    other = spawned(p)
    monkeypatch.setenv("FAKE_PODMAN_FAIL", "network")
    with pytest.raises(PodmanError, match=r"its network %s is not .* and its rootfs .* is not .*reclaims them$"
                                          % other.sandbox_id):
        asyncio.run(p.destroy(other))


# --------------------------------------------------------------------------- #
#  Network
# --------------------------------------------------------------------------- #

def test_set_network_toggles_and_reads_podmans_refusals_as_done(store, runtime_bin, tools, ready, monkeypatch):
    p = pool(store, runtime_bin)
    sb = spawned(p)
    sid = sb.sandbox_id
    before = len(podman_calls(store.parent))
    asyncio.run(p.set_network(sb, True))
    assert len(podman_calls(store.parent)) == before, "spawned online: nothing to do"
    asyncio.run(p.set_network(sb, False))
    asyncio.run(p.set_network(sb, True))
    assert podman_calls(store.parent)[before:] == [["network", "disconnect", sid, sid],
                                                    ["network", "connect", sid, sid]]
    monkeypatch.setenv("FAKE_PODMAN_NET_REFUSE", "1")
    asyncio.run(p.set_network(sb, False))        # "is not connected": already what was asked
    asyncio.run(p.set_network(sb, True))         # "already connected"
    assert p._online[sid] is True
    monkeypatch.setenv("FAKE_PODMAN_FAIL", "network")
    with pytest.raises(PodmanError, match="podman network disconnect failed"):
        asyncio.run(p.set_network(sb, False))
    assert p._online[sid] is True, "a refusal for any other reason is an error and changes nothing"


# --------------------------------------------------------------------------- #
#  Teardown and files
# --------------------------------------------------------------------------- #

def test_destroy_removes_the_container_before_its_network_rootfs_and_socket_dir(store, runtime_bin, tools, ready):
    p = pool(store, runtime_bin)
    sb = spawned(p)
    sid = sb.sandbox_id
    podman_before, btrfs_before = len(podman_calls(store.parent)), len(btrfs_calls(store.parent))
    asyncio.run(p.destroy(sb))
    rootfs = store / "sandboxes" / sid
    assert podman_calls(store.parent)[podman_before:] == [
        ["rm", "-f", sid], ["network", "rm", sid],
        ["unshare", "find", str(rootfs), "-mindepth", "1", "-inum", "256"]]
    assert btrfs_calls(store.parent)[btrfs_before:] == [
        ["property", "set", str(rootfs), "ro", "false"],
        ["subvolume", "delete", str(rootfs)],
    ]
    sequence = recorded_sequence(store.parent)
    assert sequence.index(["podman", "rm", "-f", sid]) < sequence.index(["btrfs", "subvolume", "delete", str(rootfs)]), \
        "the container first: a mounted subvolume cannot be deleted"
    assert not (store / "sandboxes" / sid).exists() and not config_path(store / "sandboxes" / sid).exists()
    assert not (store / "sockets" / sid).exists()
    assert p.list() == [] and sb.sandbox_id is None


def test_destroy_tolerates_a_container_that_is_already_gone(store, runtime_bin, tools, ready, monkeypatch):
    p = pool(store, runtime_bin)
    sb = spawned(p)
    sid = sb.sandbox_id
    monkeypatch.setenv("FAKE_PODMAN_GONE", "1")
    asyncio.run(p.destroy(sb))
    assert p.list() == [] and not (store / "sandboxes" / sid).exists(), "the rootfs is still reclaimed"


def test_upload_and_download_are_podman_cp(store, runtime_bin, tools, ready):
    p = pool(store, runtime_bin)
    sb = spawned(p)
    asyncio.run(p.upload_file(sb, store.parent / "fixture.bin", "/testbed/fixture.bin"))
    asyncio.run(p.download_file(sb, "/testbed/model.patch", store.parent / "model.patch"))
    cps = [c for c in podman_calls(store.parent) if c[0] == "cp"]
    assert cps == [
        ["cp", str(store.parent / "fixture.bin"), "%s:/testbed/fixture.bin" % sb.sandbox_id],
        ["cp", "%s:/testbed/model.patch" % sb.sandbox_id, str(store.parent / "model.patch")],
    ]


def test_a_failing_command_names_the_tool_and_reports_its_stderr(store, runtime_bin, tools, monkeypatch):
    p = pool(store, runtime_bin)
    with pytest.raises(PodmanError, match=r"podman frobnicate failed \(exit 2\): fake podman: unknown command"):
        asyncio.run(p._containers.podman("frobnicate"))
    monkeypatch.setenv("FAKE_BTRFS_FAIL", "create")
    with pytest.raises(PodmanError, match=r"btrfs subvolume create failed \(exit 1\): ERROR: simulated"):
        asyncio.run(p.store.create(store / "x"))


# --------------------------------------------------------------------------- #
#  HttpBackend
# --------------------------------------------------------------------------- #

class FakeRuntime(BaseHTTPRequestHandler):
    """Answers JSON-RPC the way sprout-runtime does: tools/list is a bare array."""

    requests: list = []

    def do_POST(self):
        length = int(self.headers.get("Content-Length") or 0)
        body = json.loads(self.rfile.read(length) or b"{}")
        FakeRuntime.requests.append(body)
        if body["method"] == "tools/list":
            payload = {"jsonrpc": "2.0", "id": 1, "result": [{"name": "shell"}]}
        elif body["params"]["name"] == "shell":
            payload = {"jsonrpc": "2.0", "id": 1, "result": {
                "content": [{"type": "text", "text": "hi\n"}], "isError": False}}
        else:
            payload = {"jsonrpc": "2.0", "id": 1,
                       "error": {"code": -32601, "message": "unknown tool"}}
        raw = json.dumps(payload).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def log_message(self, *args):
        pass


class UnixHTTPServer(HTTPServer):
    """The same server on a Unix socket, as the runtime listens in a container."""

    address_family = socket.AF_UNIX

    def server_bind(self):
        self.socket.bind(self.server_address)
        self.server_name, self.server_port = "unix", 0


@pytest.fixture(params=["tcp", "unix"])
def runtime(request, tmp_path):
    """A backend on the fake runtime: over TCP, or over a Unix socket -- the
    podman pool's transport, where the URL only supplies the host header.
    Yields the backend and the socket path it was given, None over TCP."""
    FakeRuntime.requests = []
    if request.param == "tcp":
        server = HTTPServer(("127.0.0.1", 0), FakeRuntime)
        uds = None
        backend = HttpBackend("http://127.0.0.1:%d/" % server.server_port)
    else:
        uds = str(tmp_path / "runtime.sock")
        server = UnixHTTPServer(uds, FakeRuntime)
        backend = HttpBackend("http://sprout-runtime", uds=uds)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield backend, uds
    server.shutdown()


def test_http_backend_speaks_the_runtimes_json_rpc(runtime):
    backend, uds = runtime
    assert backend.uds == uds

    async def scenario():
        tools = await backend.list_tools()
        result = await backend.call("shell", {"command": "echo hi"}, agent_id="agent")
        refused = await backend.call("nope", {})
        await backend.close()
        return tools, result, refused

    tools, result, refused = asyncio.run(scenario())
    assert [t["name"] for t in tools] == ["shell"]
    assert result.output == "hi\n" and not result.is_error
    assert refused.is_error and refused.output == "unknown tool"
    listed, called, _ = FakeRuntime.requests
    assert listed == {"jsonrpc": "2.0", "id": 1, "method": "tools/list"}
    assert called["params"] == {"name": "shell", "arguments": {"command": "echo hi"},
                                "agent_id": "agent"}


def test_http_backend_reads_a_wrapped_tool_list():
    """An MCP-shaped server wraps the array; both are read."""
    import httpx

    backend = HttpBackend("http://runtime")

    async def scenario():
        await backend._client.aclose()
        backend._client = httpx.AsyncClient(transport=httpx.MockTransport(
            lambda request: httpx.Response(200, json={"jsonrpc": "2.0", "id": 1,
                                                      "result": {"tools": [{"name": "shell"}]}})))
        tools = await backend.list_tools()
        await backend.close()
        return tools

    assert asyncio.run(scenario()) == [{"name": "shell"}]


def test_the_default_podman_is_the_variable_else_the_installed_one_else_path(store, runtime_bin, tools, monkeypatch, tmp_path):
    """One rule for every entry point (PodmanPool, image_info, the reaper, the Run Store
    worker and its children, the live tests): $SPROUT_PODMAN_BIN names the podman (a path,
    `~` expanded, or the bare name `podman` for PATH); unset, the podman 6 that
    ops/setup_podman.sh installed under ~/.local/opt/podman when it is there; else `podman`
    on PATH as before. An explicit podman_bin wins over all of it."""
    home = tmp_path / "home"
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: home))
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.delenv("SPROUT_PODMAN_BIN", raising=False)
    assert default_podman_bin() == "podman"
    installed = home / ".local" / "opt" / "podman" / "bin" / "podman"
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\n")
    installed.chmod(0o755)
    assert default_podman_bin() == str(installed)
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "~/elsewhere/podman")
    assert default_podman_bin() == str(home / "elsewhere" / "podman")
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "podman")
    assert default_podman_bin() == "podman" and pool(store, runtime_bin).podman_bin == "podman"
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "/opt/p6/bin/podman")
    assert pool(store, runtime_bin).podman_bin == "/opt/p6/bin/podman"
    assert pool(store, runtime_bin, podman_bin="podman").podman_bin == "podman"

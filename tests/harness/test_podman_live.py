"""The podman backend driven the way a run drives it: through SandboxSession.

``tests/sdk/test_podman_live.py`` proves the pool against podman and a btrfs
store; this proves the harness plumbing on top of it: a session built from the
``podman`` config section creates a sandbox with the task's shape, its snapshot
is a branch point a second session can start from (shaped again), a memory
snapshot is declined rather than faked, the checkpointer records every capture
and never re-boards (a subvolume reports no layer count), ``harness sweep``
deletes a podman journal's snapshots through the store, and ``harness reap``
removes a leaked container with its rootfs and socket directory -- and a rootfs
whose container is already gone, found through the store the ledger names.

Skipped under the same conditions as the SDK live test (``mount_problem``
decides whether the store will do). Each test destroys the sandboxes it made
and the snapshots it took, and asserts nothing is left; run the file alone.

    pytest tests/harness/test_podman_live.py -v -m slow
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from tests.conftest import default_podman_bin, podman_store_in_use, sandbox_networks

import sprout.worker.cli as cli
from sprout.worker.checkpoint.checkpointer import Checkpointer
from sprout.worker.run.reap import PodmanClient, Reaper
from sprout.worker.run.resources import ResourceLedger
from sprout.worker.sandbox.commands import checked, shell
from sprout.worker.sandbox.session import SandboxSession
from sprout_sandbox import PodmanPool
from sprout_sandbox.btrfs_store import mount_problem

REPO = Path(__file__).resolve().parents[2]
RUNTIME_BIN = REPO / "runtime" / "sprout-runtime"
STORE = Path(os.environ.get("SPROUT_PODMAN_STORE", "~/sprout-btrfs")).expanduser()
#: The podman the pool runs: $SPROUT_PODMAN_BIN (the one ops/setup_podman.sh installed), else PATH.
PODMAN = default_podman_bin()
IMAGE = "docker.io/library/python:3.11-slim"

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
                       reason="needs a btrfs store with user_subvol_rm_allowed at SPROUT_PODMAN_STORE"),
]


def _backend() -> dict:
    return {"backend": "podman", "podman": {"store": str(STORE), "runtime_bin": str(RUNTIME_BIN)}}


def _containers() -> set[str]:
    return set(subprocess.run([PODMAN, "ps", "-a", "--no-trunc", "-q"],
                              capture_output=True, text=True, check=True).stdout.split())


def _entries(directory: Path) -> set[str]:
    return {p.name for p in directory.iterdir()} if directory.is_dir() else set()


def _shape(container: str) -> str:
    return subprocess.run([PODMAN, "inspect", "--type", "container", "--format",
                           "{{.HostConfig.Memory}} {{.HostConfig.NanoCpus}}", container],
                          capture_output=True, text=True, check=True).stdout.strip()


@pytest.fixture
def cleanup():
    """Destroy every session a test made and the snapshots it took; assert nothing leaked."""
    before = {kind: _entries(STORE / kind) for kind in ("sandboxes", "snapshots", "sockets")}
    containers_before, networks_before = _containers(), sandbox_networks()
    sessions: list[SandboxSession] = []
    yield sessions
    for session in sessions:
        session.destroy()
    leaked = _containers() - containers_before
    if leaked:
        subprocess.run([PODMAN, "rm", "-f", *leaked], capture_output=True)
    leaked_networks = sandbox_networks() - networks_before
    if leaked_networks:
        subprocess.run([PODMAN, "network", "rm", *leaked_networks], capture_output=True)
    left = {kind: _entries(STORE / kind) - before[kind] for kind in before}
    pool = PodmanPool(STORE, RUNTIME_BIN)
    for name in sorted(left["snapshots"]):
        if not name.endswith(".json"):
            asyncio.run(pool.delete_snapshot(name))
    assert not leaked, "sessions left containers behind: %s" % sorted(leaked)
    assert not leaked_networks, "sessions left networks behind: %s" % sorted(leaked_networks)
    assert not left["sandboxes"] and not left["sockets"], left


SHAPE = {"cpu": 1, "memory_mb": 300}


def test_a_session_snapshots_and_a_second_session_branches_from_it(cleanup):
    parent = SandboxSession(quiet=True, backend=_backend())
    cleanup.append(parent)
    assert parent.create(IMAGE, SHAPE), parent.create_error
    first = parent.sandbox_id
    assert first.startswith("sprout-")
    assert _shape(first) == "314572800 1000000000", "the task's shape reached the container"
    assert checked(parent, "echo hi > /tmp/a && cat /tmp/a").strip() == "hi"
    assert "PYTHON_VERSION=" in checked(parent, "env"), "the image's ENV reached the runtime"

    snap = parent.snapshot()
    assert snap is not None and snap.disk_only and snap.rootfs_layers is None, snap
    assert snap.id.startswith(first + "-0001")
    assert parent.sandbox_id == first, "a snapshot does not replace the sandbox"
    assert parent.snapshot(disk_only=False) is None, "a memory snapshot is declined, not faked"

    checkpointer = Checkpointer(session=parent)
    records = [checkpointer.after_step(turn) for turn in (1, 2)]
    assert all(r.captured and not r.reboarded and r.disk_only for r in records), records
    assert parent.sandbox_id == first, "nothing re-boards: a subvolume has no layer count"

    branch = SandboxSession(quiet=True, backend=_backend())
    cleanup.append(branch)
    assert branch.create(snap.id, SHAPE), branch.create_error
    assert branch.sandbox_id != first
    assert _shape(branch.sandbox_id) == "314572800 1000000000", "a branch is shaped again: a subvolume carries none"
    assert checked(branch, "cat /tmp/a").strip() == "hi"
    checked(branch, "echo branch > /tmp/b")
    _, _, code = shell(parent, "test -e /tmp/b")
    assert code != 0, "the branch's write reached the parent: no isolation"


def test_sweep_deletes_a_podman_journals_snapshots_through_the_store(cleanup, tmp_path, capsys):
    """The journal names the backend and the store; sweep needs nothing else --
    not the pool, not the runtime binary, not AgentENV."""
    session = SandboxSession(quiet=True, backend=_backend())
    cleanup.append(session)
    assert session.create(IMAGE), session.create_error
    snapshots = [session.snapshot(), session.snapshot(name="final")]
    assert all(snapshots)
    journal = tmp_path / "run.jsonl"
    lines = [{"v": 2, "type": "run.started", "seq": 1, "run_id": "r"},
             {"v": 2, "type": "sandbox.created", "seq": 2, "run_id": "r", "sandbox_id": session.sandbox_id,
              "backend": "podman", "image": IMAGE, "store": str(STORE)}]
    lines += [{"v": 2, "type": "checkpoint.captured", "seq": 3 + n, "run_id": "r", "snapshot_id": snap.id}
              for n, snap in enumerate(snapshots)]
    journal.write_text("\n".join(json.dumps(line) for line in lines) + "\n")

    assert cli.main(["sweep", str(journal)]) == 0
    assert "would delete 2 podman snapshot(s)" in capsys.readouterr().out
    assert all((STORE / "snapshots" / snap.id).is_dir() for snap in snapshots)
    assert cli.main(["sweep", str(journal), "--yes"]) == 0
    assert "deleted 2/2" in capsys.readouterr().out
    for snap in snapshots:
        assert not (STORE / "snapshots" / snap.id).exists()
        assert not (STORE / "snapshots" / (snap.id + ".json")).exists()
    marker = json.loads(journal.read_text().splitlines()[-1])
    assert (marker["type"], marker["backend"], marker["deleted"]) == ("snapshots.swept", "podman", 2)
    assert checked(session, "echo alive").strip() == "alive", "the sandbox the snapshots came from is untouched"


def test_reap_removes_a_leaked_container_its_rootfs_and_socket_dir(tmp_path):
    """A killed run leaves its container running on a subvolume nothing else
    references; the ledger knows the name and the dead pid, the container's own
    labels and mounts say where its rootfs and socket directory are."""
    before = {kind: _entries(STORE / kind) for kind in ("sandboxes", "sockets")}
    pool = PodmanPool(STORE, RUNTIME_BIN)
    leaked = asyncio.run(pool.spawn(IMAGE))
    live = asyncio.run(pool.spawn(IMAGE))
    try:
        ledger = ResourceLedger(tmp_path / "res.jsonl")
        ledger._append("claim", run_id="dead", kind="sandbox", id=leaked.sandbox_id,
                       pid=999999999, backend="podman")
        ledger._append("claim", run_id="mine", kind="sandbox", id=live.sandbox_id,
                       pid=os.getpid(), backend="podman")
        reaper = Reaper({"podman": PodmanClient()}, ledger)
        plan = reaper.plan()
        assert plan.sandboxes == [leaked.sandbox_id], "the live run's container is in use"
        assert plan.backends == {leaked.sandbox_id: "podman"}
        assert reaper.apply(plan) == {"sandboxes": [leaked.sandbox_id], "failed": []}
        assert leaked.sandbox_id not in subprocess.run(
            [PODMAN, "ps", "-a", "--format", "{{.Names}}"], capture_output=True, text=True).stdout
        assert not (STORE / "sandboxes" / leaked.sandbox_id).exists()
        assert not (STORE / "sandboxes" / (leaked.sandbox_id + ".json")).exists()
        assert not (STORE / "sockets" / leaked.sandbox_id).exists()
        assert (STORE / "sandboxes" / live.sandbox_id).is_dir()
        assert reaper.plan().total() == 0, "released in the ledger: a second pass finds nothing"
    finally:
        asyncio.run(pool.close())
    assert {kind: _entries(STORE / kind) - before[kind] for kind in before} == {"sandboxes": set(), "sockets": set()}
    assert not [name for name in subprocess.run([PODMAN, "ps", "-a", "--format", "{{.Names}}"],
                                                capture_output=True, text=True).stdout.split()
                if name in (leaked.sandbox_id, live.sandbox_id)]


def test_reap_reclaims_a_sandbox_whose_container_is_already_gone(tmp_path):
    """Two leftovers with no container to find them by. A container removed by
    hand leaves its network, whose `sprout.store` label leads to the rootfs, so
    a reaper given no store reclaims all of it. A destroy that failed at the
    rootfs leaves nothing but the rootfs: the run's ledger claim names the
    store (as `harness run` records it), so the client lists it there, the
    dead pid makes it an orphan, and it goes with its config and socket
    directory."""
    before = {kind: _entries(STORE / kind) for kind in ("sandboxes", "sockets")}
    networks_before = sandbox_networks()
    pool = PodmanPool(STORE, RUNTIME_BIN)
    by_hand, stranded = (asyncio.run(pool.spawn(IMAGE)) for _ in range(2))
    ids = [by_hand.sandbox_id, stranded.sandbox_id]
    try:
        for sid in ids:
            subprocess.run([PODMAN, "rm", "-f", sid], check=True, capture_output=True)
        subprocess.run([PODMAN, "network", "rm", stranded.sandbox_id], check=True, capture_output=True)
        asyncio.run(pool.forget(by_hand, stranded))
        assert all((STORE / "sandboxes" / sid).is_dir() and (STORE / "sockets" / sid).is_dir() for sid in ids)
        ledger = ResourceLedger(tmp_path / "res.jsonl")
        for sid in ids:
            ledger._append("claim", run_id="dead", kind="sandbox", id=sid, pid=999999999, backend="podman",
                           store=str(STORE))
        listed = {s["id"] for s in PodmanClient().list_sandboxes()}
        assert ids[0] in listed and ids[1] not in listed, \
            "without a store to look in, only the sandbox whose network is left is visible"
        reaper = Reaper({"podman": PodmanClient()}, ledger)
        assert reaper.apply(reaper.plan()) == {"sandboxes": [ids[0]], "failed": []}
        assert ids[0] not in sandbox_networks() and not (STORE / "sandboxes" / ids[0]).exists()
        assert not (STORE / "sockets" / ids[0]).exists(), "found through the network's store label"
        client = PodmanClient(stores=[resource.store for resource in ledger.resources() if resource.store])
        assert ids[1] in {s["id"] for s in client.list_sandboxes()}
        reaper = Reaper({"podman": client}, ledger)
        plan = reaper.plan()
        assert plan.sandboxes == [ids[1]] and plan.backends == {ids[1]: "podman"}
        assert reaper.apply(plan) == {"sandboxes": [ids[1]], "failed": []}
        assert not (STORE / "sandboxes" / ids[1]).exists() and not (STORE / "sandboxes" / (ids[1] + ".json")).exists()
        assert not (STORE / "sockets" / ids[1]).exists()
        assert reaper.plan().total() == 0
    finally:
        asyncio.run(pool.close())
        for sid in ids:
            subprocess.run([PODMAN, "rm", "-f", sid], capture_output=True)
            subprocess.run([PODMAN, "network", "rm", sid], capture_output=True)
    assert {kind: _entries(STORE / kind) - before[kind] for kind in before} == {"sandboxes": set(), "sockets": set()}
    assert sandbox_networks() - networks_before == set()

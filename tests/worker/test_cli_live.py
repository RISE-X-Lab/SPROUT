"""``sweep`` and ``reap`` on the real podman and btrfs store, the fidelity tests of their fakes.

``test_cli_journals.py`` and ``test_cli_reap.py`` run the same commands against the fake podman and
btrfs. These tests run them against real sandboxes. They need rootless podman, a btrfs store at
``$SPROUT_PODMAN_STORE`` that no run uses, and the runtime binary. Each test skips with what is missing.
Run the file alone, while the store is idle::

    pytest tests/worker/test_cli_live.py -m slow
"""

from __future__ import annotations

import asyncio
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest

from sprout.worker.cli import main
from sprout.worker.run.events import CHECKPOINT_CAPTURED, SANDBOX_CREATED
from sprout.worker.run.journal import JournalWriter
from sprout.worker.run.resources import ResourceLedger
from sprout.worker.sandbox.commands import checked
from sprout.worker.sandbox.session import SandboxSession
from sprout_sandbox import PodmanPool, Sandbox
from sprout_sandbox.podman_cli import default_podman_bin
from tests.support.live import sandbox_networks
from tests.support.paths import RUNTIME_BIN

pytestmark = pytest.mark.slow

IMAGE = "docker.io/library/python:3.11-slim"


def podman(*args: str) -> str:
    return subprocess.run([default_podman_bin(), *args], capture_output=True, text=True, check=True).stdout


def containers() -> set[str]:
    return set(podman("ps", "-a", "--format", "{{.Names}}").split())


def entries(directory: Path) -> set[str]:
    return {path.name for path in directory.iterdir()} if directory.is_dir() else set()


def leftovers(store: Path, sandbox_id: str) -> dict[str, bool]:
    """Whether each part of ``sandbox_id`` exists: its container, network, rootfs and socket directory."""
    return {
        "container": sandbox_id in containers(),
        "network": sandbox_id in sandbox_networks(),
        "rootfs": (store / "sandboxes" / sandbox_id).exists(),
        "sockets": (store / "sockets" / sandbox_id).exists(),
    }


NOTHING_LEFT = {"container": False, "network": False, "rootfs": False, "sockets": False}


class Live:
    """The store of one test and the pool that makes its sandboxes."""

    def __init__(self, store: Path) -> None:
        self.store = store
        self.pool = PodmanPool(store, RUNTIME_BIN)
        self.sessions: list[SandboxSession] = []

    def sandbox(self) -> Sandbox:
        """A running sandbox of the pool, as a run that the test then kills would leave it."""
        return asyncio.run(self.pool.spawn(IMAGE))

    def session(self) -> SandboxSession:
        """A running session on the store, as a run makes one."""
        session = SandboxSession(
            quiet=True, backend={"podman": {"store": str(self.store), "runtime_bin": str(RUNTIME_BIN)}}
        )
        self.sessions.append(session)
        assert session.create(IMAGE), session.create_error
        return session


@pytest.fixture
def live(live_podman_store: Path) -> Iterator[Live]:
    """Free each sandbox and snapshot of the test, then fail if a container, network or rootfs is left."""
    if not RUNTIME_BIN.is_file():
        pytest.skip(
            "needs the runtime binary: cd runtime && go generate ./... && go build -o sprout-runtime ."
        )
    kinds = ("sandboxes", "snapshots", "sockets")
    before = {kind: entries(live_podman_store / kind) for kind in kinds}
    containers_before, networks_before = containers(), sandbox_networks()
    made = Live(live_podman_store)
    yield made
    for session in made.sessions:
        session.destroy()
    asyncio.run(made.pool.close())
    for name in sorted(entries(live_podman_store / "snapshots") - before["snapshots"]):
        if not name.endswith(".json"):
            asyncio.run(made.pool.delete_snapshot(name))
    left = {kind: entries(live_podman_store / kind) - before[kind] for kind in kinds}
    leaked = (containers() - containers_before, sandbox_networks() - networks_before)
    assert (left, leaked) == ({kind: set() for kind in kinds}, (set(), set())), "the test left these behind"


def test_a_sweep_deletes_the_snapshots_of_a_podman_journal_from_its_store_and_leaves_the_sandbox(
    live: Live, tmp_path, capsys
):
    """The journal names the backend and the store, and the sweep needs nothing else."""
    session = live.session()
    snapshots = [session.snapshot(), session.snapshot(name="final")]
    path = tmp_path / "run.jsonl"
    with JournalWriter(path, run_id="run-1") as journal:
        journal.emit(
            SANDBOX_CREATED,
            sandbox_id=session.sandbox_id,
            backend="podman",
            image=IMAGE,
            store=str(live.store),
        )
        journal.emit(CHECKPOINT_CAPTURED, step=1, snapshot_id=snapshots[0].id)
        journal.emit(CHECKPOINT_CAPTURED, step=2, snapshot_id=snapshots[1].id)

    code = main(["sweep", str(path), "--yes"])

    assert (code, capsys.readouterr().out) == (0, "%s: deleted 2/2 snapshot(s)\n" % path)
    assert entries(live.store / "snapshots") & {snapshots[0].id, snapshots[1].id} == set()
    assert checked(session, "echo alive") == "alive\n"


def test_a_reap_frees_the_sandbox_of_a_finished_run_and_leaves_the_sandbox_of_a_running_one(
    live: Live, tmp_path, capsys
):
    """A killed run leaves its container running. Its claim in the ledger is all that names it."""
    leaked, running = live.sandbox(), live.sandbox()
    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    with ledger.run("run-1") as claim:
        claim.sandbox(leaked.sandbox_id, backend="podman", store=str(live.store))

    with ledger.run("run-2") as claim:
        claim.sandbox(running.sandbox_id, backend="podman", store=str(live.store))
        code = main(["reap", "--ledger", str(ledger.path)])

    assert (code, capsys.readouterr().out.splitlines()[-1]) == (0, "freed 1 sandbox(es)")
    assert leftovers(live.store, leaked.sandbox_id) == NOTHING_LEFT
    assert leftovers(live.store, running.sandbox_id) == {
        "container": True,
        "network": True,
        "rootfs": True,
        "sockets": True,
    }


def test_a_reap_frees_the_rootfs_of_a_container_removed_by_hand_through_the_label_of_its_network(
    live: Live, tmp_path
):
    """The claim names no store. The network that ``podman rm`` leaves carries the store in a label."""
    sandbox = live.sandbox()
    sandbox_id = sandbox.sandbox_id
    podman("rm", "-f", sandbox_id)
    asyncio.run(live.pool.forget(sandbox))
    with ResourceLedger(tmp_path / "resources.jsonl").run("run-1") as claim:
        claim.sandbox(sandbox_id, backend="podman")

    code = main(["reap", "--ledger", str(tmp_path / "resources.jsonl")])

    assert (code, leftovers(live.store, sandbox_id)) == (0, NOTHING_LEFT)


def test_a_reap_frees_a_rootfs_that_only_the_store_of_its_claim_still_holds(live: Live, tmp_path):
    """A destroy that failed at the rootfs leaves no container and no network to find it by."""
    sandbox = live.sandbox()
    sandbox_id = sandbox.sandbox_id
    podman("rm", "-f", sandbox_id)
    podman("network", "rm", sandbox_id)
    asyncio.run(live.pool.forget(sandbox))
    with ResourceLedger(tmp_path / "resources.jsonl").run("run-1") as claim:
        claim.sandbox(sandbox_id, backend="podman", store=str(live.store))

    code = main(["reap", "--ledger", str(tmp_path / "resources.jsonl")])

    assert (code, leftovers(live.store, sandbox_id)) == (0, NOTHING_LEFT)

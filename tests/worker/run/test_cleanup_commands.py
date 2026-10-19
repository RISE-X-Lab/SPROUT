"""``python -m sprout.worker sweep`` and ``reap``: the commands that delete what runs leave behind.

A sweep deletes the snapshots that a named journal records, the only way that a snapshot dies.
A reap frees the sandboxes of dead runs, and never a snapshot. Both run against the fake podman,
btrfs and findmnt, and a stub of the AgentENV API.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from sprout.worker import cli
from sprout.worker.run.journal import JournalWriter, journal_events
from tests.support.tools import FakeBtrfs, FakeFindmnt, FakePodman, ManagedContainer
from tests.worker.run.agentenv import AgentEnvStub, agentenv_stub

SANDBOX = "sprout-000000000009"
SNAPSHOTS = ["sprout-000000000009-0001", "sprout-000000000009-0002", "sprout-000000000009-0003-final"]
DEAD_PID = 999_999_999


def podman_journal(path: Path, *, store: Path | None, snapshot_ids: list[str]) -> Path:
    """The journal of a podman run in ``store``, or in no store it names, that recorded ``snapshot_ids``."""
    with JournalWriter(path, run_id="run-1") as journal:
        located = {"store": str(store)} if store else {}
        journal.emit("sandbox.created", sandbox_id=SANDBOX, backend="podman", image="image", **located)
        for step, snapshot_id in enumerate(snapshot_ids, 1):
            journal.emit("checkpoint.captured", step=step, snapshot_id=snapshot_id, reason="captured")
    return path


def microvm_journal(path: Path, *, snapshot_ids: list[str]) -> Path:
    """The journal of a microVM run that recorded ``snapshot_ids``, from before runs named their backend."""
    with JournalWriter(path, run_id="run-1") as journal:
        for step, snapshot_id in enumerate(snapshot_ids, 1):
            journal.emit("checkpoint.captured", step=step, snapshot_id=snapshot_id, reason="captured")
    return path


def snapshot_subvolumes(store: Path, *snapshot_ids: str) -> None:
    for snapshot_id in snapshot_ids:
        (store / "snapshots" / snapshot_id).mkdir(parents=True)
        (store / "snapshots" / (snapshot_id + ".json")).write_text("{}")


def snapshots_left(store: Path) -> list[str]:
    """The names in the store's ``snapshots`` directory: subvolumes and their configs."""
    return sorted(path.name for path in (store / "snapshots").iterdir())


def last_record(journal: Path) -> dict:
    return journal_events(journal)[-1]


@pytest.fixture
def agentenv(monkeypatch: pytest.MonkeyPatch) -> Iterator[AgentEnvStub]:
    """The AgentENV that the commands reach through ``AENV_SERVER_URL``."""
    with agentenv_stub() as stub:
        monkeypatch.setenv("AENV_SERVER_URL", stub.url)
        monkeypatch.delenv("AENV_API_KEY", raising=False)
        yield stub


def test_a_sweep_without_yes_names_the_snapshots_and_deletes_nothing(tmp_path, capsys, fake_btrfs_store):
    journal = podman_journal(tmp_path / "run.jsonl", store=fake_btrfs_store, snapshot_ids=SNAPSHOTS)
    snapshot_subvolumes(fake_btrfs_store, *SNAPSHOTS)
    before = journal.read_bytes()

    status = cli.main(["sweep", str(journal)])

    assert (status, capsys.readouterr().out) == (
        0,
        "%s: would delete 3 podman snapshot(s)  (dry run; pass --yes)\n" % journal,
    )
    assert (len(snapshots_left(fake_btrfs_store)), journal.read_bytes()) == (6, before)


def test_a_sweep_deletes_the_snapshots_of_a_podman_journal_from_its_store_and_marks_the_journal(
    tmp_path, capsys, fake_btrfs: FakeBtrfs, fake_btrfs_store
):
    """A snapshot that is already gone counts as deleted."""
    journal = podman_journal(tmp_path / "run.jsonl", store=fake_btrfs_store, snapshot_ids=SNAPSHOTS)
    snapshot_subvolumes(fake_btrfs_store, *SNAPSHOTS[:2])

    status = cli.main(["sweep", str(journal), "--yes"])

    assert (status, capsys.readouterr().out) == (0, "%s: deleted 3/3 snapshot(s)\n" % journal)
    assert snapshots_left(fake_btrfs_store) == []
    assert last_record(journal) == {
        "v": 2, "type": "snapshots.swept", "seq": 5, "run_id": "sweep", "agent_id": "sweep",
        "backend": "podman", "deleted": 3, "failed": 0,
    }  # fmt: skip


def test_a_swept_journal_is_skipped_by_the_next_sweep(tmp_path, capsys, agentenv: AgentEnvStub):
    journal = microvm_journal(tmp_path / "run.jsonl", snapshot_ids=["snapshot-1"])
    cli.main(["sweep", str(journal), "--yes"])
    capsys.readouterr()

    status = cli.main(["sweep", str(journal), "--yes"])

    assert (status, capsys.readouterr().out) == (0, "%s: already swept, skipping\n" % journal)
    assert agentenv.requests == [("DELETE", "/snapshots/snapshot-1", None)]


def test_a_sweep_of_a_killed_runs_torn_journal_keeps_the_journal_a_file_of_whole_records(
    tmp_path, fake_btrfs_store
):
    """Glued to the torn tail, the marker would make one corrupt line. The marker carries the tail instead."""
    journal = podman_journal(tmp_path / "run.jsonl", store=fake_btrfs_store, snapshot_ids=SNAPSHOTS[:2])
    snapshot_subvolumes(fake_btrfs_store, *SNAPSHOTS[:2])
    with journal.open("a") as stream:
        stream.write('{"v": 2, "type": "tool.finished", "output": "cut off')

    cli.main(["sweep", str(journal), "--yes"])

    lines = journal.read_bytes().splitlines(keepends=True)
    assert [json.loads(line)["type"] for line in lines] == [
        "sandbox.created",
        "checkpoint.captured",
        "checkpoint.captured",
        "snapshots.swept",
    ]
    assert (
        lines[-1].endswith(b"\n"),
        json.loads(lines[-1])["torn_tail"],
        snapshots_left(fake_btrfs_store),
    ) == (
        True,
        '{"v": 2, "type": "tool.finished", "output": "cut off',
        [],
    )


def test_a_podman_journal_that_names_no_store_is_not_swept_without_the_store_option(
    tmp_path, capsys, fake_btrfs_store
):
    journal = podman_journal(tmp_path / "run.jsonl", store=None, snapshot_ids=SNAPSHOTS[:1])
    snapshot_subvolumes(fake_btrfs_store, SNAPSHOTS[0])

    status = cli.main(["sweep", str(journal), "--yes"])

    assert (status, capsys.readouterr().err) == (
        1,
        "%s: a podman run whose journal does not name its store; pass --store <dir> "
        "(the store its subvolumes live in)\n" % journal,
    )
    assert (snapshots_left(fake_btrfs_store), last_record(journal)["type"]) == (
        [SNAPSHOTS[0], SNAPSHOTS[0] + ".json"],
        "checkpoint.captured",
    )


def test_the_store_option_names_the_store_of_a_podman_journal(tmp_path, fake_btrfs_store):
    journal = podman_journal(tmp_path / "run.jsonl", store=None, snapshot_ids=SNAPSHOTS[:1])
    snapshot_subvolumes(fake_btrfs_store, SNAPSHOTS[0])

    status = cli.main(["sweep", str(journal), "--yes", "--store", str(fake_btrfs_store)])

    assert (status, snapshots_left(fake_btrfs_store), last_record(journal)["type"]) == (
        0,
        [],
        "snapshots.swept",
    )


def test_a_store_that_is_not_a_btrfs_store_is_refused_and_nothing_is_swept(
    tmp_path, capsys, fake_findmnt: FakeFindmnt, fake_btrfs_store
):
    journal = podman_journal(tmp_path / "run.jsonl", store=fake_btrfs_store, snapshot_ids=SNAPSHOTS[:1])
    snapshot_subvolumes(fake_btrfs_store, SNAPSHOTS[0])
    fake_findmnt.report(fstype="ext4", options="rw,relatime")

    status = cli.main(["sweep", str(journal), "--yes"])

    assert (status, capsys.readouterr().err) == (
        1,
        "%s: a podman run: store %s is on ext4 mounted with rw,relatime; the podman backend needs btrfs "
        "mounted with user_subvol_rm_allowed (sudo ops/btrfs_store.sh <mount point> <image file> <size>)\n"
        % (journal, fake_btrfs_store.resolve()),
    )
    assert last_record(journal)["type"] == "checkpoint.captured"


def test_a_journal_that_names_no_backend_is_swept_through_agentenv_once_for_each_snapshot(
    tmp_path, capsys, agentenv: AgentEnvStub
):
    """A clean step records the snapshot of an earlier step again."""
    journal = microvm_journal(tmp_path / "run.jsonl", snapshot_ids=["snapshot-1", "snapshot-1", "snapshot-2"])

    status = cli.main(["sweep", str(journal), "--yes"])

    assert (status, capsys.readouterr().out) == (0, "%s: deleted 2/2 snapshot(s)\n" % journal)
    assert agentenv.requests == [
        ("DELETE", "/snapshots/snapshot-1", None),
        ("DELETE", "/snapshots/snapshot-2", None),
    ]
    assert (last_record(journal)["backend"], last_record(journal)["deleted"]) == ("microvm", 2)


def test_a_snapshot_that_fails_to_delete_is_counted_and_fails_the_sweep(
    tmp_path, capsys, agentenv: AgentEnvStub
):
    journal = microvm_journal(tmp_path / "run.jsonl", snapshot_ids=["snapshot-1", "snapshot-2"])
    agentenv.delete_statuses["/snapshots/snapshot-2"] = 500

    status = cli.main(["sweep", str(journal), "--yes"])

    assert (status, capsys.readouterr().out) == (1, "%s: deleted 1/2 snapshot(s), 1 failed\n" % journal)
    assert (last_record(journal)["deleted"], last_record(journal)["failed"]) == (1, 1)


def test_a_sweep_of_a_missing_journal_fails(tmp_path, capsys):
    missing = tmp_path / "missing.jsonl"

    status = cli.main(["sweep", str(missing)])

    assert (status, capsys.readouterr().err) == (2, "no such journal: %s\n" % missing)


def write_claims(path: Path, *claims: dict) -> Path:
    """A ledger at ``path`` with ``claims`` of run ``run-1``, whose process is gone."""
    with path.open("w") as stream:
        for fields in claims:
            entry = {"event": "claim", "run_id": "run-1", "kind": "sandbox", "pid": DEAD_PID, **fields}
            stream.write(json.dumps(entry) + "\n")
    return path


def leaked_rootfs(store: Path, name: str) -> Path:
    """The rootfs and the socket directory that a sandbox ``name`` without a container leaves in ``store``."""
    (store / "sandboxes" / name).mkdir(parents=True)
    (store / "sockets" / name).mkdir(parents=True)
    return store / "sandboxes" / name


def test_a_dry_run_reap_lists_the_orphans_of_both_backends_and_frees_nothing(
    tmp_path, capsys, agentenv: AgentEnvStub, fake_podman: FakePodman, fake_btrfs_store
):
    ledger = write_claims(tmp_path / "resources.jsonl", {"id": "vm-1"}, {"id": SANDBOX, "backend": "podman"})
    agentenv.sandboxes = [{"sandboxID": "vm-1", "startedAt": None}]
    sockets = fake_btrfs_store / "sockets" / SANDBOX
    fake_podman.containers_exist(
        ManagedContainer(SANDBOX, store=fake_btrfs_store, sockets=sockets, created=1)
    )

    status = cli.main(["reap", "--ledger", str(ledger), "--dry-run"])

    assert (status, capsys.readouterr().out) == (
        0,
        "  sandbox   microvm  vm-1  orphan of run run-1 (pid 999999999 gone)\n"
        "  sandbox   podman   %s  orphan of run run-1 (pid 999999999 gone)\n"
        "\n2 sandbox(es) would be freed (dry run)\n" % SANDBOX,
    )
    assert ([request[0] for request in agentenv.requests], ["rm", "-f", SANDBOX] in fake_podman.calls()) == (
        ["GET"],
        False,
    )


def test_a_reap_frees_the_orphans_through_the_backend_of_each(
    tmp_path, capsys, agentenv: AgentEnvStub, fake_podman: FakePodman, fake_btrfs_store
):
    ledger = write_claims(tmp_path / "resources.jsonl", {"id": "vm-1"}, {"id": SANDBOX, "backend": "podman"})
    agentenv.sandboxes = [{"sandboxID": "vm-1", "startedAt": None}]
    sockets = fake_btrfs_store / "sockets" / SANDBOX
    fake_podman.containers_exist(
        ManagedContainer(SANDBOX, store=fake_btrfs_store, sockets=sockets, created=1)
    )

    status = cli.main(["reap", "--ledger", str(ledger)])

    assert (status, capsys.readouterr().out.splitlines()[-1]) == (0, "freed 2 sandbox(es)")
    assert (agentenv.requests[-1], ["rm", "-f", SANDBOX] in fake_podman.calls()) == (
        ("DELETE", "/sandboxes/vm-1", None),
        True,
    )


def test_a_reap_looks_for_a_rootfs_in_the_stores_of_the_claims_and_of_the_store_option(
    tmp_path, agentenv: AgentEnvStub, fake_podman: FakePodman, fake_btrfs_store
):
    """A destroy that failed after ``podman rm -f`` leaves a rootfs that no container label finds."""
    other_store = tmp_path / "other-store"
    other_store.mkdir()
    claimed = leaked_rootfs(fake_btrfs_store, "sprout-00000000000a")
    named = leaked_rootfs(other_store, "sprout-00000000000b")
    ledger = write_claims(
        tmp_path / "resources.jsonl",
        {"id": "sprout-00000000000a", "backend": "podman", "store": str(fake_btrfs_store)},
        {"id": "sprout-00000000000b", "backend": "podman"},
    )

    status = cli.main(["reap", "--ledger", str(ledger), "--store", str(other_store)])

    assert (status, claimed.exists(), named.exists()) == (0, False, False)


def test_a_reap_of_unknown_sandboxes_needs_an_age_cutoff(tmp_path, capsys):
    ledger = write_claims(tmp_path / "resources.jsonl")

    status = cli.main(["reap", "--ledger", str(ledger), "--include-unknown"])

    assert (status, capsys.readouterr().err) == (2, "refusing --include-unknown without --older-than\n")


def test_a_reap_whose_deletion_fails_names_the_sandbox_and_fails(
    tmp_path, capsys, agentenv: AgentEnvStub, fake_podman: FakePodman
):
    ledger = write_claims(tmp_path / "resources.jsonl", {"id": "vm-1"})
    agentenv.sandboxes = [{"sandboxID": "vm-1", "startedAt": None}]
    agentenv.delete_statuses["/sandboxes/vm-1"] = 500

    status = cli.main(["reap", "--ledger", str(ledger)])

    assert (status, capsys.readouterr().err) == (1, "failed: vm-1\n")


def test_a_reap_with_nothing_to_free_says_so_and_counts_the_kept_sandboxes(
    tmp_path, capsys, agentenv: AgentEnvStub, fake_podman: FakePodman
):
    ledger = write_claims(tmp_path / "resources.jsonl", {"id": "vm-kept", "keep": True})

    status = cli.main(["reap", "--ledger", str(ledger)])

    assert (status, capsys.readouterr().out) == (0, "nothing to reap (1 kept)\n")

"""Resource ledger + reaper + sweep.

The reaper holds one client per backend: AgentENV for microVMs, podman for the
container backend. The podman client is exercised against the fake ``podman``
and ``btrfs`` from ``tests/conftest.py``, so what is verified is the exact
command line and the teardown order across the two tools (their shared log);
``tests/harness/test_podman_live.py`` runs the same client against the real
tools.
"""

from __future__ import annotations

import inspect
import json
import os
from datetime import timedelta

import pytest

from sprout.harness.orchestrator.reap import PodmanClient, ReapPlan, Reaper, parse_duration
from sprout.harness.orchestrator.resources import ResourceLedger
from tests.conftest import fake_networks, recorded_calls, recorded_sequence


def orphans(ledger):
    return [r for r in ledger.resources() if r.orphaned()]


# --- ledger ----------------------------------------------------------------
def _released_claim(ledger):
    """What a run that freed its sandbox leaves: the claim, its release, run done."""
    with ledger.run("run-1") as claim:
        claim.sandbox("sb-1")
        claim.released("sandbox", "sb-1")


@pytest.mark.parametrize("record,expected", [
    pytest.param(_released_claim, [], id="released"),
    # A pid that cannot exist: recorded by hand because the live process is us.
    pytest.param(lambda ledger: ledger._append("claim", run_id="run-2", kind="sandbox", id="sb-9", pid=999999999),
                 [("sandbox", "sb-9")], id="dead pid"),
    # Reaping a resource a running batch still needs is the worst outcome.
    pytest.param(lambda ledger: ledger._append("claim", run_id="mine", kind="sandbox", id="sb-live", pid=os.getpid()),
                 [], id="live pid"),
    # A sandbox kept because its final snapshot failed is the only copy of the
    # rollout; it must survive its creating run.
    pytest.param(lambda ledger: ledger._append("claim", run_id="r", kind="sandbox", id="pinned",
                                               pid=999999999, keep=True),
                 [], id="kept"),
])
def test_which_claims_are_orphans(tmp_path, record, expected):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    record(ledger)
    assert [(o.kind, o.id) for o in orphans(ledger)] == expected


def test_run_done_without_release_is_still_an_orphan(tmp_path):
    """A kill between "run finished" and "resource freed" must not hide a leak."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    with ledger.run("run-3") as claim:
        claim.sandbox("sb-x")
    assert [o.id for o in orphans(ledger)] == ["sb-x"]


def test_a_claim_names_its_backend_and_old_ledgers_read_as_microvm(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    with ledger.run("run-p") as claim:
        claim.sandbox("ctr", backend="podman", store="/srv/store")
        claim.sandbox("vm")
    ledger._append("claim", run_id="old", kind="sandbox", id="sb-old", pid=999999999)
    by_id = {r.id: (r.backend, r.store) for r in ledger.resources()}
    assert by_id == {"ctr": ("podman", "/srv/store"), "vm": ("microvm", None), "sb-old": ("microvm", None)}, \
        "a podman claim names the store its rootfs lives in; the reaper looks there once the container is gone"


def test_torn_line_does_not_abort_the_reap(tmp_path):
    path = tmp_path / "res.jsonl"
    ledger = ResourceLedger(path)
    ledger._append("claim", run_id="r", kind="sandbox", id="good", pid=999999999)
    with path.open("a") as fh:
        fh.write('{"event": "claim", "kind": "sandb\n')   # truncated write
    assert [o.id for o in orphans(ledger)] == ["good"]


# --- reaper ----------------------------------------------------------------
class FakeClient:
    def __init__(self, sandboxes=()):
        self._sandboxes = list(sandboxes)
        self.deleted = []
        self.fail = set()

    def list_sandboxes(self):
        return list(self._sandboxes)

    def delete_sandbox(self, sandbox_id):
        if sandbox_id in self.fail:
            return False
        self.deleted.append(sandbox_id)
        return True


def reaper(ledger, microvm=None, podman=None):
    clients = {}
    if microvm is not None:
        clients["microvm"] = microvm
    if podman is not None:
        clients["podman"] = podman
    return Reaper(clients, ledger)


def test_old_snapshot_claims_are_ignored_never_planned(tmp_path):
    """Ledgers written before snapshot claims were dropped carry `kind:
    snapshot` rows. They are neither planned nor counted: "owning run died"
    describes EVERY snapshot worth keeping, and snapshots die only by
    `harness sweep <journal> --yes`, where naming the journal is the consent."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="snapshot", id="orphan", pid=999999999)
    ledger._append("claim", run_id="dead", kind="sandbox", id="sb", pid=999999999)
    client = FakeClient(sandboxes=[
        {"id": "sb", "created_at": None, "names": []},
        {"id": "orphan", "created_at": None, "names": []},
    ])
    plan = reaper(ledger, microvm=client).plan()
    assert plan.sandboxes == ["sb"] and plan.total() == 1
    assert "orphan" not in plan.kept
    assert reaper(ledger, microvm=client).apply(plan)["sandboxes"] == ["sb"]


def test_include_unknown_needs_an_age_cutoff_and_spares_named(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    client = FakeClient(sandboxes=[
        {"id": "old", "created_at": "2020-01-01T00:00:00Z", "names": []},
        {"id": "recent", "created_at": "2999-01-01T00:00:00Z", "names": []},
        {"id": "named", "created_at": "2020-01-01T00:00:00Z", "names": ["golden-base"]},
        {"id": "undated", "created_at": None, "names": []},
    ])
    assert reaper(ledger, microvm=client).plan().total() == 0, "strangers are left alone unless asked for"
    plan = reaper(ledger, microvm=client).plan(
        include_unknown=True, older_than=timedelta(hours=24)
    )
    assert plan.sandboxes == ["old"]
    assert "named" in plan.kept
    assert plan.reasons["old"].startswith("not in ledger")


def test_include_unknown_reaps_every_backends_strangers_through_their_own_client(tmp_path):
    """A leaked podman container the ledger never saw is as much a leak as a
    microVM; each goes back through the client that listed it."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    microvm = FakeClient(sandboxes=[{"id": "vm", "created_at": "2020-01-01T00:00:00Z", "names": []}])
    podman = FakeClient(sandboxes=[{"id": "ctr", "created_at": "2020-01-01T00:00:00Z", "names": []}])
    r = reaper(ledger, microvm=microvm, podman=podman)
    plan = r.plan(include_unknown=True, older_than=timedelta(hours=24))
    assert sorted(plan.sandboxes) == ["ctr", "vm"]
    assert plan.backends == {"vm": "microvm", "ctr": "podman"}
    assert r.apply(plan)["sandboxes"] == plan.sandboxes
    assert (microvm.deleted, podman.deleted) == (["vm"], ["ctr"])


def test_an_orphan_of_a_backend_without_a_client_is_reported_not_planned(tmp_path):
    """A claim naming a backend this reaper has no client for is not silently
    "already gone" -- that rule once hid every podman container while AgentENV
    was the only client -- and it is not anyone else's to free either."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="sandbox", id="ctr", pid=999999999, backend="docker")
    client = FakeClient()
    plan = reaper(ledger, microvm=client).plan()
    assert plan.total() == 0 and plan.foreign == ["ctr"]
    assert plan.reasons["ctr"] == ("docker sandbox of run dead (pid 999999999 gone): "
                                   "no client for that backend")
    assert reaper(ledger, microvm=client).apply(plan) == {"sandboxes": [], "failed": []}
    assert client.deleted == []


def test_apply_deletes_planned_sandboxes_and_records_the_release(tmp_path):
    """Each ledger claim is checked against its own backend's listing and freed
    through that backend's client: a podman container is planned when podman
    lists it, whatever AgentENV knows."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="sandbox", id="ctr", pid=999999999, backend="podman")
    ledger._append("claim", run_id="dead", kind="sandbox", id="vm", pid=999999999)
    microvm = FakeClient(sandboxes=[{"id": "vm", "created_at": None, "names": []}])
    podman = FakeClient(sandboxes=[{"id": "ctr", "created_at": None, "names": []}])
    r = reaper(ledger, microvm=microvm, podman=podman)
    plan = r.plan()
    assert sorted(plan.sandboxes) == ["ctr", "vm"] and plan.backends == {"ctr": "podman", "vm": "microvm"}
    assert plan.reasons["ctr"] == "orphan of run dead (pid 999999999 gone)"
    assert r.apply(plan) == {"sandboxes": plan.sandboxes, "failed": []}
    assert (microvm.deleted, podman.deleted) == (["vm"], ["ctr"])
    # released entries are recorded so a second pass does not retry them
    assert all(res.released for res in ledger.resources())
    assert r.plan().total() == 0


def test_apply_reports_failures_without_raising(tmp_path):
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    client = FakeClient()
    client.fail.add("sb-bad")
    plan = ReapPlan(sandboxes=["sb-bad", "sb-ok"], backends={"sb-bad": "microvm", "sb-ok": "microvm"})
    done = reaper(ledger, microvm=client).apply(plan)
    assert done["failed"] == ["sb-bad"]
    assert done["sandboxes"] == ["sb-ok"]


@pytest.mark.parametrize(
    "text,expected",
    [("30s", 30), ("15m", 900), ("2h", 7200), ("1d", 86400), ("soon", ValueError)],
)
def test_parse_duration(text, expected):
    if expected is ValueError:
        with pytest.raises(ValueError):
            parse_duration(text)
    else:
        assert parse_duration(text).total_seconds() == expected


def test_a_dead_runs_sandbox_is_reaped_while_a_live_runs_is_not(tmp_path):
    """A dead run's sandbox burns compute right now; a live run's is in use;
    a dead run's sandbox the backend no longer lists is a stale entry, harmless."""
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="sandbox", id="sb", pid=999999999)
    ledger._append("claim", run_id="mine", kind="sandbox", id="sb-live", pid=os.getpid())
    ledger._append("claim", run_id="dead", kind="sandbox", id="already-gone", pid=999999999)
    client = FakeClient(sandboxes=[
        {"id": "sb", "created_at": None, "names": []},
        {"id": "sb-live", "created_at": None, "names": []},
    ])
    plan = reaper(ledger, microvm=client).plan()
    assert plan.sandboxes == ["sb"]
    assert plan.reasons["sb"] == "orphan of run dead (pid 999999999 gone)"


# --- the podman client: podman ps / inspect / rm -f, then the store -------------
def _leaked_container(store, name: str, created: int = 1700000000):
    """A container's traces on disk -- rootfs subvolume, its config, socket dir --
    and the fake podman's knowledge of it."""
    (store / "sandboxes" / name).mkdir(parents=True)
    (store / "sandboxes" / (name + ".json")).write_text("{}")
    sockets = store / "sockets" / name
    sockets.mkdir(parents=True)
    (sockets / "runtime.sock").write_text("")
    return {"name": name, "store": str(store), "sockets": str(sockets), "created": created}


def test_the_podman_client_lists_managed_containers_with_their_age(tmp_path, store, fake_podman, monkeypatch):
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", json.dumps([
        _leaked_container(store, "sprout-000000000001", created=1700000000)]))
    listed = PodmanClient().list_sandboxes()
    assert listed == [{"id": "sprout-000000000001", "created_at": "2023-11-14T22:13:20+00:00", "names": []}], \
        "named by the pool, so never a deliberately named artifact"
    assert recorded_calls(tmp_path, "podman") == [
        ["ps", "-a", "--filter", "label=sprout.managed=1", "--format", "json"],
        ["network", "ls", "--filter", "label=sprout.managed=1", "--format", "json"]]


def test_a_host_without_podman_lists_nothing(tmp_path):
    client = PodmanClient(podman_bin=str(tmp_path / "no-such-podman"))
    assert client.list_sandboxes() == []
    assert client.delete_sandbox("sprout-000000000001") is True, "nothing exists there to reclaim"


def test_the_podman_client_removes_a_container_the_way_the_pool_destroys_it(
        tmp_path, store, fake_podman, monkeypatch):
    """Container first (its rootfs is mounted while it exists), then the
    network named after it, the rootfs subvolume with its config and the
    socket directory -- each located from the container's name, labels and
    mounts, since the ledger records only the name."""
    name = "sprout-000000000002"
    leaked = _leaked_container(store, name)
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", json.dumps([leaked]))
    monkeypatch.setenv("FAKE_PODMAN_NETWORKS", json.dumps([
        {"name": name, "store": str(store), "created": "2023-11-14T22:13:20.123456789Z"}]))
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="sandbox", id=name, pid=999999999, backend="podman")
    r = reaper(ledger, podman=PodmanClient())
    plan = r.plan()
    assert plan.sandboxes == [name] and plan.backends == {name: "podman"}
    assert r.apply(plan) == {"sandboxes": [name], "failed": []}
    podman = recorded_calls(tmp_path, "podman")
    rootfs = store / "sandboxes" / name
    assert podman[-5:] == [["inspect", "--type", "container", "--format", "json", name], ["rm", "-f", name],
                           ["network", "exists", name], ["network", "rm", name],
                           ["unshare", "find", str(rootfs), "-mindepth", "1", "-inum", "256"]]
    assert fake_networks(tmp_path) == {}
    assert recorded_calls(tmp_path, "btrfs")[-2:] == [
        ["property", "set", str(rootfs), "ro", "false"], ["subvolume", "delete", str(rootfs)]]
    sequence = recorded_sequence(tmp_path)
    assert sequence.index(["podman", "rm", "-f", name]) < sequence.index(["btrfs", "subvolume", "delete", str(rootfs)]), \
        "the container first: its rootfs is mounted while it exists"
    assert not rootfs.exists() and not (store / "sandboxes" / (name + ".json")).exists()
    assert not (store / "sockets" / name).exists()
    assert all(res.released for res in ledger.resources())


def test_a_rootfs_whose_container_is_gone_is_found_through_the_store_and_reclaimed(
        tmp_path, store, fake_podman, monkeypatch):
    """What a destroy that failed after `podman rm -f` leaves: a rootfs (here
    with a subvolume made inside it, the usual cause), its config and socket
    directory, and no container to find them by. The ledger's claim names the
    store, so the client lists the rootfs as a sandbox, the dead pid makes it
    an orphan, and removal deletes the nested subvolume first."""
    name = "sprout-000000000005"
    leaked = _leaked_container(store, name)
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", "[]")
    rootfs = store / "sandboxes" / name
    (rootfs / "adv" / "nested").mkdir(parents=True)
    monkeypatch.setenv("FAKE_PODMAN_NESTED", "adv/nested")
    ledger = ResourceLedger(tmp_path / "res.jsonl")
    ledger._append("claim", run_id="dead", kind="sandbox", id=name, pid=999999999, backend="podman",
                   store=leaked["store"])
    client = PodmanClient(stores=[resource.store for resource in ledger.resources() if resource.store])
    assert client.stores == [str(store.resolve())]
    [listed] = client.list_sandboxes()
    assert listed["id"] == name and listed["names"] == [] and listed["created_at"].endswith("+00:00"), \
        "a rootfs without a container is listed, dated by its directory"
    r = reaper(ledger, podman=client)
    plan = r.plan()
    assert plan.sandboxes == [name] and plan.reasons[name] == "orphan of run dead (pid 999999999 gone)"
    assert r.apply(plan) == {"sandboxes": [name], "failed": []}
    sequence = recorded_sequence(tmp_path)
    nested_gone = sequence.index(["podman", "unshare", "btrfs", "subvolume", "delete", str(rootfs / "adv" / "nested")])
    rootfs_gone = sequence.index(["btrfs", "subvolume", "delete", str(rootfs)])
    assert nested_gone < rootfs_gone, "deepest first: a subvolume holding another cannot be deleted"
    assert not [c for c in recorded_calls(tmp_path, "podman") if c[0] == "rm"], "no container to remove"
    assert not rootfs.exists() and not (store / "sandboxes" / (name + ".json")).exists()
    assert not (store / "sockets" / name).exists(), "the socket directory the pool gives such a sandbox by default"
    assert PodmanClient(stores=[str(store)]).list_sandboxes() == [] and reaper(ledger, podman=client).plan().total() == 0
    assert PodmanClient().list_sandboxes() == [], "without a store to look in, only containers are listed"


@pytest.mark.parametrize("rootfs_left", [True, False], ids=["rm -f by hand", "network alone"])
def test_a_container_removed_by_hand_is_reclaimed_whole_through_its_networks_store_label(
        tmp_path, store, fake_podman, monkeypatch, rootfs_left):
    """`podman rm -f` by hand leaves the network, the rootfs and the socket
    directory; the network's `sprout.store` label says where the rest is, so
    no store has to be given. A run killed between `network create` and `run`,
    or a network whose `network rm` failed, leaves the network alone: nothing
    else names it, but the pool labelled it."""
    name = "sprout-000000000008"
    if rootfs_left:
        _leaked_container(store, name)
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", "[]")
    monkeypatch.setenv("FAKE_PODMAN_NETWORKS", json.dumps([
        {"name": name, "store": str(store), "created": "2023-11-14T22:13:20.123456789Z"}]))
    client = PodmanClient()
    assert client.list_sandboxes() == [{"id": name, "created_at": "2023-11-14T22:13:20.123456+00:00", "names": []}], \
        "listed from the network, dated by it"
    assert client.delete_sandbox(name) is True
    assert fake_networks(tmp_path) == {} and ["network", "rm", name] in recorded_calls(tmp_path, "podman")
    assert not (store / "sandboxes" / name).exists() and not (store / "sandboxes" / (name + ".json")).exists()
    assert not (store / "sockets" / name).exists()


def test_a_container_gone_between_plan_and_apply_counts_as_reclaimed(tmp_path, store, fake_podman, monkeypatch):
    monkeypatch.setenv("FAKE_PODMAN_FAIL", "inspect")
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", "[]")
    assert PodmanClient().delete_sandbox("sprout-000000000003") is True
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", json.dumps([_leaked_container(store, "sprout-000000000003")]))
    assert PodmanClient().delete_sandbox("sprout-000000000003") is False, \
        "still listed but not inspectable: not reclaimed, and not claimed to be"


def test_a_failed_rm_leaves_the_rootfs_alone_and_reports(tmp_path, store, fake_podman, monkeypatch, capsys):
    """A subvolume cannot be deleted while its container mounts it; a failed
    `podman rm -f` therefore stops the sequence and the container is reported."""
    name = "sprout-000000000004"
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", json.dumps([_leaked_container(store, name)]))
    monkeypatch.setenv("FAKE_PODMAN_FAIL", "rm")
    assert PodmanClient().delete_sandbox(name) is False
    assert (store / "sandboxes" / name).is_dir() and (store / "sockets" / name).is_dir()
    assert "podman rm" in capsys.readouterr().err
    assert not [c for c in recorded_calls(tmp_path, "btrfs") if c[:2] == ["subvolume", "delete"]]


# --- sweep: the one legitimate way a snapshot dies -----------------------------
def _journal_with_snapshots(path, ids, backend=None, store=None):
    lines = [{"v": 2, "type": "run.started", "seq": 1, "run_id": "r"}]
    for n, sid in enumerate(ids, 2):
        lines.append({"v": 2, "type": "checkpoint.captured", "seq": n,
                      "run_id": "r", "snapshot_id": sid})
    if backend:
        created = {"v": 2, "type": "sandbox.created", "seq": len(lines) + 1, "run_id": "r",
                   "sandbox_id": "sprout-000000000009", "backend": backend}
        if store:
            created["store"] = str(store)
        lines.append(created)
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")


class NoAgentEnv:
    """AgentENV, consulted when it must not be."""

    def __init__(self, *args, **kwargs):
        raise AssertionError("AgentENV consulted for a run that is not its own")


def _snapshot_subvolumes(store, ids):
    for sid in ids:
        (store / "snapshots" / sid).mkdir(parents=True)
        (store / "snapshots" / (sid + ".json")).write_text("{}")


def test_sweep_deletes_a_podman_journals_subvolumes_through_the_store(tmp_path, store, monkeypatch, capsys):
    """A podman run's snapshot ids are subvolumes of the store its journal
    names; sweep deletes them through BtrfsStore -- no pool, no runtime binary,
    no AgentENV -- and one already gone counts as deleted, as a 404 does."""
    import sprout.harness.cli as cli

    monkeypatch.setattr(cli, "AgentEnvClient", NoAgentEnv)
    ids = ["sprout-000000000009-0001", "sprout-000000000009-0002", "sprout-000000000009-0003-final"]
    journal = tmp_path / "run.jsonl"
    _journal_with_snapshots(journal, ids, backend="podman", store=store)
    _snapshot_subvolumes(store, ids[:2])

    assert cli.main(["sweep", str(journal)]) == 0
    assert "would delete 3 podman snapshot(s)" in capsys.readouterr().out
    assert all((store / "snapshots" / sid).is_dir() for sid in ids[:2]), "a dry run deletes nothing"

    assert cli.main(["sweep", str(journal), "--yes"]) == 0
    assert "deleted 3/3" in capsys.readouterr().out
    for sid in ids:
        assert not (store / "snapshots" / sid).exists() and not (store / "snapshots" / (sid + ".json")).exists()
    deletions = [c for c in recorded_calls(tmp_path, "btrfs") if c[:2] == ["subvolume", "delete"]]
    assert deletions == [["subvolume", "delete", str(store / "snapshots" / sid)] for sid in ids[:2]]
    marker = json.loads(journal.read_text().splitlines()[-1])
    assert (marker["type"], marker["backend"], marker["deleted"], marker["failed"]) == \
        ("snapshots.swept", "podman", 3, 0)
    assert cli.main(["sweep", str(journal), "--yes"]) == 0, "idempotent"
    assert "already swept" in capsys.readouterr().out


def test_sweep_reads_a_killed_runs_torn_journal(tmp_path, store, monkeypatch, capsys):
    """A run killed mid-write leaves half a record at the end of its journal. The
    snapshots the complete records name are deleted all the same; the fragment
    moves into the swept marker, so the journal stays a file of whole lines
    that every reader parses (glued to the fragment, the marker would have made
    one corrupt line)."""
    import sprout.harness.cli as cli
    from sprout.harness.core.journal import journal_events

    monkeypatch.setattr(cli, "AgentEnvClient", NoAgentEnv)
    ids = ["sprout-000000000009-0001", "sprout-000000000009-0002"]
    journal = tmp_path / "run.jsonl"
    _journal_with_snapshots(journal, ids, backend="podman", store=store)
    _snapshot_subvolumes(store, ids)
    with journal.open("a") as fh:
        fh.write('{"v": 2, "type": "tool.finished", "seq": 99, "output": "cut off he')
    assert cli.main(["sweep", str(journal), "--yes"]) == 0
    assert "deleted 2/2" in capsys.readouterr().out
    assert not any((store / "snapshots" / sid).exists() for sid in ids)
    events = journal_events(journal)
    assert events[-1]["type"] == "snapshots.swept" and events[-1]["deleted"] == 2
    assert events[-1]["torn_tail"] == '{"v": 2, "type": "tool.finished", "seq": 99, "output": "cut off he'
    lines = journal.read_bytes().splitlines(keepends=True)
    assert all(line.endswith(b"\n") for line in lines) and len(lines) == len(events), \
        "every line is a whole record again"
    assert [json.loads(line) for line in lines] == events


def test_sweep_needs_the_store_for_a_podman_journal_that_names_none(tmp_path, store, monkeypatch, capsys):
    """A journal written before the store was recorded, or a store that moved:
    --store says where the subvolumes are; without it, or with a store that is
    not on btrfs, nothing is deleted and nothing is marked swept."""
    import sprout.harness.cli as cli

    monkeypatch.setattr(cli, "AgentEnvClient", NoAgentEnv)
    ids = ["sprout-000000000009-0001"]
    journal = tmp_path / "run.jsonl"
    _journal_with_snapshots(journal, ids, backend="podman")
    _snapshot_subvolumes(store, ids)
    assert cli.main(["sweep", str(journal), "--yes"]) == 1
    assert "--store" in capsys.readouterr().err
    assert (store / "snapshots" / ids[0]).is_dir() and "snapshots.swept" not in journal.read_text()

    monkeypatch.setenv("FAKE_FINDMNT_FSTYPE", "ext4")
    assert cli.main(["sweep", str(journal), "--yes", "--store", str(store)]) == 1
    assert "needs btrfs mounted with user_subvol_rm_allowed" in capsys.readouterr().err
    assert (store / "snapshots" / ids[0]).is_dir() and "snapshots.swept" not in journal.read_text()
    monkeypatch.delenv("FAKE_FINDMNT_FSTYPE")

    assert cli.main(["sweep", str(journal), "--yes", "--store", str(store)]) == 0
    assert not (store / "snapshots" / ids[0]).exists()
    assert "snapshots.swept" in journal.read_text()


def test_sweep_is_a_dry_run_unless_told_otherwise(tmp_path, monkeypatch, capsys):
    """Deleting history must take two deliberate acts: naming the journal AND
    typing --yes. The default prints what WOULD die and touches nothing."""
    import sprout.harness.cli as cli

    journal = tmp_path / "run.jsonl"
    _journal_with_snapshots(journal, ["s1", "s2"])
    deleted = []

    class FakeEnv:
        def delete_snapshot(self, sid):
            deleted.append(sid)
            return True

    monkeypatch.setattr(cli, "AgentEnvClient", FakeEnv)
    assert cli.main(["sweep", str(journal)]) == 0
    assert deleted == []
    assert "would delete 2 microvm" in capsys.readouterr().out
    # the journal is untouched -- no swept marker on a dry run
    assert "snapshots.swept" not in journal.read_text()

    assert cli.main(["sweep", str(journal), "--yes"]) == 0
    assert deleted == ["s1", "s2"]
    marker = [json.loads(line) for line in journal.read_text().splitlines()][-1]
    assert marker["type"] == "snapshots.swept" and marker["deleted"] == 2 and marker["backend"] == "microvm"


def test_sweep_refuses_to_double_sweep_and_missing_journals(tmp_path, monkeypatch, capsys):
    import sprout.harness.cli as cli

    journal = tmp_path / "run.jsonl"
    _journal_with_snapshots(journal, ["s1"])
    calls = []

    class FakeEnv:
        def delete_snapshot(self, sid):
            calls.append(sid)
            return True

    monkeypatch.setattr(cli, "AgentEnvClient", FakeEnv)
    assert cli.main(["sweep", str(journal), "--yes"]) == 0
    assert cli.main(["sweep", str(journal), "--yes"]) == 0  # idempotent
    assert calls == ["s1"], "a swept journal must not be swept again"
    assert cli.main(["sweep", str(tmp_path / "nope.jsonl")]) == 2


def test_reap_cli_plans_sandboxes_of_both_backends(tmp_path, monkeypatch, capsys):
    """The command a person types habitually: it names sandboxes, never
    snapshots, and frees each through the backend that made it."""
    import sprout.harness.cli as cli

    ledger = tmp_path / "res.jsonl"
    ResourceLedger(ledger)._append("claim", run_id="dead", kind="sandbox", id="sb", pid=999999999)
    ResourceLedger(ledger)._append("claim", run_id="dead", kind="sandbox", id="ctr",
                                   pid=999999999, backend="podman")
    ResourceLedger(ledger)._append("claim", run_id="dead", kind="sandbox", id="gone-ctr",
                                   pid=999999999, backend="podman", store="/srv/store-a")
    microvm = FakeClient(sandboxes=[{"id": "sb", "created_at": None, "names": []}])
    podman = FakeClient(sandboxes=[{"id": "ctr", "created_at": None, "names": []}])
    stores_given = []
    monkeypatch.setattr(cli, "AgentEnvClient", lambda: microvm)
    monkeypatch.setattr(cli, "PodmanClient", lambda **kwargs: (stores_given.append(kwargs["stores"]), podman)[1])
    assert cli.main(["reap", "--ledger", str(ledger), "--dry-run", "--store", "/srv/store-b"]) == 0
    assert stores_given == [{"/srv/store-a", "/srv/store-b"}], \
        "the podman client looks in the stores the ledger names and the ones on the command line"
    out = capsys.readouterr().out
    assert "sandbox   microvm  sb" in out and "sandbox   podman   ctr" in out
    assert "2 sandbox(es) would be freed" in out
    assert microvm.deleted == [] and podman.deleted == []
    assert cli.main(["reap", "--ledger", str(ledger)]) == 0
    assert (microvm.deleted, podman.deleted) == (["sb"], ["ctr"])
    assert "freed 2 sandbox(es)" in capsys.readouterr().out


def test_nothing_in_the_run_lifecycle_calls_snapshot_deletion():
    """The design boundary in one test: deletion is never wired into teardown.
    The orchestrator, the checkpointing path and the session may CREATE
    snapshots; the only callers of deletion are the two explicit surfaces
    (harness sweep, and methods.evaluate releasing a failed attempt's memory
    snapshot) -- and the reaper is not one of them."""
    from pathlib import Path

    root = Path(__file__).resolve().parents[2]
    for module in ("src/sprout/harness/orchestrator/run.py", "src/sprout/harness/tools/access.py",
                   "src/sprout/harness_checkpoint/checkpointer.py", "src/sprout/harness_sandbox/session.py"):
        text = (root / module).read_text()
        assert "delete_snapshot" not in text, \
            "%s must not delete snapshots -- runs end, history stays" % module
    for cls in (Reaper, PodmanClient):
        assert "delete_snapshot" not in inspect.getsource(cls), "the reaper reclaims compute, never history"

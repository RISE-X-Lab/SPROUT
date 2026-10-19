"""The reaper: which sandboxes a reap plan frees, which it leaves, and how it frees them.

The clients here are stubs that list what a test gives them and record each deletion, because the
deletion through the right backend is the behavior. ``test_podman_client.py`` checks the podman client
against the fake podman and btrfs, and this file checks the AgentENV client against a stub server.
"""

from __future__ import annotations

import json
import os
from datetime import timedelta
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.run.reap import AgentEnvClient, Reaper, parse_duration
from sprout.worker.run.resources import ResourceLedger
from tests.worker.run.agentenv import agentenv_stub

#: Above the largest pid_max of Linux (2**22), so no process has it.
DEAD_PID = 999_999_999


class StubClient:
    """A backend that lists ``sandboxes`` and records each deletion. Deleting one of ``failing`` fails."""

    def __init__(self, *sandboxes: dict[str, Any], failing: tuple[str, ...] = ()) -> None:
        self.sandboxes = list(sandboxes)
        self.failing = set(failing)
        self.deleted: list[str] = []
        self.deleted_snapshots: list[str] = []

    def list_sandboxes(self) -> list[dict[str, Any]]:
        return list(self.sandboxes)

    def delete_sandbox(self, sandbox_id: str) -> bool:
        if sandbox_id in self.failing:
            return False
        self.deleted.append(sandbox_id)
        return True

    def delete_snapshot(self, snapshot_id: str) -> bool:
        self.deleted_snapshots.append(snapshot_id)
        return True


def listed(sandbox_id: str, *, created_at: str | None = None, names: tuple[str, ...] = ()) -> dict[str, Any]:
    """A sandbox as a backend lists it."""
    return {"id": sandbox_id, "created_at": created_at, "names": list(names)}


def ledger_with(path: Path, *claims: dict[str, Any]) -> ResourceLedger:
    """The ledger at ``path`` with the ``claims`` of runs whose processes are gone, unless a claim says."""
    with path.open("a") as stream:
        for fields in claims:
            entry = {"event": "claim", "run_id": "run-1", "kind": "sandbox", "pid": DEAD_PID, **fields}
            stream.write(json.dumps(entry) + "\n")
    return ResourceLedger(path)


def test_a_listed_sandbox_of_a_dead_process_is_planned_and_one_of_a_live_process_is_not(tmp_path):
    ledger = ledger_with(
        tmp_path / "resources.jsonl",
        {"id": "sandbox-dead"},
        {"id": "sandbox-live", "run_id": "run-2", "pid": os.getpid()},
    )
    microvm = StubClient(listed("sandbox-dead"), listed("sandbox-live"))

    plan = Reaper({"microvm": microvm}, ledger).plan()

    assert (plan.sandboxes, plan.backends, plan.reasons) == (
        ["sandbox-dead"],
        {"sandbox-dead": "microvm"},
        {"sandbox-dead": "orphan of run run-1 (pid 999999999 gone)"},
    )


def test_an_orphan_that_its_backend_no_longer_lists_is_not_planned(tmp_path):
    ledger = ledger_with(tmp_path / "resources.jsonl", {"id": "sandbox-gone"})

    plan = Reaper({"microvm": StubClient()}, ledger).plan()

    assert (plan.sandboxes, plan.kept, plan.foreign) == ([], [], [])


def test_a_kept_sandbox_is_left_until_its_keep_until_has_passed(tmp_path):
    ledger = ledger_with(
        tmp_path / "resources.jsonl",
        {"id": "kept-until-released", "keep": True},
        {"id": "kept-until-2999", "keep": True, "keep_until": "2999-01-01T00:00:00+00:00"},
        {"id": "kept-until-2020", "keep": True, "keep_until": "2020-01-01T00:00:00+00:00"},
    )
    microvm = StubClient(listed("kept-until-released"), listed("kept-until-2999"), listed("kept-until-2020"))

    plan = Reaper({"microvm": microvm}, ledger).plan()

    assert (plan.sandboxes, plan.kept, plan.reasons) == (
        ["kept-until-2020"],
        ["kept-until-released", "kept-until-2999"],
        {"kept-until-2020": "kept by run run-1 until 2020-01-01T00:00:00+00:00"},
    )


def test_an_orphan_of_a_backend_without_a_client_is_reported_as_foreign_and_never_deleted(tmp_path):
    """An orphan that no client of the reaper can look for is not taken for one that is already gone."""
    ledger = ledger_with(tmp_path / "resources.jsonl", {"id": "sandbox-docker", "backend": "docker"})
    microvm = StubClient(listed("sandbox-docker"))
    reaper = Reaper({"microvm": microvm}, ledger)

    plan = reaper.plan()
    done = reaper.apply(plan)

    assert (plan.sandboxes, plan.foreign, plan.reasons) == (
        [],
        ["sandbox-docker"],
        {"sandbox-docker": "docker sandbox of run run-1 (pid 999999999 gone): no client for that backend"},
    )
    assert (done, microvm.deleted) == ({"sandboxes": [], "failed": []}, [])


def test_each_planned_sandbox_is_deleted_through_the_client_of_its_backend(tmp_path):
    ledger = ledger_with(
        tmp_path / "resources.jsonl",
        {"id": "container-1", "backend": "podman"},
        {"id": "vm-1", "backend": "microvm"},
    )
    microvm, podman = StubClient(listed("vm-1")), StubClient(listed("container-1"))
    reaper = Reaper({"microvm": microvm, "podman": podman}, ledger)

    done = reaper.apply(reaper.plan())

    assert (done, microvm.deleted, podman.deleted) == (
        {"sandboxes": ["container-1", "vm-1"], "failed": []},
        ["vm-1"],
        ["container-1"],
    )


def test_a_deleted_sandbox_is_recorded_as_released_so_that_the_next_plan_leaves_it(tmp_path):
    ledger = ledger_with(tmp_path / "resources.jsonl", {"id": "sandbox-1"})
    reaper = Reaper({"microvm": StubClient(listed("sandbox-1"))}, ledger)
    reaper.apply(reaper.plan())

    plan = reaper.plan()

    assert (plan.sandboxes, [resource.released for resource in ledger.resources()]) == ([], [True])


def test_a_failed_deletion_is_reported_not_raised_and_planned_again(tmp_path):
    ledger = ledger_with(tmp_path / "resources.jsonl", {"id": "sandbox-stuck"}, {"id": "sandbox-1"})
    microvm = StubClient(listed("sandbox-stuck"), listed("sandbox-1"), failing=("sandbox-stuck",))
    reaper = Reaper({"microvm": microvm}, ledger)

    done = reaper.apply(reaper.plan())

    assert (done, reaper.plan().sandboxes) == (
        {"sandboxes": ["sandbox-1"], "failed": ["sandbox-stuck"]},
        ["sandbox-stuck"],
    )


def test_a_snapshot_claim_of_an_older_ledger_is_neither_planned_nor_kept_nor_deleted(tmp_path):
    """Each snapshot of a finished run is a branch point. Only a sweep of its journal deletes it."""
    ledger = ledger_with(
        tmp_path / "resources.jsonl", {"kind": "snapshot", "id": "snapshot-1"}, {"id": "sandbox-1"}
    )
    microvm = StubClient(listed("snapshot-1"), listed("sandbox-1"))
    reaper = Reaper({"microvm": microvm}, ledger)

    plan = reaper.plan()
    reaper.apply(plan)

    assert (plan.sandboxes, plan.kept, microvm.deleted, microvm.deleted_snapshots) == (
        ["sandbox-1"],
        [],
        ["sandbox-1"],
        [],
    )


def test_a_sandbox_that_the_ledger_never_saw_is_planned_only_when_asked(tmp_path):
    """A sweep by age alone can delete a colleague's experiment in progress."""
    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    microvm = StubClient(listed("stranger", created_at="2020-01-01T00:00:00Z"))

    plan = Reaper({"microvm": microvm}, ledger).plan()

    assert plan.sandboxes == []


def test_unknown_sandboxes_are_planned_when_unnamed_and_older_than_the_cutoff(tmp_path):
    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    microvm = StubClient(
        listed("old", created_at="2020-01-01T00:00:00Z"),
        listed("recent", created_at="2999-01-01T00:00:00Z"),
        listed("named", created_at="2020-01-01T00:00:00Z", names=("golden-base",)),
        listed("undated"),
    )

    plan = Reaper({"microvm": microvm}, ledger).plan(include_unknown=True, older_than=timedelta(hours=24))

    assert (plan.sandboxes, plan.kept, plan.reasons) == (
        ["old"],
        ["named"],
        {"old": "not in ledger, created 2020-01-01T00:00:00Z"},
    )


def test_unknown_sandboxes_are_deleted_through_the_client_that_listed_them(tmp_path):
    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    microvm = StubClient(listed("vm-1", created_at="2020-01-01T00:00:00Z"))
    podman = StubClient(listed("container-1", created_at="2020-01-01T00:00:00Z"))
    reaper = Reaper({"microvm": microvm, "podman": podman}, ledger)

    reaper.apply(reaper.plan(include_unknown=True, older_than=timedelta(hours=24)))

    assert (microvm.deleted, podman.deleted) == (["vm-1"], ["container-1"])


@pytest.mark.parametrize(
    "text,seconds",
    [("30s", 30), ("15m", 900), ("2h", 7200), ("1d", 86400), ("1.5h", 5400), (" 24H ", 86400)],
)
def test_a_duration_is_a_number_and_a_unit(text, seconds):
    assert parse_duration(text) == timedelta(seconds=seconds)


@pytest.mark.parametrize("text", ["soon", "10", "10w", "-5m", "1h30m", ""])
def test_a_duration_without_a_known_unit_is_refused(text):
    with pytest.raises(ValueError, match="^bad duration"):
        parse_duration(text)


def test_the_agentenv_client_lists_each_sandbox_with_its_start_time_and_names():
    with agentenv_stub() as server:
        server.sandboxes = [
            {"sandboxID": "vm-1", "startedAt": "2030-01-01T00:00:00Z", "names": ["golden-base"]},
            {"sandboxID": "vm-2", "createdAt": "2030-01-02T00:00:00Z"},
        ]

        sandboxes = AgentEnvClient(base_url=server.url).list_sandboxes()

    assert sandboxes == [
        {"id": "vm-1", "created_at": "2030-01-01T00:00:00Z", "names": ["golden-base"]},
        {"id": "vm-2", "created_at": "2030-01-02T00:00:00Z", "names": []},
    ]


def test_the_agentenv_client_sends_its_api_key_with_each_request():
    with agentenv_stub() as server:
        client = AgentEnvClient(base_url=server.url + "/", api_key="fixture-key")

        client.list_sandboxes()
        client.delete_sandbox("vm-1")

    assert server.requests == [
        ("GET", "/sandboxes", "fixture-key"),
        ("DELETE", "/sandboxes/vm-1", "fixture-key"),
    ]


def test_an_agentenv_that_does_not_answer_lists_no_sandboxes():
    with agentenv_stub() as server:
        url = server.url

    assert AgentEnvClient(base_url=url).list_sandboxes() == []


@pytest.mark.parametrize(
    "status,deleted",
    [(200, True), (204, True), (404, True), (409, False), (500, False)],
    ids=["ok", "no-content", "already-gone", "conflict", "server-error"],
)
def test_an_agentenv_sandbox_counts_as_deleted_when_nothing_of_it_is_left(status, deleted):
    with agentenv_stub() as server:
        server.delete_statuses["/sandboxes/vm-1"] = status

        result = AgentEnvClient(base_url=server.url).delete_sandbox("vm-1")

    assert result is deleted

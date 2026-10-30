"""One run from start to finish: its sandbox, its checkpoints, its records, and the teardown on each path.

The sandbox is a ``DirectorySession`` (``conftest.py``). A run of the ``nop`` agent calls no tool, so
it shows the lifecycle alone. A run of mini on a scripted model shows the per-turn checkpoints.
"""

from __future__ import annotations

import shutil
import time
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.contract.run_spec import RunSpec
from sprout.worker.checkpoint.rollback import branch_checkpoints
from sprout.worker.run.journal import read_journal
from sprout.worker.run.resources import ResourceLedger
from sprout.worker.run.runner import Runner
from tests.support.http import chat_completion, chat_server
from tests.support.paths import HARBOR_SHELL_TASK
from tests.worker.run.directory_session import DirectorySession
from tests.worker.run.rollouts import SUBMIT, events, needs_mini, rollout_spec

PODMAN = {
    "backend": "podman",
    "podman": {"store": "/path/to/store", "runtime_bin": "/path/to/sprout-runtime"},
}
MICROVM = {"backend": "microvm", "microvm": {"server_url": "http://127.0.0.1:8000"}}


def nop_spec(tmp_path: Path, **changes: Any) -> RunSpec:
    """The spec of a run of the ``nop`` agent, from the image ``image``, with its journal in ``tmp_path``."""
    fields: dict[str, Any] = {
        "prompt": "Do nothing.",
        "agent": "nop",
        "sandbox_image": "image",
        "journal_path": tmp_path / "trajectory.jsonl",
        **changes,
    }
    return RunSpec(**fields)


def test_a_run_takes_its_final_disk_snapshot_before_it_destroys_its_sandbox(
    tmp_path, sandbox: DirectorySession
):
    outcome = Runner().run(nop_spec(tmp_path))

    assert (outcome.status, outcome.final_snapshot_id, outcome.sandbox_id) == (
        "completed",
        "snapshot-1",
        "sandbox-1",
    )
    assert sandbox.lifecycle == [("create", "image"), ("snapshot", "snapshot-1"), ("destroy", "sandbox-1")]
    assert [
        (r["snapshot_id"], r["final"], r["disk_only"]) for r in events(outcome, "checkpoint.captured")
    ] == [("snapshot-1", True, True)]


def test_a_run_that_keeps_its_sandbox_does_not_destroy_it(tmp_path, sandbox: DirectorySession):
    Runner().run(nop_spec(tmp_path, keep_sandbox=True))

    assert sandbox.lifecycle == [("create", "image"), ("snapshot", "snapshot-1")]


def test_the_ledger_records_the_claim_and_the_release_of_the_sandbox_and_the_end_of_the_run(
    tmp_path, sandbox: DirectorySession
):
    """A killed process releases nothing, so ``sprout.worker reap`` works from these records."""
    ledger = ResourceLedger(tmp_path / "resources.jsonl")

    Runner(ledger=ledger).run(nop_spec(tmp_path, run_id="run-1", backend=PODMAN))

    assert [
        (entry["event"], entry.get("id"), entry.get("backend"), entry.get("store"))
        for entry in ledger.entries()
    ] == [
        ("claim", "sandbox-1", "podman", "/path/to/store"),
        ("release", "sandbox-1", None, None),
        ("run_done", None, None, None),
    ]


def test_a_failed_final_snapshot_keeps_the_sandbox_and_records_it_as_kept(
    tmp_path, sandbox: DirectorySession
):
    """Until the final snapshot exists, the sandbox holds the only copy of the result of the run."""
    sandbox.failing_snapshots = True
    ledger = ResourceLedger(tmp_path / "resources.jsonl")

    outcome = Runner(ledger=ledger).run(nop_spec(tmp_path, run_id="run-1"))

    assert (outcome.status, outcome.error, sandbox.destroyed) == (
        "error",
        "Final snapshot failed (snapshot failed); sandbox sandbox-1 retained",
        False,
    )
    [retained] = events(outcome, "sandbox.retained")
    assert (retained["sandbox_id"], retained["reason"], retained["error"]) == (
        "sandbox-1",
        "snapshot_pending",
        "snapshot failed",
    )
    assert [(e["event"], e.get("keep")) for e in ledger.entries()] == [
        ("claim", False),
        ("claim", True),
        ("run_done", None),
    ]


def test_a_run_on_a_backend_that_cannot_snapshot_ends_without_a_final_snapshot(
    tmp_path, sandbox: DirectorySession
):
    sandbox.can_snapshot = False

    outcome = Runner().run(nop_spec(tmp_path))

    assert (outcome.status, outcome.final_snapshot_id, sandbox.destroyed) == ("completed", None, True)


def test_a_final_memory_snapshot_follows_the_final_disk_snapshot_when_the_run_asks(
    tmp_path, sandbox: DirectorySession
):
    outcome = Runner().run(nop_spec(tmp_path, final_memory_snapshot=True))

    [memory] = events(outcome, "snapshot.memory")
    assert (outcome.final_snapshot_id, outcome.final_memory_snapshot_id) == ("snapshot-1", "snapshot-2")
    assert (memory["snapshot_id"], memory["after"]) == ("snapshot-2", "snapshot-1")


def test_a_sandbox_that_a_failed_wiring_step_leaves_unreachable_is_destroyed(
    tmp_path, sandbox: DirectorySession
):
    def refuse(kind: str, payload: dict[str, Any]) -> None:
        raise RuntimeError("observer refused the %s" % kind)

    outcome = Runner(on_event=refuse).run(nop_spec(tmp_path))

    assert (outcome.status, outcome.error, sandbox.lifecycle) == (
        "error",
        "RuntimeError: observer refused the sandbox",
        [("create", "image"), ("destroy", "sandbox-1")],
    )


def task_without_solve_sh(tmp_path: Path) -> Path:
    """A copy of the shell task of the fixtures, in ``tmp_path``, whose solution has no ``solve.sh``."""
    task = Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "task"))
    (task / "solution" / "solve.sh").unlink()
    return task


def test_an_agent_that_raises_fails_the_run_and_the_runner_still_snapshots_and_destroys_its_sandbox(
    tmp_path, sandbox: DirectorySession
):
    task = task_without_solve_sh(tmp_path)

    outcome = Runner().run(nop_spec(tmp_path, agent="oracle", harbor_task=str(task)))

    error = "ValueError: %s: missing, and the oracle runs it" % (task / "solution" / "solve.sh")
    assert (outcome.status, outcome.error, sandbox.lifecycle) == (
        "error",
        error,
        [("create", "image"), ("snapshot", "snapshot-1"), ("destroy", "sandbox-1")],
    )
    assert [(r["status"], r["error"]) for r in events(outcome, "run.finished")] == [("error", error)]


@pytest.mark.parametrize(
    "agent,status", [("nop", "completed"), ("oracle", "error")], ids=["agent completes", "agent raises"]
)
def test_the_gateway_of_a_run_stops_when_the_run_ends(tmp_path, sandbox: DirectorySession, agent, status):
    """A budget starts a gateway. A gateway left running holds its port and its thread for nobody."""
    heard: dict[str, dict[str, Any]] = {}
    task = task_without_solve_sh(tmp_path)

    outcome = Runner(on_event=heard.setdefault).run(
        nop_spec(tmp_path, agent=agent, harbor_task=str(task), budget_usd=1.0)
    )

    assert outcome.status == status
    with pytest.raises(httpx.ConnectError):
        httpx.get(heard["gateway"]["url"].rstrip("/") + "/health", timeout=5)


def test_a_sandbox_that_cannot_be_destroyed_leaves_the_run_completed_and_its_claim_to_the_reaper(
    tmp_path, sandbox: DirectorySession
):
    sandbox.destroy_error = "backend unreachable"
    ledger = ResourceLedger(tmp_path / "resources.jsonl")

    outcome = Runner(ledger=ledger).run(nop_spec(tmp_path, run_id="run-1"))

    assert (outcome.status, outcome.error) == ("completed", None)
    assert [(entry["event"], entry.get("id")) for entry in ledger.entries()] == [
        ("claim", "sandbox-1"),
        ("run_done", None),
    ]


@pytest.mark.parametrize(
    "changes,error",
    [
        pytest.param(
            {"agent": "codex"},
            "ValueError: unknown agent 'codex'; the agents are mini, harness, oracle and nop",
            id="agent",
        ),
        pytest.param(
            {"slot": "codex"},
            "ValueError: unknown slot 'codex'; the slots are mini-swe-agent and sprout-harness",
            id="slot",
        ),
        pytest.param(
            {"agent": "oracle"},
            "ValueError: an oracle run needs the harbor_task whose solution it runs",
            id="oracle-without-a-task",
        ),
        pytest.param(
            {"sandbox_image": None},
            "ValueError: RunSpec.sandbox_image is required: the orchestrator owns the sandbox",
            id="no-image",
        ),
    ],
)
def test_a_run_that_cannot_start_is_refused_before_a_sandbox_exists(
    tmp_path, sandbox: DirectorySession, changes, error
):
    outcome = Runner().run(nop_spec(tmp_path, **changes))

    assert (outcome.status, outcome.error, sandbox.lifecycle) == ("error", error, [])
    assert [(r["status"], r["error"]) for r in events(outcome, "run.finished")] == [("error", error)]


def test_a_sandbox_that_cannot_be_created_fails_the_run_with_the_reason_of_the_backend(
    tmp_path, sandbox: DirectorySession
):
    """The session is quiet, so the reason travels in the error, or nobody sees it."""
    sandbox.create_error = "backend said no"

    outcome = Runner().run(nop_spec(tmp_path))

    assert (outcome.status, outcome.error) == (
        "error",
        "RuntimeError: could not create a sandbox from image: backend said no",
    )


def test_an_expired_rollout_group_creates_no_sandbox(tmp_path, sandbox: DirectorySession):
    contract = {"model_endpoint": "http://127.0.0.1:1", "deadline_at": time.time() - 1, "max_turns": 5}

    outcome = Runner().run(nop_spec(tmp_path, extra={"rollout_contract": contract}))

    assert (outcome.error, sandbox.lifecycle) == (
        "ValueError: Rollout group wall-time budget expired before execution",
        [],
    )


@pytest.mark.parametrize(
    "origin,first_types",
    [
        pytest.param(
            {"point_id": "point-1", "snapshot_id": "snapshot-x", "tool_depth": 2},
            ["fork.origin", "sandbox.created"],
            id="branch",
        ),
        # A reader takes a fork.origin event for a branch, and looks for its parent.
        pytest.param(None, ["sandbox.created", "checkpoint.captured"], id="root"),
    ],
)
def test_the_journal_of_a_branch_opens_with_its_origin(
    tmp_path, sandbox: DirectorySession, origin, first_types
):
    outcome = Runner().run(nop_spec(tmp_path, origin=origin))

    records = read_journal(outcome.journal_path)
    assert [record["type"] for record in records[:2]] == first_types


def test_the_origin_of_a_branch_is_journaled_as_given(tmp_path, sandbox: DirectorySession):
    origin = {"point_id": "point-1", "snapshot_id": "snapshot-x", "tool_depth": 2}

    outcome = Runner().run(nop_spec(tmp_path, origin=origin))

    [record] = events(outcome, "fork.origin")
    assert {key: record[key] for key in origin} == origin


@pytest.mark.parametrize(
    "backend,expected",
    [
        pytest.param(
            PODMAN,
            {"sandbox_id": "sandbox-1", "backend": "podman", "image": "image", "store": "/path/to/store"},
            id="podman",
        ),
        # An AgentENV snapshot is a record of the server, not a file in a store.
        pytest.param(
            MICROVM, {"sandbox_id": "sandbox-1", "backend": "microvm", "image": "image"}, id="microvm"
        ),
    ],
)
def test_the_journal_names_the_sandbox_its_backend_and_the_store_of_its_snapshots(
    tmp_path, sandbox: DirectorySession, backend, expected
):
    """``sprout.worker sweep`` deletes the snapshots of a journal where this record says that they are."""
    outcome = Runner().run(nop_spec(tmp_path, backend=backend))

    [record] = events(outcome, "sandbox.created")
    assert {
        key: record[key] for key in record if key in ("sandbox_id", "backend", "image", "store")
    } == expected


@pytest.mark.parametrize("network_mode,allow_internet", [("no-network", False), ("public", True)])
def test_the_sandbox_of_a_harbor_task_starts_on_the_agent_network_of_the_task(
    tmp_path, sandbox: DirectorySession, network_mode, allow_internet
):
    task = Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "task"))
    toml = task / "task.toml"
    toml.write_text(toml.read_text().replace("[agent]\n", '[agent]\nnetwork_mode = "%s"\n' % network_mode))

    Runner().run(nop_spec(tmp_path, harbor_task=str(task), backend=PODMAN))

    assert sandbox.opened_with["backend"]["podman"]["allow_internet"] is allow_internet


@needs_mini
def test_each_closed_turn_is_a_branch_point_whose_snapshot_holds_the_files_of_that_turn(
    tmp_path, sandbox: DirectorySession
):
    """A turn's snapshot is taken when the next turn starts. The last turn's is the final snapshot."""
    replies = [
        chat_completion("printf first > answer", "cat answer"),
        chat_completion("printf second > answer"),
        chat_completion(SUBMIT),
    ]
    with chat_server(replies) as server:
        outcome = Runner().run(rollout_spec(tmp_path, server.url))

    points = branch_checkpoints(outcome.journal_path)
    assert {step: sandbox.snapshots[point.snapshot_id] for step, point in points.items()} == {
        2: {"answer": b"first"},
        3: {"answer": b"second"},
        4: {"answer": b"second"},
    }
    assert (outcome.status, outcome.checkpoints, outcome.final_snapshot_id) == (
        "completed",
        2,
        points[4].snapshot_id,
    )


@needs_mini
def test_a_run_whose_tool_call_did_not_settle_ends_as_an_error_and_its_final_snapshot_is_no_branch_point(
    tmp_path, sandbox: DirectorySession
):
    """Nothing after a call whose outcome is unknown pairs with a known filesystem."""
    sandbox.unsettled = "printf second"
    replies = [chat_completion("printf first > answer"), chat_completion("printf second > answer")]
    with chat_server(replies) as server:
        outcome = Runner().run(rollout_spec(tmp_path, server.url))

    assert (outcome.status, outcome.error, outcome.stop_reason) == (
        "error",
        "execution uncertain at step 2 (call call-2)",
        None,
    )
    assert (sorted(branch_checkpoints(outcome.journal_path)), outcome.final_snapshot_id) == (
        [1],
        "snapshot-2",
    )


@needs_mini
def test_a_re_board_moves_the_claim_and_the_name_of_the_sandbox_of_the_run_to_the_replacement(
    tmp_path, sandbox: DirectorySession
):
    """A process killed after the re-board must leave a sandbox that the reaper can attribute."""
    sandbox.compacting = True
    ledger = ResourceLedger(tmp_path / "resources.jsonl")
    replies = [
        chat_completion("printf a > answer"),
        chat_completion("printf b > answer"),
        chat_completion(SUBMIT),
    ]
    with chat_server(replies) as server:
        outcome = Runner(ledger=ledger).run(
            rollout_spec(tmp_path, server.url, run_id="run-1", backend=MICROVM)
        )

    assert [(entry["event"], entry.get("id")) for entry in ledger.entries()] == [
        ("claim", "sandbox-1"),
        ("claim", "sandbox-2"),
        ("release", "sandbox-2"),
        ("run_done", None),
    ]
    assert (outcome.sandbox_id, sandbox.on_swap) == ("sandbox-2", [])

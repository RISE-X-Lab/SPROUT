"""The index of an attempt's tool calls and recovery points, and when a branch can start from a point.

The tests of ``Index.valid`` are small. The others use the PostgreSQL test database (``run_store``), and
skip without it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import NativePoint
from sprout.runstore.index import Index
from sprout.runstore.store import Conflict, Store
from tests.runstore.builders import job, native_history
from tests.support.runstore import registered

#: A store that ``Index.valid`` never connects to.
UNUSED_STORE = Store("dbname=never-connected")


def started(call_id: str, command: str) -> dict[str, Any]:
    return {"type": "tool.started", "call_id": call_id, "name": "bash", "args": {"command": command}}


def finished(call_id: str, output: str) -> dict[str, Any]:
    return {"type": "tool.finished", "call_id": call_id, "output": output}


def claimed_job(store: Store, key: str = "key-1") -> dict[str, Any]:
    registered(store, "worker-1")
    store.submit(job("rollout"), key)
    claimed = store.claim("worker-1")
    assert claimed is not None
    return claimed


def project(
    store: Store, claimed: dict[str, Any], events: list[dict[str, Any]], *points: NativePoint
) -> None:
    """Index ``events`` and ``points`` for the attempt of ``claimed``, under one fixed scope."""
    Index(store).project(claimed["id"], claimed["lease_token"], {"slot": "mini-swe-agent"}, events, points)


def calls(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [(row["depth"], row["call_id"], row["call"], row["response"]) for row in rows]


def projection(row: dict[str, Any], *keys: str) -> dict[str, Any]:
    return {key: row[key] for key in keys}


def column(rows: list[dict[str, Any]], name: str) -> list[Any]:
    return [row[name] for row in rows]


def snapshot_check(present: bool) -> Callable[[dict[str, Any]], bool]:
    return lambda point: present


# Tool calls


def test_project_indexes_each_started_call_in_journal_order_with_its_first_response(run_store):
    claimed = claimed_job(run_store)

    project(
        run_store,
        claimed,
        [started("c1", "ls"), started("c2", "pwd"), finished("c1", "first"), finished("c1", "again")],
    )

    assert calls(Index(run_store).tools(claimed["active_attempt"])) == [
        (1, "c1", {"name": "bash", "arguments": {"command": "ls"}}, finished("c1", "first")),
        (2, "c2", {"name": "bash", "arguments": {"command": "pwd"}}, None),
    ]


def test_a_later_projection_adds_a_response_that_arrived_after_its_call(run_store):
    claimed = claimed_job(run_store)
    project(run_store, claimed, [started("c1", "ls")])

    project(run_store, claimed, [started("c1", "ls"), finished("c1", "answer")])

    assert column(Index(run_store).tools(claimed["active_attempt"]), "response") == [finished("c1", "answer")]


def test_a_call_that_changes_at_an_indexed_depth_is_a_conflict_and_the_first_call_stays(run_store):
    claimed = claimed_job(run_store)
    project(run_store, claimed, [started("c1", "ls")])

    with pytest.raises(Conflict, match="Indexed tool prefix changed"):
        project(run_store, claimed, [started("c1", "rm -rf build")])

    assert column(Index(run_store).tools(claimed["active_attempt"]), "call") == [
        {"name": "bash", "arguments": {"command": "ls"}}
    ]


def test_tools_returns_the_calls_deeper_than_a_cursor_up_to_a_limit(run_store):
    claimed = claimed_job(run_store)
    project(run_store, claimed, [started("c1", "ls"), started("c2", "pwd"), started("c3", "cat answer")])

    page = Index(run_store).tools(claimed["active_attempt"], after=1, limit=1)

    assert column(page, "call_id") == ["c2"]


def test_attempts_that_made_the_same_calls_under_one_scope_share_their_prefix_nodes(run_store):
    """The prefix of calls is a DAG keyed by the scope, so two scopes never share a node."""
    first, second = claimed_job(run_store, "first"), claimed_job(run_store, "second")
    other_scope = claimed_job(run_store, "other-scope")
    events = [started("c1", "ls"), started("c2", "pwd")]
    project(run_store, first, events)
    project(run_store, second, events)
    Index(run_store).project(other_scope["id"], other_scope["lease_token"], {"slot": "other"}, events)

    index = Index(run_store)
    nodes = column(index.tools(first["active_attempt"]), "prefix_node")
    assert column(index.tools(second["active_attempt"]), "prefix_node") == nodes
    assert set(column(index.tools(other_scope["active_attempt"]), "prefix_node")).isdisjoint(nodes)


def test_a_tool_call_needs_a_name_and_object_arguments(run_store):
    claimed = claimed_job(run_store)
    malformed = {"type": "tool.started", "call_id": "c1", "name": "bash", "args": ["ls"]}

    with pytest.raises(ValueError, match="Tool prefix requires tool name and JSON object arguments"):
        project(run_store, claimed, [malformed])


# Recovery points


def test_a_published_point_pairs_its_snapshot_and_history_with_the_prefix_node_at_its_depth(
    run_store, tmp_path
):
    claimed = claimed_job(run_store)
    native = native_history(tmp_path)

    project(
        run_store,
        claimed,
        [started("c1", "ls"), started("c2", "pwd")],
        NativePoint(1, 1, "snapshot-1", native),
    )

    index = Index(run_store)
    point = index.points(claimed["active_attempt"])[0]
    assert projection(
        point, "message_step", "tool_depth", "snapshot_id", "native", "valid", "prefix_node"
    ) == {
        "message_step": 1,
        "tool_depth": 1,
        "snapshot_id": "snapshot-1",
        "native": native,
        "valid": True,
        "prefix_node": index.tools(claimed["active_attempt"])[0]["prefix_node"],
    }


def test_points_are_listed_in_message_step_order(run_store, tmp_path):
    claimed = claimed_job(run_store)
    native = native_history(tmp_path)

    project(
        run_store,
        claimed,
        [started("c1", "ls"), started("c2", "pwd")],
        NativePoint(2, 2, "snapshot-2", native),
        NativePoint(1, 1, "snapshot-1", native),
    )

    assert column(Index(run_store).points(claimed["active_attempt"]), "snapshot_id") == [
        "snapshot-1",
        "snapshot-2",
    ]


def test_a_point_past_the_recorded_calls_is_refused(run_store, tmp_path):
    claimed = claimed_job(run_store)

    with pytest.raises(ValueError, match="Recovery point extends beyond recorded tools"):
        project(
            run_store,
            claimed,
            [started("c1", "ls")],
            NativePoint(1, 2, "snapshot-1", native_history(tmp_path)),
        )


def test_a_published_point_that_changes_is_a_conflict(run_store, tmp_path):
    claimed = claimed_job(run_store)
    events, native = [started("c1", "ls")], native_history(tmp_path)
    project(run_store, claimed, events, NativePoint(1, 1, "snapshot-1", native))

    with pytest.raises(Conflict, match="Recovery point changed after publication"):
        project(run_store, claimed, events, NativePoint(1, 1, "snapshot-other", native))


def test_a_point_that_a_later_projection_no_longer_names_becomes_invalid(run_store, tmp_path):
    claimed = claimed_job(run_store)
    events = [started("c1", "ls")]
    project(run_store, claimed, events, NativePoint(1, 1, "snapshot-1", native_history(tmp_path)))

    project(run_store, claimed, events)

    point = Index(run_store).points(claimed["active_attempt"])[0]
    assert (point["valid"], point["invalid_reason"]) == (False, "boundary_no_longer_proven")


def test_a_point_named_again_after_its_invalidation_is_valid_again(run_store, tmp_path):
    claimed = claimed_job(run_store)
    events, point = [started("c1", "ls")], NativePoint(1, 1, "snapshot-1", native_history(tmp_path))
    project(run_store, claimed, events, point)
    project(run_store, claimed, events)

    project(run_store, claimed, events, point)

    restored = Index(run_store).points(claimed["active_attempt"])[0]
    assert (restored["valid"], restored["invalid_reason"]) == (True, None)


def test_get_point_names_the_job_of_the_points_attempt(run_store, tmp_path):
    claimed = claimed_job(run_store)
    project(
        run_store, claimed, [started("c1", "ls")], NativePoint(1, 1, "snapshot-1", native_history(tmp_path))
    )
    published = Index(run_store).points(claimed["active_attempt"])[0]

    point = Index(run_store).get_point(published["id"])

    assert point == {**published, "job_id": claimed["id"]}


def test_get_point_of_an_unknown_point_raises_key_error(run_store):
    with pytest.raises(KeyError, match="no-such-point"):
        Index(run_store).get_point("no-such-point")


# Whether a branch can start from a point


def point_row(native: dict[str, Any], *, valid: bool = True) -> dict[str, Any]:
    """The fields of a stored recovery point that ``Index.valid`` reads."""
    return {"id": "point-1", "snapshot_id": "snapshot-1", "native": native, "valid": valid}


def test_a_valid_point_with_its_history_and_its_snapshot_can_start_a_branch(tmp_path):
    index = Index(UNUSED_STORE, snapshot_check(True))

    assert index.valid(point_row(native_history(tmp_path))) is True


def test_an_invalidated_point_cannot_start_a_branch(tmp_path):
    index = Index(UNUSED_STORE, snapshot_check(True))

    assert index.valid(point_row(native_history(tmp_path), valid=False)) is False


def test_a_point_whose_history_changed_cannot_start_a_branch(tmp_path):
    native = native_history(tmp_path)
    Path(native["path"]).write_text(Path(native["path"]).read_text().replace("answer", "other!"))

    assert Index(UNUSED_STORE, snapshot_check(True)).valid(point_row(native)) is False


def test_a_point_whose_snapshot_is_gone_cannot_start_a_branch(tmp_path):
    assert Index(UNUSED_STORE, snapshot_check(False)).valid(point_row(native_history(tmp_path))) is False


def test_without_a_snapshot_check_no_point_can_start_a_branch(tmp_path):
    assert Index(UNUSED_STORE).valid(point_row(native_history(tmp_path))) is False

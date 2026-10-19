"""What a branch request carries, where the Run Store refuses one, and where each attempt of it starts.

The request tests are small. The tests of the branch endpoint and of the index of recovery points need
the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.continuation import render_reasoning
from sprout.contract.mini_history import NativePoint
from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from sprout.worker.attempts import apply_branch_budget
from tests.contract.builders import assistant_turn, bash_call
from tests.support.runstore import registered

OBJECT_SHAPE = (
    "A continuation is an object with guidance and, for its mode, "
    "assistant_turn (and optionally assistant_turn_rendering) or hint"
)
#: The native history of a parent whose one closed turn called ``parent-call``.
PARENT_HISTORY = [
    {
        "type": "mini.session",
        "format": "sprout-mini-v1",
        "mini_version": "2.4.6",
        "session_id": "parent-session",
        "workspace": {"cwd": "/testbed", "repository_dir": None},
    },
    {"type": "mini.message", "message": {"role": "user", "content": "Fix the bug."}},
    {
        "type": "mini.message",
        "message": {"role": "assistant", "content": "", "tool_calls": [bash_call("ls", "parent-call")]},
    },
    {
        "type": "mini.message",
        "message": {"role": "tool", "tool_call_id": "parent-call", "content": "README\n"},
    },
    {"type": "mini.turn", "turn_id": "turn-1", "call_ids": ["parent-call"], "source": "actor"},
]
#: The journal events of the parent's one call.
PARENT_CALLS = [
    {"type": "tool.started", "call_id": "parent-call", "name": "shell", "args": {"command": "ls"}},
    {"type": "tool.finished", "call_id": "parent-call", "output": "README\n"},
]


def contract() -> dict[str, Any]:
    return {
        "message_export": True,
        "model_endpoint": "http://127.0.0.1:30000",
        "model": "fixture",
        "deadline_at": 1000.0,
        "max_turns": 40,
    }


def test_a_rollout_request_that_is_no_branch_keeps_its_body_without_branch_fields():
    """A job stored before these fields existed gets the same body and hash when it is submitted again."""
    body = JobSpec("rollout", {"prompt": "Fix the bug."}).validate()

    assert body == {
        "kind": "rollout",
        "spec": {"prompt": "Fix the bug."},
        "profile": "default",
        "context": {},
        "max_infra_retries": 2,
        "parent_point": None,
    }


@pytest.mark.parametrize(
    "continuation",
    [
        {"guidance": "none"},
        {
            "guidance": "assistant-turn",
            "assistant_turn": assistant_turn(),
            "assistant_turn_rendering": "reasoning",
        },
    ],
    ids=["none", "reasoning-rendering"],
)
def test_a_branch_request_carries_its_continuation_as_given(continuation):
    body = JobSpec(
        "rollout", {"prompt": "Fix the bug."}, parent_point="point-1", continuation=continuation
    ).validate()

    assert body["continuation"] == continuation


def test_a_branch_request_without_a_continuation_is_refused():
    with pytest.raises(ValueError, match="^%s$" % re.escape(OBJECT_SHAPE)):
        JobSpec("rollout", {"prompt": "Fix the bug."}, parent_point="point-1").validate()


def test_a_continuation_without_the_point_that_it_continues_is_refused():
    with pytest.raises(
        ValueError, match="^A continuation belongs to a rollout branch, with the parent_point it continues$"
    ):
        JobSpec("rollout", {"prompt": "Fix the bug."}, continuation={"guidance": "none"}).validate()


def test_a_branch_request_carries_its_own_budget():
    body = JobSpec(
        "rollout",
        {"prompt": "Fix the bug."},
        parent_point="point-1",
        continuation={"guidance": "none"},
        branch_budget={"max_turns": 12},
    ).validate()

    assert body["branch_budget"] == {"max_turns": 12}


@pytest.mark.parametrize(
    "budget, error",
    [
        pytest.param({}, "A branch budget sets max_turns and/or deadline_at", id="empty"),
        pytest.param({"turns": 3}, "A branch budget sets max_turns and/or deadline_at", id="unknown-key"),
        pytest.param({"max_turns": 0}, "branch budget max_turns must be a positive integer", id="no-turns"),
        pytest.param(
            {"max_turns": 2.5}, "branch budget max_turns must be a positive integer", id="half-turns"
        ),
        pytest.param(
            {"max_turns": True}, "branch budget max_turns must be a positive integer", id="boolean-turns"
        ),
        pytest.param(
            {"deadline_at": -5}, "branch budget deadline_at must be a finite positive timestamp", id="past"
        ),
        pytest.param(
            {"deadline_at": "soon"},
            "branch budget deadline_at must be a finite positive timestamp",
            id="text-time",
        ),
    ],
)
def test_a_branch_budget_that_sets_no_valid_limit_is_refused(budget, error):
    request = JobSpec(
        "rollout",
        {"prompt": "Fix the bug."},
        parent_point="point-1",
        continuation={"guidance": "none"},
        branch_budget=budget,
    )

    with pytest.raises(ValueError, match="^%s$" % re.escape(error)):
        request.validate()


def test_a_budget_without_a_branch_is_refused():
    with pytest.raises(ValueError, match="^A branch budget belongs to a branch$"):
        JobSpec("rollout", {"prompt": "Fix the bug."}, branch_budget={"max_turns": 3}).validate()


def test_a_branch_budget_lays_its_limits_over_the_inherited_contract_and_changes_nothing_else():
    """The inherited ``deadline_at`` is the parent group's, and can have passed. Turns count from zero."""
    effective = {
        "prompt": "Fix the bug.",
        "extra": {"rollout_contract": contract(), "mini": {"environment": {"cwd": "/w"}}},
    }

    budgeted = apply_branch_budget(effective, {"max_turns": 12, "deadline_at": 2000.0})

    assert budgeted == {
        "prompt": "Fix the bug.",
        "extra": {
            "rollout_contract": {**contract(), "max_turns": 12, "deadline_at": 2000.0},
            "mini": {"environment": {"cwd": "/w"}},
        },
    }


def test_a_branch_budget_leaves_the_inherited_spec_as_it_was():
    effective = {"prompt": "Fix the bug.", "extra": {"rollout_contract": contract()}}

    apply_branch_budget(effective, {"max_turns": 12})

    assert effective == {"prompt": "Fix the bug.", "extra": {"rollout_contract": contract()}}


def test_a_branch_without_a_budget_runs_the_inherited_spec():
    effective = {"prompt": "Fix the bug.", "extra": {"rollout_contract": contract()}}

    assert apply_branch_budget(effective, None) == {
        "prompt": "Fix the bug.",
        "extra": {"rollout_contract": contract()},
    }


def test_a_branch_budget_for_a_run_without_a_contract_is_refused():
    with pytest.raises(ValueError, match="^A branch budget bounds a rollout contract; this run has none$"):
        apply_branch_budget({"prompt": "Fix the bug.", "extra": {}}, {"max_turns": 3})


@dataclass(frozen=True)
class Parent:
    """A parent rollout job whose active attempt has published one recovery point."""

    job_id: str
    attempt_id: str
    point_id: str


def write_history(path: Path) -> dict[str, Any]:
    """The native reference of ``PARENT_HISTORY``, written to ``path``."""
    data = b"".join(json.dumps(entry).encode() + b"\n" for entry in PARENT_HISTORY)
    path.write_bytes(data)
    return {
        "slot": "mini-swe-agent",
        "path": str(path),
        "byte_length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
    }


def published_parent(
    store: Store, index: Index, native: dict[str, Any], spec: dict[str, Any], key: str
) -> Parent:
    """A parent job from ``spec``, claimed, with the one point of ``PARENT_HISTORY`` published."""
    registered(store, "fixture-worker")
    job = store.submit(JobSpec("rollout", spec), key)
    claim = store.claim("fixture-worker")
    index.project(
        job["id"], claim["lease_token"], {}, PARENT_CALLS, [NativePoint(1, 1, "snapshot-1", native)]
    )
    [point] = index.points(claim["active_attempt"])
    return Parent(job_id=job["id"], attempt_id=claim["active_attempt"], point_id=point["id"])


@pytest.fixture
def index(run_store: Store) -> Index:
    """The index of ``run_store``, for which each snapshot exists."""
    return Index(run_store, lambda point: True)


@pytest.fixture
def parent(run_store: Store, index: Index, tmp_path: Path) -> Parent:
    return published_parent(
        run_store, index, write_history(tmp_path / "parent.jsonl"), {"prompt": "Fix the bug."}, "parent"
    )


@pytest.fixture
def api(run_store: Store, index: Index) -> Iterator[Any]:
    """A client of the Run Store API over ``run_store`` and ``index``, with the bearer token."""
    from fastapi.testclient import TestClient

    from sprout.runstore.api import create_app

    with TestClient(create_app(run_store, "fixture-token", index=index)) as client:
        client.headers["Authorization"] = "Bearer fixture-token"
        yield client


def branch(api: Any, job_id: str, body: dict[str, Any], key: str) -> Any:
    return api.post("/v1/jobs/%s/branch" % job_id, json=body, headers={"Idempotency-Key": key})


def test_a_branch_is_queued_with_the_spec_of_its_parent_and_its_continuation(api, run_store, parent):
    continuation = {"guidance": "assistant-turn", "assistant_turn": assistant_turn(identifier="fresh-call")}

    accepted = branch(
        api, parent.job_id, {"point_id": parent.point_id, "continuation": continuation}, "fresh"
    )

    request = run_store.get(accepted.json()["id"])["request"]
    assert (accepted.status_code, request["spec"], request["parent_point"], request["continuation"]) == (
        202,
        {"prompt": "Fix the bug."},
        parent.point_id,
        continuation,
    )


def test_a_branch_keeps_the_rendering_of_its_turn(api, run_store, parent):
    continuation = {
        "guidance": "assistant-turn",
        "assistant_turn": assistant_turn(identifier="fresh-call"),
        "assistant_turn_rendering": "reasoning",
    }

    accepted = branch(
        api, parent.job_id, {"point_id": parent.point_id, "continuation": continuation}, "rendered"
    )

    assert run_store.get(accepted.json()["id"])["request"]["continuation"] == continuation


@pytest.mark.parametrize(
    "body, detail",
    [
        pytest.param(
            {
                "continuation": {
                    "guidance": "assistant-turn",
                    "assistant_turn": render_reasoning(assistant_turn(identifier="fresh-call")),
                    "assistant_turn_rendering": "reasoning",
                }
            },
            "assistant_turn requires exactly role, content and tool_calls",
            id="turn-already-rendered",
        ),
        pytest.param(
            {
                "continuation": {
                    "guidance": "assistant-turn",
                    "assistant_turn": assistant_turn(identifier="parent-call"),
                }
            },
            "assistant_turn tool call ids must be non-empty and unique across history",
            id="call-id-of-the-history",
        ),
        pytest.param(
            {"continuation": {"guidance": "none"}, "overrides": {"prompt": "Try a different fix."}},
            "A branch can change only budget_usd, model, timeout_s; "
            "what the actor is given next is its continuation",
            id="new-prompt",
        ),
        pytest.param({}, OBJECT_SHAPE, id="no-continuation"),
        pytest.param(
            {"continuation": {"guidance": "none"}, "prompt": "Try a different fix."},
            "A branch names point_id and continuation, and optionally overrides and budget",
            id="unknown-field",
        ),
        pytest.param(
            {"continuation": {"guidance": "none"}, "budget": {"max_turns": 5}},
            "A branch budget bounds a rollout contract; the parent run has none",
            id="budget-without-contract",
        ),
        pytest.param(
            {"continuation": {"guidance": "none"}, "budget": {"max_turns": 0}},
            "branch budget max_turns must be a positive integer",
            id="invalid-budget",
        ),
    ],
)
def test_a_branch_that_cannot_continue_its_point_is_refused_with_the_reason(
    api, run_store, parent, body, detail
):
    refused = branch(api, parent.job_id, {"point_id": parent.point_id, **body}, "refused")

    assert (refused.status_code, refused.json()) == (422, {"detail": detail})


def test_a_branch_budget_is_stored_beside_the_inherited_contract_that_it_bounds(
    api, run_store, index, tmp_path, parent
):
    """The worker lays the budget over the contract when an attempt starts (``apply_branch_budget``)."""
    spec = {"prompt": "Fix the bug.", "extra": {"rollout_contract": contract()}}
    bounded = published_parent(run_store, index, write_history(tmp_path / "bounded.jsonl"), spec, "bounded")
    body = {"point_id": bounded.point_id, "continuation": {"guidance": "none"}, "budget": {"max_turns": 5}}

    accepted = branch(api, bounded.job_id, body, "budgeted")

    request = run_store.get(accepted.json()["id"])["request"]
    assert (request["branch_budget"], request["spec"]["extra"]["rollout_contract"]) == (
        {"max_turns": 5},
        contract(),
    )


def test_a_published_point_that_a_later_projection_no_longer_names_is_no_longer_valid(
    run_store, index, parent
):
    claim_token = run_store.get(parent.job_id)["lease_token"]

    index.project(parent.job_id, claim_token, {}, PARENT_CALLS, [])

    assert [(point["valid"], point["invalid_reason"]) for point in index.points(parent.attempt_id)] == [
        (False, "boundary_no_longer_proven")
    ]


def test_a_point_that_each_projection_names_stays_valid(run_store, index, tmp_path, parent):
    claim_token = run_store.get(parent.job_id)["lease_token"]
    native = write_history(tmp_path / "parent.jsonl")

    index.project(parent.job_id, claim_token, {}, PARENT_CALLS, [NativePoint(1, 1, "snapshot-1", native)])

    assert [(point["valid"], point["invalid_reason"]) for point in index.points(parent.attempt_id)] == [
        (True, None)
    ]


def test_a_point_whose_history_changed_is_not_available_for_a_branch(run_store, index, tmp_path, parent):
    (tmp_path / "parent.jsonl").write_bytes(
        (tmp_path / "parent.jsonl").read_bytes().replace(b"README", b"readme")
    )

    [point] = index.points(parent.attempt_id)

    assert index.valid(point) is False

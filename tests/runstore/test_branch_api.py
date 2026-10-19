"""A branch through the HTTP API: a new rollout from a recovery point of its parent, with its continuation.

These tests use the PostgreSQL test database (``run_store``), and skip without it. Each test calls the API
in process, through FastAPI's ``TestClient``. A stub says whether the snapshot of a point still exists.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import NativePoint
from sprout.runstore.api import create_app
from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from tests.runstore.builders import bash_call, native_history, rollout_spec
from tests.support.runstore import registered

TOKEN = "fixture-token"
#: A parent rollout with a rollout contract, the limits that a branch budget lays its own over.
CONTRACT_SPEC = rollout_spec(
    model="policy", extra={"rollout_contract": {"max_turns": 10, "deadline_at": 1700000000}}
)


def client_for(app: Any) -> Any:
    """A ``TestClient`` of ``app`` that sends the bearer token."""
    testclient = pytest.importorskip(
        "fastapi.testclient", reason="install the runstore extra: pip install -e '.[runstore]'"
    )
    return testclient.TestClient(app, headers={"Authorization": "Bearer " + TOKEN})


@pytest.fixture
def index(run_store: Store) -> Index:
    """The index of ``run_store``, on which every snapshot exists."""
    return Index(run_store, lambda point: True)


@pytest.fixture
def api(run_store: Store, index: Index) -> Any:
    return client_for(create_app(run_store, TOKEN, index=index))


def parent_with_point(
    store: Store,
    directory: Path,
    *,
    spec: dict[str, Any] | None = None,
    key: str = "parent",
    slot: str = "mini-swe-agent",
) -> tuple[str, str]:
    """The ids of a parent rollout and of its recovery point after one closed turn.

    The point's history is in ``directory``. Its last call has the id ``call-1``.
    """
    registered(store, "worker-1")
    store.submit(JobSpec("rollout", spec or rollout_spec(model="policy")), key)
    claimed = store.claim("worker-1")
    native = {**native_history(directory), "slot": slot}
    events = [{"type": "tool.started", "call_id": "call-1", "name": "bash", "args": {"command": "ls"}}]
    Index(store).project(
        claimed["id"], claimed["lease_token"], {}, events, [NativePoint(1, 1, "snapshot-1", native)]
    )
    return claimed["id"], Index(store).points(claimed["active_attempt"])[0]["id"]


def branch(client: Any, job_id: str, request: dict[str, Any], key: str = "branch-1") -> Any:
    return client.post("/v1/jobs/%s/branch" % job_id, json=request, headers={"Idempotency-Key": key})


def answer(response: Any) -> tuple[int, Any]:
    return response.status_code, response.json()


def assistant_turn(call_id: str) -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": "Fix the parser.",
        "tool_calls": [bash_call(call_id, "sed -i s/a/b/ parser.py")],
    }


def test_a_branch_is_a_queued_rollout_of_the_parents_request_from_the_point(run_store, api, tmp_path):
    parent_id, point_id = parent_with_point(run_store, tmp_path)
    continuation = {"guidance": "user-hint", "hint": "Check the parser."}

    response = branch(
        api, parent_id, {"point_id": point_id, "continuation": continuation, "overrides": {"model": "other"}}
    )

    assert (response.status_code, response.json()["state"], response.json()["request"]) == (
        202,
        "queued",
        {
            "kind": "rollout",
            "spec": rollout_spec(model="other"),
            "profile": "default",
            "context": {},
            "max_infra_retries": 2,
            "parent_point": point_id,
            "continuation": continuation,
        },
    )


def test_a_repeated_branch_under_one_key_returns_the_same_job(run_store, api, tmp_path):
    parent_id, point_id = parent_with_point(run_store, tmp_path)
    request = {"point_id": point_id, "continuation": {"guidance": "none"}}
    first = branch(api, parent_id, request)

    again = branch(api, parent_id, request)

    assert again.json()["id"] == first.json()["id"]


@pytest.mark.parametrize(
    "overrides", [{"prompt": "Another task."}, {"sandbox_image": "another:1"}], ids=["prompt", "environment"]
)
def test_a_branch_may_override_only_the_model_the_timeout_and_the_budget(run_store, api, tmp_path, overrides):
    parent_id, point_id = parent_with_point(run_store, tmp_path)

    response = branch(
        api, parent_id, {"point_id": point_id, "continuation": {"guidance": "none"}, "overrides": overrides}
    )

    assert answer(response) == (
        422,
        {
            "detail": "A branch can change only budget_usd, model, timeout_s; "
            "what the actor is given next is its continuation"
        },
    )


def test_a_branch_body_names_only_its_point_continuation_overrides_and_budget(run_store, api, tmp_path):
    parent_id, point_id = parent_with_point(run_store, tmp_path)

    response = branch(
        api, parent_id, {"point_id": point_id, "continuation": {"guidance": "none"}, "priority": 1}
    )

    assert answer(response) == (
        422,
        {"detail": "A branch names point_id and continuation, and optionally overrides and budget"},
    )


def test_a_branch_from_a_point_of_another_job_answers_422(run_store, api, tmp_path):
    parent_id, _ = parent_with_point(run_store, tmp_path / "parent", key="parent")
    _, other_point = parent_with_point(run_store, tmp_path / "other", key="other")

    response = branch(api, parent_id, {"point_id": other_point, "continuation": {"guidance": "none"}})

    assert answer(response) == (422, {"detail": "Branch point does not belong to this rollout"})


@pytest.mark.parametrize(
    "index_of",
    [lambda store: Index(store, lambda point: False), lambda store: None],
    ids=["snapshot-gone", "no-snapshot-check"],
)
def test_a_branch_from_a_point_that_is_no_longer_available_answers_409(run_store, tmp_path, index_of):
    client = client_for(create_app(run_store, TOKEN, index=index_of(run_store)))
    parent_id, point_id = parent_with_point(run_store, tmp_path)

    response = branch(client, parent_id, {"point_id": point_id, "continuation": {"guidance": "none"}})

    assert answer(response) == (
        409,
        {"detail": "Recovery point no longer has both native prefix and snapshot"},
    )


def test_a_branch_from_a_point_of_another_history_slot_answers_422(run_store, api, tmp_path):
    parent_id, point_id = parent_with_point(run_store, tmp_path, slot="other-slot")

    response = branch(api, parent_id, {"point_id": point_id, "continuation": {"guidance": "none"}})

    assert answer(response) == (422, {"detail": "Continuation must use its native history slot"})


@pytest.mark.parametrize(
    "continuation,error",
    [
        ({"guidance": "rewrite"}, "Branch guidance must be assistant-turn, user-hint or none"),
        ({"guidance": "user-hint"}, "A user-hint branch needs a non-empty hint"),
        (
            None,
            "A continuation is an object with guidance and, for its mode, "
            "assistant_turn (and optionally assistant_turn_rendering) or hint",
        ),
    ],
    ids=["unknown-guidance", "hint-without-text", "no-continuation"],
)
def test_a_branch_with_an_invalid_continuation_answers_422(run_store, api, tmp_path, continuation, error):
    parent_id, point_id = parent_with_point(run_store, tmp_path)

    response = branch(api, parent_id, {"point_id": point_id, "continuation": continuation})

    assert answer(response) == (422, {"detail": error})


@pytest.mark.parametrize("call_id,status", [("call-1", 422), ("call-2", 202)], ids=["used-id", "fresh-id"])
def test_an_inserted_turn_must_call_with_ids_that_the_history_has_not_used(
    run_store, api, tmp_path, call_id, status
):
    """The history up to the point already holds the call ``call-1``."""
    parent_id, point_id = parent_with_point(run_store, tmp_path)
    continuation = {"guidance": "assistant-turn", "assistant_turn": assistant_turn(call_id)}

    response = branch(api, parent_id, {"point_id": point_id, "continuation": continuation})

    assert response.status_code == status, response.text


def test_a_branch_budget_is_stored_with_the_branch(run_store, api, tmp_path):
    parent_id, point_id = parent_with_point(run_store, tmp_path, spec=CONTRACT_SPEC)

    response = branch(
        api,
        parent_id,
        {"point_id": point_id, "continuation": {"guidance": "none"}, "budget": {"max_turns": 3}},
    )

    assert response.json()["request"]["branch_budget"] == {"max_turns": 3}


def test_a_branch_budget_for_a_parent_without_a_rollout_contract_answers_422(run_store, api, tmp_path):
    parent_id, point_id = parent_with_point(run_store, tmp_path)

    response = branch(
        api,
        parent_id,
        {"point_id": point_id, "continuation": {"guidance": "none"}, "budget": {"max_turns": 3}},
    )

    assert answer(response) == (
        422,
        {"detail": "A branch budget bounds a rollout contract; the parent run has none"},
    )


def test_an_invalid_branch_budget_answers_422(run_store, api, tmp_path):
    parent_id, point_id = parent_with_point(run_store, tmp_path, spec=CONTRACT_SPEC)

    response = branch(
        api, parent_id, {"point_id": point_id, "continuation": {"guidance": "none"}, "budget": {}}
    )

    assert answer(response) == (422, {"detail": "A branch budget sets max_turns and/or deadline_at"})


def test_a_submission_with_a_parent_point_must_keep_the_parents_spec(run_store, api, tmp_path):
    _, point_id = parent_with_point(run_store, tmp_path)
    request = {
        "kind": "rollout",
        "spec": rollout_spec(model="policy", prompt="Another task."),
        "parent_point": point_id,
        "continuation": {"guidance": "none"},
    }

    response = api.post("/v1/jobs", json=request, headers={"Idempotency-Key": "branch-1"})

    assert answer(response) == (
        422,
        {"detail": "Continuation must retain its source environment, slot and tool configuration"},
    )

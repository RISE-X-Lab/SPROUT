"""The Run Store ``Client`` against the API served on a local port.

These tests use the PostgreSQL test database (``run_store``), and skip without it. The API runs in
uvicorn on 127.0.0.1 (``tests.support.http.serving``).
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.contract.mini_history import NativePoint
from sprout.runstore.api import create_app
from sprout.runstore.client import Client
from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from tests.runstore.builders import job, native_history, rollout_spec
from tests.support.http import serving
from tests.support.runstore import registered

TOKEN = "fixture-token"


@dataclass(frozen=True)
class Served:
    """A client of the API at ``url`` over ``store``, and the index whose snapshots all exist."""

    url: str
    client: Client
    store: Store
    index: Index


@pytest.fixture
def served(run_store: Store) -> Iterator[Served]:
    pytest.importorskip("uvicorn", reason="install the runstore extra: pip install -e '.[runstore]'")
    index = Index(run_store, lambda point: True)
    with serving(create_app(run_store, TOKEN, index=index)) as url:
        client = Client(url, TOKEN)
        yield Served(url=url, client=client, store=run_store, index=index)
        client.close()


def finished(store: Store, job_id: str, result: dict[str, Any]) -> None:
    registered(store, "worker-1")
    claimed = store.claim("worker-1")
    store.finish(job_id, claimed["lease_token"], result)


def published_point(store: Store, index: Index, directory: Path, job_id: str) -> str:
    """The id of a recovery point that the claimed job ``job_id`` published after one call."""
    registered(store, "worker-1")
    claimed = store.claim("worker-1")
    events = [{"type": "tool.started", "call_id": "call-1", "name": "bash", "args": {"command": "ls"}}]
    index.project(
        job_id,
        claimed["lease_token"],
        {},
        events,
        [NativePoint(1, 1, "snapshot-1", native_history(directory))],
    )
    return str(index.points(claimed["active_attempt"])[0]["id"])


def availability(points: list[dict[str, Any]]) -> list[tuple[str, bool]]:
    return [(point["id"], point["available"]) for point in points]


def test_submit_returns_the_job_id_and_the_same_id_for_a_repeat(served):
    job_id = served.client.submit(job("rollout"), "key-1")

    again = served.client.submit(job("rollout"), "key-1")

    assert (again, served.client.get(job_id)["state"]) == (job_id, "queued")


def test_submit_of_another_job_under_a_used_key_raises_a_409(served):
    served.client.submit(job("rollout"), "key-1")

    with pytest.raises(httpx.HTTPStatusError) as refused:
        served.client.submit(JobSpec("rollout", rollout_spec(prompt="Another task.")), "key-1")

    assert refused.value.response.status_code == 409


def test_a_client_with_the_wrong_token_is_refused(served):
    intruder = Client(served.url, "wrong-token")

    with pytest.raises(httpx.HTTPStatusError) as refused:
        intruder.get_result("job-1")
    intruder.close()

    assert refused.value.response.status_code == 401


def test_wait_returns_the_result_of_a_finished_job(served):
    job_id = served.client.submit(job("rollout"), "key-1")
    finished(served.store, job_id, {"final_text": "done"})

    result = served.client.wait(job_id, timeout_s=10, interval_s=0.01)

    assert result == {
        "job_id": job_id,
        "state": "succeeded",
        "result": {"final_text": "done"},
        "error": None,
        "ready": True,
    }


def test_a_wait_that_times_out_leaves_the_job_queued(served):
    job_id = served.client.submit(job("rollout"), "key-1")

    with pytest.raises(TimeoutError, match="Client wait timed out; job remains queued/running"):
        served.client.wait(job_id, timeout_s=0)

    assert served.client.get(job_id)["state"] == "queued"


def test_recovery_points_are_those_of_the_active_attempt_with_their_availability(served, tmp_path):
    job_id = served.client.submit(job("rollout"), "key-1")
    point_id = published_point(served.store, served.index, tmp_path, job_id)

    points = served.client.recovery_points(job_id)

    assert availability(points) == [(point_id, True)]


def test_branch_submits_the_overrides_and_the_budget_and_returns_the_branch_id(served, tmp_path):
    contract = {"max_turns": 10, "deadline_at": 1700000000}
    job_id = served.client.submit(
        JobSpec("rollout", rollout_spec(extra={"rollout_contract": contract})), "key-1"
    )
    point_id = published_point(served.store, served.index, tmp_path, job_id)

    branch_id = served.client.branch(
        job_id,
        point_id,
        idempotency_key="branch-1",
        continuation={"guidance": "none"},
        budget={"max_turns": 3},
        model="other",
    )

    request = served.client.get(branch_id)["request"]
    assert (request["spec"]["model"], request["branch_budget"], request["parent_point"]) == (
        "other",
        {"max_turns": 3},
        point_id,
    )

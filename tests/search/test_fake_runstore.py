"""The fake Run Store answers a ``RunStoreClient`` the way the real Run Store API answers it.

Each test runs on the ``FakeRunStore``, as a small test, and on the real API over a Run Store on the
PostgreSQL test database, as a medium test that skips without the database. A test plays the worker on
each side with the same steps, and checks only the fields that the coordinator reads.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.contract.mini_history import NativePoint
from sprout.runstore.api import create_app
from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from sprout.search.client import RunStoreClient
from tests.search.documents import RUN
from tests.search.fake_runstore import FakeRunStore, complete_tools
from tests.support.http import serving
from tests.support.runstore import registered

ROLLOUT = JobSpec.from_dict({"kind": "rollout", **RUN})
#: A snapshot that no longer exists: a branch cannot start from its point.
GONE = "GONE"


class FakeSide:
    """The worker's steps on a ``FakeRunStore``."""

    def __init__(self, fake: FakeRunStore) -> None:
        self.client = fake.client
        self._fake = fake

    def claim(self, job_id: str) -> None:
        self._fake.run(job_id)

    def record(self, job_id: str, *, depth: int, snapshots: dict[int, str]) -> None:
        points = [
            {
                "id": "%s-step%d" % (job_id, tool_depth),
                "tool_depth": tool_depth,
                "message_step": tool_depth - 1,
                "snapshot_id": snapshot,
                "available": snapshot != GONE,
            }
            for tool_depth, snapshot in snapshots.items()
        ]
        self._fake.record(job_id, tools=complete_tools(depth), points=points)

    def finish(self, job_id: str, result: dict[str, Any], *, state: str = "succeeded") -> None:
        self._fake.finish(job_id, result, state=state)


class RealSide:
    """The worker's steps on a real Run Store, whose real API serves at ``url``."""

    def __init__(self, store: Store, *, url: str, native: Path) -> None:
        registered(store, "worker")
        self.client = RunStoreClient(url, "fixture")
        self._store = store
        self._native = native
        self._tokens: dict[str, str] = {}

    def claim(self, job_id: str) -> None:
        claimed = self._store.claim("worker")
        assert claimed is not None and claimed["id"] == job_id, "the job is the only queued one"
        self._tokens[job_id] = claimed["lease_token"]

    def record(self, job_id: str, *, depth: int, snapshots: dict[int, str]) -> None:
        events = []
        for n in range(1, depth + 1):
            events.append(
                {"type": "tool.started", "call_id": "call-%d" % n, "name": "bash", "args": {"n": n}}
            )
            events.append({"type": "tool.finished", "call_id": "call-%d" % n, "output": "ok"})
        reference = {
            "path": str(self._native),
            "byte_length": self._native.stat().st_size,
            "sha256": hashlib.sha256(self._native.read_bytes()).hexdigest(),
            "slot": "mini-swe-agent",
        }
        points = [
            NativePoint(tool_depth - 1, tool_depth, snapshot, reference)
            for tool_depth, snapshot in snapshots.items()
        ]
        Index(self._store).project(job_id, self._tokens[job_id], {}, events, points)

    def finish(self, job_id: str, result: dict[str, Any], *, state: str = "succeeded") -> None:
        self._store.finish(job_id, self._tokens[job_id], result, state=state)


@pytest.fixture(params=["fake", "real"])
def side(request: pytest.FixtureRequest, tmp_path: Path) -> Iterator[FakeSide | RealSide]:
    """The fake Run Store, or a real one of its own on the PostgreSQL test database."""
    if request.param == "fake":
        yield FakeSide(request.getfixturevalue("fake_run_store"))
        return
    pytest.importorskip("uvicorn", reason="install the runstore extra: pip install -e '.[runstore]'")
    store = request.getfixturevalue("run_store")
    native = tmp_path / "native.jsonl"
    native.write_bytes(b'{"type": "mini.turn"}\n')
    # Each snapshot exists but GONE.
    app = create_app(store, "fixture", index=Index(store, lambda point: point["snapshot_id"] != GONE))
    with serving(app) as url:
        real = RealSide(store, url=url, native=native)
        yield real
        real.client.close()


def finished_rollout(
    side: FakeSide | RealSide, *, snapshots: dict[int, str]
) -> tuple[str, list[dict[str, Any]]]:
    """A rollout that ran three tool calls with a recovery point after each depth in ``snapshots``, and
    succeeded. Returns its id and its recovery points."""
    job_id = side.client.submit(ROLLOUT, "rollout")
    side.claim(job_id)
    side.record(job_id, depth=3, snapshots=snapshots)
    side.finish(job_id, {"status": "completed"})
    attempt = side.client.get(job_id)["active_attempt"]
    return job_id, side.client.points(job_id, attempt_id=attempt)


def depths(page: list[dict[str, Any]]) -> list[tuple[int, bool]]:
    """The depth of each tool call of ``page``, and whether the call has its response."""
    return [(row["depth"], row["response"] is not None) for row in page]


def boundaries(points: list[dict[str, Any]]) -> list[tuple[int, int, str, bool]]:
    """The tool depth, the message step, the snapshot and the availability of each recovery point."""
    return [
        (point["tool_depth"], point["message_step"], point["snapshot_id"], point["available"])
        for point in points
    ]


def test_a_new_job_is_queued_under_its_key_and_has_no_attempt(side):
    job_id = side.client.submit(ROLLOUT, "rollout")

    job = side.client.get(job_id)

    assert (
        job["id"],
        job["state"],
        job["active_attempt"],
        job["result"],
        job["error"],
        job["idempotency_key"],
    ) == (
        job_id,
        "queued",
        None,
        None,
        None,
        "rollout",
    )


def test_the_same_request_under_the_same_key_returns_the_first_job(side):
    first = side.client.submit(ROLLOUT, "rollout")

    assert side.client.submit(ROLLOUT, "rollout") == first


def test_another_request_under_a_used_key_is_refused_with_409(side):
    side.client.submit(ROLLOUT, "rollout")
    other = JobSpec.from_dict({"kind": "rollout", **RUN, "context": {"another": "request"}})

    with pytest.raises(httpx.HTTPStatusError) as refused:
        side.client.submit(other, "rollout")

    assert refused.value.response.status_code == 409


def test_a_finished_job_has_its_state_its_result_its_error_and_its_attempt(side):
    job_id = side.client.submit(ROLLOUT, "rollout")
    side.claim(job_id)
    side.finish(job_id, {"status": "error", "error": "the agent crashed"}, state="failed")

    job = side.client.get(job_id)

    assert (job["state"], job["result"], job["error"], job["active_attempt"] is not None) == (
        "failed",
        {"status": "error", "error": "the agent crashed"},
        "the agent crashed",
        True,
    )


def test_a_stop_cancels_a_queued_job(side):
    job_id = side.client.submit(ROLLOUT, "rollout")

    stopped = side.client.stop(job_id)

    assert (stopped["state"], side.client.get(job_id)["state"]) == ("cancelled", "cancelled")


def test_a_stop_of_a_running_job_leaves_it_running_and_records_the_request(side):
    job_id = side.client.submit(ROLLOUT, "rollout")
    side.claim(job_id)

    stopped = side.client.stop(job_id)

    assert (stopped["state"], stopped["stop_requested_at"] is not None) == ("running", True)


def test_the_tool_calls_after_a_depth_come_in_depth_order_with_their_responses(side):
    job_id = side.client.submit(ROLLOUT, "rollout")
    side.claim(job_id)
    side.record(job_id, depth=3, snapshots={})
    attempt = side.client.get(job_id)["active_attempt"]

    page = side.client.tools(job_id, attempt_id=attempt, after=1)

    assert depths(page) == [(2, True), (3, True)]


def test_each_recovery_point_says_whether_a_branch_can_start_from_it(side):
    _, points = finished_rollout(side, snapshots={2: "AB", 3: GONE})

    assert boundaries(points) == [(2, 1, "AB", True), (3, 2, GONE, False)]


def test_a_branch_from_a_point_of_the_rollout_is_a_new_queued_rollout_and_a_repeat_returns_it(side):
    job_id, points = finished_rollout(side, snapshots={2: "AB"})

    branch = side.client.branch(
        job_id,
        points[0]["id"],
        idempotency_key="branch",
        continuation={"guidance": "none"},
        budget={"max_turns": 3},
    )
    again = side.client.branch(
        job_id,
        points[0]["id"],
        idempotency_key="branch",
        continuation={"guidance": "none"},
        budget={"max_turns": 3},
    )

    job = side.client.get(branch)
    assert (again, branch != job_id, job["kind"], job["state"]) == (branch, True, "rollout", "queued")


def test_a_branch_from_a_point_of_another_job_is_refused_with_422(side):
    _, points = finished_rollout(side, snapshots={2: "AB"})
    other = side.client.submit(ROLLOUT, "other")

    with pytest.raises(httpx.HTTPStatusError) as refused:
        side.client.branch(
            other, points[0]["id"], idempotency_key="branch", continuation={"guidance": "none"}
        )

    assert refused.value.response.status_code == 422


def test_an_unknown_job_is_not_found(side):
    with pytest.raises(httpx.HTTPStatusError) as refused:
        side.client.get("0123456789abcdef0123456789abcdef")

    assert refused.value.response.status_code == 404

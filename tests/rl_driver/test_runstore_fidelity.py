"""The driver does the same on ``FakeRunStore`` as on the real Run Store: the fake's fidelity tests.

Each test runs twice. On the fake, it is a small test. On the real Run Store, it is a medium test: the
Run Store API on a local port, over a Store in a fresh schema of the test database (``run_store``). That
side skips without ``SPROUT_RUNSTORE_TEST_DSN``. The real API gets a stub index of recovery points,
because a real one needs the native history of a run. Each point is also a row of the Run Store. A
``Backend`` plays the worker on each side.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.runstore.api import create_app
from sprout.runstore.store import Store
from sprout.search.client import RunStoreClient
from tests.rl_driver.fake_runstore import FakeRunStore
from tests.rl_driver.miles import GRADE_TEMPLATE
from tests.support.http import serving

TOKEN = "runstore-token"
IMAGE = "docker.io/swerebenchv2/task-1:base"


class FakeBackend:
    """The fake Run Store and its worker."""

    def __init__(self) -> None:
        self._fake = FakeRunStore()
        self.client = self._fake.client()

    def claim(self, kind: str) -> str:
        job_id = next(
            job_id for job_id in self._fake.ids(kind) if self._fake.jobs[job_id]["state"] == "queued"
        )
        self._fake.start(job_id)
        return job_id

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        self._fake.finish(job_id, result=result)

    def publish_point(self, job_id: str, point: dict[str, Any]) -> None:
        self._fake.publish(job_id, tools=[], points=[{**point, "available": True}])

    def job(self, job_id: str) -> dict[str, Any]:
        return self._fake.jobs[job_id]

    def job_ids(self) -> list[str]:
        return list(self._fake.jobs)

    def lose_next_reply(self) -> None:
        self._fake.lose_next_reply()


class StubIndex:
    """The recovery points of the real API: each one valid, with the native history of mini."""

    def __init__(self) -> None:
        self._points: list[dict[str, Any]] = []

    def add(self, point: dict[str, Any]) -> None:
        self._points.append(point)

    def tools(self, attempt_id: str, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        return []

    def points(self, attempt_id: str) -> list[dict[str, Any]]:
        return [point for point in self._points if point["attempt_id"] == attempt_id]

    def get_point(self, point_id: str) -> dict[str, Any]:
        return next(point for point in self._points if point["id"] == point_id)

    def valid(self, point: dict[str, Any]) -> bool:
        return True


class LosingReplies(httpx.HTTPTransport):
    """An HTTP transport that loses the reply to the next submission, after the server handled it."""

    def __init__(self) -> None:
        super().__init__()
        self.armed = False

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        response = super().handle_request(request)
        if self.armed and request.method == "POST" and request.url.path.endswith(("/jobs", "/branch")):
            self.armed = False
            response.close()
            raise httpx.ReadError("The reply was lost after the Run Store accepted the job", request=request)
        return response


class RealBackend:
    """The Run Store API at ``url`` over ``store``, and a worker that claims and finishes its jobs."""

    def __init__(self, *, store: Store, index: StubIndex, url: str) -> None:
        # The worker grades the dataset and the parser of the Miles tasks' grade template.
        grade = (GRADE_TEMPLATE["spec"]["dataset_sha256"], GRADE_TEMPLATE["spec"]["grader_revision"])
        store.register_worker("fidelity-worker", host="host-1", session="fidelity", grades=[grade])
        self._store = store
        self._index = index
        self._tokens: dict[str, str] = {}
        self._transport = LosingReplies()
        self.client = RunStoreClient(url, TOKEN)
        self.client.http.close()
        self.client.http = httpx.Client(
            base_url=url, headers=self.client.http.headers, transport=self._transport
        )

    def claim(self, kind: str) -> str:
        job = self._store.claim("fidelity-worker", kinds=(kind,))
        assert job is not None, f"no {kind} job is queued"
        self._tokens[job["id"]] = job["lease_token"]
        return str(job["id"])

    def finish(self, job_id: str, result: dict[str, Any]) -> None:
        self._store.finish(job_id, self._tokens[job_id], result)

    def publish_point(self, job_id: str, point: dict[str, Any]) -> None:
        """Publish ``point`` in the stub index, and as a row of the Run Store, for its job.

        The Run Store gives a branch the affinity key of the job of its point, from the row.
        """
        attempt = self._store.get(job_id)["active_attempt"]
        native = {"slot": "mini-swe-agent"}
        self._index.add({**point, "attempt_id": attempt, "job_id": job_id, "native": native})
        with self._store.transaction() as cursor:
            cursor.execute(
                "INSERT INTO rs_prefix_nodes(node,scope,parent,depth,call) VALUES(%s,'',%s,%s,'{}')",
                (point["id"], point["id"], point["tool_depth"]),
            )
            cursor.execute(
                """INSERT INTO rs_recoveries
                (id,attempt_id,message_step,tool_depth,prefix_node,snapshot_id,native)
                VALUES(%s,%s,%s,%s,%s,%s,%s::jsonb)""",
                (
                    point["id"],
                    attempt,
                    point["message_step"],
                    point["tool_depth"],
                    point["id"],
                    point["snapshot_id"],
                    json.dumps(native),
                ),
            )

    def job(self, job_id: str) -> dict[str, Any]:
        return self._store.get(job_id)

    def job_ids(self) -> list[str]:
        return [job["id"] for job in reversed(self._store.list_jobs())]

    def lose_next_reply(self) -> None:
        self._transport.armed = True


@pytest.fixture(params=["fake", "real"])
def backend(request: pytest.FixtureRequest) -> Iterator[FakeBackend | RealBackend]:
    """The fake Run Store, or the real one on the test database. The real one skips without it."""
    if request.param == "fake":
        yield FakeBackend()
        return
    store = request.getfixturevalue("run_store")
    index = StubIndex()
    with serving(create_app(store, TOKEN, index=index)) as url:
        real = RealBackend(store=store, index=index, url=url)
        yield real
        real.client.close()


def run_sample(slot: str, *, graded: bool = False) -> dict[str, Any]:
    spec = {
        "prompt": "Fix the parser",
        "sandbox_image": IMAGE,
        "extra": {"rollout_contract": {"max_turns": 10}},
    }
    sample: dict[str, Any] = {"sample_slot_id": slot, "run": {"profile": "mini", "spec": spec}}
    grade = {**GRADE_TEMPLATE, "spec": {**GRADE_TEMPLATE["spec"], "image": IMAGE}}
    return {**sample, "grade": grade} if graded else sample


def group(group_id: str, *samples: dict[str, Any]) -> dict[str, Any]:
    return {"rollout_job_id": group_id, "prompt_group_id": "prompt-1", "samples": list(samples)}


def new_driver(backend: FakeBackend | RealBackend, directory: Path) -> Driver:
    return Driver(backend.client, Ledger(directory / "ledger.sqlite3"))


def test_a_graded_group_runs_its_actor_then_grades_the_actors_final_snapshot(backend, tmp_path):
    driver = new_driver(backend, tmp_path)
    driver.submit(group("group-1", run_sample("slot-0", graded=True)))
    driver.tick()
    actor = backend.claim("rollout")
    backend.finish(actor, {"status": "completed", "final_snapshot_id": "snap-final"})
    driver.tick()
    grade = backend.claim("grade")
    backend.finish(grade, {"status": "completed", "resolved": False})

    driver.tick()

    view = driver.get("group-1")
    record = backend.job(grade)["request"]["context"]["rl_driver"]
    assert (
        view["status"],
        view["samples"][0]["grade"]["result"],
        backend.job(grade)["request"]["spec"]["snapshot_id"],
    ) == (
        "completed",
        {"status": "completed", "resolved": False},
        "snap-final",
    )
    assert (record["actor_job_id"], record["actor_attempt_id"]) == (
        actor,
        backend.job(actor)["active_attempt"],
    )


def test_cancelling_a_group_cancels_its_queued_job_and_asks_its_running_job_to_stop(backend, tmp_path):
    driver = new_driver(backend, tmp_path)
    driver.submit(group("group-1", run_sample("slot-0"), run_sample("slot-1")))
    driver.tick()
    running = backend.claim("rollout")
    driver.release("group-1")

    driver.tick()

    (queued,) = set(backend.job_ids()) - {running}
    assert (backend.job(running)["state"], backend.job(running)["stop_requested_at"] is not None) == (
        "running",
        True,
    )
    assert (backend.job(queued)["state"], driver.get("group-1")["status"]) == ("cancelled", "cancelling")


def test_a_restarted_driver_retries_a_submission_whose_reply_was_lost_onto_the_same_job(backend, tmp_path):
    new_driver(backend, tmp_path).submit(group("group-1", run_sample("slot-0")))
    backend.lose_next_reply()
    new_driver(backend, tmp_path).tick()

    restarted = new_driver(backend, tmp_path)
    restarted.tick()

    (job_id,) = backend.job_ids()
    assert restarted.get("group-1")["samples"][0]["actor"]["job_id"] == job_id


def test_a_lost_submission_of_a_cancelled_group_is_reconciled_onto_its_job_and_cancelled(backend, tmp_path):
    driver = new_driver(backend, tmp_path)
    driver.submit(group("group-1", run_sample("slot-0")))
    backend.lose_next_reply()
    driver.tick()
    driver.release("group-1")
    driver.tick()
    (job_id,) = backend.job_ids()

    driver.reconcile_submission("group-1", "slot-0", "actor", job_id)
    driver.tick()

    assert (driver.get("group-1")["status"], backend.job(job_id)["state"], backend.job_ids()) == (
        "cancelled",
        "cancelled",
        [job_id],
    )


def test_a_branch_sample_runs_as_a_branch_of_its_parents_point_with_its_own_budget(backend, tmp_path):
    driver = new_driver(backend, tmp_path)
    driver.submit(group("parent-group", run_sample("slot-0")))
    driver.tick()
    parent = backend.claim("rollout")
    backend.finish(parent, {"status": "completed"})
    backend.publish_point(
        parent, {"id": "point-2", "tool_depth": 2, "message_step": 1, "snapshot_id": "snap-2"}
    )
    continuation = {"guidance": "user-hint", "hint": "Run the tests first"}
    branch = {
        "job_id": parent,
        "point_id": "point-2",
        "continuation": continuation,
        "budget": {"max_turns": 7},
    }

    driver.submit(group("branch-group", {"sample_slot_id": "slot-0", "branch": branch}))
    driver.tick()

    request = backend.job(driver.get("branch-group")["samples"][0]["actor"]["job_id"])["request"]
    assert (request["parent_point"], request["continuation"], request["branch_budget"], request["spec"]) == (
        "point-2",
        continuation,
        {"max_turns": 7},
        backend.job(parent)["request"]["spec"],
    )

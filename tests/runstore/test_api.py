"""The Run Store's HTTP API: authentication, submission with profile defaults, reads, cancel and stop.

The authentication tests are small. The others use the PostgreSQL test database (``run_store``), and skip
without it. Each test calls the API in process, through FastAPI's ``TestClient``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any

import pytest

from sprout.contract.mini_history import NativePoint
from sprout.runstore.api import create_app
from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from tests.runstore.builders import SPECS, harbor_grade_spec, native_history, rollout_spec, stored_job
from tests.support.paths import REPO
from tests.support.runstore import registered

TOKEN = "fixture-token"
AUTH = {"Authorization": "Bearer " + TOKEN}
#: A store that the API never connects to, because each request is refused first.
UNUSED_STORE = Store("dbname=never-connected")


def client_for(app: Any) -> Any:
    """A ``TestClient`` of ``app`` that sends the bearer token."""
    testclient = pytest.importorskip(
        "fastapi.testclient", reason="install the runstore extra: pip install -e '.[runstore]'"
    )
    return testclient.TestClient(app, headers=AUTH)


@pytest.fixture
def api(run_store: Store) -> Any:
    """The API over ``run_store``, without profiles."""
    return client_for(create_app(run_store, TOKEN))


def body(kind: str = "rollout", **fields: Any) -> dict[str, Any]:
    """The JSON body of a valid request for a job of ``kind``, with ``fields`` over the request's fields."""
    return {"kind": kind, "spec": SPECS[kind](), **fields}


def submit(client: Any, request: dict[str, Any], key: str = "key-1") -> Any:
    return client.post("/v1/jobs", json=request, headers={"Idempotency-Key": key})


def answer(response: Any) -> tuple[int, Any]:
    return response.status_code, response.json()


def column(rows: list[dict[str, Any]], name: str) -> list[Any]:
    return [row[name] for row in rows]


def finish_claimed(store: Store, state: str) -> None:
    claimed = store.claim("worker-1")
    store.finish(claimed["id"], claimed["lease_token"], {"status": "fixture"}, state=state)


#: The step that takes a queued job to each state.
TO_STATE: dict[str, Callable[[Store, str], object]] = {
    "queued": lambda store, job_id: None,
    "running": lambda store, job_id: store.claim("worker-1"),
    "succeeded": lambda store, job_id: finish_claimed(store, "succeeded"),
    "failed": lambda store, job_id: finish_claimed(store, "failed"),
    "cancelled": lambda store, job_id: store.cancel_queued(job_id),
}


# Authentication

ROUTES = [
    ("POST", "/v1/jobs"),
    ("GET", "/v1/jobs"),
    ("GET", "/v1/jobs/job-1"),
    ("GET", "/v1/jobs/job-1/result"),
    ("GET", "/v1/jobs/job-1/attempts"),
    ("GET", "/v1/jobs/job-1/events"),
    ("GET", "/v1/jobs/job-1/tools"),
    ("GET", "/v1/jobs/job-1/recovery-points"),
    ("POST", "/v1/jobs/job-1/branch"),
    ("POST", "/v1/jobs/job-1/cancel"),
    ("POST", "/v1/jobs/job-1/stop"),
]
BAD_CREDENTIALS = {
    "no-header": {"Authorization": ""},
    "wrong-token": {"Authorization": "Bearer wrong-token"},
    "no-scheme": {"Authorization": TOKEN},
}


@pytest.mark.parametrize("credentials", BAD_CREDENTIALS.values(), ids=BAD_CREDENTIALS.keys())
@pytest.mark.parametrize("method,path", ROUTES, ids=["-".join(route) for route in ROUTES])
def test_every_route_refuses_a_request_without_the_bearer_token(method, path, credentials):
    client = client_for(create_app(UNUSED_STORE, TOKEN))

    response = client.request(method, path, json={}, headers=credentials)

    assert answer(response) == (401, {"detail": "Invalid control-plane token"})


def test_the_api_needs_a_token():
    with pytest.raises(ValueError, match="A control-plane bearer token is required"):
        create_app(UNUSED_STORE, "")


# Submission


def test_a_submission_is_accepted_with_its_queued_job(api):
    response = submit(api, body("rollout"))

    assert (response.status_code, response.json()["state"], response.json()["request"]["spec"]) == (
        202,
        "queued",
        rollout_spec(),
    )


def test_a_repeated_submission_under_one_key_returns_the_same_job(api):
    first = submit(api, body("rollout"))

    again = submit(api, body("rollout"))

    assert (again.status_code, again.json()["id"]) == (202, first.json()["id"])


def test_another_request_under_a_used_key_answers_409(api):
    submit(api, body("rollout"))

    response = submit(api, body("rollout", spec=rollout_spec(prompt="Another task.")))

    assert answer(response) == (409, {"detail": "Idempotency key already names a different request"})


def test_a_request_that_the_specs_refuse_answers_422_with_the_reason(api):
    response = submit(api, body("rollout", spec=rollout_spec(prompt=" ")))

    assert answer(response) == (422, {"detail": "RunSpec requires a nonempty prompt"})


@pytest.mark.parametrize(
    "request_body",
    [body("rollout", priority=1), body("grade", spec=harbor_grade_spec(priority=1))],
    ids=["job-spec-field", "grade-spec-field"],
)
def test_a_request_with_an_unknown_field_answers_422(api, request_body):
    assert answer(submit(api, request_body)) == (422, {"detail": "Invalid JobSpec fields"})


def test_a_submission_without_an_idempotency_key_answers_422(api):
    response = api.post("/v1/jobs", json=body("rollout"))

    assert answer(response) == (422, {"detail": "An idempotency key of 1..256 characters is required"})


# Profiles

#: A profile whose sections each give a different timeout, so that each kind shows which one it took.
PROFILE = {
    "python": "/path/to/python",
    "run_defaults": {"timeout_s": 100},
    "grade_defaults": {"timeout_s": 200},
    "review_defaults": {"timeout_s": 300},
}


@pytest.fixture
def profiled_api(run_store: Store) -> Any:
    """The API over ``run_store`` with the one profile ``mini``, which is ``PROFILE``."""
    return client_for(create_app(run_store, TOKEN, profiles={"mini": PROFILE}))


def test_with_profiles_a_submission_must_name_one_of_them(profiled_api):
    response = submit(profiled_api, body("rollout", profile="missing"))

    assert answer(response) == (422, {"detail": "Unknown worker profile"})


@pytest.mark.parametrize("kind,timeout_s", [("rollout", 100), ("grade", 200), ("review", 300)])
def test_a_job_takes_the_defaults_of_its_kinds_profile_section(run_store, profiled_api, kind, timeout_s):
    """The spec keeps the defaults of the submission, so a later edit of the profile does not change them."""
    job = stored_job(run_store, kind)
    spec = job.spec

    response = submit(profiled_api, {"kind": kind, "spec": spec, "profile": "mini", "affinity": job.affinity})

    assert response.json()["request"]["spec"] == {**spec, "timeout_s": timeout_s}


def test_a_value_in_the_request_wins_over_the_profiles_default(profiled_api):
    request = submit(profiled_api, body("rollout", profile="mini", spec=rollout_spec(timeout_s=50))).json()[
        "request"
    ]

    assert request["spec"] == rollout_spec(timeout_s=50)


#: A valid request, in a store, for each job kind that a profile of the example config runs.
EXAMPLE_JOBS: list[tuple[str, Callable[[Store], JobSpec]]] = [
    ("mini", lambda store: JobSpec("rollout", rollout_spec(sandbox_image="registry.example/task:1"))),
    ("mini-microvm", lambda store: JobSpec("rollout", rollout_spec(sandbox_image="registry.example/task:1"))),
    ("grade", lambda store: JobSpec("grade", harbor_grade_spec(), affinity="group-1")),
    ("grade-microvm", lambda store: JobSpec("grade", harbor_grade_spec(), affinity="group-1")),
    ("review", lambda store: stored_job(store, "review")),
]


@pytest.mark.parametrize("profile,request_of", EXAMPLE_JOBS, ids=[row[0] for row in EXAMPLE_JOBS])
def test_each_profile_of_the_example_config_makes_a_valid_job_of_its_kind(run_store, profile, request_of):
    """A default that names a field the spec no longer has would refuse each job of that profile."""
    profiles = json.loads((REPO / "resources" / "runstore" / "config.example.json").read_text())["profiles"]
    client = client_for(create_app(run_store, TOKEN, profiles=profiles))
    request = request_of(run_store)

    response = submit(
        client, {"kind": request.kind, "spec": request.spec, "profile": profile, "affinity": request.affinity}
    )

    assert response.status_code == 202, response.text


# Reads


def test_jobs_are_listed_newest_first_and_filtered_by_state(run_store, api):
    registered(run_store, "worker-1")
    first = submit(api, body("rollout"), "key-1").json()["id"]
    second = submit(api, body("rollout"), "key-2").json()["id"]
    third = submit(api, body("rollout"), "key-3").json()["id"]
    run_store.claim("worker-1")

    every, queued = api.get("/v1/jobs").json(), api.get("/v1/jobs", params={"state": "queued"}).json()

    assert (column(every, "id"), column(queued, "id")) == ([third, second, first], [third, second])


@pytest.mark.parametrize(
    "state,ready",
    [
        ("queued", False),
        ("running", False),
        ("succeeded", True),
        ("failed", True),
        ("cancelled", True),
    ],
)
def test_a_result_is_ready_once_its_job_succeeded_failed_or_was_cancelled(run_store, api, state, ready):
    registered(run_store, "worker-1")
    job_id = submit(api, body("rollout")).json()["id"]
    TO_STATE[state](run_store, job_id)

    result = api.get("/v1/jobs/%s/result" % job_id).json()

    assert (result["state"], result["ready"]) == (state, ready)


def test_reading_a_result_does_not_consume_it(run_store, api):
    registered(run_store, "worker-1")
    job_id = submit(api, body("rollout")).json()["id"]
    claimed = run_store.claim("worker-1")
    run_store.finish(job_id, claimed["lease_token"], {"final_text": "durable"})

    first, second = (
        api.get("/v1/jobs/%s/result" % job_id).json(),
        api.get("/v1/jobs/%s/result" % job_id).json(),
    )

    expected = {
        "job_id": job_id,
        "state": "succeeded",
        "result": {"final_text": "durable"},
        "error": None,
        "ready": True,
    }
    assert (first, second) == (expected, expected)


def test_a_job_is_written_with_the_iso_form_of_each_timestamp(run_store, api):
    """A response model would write a UTC time as ...Z. The clients read the ISO form, ...+00:00."""
    job_id = submit(api, body("rollout")).json()["id"]

    written = api.get("/v1/jobs/%s" % job_id).json()

    assert written["created_at"] == run_store.get(job_id)["created_at"].isoformat()


def test_the_attempts_of_a_job_are_listed_by_number(run_store, api):
    registered(run_store, "worker-1")
    job_id = submit(api, body("rollout")).json()["id"]
    run_store.claim("worker-1")

    attempts = api.get("/v1/jobs/%s/attempts" % job_id).json()

    assert column(attempts, "number") == [1]


def test_events_are_those_of_the_active_attempt_after_a_cursor(run_store, api):
    registered(run_store, "worker-1")
    job_id = submit(api, body("rollout")).json()["id"]
    claimed = run_store.claim("worker-1")
    run_store.append_events(
        job_id, claimed["lease_token"], [{"seq": 1, "type": "a"}, {"seq": 2, "type": "b"}]
    )

    events = api.get("/v1/jobs/%s/events" % job_id, params={"after": 1}).json()

    assert events == [{"seq": 2, "type": "b"}]


def test_events_of_an_attempt_of_another_job_answer_404(run_store, api):
    registered(run_store, "worker-1")
    job_id = submit(api, body("rollout"), "key-1").json()["id"]
    run_store.claim("worker-1")
    submit(api, body("rollout"), "key-2")
    other = run_store.claim("worker-1")

    response = api.get("/v1/jobs/%s/events" % job_id, params={"attempt_id": other["active_attempt"]})

    assert answer(response) == (404, {"detail": "Reference not found"})


@pytest.mark.parametrize("present", [True, False], ids=["snapshot-present", "snapshot-gone"])
def test_tools_and_recovery_points_are_those_of_the_active_attempt(run_store, tmp_path, present):
    registered(run_store, "worker-1")
    index = Index(run_store, lambda point: present)
    client = client_for(create_app(run_store, TOKEN, index=index))
    job_id = submit(client, body("rollout")).json()["id"]
    claimed = run_store.claim("worker-1")
    events = [{"type": "tool.started", "call_id": "c1", "name": "bash", "args": {"command": "ls"}}]
    index.project(
        job_id,
        claimed["lease_token"],
        {},
        events,
        [NativePoint(1, 1, "snapshot-1", native_history(tmp_path))],
    )

    tools = client.get("/v1/jobs/%s/tools" % job_id).json()
    points = client.get("/v1/jobs/%s/recovery-points" % job_id).json()

    assert (tools, points) == (
        index.tools(claimed["active_attempt"]),
        [{**index.points(claimed["active_attempt"])[0], "available": present}],
    )


@pytest.mark.parametrize(
    "path",
    ["", "/result", "/attempts", "/events", "/tools", "/recovery-points"],
    ids=["job", "result", "attempts", "events", "tools", "recovery-points"],
)
def test_a_read_of_an_unknown_job_answers_404(api, path):
    assert answer(api.get("/v1/jobs/no-such-job" + path)) == (404, {"detail": "Reference not found"})


# Cancel and stop


def test_cancel_cancels_a_queued_job(api):
    job_id = submit(api, body("rollout")).json()["id"]

    response = api.post("/v1/jobs/%s/cancel" % job_id)

    assert (response.status_code, response.json()["state"]) == (200, "cancelled")


def test_cancel_of_a_running_job_answers_409(run_store, api):
    registered(run_store, "worker-1")
    job_id = submit(api, body("rollout")).json()["id"]
    run_store.claim("worker-1")

    response = api.post("/v1/jobs/%s/cancel" % job_id)

    assert answer(response) == (
        409,
        {"detail": "Only a queued job can be cancelled; a running one needs a stop request"},
    )


def test_stop_cancels_a_queued_job(api):
    job_id = submit(api, body("rollout")).json()["id"]

    assert api.post("/v1/jobs/%s/stop" % job_id).json()["state"] == "cancelled"


def test_stop_marks_a_running_job_for_its_worker(run_store, api):
    registered(run_store, "worker-1")
    job_id = submit(api, body("rollout")).json()["id"]
    run_store.claim("worker-1")

    stopped = api.post("/v1/jobs/%s/stop" % job_id).json()

    assert (stopped["state"], stopped["stop_requested_at"]) == (
        "running",
        run_store.get(job_id)["stop_requested_at"].isoformat(),
    )


def test_stop_of_an_unknown_job_answers_404(api):
    assert answer(api.post("/v1/jobs/no-such-job/stop")) == (404, {"detail": "Unknown job"})

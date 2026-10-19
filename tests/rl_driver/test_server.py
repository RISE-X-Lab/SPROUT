"""The driver's HTTP API: its token, its routes and status codes, its hold on the ledger and its poller.

docs/RL_DRIVER.md ("HTTP interface") lists the routes and the error codes. Each test serves the app
in process with ``TestClient``. The Run Store is ``FakeRunStore``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Conflict, Ledger
from sprout.rl_driver.messages import MessageAdapter, internal_id
from sprout.rl_driver.server import create_app
from tests.rl_driver.fake_runstore import FakeRunStore
from tests.rl_driver.miles import adapter_config, miles_request
from tests.support.waiting import wait_until

AUTHORIZED = {"Authorization": "Bearer driver-token"}


def run_group(*, prompt: str = "Fix the parser") -> dict[str, Any]:
    spec = {"prompt": prompt, "sandbox_image": "docker.io/swerebenchv2/task-1:base"}
    return {
        "rollout_job_id": "group-1",
        "prompt_group_id": "prompt-1",
        "samples": [{"sample_slot_id": "slot-0", "run": {"profile": "mini", "spec": spec}}],
    }


def new_driver(fake: FakeRunStore, directory: Path) -> Driver:
    return Driver(fake.client(), Ledger(directory / "ledger.sqlite3"))


def served(driver: Driver, *, miles: bool = False, **options: Any) -> TestClient:
    """A client of the driver's app, with the token ``driver-token``, and no poller unless ``options`` ask."""
    adapter = MessageAdapter(driver, adapter_config()) if miles else None
    app = create_app(driver, "driver-token", miles=adapter, **{"background": False, **options})
    return TestClient(app, headers=AUTHORIZED)


@pytest.mark.parametrize(
    "headers, status",
    [
        ({}, 401),
        ({"Authorization": "Bearer another-token"}, 401),
        ({"Authorization": "driver-token"}, 401),
        (AUTHORIZED, 200),
    ],
)
def test_each_request_needs_the_driver_token_as_a_bearer_token(fake_run_store, tmp_path, headers, status):
    app = create_app(new_driver(fake_run_store, tmp_path), "driver-token", background=False)

    with TestClient(app) as http:
        response = http.get("/health", headers=headers)

    assert response.status_code == status


def test_an_app_without_a_token_serves_a_request_without_authorization(fake_run_store, tmp_path):
    app = create_app(new_driver(fake_run_store, tmp_path), None, background=False)

    with TestClient(app) as http:
        response = http.get("/health")

    assert (response.status_code, response.json()) == (200, {"role": "driver", "miles_configured": False})


@pytest.mark.parametrize(
    "token, interval, message",
    [
        ("", 1, "Use a nonempty token or None for trusted local access"),
        ("driver-token", 0, "poll_interval_s must be positive"),
        ("driver-token", -1, "poll_interval_s must be positive"),
    ],
)
def test_an_empty_token_or_a_poll_interval_that_is_not_positive_is_refused(
    fake_run_store, tmp_path, token, interval, message
):
    driver = new_driver(fake_run_store, tmp_path)

    with pytest.raises(ValueError, match=message):
        create_app(driver, token, poll_interval_s=interval)


def test_the_health_route_says_that_the_miles_section_is_configured(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path), miles=True) as http:
        response = http.get("/health")

    assert response.json() == {"role": "driver", "miles_configured": True}


@pytest.mark.parametrize(
    "method, path",
    [
        ("POST", "/rollout-groups"),
        ("GET", "/rollout-groups/rollout-1"),
        ("DELETE", "/rollout-groups/rollout-1"),
        ("GET", "/miles-executions/rollout-1"),
    ],
)
def test_each_miles_route_answers_503_without_the_miles_section(fake_run_store, tmp_path, method, path):
    with served(new_driver(fake_run_store, tmp_path)) as http:
        response = http.request(method, path, json=miles_request())

    assert response.status_code == 503


def test_a_miles_group_whose_id_holds_a_slash_is_read_at_its_path(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path), miles=True) as http:
        http.post("/rollout-groups", json=miles_request(rollout_job_id="run-7/step-3/group-1"))

        response = http.get("/rollout-groups/run-7/step-3/group-1")

    assert (response.status_code, response.json()["rollout_job_id"]) == (200, "run-7/step-3/group-1")


def test_the_miles_execution_route_answers_the_execution_view_behind_a_miles_group(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    with served(driver, miles=True) as http:
        http.post("/rollout-groups", json=miles_request(rollout_job_id="rollout-1"))

        response = http.get("/miles-executions/rollout-1")

    assert response.json() == driver.get(internal_id("rollout-1"))


def test_submitting_an_execution_group_answers_202_with_its_id_and_status(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path)) as http:
        response = http.post("/execution-groups", json=run_group())

    assert (response.status_code, response.json()) == (202, {"rollout_job_id": "group-1", "status": "queued"})


@pytest.mark.parametrize("path", ["/execution-groups/group-1", "/execution-groups/group-1/result"])
def test_an_execution_group_is_read_at_its_path_and_at_its_result_path(fake_run_store, tmp_path, path):
    driver = new_driver(fake_run_store, tmp_path)
    with served(driver) as http:
        http.post("/execution-groups", json=run_group())

        response = http.get(path)

    assert (response.status_code, response.json()) == (200, driver.get("group-1"))


def test_deleting_an_active_group_answers_202_and_says_that_it_is_cancelling(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path)) as http:
        http.post("/execution-groups", json=run_group())

        response = http.delete("/execution-groups/group-1")

    assert (response.status_code, response.json()) == (
        202,
        {"rollout_job_id": "group-1", "status": "cancelling", "acknowledged_at": None},
    )


def test_deleting_a_finished_group_answers_200_with_the_time_of_the_acknowledgement(
    fake_run_store, tmp_path, clock
):
    driver = new_driver(fake_run_store, tmp_path)
    with served(driver) as http:
        http.post("/execution-groups", json=run_group())
        driver.tick()
        fake_run_store.finish("job-0", result={"status": "completed"})
        driver.tick()

        response = http.delete("/execution-groups/group-1")

    assert (response.status_code, response.json()) == (
        200,
        {"rollout_job_id": "group-1", "status": "completed", "acknowledged_at": clock.now},
    )


def test_a_changed_request_under_a_stored_id_answers_409(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path)) as http:
        http.post("/execution-groups", json=run_group(prompt="Fix the parser"))

        response = http.post("/execution-groups", json=run_group(prompt="Fix the lexer"))

    assert (response.status_code, response.json()) == (
        409,
        {"detail": "rollout_job_id already names a different request"},
    )


@pytest.mark.parametrize(
    "method, path",
    [
        ("GET", "/execution-groups/group-9"),
        ("DELETE", "/execution-groups/group-9"),
        ("GET", "/rollout-groups/rollout-9"),
        ("GET", "/miles-executions/rollout-9"),
    ],
)
def test_an_unknown_group_answers_404(fake_run_store, tmp_path, method, path):
    with served(new_driver(fake_run_store, tmp_path), miles=True) as http:
        response = http.request(method, path)

    assert (response.status_code, response.json()) == (
        404,
        {"detail": "Group, sample or execution reference not found"},
    )


@pytest.mark.parametrize(
    "path, body, status",
    [
        ("/execution-groups", {"rollout_job_id": "group-1"}, 422),
        ("/rollout-groups", miles_request(max_turns=0), 400),
    ],
)
def test_an_invalid_request_answers_400_on_the_miles_routes_and_422_on_the_others(
    fake_run_store, tmp_path, path, body, status
):
    with served(new_driver(fake_run_store, tmp_path), miles=True) as http:
        response = http.post(path, json=body)

    assert response.status_code == status


@pytest.mark.parametrize(
    "body, detail",
    [
        ({"phase": "actor"}, "Reconciliation requires phase and job_id"),
        ({"job_id": "job-0"}, "Reconciliation requires phase and job_id"),
        ({"phase": "actor", "job_id": "job-0", "force": True}, "Reconciliation requires phase and job_id"),
        ({"phase": "actor", "job_id": "job/0"}, "job_id must be a URL-safe identifier of 1..200 characters"),
    ],
)
def test_a_reconciliation_needs_a_phase_and_a_url_safe_job_id_alone(fake_run_store, tmp_path, body, detail):
    with served(new_driver(fake_run_store, tmp_path)) as http:
        http.post("/execution-groups", json=run_group())

        response = http.post("/execution-groups/group-1/samples/slot-0/reconcile-submission", json=body)

    assert (response.status_code, response.json()) == (422, {"detail": detail})


def test_a_failed_run_store_call_answers_502_and_says_that_nothing_needs_resubmitting(
    fake_run_store, tmp_path
):
    driver = new_driver(fake_run_store, tmp_path)
    with served(driver) as http:
        http.post("/execution-groups", json=run_group())
        fake_run_store.lose_next_reply()
        driver.tick()
        fake_run_store.go_down()

        response = http.post(
            "/execution-groups/group-1/samples/slot-0/reconcile-submission",
            json={"phase": "actor", "job_id": "job-0"},
        )

    assert (response.status_code, response.json()) == (
        502,
        {"detail": "Run Store query failed; retry without resubmitting execution"},
    )


def test_a_second_app_cannot_start_on_a_ledger_that_a_running_app_holds(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path)):
        with pytest.raises(Conflict, match="Another driver already owns this ledger"):
            with served(new_driver(fake_run_store, tmp_path)):
                pytest.fail("a second app served the ledger")


def test_the_poller_submits_a_new_groups_jobs_without_waiting_for_its_interval(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path), background=True, poll_interval_s=3600) as http:
        http.post("/execution-groups", json=run_group())

        wait_until(
            lambda: fake_run_store.ids("rollout") == ["job-0"], what="the poller's submission of job-0"
        )


def test_the_poller_cancels_a_released_groups_job_without_waiting_for_its_interval(fake_run_store, tmp_path):
    with served(new_driver(fake_run_store, tmp_path), background=True, poll_interval_s=3600) as http:
        http.post("/execution-groups", json=run_group())
        wait_until(
            lambda: fake_run_store.ids("rollout") == ["job-0"], what="the poller's submission of job-0"
        )

        http.delete("/execution-groups/group-1")

        wait_until(
            lambda: fake_run_store.jobs["job-0"]["state"] == "cancelled", what="the poller's cancel of job-0"
        )

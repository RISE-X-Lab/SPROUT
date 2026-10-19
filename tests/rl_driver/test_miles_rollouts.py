"""A Miles request without a search: its rollout and grade jobs, and the result that Miles trains on.

The Run Store is ``FakeRunStore``, and the test plays its worker (``tests/rl_driver/miles.py``). A test
that takes the ``clock`` fixture runs at ``NOW``, so each deadline is exact. docs/RL_DRIVER.md
("The Miles protocol", "Results") describes the request and the result.
"""

from __future__ import annotations

import json
import re

import pytest
from fastapi.testclient import TestClient

from sprout.contract.messages import HINT_START, mark_hint
from sprout.rl_driver.ledger import Conflict
from sprout.rl_driver.messages import MessageAdapter, internal_id
from sprout.rl_driver.server import create_app
from tests.rl_driver.conftest import NOW
from tests.rl_driver.fake_runstore import FakeRunStore
from tests.rl_driver.miles import (
    GRADE_TEMPLATE,
    MESSAGES,
    REVIEW_CONFIG,
    adapter_config,
    data_lost,
    exported,
    failed,
    finish_open,
    miles_adapter,
    miles_request,
    run_samples,
    search_request,
    slots,
    succeeded,
    verdict,
)
from tests.support.paths import REPO

EXAMPLES = REPO / "examples" / "rl_driver"
IMAGE = "docker.io/swerebenchv2/task-1:base"


def test_a_miles_id_maps_to_the_same_internal_id_on_every_driver():
    """The ledger keeps a Miles group under this id: another id would lose every stored group."""
    assert (
        internal_id("rollout-1") == "miles-dda26b22f1d382fb5f9642cca3462d1b36454e8ea0602325fd8db11e5f2bac5d"
    )


def test_the_shipped_request_example_returns_the_shipped_result_example(fake_run_store, tmp_path, clock):
    """Miles's importer is tested against the two files: the adapter must return exactly the result."""
    request = json.loads((EXAMPLES / "request.example.json").read_text())
    expected = json.loads((EXAMPLES / "result.example.json").read_text())
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    export = exported(final_snapshot_id="ABC", training_messages=expected["trajectories"][0]["messages"])
    with TestClient(create_app(adapter.driver, None, background=False, miles=adapter)) as http:
        submitted = http.post("/rollout-groups", json=request)
        run_samples(
            adapter,
            fake_run_store,
            actors=[succeeded(export)],
            grades=[succeeded(verdict(True, touched_test_paths=[]))],
        )

        result = http.get("/rollout-groups/" + request["rollout_job_id"])

    assert (submitted.status_code, result.json()) == (202, expected)


def test_each_slot_runs_as_one_rollout_job_under_the_requests_rollout_contract(
    fake_run_store, tmp_path, clock
):
    """The worker reads the rollout contract: the endpoint, the sampling, the turn limit and the deadline."""
    config = adapter_config(
        api_key_env="MODEL_API_KEY", run_defaults={"slot": "mini-swe-agent", "extra": {"x": 1}}
    )
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3", config)
    adapter.submit(miles_request(sample_slots=slots(2), max_samples=2, sampling_params={"top_k": -1}))

    adapter.driver.tick()

    contract = {
        "message_export": True,
        "model_endpoint": "http://model.test:30000",
        "model": "policy",
        "sampling_params": {"top_k": -1},
        "deadline_at": NOW + 900.0,
        "max_turns": 10,
        "api_key_env": "MODEL_API_KEY",
    }
    assert fake_run_store.request("job-0") == {
        "kind": "rollout",
        "profile": "mini",
        "spec": {
            "slot": "mini-swe-agent",
            "extra": {"x": 1, "rollout_contract": contract},
            "prompt": "Fix the parser",
            "model": "policy",
            "sandbox_image": IMAGE,
            "sandbox_resources": {"cpu": 2, "memory_mb": 12288},
            "timeout_s": 900.0,
        },
        "context": {
            "task_id": "task-1",
            "image": IMAGE,
            "sample_slot_id": "slot-0",
            "rl_driver": {
                "rollout_job_id": internal_id("rollout-1"),
                "prompt_group_id": internal_id("prompt-1"),
                "sample_slot_id": internal_id("slot-0"),
                "context": {"deadline_at": NOW + 900.0},
            },
        },
        "max_infra_retries": 0,
        "parent_point": None,
        "affinity": internal_id("rollout-1"),
    }
    assert fake_run_store.request("job-1")["spec"] == fake_run_store.request("job-0")["spec"], (
        "each trajectory gets the whole turn limit and the same deadline"
    )


def test_only_the_first_max_samples_slots_run(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request(sample_slots=slots(3), max_samples=2))

    adapter.driver.tick()

    contexts = [fake_run_store.request(job)["context"] for job in fake_run_store.ids("rollout")]
    assert [context["sample_slot_id"] for context in contexts] == ["slot-0", "slot-1"]


@pytest.mark.parametrize(
    "image, runs_from",
    [
        ("prime/primeintellect/task-1:base", "docker.io/swerebenchv2/task-1:base"),
        ("registry.test/team/task-1:base", "registry.test/team/task-1:base"),
    ],
)
def test_a_prime_image_runs_and_is_graded_from_the_public_registry(
    fake_run_store, tmp_path, image, runs_from
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request(image=image))
    adapter.driver.tick()
    finish_open(fake_run_store, "rollout", [succeeded(exported())])

    adapter.driver.tick()

    actor, grade = fake_run_store.request("job-0"), fake_run_store.request("job-1")
    assert (actor["spec"]["sandbox_image"], grade["spec"]["image"], actor["context"]["image"]) == (
        runs_from,
        runs_from,
        image,
    )


@pytest.mark.parametrize(
    "run_defaults, timeout",
    [
        ({"slot": "mini-swe-agent"}, 900.0),
        ({"slot": "mini-swe-agent", "timeout_s": 300}, 300),
        ({"slot": "mini-swe-agent", "timeout_s": 1200}, 900.0),
    ],
)
def test_a_rollout_times_out_at_the_shorter_of_its_run_defaults_and_the_wall_time(
    fake_run_store, tmp_path, run_defaults, timeout
):
    adapter = miles_adapter(
        fake_run_store, tmp_path / "ledger.sqlite3", adapter_config(run_defaults=run_defaults)
    )
    adapter.submit(miles_request(budgets={"max_wall_time_seconds": 900.0}))

    adapter.driver.tick()

    assert fake_run_store.request("job-0")["spec"]["timeout_s"] == timeout


@pytest.mark.parametrize(
    "request_fields, model",
    [({"model": "policy"}, "policy"), ({"model": None}, "base-model")],
)
def test_the_rollout_samples_the_requests_model_or_else_the_run_defaults_model(
    fake_run_store, tmp_path, request_fields, model
):
    config = adapter_config(run_defaults={"slot": "mini-swe-agent", "model": "base-model"})
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3", config)
    adapter.submit(miles_request(**request_fields))

    adapter.driver.tick()

    spec = fake_run_store.request("job-0")["spec"]
    assert (spec["model"], spec["extra"]["rollout_contract"]["model"]) == (model, model)


@pytest.mark.parametrize(
    "body, config, message",
    [
        pytest.param(
            miles_request(model_endpoint="http://user:secret@model.test"),
            adapter_config(),
            "Model/session endpoint must be HTTP(S) without credentials/query/fragment",
            id="endpoint-credentials",
        ),
        pytest.param(
            miles_request(model_endpoint="http://model.test/v1?key=1"),
            adapter_config(),
            "Model/session endpoint must be HTTP(S) without credentials/query/fragment",
            id="endpoint-query",
        ),
        pytest.param(
            miles_request(model_endpoint="http://model.test/v1#chat"),
            adapter_config(),
            "Model/session endpoint must be HTTP(S) without credentials/query/fragment",
            id="endpoint-fragment",
        ),
        pytest.param(
            miles_request(model_endpoint="ftp://model.test"),
            adapter_config(),
            "Model/session endpoint must be HTTP(S) without credentials/query/fragment",
            id="endpoint-scheme",
        ),
        pytest.param(
            miles_request(model_endpoint="http://"),
            adapter_config(),
            "Model/session endpoint must be HTTP(S) without credentials/query/fragment",
            id="endpoint-host",
        ),
        pytest.param(
            miles_request(sampling_params={"stop_token_ids": [7]}),
            adapter_config(),
            "Unknown message sampling controls",
            id="sampling-control",
        ),
        pytest.param(
            miles_request(sample_slots=slots(2), max_samples=2),
            adapter_config(max_samples=1),
            "Group exceeds deployment max_samples",
            id="group-too-large",
        ),
        pytest.param(
            miles_request(model=None),
            adapter_config(),
            "Provide model or run_defaults.model",
            id="no-model",
        ),
        pytest.param(
            miles_request(),
            {key: value for key, value in adapter_config().items() if key != "image_resources"},
            "Sprout must configure positive image_resources cpu/memory_mb",
            id="no-resources",
        ),
        pytest.param(
            miles_request(),
            adapter_config(image_resources={"cpu": 0, "memory_mb": 12288}),
            "Sprout must configure positive image_resources cpu/memory_mb",
            id="no-cpu",
        ),
        pytest.param(
            miles_request(),
            adapter_config(image_resources={"cpu": True, "memory_mb": 12288}),
            "Sprout must configure positive image_resources cpu/memory_mb",
            id="bool-cpu",
        ),
        pytest.param(
            miles_request(),
            adapter_config(image_resources={"cpu": 2, "memory_mb": 12288, "gpu": 1}),
            "Sprout must configure positive image_resources cpu/memory_mb",
            id="unknown-resource",
        ),
        pytest.param(
            miles_request(),
            adapter_config(tasks={}),
            "Configure Sprout tasks['task-1'].grade for message rollout rewards",
            id="no-grade",
        ),
        pytest.param(
            miles_request(prompt=[{"role": "user", "content": "Fix it"}, {"role": "user", "content": "Now"}]),
            adapter_config(),
            "Native task input must be text or one user message",
            id="two-messages",
        ),
        pytest.param(
            miles_request(prompt=[{"role": "system", "content": "Fix the parser"}]),
            adapter_config(),
            "Native task input must be text or one user message",
            id="system-message",
        ),
        pytest.param(
            miles_request(prompt=[{"role": "user", "content": "Fix the parser", "name": "trainer"}]),
            adapter_config(),
            "Native task input must be text or one user message",
            id="message-field",
        ),
        pytest.param(
            miles_request(prompt=[{"role": "user", "content": [{"type": "text", "text": "Fix the parser"}]}]),
            adapter_config(),
            "Native task input must be text or one user message",
            id="message-parts",
        ),
        pytest.param(
            search_request(slot_count=3),
            adapter_config(),
            "A search request needs the adapter's review config (the review worker profile)",
            id="search-without-review",
        ),
        pytest.param(
            search_request(slot_count=3),
            adapter_config(review={**REVIEW_CONFIG, "max_concurrent_requests": 0}),
            "Review max_concurrent_requests must be a positive integer",
            id="bad-reviewer-setting",
        ),
    ],
)
def test_a_request_that_cannot_run_is_refused_before_it_is_stored_or_submitted(
    fake_run_store, tmp_path, body, config, message
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3", config)

    with pytest.raises(ValueError, match=re.escape(message)):
        adapter.submit(body)

    adapter.driver.tick()
    assert (adapter.driver.ledger.active(), fake_run_store.posts) == ([], [])


def test_a_group_as_large_as_the_deployment_allows_is_accepted(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3", adapter_config(max_samples=2))

    answer = adapter.submit(miles_request(sample_slots=slots(2), max_samples=2))

    assert answer == {"rollout_job_id": "rollout-1", "status": "queued"}


def test_a_resubmitted_request_returns_its_groups_status_and_makes_no_second_job(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[succeeded(verdict(True))])

    answer = adapter.submit(miles_request())
    adapter.driver.tick()

    assert (answer, len(fake_run_store.posts)) == ({"rollout_job_id": "rollout-1", "status": "completed"}, 2)


def test_a_changed_request_under_a_stored_rollout_job_id_is_a_conflict(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request(prompt="Fix the parser"))

    with pytest.raises(Conflict, match="rollout_job_id already names a different request"):
        adapter.submit(miles_request(prompt="Fix the lexer"))


@pytest.mark.parametrize(
    "advance, status",
    [
        pytest.param(lambda driver: None, "queued", id="before-the-first-tick"),
        pytest.param(lambda driver: driver.tick(), "running", id="after-the-first-tick"),
    ],
)
def test_a_group_that_is_not_ready_returns_its_status_and_no_trajectory(
    fake_run_store, tmp_path, advance, status
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    advance(adapter.driver)

    result = adapter.get("rollout-1")

    assert result == {
        "rollout_job_id": "rollout-1",
        "prompt_group_id": "prompt-1",
        "max_samples": 1,
        "actual_samples": 0,
        "trajectories": [],
        "failed_samples": [],
        "search_branches": 0,
        "consumed_budget": {},
        "status": status,
        "stop_reason": None,
    }


def test_the_first_read_of_a_ready_group_ends_it_and_a_restarted_driver_reads_the_same_result(
    fake_run_store, tmp_path
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[succeeded(verdict(True))])

    first = adapter.get("rollout-1")

    restarted = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    assert (restarted.driver.ledger.active(), restarted.get("rollout-1")) == ([], first)


def test_releasing_a_running_group_cancels_it_and_its_result_has_no_trajectory(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    adapter.driver.tick()

    answer = adapter.release("rollout-1")

    result = adapter.get("rollout-1")
    assert (answer, result["status"], result["trajectories"]) == (
        {"rollout_job_id": "rollout-1", "status": "cancelled"},
        "cancelled",
        [],
    )


def test_releasing_a_finished_group_returns_its_status_and_records_the_acknowledgement(
    fake_run_store, tmp_path, clock
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[succeeded(verdict(True))])

    answer = adapter.release("rollout-1")

    assert (answer, adapter.execution("rollout-1")["acknowledged_at"]) == (
        {"rollout_job_id": "rollout-1", "status": "completed"},
        NOW,
    )


@pytest.mark.parametrize(
    "actor_fields, status, stop_reason",
    [
        ({"status": "truncated", "stop_reason": "timeout"}, "truncated", "timeout"),
        ({"status": "truncated", "stop_reason": "max_turns_reached"}, "truncated", "max_turns_reached"),
        ({"status": "truncated", "stop_reason": "interrupted"}, "truncated", "interrupted"),
        ({"status": "completed", "stop_reason": "submitted"}, "completed", None),
    ],
)
def test_a_rollout_cut_at_a_limit_is_graded_and_returned_as_truncated(
    fake_run_store, tmp_path, actor_fields, status, stop_reason
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    run_samples(
        adapter,
        fake_run_store,
        actors=[succeeded(exported(final_snapshot_id="snap-cut", **actor_fields))],
        grades=[succeeded(verdict(True))],
    )

    (trajectory,) = adapter.get("rollout-1")["trajectories"]

    assert (trajectory["status"], trajectory["stop_reason"], trajectory["reward"]) == (
        status,
        stop_reason,
        1.0,
    )
    assert trajectory["metadata"]["graded_snapshot_id"] == "snap-cut"


def test_a_group_past_its_execution_deadline_is_still_graded(fake_run_store, tmp_path, clock):
    """The ``finalization_timeout_seconds`` after the execution deadline are for the snapshot and grade."""
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request(budgets={"max_wall_time_seconds": 900.0}))
    adapter.driver.tick()
    finish_open(fake_run_store, "rollout", [succeeded(exported(status="truncated", stop_reason="timeout"))])
    clock.now = NOW + 901.0

    adapter.driver.tick()

    assert (fake_run_store.ids("grade"), adapter.get("rollout-1")["status"]) == (["job-1"], "running")


@pytest.mark.parametrize(
    "elapsed, status, stopped",
    [(900.0 + 1800.0 - 1.0, "running", False), (900.0 + 1800.0, "cancelled", True)],
)
def test_a_group_is_cancelled_when_its_deadline_comes(
    fake_run_store, tmp_path, clock, elapsed, status, stopped
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(
        miles_request(budgets={"max_wall_time_seconds": 900.0}, finalization_timeout_seconds=1800.0)
    )
    adapter.driver.tick()
    fake_run_store.start("job-0")
    clock.now = NOW + elapsed

    adapter.driver.tick()

    assert (adapter.get("rollout-1")["status"], fake_run_store.stop_requested("job-0")) == (status, stopped)


@pytest.mark.parametrize(
    "actor, grades, reason",
    [
        pytest.param(failed("sandbox did not start"), [], "Actor did not complete", id="actor-failed"),
        pytest.param(
            succeeded(exported(training_export_error="Native history has an incomplete final record")),
            [succeeded(verdict(True))],
            "Native history has an incomplete final record",
            id="export-failed",
        ),
        pytest.param(
            succeeded(exported(training_messages=[])),
            [succeeded(verdict(True))],
            "Execution has no exported native messages",
            id="no-messages",
        ),
        pytest.param(
            succeeded(exported(training_provenance=None)),
            [succeeded(verdict(True))],
            "Execution does not say which exported messages the policy sampled",
            id="no-provenance",
        ),
        pytest.param(
            succeeded(
                exported(training_messages=[*MESSAGES, {"role": "user", "content": mark_hint("Run tests")}])
            ),
            [succeeded(verdict(True))],
            "Exported messages are not in their cleaned training form",
            id="hint-left-in",
        ),
        pytest.param(
            succeeded(exported(status="truncated", stop_reason="agent_crashed")),
            [succeeded(verdict(True))],
            "Truncated actor lacks a verified cutoff result",
            id="cutoff-unverified",
        ),
        pytest.param(
            succeeded(
                exported(training_messages=[*MESSAGES, {"role": "user", "content": HINT_START + "Run tests"}])
            ),
            [succeeded(verdict(True))],
            "Incomplete hint boundary",
            id="hint-boundary-cut",
        ),
    ],
)
def test_a_sample_with_no_trajectory_to_train_on_is_listed_as_failed(
    fake_run_store, tmp_path, actor, grades, reason
):
    """The trainer counts such a slot as missing, never as a reward of 0."""
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    run_samples(adapter, fake_run_store, actors=[actor], grades=grades)

    result = adapter.get("rollout-1")

    assert (result["status"], result["trajectories"], result["failed_samples"], result["stop_reason"]) == (
        "failed",
        [],
        [{"sample_slot_id": "slot-0", "reason": reason}],
        "slot-0: " + reason,
    )


@pytest.mark.parametrize(
    "minimum, second_actor, grades, status",
    [
        (1, failed("sandbox did not start"), [succeeded(verdict(True))], "early_stopped"),
        (2, failed("sandbox did not start"), [succeeded(verdict(True))], "failed"),
        (2, succeeded(exported()), 2 * [succeeded(verdict(True))], "completed"),
    ],
)
def test_a_group_with_fewer_trajectories_than_its_minimum_fails_and_one_with_fewer_than_all_stops_early(
    fake_run_store, tmp_path, minimum, second_actor, grades, status
):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request(sample_slots=slots(2), max_samples=2, minimum_returned_samples=minimum))
    run_samples(adapter, fake_run_store, actors=[succeeded(exported()), second_actor], grades=grades)

    result = adapter.get("rollout-1")

    assert result["status"] == status


def lose_and_run_again(adapter: MessageAdapter, fake: FakeRunStore, job_id: str) -> None:
    """Lose ``job_id`` with its host. The driver plans the group's next run, then submits its roots."""
    fake.finish(job_id, **data_lost())
    adapter.driver.tick()
    adapter.driver.tick()


def test_a_group_whose_third_run_lost_a_job_exports_no_sample_and_gives_its_error(fake_run_store, tmp_path):
    """Miles gets only the failed status and the error, not the samples that settled before the loss."""
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request(sample_slots=slots(2), max_samples=2, minimum_returned_samples=1))
    adapter.driver.tick()
    lose_and_run_again(adapter, fake_run_store, "job-0")
    lose_and_run_again(adapter, fake_run_store, "job-2")
    # The third run: slot-1's actor (job-5) and its grade (job-6) succeed, then slot-0's actor is lost.
    fake_run_store.finish("job-5", **succeeded(exported()))
    adapter.driver.tick()
    fake_run_store.finish("job-6", **succeeded(verdict(True)))
    fake_run_store.finish("job-4", **data_lost())
    adapter.driver.tick()

    result = adapter.get("rollout-1")

    group = internal_id("rollout-1")
    assert (adapter.execution("rollout-1")["samples"][1]["grade"]["state"], result) == (
        "succeeded",
        {
            "rollout_job_id": "rollout-1",
            "prompt_group_id": "prompt-1",
            "max_samples": 2,
            "actual_samples": 0,
            "trajectories": [],
            "failed_samples": [],
            "search_branches": 0,
            "consumed_budget": {},
            "status": "failed",
            "stop_reason": f"A host that held the group's jobs was lost in each of its 3 runs: {group},"
            f" {group}-r1, {group}-r2. A group runs again at most 2 times.",
        },
    )


@pytest.mark.parametrize(
    "grade, zero_reason",
    [
        (failed("sandbox did not start"), "grade_failed"),
        (succeeded({"status": "completed"}), "grade_no_verdict"),
        (succeeded({"status": "completed", "resolved": "yes", "progress": 1.0}), "grade_no_verdict"),
    ],
)
def test_a_trajectory_whose_grade_gave_no_verdict_is_returned_as_missing(
    fake_run_store, tmp_path, grade, zero_reason
):
    """Missing is not a 0: the wire needs a reward, and the metadata says that there is no verdict."""
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[grade])

    result = adapter.get("rollout-1")

    (trajectory,) = result["trajectories"]
    metadata = trajectory["metadata"]
    assert (result["status"], trajectory["reward"], metadata["zero_reason"]) == (
        "completed",
        0.0,
        zero_reason,
    )
    assert (metadata["progress"], metadata["touched_test_paths"]) == (None, None)


def run_to_grade(adapter: MessageAdapter, fake: FakeRunStore) -> None:
    """Submit ``miles_request()``, finish its actor, and let the driver submit its grade ``job-1``."""
    adapter.submit(miles_request())
    adapter.driver.tick()
    finish_open(fake, "rollout", [succeeded(exported())])
    adapter.driver.tick()


@pytest.mark.parametrize("elapsed, status, stopped", [(599.0, "running", False), (600.0, "completed", True)])
def test_a_grade_is_given_up_and_stopped_once_it_has_run_for_its_wall_time(
    fake_run_store, tmp_path, clock, elapsed, status, stopped
):
    adapter = miles_adapter(
        fake_run_store, tmp_path / "ledger.sqlite3", adapter_config(grade_wall_seconds=600)
    )
    run_to_grade(adapter, fake_run_store)
    fake_run_store.start("job-1")
    adapter.driver.tick()
    clock.now += elapsed

    adapter.driver.tick()

    assert (adapter.get("rollout-1")["status"], fake_run_store.stop_requested("job-1")) == (status, stopped)


def test_a_given_up_grade_leaves_its_trajectory_missing(fake_run_store, tmp_path, clock):
    adapter = miles_adapter(
        fake_run_store, tmp_path / "ledger.sqlite3", adapter_config(grade_wall_seconds=600)
    )
    run_to_grade(adapter, fake_run_store)
    fake_run_store.start("job-1")
    adapter.driver.tick()
    clock.now += 600.0
    adapter.driver.tick()

    (trajectory,) = adapter.get("rollout-1")["trajectories"]

    assert (
        trajectory["reward"],
        trajectory["metadata"]["zero_reason"],
        trajectory["metadata"]["progress"],
    ) == (
        0.0,
        "grade_expired",
        None,
    )


def test_a_grade_is_not_charged_its_time_in_the_queue(fake_run_store, tmp_path, clock):
    adapter = miles_adapter(
        fake_run_store, tmp_path / "ledger.sqlite3", adapter_config(grade_wall_seconds=600)
    )
    run_to_grade(adapter, fake_run_store)
    clock.now += 601.0

    adapter.driver.tick()

    assert (adapter.get("rollout-1")["status"], fake_run_store.jobs["job-1"]["state"]) == (
        "running",
        "queued",
    )


def test_hint_markers_in_assistant_or_tool_text_reach_miles_as_the_policy_wrote_them(
    fake_run_store, tmp_path
):
    """Only system and user messages carry hints. A marker elsewhere is text that the policy saw."""
    call = {"id": "call-1", "type": "function", "function": {"name": "bash", "arguments": "{}"}}
    messages = [
        {"role": "user", "content": "Fix the parser"},
        {"role": "assistant", "content": "I will print " + mark_hint("a marker"), "tool_calls": [call]},
        {"role": "tool", "tool_call_id": "call-1", "content": mark_hint("tool output")},
        {"role": "assistant", "content": "Fixed."},
    ]
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    run_samples(
        adapter,
        fake_run_store,
        actors=[succeeded(exported(training_messages=messages))],
        grades=[succeeded(verdict(True))],
    )

    (trajectory,) = adapter.get("rollout-1")["trajectories"]

    assert trajectory["messages"] == messages


def test_the_grades_verdict_and_its_hidden_test_facts_reach_the_trajectory(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request())
    grade = verdict(False, progress=0.5, touched_test_paths=["tests/test_parser.py"])
    run_samples(adapter, fake_run_store, actors=[succeeded(exported())], grades=[succeeded(grade)])

    (trajectory,) = adapter.get("rollout-1")["trajectories"]

    metadata = trajectory["metadata"]
    assert (
        trajectory["reward"],
        metadata["progress"],
        metadata["touched_test_paths"],
        metadata["zero_reason"],
    ) == (
        0.0,
        0.5,
        ["tests/test_parser.py"],
        None,
    )


def test_the_consumed_budget_sums_the_usage_of_the_trajectories(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")
    adapter.submit(miles_request(sample_slots=slots(2), max_samples=2))
    actors = [
        succeeded(exported(rollout_usage={"model_calls": 3, "tool_calls": 5})),
        succeeded(exported(rollout_usage={"model_calls": 2, "tool_calls": 1})),
    ]
    run_samples(adapter, fake_run_store, actors=actors, grades=2 * [succeeded(verdict(True))])

    result = adapter.get("rollout-1")

    assert result["consumed_budget"] == {"model_calls": 5, "tool_calls": 6}


def test_the_grade_job_grades_the_final_snapshot_once_with_the_tasks_template(fake_run_store, tmp_path):
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3")

    run_to_grade(adapter, fake_run_store)

    assert fake_run_store.request("job-1") == {
        "kind": "grade",
        "profile": "grade",
        "spec": {**GRADE_TEMPLATE["spec"], "image": IMAGE, "timeout_s": 480.0, "snapshot_id": "snap-final"},
        "context": {
            "rl_driver": {
                "rollout_job_id": internal_id("rollout-1"),
                "sample_slot_id": internal_id("slot-0"),
                "actor_job_id": "job-0",
                "actor_attempt_id": "attempt-job-0",
                "point_id": None,
            }
        },
        "max_infra_retries": 0,
        "parent_point": None,
        "affinity": internal_id("rollout-1"),
    }


@pytest.mark.parametrize(
    "template_spec, wall, timeout", [({}, 600, 480.0), ({"timeout_s": 100}, 600, 100), ({}, 240, 120.0)]
)
def test_the_grades_tests_end_within_its_wall_time_less_the_preparation_time(
    fake_run_store, tmp_path, template_spec, wall, timeout
):
    grade = {**GRADE_TEMPLATE, "spec": {**GRADE_TEMPLATE["spec"], **template_spec}}
    config = adapter_config(tasks={"task-1": {"grade": grade}}, grade_wall_seconds=wall)
    adapter = miles_adapter(fake_run_store, tmp_path / "ledger.sqlite3", config)

    run_to_grade(adapter, fake_run_store)

    assert fake_run_store.request("job-1")["spec"]["timeout_s"] == timeout

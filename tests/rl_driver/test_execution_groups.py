"""A driver runs each execution group as Run Store jobs, once, across lost replies, restarts and cancels.

The Run Store is ``FakeRunStore``, and the test plays its worker. ``test_runstore_fidelity.py`` runs
scenarios of this file on the real Run Store too.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Conflict, Ledger
from sprout.runstore.specs import JobSpec
from tests.rl_driver.fake_runstore import FakeRunStore
from tests.rl_driver.miles import GRADE_TEMPLATE
from tests.support.paths import REPO

#: The task image of the samples here.
IMAGE = "docker.io/swerebenchv2/task-1:base"
#: The idempotency key of the actor of slot-0 in group-1.
ACTOR_KEY = "rl-driver/7df1b9aad21dccc01af7495f107092dc111f25463d0ad495318678ce1ceb76c7"
#: The grade template of the graded samples here.
GRADE = {**GRADE_TEMPLATE, "spec": {**GRADE_TEMPLATE["spec"], "image": IMAGE}}
#: The tool calls of an attempt that ends at tool depth 3.
THREE_TOOLS = [{"depth": depth, "response": {"output": "ok"}} for depth in (1, 2, 3)]


def run_sample(
    slot: str = "slot-0", *, graded: bool = False, prompt: str = "Fix the parser"
) -> dict[str, Any]:
    sample: dict[str, Any] = {
        "sample_slot_id": slot,
        "run": {"profile": "mini", "spec": {"prompt": prompt, "sandbox_image": IMAGE, "timeout_s": 900}},
    }
    return {**sample, "grade": GRADE} if graded else sample


def group(*samples: dict[str, Any], group_id: str = "group-1") -> dict[str, Any]:
    return {"rollout_job_id": group_id, "prompt_group_id": "prompt-1", "samples": list(samples)}


def new_driver(fake: FakeRunStore, directory: Path) -> Driver:
    """A driver on the ledger file in ``directory``: a second call there is the same driver restarted."""
    return Driver(fake.client(), Ledger(directory / "ledger.sqlite3"))


def parent_rollout(fake: FakeRunStore) -> str:
    """A finished rollout ``job-0`` under a turn limit, with an available recovery point ``point-2``."""
    run = run_sample()["run"]
    spec = {**run["spec"], "extra": {"rollout_contract": {"max_turns": 10}}}
    job_id = fake.client().submit(JobSpec.from_dict({**run, "kind": "rollout", "spec": spec}), "parent")
    fake.finish(job_id, result={"status": "completed"})
    fake.publish(
        job_id,
        tools=THREE_TOOLS,
        points=[
            {"id": "point-2", "tool_depth": 2, "message_step": 1, "snapshot_id": "snap-2", "available": True}
        ],
    )
    return job_id


def branch_sample(parent: str) -> dict[str, Any]:
    return {
        "sample_slot_id": "slot-0",
        "branch": {
            "job_id": parent,
            "point_id": "point-2",
            "continuation": {"guidance": "user-hint", "hint": "Run the tests first"},
            "budget": {"max_turns": 7},
        },
    }


def recorded(stage: dict[str, Any]) -> dict[str, Any]:
    """What docs/RL_DRIVER.md says a view records of a stage: its job, attempt, state and result."""
    return {key: stage.get(key) for key in ("job_id", "attempt_id", "state", "result")}


def test_a_group_whose_jobs_all_succeeded_is_completed_with_each_jobs_result(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample("slot-0"), run_sample("slot-1")))
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed", "final_text": "first"})
    fake_run_store.finish("job-1", result={"status": "completed", "final_text": "second"})

    driver.tick()

    view = driver.get("group-1")
    assert (view["status"], view["ready"], [recorded(sample["actor"]) for sample in view["samples"]]) == (
        "completed",
        True,
        [
            {
                "job_id": "job-0",
                "attempt_id": "attempt-job-0",
                "state": "succeeded",
                "result": {"status": "completed", "final_text": "first"},
            },
            {
                "job_id": "job-1",
                "attempt_id": "attempt-job-1",
                "state": "succeeded",
                "result": {"status": "completed", "final_text": "second"},
            },
        ],
    )


def test_a_group_is_not_ready_while_one_of_its_jobs_runs(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample("slot-0"), run_sample("slot-1")))
    driver.tick()
    fake_run_store.start("job-0")
    fake_run_store.finish("job-1", result={"status": "completed"})

    driver.tick()

    view = driver.get("group-1")
    assert (view["status"], view["ready"]) == ("running", False)


def test_a_failed_job_fails_the_group_and_the_other_results_are_kept(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample("slot-0"), run_sample("slot-1")))
    driver.tick()
    fake_run_store.finish("job-0", state="failed", error="sandbox did not start")
    fake_run_store.finish("job-1", result={"status": "completed"})

    driver.tick()

    view = driver.get("group-1")
    assert (view["status"], view["ready"], recorded(view["samples"][1]["actor"])) == (
        "failed",
        True,
        {
            "job_id": "job-1",
            "attempt_id": "attempt-job-1",
            "state": "succeeded",
            "result": {"status": "completed"},
        },
    )


def test_a_view_leaves_out_the_submission_bodies_that_the_ledger_keeps(fake_run_store, tmp_path):
    """A submission body can hold a long prompt. The view names the jobs, and a restart needs the bodies."""
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed", "final_snapshot_id": "snap-final"})
    driver.tick()

    view = driver.get("group-1")["samples"][0]

    stored = driver.ledger.get("group-1")["document"]["samples"][0]
    bodies = {"submission", "template"}
    assert [bodies & set(view["actor"]), bodies & set(view["grade"])] == [set(), set()]
    assert [bodies & set(stored["actor"]), bodies & set(stored["grade"])] == [{"submission"}, bodies]


def test_a_view_leaves_out_the_submission_bodies_of_the_earlier_runs_too(fake_run_store, tmp_path):
    """A group whose job lost its data with its host runs again, and keeps the run before in the ledger."""
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.tick()
    fake_run_store.finish("job-0", state="failed", result={"reason": "data_lost"})
    driver.tick()

    (earlier,) = driver.get("group-1")["earlier_runs"]

    (stored,) = driver.ledger.get("group-1")["document"]["earlier_runs"]
    bodies = {"submission", "template"}
    sample, kept = earlier["samples"][0], stored["samples"][0]
    assert [bodies & set(sample["actor"]), bodies & set(sample["grade"])] == [set(), set()]
    assert [bodies & set(kept["actor"]), bodies & set(kept["grade"])] == [{"submission"}, {"template"}]


def test_a_resubmitted_group_returns_its_view_and_makes_no_second_job(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample()))
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed"})
    driver.tick()

    view = driver.submit(group(run_sample()))
    driver.tick()

    assert (view["status"], list(fake_run_store.jobs)) == ("completed", ["job-0"])


def test_a_changed_request_under_a_stored_group_id_is_a_conflict(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(prompt="Fix the parser")))

    with pytest.raises(Conflict, match="rollout_job_id already names a different request"):
        driver.submit(group(run_sample(prompt="Fix the lexer")))


def test_the_jobs_of_a_group_carry_the_drivers_idempotency_key_and_record(fake_run_store, tmp_path):
    """A restarted driver finds the jobs of its stored groups by these keys: a new key submits them again."""
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample()) | {"context": {"experiment": "e-1"}})

    driver.tick()

    job = fake_run_store.jobs["job-0"]
    assert (job["idempotency_key"], job["request"]["context"]) == (
        ACTOR_KEY,
        {
            "rl_driver": {
                "rollout_job_id": "group-1",
                "prompt_group_id": "prompt-1",
                "sample_slot_id": "slot-0",
                "context": {"experiment": "e-1"},
            }
        },
    )


def test_a_restarted_driver_follows_the_jobs_of_the_stored_groups(fake_run_store, tmp_path):
    new_driver(fake_run_store, tmp_path).submit(group(run_sample()))
    new_driver(fake_run_store, tmp_path).tick()
    restarted = new_driver(fake_run_store, tmp_path)
    fake_run_store.finish("job-0", result={"status": "completed"})

    restarted.tick()

    view = restarted.get("group-1")
    assert (view["status"], list(fake_run_store.jobs)) == ("completed", ["job-0"])


@pytest.mark.parametrize(
    "sample, path",
    [
        pytest.param(run_sample(), "/v1/jobs", id="run"),
        pytest.param(branch_sample("job-0"), "/v1/jobs/job-0/branch", id="branch"),
    ],
)
def test_a_submission_whose_reply_was_lost_is_retried_under_its_key_and_makes_one_job(
    fake_run_store, tmp_path, sample, path
):
    parent_rollout(fake_run_store)
    new_driver(fake_run_store, tmp_path).submit(group(sample))
    fake_run_store.lose_next_reply()
    new_driver(fake_run_store, tmp_path).tick()

    restarted = new_driver(fake_run_store, tmp_path)
    restarted.tick()

    actor = restarted.get("group-1")["samples"][0]["actor"]
    assert (actor["job_id"], list(fake_run_store.jobs), fake_run_store.posts[1:]) == (
        "job-1",
        ["job-0", "job-1"],
        [(path, ACTOR_KEY), (path, ACTOR_KEY)],
    )


def test_a_branch_sample_is_posted_as_a_branch_from_its_parents_point(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    parent = parent_rollout(fake_run_store)
    driver.submit(group(branch_sample(parent)))

    driver.tick()

    assert fake_run_store.branch_posts == [
        (
            parent,
            {
                "point_id": "point-2",
                "continuation": {"guidance": "user-hint", "hint": "Run the tests first"},
                "overrides": {},
                "budget": {"max_turns": 7},
            },
        )
    ]


def test_one_groups_failed_submission_does_not_stop_the_next_group(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(), group_id="group-1"))
    driver.submit(group(run_sample(), group_id="group-2"))
    fake_run_store.lose_next_reply()

    driver.tick()

    assert driver.get("group-2")["samples"][0]["actor"]["job_id"] == "job-1"


def test_cancelling_a_group_cancels_its_queued_job_and_asks_its_running_job_to_stop(fake_run_store, tmp_path):
    """A running job ends when its worker sees the stop request, so the group waits for it."""
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample("slot-0", graded=True), run_sample("slot-1", graded=True)))
    driver.tick()
    fake_run_store.start("job-0")
    driver.release("group-1")

    driver.tick()

    jobs = fake_run_store.jobs
    assert (jobs["job-0"]["state"], fake_run_store.stop_requested("job-0"), jobs["job-1"]["state"]) == (
        "running",
        True,
        "cancelled",
    )
    view = driver.get("group-1")
    assert (view["status"], view["ready"]) == ("cancelling", False)


def test_a_cancelled_group_grades_nothing_and_is_cancelled_once_its_running_job_ends(
    fake_run_store, tmp_path
):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.tick()
    fake_run_store.start("job-0")
    driver.release("group-1")
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed", "final_snapshot_id": "snap-final"})

    driver.tick()

    view = driver.get("group-1")
    assert (view["status"], view["ready"], fake_run_store.ids("grade")) == ("cancelled", True, [])


def test_a_group_cancelled_before_its_first_tick_submits_no_job(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.release("group-1")

    driver.tick()

    view = driver.get("group-1")
    assert (view["status"], view["ready"], fake_run_store.posts) == ("cancelled", True, [])


def test_a_cancelled_submission_whose_reply_was_lost_is_not_posted_again(fake_run_store, tmp_path):
    """A second post could make a new job after the cancellation, if the first never arrived."""
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample()))
    fake_run_store.lose_next_reply()
    driver.tick()
    driver.release("group-1")

    driver.tick()

    view = driver.get("group-1")
    actor = view["samples"][0]["actor"]
    assert (view["status"], len(fake_run_store.posts), actor["idempotency_key"], actor["last_error"]) == (
        "cancelling",
        1,
        ACTOR_KEY,
        "Submission acknowledgement unknown; inspect Run Store using the idempotency key",
    )


def test_reconciling_a_lost_submission_attaches_the_job_with_its_key_and_lets_the_cancel_finish(
    fake_run_store, tmp_path
):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample()))
    fake_run_store.lose_next_reply()
    driver.tick()
    driver.release("group-1")
    driver.tick()

    driver.reconcile_submission("group-1", "slot-0", "actor", "job-0")
    driver.tick()

    view = driver.get("group-1")
    assert (view["status"], fake_run_store.jobs["job-0"]["state"], len(fake_run_store.posts)) == (
        "cancelled",
        "cancelled",
        1,
    )


def test_reconciling_refuses_a_job_submitted_under_another_key(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample()))
    fake_run_store.lose_next_reply()
    driver.tick()
    driver.release("group-1")
    driver.tick()
    other = fake_run_store.client().submit(
        JobSpec.from_dict(run_sample()["run"] | {"kind": "rollout"}), "other"
    )

    with pytest.raises(ValueError, match="Job does not match this submission's idempotency key"):
        driver.reconcile_submission("group-1", "slot-0", "actor", other)


@pytest.mark.parametrize(
    "sample, phase, error, message",
    [
        ("slot-0", "actor", ValueError, "Only an unacknowledged submission can be reconciled"),
        ("slot-0", "grade", ValueError, "Only an unacknowledged submission can be reconciled"),
        ("slot-9", "actor", KeyError, "slot-9"),
        ("slot-0", "rollout", ValueError, "phase must be actor, grade or review"),
    ],
)
def test_reconciling_a_stage_that_is_no_lost_submission_is_refused(
    fake_run_store, tmp_path, sample, phase, error, message
):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample()))
    driver.tick()

    with pytest.raises(error, match=message):
        driver.reconcile_submission("group-1", sample, phase, "job-0")


def test_the_grade_runs_from_the_last_available_point_at_the_actors_final_tool_call(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed"})
    fake_run_store.publish(
        "job-0",
        tools=THREE_TOOLS,
        points=[
            {"id": "point-2", "tool_depth": 2, "message_step": 1, "snapshot_id": "snap-2", "available": True},
            {"id": "point-3", "tool_depth": 3, "message_step": 2, "snapshot_id": "snap-3", "available": True},
            {"id": "point-4", "tool_depth": 3, "message_step": 3, "snapshot_id": "snap-4", "available": True},
        ],
    )

    driver.tick()

    grade = fake_run_store.request("job-1")
    assert (grade["spec"]["snapshot_id"], grade["context"]) == (
        "snap-4",
        {
            "rl_driver": {
                "rollout_job_id": "group-1",
                "sample_slot_id": "slot-0",
                "actor_job_id": "job-0",
                "actor_attempt_id": "attempt-job-0",
                "point_id": "point-4",
            }
        },
    )


def point(step: int, *, depth: int, available: bool = True) -> dict[str, Any]:
    return {
        "id": f"point-{step}",
        "tool_depth": depth,
        "message_step": step,
        "snapshot_id": f"snap-{step}",
        "available": available,
    }


@pytest.mark.parametrize(
    "tools, points, error",
    [
        ([], [point(1, depth=1)], "Cannot grade without a complete final tool boundary"),
        (
            [{"depth": 1, "response": {"output": "ok"}}, {"depth": 2, "response": None}],
            [point(1, depth=1)],
            "Cannot grade without a complete final tool boundary",
        ),
        (
            [{"depth": 1, "response": {}}, {"depth": 3, "response": {}}],
            [point(1, depth=3)],
            "Final tool prefix has a gap",
        ),
        (
            THREE_TOOLS,
            [point(1, depth=2), point(2, depth=3, available=False)],
            "Final message has no available recovery point; refusing an older snapshot",
        ),
    ],
)
def test_an_actor_whose_final_state_cannot_be_proved_fails_its_grade_and_no_older_snapshot_is_graded(
    fake_run_store, tmp_path, tools, points, error
):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed"})
    fake_run_store.publish("job-0", tools=tools, points=points)

    driver.tick()

    view = driver.get("group-1")
    grade = view["samples"][0]["grade"]
    assert (view["status"], grade["state"], grade["error"], fake_run_store.ids("grade")) == (
        "failed",
        "failed",
        error,
        [],
    )


def test_a_grade_whose_reply_was_lost_is_retried_from_the_snapshot_chosen_first(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed"})
    fake_run_store.publish("job-0", tools=THREE_TOOLS, points=[point(2, depth=3)])
    fake_run_store.lose_next_reply()
    driver.tick()
    fake_run_store.publish("job-0", tools=THREE_TOOLS, points=[point(2, depth=3), point(3, depth=3)])

    new_driver(fake_run_store, tmp_path).tick()

    grade = new_driver(fake_run_store, tmp_path).get("group-1")["samples"][0]["grade"]
    assert (
        grade["job_id"],
        fake_run_store.ids("grade"),
        fake_run_store.request("job-1")["spec"]["snapshot_id"],
    ) == (
        "job-1",
        ["job-1"],
        "snap-2",
    )


def test_a_cancellation_while_the_grade_is_prepared_submits_no_grade(fake_run_store, tmp_path):
    driver = new_driver(fake_run_store, tmp_path)
    driver.submit(group(run_sample(graded=True)))
    driver.tick()
    fake_run_store.finish("job-0", result={"status": "completed"})
    fake_run_store.publish("job-0", tools=THREE_TOOLS, points=[point(2, depth=3)])
    fake_run_store.before_points_reply(lambda: driver.release("group-1"))

    driver.tick()

    view = driver.get("group-1")
    assert (view["status"], fake_run_store.ids("grade")) == ("cancelled", [])


def with_run(group_body: dict[str, Any], **spec: Any) -> dict[str, Any]:
    """``group_body`` with ``spec`` added to the run spec of its first sample."""
    sample = group_body["samples"][0]
    run = {**sample["run"], "spec": {**sample["run"]["spec"], **spec}}
    return {**group_body, "samples": [{**sample, "run": run}, *group_body["samples"][1:]]}


def with_branch(**fields: Any) -> dict[str, Any]:
    """A group of one branch sample whose ``branch`` has ``fields`` in place of its own."""
    sample = branch_sample("job-0")
    return group({**sample, "branch": {**sample["branch"], **fields}})


@pytest.mark.parametrize(
    "body, message",
    [
        (group(run_sample()) | {"protocol_version": "1"}, "Invalid group fields"),
        (group(run_sample()) | {"budgets": {"max_model_calls": 2}}, "Invalid group fields"),
        (group(run_sample(), run_sample()), "sample_slot_id values must be unique"),
        (group(run_sample() | {"branch": {}}), "exactly one run, branch or review"),
        (
            with_run(group(run_sample()), sampling_params={"temperature": 1}),
            "Unknown RunSpec fields",
        ),
        (with_run(group(run_sample()), extra={"api_key": "inline-secret"}), r"\$env reference"),
        (with_branch(continuation=None), "A continuation is an object"),
        (
            with_branch(overrides={"prompt": "Try a different fix"}),
            "Invalid branch overrides fields",
        ),
        (with_branch(continuation={"guidance": "assistant-turn"}), "needs its assistant_turn"),
        (with_branch(budget={"turns": 3}), "A branch budget sets max_turns and/or deadline_at"),
        (
            group(run_sample() | {"grade": {**GRADE, "spec": {**GRADE["spec"], "snapshot_id": "snap-1"}}}),
            "A grade's snapshot_id or in_place_job is selected from its actor",
        ),
    ],
)
def test_an_invalid_group_is_refused_before_it_is_stored_or_submitted(
    fake_run_store, tmp_path, body, message
):
    driver = new_driver(fake_run_store, tmp_path)

    with pytest.raises(ValueError, match=message):
        driver.submit(body)

    driver.tick()
    assert (driver.ledger.active(), fake_run_store.posts) == ([], [])


def test_the_shipped_execution_group_example_is_accepted_and_runs_one_job_for_each_sample(
    fake_run_store, tmp_path
):
    """docs/RL_DRIVER.md gives this file as the template of a group."""
    body = json.loads((REPO / "examples" / "rl_driver" / "execution-group.example.json").read_text())
    driver = new_driver(fake_run_store, tmp_path)

    driver.submit(body)
    driver.tick()

    assert (driver.get("experiment-1-group-0")["status"], fake_run_store.ids("rollout")) == (
        "running",
        ["job-0", "job-1"],
    )

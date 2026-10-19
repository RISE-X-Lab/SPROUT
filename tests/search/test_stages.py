"""The coordinator runs the actor and the grade of each sample as Run Store jobs, and settles the group.

The Run Store is a ``FakeRunStore``, and the group store a ``FileGroups`` in ``tmp_path``. Between two
ticks, the test plays the worker. These groups have no search.
"""

from __future__ import annotations

import threading
from collections.abc import Iterator
from functools import partial
from typing import Any

import pytest

from sprout.eval.groups import FileGroups
from sprout.runstore.specs import JobSpec
from tests.search.documents import (
    DATA_LOST,
    GRADE,
    GROUP,
    REVIEW,
    RUN,
    SUBMITTER,
    coordinator_for,
    document,
    exported,
    lose,
    run,
    stage,
    stored,
    verdict,
)
from tests.search.fake_runstore import STOP_REQUESTED_AT, FakeRunStore, complete_tools, point

#: A grade template that runs a Harbor verifier in the sandbox that the actor keeps.
IN_PLACE_GRADE = {
    "profile": "grade",
    "spec": {"benchmark": "harbor", "task_dir": "/path/to/task"},
    "in_place": True,
}
#: A failed root that a review sample reads.
PARENT = {"name": "root", "job_id": "job-x", "attempt_id": "attempt-x", "grade": {"resolved": False}}
FAILED = {"status": "error", "error": "the agent crashed"}
#: What the fake Run Store answers to a request that ``fail_next`` refuses with a status.
REFUSAL = '{"detail":"the fake Run Store refused it"}'


class Gate:
    """Holds a request inside the fake Run Store until the test opens the gate."""

    def __init__(self) -> None:
        self.entered = threading.Event()
        self._opened = threading.Event()

    def hold(self) -> None:
        self.entered.set()
        assert self._opened.wait(10), "the test did not open the gate"

    def open(self) -> None:
        self._opened.set()


@pytest.fixture
def gate() -> Iterator[Gate]:
    """A closed ``Gate``, opened after the test, so that no thread waits on it after a failure."""
    opened = Gate()
    yield opened
    opened.open()


def parent_rollout(fake: FakeRunStore) -> str:
    """A finished rollout, ``job-0``, with the recovery points ``job-0-step2`` and ``job-0-step3``."""
    job_id = fake.client.submit(JobSpec.from_dict({"kind": "rollout", **RUN}), "parent")
    fake.finish_rollout(job_id, exported())
    return job_id


def contexts(fake: FakeRunStore, kind: str) -> list[dict[str, Any]]:
    """The context of each job of ``kind`` that the coordinator submitted, in order."""
    return [body["context"] for body in fake.submitted(kind)]


def grade_specs(fake: FakeRunStore) -> list[dict[str, Any]]:
    """The spec of each grade job that the coordinator submitted, in order."""
    return [body["spec"] for body in fake.submitted("grade")]


BRANCH = {
    "sample_slot_id": "slot-0",
    "branch": {"job_id": "job-0", "point_id": "job-0-step2", "continuation": {"guidance": "none"}},
}


def test_a_run_and_its_grade_settle_the_group_as_completed(fake_run_store, groups):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)

    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()
    fake_run_store.finish("job-1", verdict(True))
    coordinator.tick()

    row = groups.get(GROUP)
    assert (row["terminal"], row["document"]["ready"], row["document"]["status"]) == (True, True, "completed")


def test_each_job_carries_the_submitters_record_of_its_sample_in_its_context(fake_run_store, groups):
    review = {
        "sample_slot_id": "review-1",
        "review": {"profile": "review", "spec": {**REVIEW, "parents": [PARENT]}},
    }
    stored(groups, document(run("slot-0"), review))
    coordinator = coordinator_for(fake_run_store, groups)

    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()

    assert [
        contexts(fake_run_store, "rollout"),
        contexts(fake_run_store, "review"),
        contexts(fake_run_store, "grade"),
    ] == [
        [
            {
                "eval": {
                    "rollout_job_id": "group",
                    "prompt_group_id": "prompt",
                    "sample_slot_id": "slot-0",
                    "context": {},
                }
            }
        ],
        [
            {
                "eval": {
                    "rollout_job_id": "group",
                    "prompt_group_id": "prompt",
                    "sample_slot_id": "review-1",
                    "context": {},
                }
            }
        ],
        [
            {
                "eval": {
                    "rollout_job_id": "group",
                    "sample_slot_id": "slot-0",
                    "actor_job_id": "job-0",
                    "actor_attempt_id": "attempt-job-0",
                    "point_id": "job-0-step3",
                }
            }
        ],
    ]


def test_each_job_carries_the_key_that_the_caller_gave_its_group(fake_run_store, groups):
    """The Run Store runs the jobs of one key on one host, where the grade finds its actor's snapshot.

    Manual evaluation gives one key to each group of a task's search, so the key is not the group id."""
    review = {
        "sample_slot_id": "review-1",
        "review": {"profile": "review", "spec": {**REVIEW, "parents": [PARENT]}},
    }
    stored(groups, document(run("slot-0"), review, affinity="search"))
    coordinator = coordinator_for(fake_run_store, groups)

    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()

    assert [
        [body["affinity"] for body in fake_run_store.submitted(kind)]
        for kind in ("rollout", "review", "grade")
    ] == [["search"], ["search"], ["search"]]


def test_each_stage_is_submitted_under_its_submitters_key_for_that_stage(fake_run_store, groups):
    stored(groups, document(run("slot-0"), run("slot-1", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)

    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()

    assert fake_run_store.keys() == [
        SUBMITTER.key(affinity="group", sample_id="slot-0", phase="actor"),
        SUBMITTER.key(affinity="group", sample_id="slot-1", phase="actor"),
        SUBMITTER.key(affinity="group", sample_id="slot-0", phase="grade"),
    ]


def test_a_grade_runs_from_the_latest_available_snapshot_at_the_actors_final_tool_boundary(
    fake_run_store, groups
):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish("job-0", exported())
    final = point("job-0", 3)
    fake_run_store.record(
        "job-0",
        tools=complete_tools(3),
        points=[
            point("job-0", 2),
            {**final, "id": "final-1", "message_step": 2, "snapshot_id": "FINAL-1"},
            {**final, "id": "final-2", "message_step": 3, "snapshot_id": "FINAL-2"},
            {**final, "id": "final-3", "message_step": 4, "snapshot_id": "GONE", "available": False},
        ],
    )

    coordinator.tick()

    assert (grade_specs(fake_run_store), stage(groups, "slot-0", "grade")["snapshot_id"]) == (
        [{**GRADE["spec"], "snapshot_id": "FINAL-2"}],
        "FINAL-2",
    )


def test_a_grade_runs_from_the_final_snapshot_that_its_rollout_reports_without_reading_its_tools(
    fake_run_store, groups
):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish("job-0", exported(final_snapshot_id="FINAL"))

    coordinator.tick()

    reads = (
        fake_run_store.requests.count(("GET", "tools")),
        fake_run_store.requests.count(("GET", "recovery-points")),
    )
    assert (grade_specs(fake_run_store), reads) == ([{**GRADE["spec"], "snapshot_id": "FINAL"}], (0, 0))


def test_an_in_place_grade_runs_in_the_sandbox_that_its_actor_kept(fake_run_store, groups):
    """It names the actor's job and no snapshot, so it needs no tool boundary: this actor recorded none."""
    stored(groups, document(run("slot-0", grade=IN_PLACE_GRADE)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish("job-0", exported())

    coordinator.tick()

    assert (grade_specs(fake_run_store), contexts(fake_run_store, "grade")) == (
        [{"benchmark": "harbor", "task_dir": "/path/to/task", "in_place_job": "job-0"}],
        [
            {
                "eval": {
                    "rollout_job_id": "group",
                    "sample_slot_id": "slot-0",
                    "actor_job_id": "job-0",
                    "actor_attempt_id": "attempt-job-0",
                }
            }
        ],
    )


NO_BOUNDARY = "Cannot grade without a complete final tool boundary"
NO_FINAL_POINT = "Final message has no available recovery point; refusing an older snapshot"


@pytest.mark.parametrize(
    "tools, points, error",
    [
        ([], [], NO_BOUNDARY),
        (
            [*complete_tools(2), {"depth": 3, "call_id": "call-3", "response": None}],
            [point("job-0", 3)],
            NO_BOUNDARY,
        ),
        ([complete_tools(3)[0], complete_tools(3)[2]], [point("job-0", 3)], "Final tool prefix has a gap"),
        (complete_tools(3), [point("job-0", 2), point("job-0", 3, available=False)], NO_FINAL_POINT),
        (complete_tools(3), [point("job-0", 2)], NO_FINAL_POINT),
    ],
    ids=["no tool call", "a call without its response", "a gap", "final point unavailable", "no final point"],
)
def test_a_grade_whose_source_cannot_be_proved_fails_and_submits_no_job(
    fake_run_store, groups, tools, points, error
):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish("job-0", exported())
    fake_run_store.record("job-0", tools=tools, points=points)

    coordinator.tick()

    grade = stage(groups, "slot-0", "grade")
    assert (grade["state"], grade["error"], fake_run_store.job_ids("grade")) == ("failed", error, [])


@pytest.mark.parametrize("state", ["failed", "cancelled"])
def test_the_grade_of_an_actor_that_did_not_succeed_is_skipped(fake_run_store, groups, state):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish("job-0", FAILED, state=state)

    coordinator.tick()

    grade = stage(groups, "slot-0", "grade")
    assert (grade["state"], grade["error"], fake_run_store.job_ids("grade")) == (
        "skipped",
        "Actor did not succeed",
        [],
    )


def test_a_group_with_a_failed_stage_is_failed_and_keeps_the_results_of_the_others(fake_run_store, groups):
    stored(groups, document(run("slot-0", grade=None), run("slot-1", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish("job-0", exported())
    fake_run_store.finish("job-1", FAILED, state="failed")

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (settled["ready"], settled["status"], settled["samples"][0]["actor"]["result"]) == (
        True,
        "failed",
        exported(),
    )


def test_a_job_that_lost_its_data_runs_its_group_again_under_new_keys(fake_run_store, groups):
    """The new keys make new jobs, and the Run Store runs the jobs of the new affinity key on a live host."""
    stored(groups, document(run("slot-0", grade=None), run("slot-1", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    lose(fake_run_store, "job-0")

    coordinator.tick()
    coordinator.tick()

    assert (
        [body["affinity"] for body in fake_run_store.submitted("rollout")],
        fake_run_store.keys()[2:],
    ) == (
        ["group", "group", "group-r1", "group-r1"],
        [
            SUBMITTER.key(affinity="group-r1", sample_id="slot-0", phase="actor"),
            SUBMITTER.key(affinity="group-r1", sample_id="slot-1", phase="actor"),
        ],
    )


def test_a_group_that_runs_again_plans_each_stage_again_and_keeps_its_earlier_run(fake_run_store, groups):
    initial = document(run("slot-0"), run("slot-1", grade=None))
    stored(groups, initial)
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.run("job-1")
    lose(fake_run_store, "job-0")

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    (earlier,) = settled["earlier_runs"]
    assert (
        settled["status"],
        settled["samples"],
        earlier["affinity"],
        [(sample["actor"]["job_id"], sample["actor"]["state"]) for sample in earlier["samples"]],
        earlier["samples"][0]["actor"]["result"],
    ) == ("running", initial["samples"], "group", [("job-0", "failed"), ("job-1", "running")], DATA_LOST)


def test_a_group_that_runs_again_stops_no_job_of_its_earlier_run(fake_run_store, groups):
    """The Run Store keeps every job of a lost host. A job of the run on another host runs to its end."""
    stored(groups, document(run("slot-0", grade=None), run("slot-1", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.run("job-1")
    lose(fake_run_store, "job-0")

    coordinator.tick()
    coordinator.tick()

    assert (("POST", "stop") in fake_run_store.requests, fake_run_store.job("job-1")["state"]) == (
        False,
        "running",
    )


def test_a_grade_that_lost_its_data_runs_the_group_again(fake_run_store, groups):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()
    lose(fake_run_store, "job-1")

    coordinator.tick()

    assert groups.get(GROUP)["document"]["affinity"] == "group-r1"


def test_a_group_whose_third_run_loses_a_job_fails_with_the_reason(fake_run_store, groups):
    """A group runs again at most 2 times. Its failed group is terminal, so no tick makes a fourth run."""
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    lose(fake_run_store, "job-0")
    coordinator.tick()
    coordinator.tick()
    lose(fake_run_store, "job-1")
    coordinator.tick()
    coordinator.tick()
    lose(fake_run_store, "job-2")

    coordinator.tick()
    coordinator.tick()

    row = groups.get(GROUP)
    assert (
        row["terminal"],
        row["document"]["status"],
        row["document"]["error"],
        fake_run_store.job_ids("rollout"),
    ) == (
        True,
        "failed",
        "A host that held the group's jobs was lost in each of its 3 runs: group, group-r1, group-r2."
        " A group runs again at most 2 times.",
        ["job-0", "job-1", "job-2"],
    )


def test_a_group_runs_again_under_the_key_of_its_first_run_and_the_number_of_the_run(fake_run_store, groups):
    """ADR-0007: the n-th run again of a search whose key is ``search`` has the key ``search-r<n>``."""
    stored(groups, document(run("slot-0", grade=None), affinity="search"))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    lose(fake_run_store, "job-0")
    coordinator.tick()
    coordinator.tick()
    lose(fake_run_store, "job-1")

    coordinator.tick()
    coordinator.tick()

    assert [body["affinity"] for body in fake_run_store.submitted("rollout")] == [
        "search",
        "search-r1",
        "search-r2",
    ]


def test_a_coordinator_that_runs_no_group_again_fails_a_group_whose_job_lost_its_data(fake_run_store, groups):
    """Manual evaluation runs its search again itself, so the coordinator submits no job of a new run.

    The group does not wait for its other jobs, which fail with the same host."""
    stored(groups, document(run("slot-0", grade=None), run("slot-1", grade=None), affinity="search"))
    coordinator = coordinator_for(fake_run_store, groups, runs_again=False)
    coordinator.tick()
    lose(fake_run_store, "job-0")

    coordinator.tick()
    coordinator.tick()

    row = groups.get(GROUP)
    assert (
        row["terminal"],
        row["document"]["status"],
        row["document"]["affinity"],
        row["document"].get("error"),
        fake_run_store.job_ids("rollout"),
    ) == (
        True,
        "failed",
        "search",
        "A host that held the jobs of the key search was lost, and this coordinator runs no group again.",
        ["job-0", "job-1"],
    )


def test_a_cancelled_group_whose_job_lost_its_data_does_not_run_again(fake_run_store, groups):
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    groups.release(GROUP)
    lose(fake_run_store, "job-0")

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (settled["status"], settled["affinity"], fake_run_store.job_ids("rollout")) == (
        "cancelled",
        "group",
        ["job-0"],
    )


def test_a_review_sample_is_an_actor_whose_result_the_group_keeps(fake_run_store, groups):
    review = {
        "sample_slot_id": "review-1",
        "review": {"profile": "review", "spec": {**REVIEW, "parents": [PARENT]}},
    }
    stored(groups, document(review))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish("job-0", {"status": "completed", "points": []})

    coordinator.tick()

    settled = groups.get(GROUP)["document"]
    assert (settled["status"], settled["samples"][0]["actor"]["result"]) == (
        "completed",
        {"status": "completed", "points": []},
    )


@pytest.mark.parametrize("sample", [run("slot-0", grade=None), BRANCH], ids=["run", "branch"])
def test_a_submission_whose_reply_was_lost_is_sent_again_after_a_restart_and_makes_one_job(
    fake_run_store, groups, tmp_path, sample
):
    """The key of a stage is the same on every try, so the Run Store answers the retry with the first job."""
    parent_rollout(fake_run_store)
    stored(groups, document(sample))
    fake_run_store.lose_next_reply()
    coordinator_for(fake_run_store, groups).tick()

    coordinator_for(fake_run_store, FileGroups(tmp_path / "groups")).tick()

    actor = stage(groups, "slot-0", "actor")
    assert (actor["job_id"], actor["state"], fake_run_store.job_ids("rollout")) == (
        "job-1",
        "queued",
        ["job-0", "job-1"],
    )


@pytest.mark.parametrize("status", [400, 404, 409, 422])
def test_a_submission_that_the_run_store_refuses_fails_its_stage_with_the_answer(
    fake_run_store, groups, status
):
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    fake_run_store.fail_next("POST", "jobs", status=status)

    coordinator.tick()
    coordinator.tick()

    actor = stage(groups, "slot-0", "actor")
    assert (actor["state"], actor["error"], fake_run_store.requests) == (
        "failed",
        REFUSAL,
        [("POST", "jobs")],
    )


@pytest.mark.parametrize("status", [503, None], ids=["server error", "unreachable"])
def test_a_submission_that_did_not_reach_a_verdict_is_sent_again_on_the_next_tick(
    fake_run_store, groups, status
):
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    fake_run_store.fail_next("POST", "jobs", status=status)

    coordinator.tick()
    coordinator.tick()

    actor = stage(groups, "slot-0", "actor")
    assert (actor["state"], actor["job_id"], fake_run_store.requests.count(("POST", "jobs"))) == (
        "queued",
        "job-0",
        2,
    )


def test_a_refusal_after_a_lost_reply_leaves_the_stage_waiting_for_the_job_that_may_exist(
    fake_run_store, groups
):
    """The first try may have made the job. A retry can still be refused: the point of a branch can expire."""
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    fake_run_store.lose_next_reply()
    coordinator.tick()
    fake_run_store.fail_next("POST", "jobs", status=409)

    coordinator.tick()

    actor = stage(groups, "slot-0", "actor")
    assert (actor["state"], actor["job_id"], actor["submission_uncertain"]) == ("submitting", None, True)


def test_a_failed_poll_is_the_stages_last_error_until_a_poll_succeeds(fake_run_store, groups):
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.fail_next("GET", "job")

    coordinator.tick()
    after_the_failure = stage(groups, "slot-0", "actor")
    coordinator.tick()

    assert (after_the_failure["last_error"], "last_error" in stage(groups, "slot-0", "actor")) == (
        "the fake Run Store is unreachable",
        False,
    )


def test_a_failed_run_store_call_of_one_group_does_not_stop_the_other_group(fake_run_store, groups):
    stored(groups, document(run("slot-0", grade=None), group_id="first"))
    stored(groups, document(run("slot-0", grade=None), group_id="second"))
    fake_run_store.fail_next("POST", "jobs", status=503)

    coordinator_for(fake_run_store, groups).tick()

    first = stage(groups, "slot-0", "actor", group_id="first")
    second = stage(groups, "slot-0", "actor", group_id="second")
    assert {first["job_id"], second["job_id"]} == {None, "job-0"}, "the group that was refused, and the other"


def test_a_group_cancelled_before_its_first_tick_submits_nothing_and_is_cancelled(fake_run_store, groups):
    stored(groups, document(run("slot-0")))
    groups.release(GROUP)

    coordinator_for(fake_run_store, groups).tick()

    settled = groups.get(GROUP)["document"]
    sample = settled["samples"][0]
    assert (
        settled["status"],
        sample["actor"]["state"],
        sample["grade"]["state"],
        fake_run_store.requests,
    ) == (
        "cancelled",
        "cancelled",
        "skipped",
        [],
    )


def test_a_cancelled_group_stops_its_jobs_and_is_cancelled_once_the_running_one_ends(fake_run_store, groups):
    """A queued job is cancelled at once. A running one is asked to stop, and the group waits for its end."""
    stored(groups, document(run("slot-0", grade=None), run("slot-1", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.run("job-0")
    groups.release(GROUP)

    coordinator.tick()
    waiting = groups.get(GROUP)["document"]["ready"]
    fake_run_store.finish("job-0", exported())
    coordinator.tick()

    assert (
        fake_run_store.job("job-0")["stop_requested_at"],
        fake_run_store.job("job-1")["state"],
        waiting,
        groups.get(GROUP)["document"]["status"],
    ) == (STOP_REQUESTED_AT, "cancelled", False, "cancelled")


def test_a_cancelled_group_does_not_grade_an_actor_that_ends_after_the_cancel(fake_run_store, groups):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.run("job-0")
    groups.release(GROUP)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())

    coordinator.tick()

    assert (stage(groups, "slot-0", "grade")["state"], fake_run_store.job_ids("grade")) == ("cancelled", [])


def test_a_cancel_that_arrives_while_a_grade_is_prepared_submits_no_grade(fake_run_store, groups):
    """The cancel arrives while the coordinator reads the actor's recovery points."""
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    fake_run_store.before_next("GET", "recovery-points", partial(groups.release, GROUP))

    coordinator.tick()

    assert (stage(groups, "slot-0", "grade")["state"], fake_run_store.job_ids("grade")) == ("cancelled", [])


def test_a_cancelled_group_does_not_post_again_a_submission_whose_reply_was_lost(fake_run_store, groups):
    """The first try may have made no job. A second one would make a job after the cancel."""
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    fake_run_store.lose_next_reply()
    coordinator.tick()
    groups.release(GROUP)

    coordinator.tick()

    actor = stage(groups, "slot-0", "actor")
    assert (
        fake_run_store.requests,
        actor.get("last_error"),
        actor.get("idempotency_key"),
        groups.get(GROUP)["document"]["ready"],
    ) == (
        [("POST", "jobs")],
        "Submission acknowledgement unknown; inspect Run Store using the idempotency key",
        fake_run_store.job("job-0")["idempotency_key"],
        False,
    )


def test_a_reconciled_submission_is_followed_and_the_cancel_stops_its_job(fake_run_store, groups):
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    fake_run_store.lose_next_reply()
    coordinator.tick()
    groups.release(GROUP)
    coordinator.tick()

    coordinator.reconcile_submission(GROUP, sample_id="slot-0", phase="actor", job_id="job-0")
    coordinator.tick()

    assert (
        fake_run_store.job("job-0")["state"],
        groups.get(GROUP)["document"]["status"],
        fake_run_store.requests.count(("POST", "jobs")),
    ) == ("cancelled", "cancelled", 1)


def test_reconciling_refuses_a_job_submitted_under_another_key(fake_run_store, groups):
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    fake_run_store.lose_next_reply()
    coordinator.tick()
    other = fake_run_store.client.submit(JobSpec.from_dict({"kind": "rollout", **RUN}), "another-key")

    with pytest.raises(ValueError, match="Job does not match this submission's idempotency key"):
        coordinator.reconcile_submission(GROUP, sample_id="slot-0", phase="actor", job_id=other)


@pytest.mark.parametrize(
    "group_id, sample_id, phase, error, message",
    [
        (GROUP, "slot-0", "verify", ValueError, "phase must be actor, grade or review"),
        (GROUP, "slot-9", "actor", KeyError, "slot-9"),
        ("other", "slot-0", "actor", KeyError, "other"),
        (GROUP, "slot-0", "actor", ValueError, "Only an unacknowledged submission can be reconciled"),
        (GROUP, "slot-0", "grade", ValueError, "Only an unacknowledged submission can be reconciled"),
    ],
    ids=["unknown phase", "unknown sample", "unknown group", "acknowledged", "never submitted"],
)
def test_reconciling_refuses_a_stage_that_is_not_an_unacknowledged_submission(
    fake_run_store, groups, group_id, sample_id, phase, error, message
):
    stored(groups, document(run("slot-0")))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()

    with pytest.raises(error, match=message):
        coordinator.reconcile_submission(group_id, sample_id=sample_id, phase=phase, job_id="job-0")


@pytest.mark.parametrize("now, cancel_requested", [(999.0, False), (1000.0, True)])
def test_a_group_is_cancelled_from_its_deadline_on(fake_run_store, groups, clock, now, cancel_requested):
    stored(groups, document(run("slot-0", grade=None), deadline_at=1000.0))
    clock.now = now

    coordinator_for(fake_run_store, groups).tick()

    assert groups.get(GROUP)["cancel_requested"] is cancel_requested


@pytest.mark.parametrize(
    "running_s, state, error, stop_requested_at",
    [
        (599.0, "running", None, None),
        (600.0, "expired", "Grading did not settle within 600 s", STOP_REQUESTED_AT),
    ],
)
def test_a_grade_that_runs_for_its_wall_time_is_expired_and_stopped(
    fake_run_store, groups, clock, running_s, state, error, stop_requested_at
):
    stored(groups, document(run("slot-0"), grade_wall_seconds=600.0))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()
    fake_run_store.run("job-1")
    coordinator.tick()
    clock.now += running_s

    coordinator.tick()

    grade = stage(groups, "slot-0", "grade")
    assert (grade["state"], grade["error"], fake_run_store.job("job-1")["stop_requested_at"]) == (
        state,
        error,
        stop_requested_at,
    )


def test_the_wall_time_of_a_grade_starts_when_it_leaves_the_queue(fake_run_store, groups, clock):
    stored(groups, document(run("slot-0"), grade_wall_seconds=600.0))
    coordinator = coordinator_for(fake_run_store, groups)
    coordinator.tick()
    fake_run_store.finish_rollout("job-0", exported())
    coordinator.tick()
    clock.now += 5000.0
    fake_run_store.run("job-1")
    coordinator.tick()
    clock.now += 599.0

    coordinator.tick()

    assert stage(groups, "slot-0", "grade")["state"] == "running"


def test_a_tick_of_a_stopping_coordinator_submits_nothing(fake_run_store, groups):
    stored(groups, document(run("slot-0", grade=None)))
    stopping = threading.Event()
    stopping.set()

    coordinator_for(fake_run_store, groups, stopping=stopping).tick()

    assert fake_run_store.requests == []


def test_a_tick_returns_at_once_while_another_tick_runs(fake_run_store, groups, gate):
    stored(groups, document(run("slot-0", grade=None)))
    coordinator = coordinator_for(fake_run_store, groups)
    fake_run_store.before_next("POST", "jobs", gate.hold)
    first = threading.Thread(target=coordinator.tick)
    first.start()
    assert gate.entered.wait(10), "the first tick did not submit"

    coordinator.tick()
    gate.open()
    first.join(10)

    assert fake_run_store.requests == [("POST", "jobs"), ("GET", "job")]

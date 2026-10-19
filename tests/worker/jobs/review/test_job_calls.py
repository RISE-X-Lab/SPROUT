"""The calls of a review side by side: which start when, in what order the results come, and failures.

Each test orders the calls through what the review exposes: the transport that it makes when a call
starts, the stop that it gives each transport, and the thread that each call's requests run on. A call
that ends puts its result on the review's queue, so ``ReviewModel.join`` orders two results.
"""

from __future__ import annotations

import json
import logging
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from sprout.worker.jobs.review import job as review_job
from sprout.worker.jobs.review.job import run_review
from tests.worker.jobs.review.plain_prompts import prompts_record
from tests.worker.jobs.review.review_model import ReviewModel, failed_roots, review_spec, stage_of

ACROSS = {"points": [("root-1", 2), ("root-1", 3)]}
PER_ATTEMPT = {"points-1": [("root-1", 2), ("root-1", 3)], "points-2": [("root-2", 2), ("root-2", 3)]}


def two_roots(tmp_path: Path, **changes: Any) -> dict[str, Any]:
    root = tmp_path / "artifacts"
    return review_spec(root, failed_roots(root, 2), **changes)


def plans(directory: Path) -> dict[str, Any]:
    return {path.name: json.loads(path.read_text()) for path in sorted(directory.glob("plan-*.json"))}


@pytest.mark.parametrize(
    "per_attempt, points, calls",
    [
        (
            False,
            ACROSS,
            ["analyst-1", "analyst-2", "points"]
            + [f"point{p}-candidate{s}" for p in (1, 2) for s in (1, 2, 3)],
        ),
        (
            True,
            PER_ATTEMPT,
            ["analyst-1", "analyst-2", "points-1", "points-2"]
            + [f"point{a}.{p}-candidate{s}" for a in (1, 2) for p in (1, 2) for s in (1, 2, 3)],
        ),
    ],
    ids=["across-attempts", "points-per-attempt"],
)
def test_with_one_slot_the_calls_run_one_at_a_time_in_call_order(tmp_path, per_attempt, points, calls):
    """Every analysis, then every point review, then the samples point by point."""
    model = ReviewModel(points=points)

    run_review(
        two_roots(tmp_path, points_per_attempt=per_attempt, max_concurrent_requests=1),
        tmp_path / "review",
        transport_factory=model.factory,
    )

    assert (model.calls, model.most_in_flight) == (calls, 1)


def test_a_call_starts_only_when_a_slot_is_free(tmp_path):
    """Two slots and three analyses: the third starts after one of the first two has returned."""
    root = tmp_path / "artifacts"
    spec = review_spec(
        root,
        failed_roots(root, 3),
        guidance="none",
        candidates=1,
        points_per_attempt=True,
        points=1,
        max_concurrent_requests=2,
    )
    first_two = threading.Barrier(2, timeout=30)

    def hold(call: str) -> None:
        if call in ("analyst-1", "analyst-2"):
            first_two.wait()
            # A review that ignored its slots would start the third analysis now. It gets 0.5 s to show it.
            model.made_transport("analyst-3").wait(0.5)

    model = ReviewModel(points={f"points-{n}": [(f"root-{n}", 2)] for n in (1, 2, 3)}, hold=hold)

    run_review(spec, tmp_path / "review", transport_factory=model.factory)

    first_return = min(model.log.index(("finished", "analyst-1")), model.log.index(("finished", "analyst-2")))
    assert (model.log.index(("made", "analyst-3")) > first_return, model.most_in_flight) == (True, 2)


def test_across_attempts_the_point_review_waits_for_every_analysis_and_then_every_sample_runs_at_once(
    tmp_path,
):
    analyses, samples = threading.Barrier(2, timeout=30), threading.Barrier(6, timeout=30)

    def hold(call: str) -> None:
        if stage_of(call) == "analyst":
            analyses.wait()
        elif stage_of(call) == "points":
            assert model.finished("analyst-1").is_set() and model.finished("analyst-2").is_set()
        else:
            samples.wait()

    model = ReviewModel(points=ACROSS, hold=hold)

    result = run_review(two_roots(tmp_path), tmp_path / "review", transport_factory=model.factory)

    assert (model.most_in_flight, result["review_requests"]) == (6, 2 + 1 + 6)


def test_with_points_per_attempt_each_failed_attempt_is_a_pipeline_and_the_pipelines_run_side_by_side(
    tmp_path,
):
    """Root 2's point review starts, and its samples run, while root 1's analysis is still out."""

    def hold(call: str) -> None:
        if call == "analyst-1":
            assert model.finished("point2.1-candidate1").wait(30)

    model = ReviewModel(points={"points-1": [("root-1", 2)], "points-2": [("root-2", 3)]}, hold=hold)

    result = run_review(
        two_roots(tmp_path, points=1, points_per_attempt=True, candidates=1),
        tmp_path / "review",
        transport_factory=model.factory,
    )

    assert (model.answered[-1], [p["base"] for p in result["points"]], list(result["reports"])) == (
        "point1.1-candidate1",
        ["root-1", "root-2"],
        ["root-1", "root-2"],
    )


@pytest.mark.parametrize(
    "per_attempt, points",
    [(False, ACROSS), (True, PER_ATTEMPT)],
    ids=["across-attempts", "points-per-attempt"],
)
def test_the_review_is_the_same_whatever_order_its_calls_return_in(tmp_path, per_attempt, points):
    """Root 2's analysis returns first, and at each point the second sample, then the first, then the third.

    The first and the third samples agree, so a merge in the order of return would put the second first."""

    def propose(call: str) -> str:
        return f"ls {call}" if call.endswith("candidate2") else "pytest -q"

    spec = two_roots(tmp_path, points_per_attempt=per_attempt)
    serial = ReviewModel(points=points, proposals=propose)
    one = run_review(
        {**spec, "max_concurrent_requests": 1}, tmp_path / "one", transport_factory=serial.factory
    )

    def hold(call: str) -> None:
        if call == "analyst-1":
            model.join("analyst-2")
        elif call.endswith("candidate1"):
            model.join(call[:-1] + "2")
        elif call.endswith("candidate3"):
            model.join(call[:-1] + "1")

    model = ReviewModel(points=points, proposals=propose, hold=hold)
    eight = run_review(
        {**spec, "max_concurrent_requests": 8}, tmp_path / "eight", transport_factory=model.factory
    )

    assert (json.dumps(eight), plans(tmp_path / "eight")) == (json.dumps(one), plans(tmp_path / "one"))
    assert model.answered.index("analyst-2") < model.answered.index("analyst-1")
    assert [[c["multiplicity"] for c in p["candidates"]] for p in eight["points"]] == [[2, 1]] * len(
        eight["points"]
    )


def test_a_failed_review_raises_the_error_of_its_first_call_in_call_order_with_the_others_noted(tmp_path):
    """Root 2's analysis fails first. Root 1's analysis comes before it, so it is not stopped: its correction
    is still sent, and its error is the review's. Root 3's analysis comes after, so its correction is not."""
    root = tmp_path / "artifacts"
    spec = review_spec(root, failed_roots(root, 3), points=1, points_per_attempt=True, candidates=1)

    def hold(call: str) -> None:
        if call == "analyst-1" and len(model.prompts_of(call)) == 1:
            assert model.stop_of("analyst-3").wait(30), "the review stopped root 3 once root 2 failed"
        elif call == "analyst-2":
            assert model.arrived("analyst-3").wait(30)
        elif call == "analyst-3":
            assert model.stop_of("analyst-3").wait(30)

    model = ReviewModel(
        points={},
        hold=hold,
        fail={"analyst-2": RuntimeError("root 2 refused")},
        replies={
            "analyst-1": ["not a report", RuntimeError("root 1 refused")],
            "analyst-3": ["not a report"],
        },
    )

    with pytest.raises(RuntimeError, match="^root 1 refused\n") as raised:
        run_review(spec, tmp_path / "review", transport_factory=model.factory)

    assert raised.value.__notes__ == [
        "in review call analyst-1",
        "review call analyst-2 failed too: RuntimeError: root 2 refused",
    ]
    assert ([len(model.prompts_of(f"analyst-{n}")) for n in (1, 2, 3)], model.made) == (
        [2, 1, 1],
        ["analyst-1", "analyst-2", "analyst-3"],
    )


def test_a_running_call_after_a_failure_asks_for_nothing_more_and_its_plan_says_why(tmp_path):
    def hold(call: str) -> None:
        if call == "analyst-1":
            assert model.arrived("points-2").wait(30)
        elif call == "points-2":
            assert model.stop_of("points-2").wait(30)

    model = ReviewModel(
        points={},
        hold=hold,
        fail={"analyst-1": RuntimeError("root 1 refused")},
        replies={"points-2": ["not a plan"]},
    )

    with pytest.raises(RuntimeError, match="^root 1 refused\n") as raised:
        run_review(
            two_roots(tmp_path, points=1, points_per_attempt=True, candidates=1),
            tmp_path / "review",
            transport_factory=model.factory,
        )

    plan = json.loads((tmp_path / "review" / "plan-points-2.json").read_text())
    assert (raised.value.__notes__, model.calls.count("points-2"), model.made) == (
        ["in review call analyst-1"],
        1,
        ["analyst-1", "analyst-2", "points-2"],
    )
    assert (
        plan["validation_error"]
        == "reviewer failed: review call points-2 stopped: review call analyst-1 failed"
    )


def unwritable_plan(review: Path, call: str) -> None:
    """Turn the plan file of ``call`` into a directory, so that the call's next save of its plan raises."""
    (review / f"plan-{call}.json").unlink()
    (review / f"plan-{call}.json").mkdir()


def test_an_earlier_stage_outranks_an_earlier_attempt(tmp_path):
    """Root 1's point review fails before root 2's analysis. Each analysis comes before each point review."""
    review = tmp_path / "review"

    def hold(call: str) -> None:
        if call == "points-1":
            unwritable_plan(review, "points-1")
        elif call == "analyst-2":
            model.join("points-1")

    model = ReviewModel(
        points={"points-1": [("root-1", 2)]}, hold=hold, fail={"analyst-2": RuntimeError("root 2 refused")}
    )

    with pytest.raises(RuntimeError, match="^root 2 refused\n") as raised:
        run_review(
            two_roots(tmp_path, points=1, points_per_attempt=True, candidates=1),
            review,
            transport_factory=model.factory,
        )

    assert raised.value.__notes__ == [
        "in review call analyst-2",
        "review call points-1 failed too: IsADirectoryError: "
        f"[Errno 21] Is a directory: '{review / 'plan-points-1.json'}'",
    ]


def test_a_call_before_the_first_failure_still_starts_and_its_error_wins(tmp_path):
    """Root 2's point review fails while root 1's analysis is out. Root 1's point review comes before it."""
    review = tmp_path / "review"

    def hold(call: str) -> None:
        if call.startswith("points-"):
            unwritable_plan(review, call)
        elif call == "analyst-1":
            model.join("points-2")

    model = ReviewModel(points={"points-1": [("root-1", 2)], "points-2": [("root-2", 2)]}, hold=hold)

    with pytest.raises(IsADirectoryError) as raised:
        run_review(
            two_roots(tmp_path, points=1, points_per_attempt=True, candidates=1),
            review,
            transport_factory=model.factory,
        )

    assert (str(raised.value), model.made) == (
        f"[Errno 21] Is a directory: '{review / 'plan-points-1.json'}'",
        ["analyst-1", "analyst-2", "points-2", "points-1"],
    )
    assert raised.value.__notes__ == [
        "in review call points-1",
        "review call points-2 failed too: IsADirectoryError: "
        f"[Errno 21] Is a directory: '{review / 'plan-points-2.json'}'",
    ]


def test_a_sample_that_fails_fails_the_review(tmp_path):
    """No result comes back without every sample of every point."""
    review = tmp_path / "review"

    def hold(call: str) -> None:
        if call == "point1-candidate2":
            unwritable_plan(review, call)

    model = ReviewModel(points={"points": [("root-1", 2)]}, hold=hold)

    with pytest.raises(IsADirectoryError) as raised:
        run_review(two_roots(tmp_path, points=1, candidates=3), review, transport_factory=model.factory)

    assert raised.value.__notes__ == ["in review call point1-candidate2"]


def test_with_one_slot_no_call_starts_after_a_failure(tmp_path):
    model = ReviewModel(points={}, fail={"analyst-1": RuntimeError("root 1 refused")})

    with pytest.raises(RuntimeError, match="^root 1 refused\n") as raised:
        run_review(
            two_roots(tmp_path, points_per_attempt=True, max_concurrent_requests=1),
            tmp_path / "review",
            transport_factory=model.factory,
        )

    assert (raised.value.__notes__, model.made) == (["in review call analyst-1"], ["analyst-1"])


def test_a_transport_that_cannot_be_made_fails_its_call_and_no_later_call_starts(tmp_path):
    model = ReviewModel(points={"points-1": [("root-1", 2)]})

    def factory(directory: Path, **settings: Any) -> Any:
        if Path(directory).name == "analyst-2":
            raise ValueError("no transport for root 2")
        return model.factory(directory, **settings)

    with pytest.raises(ValueError, match="^no transport for root 2\n"):
        run_review(
            two_roots(tmp_path, points=1, points_per_attempt=True, candidates=1),
            tmp_path / "review",
            transport_factory=factory,
        )

    assert (model.made, model.calls, plans(tmp_path / "review")) == (["analyst-1"], ["analyst-1"], {})


def test_an_interrupted_review_raises_at_once_and_its_running_call_asks_for_nothing_more(tmp_path):
    """The worker stops a review with SIGTERM, raised as KeyboardInterrupt.

    A request in flight can last ``timeout_s``."""
    release = threading.Event()
    model = ReviewModel(
        points={}, hold=lambda call: release.wait(30), replies={"analyst-1": ["not a report"]}
    )

    def factory(directory: Path, **settings: Any) -> Any:
        if Path(directory).name == "analyst-2":
            assert model.arrived("analyst-1").wait(30)
            raise KeyboardInterrupt("Worker requested shutdown")
        return model.factory(directory, **settings)

    with pytest.raises(KeyboardInterrupt):
        run_review(
            two_roots(tmp_path, points_per_attempt=True), tmp_path / "review", transport_factory=factory
        )

    returned_before_the_call = not model.finished("analyst-1").is_set()
    release.set()
    model.join("analyst-1")
    assert (returned_before_the_call, model.calls) == (True, ["analyst-1"])


def test_the_plan_of_a_point_review_cut_off_by_an_interrupt_says_why(tmp_path):
    release = threading.Event()

    def hold(call: str) -> None:
        if call == "analyst-2":
            assert model.arrived("points-1").wait(30)
        elif call == "points-1":
            assert release.wait(30)

    model = ReviewModel(points={}, hold=hold, replies={"points-1": ["not a plan"]})

    def factory(directory: Path, **settings: Any) -> Any:
        if Path(directory).name == "points-2":
            raise KeyboardInterrupt("Worker requested shutdown")
        return model.factory(directory, **settings)

    with pytest.raises(KeyboardInterrupt):
        run_review(
            two_roots(tmp_path, points=1, points_per_attempt=True),
            tmp_path / "review",
            transport_factory=factory,
        )

    release.set()
    model.join("points-1")
    plan = json.loads((tmp_path / "review" / "plan-points-1.json").read_text())
    assert (plan["validation_error"], model.calls.count("points-1")) == (
        "reviewer failed: review call points-1 stopped: the review was interrupted",
        1,
    )


def test_an_interrupt_keeps_the_failures_already_known_on_it_and_in_the_log(tmp_path, caplog):
    """Root 2's point review has failed when root 1's analysis returns and the worker's SIGTERM lands."""
    review = tmp_path / "review"

    def hold(call: str) -> None:
        if call == "points-2":
            unwritable_plan(review, "points-2")
        elif call == "analyst-1":
            model.join("points-2")

    model = ReviewModel(points={"points-2": [("root-2", 2)]}, hold=hold)

    def factory(directory: Path, **settings: Any) -> Any:
        if Path(directory).name == "points-1":
            raise KeyboardInterrupt("Worker requested shutdown")
        return model.factory(directory, **settings)

    with (
        caplog.at_level(logging.ERROR, logger=review_job.__name__),
        pytest.raises(KeyboardInterrupt) as raised,
    ):
        run_review(
            two_roots(tmp_path, points=1, points_per_attempt=True, candidates=1),
            review,
            transport_factory=factory,
        )

    failure = f"IsADirectoryError: [Errno 21] Is a directory: '{review / 'plan-points-2.json'}'"
    assert raised.value.__notes__ == [f"review call points-2 had failed: {failure}"]
    (logged,) = [record for record in caplog.records if record.name == review_job.__name__]
    assert (logged.getMessage(), type(logged.exc_info[1]).__name__) == (
        "Review call points-2 had failed when the review was cut off",
        "IsADirectoryError",
    )


class Lingering(queue.SimpleQueue):  # type: ignore[type-arg]
    """A queue on which each call's thread stays alive a moment after it puts its result."""

    def put(self, item: Any, block: bool = True, timeout: float | None = None) -> None:
        super().put(item, block, timeout)
        threading.Event().wait(0.1)


def test_no_call_thread_outlives_the_review(tmp_path, monkeypatch):
    """The worker's process exits once the review returns, and must not cut off a call that still runs."""
    root = tmp_path / "artifacts"
    monkeypatch.setattr(review_job, "queue", SimpleNamespace(SimpleQueue=Lingering))
    model = ReviewModel(points={"points": [("root-1", 2)]})

    run_review(
        review_spec(root, failed_roots(root, 1), guidance="none", candidates=1),
        tmp_path / "review",
        transport_factory=model.factory,
    )

    assert [call for call, thread in model.threads.items() if thread.is_alive()] == []


class Shared:
    """One transport that the factory hands to every call. It answers by the template of each prompt."""

    def __init__(self) -> None:
        self.sequence = 0

    def __call__(self, model: str, prompt: str) -> str:
        self.sequence += 1
        if prompt.startswith("PROBLEM "):
            return json.dumps({"failure_reason": "r", "lesson": "l", "salvage": "s", "branch_candidates": []})
        branches = [{"name": "p", "base": "root-1", "branch_step": 2, "why": "w"}]
        return "```branch-plan\n" + json.dumps({"synthesis": "s", "branches": branches}) + "\n```"


def test_a_transport_shared_by_every_call_counts_its_requests_once(tmp_path):

    root = tmp_path / "artifacts"
    shared = Shared()
    spec = review_spec(
        root, failed_roots(root, 1), guidance="none", candidates=1, prompts=prompts_record(tmp_path)
    )

    result = run_review(spec, tmp_path / "review", transport_factory=lambda directory, **settings: shared)

    assert (result["review_requests"], shared.sequence) == (2, 2)

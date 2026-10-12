"""A Miles search request: roots, then a review of the failed ones, then branches where it points.

The Run Store is the fake peer of the driver tests; the review's result is
what ``runstore.review`` returns, supplied here by hand. How the roots came
out decides what follows (``rl_driver.outcome``): the full review and its
repairs when every root failed (FF), a review for points and unaided students
alone when the roots split or one is missing beside failures (TF, FF_partial),
nothing when no graded root failed or none was graded.
"""

from copy import deepcopy
import json

import pytest

import sprout.rl_driver.driver as driver_module
from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.messages import MessageAdapter, MessageRequest, internal_id
from sprout.rl_driver.outcome import search_outcome, search_shape
from sprout.runstore.specs import JobSpec
from tests.harness.test_assistant_turn import assistant_turn
from tests.rl_driver.test_driver import peer as peer
from tests.rl_driver.test_message_rollout import config, exported, request

_ = peer

#: What a grade says about the task's hidden tests beside its verdict (``grading.patch_facts``).
HIDDEN = {"fail_to_pass": ["tests/test_hidden.py::test_secret_behaviour"], "test_patch_paths": ["tests/test_hidden.py"]}


def search_request(*, roots=1, points=1, candidates=1, student=1, per_candidate=1, tf=2, rounds=1):
    body = request()
    # Room for every root to have failed, or for all but one to have and their tf students.
    slots = roots + points * max(roots * (student + candidates * per_candidate * rounds), (roots - 1) * tf)
    body["sample_slots"] = [{"sample_slot_id": f"slot-{i}", "sample_index": i} for i in range(slots)]
    body.update(max_samples=slots, minimum_returned_samples=roots, max_turns=10)
    body["search"] = {"roots": roots, "points": points, "candidates": candidates,
                      "student_continuations": student, "candidate_continuations": per_candidate,
                      "tf_student_continuations": tf, "repair_rounds": rounds,
                      "review_seconds": 600.0, "wall_time_seconds": 900.0}
    return body


def search_config(body):
    return {**config(body), "review": {"profile": "review", "temperature": 0.7, "max_output_tokens": 4096}}


def review_result(root_job, *, step=2, candidates=1):
    """One point in ``root_job`` with ``candidates`` repair turns; the second names a hidden test file."""
    return {"status": "completed", "guidance": "assistant-turn", "analysed": ["root"], "review_requests": 3,
            "points": [{"base": "root", "job_id": root_job, "attempt_id": f"attempt-{root_job}", "step": step,
                        "snapshot_id": "AB", "why": "before the edit", "samples": 4,
                        "candidates": [{"assistant_turn": assistant_turn("pytest -q", identifier=f"repair-{i}"),
                                        "why": "check first", "multiplicity": 2 if i == 1 else 1,
                                        "leak_terms": [] if i == 1 else ["tests/test_hidden.py"]}
                                       for i in range(1, candidates + 1)]}]}


def finish_actor(queue, job_id, *, points=True):
    queue.finish(job_id)
    queue.jobs[job_id]["result"].update(exported())
    if points:
        queue.final_point(job_id)


def finish_branch(queue, job_id, root):
    finish_actor(queue, job_id)
    # a branch's export says what it restored and inserted
    queue.jobs[job_id]["result"].update(training_origin={"job_id": root, "point_id": "old", "tool_depth": 2},
                                        training_provenance={"prefix_messages": 1, "inserted_messages": []})


def jobs_of(queue, kind):
    return [job_id for job_id, job in queue.jobs.items() if job["kind"] == kind]


def review_submission(queue):
    """The body of the one review job the driver submitted."""
    (body,) = [row[2] for row in queue.requests
               if row[0] == "POST" and row[1] == "/v1/jobs" and row[2]["kind"] == "review"]
    return body


def settle(queue, grade_id, verdict):
    """Settle a root's grade: resolved or not (True, False), or ``"failed"``, a grade that could not be made.

    ``"unexported"`` is a root graded unresolved whose actor exported nothing to train on.
    """
    if verdict == "failed":
        queue.jobs[grade_id].update(state="failed", active_attempt=f"attempt-{grade_id}",
                                    error="sandbox did not start",
                                    result={"status": "error", "failure_kind": "infrastructure"})
    else:
        queue.finish(grade_id, resolved=verdict is True, **HIDDEN)


def run_roots(queue, driver, *verdicts):
    """Run a search's roots through their actors and grades, root i settled as ``verdicts[i]`` (``settle``)."""
    driver.tick()
    roots = jobs_of(queue, "rollout")
    for job_id, verdict in zip(roots, verdicts, strict=True):
        finish_actor(queue, job_id)
        if verdict == "unexported":
            queue.jobs[job_id]["result"]["training_export_error"] = "Native history has an incomplete final record"
    driver.tick()
    for grade_id, verdict in zip(jobs_of(queue, "grade"), verdicts, strict=True):
        settle(queue, grade_id, verdict)
    driver.tick()
    return roots


def searching(tmp_path, peer, body):
    _, client = peer
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, search_config(body))
    adapter.submit(body)
    return driver, adapter


@pytest.mark.parametrize("counts, label, shape", [
    ((2, 2, 2), "TT", "none"),
    ((2, 1, 1), "TT_partial", "none"),
    ((2, 0, 0), "ungraded", "none"),
    ((2, 2, 1), "TF", "points_only"),
    ((3, 2, 1), "TF", "points_only"),   # a split beside a missing root is still a split
    ((2, 1, 0), "FF_partial", "points_only"),
    ((2, 2, 0), "FF", "full"),
    ((1, 1, 0), "FF", "full"),
])
def test_the_roots_label_decides_what_the_search_samples(counts, label, shape):
    assert search_outcome(*counts) == label
    assert search_shape(label, {"tf_student_continuations": 2}) == shape
    # With no unaided student to run from its points, a points-only search samples nothing.
    assert search_shape(label, {"tf_student_continuations": 0}) == ("none" if shape == "points_only" else shape)


@pytest.mark.parametrize("counts", [(1, 2, 0), (2, 1, 2), (2, -1, 0), (2, 2, True), (2.0, 2, 0)])
def test_impossible_root_counts_have_no_label(counts):
    with pytest.raises(ValueError, match="resolved <= graded <= planned"):
        search_outcome(*counts)


def test_search_request_is_checked_and_round_trips():
    body = search_request(roots=2, points=1, candidates=2, student=2, per_candidate=2, tf=2)
    parsed = MessageRequest.from_dict(body)
    assert parsed.search.branches() == 12 and len(body["sample_slots"]) == 14, "both roots failed: 2 x (2 + 2 x 2)"
    assert parsed.to_dict() == {**body, "finalization_timeout_seconds": 1800.0}
    split = MessageRequest.from_dict(search_request(roots=2, points=2, candidates=0, student=1, per_candidate=0, tf=5))
    assert split.search.branches() == 10, "a split task's students may outnumber a failed one's: 2 x (2 - 1) x 5"
    assert "search" not in MessageRequest.from_dict(request()).to_dict()
    for change, message in [
        ({"roots": 0}, "search.roots must be positive"),
        ({"candidates": 0}, "both zero or both positive"),
        ({"tf_student_continuations": -1}, "search.tf_student_continuations must be a non-negative integer"),
        ({"tf_student_continuations": True}, "search.tf_student_continuations must be a non-negative integer"),
        ({"tf_student_continuations": 13}, "cover the roots and every branch"),
        ({"review_seconds": 0}, "finite and positive"),
        ({"points": 3}, "cover the roots and every branch"),
    ]:
        bad = deepcopy(body)
        bad["search"].update(change)
        with pytest.raises(ValueError, match=message):
            MessageRequest.from_dict(bad)
    bad = deepcopy(body)
    bad["minimum_returned_samples"] = 3
    with pytest.raises(ValueError, match="only the roots are guaranteed"):
        MessageRequest.from_dict(bad)
    bad = deepcopy(body)
    bad["search"]["rounds"] = 2
    with pytest.raises(ValueError, match="search names"):
        MessageRequest.from_dict(bad)
    bad = deepcopy(body)
    del bad["search"]["tf_student_continuations"]
    with pytest.raises(ValueError, match="search names .*tf_student_continuations"):
        MessageRequest.from_dict(bad)


def test_a_search_request_needs_the_review_config(tmp_path, peer):
    _, client = peer
    body = search_request()
    adapter = MessageAdapter(Driver(client, Ledger(tmp_path / "ledger")), config(body))
    with pytest.raises(ValueError, match="review config"):
        adapter.submit(body)


def test_the_review_config_may_bound_the_review_jobs_concurrent_requests(tmp_path, peer):
    """``max_concurrent_requests`` is one of the reviewer's settings: it reaches
    both review templates, and a bad one fails the request at submission. A
    config without it leaves the review job its own default."""
    _, client = peer
    body = search_request()

    def templates(review_changes, ledger):
        driver = Driver(client, Ledger(tmp_path / ledger))
        configured = search_config(body)
        configured["review"].update(review_changes)
        MessageAdapter(driver, configured).submit(body)
        return driver.ledger.get(internal_id(body["rollout_job_id"]))["document"]["search"]["review_templates"]

    bounded = templates({"max_concurrent_requests": 2}, "bounded")
    assert {shape: template["spec"]["max_concurrent_requests"] for shape, template in bounded.items()} == {
        "full": 2, "points_only": 2}
    assert not any("max_concurrent_requests" in template["spec"] for template in templates({}, "default").values())
    with pytest.raises(ValueError, match="max_concurrent_requests must be a positive integer"):
        templates({"max_concurrent_requests": 0}, "refused")


def test_roots_review_and_branches_fill_the_slots_in_order(tmp_path, peer):
    queue, client = peer
    body = search_request(roots=1, points=1, candidates=2, student=1, per_candidate=2)   # 1 + 1 + 4 = 6 slots
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, search_config(body))
    adapter.submit(body)
    driver.tick()
    (root,) = jobs_of(queue, "rollout")
    submitted = next(row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs")
    assert submitted["spec"]["extra"]["rollout_contract"]["max_turns"] == 10
    assert adapter.get(body["rollout_job_id"])["status"] == "running"

    finish_actor(queue, root)
    driver.tick()
    (grade,) = jobs_of(queue, "grade")
    queue.finish(grade, resolved=False, touched_test_paths=["tests/test_hidden.py"], **HIDDEN)
    driver.tick()
    # The failed root is reviewed: one review job, naming the root's attempt and verdict.
    (review,) = jobs_of(queue, "review")
    review_body = review_submission(queue)
    spec = review_body["spec"]
    assert review_body["profile"] == "review" and spec["temperature"] == 0.7 and spec["max_output_tokens"] == 4096
    assert spec["points"] == 1 and spec["candidates"] == 2 and spec["guidance"] == "assistant-turn"
    assert spec["points_per_attempt"] is True, "each failed root is branched from on its own"
    assert spec["timeout_s"] == 600.0 and spec["model_endpoint"] == body["model_endpoint"]
    # The verdict as the analyst reads it, and the hidden tests the leak check reads, never
    # which of them the diff touched.
    assert spec["parents"] == [{"name": "miles-" + spec["parents"][0]["name"][6:], "job_id": root,
                                "attempt_id": f"attempt-{root}",
                                "grade": {"resolved": False, "progress": 0.0, **HIDDEN}}]
    assert review_body["context"]["rl_driver"] == {
        "rollout_job_id": internal_id(body["rollout_job_id"]), "phase": "review", "outcome": "FF"}
    counts = {"planned": 1, "graded": 1, "resolved": 0}
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert search["state"] == "reviewing" and search["outcome"] == "FF" and search["shape"] == "full"
    assert search["root_counts"] == counts

    plan = review_result(root, candidates=2)
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result=plan)
    driver.tick()
    # One student branch and two per candidate, each from the point the review named, with its own budget.
    branch_calls = [row for row in queue.requests if row[0] == "POST" and row[1].endswith("/branch")]
    assert len(branch_calls) == 5, adapter.execution(body["rollout_job_id"])["search"]
    assert all(row[1] == f"/v1/jobs/{root}/branch" for row in branch_calls)
    continuations = [row[2]["continuation"] for row in branch_calls]
    assert continuations[0] == {"guidance": "none"}
    assert [c["guidance"] for c in continuations[1:]] == ["assistant-turn"] * 4
    assert all(c["assistant_turn_rendering"] == "reasoning" for c in continuations[1:]), \
        "a repair inserts the reviewer's text as the turn's reasoning"
    assert continuations[1]["assistant_turn"]["tool_calls"][0]["id"] == "repair-1"
    assert continuations[3]["assistant_turn"]["tool_calls"][0]["id"] == "repair-2"
    for row in branch_calls:
        assert row[2]["point_id"] == "old" and row[2]["budget"]["max_turns"] == 10 - 1
        assert row[2]["budget"]["deadline_at"] > 0 and row[2]["overrides"] == {}
    view = adapter.execution(body["rollout_job_id"])
    assert view["search"]["state"] == "branching" and view["search"]["free_slots"] == []
    assert [s["search"]["group"] for s in view["samples"][1:]] == [
        "student:old", "repair:old:1", "repair:old:1", "repair:old:2", "repair:old:2"]
    assert view["samples"][1]["search"]["kind"] == "student" and view["samples"][2]["search"]["multiplicity"] == 2
    assert not {"review_templates", "review_template"} & set(view["search"])
    assert "submission" not in view["search"]["review"]

    for job_id in jobs_of(queue, "rollout"):
        if job_id != root:
            finish_branch(queue, job_id, root)
    driver.tick()
    grades = [job_id for job_id in jobs_of(queue, "grade") if job_id != grade]
    assert len(grades) == 5
    for index, job_id in enumerate(grades):
        queue.finish(job_id, resolved=index % 2 == 1)
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed", result["stop_reason"]
    assert result["actual_samples"] == 6 and result["search_branches"] == 5
    groups = [t["group"] for t in result["trajectories"]]
    assert groups == ["root", "student:old", "repair:old:1", "repair:old:1", "repair:old:2", "repair:old:2"]
    assert [t["sample_slot_id"] for t in result["trajectories"]] == [f"slot-{i}" for i in range(6)]
    assert [t["reward"] for t in result["trajectories"]] == [0.0, 0.0, 1.0, 0.0, 1.0, 0.0]
    # Every trajectory is labelled with the roots' outcome and counts, which the trainer checks.
    assert result["trajectories"][0]["metadata"]["search"] == {"kind": "root", "outcome": "FF", "root_counts": counts}
    assert result["trajectories"][0]["metadata"]["touched_test_paths"] == ["tests/test_hidden.py"]
    student = result["trajectories"][1]["metadata"]["search"]
    assert student == {"kind": "student", "point": 1, "point_id": "old", "step": 2, "message_step": 1,
                       "parent_job_id": root, "outcome": "FF", "root_counts": counts}
    repair = result["trajectories"][2]["metadata"]
    assert repair["search"] == {**student, "kind": "repair", "candidate": 1, "multiplicity": 2, "samples": 4,
                                "leak_terms": [], "assistant_turn_rendering": "reasoning", "round": 1}
    assert [t["metadata"]["search"]["leak_terms"] for t in result["trajectories"][2:]] == [
        candidate["leak_terms"] for candidate in plan["points"][0]["candidates"] for _ in range(2)], \
        "each repair carries its candidate's leak_terms as the review found them"
    assert repair["progress"] == 1.0 and repair["provenance"]["prefix_messages"] == 1
    assert repair["touched_test_paths"] == [] and repair["zero_reason"] is None
    assert result["trajectories"][1]["parent_branch_id"] == root
    # The whole result is still the wire Miles's model accepts: nothing beyond group was added.
    assert set(result["trajectories"][0]) == {"sample_slot_id", "branch_id", "parent_branch_id", "messages", "group",
                                              "tools", "reward", "status", "stop_reason", "hints_removed", "metadata"}


def test_split_roots_get_a_review_for_points_alone_and_unaided_students(tmp_path, peer):
    queue, _ = peer
    body = search_request(roots=2, points=1, candidates=2, student=2, per_candidate=2, tf=3)   # 2 + 12 slots
    driver, adapter = searching(tmp_path, peer, body)
    resolved_root, failed_root = run_roots(queue, driver, True, False)
    (review,) = jobs_of(queue, "review")
    review_body = review_submission(queue)
    spec = review_body["spec"]
    assert spec["guidance"] == "none" and spec["candidates"] == 1 and spec["points_per_attempt"] is True
    assert [parent["job_id"] for parent in spec["parents"]] == [failed_root], "only the root that failed"
    assert review_body["context"]["rl_driver"]["outcome"] == "TF"
    counts = {"planned": 2, "graded": 2, "resolved": 1}
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert (search["outcome"], search["shape"], search["root_counts"]) == ("TF", "points_only", counts)

    # The plan carries repair turns all the same: a split task runs none of them.
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}",
                              result=review_result(failed_root, candidates=2))
    driver.tick()
    branch_calls = [row for row in queue.requests if row[0] == "POST" and row[1].endswith("/branch")]
    assert [row[1] for row in branch_calls] == [f"/v1/jobs/{failed_root}/branch"] * 3
    assert [row[2]["continuation"] for row in branch_calls] == [{"guidance": "none"}] * 3
    for job_id in jobs_of(queue, "rollout")[2:]:
        finish_branch(queue, job_id, failed_root)
    driver.tick()
    for job_id in jobs_of(queue, "grade")[2:]:
        queue.finish(job_id, resolved=False)
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed", result["stop_reason"]
    assert [t["group"] for t in result["trajectories"]] == ["root", "root"] + ["student:old"] * 3
    assert [t["reward"] for t in result["trajectories"]] == [1.0, 0.0, 0.0, 0.0, 0.0]
    assert all(t["metadata"]["search"]["outcome"] == "TF" and t["metadata"]["search"]["root_counts"] == counts
               for t in result["trajectories"])
    assert adapter.execution(body["rollout_job_id"])["search"]["state"] == "done"


def test_two_failed_roots_are_reviewed_together_in_full(tmp_path, peer):
    queue, _ = peer
    body = search_request(roots=2, points=1, candidates=2, student=2, per_candidate=2)
    driver, adapter = searching(tmp_path, peer, body)
    roots = run_roots(queue, driver, False, False)
    assert len(jobs_of(queue, "review")) == 1, "one review job per group"
    review_body = review_submission(queue)
    assert review_body["spec"]["guidance"] == "assistant-turn" and review_body["spec"]["candidates"] == 2
    assert [parent["job_id"] for parent in review_body["spec"]["parents"]] == roots
    assert review_body["context"]["rl_driver"]["outcome"] == "FF"
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert search["shape"] == "full" and search["root_counts"] == {"planned": 2, "graded": 2, "resolved": 0}


@pytest.mark.parametrize("missing", ["failed", "unexported"])
def test_a_failed_root_beside_a_missing_one_gets_a_review_for_points_alone(tmp_path, peer, missing):
    queue, _ = peer
    body = search_request(roots=2, points=1, candidates=2, student=2, per_candidate=2)
    body["minimum_returned_samples"] = 1   # as Miles sends it
    driver, adapter = searching(tmp_path, peer, body)
    missing_root, failed_root = run_roots(queue, driver, missing, False)
    (review,) = jobs_of(queue, "review")
    review_body = review_submission(queue)
    assert review_body["spec"]["guidance"] == "none" and review_body["spec"]["candidates"] == 1
    assert [parent["job_id"] for parent in review_body["spec"]["parents"]] == [failed_root], \
        "a root with no verdict or no trajectory is missing: neither reviewed nor counted"
    assert review_body["context"]["rl_driver"]["outcome"] == "FF_partial"
    counts = {"planned": 2, "graded": 1, "resolved": 0}
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert search["shape"] == "points_only" and search["root_counts"] == counts

    queue.jobs[review].update(state="failed", active_attempt=f"attempt-{review}", error="model endpoint refused")
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    # The trainer counts the roots it is sent, so they must be the roots counted here.
    gradable = [t for t in result["trajectories"] if t["metadata"]["zero_reason"] is None]
    assert [t["branch_id"] for t in gradable] == [failed_root]
    assert all(t["metadata"]["search"]["root_counts"] == counts for t in result["trajectories"])
    assert [f["reason"] for f in result["failed_samples"]] == (
        ["Native history has an incomplete final record"] if missing == "unexported" else [])
    assert missing_root not in [t["branch_id"] for t in gradable]


def test_a_resolved_root_beside_an_expired_grade_is_not_reviewed(tmp_path, peer, monkeypatch):
    queue, _ = peer
    body = search_request(roots=2)
    driver, adapter = searching(tmp_path, peer, body)
    driver.tick()
    for job_id in jobs_of(queue, "rollout"):
        finish_actor(queue, job_id)
    driver.tick()
    resolved_grade, hung_grade = jobs_of(queue, "grade")
    queue.finish(resolved_grade, resolved=True)
    queue.jobs[hung_grade].update(state="running", active_attempt=f"attempt-{hung_grade}")
    driver.tick()   # seen running: its wall time starts now
    now = driver_module.time.time()
    monkeypatch.setattr(driver_module.time, "time", lambda: now + 601)
    driver.tick()
    assert jobs_of(queue, "review") == []
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert search["state"] == "done" and search["outcome"] == "TT_partial" and search["shape"] == "none"
    assert search["error"] == "no gradable failed root: every graded root resolved, 1 grade(s) missing"
    result = adapter.get(body["rollout_job_id"])
    assert [t["metadata"]["zero_reason"] for t in result["trajectories"]] == [None, "grade_expired"]
    assert [t["metadata"]["search"] for t in result["trajectories"]] == 2 * [
        {"kind": "root", "outcome": "TT_partial", "root_counts": {"planned": 2, "graded": 1, "resolved": 1}}]


def test_roots_that_could_not_be_graded_are_not_reviewed(tmp_path, peer):
    queue, _ = peer
    body = search_request()
    driver, adapter = searching(tmp_path, peer, body)
    run_roots(queue, driver, "failed")
    assert jobs_of(queue, "review") == []
    assert adapter.execution(body["rollout_job_id"])["search"]["error"] == "no root could be graded"
    (trajectory,) = adapter.get(body["rollout_job_id"])["trajectories"]
    assert trajectory["metadata"]["zero_reason"] == "grade_failed"
    assert trajectory["metadata"]["search"] == {"kind": "root", "outcome": "ungraded",
                                                "root_counts": {"planned": 1, "graded": 0, "resolved": 0}}


def test_split_roots_are_not_branched_without_unaided_students(tmp_path, peer):
    queue, _ = peer
    body = search_request(roots=2, tf=0)
    driver, adapter = searching(tmp_path, peer, body)
    run_roots(queue, driver, True, False)
    assert jobs_of(queue, "review") == []
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert search["outcome"] == "TF" and search["shape"] == "none"
    assert search["error"] == "tf_student_continuations is 0: a task whose roots split is not branched"


@pytest.mark.parametrize("verdicts, label", [((False,), "FF"), ((True, False), "TF")])
def test_a_document_from_before_outcome_keyed_review_ends_at_its_roots(tmp_path, peer, verdicts, label):
    queue, _ = peer
    body = search_request(roots=len(verdicts))
    driver, adapter = searching(tmp_path, peer, body)
    group = internal_id(body["rollout_job_id"])
    document = driver.ledger.get(group)["document"]
    # What the adapter wrote before: one review template, and no unaided students in the config.
    document["search"]["review_template"] = document["search"].pop("review_templates")["full"]
    del document["search"]["config"]["tf_student_continuations"]
    driver.ledger.save(group, document)
    run_roots(queue, driver, *verdicts)
    assert jobs_of(queue, "review") == []
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert search["state"] == "done" and search["error"] == "search document predates outcome-keyed review"
    assert search["outcome"] == label and "review_template" not in search, "its roots are labelled all the same"


def test_a_review_that_fails_or_finds_nothing_returns_the_roots(tmp_path, peer):
    queue, client = peer
    body = search_request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, search_config(body))
    adapter.submit(body)
    driver.tick()
    (root,) = jobs_of(queue, "rollout")
    finish_actor(queue, root)
    driver.tick()
    queue.finish(jobs_of(queue, "grade")[0], resolved=False)
    driver.tick()
    (review,) = jobs_of(queue, "review")
    queue.jobs[review].update(state="failed", active_attempt=f"attempt-{review}", error="model endpoint refused")
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    # Unfilled slots are not failures: every trajectory made was exported.
    assert result["status"] == "completed" and result["actual_samples"] == 1 and result["max_samples"] == 3
    assert adapter.execution(body["rollout_job_id"])["search"]["error"] == "model endpoint refused"
    assert result["trajectories"][0]["group"] == "root"


def test_a_resolved_root_is_not_reviewed(tmp_path, peer):
    queue, client = peer
    body = search_request()
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, search_config(body))
    adapter.submit(body)
    driver.tick()
    (root,) = jobs_of(queue, "rollout")
    finish_actor(queue, root)
    driver.tick()
    queue.finish(jobs_of(queue, "grade")[0], resolved=True)
    driver.tick()
    assert jobs_of(queue, "review") == []
    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed" and result["trajectories"][0]["reward"] == 1.0
    assert adapter.execution(body["rollout_job_id"])["search"]["error"] == "every root resolved the task"
    assert JobSpec.from_dict(json.loads(json.dumps(
        next(row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs")))).kind == "rollout"


def test_a_review_whose_reply_was_lost_before_a_cancel_is_reconciled_not_reposted(tmp_path, peer):
    """The review is submitted like any stage, under its own idempotency key (slot
    ``REVIEW_SLOT``, phase ``review``). Cancelled while its reply is unknown, it is never
    posted again; attaching the job Run Store holds under that key lets the group settle."""
    from fastapi.testclient import TestClient
    from sprout.rl_driver.driver import REVIEW_SLOT
    from sprout.rl_driver.server import create_app

    queue, _ = peer
    body = search_request()
    driver, _ = searching(tmp_path, peer, body)
    group = internal_id(body["rollout_job_id"])
    driver.tick()
    (root,) = jobs_of(queue, "rollout")
    finish_actor(queue, root)
    driver.tick()
    queue.finish(jobs_of(queue, "grade")[0], resolved=False, **HIDDEN)
    queue.lose_ack = True
    driver.tick()   # the review job is accepted, and the reply saying so lost
    (review,) = jobs_of(queue, "review")
    driver.release(group)
    calls = len(queue.requests)
    driver.tick()
    assert len(queue.requests) == calls, "a submission whose reply is unknown is not posted again after a cancel"
    view = driver.get(group)
    assert view["status"] == "cancelling" and not view["ready"] and view["search"]["state"] == "reviewing"
    assert "acknowledgement unknown" in view["search"]["review"]["last_error"]
    assert view["search"]["review"]["idempotency_key"] == queue.jobs[review]["idempotency_key"]
    queue.jobs["wrong-job"] = {**queue.jobs[review], "id": "wrong-job", "idempotency_key": "different"}
    with pytest.raises(ValueError, match="idempotency key"):
        driver.reconcile_submission(group, REVIEW_SLOT, "review", "wrong-job")
    with pytest.raises(KeyError):
        driver.reconcile_submission(group, "slot-0", "review", review)
    with TestClient(create_app(driver, None, background=False)) as http:
        reply = http.post(f"/execution-groups/{group}/samples/{REVIEW_SLOT}/reconcile-submission",
                          json={"phase": "review", "job_id": review})
        assert reply.status_code == 200 and reply.json()["search"]["review"]["job_id"] == review
        driver.tick()
    assert queue.jobs[review]["state"] == "cancelled", "the attached review is stopped like any cancelled stage"
    view = driver.get(group)
    assert view["ready"] and view["status"] == "cancelled" and view["search"]["state"] == "done"
    assert not [r for r in queue.requests[calls:] if r[0] == "POST" and r[1] == "/v1/jobs"]


def _settle_new(queue, driver, known, verdicts):
    """Finish the branch actors not in ``known``, then their grades as ``verdicts``; the new job ids."""
    actors = [job_id for job_id in jobs_of(queue, "rollout") if job_id not in known]
    for job_id in actors:
        finish_branch(queue, job_id, "parent")
    driver.tick()
    grades = [job_id for job_id in jobs_of(queue, "grade") if job_id not in known]
    for job_id, verdict in zip(grades, verdicts, strict=True):
        queue.finish(job_id, resolved=verdict, **HIDDEN)
    driver.tick()
    known.update(actors, grades)
    return actors


def _chain_review(queue, known):
    (review,) = [job_id for job_id in jobs_of(queue, "review") if job_id not in known]
    known.add(review)
    return review


def test_repair_rounds_chain_a_failed_repair_until_one_resolves(tmp_path, peer):
    queue, _ = peer
    body = search_request(rounds=3)
    driver, adapter = searching(tmp_path, peer, body)
    (root,) = run_roots(queue, driver, False)
    known = {root, *jobs_of(queue, "grade")}
    review = _chain_review(queue, known)
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result=review_result(root))
    driver.tick()
    student, repair_1 = _settle_new(queue, driver, known, [False, False])

    # Round 2: the failed round-1 repair is reviewed on its own, for one point in its history and one turn.
    review_2 = _chain_review(queue, known)
    (body_2,) = [row[2] for row in queue.requests if row[0] == "POST" and row[1] == "/v1/jobs"
                 and row[2]["kind"] == "review" and row[2]["spec"].get("round_no") == 2]
    assert [parent["job_id"] for parent in body_2["spec"]["parents"]] == [repair_1]
    assert (body_2["spec"]["points"], body_2["spec"]["candidates"], body_2["spec"]["points_per_attempt"]) == (1, 1, True)
    assert body_2["context"]["rl_driver"]["round"] == 2
    view = adapter.execution(body["rollout_job_id"])["search"]
    assert view["state"] == "chaining" and view["round"] == 2
    assert "submission" not in view["chain_reviews"][0]
    queue.jobs[review_2].update(state="succeeded", active_attempt=f"attempt-{review_2}",
                                result=review_result(repair_1))
    driver.tick()
    branch_calls = [row for row in queue.requests if row[0] == "POST" and row[1].endswith("/branch")]
    assert branch_calls[-1][1] == f"/v1/jobs/{repair_1}/branch", "a round-2 repair branches from the round-1 repair"
    assert branch_calls[-1][2]["budget"]["max_turns"] == 10 - 1 - 1, "the turns its parent had left at the point"
    (repair_2,) = _settle_new(queue, driver, known, [False])

    review_3 = _chain_review(queue, known)
    plan_3 = review_result(repair_2)
    plan_3["points"][0]["candidates"][0]["leak_terms"] = ["tests/test_hidden.py"]
    queue.jobs[review_3].update(state="succeeded", active_attempt=f"attempt-{review_3}", result=plan_3)
    driver.tick()
    _settle_new(queue, driver, known, [True])

    result = adapter.get(body["rollout_job_id"])
    assert result["status"] == "completed", result["stop_reason"]
    assert [t["group"] for t in result["trajectories"]] == [
        "root", "student:old", "repair:old:1", "repair:old:2", "repair:old:3"]
    assert [t["reward"] for t in result["trajectories"]] == [0.0, 0.0, 0.0, 0.0, 1.0]
    searches = [t["metadata"]["search"] for t in result["trajectories"][2:]]
    assert [s["round"] for s in searches] == [1, 2, 3] and [s["candidate"] for s in searches] == [1, 2, 3]
    assert [s.get("chain_parent_job_id") for s in searches] == [None, repair_1, repair_2]
    assert all(s["point_id"] == "old" and s["parent_job_id"] == root for s in searches), \
        "every round's repair belongs to the root point its chain started from"
    assert [s["leak_terms"] for s in searches] == [[], [], ["tests/test_hidden.py"]], \
        "a chain carries every turn's leak terms"
    assert adapter.execution(body["rollout_job_id"])["search"]["state"] == "done"


@pytest.mark.parametrize("student_resolves, repair_resolves", [(True, False), (False, True)])
def test_no_further_round_once_the_point_resolved(tmp_path, peer, student_resolves, repair_resolves):
    queue, _ = peer
    body = search_request(rounds=3)
    driver, adapter = searching(tmp_path, peer, body)
    (root,) = run_roots(queue, driver, False)
    known = {root, *jobs_of(queue, "grade")}
    review = _chain_review(queue, known)
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result=review_result(root))
    driver.tick()
    _settle_new(queue, driver, known, [student_resolves, repair_resolves])
    assert not [job_id for job_id in jobs_of(queue, "review") if job_id not in known]
    search = adapter.execution(body["rollout_job_id"])["search"]
    assert search["state"] == "done" and search.get("round", 1) == 1
    assert search["chain_end"].startswith("after round 1")
    assert adapter.get(body["rollout_job_id"])["actual_samples"] == 3


def test_a_repair_that_touched_the_tests_does_not_end_the_chain(tmp_path, peer):
    queue, _ = peer
    body = search_request(rounds=2)
    driver, adapter = searching(tmp_path, peer, body)
    (root,) = run_roots(queue, driver, False)
    known = {root, *jobs_of(queue, "grade")}
    review = _chain_review(queue, known)
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result=review_result(root))
    driver.tick()
    actors = [job_id for job_id in jobs_of(queue, "rollout") if job_id not in known]
    for job_id in actors:
        finish_branch(queue, job_id, root)
    driver.tick()
    student_grade, repair_grade = [job_id for job_id in jobs_of(queue, "grade") if job_id not in known]
    queue.finish(student_grade, resolved=False, **HIDDEN)
    queue.finish(repair_grade, resolved=True, **HIDDEN, touched_test_paths=["tests/test_hidden.py"])
    driver.tick()
    known.update(actors, [student_grade, repair_grade])
    # Its resolution is not one the trainer credits, so the point is reviewed again.
    assert len([job_id for job_id in jobs_of(queue, "review") if job_id not in known]) == 1


def test_the_leak_judge_is_configured_fed_the_test_patch_and_its_checks_ride_the_chain(tmp_path, peer):
    queue, client = peer
    body = search_request(rounds=2)
    driver = Driver(client, Ledger(tmp_path / "ledger"))
    judge = {"threshold": 0.8, "resamples": 1}
    config_ = search_config(body)
    config_["review"]["leak_judge"] = judge
    adapter = MessageAdapter(driver, config_)
    adapter.submit(body)
    driver.tick()
    (root,) = jobs_of(queue, "rollout")
    finish_actor(queue, root)
    driver.tick()
    (grade,) = jobs_of(queue, "grade")
    queue.finish(grade, resolved=False, **HIDDEN, test_patch="/artifacts/grade/test.patch")
    driver.tick()
    # The review is configured with the judge and given the test patch the grade applied.
    submitted = review_submission(queue)
    assert submitted["spec"]["leak_judge"] == judge
    assert submitted["spec"]["parents"][0]["grade"]["test_patch"] == "/artifacts/grade/test.patch"
    known = {root, grade}
    review = _chain_review(queue, known)
    plan = review_result(root)
    flagged = {"leak": True, "leak_terms": [], "jev": {"score": 0.93, "leak": True}}
    plan["points"][0]["candidates"][0]["leak_judge"] = flagged
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result=plan)
    driver.tick()
    _, repair_1 = _settle_new(queue, driver, known, [False, False])
    review_2 = _chain_review(queue, known)
    plan_2 = review_result(repair_1)
    clean = {"leak": False, "leak_terms": [], "jev": {"score": 0.1, "leak": False}}
    plan_2["points"][0]["candidates"][0]["leak_judge"] = clean
    queue.jobs[review_2].update(state="succeeded", active_attempt=f"attempt-{review_2}", result=plan_2)
    driver.tick()
    _settle_new(queue, driver, known, [True])
    result = adapter.get(body["rollout_job_id"])
    searches = [t["metadata"]["search"] for t in result["trajectories"][2:]]
    assert [s["leak_judge"] for s in searches] == [[flagged], [flagged, clean]], \
        "a repair carries the check of every turn its chain inserted, in round order"
    assert "leak_judge" not in result["trajectories"][1]["metadata"]["search"], "a student inserts nothing"


def test_without_a_leak_judge_repairs_carry_no_leak_judge(tmp_path, peer):
    queue, _ = peer
    body = search_request()
    driver, adapter = searching(tmp_path, peer, body)
    (root,) = run_roots(queue, driver, False)
    assert "leak_judge" not in review_submission(queue)["spec"]
    (review,) = jobs_of(queue, "review")
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result=review_result(root))
    driver.tick()
    _settle_new(queue, driver, {root, *jobs_of(queue, "grade")}, [False, True])
    result = adapter.get(body["rollout_job_id"])
    assert all("leak_judge" not in t["metadata"]["search"] for t in result["trajectories"][1:])


def test_more_than_one_round_takes_one_candidate_and_one_continuation():
    with pytest.raises(ValueError, match="more than one search.repair_rounds"):
        MessageRequest.from_dict(search_request(rounds=2, candidates=2))
    spec = MessageRequest.from_dict(search_request(roots=2, rounds=3, student=1, tf=2)).search
    assert spec.branches() == 1 * max(2 * (1 + 3), 1 * 2)
    assert spec.search_seconds(600.0) == 3 * (600.0 + 900.0) + 2 * 600.0

"""The GSML wire on Sprout's side: from a grade's facts to the result Miles credits.

Miles credits a search group by what the export writes on each trajectory
(``MessageAdapter._export``): the roots' outcome and counts, a branch's point,
a repair's candidate, its ``leak_terms`` and how its inserted turn was
rendered, and the task's test files the graded diff touched. These tests
drive the real driver and adapter against the driver tests' fake Run Store and
check that each of those is consumed where it should be. The shared FF example
of the credit rules is what the driver returns for it, checked in as
``search_result.example.json``; Miles's GSML tests read a byte copy of that
file (``tests/fast/rollout/sprout/fixtures/sprout_search_result.example.json``).
"""

from copy import deepcopy
import json

import httpx
import pytest

from sprout.contract.continuation import Continuation, prepare, render_reasoning
from sprout.rl_driver.backend import RunStoreClient
from sprout.rl_driver.driver import REVIEW_GRADE_KEYS
from sprout.rl_driver.messages import SearchSpec
from sprout.runstore.grading import GRADE_FACT_KEYS, patch_facts
from sprout.methods.leak import hidden_terms_of
from sprout.contract.messages import branch_origin
from tests.rl_driver.test_driver import QueueHTTP
from tests.rl_driver.test_message_rollout import RESULT_EXAMPLE, exported, request
from tests.rl_driver.test_search import (
    HIDDEN, finish_actor, jobs_of, review_submission, run_roots, search_request, searching,
)

SEARCH_RESULT_EXAMPLE = RESULT_EXAMPLE.with_name("search_result.example.json")

#: The shared example's search, as Miles asks for it: two roots, a point in each failed one, two
#: students there and two continuations after each of two repair turns; a split task's point gets
#: two students.
SEARCH = {"roots": 2, "points": 1, "candidates": 2, "student_continuations": 2, "candidate_continuations": 2,
          "tf_student_continuations": 2, "repair_rounds": 1, "review_seconds": 600.0, "wall_time_seconds": 900.0}

#: What a branch restores: its root's history up to the point, the root's first tool result.
PREFIX = exported()["training_messages"][:3]

#: The repair turns the review writes at the point of each failed root: the text, then the command.
#: The words are ones Miles's toy tokenizer knows; its tests render the example with it.
REPAIRS = {1: [("parser broken", "sed -n 1,40p src/parser.py"), ("inspect tests", "pytest -q tests")],
           2: [("parser broken", "grep -n parse src/parser.py"), ("inspect tests", "pytest -q tests/test_hidden.py")]}
#: The hidden test names the review's leak check found in a turn, by point and candidate; [] for the rest.
LEAKS = {(2, 2): ["tests/test_hidden.py"]}

#: The FF example's verdicts after its two failed roots, slot by slot: at P1 the students (1, 0),
#: candidate 1's continuations (1, 0) and candidate 2's (0, 0); at P2 the students (0, 0),
#: candidate 1's (1, 1) and candidate 2's (1, 0).
FF_BRANCHES = (True, False, True, False, False, False, False, False, True, True, True, False)

#: A task's hidden test patch, and an agent's diff that changed the patched test file as well.
TEST_PATCH = """diff --git a/tests/test_hidden.py b/tests/test_hidden.py
index 3b18e51..8d0e412 100644
--- a/tests/test_hidden.py
+++ b/tests/test_hidden.py
@@ -1,2 +1,6 @@
 def test_parses():
     assert parse(" x ") == "x"
+
+
+def test_secret_behaviour():
+    assert parse("") == ""
"""
MODEL_PATCH = """diff --git a/src/parser.py b/src/parser.py
index 1f2a3b4..5c6d7e8 100644
--- a/src/parser.py
+++ b/src/parser.py
@@ -1,2 +1,4 @@
 def parse(text):
+    if not text:
+        return text
     return text.strip()
diff --git a/tests/test_hidden.py b/tests/test_hidden.py
index 3b18e51..9a0b1c2 100644
--- a/tests/test_hidden.py
+++ b/tests/test_hidden.py
@@ -1,2 +1,2 @@
 def test_parses():
-    assert parse(" x ") == "x"
+    assert parse("x") == "x"
"""
ROW = {"test_patch": TEST_PATCH, "FAIL_TO_PASS": ["tests/test_hidden.py::test_secret_behaviour"],
       "PASS_TO_PASS": ["tests/test_hidden.py::test_parses"]}


class Peer(QueueHTTP):
    """The driver tests' Run Store, with recovery points that are each attempt's own and each branch job's
    submission kept for the worker that runs it. Two roots' points must not share an id: their branches
    would share a group."""

    def __init__(self):
        super().__init__()
        self.branches = {}

    def __call__(self, req):
        response = super().__call__(req)
        if req.method == "POST" and req.url.path.endswith("/branch"):
            self.branches[response.json()["id"]] = {"parent": req.url.path.split("/")[3], **json.loads(req.content)}
        return response

    def final_point(self, job_id):
        super().final_point(job_id)
        for point in self.points[job_id]:
            point["id"] = f"{job_id}-step{point['tool_depth']}"


@pytest.fixture
def peer():
    queue = Peer()
    client = RunStoreClient("http://runstore", "fixture")
    client.http.close()
    client.http = httpx.Client(base_url="http://runstore", transport=httpx.MockTransport(queue))
    try:
        yield queue, client
    finally:
        client.close()


def shared_request():
    """Sprout's shipped request as Miles sends it with ``SEARCH``: the roots' slots, then one for each branch."""
    body = request()
    job = body["rollout_job_id"]
    slots = SEARCH["roots"] + SearchSpec.from_dict(SEARCH).branches()
    body["sample_slots"] = [{"sample_slot_id": f"{job}:slot:{11 + k}", "sample_index": 11 + k} for k in range(slots)]
    body.update(max_samples=slots, minimum_returned_samples=1, search=dict(SEARCH))
    return body


def repair_turn(point, candidate):
    text, command = REPAIRS[point][candidate - 1]
    return {"role": "assistant", "content": text, "tool_calls": [{
        "id": f"repair-{point}-{candidate}", "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})}}]}


def review_result(parents, guidance="assistant-turn"):
    """What ``runstore.review.run_review`` returns for these parents: a point at step 2 of each and, at
    each point, two repair turns with the ``leak_terms`` the review found in them. A points-only review
    (guidance ``none``) writes no turns; this plan carries them all the same, for the driver to ignore."""
    return {
        "status": "completed", "guidance": guidance,
        "points": [{"base": parent["name"], "job_id": parent["job_id"], "attempt_id": parent["attempt_id"],
                    "step": 2, "snapshot_id": "AB", "conversation_cut": f"turn-{parent['job_id']}-2",
                    "why": "before the edit", "samples": 2,
                    "candidates": [{"assistant_turn": repair_turn(point, candidate), "why": "check first",
                                    "multiplicity": 1, "leak_terms": LEAKS.get((point, candidate), [])}
                                   for candidate in (1, 2)]}
                   for point, parent in enumerate(parents, 1)],
        "reports": {parent["name"]: {"failure_reason": "edited the wrong file", "lesson": "read the test first",
                                     "salvage": "the exploration", "steps": 3,
                                     "branch_candidates": [{"step": 2, "why": "before the edit"}]}
                    for parent in parents},
        "analysed": [parent["name"] for parent in parents],
        # Per root an analysis and a point review, then one review per candidate.
        "review_requests": 4 * len(parents),
    }


def finish_branch(queue, job_id):
    """Settle a branch's actor as the worker exports one: the history it restored, the inserted turn as
    ``continuation.prepare`` renders it for mini and that turn's result, then the branch's own last turn."""
    branch = queue.branches[job_id]
    point = next(p for p in queue.points[branch["parent"]] if p["id"] == branch["point_id"])
    # As the Run Store API reads the continuation, and as its child hands it to mini.
    _, options = prepare(Continuation.from_dict(branch["continuation"]),
                         [{"type": "mini.message", "message": message} for message in PREFIX])
    inserted = []
    if "assistant_turn" in options:
        turn = options["assistant_turn"]
        inserted = [turn, *({"role": "tool", "tool_call_id": call["id"], "content": "tests"}
                            for call in turn["tool_calls"])]
    queue.finish(job_id)
    queue.jobs[job_id]["result"].update(
        training_messages=[*deepcopy(PREFIX), *inserted, {"role": "assistant", "content": "fixed"}],
        training_provenance={"prefix_messages": len(PREFIX), "inserted_messages": [len(PREFIX)] if inserted else []},
        training_origin=branch_origin({**point, "job_id": branch["parent"]}))
    queue.final_point(job_id)


def run_search(tmp_path, peer, roots, branches):
    """A search of ``shared_request`` through the driver and the adapter, to its result.

    The roots are graded ``roots`` (``test_search.settle``), the review proposes ``review_result`` for
    the parents it is given, and the branches it leads to are graded ``branches``, in slot order. Every
    grade reports the task's hidden tests and that the diff touched none of them. Returns the result,
    the review job's submission and its plan.
    """
    queue, _ = peer
    body = shared_request()
    driver, adapter = searching(tmp_path, peer, body)
    run_roots(queue, driver, *roots)
    (review,) = jobs_of(queue, "review")
    submitted = review_submission(queue)
    plan = review_result(submitted["spec"]["parents"], submitted["spec"]["guidance"])
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result=plan)
    driver.tick()
    for job_id in list(queue.branches):
        finish_branch(queue, job_id)
    driver.tick()
    samples = adapter.execution(body["rollout_job_id"])["samples"]
    for sample, verdict in zip(samples[len(roots):], branches, strict=True):
        queue.finish(sample["grade"]["job_id"], resolved=verdict, **HIDDEN)
    driver.tick()
    return adapter.get(body["rollout_job_id"]), submitted, plan


def test_grade_facts_reach_the_review_and_the_trajectory(tmp_path, peer):
    facts = patch_facts(MODEL_PATCH, ROW)
    assert set(facts) == set(GRADE_FACT_KEYS)
    assert facts == {"test_patch_paths": ["tests/test_hidden.py"], "touched_test_paths": ["tests/test_hidden.py"],
                     "fail_to_pass": ["tests/test_hidden.py::test_secret_behaviour"]}
    # The review is given the task's hidden tests, which only its leak check reads, and never which of
    # them the diff touched: that is the trainer's, on the trajectory.
    assert {"fail_to_pass", "test_patch_paths"} <= set(REVIEW_GRADE_KEYS)
    assert "touched_test_paths" not in REVIEW_GRADE_KEYS
    reviewed = {key: value for key, value in facts.items() if key in REVIEW_GRADE_KEYS}
    assert "tests/test_hidden.py" in hidden_terms_of(reviewed)

    queue, _ = peer
    body = search_request()   # one root, which fails
    driver, adapter = searching(tmp_path, peer, body)
    driver.tick()
    (root,) = jobs_of(queue, "rollout")
    finish_actor(queue, root)
    driver.tick()
    (grade,) = jobs_of(queue, "grade")
    queue.finish(grade, resolved=False, **facts)
    driver.tick()
    (review,) = jobs_of(queue, "review")
    (parent,) = review_submission(queue)["spec"]["parents"]
    assert parent["grade"] == {"resolved": False, "progress": 0.0, **reviewed}
    assert hidden_terms_of(parent["grade"]) == hidden_terms_of(reviewed), "the leak check has what it reads"

    # The review finds no point: the search ends at its root, exported with what its diff touched.
    queue.jobs[review].update(state="succeeded", active_attempt=f"attempt-{review}", result={
        "status": "completed", "guidance": "assistant-turn", "points": [], "reports": {},
        "analysed": [parent["name"]], "review_requests": 2})
    driver.tick()
    result = adapter.get(body["rollout_job_id"])
    assert adapter.execution(body["rollout_job_id"])["search"]["error"] == \
        "the review proposed no branch that could be made"
    (trajectory,) = result["trajectories"]
    assert trajectory["reward"] == 0.0 and trajectory["metadata"]["zero_reason"] is None
    assert trajectory["metadata"]["touched_test_paths"] == ["tests/test_hidden.py"]


def test_the_ff_search_example_is_what_the_driver_returns(tmp_path, peer):
    queue, _ = peer
    result, review_body, plan = run_search(tmp_path, peer, (False, False), FF_BRANCHES)
    spec = review_body["spec"]
    assert spec["guidance"] == "assistant-turn" and spec["points"] == 1 and spec["candidates"] == 2
    assert review_body["context"]["rl_driver"]["outcome"] == "FF"
    assert [parent["grade"] for parent in spec["parents"]] == 2 * [{"resolved": False, "progress": 0.0, **HIDDEN}]

    # At each point two students with nothing added, then two continuations after each repair turn,
    # which every one of them asks Run Store to insert as the turn's reasoning.
    expected = []
    for point in plan["points"]:
        expected += 2 * [{"guidance": "none"}]
        for candidate in point["candidates"]:
            expected += 2 * [{"guidance": "assistant-turn", "assistant_turn": candidate["assistant_turn"],
                              "assistant_turn_rendering": "reasoning"}]
    assert [branch["continuation"] for branch in queue.branches.values()] == expected

    trajectories = result["trajectories"]
    assert result["status"] == "completed", result["stop_reason"]
    assert len(trajectories) == result["actual_samples"] == 14 and result["search_branches"] == 12
    assert [t["reward"] for t in trajectories] == [0.0, 0.0, *map(float, FF_BRANCHES)]
    counts = {"planned": 2, "graded": 2, "resolved": 0}
    assert all(t["metadata"]["search"]["outcome"] == "FF" and t["metadata"]["search"]["root_counts"] == counts
               for t in trajectories), "every trajectory says how the roots came out"
    roots = [t["branch_id"] for t in trajectories[:2]]
    groups = ["root", "root"]
    for root in roots:
        groups += 2 * [f"student:{root}-step2"] + 2 * [f"repair:{root}-step2:1"] + 2 * [f"repair:{root}-step2:2"]
    assert [t["group"] for t in trajectories] == groups
    assert [t["parent_branch_id"] for t in trajectories] == [None, None, *(root for root in roots for _ in range(6))]
    assert all(t["metadata"]["zero_reason"] is None and t["metadata"]["touched_test_paths"] == []
               for t in trajectories)
    for trajectory in trajectories:
        search = trajectory["metadata"]["search"]
        if search["kind"] != "repair":
            continue
        candidate = plan["points"][search["point"] - 1]["candidates"][search["candidate"] - 1]
        assert search["leak_terms"] == candidate["leak_terms"], "a repair carries what the review found in its turn"
        assert search["assistant_turn_rendering"] == "reasoning"
        inserted = trajectory["messages"][max(trajectory["metadata"]["provenance"]["inserted_messages"])]
        assert inserted == render_reasoning(candidate["assistant_turn"])
        assert inserted["content"] == "" and inserted["reasoning_content"].strip(), "the canonical form"
    assert [t["metadata"]["search"]["leak_terms"] for t in trajectories if t["group"].endswith(":2")] == \
        2 * [[]] + 2 * [["tests/test_hidden.py"]]

    # The shared example, as Miles's GSML tests read it.
    assert result == json.loads(SEARCH_RESULT_EXAMPLE.read_text())


def test_split_roots_are_branched_by_unaided_students_alone(tmp_path, peer):
    """The TF example of the credit rules: roots (1, 0), then two students (1, 0) from the failed root."""
    queue, _ = peer
    result, review_body, plan = run_search(tmp_path, peer, (True, False), (True, False))
    spec = review_body["spec"]
    failed_root = result["trajectories"][1]["branch_id"]
    assert spec["guidance"] == "none" and spec["candidates"] == 1
    assert [parent["job_id"] for parent in spec["parents"]] == [failed_root], "only the root that failed"
    assert review_body["context"]["rl_driver"]["outcome"] == "TF"
    # The plan carries repair turns all the same: a split task runs none of them.
    assert plan["points"][0]["candidates"]
    assert [branch["continuation"] for branch in queue.branches.values()] == 2 * [{"guidance": "none"}]
    assert result["status"] == "completed", result["stop_reason"]
    trajectories = result["trajectories"]
    assert [t["group"] for t in trajectories] == ["root", "root", *2 * [f"student:{failed_root}-step2"]]
    assert [t["reward"] for t in trajectories] == [1.0, 0.0, 1.0, 0.0]
    counts = {"planned": 2, "graded": 2, "resolved": 1}
    assert [t["metadata"]["search"]["kind"] for t in trajectories] == ["root", "root", "student", "student"]
    assert all(t["metadata"]["search"]["outcome"] == "TF" and t["metadata"]["search"]["root_counts"] == counts
               for t in trajectories)

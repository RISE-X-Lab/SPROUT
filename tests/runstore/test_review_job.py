"""The review job: SPROUT's analyst and reviewer over Run Store attempts, as a worker child runs it."""

import json
import logging
import os
import queue
import re
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from sprout.worker.agents.mini_swe.history import available_branch_points
from sprout.contract.mini_history import materialize
from sprout.worker.run.runner import Orchestrator
from sprout.methods import review_transport
from sprout.methods.review_transport import RequestStopped, ReviewTransport
from sprout.worker.jobs.review import job as review_module
from sprout.contract.messages import mark_hint
from sprout.runstore.payload import encode_payload, send_payload
from sprout.worker.jobs.review.job import (
    ANALYSIS, CANDIDATE, POINTS, Call, ReviewStopped, commands_of, grade_of, run_review,
)
from sprout.runstore.specs import JobSpec, ReviewSpec, digest
from sprout.worker.attempts import Watchdog, process_identity, stop_process
from tests.harness.test_mini_swe import (
    FilesystemSession, model_server, owned_filesystem, pytestmark, reply, spec as run_spec,
)
from tests.runstore.test_manager import eventually
from tests.runstore.test_mini_native import parent_run

_ = pytestmark  # the parent run needs mini installed, like the harness tests it reuses

ANALYST_OPENING = "You are analysing ONE failed coding attempt"
POINT_OPENING = "You are the REVIEWER choosing restart points"
#: The grade's record of the hidden tests: it is input to the leak gate and is
#: never shown to the analyst or the reviewer.
HIDDEN = {"fail_to_pass": ["tests/test_hidden.py::test_secret_behaviour"],
          "test_patch_paths": ["tests/test_hidden.py"]}
LEAKED_PATH = ["test_hidden", "test_hidden.py", "tests/test_hidden.py"]


def shows_hidden(text):
    return "tests/test_hidden.py" in text or "test_secret_behaviour" in text


def answer(prompt, command="pytest -q", call="candidate"):
    """What the model a fake stands in for says to ``prompt``: an analyst report,
    a point plan with as many points as allowed from the available steps, or an
    assistant-turn plan at the one step offered, in the wire form every parser
    accepts, proposing ``command``."""
    if ANALYST_OPENING in prompt:
        return json.dumps({"failure_reason": "edited the wrong file", "lesson": "read the test first",
                           "salvage": "the exploration", "branch_candidates": [{"step": 2, "why": "before the edit"}]})
    # The reports are rendered as indented JSON: lists span lines.
    steps = [int(s) for s in re.findall(r"\d+", re.search(r'"available_steps":\s*\[([^\]]*)\]', prompt).group(1))]
    base = re.search(r'"name":\s*"([^"]+)",\s*"round"', prompt).group(1)
    if POINT_OPENING in prompt:   # the point review: as many points as allowed, from the available steps
        allowed = int(re.search(r"Return at most (\d+) branches", prompt).group(1))
        plan = {"synthesis": "points", "branches": [
            {"name": f"point-{step}", "base": base, "branch_step": step, "why": "a decision"}
            for step in steps[:allowed]]}
    else:   # an assistant-turn review of one point
        assert steps == [steps[0]], "a candidate review is confined to one point"
        plan = {"synthesis": "repair", "branches": [{
            "name": "repair", "base": base, "branch_step": steps[0], "why": "check first",
            "assistant_turn": {"role": "assistant", "content": "I will check the test before editing.",
                               "tool_calls": [{"id": f"{call}-call", "type": "function",
                                               "function": {"name": "bash", "arguments": {"command": command}}}]}}]}
    return "```branch-plan\n" + json.dumps(plan) + "\n```"


class FakeReviewModel:
    """The review model behind every call's transport.

    The review makes a transport per call (``runstore.review.Call``), for the
    call's receipts directory, so ``factory`` hands one out per call and each
    answers by its call's name, never by the order calls arrive in.
    ``proposals`` is the command every candidate proposes, or maps a candidate
    call's name (``point1-candidate2``) to its command, or is a function of
    the name, so duplicates can be arranged. ``hold(call)`` runs before each
    answer, for a test that holds calls back; ``fail`` maps a call's name to
    the error it raises instead of answering, and ``replies`` to the texts it
    answers its first requests with. Each transport is given the call's
    ``stop`` like any other, and ignores it: the review's own check before
    each request is what a fake meets.
    """

    def __init__(self, *, proposals="pytest -q", hold=None, fail=None, replies=None):
        self.proposals, self.hold, self.fail = proposals, hold, fail or {}
        self.replies = {call: list(texts) for call, texts in (replies or {}).items()}
        self.lock = threading.Lock()
        #: (call, prompt) as requests arrive; the calls transports were made for, and their
        #: settings; calls as they answer or raise.
        self.requests, self.transports, self.settings, self.answered = [], [], [], []
        self.in_flight = self.most_in_flight = 0
        self.events = {}

    @property
    def prompts(self):
        return [prompt for _, prompt in self.requests]

    @property
    def calls(self):
        return [call for call, _ in self.requests]

    def finished(self, call):
        """Set once ``call`` has answered or raised."""
        return self._event("finished", call)

    def made(self, call):
        """Set once the review has made ``call``'s transport, on the thread that starts calls."""
        return self._event("made", call)

    def arrived(self, call):
        """Set once a request of ``call`` has arrived, before ``hold`` runs: the call is in flight."""
        return self._event("arrived", call)

    def _event(self, kind, call):
        with self.lock:
            return self.events.setdefault((kind, call), threading.Event())

    def factory(self, directory, **settings):
        call = Path(directory).name
        with self.lock:
            self.transports.append(call)
            self.settings.append(settings)
        self.made(call).set()
        return CallTransport(self, call)

    def request(self, call, prompt):
        with self.lock:
            self.requests.append((call, prompt))
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        self.arrived(call).set()
        try:
            if self.hold is not None:
                self.hold(call)
            if call in self.fail:
                raise self.fail[call]
            if self.replies.get(call):
                return self.replies[call].pop(0)
            if stage_of(call) != "point":
                return answer(prompt, call=call)
            proposals = self.proposals
            command = (proposals if isinstance(proposals, str)
                       else proposals(call) if callable(proposals) else proposals[call])
            return answer(prompt, command, call)
        finally:
            finished = self.finished(call)
            with self.lock:
                self.in_flight -= 1
                self.answered.append(call)
            finished.set()


class CallTransport:
    """One call's transport, counting its requests as ``ReviewTransport`` does."""

    def __init__(self, model, call):
        self.model, self.call, self.sequence = model, call, 0

    def __call__(self, model, prompt):
        self.sequence += 1
        return self.model.request(self.call, prompt)


class Recorded:
    """Events per call, set once the review has recorded what the call
    returned (``returned``) or its failure (``failed``, with what that failure
    stops already stopped), which it does on the thread that starts calls
    (``_Review._finished``, ``_Review._fail``). A test that orders one call
    after another's outcome waits on these, not on the clock."""

    def __init__(self, monkeypatch):
        self.lock, self.events = threading.Lock(), {}
        finish, fail = review_module._Review._finished, review_module._Review._fail

        def finished(review, call, value):
            successors = finish(review, call, value)
            self.returned(call.name).set()
            return successors

        def failed(review, failures, running, call, error):
            fail(review, failures, running, call, error)
            self.failed(call.name).set()

        monkeypatch.setattr(review_module._Review, "_finished", finished)
        monkeypatch.setattr(review_module._Review, "_fail", failed)

    def returned(self, call):
        return self._event("returned", call)

    def failed(self, call):
        return self._event("failed", call)

    def _event(self, kind, call):
        with self.lock:
            return self.events.setdefault((kind, call), threading.Event())


def by_sample(commands, attempt=None):
    """The commands each point's samples propose, one list per point, keyed by
    the candidate calls' names; ``attempt`` for a points_per_attempt review."""
    prefix = "" if attempt is None else f"{attempt}."
    return {f"point{prefix}{point}-candidate{sample}": command
            for point, samples in enumerate(commands, 1) for sample, command in enumerate(samples, 1)}


def stage_of(call):
    """``analyst``, ``points`` or ``point`` (a candidate sample)."""
    return re.match(r"[a-z]+", call).group(0)


def root_of(call):
    """The failed attempt a call of a points_per_attempt review serves."""
    return int(re.match(r"[a-z]+-?(\d+)", call).group(1))


def review_threads():
    return [thread for thread in threading.enumerate() if thread.name.startswith("sprout-review")]


def review_spec(root, **changes):
    spec = {"problem": "Fix the failing test.", "model": "fixture", "model_endpoint": "http://127.0.0.1:1",
            "api_key_env": "FIXTURE_KEY", "guidance": "assistant-turn", "points": 2, "candidates": 3,
            "artifact_root": str(root),
            "parents": [{"name": "root-1", "job_id": "job-1", "attempt_id": "attempt-1",
                         "grade": {"resolved": False, "progress": 0.5, "test_statuses": {"t::a": "FAILED"},
                                   **HIDDEN}}]}
    spec.update(changes)
    return spec


def two_failed_roots(tmp_path, monkeypatch, **changes):
    """Two failed roots of one task under the artifact root, and a review spec over both."""
    root = tmp_path / "artifacts"
    parent_run(root / "job-1" / "attempt-1", monkeypatch)
    parent_run(root / "job-2" / "attempt-2", monkeypatch)
    spec = review_spec(root, **changes)
    spec["parents"].append({"name": "root-2", "job_id": "job-2", "attempt_id": "attempt-2",
                            "grade": {"resolved": False, "progress": 0.0}})
    return spec


def test_the_review_picks_points_then_writes_independent_candidates_per_point(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    _, _, points = parent_run(root / "job-1" / "attempt-1", monkeypatch)
    proposals = by_sample([["pytest -q", "pytest -q", "pytest -q -k test_secret_behaviour"],
                           ["pytest -q", "ls", "git diff"]])
    fake = FakeReviewModel(proposals=proposals)
    result = run_review(review_spec(root), tmp_path / "review", transport_factory=fake.factory)
    assert len(fake.settings) == 8, "a transport per call"
    assert all(settings["temperature"] == 0.8 and settings["api_key_env"] == "FIXTURE_KEY"
               for settings in fake.settings)
    assert result["status"] == "completed" and result["analysed"] == ["root-1"]
    assert sum(ANALYST_OPENING in p for p in fake.prompts) == 1, "one analysis per failed attempt"
    assert [p["step"] for p in result["points"]] == sorted(available_branch_points(
        root / "job-1" / "attempt-1" / "trajectory.jsonl"))[:2]
    for point in result["points"]:
        assert point["base"] == "root-1" and point["job_id"] == "job-1" and point["snapshot_id"]
        assert point["samples"] == 3 and sum(c["multiplicity"] for c in point["candidates"]) == 3
        for candidate in point["candidates"]:
            assert set(candidate) == {"assistant_turn", "why", "multiplicity", "leak_terms"}
            turn = candidate["assistant_turn"]
            assert turn["role"] == "assistant" and turn["tool_calls"][0]["function"]["name"] == "bash"
    first, second = result["points"]
    assert [c["multiplicity"] for c in first["candidates"]] == [2, 1], "identical proposals merge"
    assert [c["multiplicity"] for c in second["candidates"]] == [1, 1, 1]
    # The hidden target's name, which the parent's history never showed, is flagged;
    # the other candidates name nothing hidden. Every candidate is kept either way.
    assert [c["leak_terms"] for c in first["candidates"]] == [[], ["test_secret_behaviour"]]
    assert [c["leak_terms"] for c in second["candidates"]] == [[], [], []]
    assert result["review_requests"] == 1 + 1 + 2 * 3
    assert (tmp_path / "review" / "plan-points.json").is_file()
    assert len(list((tmp_path / "review").glob("plan-point*-candidate*.json"))) == 6
    assert "root-1" in result["reports"] and result["reports"]["root-1"]["steps"] >= 2
    # The history holds no hidden name, so no prompt (analyst, point or candidate
    # review) may hold one. The analyst still gets the verdict.
    recorded = "\n".join(path.read_text() for path in (root / "job-1" / "attempt-1").rglob("*.jsonl"))
    assert recorded and not shows_hidden(recorded)
    assert len(fake.prompts) == 8 and not any(shows_hidden(prompt) for prompt in fake.prompts)
    analyst = next(p for p in fake.prompts if ANALYST_OPENING in p)
    assert "target tests passed (FAIL_TO_PASS): 50%" in analyst and "t::a: FAILED" in analyst

    # A grade without either fact gives nothing to check against: leak_terms is None, the merge unchanged.
    spec = review_spec(root)
    for key in HIDDEN:
        del spec["parents"][0]["grade"][key]
    unchecked = run_review(spec, tmp_path / "review-unchecked",
                           transport_factory=FakeReviewModel(proposals=proposals).factory)
    assert [[c["multiplicity"] for c in p["candidates"]] for p in unchecked["points"]] == [[2, 1], [1, 1, 1]]
    assert [c["leak_terms"] for p in unchecked["points"] for c in p["candidates"]] == [None] * 5


def test_leak_terms_judge_a_turn_against_the_hint_free_history_at_its_own_point(tmp_path, monkeypatch):
    """The reviewer may borrow from anywhere in the attempt. The actor keeps
    only the prefix, and training drops the prefix's hint. Here the hint names
    the hidden test file and the actor first opens that file in its second
    turn, so the same repair leaks at step 1 and does not at step 2."""
    root = tmp_path / "artifacts"
    _, _, points = parent_run(root / "job-1" / "attempt-1", monkeypatch)
    branch = root / "job-2" / "attempt-2"
    extra = materialize(points[0].native, branch / "cwd", branch / "restoration", branch / "native-home")
    memory = FilesystemSession(branch / "sandbox")
    monkeypatch.setattr(Orchestrator, "_wire_sandbox", lambda *args: owned_filesystem(memory))
    hint = "Start from tests/test_hidden.py."
    with model_server([reply("cat answer"), reply("cat tests/test_hidden.py"),
                       reply("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT")]) as (url, requests):
        branch_spec = run_spec(branch, url, prompt=mark_hint(hint))
        branch_spec.extra.update(extra)
        outcome = Orchestrator().run(branch_spec)
    assert outcome.status == "completed", outcome.error
    assert hint in json.dumps(requests[0]["messages"]), "the actor was given the hint"
    spec = review_spec(root, candidates=1)
    spec["parents"] = [{"name": "branch-1", "job_id": "job-2", "attempt_id": "attempt-2", "hint": hint,
                        "grade": {"resolved": False, "progress": 0.0, **HIDDEN}}]
    result = run_review(spec, tmp_path / "review",
                        transport_factory=FakeReviewModel(proposals="pytest -q tests/test_hidden.py").factory)
    assert [p["step"] for p in result["points"]] == [1, 2]
    assert [p["candidates"][0]["leak_terms"] for p in result["points"]] == [LEAKED_PATH, []]


def test_points_per_attempt_branches_every_failed_attempt_once(tmp_path, monkeypatch):
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=1)
    fake = FakeReviewModel(proposals={**by_sample([["pytest -q"]], attempt=1), **by_sample([["ls"]], attempt=2)})
    result = run_review(spec, tmp_path / "review", transport_factory=fake.factory)
    assert [point["base"] for point in result["points"]] == ["root-1", "root-2"], "one point from each failure"
    assert [[commands_of(c["assistant_turn"]) for c in point["candidates"]] for point in result["points"]] == [
        [("pytest -q",)], [("ls",)]]
    assert [[c["leak_terms"] for c in point["candidates"]] for point in result["points"]] == [[[]], [None]], \
        "each point is checked against its own parent's grade"
    point_prompts = [p for p in fake.prompts if POINT_OPENING in p]
    assert len(point_prompts) == 2 and all(("root-1" in p) != ("root-2" in p) for p in point_prompts), \
        "each failure is reviewed on its own"
    assert {path.name for path in (tmp_path / "review").glob("plan-points*.json")} == {
        "plan-points-1.json", "plan-points-2.json"}
    # A point is named by its failed attempt and its place among that attempt's
    # points, so no name waits for another attempt's point review.
    assert {path.name for path in (tmp_path / "review").glob("plan-point*-candidate*.json")} == {
        "plan-point1.1-candidate1.json", "plan-point2.1-candidate1.json"}
    assert sorted(fake.calls) == ["analyst-1", "analyst-2", "point1.1-candidate1", "point2.1-candidate1",
                                  "points-1", "points-2"]
    with pytest.raises(ValueError, match="points_per_attempt"):
        ReviewSpec(**{**spec, "points_per_attempt": "yes"}).validate()


def test_a_none_review_picks_points_only(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    parent_run(root / "job-1" / "attempt-1", monkeypatch)
    fake = FakeReviewModel(proposals={})
    result = run_review(review_spec(root, guidance="none", candidates=1), tmp_path / "review",
                        transport_factory=fake.factory)
    assert len(result["points"]) == 2 and all(point["candidates"] == [] for point in result["points"])
    assert not any("samples" in point for point in result["points"])
    assert result["review_requests"] == 2 and fake.calls == ["analyst-1", "points"]
    assert not any(shows_hidden(prompt) for prompt in fake.prompts)


def test_the_analyst_never_sees_the_grade_facts_kept_for_the_leak_gate(tmp_path):
    patch = tmp_path / "model.patch"
    patch.write_text("diff --git a/src/module.py b/src/module.py\n")
    tests = tmp_path / "test.patch"
    tests.write_text("+    assert convert('SECRET-FIXTURE') == 'ok'\n")
    grade = grade_of({"resolved": False, "progress": 0.25, "model_patch": str(patch), "reason": "tests_timed_out",
                      "test_statuses": {"t::a": "FAILED"}, "touched_test_paths": ["tests/test_touched.py"],
                      "test_patch": str(tests), **HIDDEN})
    assert "target tests passed (FAIL_TO_PASS): 25%" in grade.detail and "t::a: FAILED" in grade.detail
    seen = grade.summary() + grade.patch + grade.detail
    assert "src/module.py" in seen and not shows_hidden(seen) and "test_touched" not in seen
    assert "SECRET-FIXTURE" not in seen and str(tests) not in seen


def test_nothing_to_repair_when_every_attempt_resolved(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    parent_run(root / "job-1" / "attempt-1", monkeypatch)
    spec = review_spec(root)
    spec["parents"][0]["grade"] = {"resolved": True, "progress": 1.0}
    result = run_review(spec, tmp_path / "review", transport_factory=lambda *a, **k: pytest.fail("no model call"))
    assert result["points"] == [] and "nothing to repair" in result["reason"]


def test_review_specs_are_checked_at_submission():
    spec = review_spec("/artifacts")
    spec.pop("artifact_root")   # the worker's to fill in
    assert JobSpec("review", spec).validate()["spec"]["candidates"] == 3
    with pytest.raises(ValueError, match="Unknown ReviewSpec fields"):
        JobSpec("review", {**spec, "prompt": "x"}).validate()
    with pytest.raises(ValueError, match="candidates must be 1"):
        JobSpec("review", {**spec, "guidance": "none"}).validate()
    with pytest.raises(ValueError, match="resolved verdict"):
        JobSpec("review", {**spec, "parents": [{"name": "r", "job_id": "j", "attempt_id": "a", "grade": {}}]}).validate()
    with pytest.raises(ValueError, match="distinct names"):
        JobSpec("review", {**spec, "parents": spec["parents"] * 2}).validate()
    for bound in (0, -1, True, 2.0, "8"):
        with pytest.raises(ValueError, match="max_concurrent_requests must be a positive integer"):
            JobSpec("review", {**spec, "max_concurrent_requests": bound}).validate()
    one_slot = JobSpec("review", {**spec, "max_concurrent_requests": 1}).validate()
    assert one_slot["spec"]["max_concurrent_requests"] == 1
    with pytest.raises(ValueError, match="must supply artifact_root"):
        run_review(spec, Path("/nonexistent"), transport_factory=lambda *a, **k: None)
    validated = ReviewSpec(**review_spec("/artifacts")).validate()
    assert validated["guidance"] == "assistant-turn" and validated["max_concurrent_requests"] == 8


def test_a_review_of_tests_named_after_secrets_is_accepted():
    """A parent's test_statuses is keyed by test ids, which the credential check must not read as keys.

    tfsec's suite has Test_AzureVMWithPasswordAuth and TestIsSensitiveAttribute/secret;
    its FF review was refused at submission, at the server and in the worker's payload.
    """
    spec = review_spec("/artifacts")
    spec["parents"][0]["grade"]["test_statuses"] = {
        "Test_AzureVMWithPasswordAuth": "FAILED", "TestIsSensitiveAttribute/secret": "PASSED"}
    job = JobSpec("review", spec).validate()
    envelope = {"version": 1, "job_id": "review-job", "attempt_id": "attempt-1", "kind": "review",
                "effective_spec": job["spec"], "profile_config": {}, "recovery": None}
    assert encode_payload(envelope, "review-job", "attempt-1")
    with pytest.raises(ValueError, match="inline credentials"):
        JobSpec("review", {**spec, "parents": [{**spec["parents"][0], "grade": {
            **spec["parents"][0]["grade"], "api_key": "sk-literal"}}]}).validate()


def test_each_failed_root_is_a_pipeline_and_the_pipelines_run_side_by_side(tmp_path, monkeypatch):
    """FF: two failed roots, a point each, two candidates per point -- eight
    calls, three in a row. Every call is held until each call of its stage is
    in flight, so a stage whose calls did not all overlap would break its
    barrier. Root 1's calls then wait until root 2's have come back -- its
    analysis and point review recorded, which starting its next calls shows --
    and the review still comes back in root order."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    sizes = {"analyst": 2, "points": 2, "point": 4}
    barriers = {stage: threading.Barrier(size, timeout=60) for stage, size in sizes.items()}

    def hold(call):
        stage = stage_of(call)
        barriers[stage].wait()
        if root_of(call) == 1 and stage == "point":
            assert all(fake.finished(f"point2.1-candidate{sample}").wait(60) for sample in (1, 2))
        elif root_of(call) == 1:
            assert fake.made("points-2" if stage == "analyst" else "point2.1-candidate1").wait(60)

    fake = FakeReviewModel(hold=hold, proposals={**by_sample([["pytest -q", "ls"]], attempt=1),
                                                 **by_sample([["git diff", "git diff"]], attempt=2)})
    result = run_review(spec, tmp_path / "review", transport_factory=fake.factory)
    assert fake.most_in_flight == 4, "the four candidate samples were in flight together"
    assert len(fake.requests) == result["review_requests"] == 8
    for stage, size in sizes.items():
        roots = [root_of(call) for call in fake.answered if stage_of(call) == stage]
        assert roots == [2] * (size // 2) + [1] * (size // 2), f"root 2's {stage} calls answered first"
    assert list(result["reports"]) == result["analysed"] == ["root-1", "root-2"]
    assert [point["base"] for point in result["points"]] == ["root-1", "root-2"]
    assert [[(commands_of(c["assistant_turn"]), c["multiplicity"]) for c in point["candidates"]]
            for point in result["points"]] == [[(("pytest -q",), 1), (("ls",), 1)], [(("git diff",), 2)]]
    assert {path.name for path in (tmp_path / "review").glob("plan-*.json")} == {
        "plan-points-1.json", "plan-points-2.json", *(f"plan-point{root}.1-candidate{sample}.json"
                                                      for root in (1, 2) for sample in (1, 2))}
    assert not review_threads()


@pytest.mark.parametrize("per_attempt", [False, True], ids=["across-attempts", "points-per-attempt"])
def test_one_slot_makes_the_calls_in_stage_order_and_more_slots_make_the_same_review(tmp_path, monkeypatch,
                                                                                    per_attempt):
    """With one slot the calls run one after another: every analysis, every
    point review, then the samples point by point. With eight they overlap
    and come back in another order -- root 2's analysis first, and at every
    point the second sample, then the first, then the third, so a merge in
    that order would put the odd proposal first -- and the review, its
    reports' order and every plan file are the same."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=2, points_per_attempt=per_attempt, candidates=3)

    def propose(call):   # the first and third sample at each point agree
        return f"ls {call}" if call.endswith("candidate2") else "pytest -q"

    one = FakeReviewModel(proposals=propose)
    serial = run_review({**spec, "max_concurrent_requests": 1}, tmp_path / "one", transport_factory=one.factory)
    if per_attempt:
        point_reviews = ["points-1", "points-2"]
        samples = [f"point{root}.{point}-candidate{sample}"
                   for root in (1, 2) for point in (1, 2) for sample in (1, 2, 3)]
    else:   # the fake's one point review picks both points in root 1
        point_reviews = ["points"]
        samples = [f"point{point}-candidate{sample}" for point in (1, 2) for sample in (1, 2, 3)]
    assert one.calls == ["analyst-1", "analyst-2", *point_reviews, *samples] and one.most_in_flight == 1
    assert [[c["multiplicity"] for c in point["candidates"]] for point in serial["points"]] == [[2, 1]] * len(
        serial["points"]) and len(serial["points"]) == (4 if per_attempt else 2)

    analysts, recorded = threading.Barrier(2, timeout=60), Recorded(monkeypatch)

    def hold(call):
        if stage_of(call) == "analyst":
            analysts.wait()
            if call == "analyst-1":   # per attempt: until root 2's analysis is recorded and its point review made
                assert (many.made("points-2") if per_attempt else many.finished("analyst-2")).wait(60)
        elif stage_of(call) == "point":   # the review records a point's samples 2, 1, 3
            after = {"1": "2", "3": "1"}.get(call[-1])
            if after:
                assert recorded.returned(call[:-1] + after).wait(60)

    many = FakeReviewModel(proposals=propose, hold=hold)
    concurrent = run_review({**spec, "max_concurrent_requests": 8}, tmp_path / "eight", transport_factory=many.factory)
    assert many.answered.index("analyst-2") < many.answered.index("analyst-1") and many.most_in_flight > 1
    for first in samples[::3]:
        assert [call for call in many.answered if call[:-1] == first[:-1]] == [
            first[:-1] + sample for sample in "213"]
    assert json.dumps(concurrent) == json.dumps(serial), "the same review, its keys in the same order"
    assert sorted(many.calls) == sorted(one.calls)

    def plans(directory):
        return {path.name: json.loads(path.read_text()) for path in directory.glob("plan-*.json")}

    assert plans(tmp_path / "eight") == plans(tmp_path / "one") and len(plans(tmp_path / "one")) == len(
        point_reviews) + len(samples)


@pytest.mark.parametrize("per_attempt", [False, True], ids=["across-attempts", "points-per-attempt"])
def test_each_candidate_review_is_offered_its_own_point_only(tmp_path, monkeypatch, per_attempt):
    """Two points from each point review, two samples at each: every sample's
    plan offers its own point's step and nothing else, and chose it."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=2, points_per_attempt=per_attempt, candidates=2)
    result = run_review(spec, tmp_path / "review", transport_factory=FakeReviewModel().factory)
    points, seen = result["points"], {}
    assert len(points) == (4 if per_attempt else 2) and points[0]["step"] != points[1]["step"]
    for index, point in enumerate(points, 1):
        root = point["base"].removeprefix("root-")
        seen[root] = seen.get(root, 0) + 1
        name = f"{root}.{seen[root]}" if per_attempt else str(index)
        for sample in (1, 2):
            plan = json.loads((tmp_path / "review" / f"plan-point{name}-candidate{sample}.json").read_text())
            assert plan["available_steps"] == {point["base"]: [point["step"]]}, (name, sample)
            assert [branch["branch_step"] for branch in plan["selected_branches"]] == [point["step"]]


def test_across_attempts_the_samples_at_every_point_run_together(tmp_path, monkeypatch):
    """Without points_per_attempt the analyses overlap, the one point review
    waits for both, and then the six samples at its two points are in flight
    together: each is held until all six are."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=2, candidates=3)
    analysts, samples = threading.Barrier(2, timeout=60), threading.Barrier(6, timeout=60)

    def hold(call):
        if stage_of(call) == "analyst":
            analysts.wait()
        elif stage_of(call) == "points":
            assert fake.finished("analyst-1").is_set() and fake.finished("analyst-2").is_set()
        else:
            samples.wait()

    fake = FakeReviewModel(hold=hold)
    result = run_review(spec, tmp_path / "review", transport_factory=fake.factory)
    assert fake.most_in_flight == 6 and len(fake.requests) == result["review_requests"] == 2 + 1 + 6
    assert [[c["multiplicity"] for c in point["candidates"]] for point in result["points"]] == [[3], [3]]


def test_the_bound_holds_and_with_one_slot_nothing_starts_after_a_failure(tmp_path, monkeypatch):
    """Two slots: the analyses run together, then the four samples two at a
    time. One slot, and root 1's analysis fails: no other call is even made."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    pairs = threading.Barrier(2, timeout=60)

    def hold(call):
        if stage_of(call) != "points":
            pairs.wait()

    two = FakeReviewModel(hold=hold)
    result = run_review({**spec, "max_concurrent_requests": 2}, tmp_path / "two", transport_factory=two.factory)
    assert two.most_in_flight == 2 and len(two.requests) == result["review_requests"] == 8

    one = FakeReviewModel(fail={"analyst-1": RuntimeError("root 1 refused")})
    with pytest.raises(RuntimeError) as raised:
        run_review({**spec, "max_concurrent_requests": 1}, tmp_path / "one", transport_factory=one.factory)
    assert str(raised.value) == "root 1 refused" and one.transports == ["analyst-1"]


def test_no_call_thread_outlives_the_review(tmp_path, monkeypatch):
    """A call's thread puts its result on the review's queue and then ends;
    the review joins it before it goes on, so none is left once the review
    returns -- here even with every thread lingering after its put."""
    root = tmp_path / "artifacts"
    parent_run(root / "job-1" / "attempt-1", monkeypatch)

    class Lingering(queue.SimpleQueue):
        def put(self, item, block=True, timeout=None):
            super().put(item, block, timeout)
            threading.Event().wait(0.2)

    monkeypatch.setattr(review_module, "queue", SimpleNamespace(SimpleQueue=Lingering))
    result = run_review(review_spec(root, guidance="none", candidates=1), tmp_path / "review",
                        transport_factory=FakeReviewModel().factory)
    assert len(result["points"]) == 2 and not review_threads()


def test_a_transport_shared_between_calls_counts_once(tmp_path, monkeypatch):
    """A factory may hand every call the same transport, as one per job was
    the rule before the calls ran side by side: review_requests counts the
    requests sent, not that transport once per call."""
    root = tmp_path / "artifacts"
    parent_run(root / "job-1" / "attempt-1", monkeypatch)
    monkeypatch.setenv("FIXTURE_KEY", "fixture-key")
    request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
    sent = []

    def post(url, **kwargs):
        prompt = kwargs["json"]["messages"][0]["content"]
        sent.append(prompt)
        return httpx.Response(200, request=request, json={"choices": [{"message": {"content": answer(prompt)}}]})

    monkeypatch.setattr("sprout.methods.review_transport.httpx.post", post)
    shared = []

    def factory(directory, **settings):
        if not shared:
            shared.append(ReviewTransport(tmp_path / "receipts", **settings))
        return shared[0]

    result = run_review(review_spec(root), tmp_path / "review", transport_factory=factory)
    assert result["review_requests"] == len(sent) == shared[0].sequence == 1 + 1 + 2 * 3


def test_receipts_are_named_by_call_whatever_order_the_calls_finish_in(tmp_path, monkeypatch):
    """The real transport behind every call: each call numbers its receipts in
    a directory of its own, so the receipts are the same with one slot as with
    eight. With eight, the four candidates' first tries are out together
    before each times out, and each retry still names its own call's first try."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    monkeypatch.setenv("FIXTURE_KEY", "fixture-key")
    request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")
    run = {"reversed": False, "analysed": [], "root 2 analysed": threading.Event()}
    together = threading.Barrier(4, timeout=60)

    def post(url, **kwargs):
        body = kwargs["json"]
        prompt = body["messages"][0]["content"]
        root = (1 if "FAIL_TO_PASS): 50%" in prompt else 2) if ANALYST_OPENING in prompt else None
        if root == 1 and run["reversed"]:   # root 1's analysis waits for root 2's
            assert run["root 2 analysed"].wait(60)
        if root is None and POINT_OPENING not in prompt and body["max_tokens"] == 16384:
            if run["reversed"]:
                together.wait()
            raise httpx.ReadTimeout("timed out")   # every candidate's first try
        reply = httpx.Response(200, request=request, json={"choices": [{"message": {"content": answer(prompt)}}]})
        if root is not None:
            run["analysed"].append(root)
        if root == 2:
            run["root 2 analysed"].set()
        return reply

    monkeypatch.setattr("sprout.methods.review_transport.httpx.post", post)

    def receipts(directory):
        return {str(path.relative_to(directory / "reviews")): json.loads(path.read_text())
                for path in (directory / "reviews").rglob("review-*.json")}

    one = run_review({**spec, "max_concurrent_requests": 1}, tmp_path / "one")
    assert run["analysed"] == [1, 2]
    run.update({"reversed": True, "analysed": [], "root 2 analysed": threading.Event()})
    eight = run_review({**spec, "max_concurrent_requests": 8}, tmp_path / "eight")
    assert run["analysed"] == [2, 1], "eight slots: root 2's analysis came back first"
    assert json.dumps(eight) == json.dumps(one) and one["review_requests"] == 2 + 2 + 4 * 2
    first, second = receipts(tmp_path / "one"), receipts(tmp_path / "eight")
    calls = ["analyst-1", "analyst-2", "points-1", "points-2"]
    candidates = [f"point{root}.1-candidate{sample}" for root in (1, 2) for sample in (1, 2)]
    assert sorted(first) == sorted(second) == sorted(
        [f"{call}/review-001.json" for call in calls]
        + [f"{call}/review-00{n}.json" for call in candidates for n in (1, 2)])
    for name, receipt in second.items():
        assert receipt["request"] == first[name]["request"], name
    for call in candidates:
        tried, retried = second[f"{call}/review-001.json"], second[f"{call}/review-002.json"]
        assert tried["error"].startswith("ReadTimeout") and "retry_of" not in tried
        assert (retried["retry_of"], retried["retry_reason"], retried["request"]["max_tokens"]) == (
            "review-001.json", "timeout", 8192)
    assert all(point["candidates"] for point in eight["points"])


def test_the_review_fails_with_the_error_one_slot_fails_with_whatever_the_timing(tmp_path, monkeypatch):
    """Both analyses fail. With one slot root 1's runs first and is the error;
    root 2's never runs. With eight, root 1's analysis does not even send its
    first request before the review has recorded root 2's failure, and the
    review still fails with root 1's error, root 2's noted on it: a failure
    stops only the calls after it, and root 1's analysis comes before."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)

    def refusals():
        return {"analyst-1": RuntimeError("root 1 refused"), "analyst-2": RuntimeError("root 2 refused")}

    one = FakeReviewModel(fail=refusals())
    with pytest.raises(RuntimeError) as serial:
        run_review({**spec, "max_concurrent_requests": 1}, tmp_path / "one", transport_factory=one.factory)
    assert str(serial.value) == "root 1 refused" and serial.value.__notes__ == ["in review call analyst-1"]
    assert one.transports == ["analyst-1"]

    recorded = Recorded(monkeypatch)
    branch_points = review_module.available_branch_points

    def late(journal):   # what root 1's analysis reads just before its first request
        if "job-1" in str(journal):
            assert recorded.failed("analyst-2").wait(60)
        return branch_points(journal)

    monkeypatch.setattr(review_module, "available_branch_points", late)
    many = FakeReviewModel(fail=refusals())
    with pytest.raises(RuntimeError) as concurrent:
        run_review(spec, tmp_path / "eight", transport_factory=many.factory)
    assert str(concurrent.value) == "root 1 refused"
    assert concurrent.value.__notes__ == [
        "in review call analyst-1", "review call analyst-2 failed too: RuntimeError: root 2 refused"]
    assert many.answered == ["analyst-2", "analyst-1"] and many.in_flight == 0
    assert not list((tmp_path / "eight").glob("plan-*.json")) and not review_threads()


def test_a_call_before_the_failure_still_starts_and_its_error_wins(tmp_path, monkeypatch):
    """Root 2's point review fails while root 1's analysis is still out. Root
    1's point review comes before it -- with one slot it runs first and fails
    first -- so it still starts once root 1's analysis is in, and its error is
    the review's."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    original = review_module.review

    def review(*args, plan_path, **kwargs):   # a point review's request error is its plan's; this raises
        if plan_path.name in ("plan-points-1.json", "plan-points-2.json"):
            raise OSError(f"no room for {plan_path.name}")
        return original(*args, plan_path=plan_path, **kwargs)

    monkeypatch.setattr(review_module, "review", review)
    one = FakeReviewModel()
    with pytest.raises(OSError) as serial:
        run_review({**spec, "max_concurrent_requests": 1}, tmp_path / "one", transport_factory=one.factory)
    assert str(serial.value) == "no room for plan-points-1.json"
    assert one.transports == ["analyst-1", "analyst-2", "points-1"]

    recorded = Recorded(monkeypatch)

    def hold(call):
        if call == "analyst-1":
            assert recorded.failed("points-2").wait(60)

    many = FakeReviewModel(hold=hold)
    with pytest.raises(OSError) as concurrent:
        run_review(spec, tmp_path / "eight", transport_factory=many.factory)
    assert str(concurrent.value) == "no room for plan-points-1.json"
    assert concurrent.value.__notes__ == [
        "in review call points-1", "review call points-2 failed too: OSError: no room for plan-points-2.json"]
    assert many.transports == ["analyst-1", "analyst-2", "points-2", "points-1"] and not review_threads()


def test_an_earlier_stage_outranks_an_earlier_root(tmp_path, monkeypatch):
    """Root 1's point review raises after root 2's analysis has failed. One
    slot never gets that far -- every analysis comes before every point
    review -- and fails with root 2's error; so does a review with eight,
    which notes root 1's."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    one = FakeReviewModel(fail={"analyst-2": RuntimeError("root 2 refused")})
    with pytest.raises(RuntimeError) as serial:
        run_review({**spec, "max_concurrent_requests": 1}, tmp_path / "one", transport_factory=one.factory)
    assert str(serial.value) == "root 2 refused" and one.transports == ["analyst-1", "analyst-2"]

    recorded = Recorded(monkeypatch)
    original = review_module.review

    def review(*args, plan_path, **kwargs):
        if plan_path.name == "plan-points-1.json":
            assert recorded.failed("analyst-2").wait(60)
            raise OSError("disk full")
        return original(*args, plan_path=plan_path, **kwargs)

    monkeypatch.setattr(review_module, "review", review)

    def hold(call):   # root 2's analysis fails once root 1's point review is under way
        if call == "analyst-2":
            assert many.made("points-1").wait(60)

    many = FakeReviewModel(hold=hold, fail={"analyst-2": RuntimeError("root 2 refused")})
    with pytest.raises(RuntimeError) as concurrent:
        run_review(spec, tmp_path / "eight", transport_factory=many.factory)
    assert str(concurrent.value) == "root 2 refused"
    assert concurrent.value.__notes__ == [
        "in review call analyst-2", "review call points-1 failed too: OSError: disk full"]
    assert many.transports == ["analyst-1", "analyst-2", "points-1"] and not review_threads()


def test_a_failure_stops_the_running_calls_after_it_and_their_plans_say_why(tmp_path, monkeypatch):
    """Root 2's point review is in flight when root 1's analysis fails. It
    comes after the failure, so the correction its reply needs is never asked
    for, its plan says why, and nothing more of root 2 runs."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    recorded = Recorded(monkeypatch)

    def hold(call):
        if call == "analyst-1":   # fails once root 2's point review is in flight
            assert fake.arrived("points-2").wait(60)
        elif call == "points-2":   # answers, with a reply to correct, once the failure is recorded
            assert recorded.failed("analyst-1").wait(60)

    fake = FakeReviewModel(hold=hold, fail={"analyst-1": RuntimeError("root 1 refused")},
                           replies={"points-2": ["not a plan"]})
    with pytest.raises(RuntimeError) as raised:
        run_review(spec, tmp_path / "review", transport_factory=fake.factory)
    assert str(raised.value) == "root 1 refused" and raised.value.__notes__ == ["in review call analyst-1"]
    assert fake.calls.count("points-2") == 1, "the correction was never asked for"
    assert fake.transports == ["analyst-1", "analyst-2", "points-2"] and not review_threads()
    plan = json.loads((tmp_path / "review" / "plan-points-2.json").read_text())
    assert plan["validation_error"] == "reviewer failed: review call points-2 stopped: review call analyst-1 failed"


def test_a_failure_cuts_short_the_retries_of_a_running_call(tmp_path, monkeypatch):
    """The real transport behind every call. Root 2's first try is out when
    root 1's analysis is refused (401), and then comes back 502. Its
    transport would try again after 10, 30 and 60 s; stopped, it waits out
    no backoff and sends nothing more, and the review fails with root 1's
    error."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    monkeypatch.setenv("FIXTURE_KEY", "fixture-key")
    recorded = Recorded(monkeypatch)
    slept, tries, out = [], [], threading.Event()
    monkeypatch.setattr(review_transport, "time", SimpleNamespace(time=time.time, sleep=slept.append))
    request = httpx.Request("POST", "http://127.0.0.1:1/v1/chat/completions")

    def post(url, **kwargs):
        root = 1 if "FAIL_TO_PASS): 50%" in kwargs["json"]["messages"][0]["content"] else 2
        tries.append(root)
        if root == 1:   # refused once root 2's first try is out
            assert out.wait(60)
            return httpx.Response(401, request=request, text="Unauthorized")
        out.set()
        assert recorded.failed("analyst-1").wait(60)
        return httpx.Response(502, request=request, text="Bad Gateway")

    monkeypatch.setattr("sprout.methods.review_transport.httpx.post", post)
    with pytest.raises(httpx.HTTPStatusError) as raised:
        run_review(spec, tmp_path / "review")
    assert raised.value.response.status_code == 401 and raised.value.__notes__ == ["in review call analyst-1"]
    assert sorted(tries) == [1, 2] and slept == [], "root 2 was tried once and slept through no backoff"
    receipts = tmp_path / "review" / "reviews" / "analyst-2"
    assert [path.name for path in receipts.glob("review-*.json")] == ["review-001.json"]
    assert json.loads((receipts / "review-001.json").read_text())["error"].startswith("HTTPStatusError")
    assert not review_threads()


def test_a_failed_call_fails_the_review_once_every_running_call_has_returned(tmp_path, monkeypatch):
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)

    # A failure known while root 1's analysis is in flight: that analysis is
    # waited for and nothing more starts, root 1's point review included.
    late = FakeReviewModel()

    def factory(directory, **settings):
        if Path(directory).name == "analyst-2":
            raise ValueError("no transport for root 2")
        return late.factory(directory, **settings)

    with pytest.raises(ValueError, match="no transport for root 2"):
        run_review(spec, tmp_path / "late", transport_factory=factory)
    assert late.transports == ["analyst-1"] and set(late.calls) <= {"analyst-1"}, "nothing started after the failure"
    assert late.in_flight == 0 and not list((tmp_path / "late").glob("plan-*.json")) and not review_threads()

    # A candidate review that raises fails the review: no result comes back without it.
    original = review_module.review

    def review(*args, plan_path, **kwargs):
        if plan_path.name == "plan-point2.1-candidate2.json":
            raise OSError("disk full")
        return original(*args, plan_path=plan_path, **kwargs)

    monkeypatch.setattr(review_module, "review", review)
    candidate = FakeReviewModel()
    with pytest.raises(OSError, match="disk full"):
        run_review(spec, tmp_path / "candidate", transport_factory=candidate.factory)
    assert candidate.in_flight == 0 and not review_threads()


def test_a_stopped_call_asks_for_nothing_says_why_and_is_never_the_error(tmp_path):
    """A stopped call is refused its next request, and told why; its transport
    is given the same stop, so it retries nothing either. A refusal is another
    call's doing, so it is never the error raised; the others rank in ``Call``
    order, an earlier stage before an earlier root."""
    fake = FakeReviewModel()
    calls = review_module._Review(ReviewSpec(**review_spec(tmp_path)), [], tmp_path, fake.factory)
    stop = review_module._Stop()
    request = calls._transport(Call(ANALYSIS, 2), stop)
    assert fake.settings[0]["stop"] is stop.event
    stop("review call analyst-1 failed")
    stop("the review was interrupted")   # the first reason stands
    with pytest.raises(ReviewStopped) as refused:
        request("fixture", "prompt")
    assert str(refused.value) == "review call analyst-2 stopped: review call analyst-1 failed"
    assert fake.requests == [] and calls.requests() == 0

    class Retrying:
        """A transport that meets the stop itself, before a retry."""
        sequence = 1

        def __call__(self, model, prompt):
            stop("the review was interrupted")
            raise RequestStopped("no further request was sent")

    stop = review_module._Stop()
    calls.transport_factory = lambda directory, **settings: Retrying()
    with pytest.raises(ReviewStopped) as refused:
        calls._transport(Call(ANALYSIS, 3), stop)("fixture", "prompt")
    assert str(refused.value) == "review call analyst-3 stopped: the review was interrupted"

    stopped, refused, later = ReviewStopped("stopped"), RuntimeError("refused"), OSError("later")
    first = review_module._Review._first([(Call(ANALYSIS, 1), stopped), (Call(CANDIDATE, 1, 1, 1), later),
                                          (Call(POINTS, 2), refused)])
    assert first is refused and first.__notes__ == [
        "in review call points-2", "review call point1.1-candidate1 failed too: OSError: later"]


def test_an_interrupted_review_leaves_at_once_and_its_running_calls_ask_for_nothing_more(tmp_path, monkeypatch):
    """The worker stops a child with SIGTERM, which the child raises as
    KeyboardInterrupt on its main thread. The review must not wait out a
    request that may last timeout_s: it leaves with root 1's analysis in
    flight, and that analysis, given a reply to correct, asks for nothing.
    Nor does a point review in flight, whose plan says why."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    arrived, release = threading.Event(), threading.Event()

    def hold(call):
        arrived.set()
        assert release.wait(60)

    fake = FakeReviewModel(hold=hold, replies={"analyst-1": ["not a report"]})

    def factory(directory, **settings):
        if Path(directory).name == "analyst-2":
            assert arrived.wait(60)   # root 1's analysis is in flight
            raise KeyboardInterrupt("Worker requested shutdown")
        return fake.factory(directory, **settings)

    with pytest.raises(KeyboardInterrupt):
        run_review(spec, tmp_path / "review", transport_factory=factory)
    assert not fake.finished("analyst-1").is_set(), "the review left before root 1's analysis came back"
    release.set()
    for thread in review_threads():
        thread.join(60)
    assert not review_threads() and fake.calls == ["analyst-1"], "the analysis never asked for its correction"

    release = threading.Event()

    def hold_points(call):
        if call == "analyst-2":   # root 2's point review is made once root 1's is in flight
            assert points.arrived("points-1").wait(60)
        elif call == "points-1":
            assert release.wait(60)

    points = FakeReviewModel(hold=hold_points, replies={"points-1": ["not a plan"]})

    def interrupting(directory, **settings):
        if Path(directory).name == "points-2":
            raise KeyboardInterrupt("Worker requested shutdown")
        return points.factory(directory, **settings)

    with pytest.raises(KeyboardInterrupt):
        run_review(spec, tmp_path / "points", transport_factory=interrupting)
    assert not points.finished("points-1").is_set()
    release.set()
    for thread in review_threads():
        thread.join(60)
    assert not review_threads() and points.calls.count("points-1") == 1
    plan = json.loads((tmp_path / "points" / "plan-points-1.json").read_text())
    assert plan["validation_error"] == "reviewer failed: review call points-1 stopped: the review was interrupted"


def test_an_interrupt_keeps_the_failures_already_known(tmp_path, monkeypatch, caplog):
    """Root 1's analysis has failed and the review waits for root 2's, which
    comes back -- and the worker's SIGTERM lands right then. The interrupt
    leaves with root 1's failure noted on it, and logged with its traceback,
    instead of dropping it."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    recorded = Recorded(monkeypatch)
    finished = review_module._Review._finished

    def interrupted(review, call, value):
        if call.name == "analyst-2":
            raise KeyboardInterrupt("Worker requested shutdown")
        return finished(review, call, value)

    monkeypatch.setattr(review_module._Review, "_finished", interrupted)

    def hold(call):
        if call == "analyst-1":   # fails once root 2's analysis is in flight
            assert fake.arrived("analyst-2").wait(60)
        elif call == "analyst-2":   # comes back once the failure is recorded
            assert recorded.failed("analyst-1").wait(60)

    refused = RuntimeError("root 1 refused")
    fake = FakeReviewModel(hold=hold, fail={"analyst-1": refused})
    with caplog.at_level(logging.ERROR, logger=review_module.__name__), pytest.raises(KeyboardInterrupt) as raised:
        run_review(spec, tmp_path / "review", transport_factory=fake.factory)
    assert raised.value.__notes__ == ["review call analyst-1 had failed: RuntimeError: root 1 refused"]
    [logged] = [record for record in caplog.records if record.name == review_module.__name__]
    assert "analyst-1" in logged.getMessage() and logged.exc_info[1] is refused
    assert not review_threads()


def test_a_stopped_review_child_exits_within_the_workers_grace(tmp_path, monkeypatch):
    """The worker stops a review child as it stops any attempt: SIGTERM to its
    process group (``Watchdog``, ``stop_process``), then five seconds for it to
    exit before the job is quarantined (``Worker.run_claimed``). Both analyses
    are in flight here, to an endpoint that never answers: the real child must
    write its outcome and exit within that grace, not when they come back."""
    spec = two_failed_roots(tmp_path, monkeypatch, points=1, points_per_attempt=True, candidates=2)
    server = socket.create_server(("127.0.0.1", 0))
    held, both = [], threading.Event()

    def accept():
        while len(held) < 2:
            try:
                held.append(server.accept()[0])
            except OSError:
                return
        both.set()

    threading.Thread(target=accept, daemon=True).start()
    spec["model_endpoint"] = "http://127.0.0.1:%d" % server.getsockname()[1]
    directory = tmp_path / "jobs" / "review-job" / "attempt-1"
    directory.mkdir(parents=True)
    envelope = {"version": 1, "job_id": "review-job", "attempt_id": "attempt-1", "kind": "review",
                "effective_spec": spec, "profile_config": {}, "recovery": None}
    repo = Path(__file__).resolve().parents[2]
    log_path = tmp_path / "worker.log"
    with log_path.open("w") as log:
        process = subprocess.Popen(
            [sys.executable, "-m", "sprout.worker.jobs.rollout", str(directory), digest(envelope)], cwd=repo,
            env={**os.environ, "FIXTURE_KEY": "fixture-key", "PYTHONPATH": f"{repo / 'src'}:{repo}"},
            stdin=subprocess.PIPE, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    identity = process_identity(process.pid)
    watchdog = Watchdog(stop_process, threading.Event(), 60)
    try:
        send_payload(process.stdin, encode_payload(envelope, "review-job", "attempt-1"), timeout_s=30)
        assert both.wait(60), log_path.read_text()
        watchdog.arm(identity, time.monotonic(), running=lambda: process.poll() is None)   # the deadline: now
        eventually(lambda: watchdog.reason is not None)
        process.wait(timeout=5)   # all Worker.run_claimed allows it
    finally:
        watchdog.close()
        stop_process(identity)
        server.close()
        for connection in held:
            connection.close()
    assert watchdog.reason == "timeout" and process.returncode == 0, "it exited by itself, not by SIGKILL"
    outcome = json.loads((directory / "outcome.json").read_text())
    assert (outcome["status"], outcome["error"]) == ("error", "KeyboardInterrupt: Worker requested shutdown")
    assert "Traceback" in log_path.read_text(), "the child's error goes to worker.log in full"

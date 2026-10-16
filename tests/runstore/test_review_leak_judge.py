"""The review's leak judge: every repair turn is checked before it is taken, and a flagged one goes back."""

import json

import pytest

from sprout.worker.jobs.review import job as review_module
from sprout.methods.leak_judge import JevError, LeakJudgement
from sprout.worker.jobs.review.job import run_review
from sprout.runstore.specs import ReviewSpec
from tests.harness.test_mini_swe import pytestmark
from tests.runstore.test_mini_native import parent_run
from tests.runstore.test_review_job import FakeReviewModel, review_spec

_ = pytestmark  # the parent run needs mini installed, like the review tests it reuses

FIXTURE = "FIXTURE-ABC-123"
LEAKING = f"python -c \"print(convert('{FIXTURE}'))\""


class FakeJudge:
    """Stands in for Jev: a turn that holds ``FIXTURE`` scores ``high``. Records what it was shown."""

    def __init__(self, high=0.95, error=None):
        self.high, self.error, self.seen, self.closed = high, error, [], 0

    def judge(self, turn, *, task_statement, hidden, visible):
        self.seen.append({"turn": turn, "task_statement": task_statement, "hidden": hidden, "visible": visible})
        if self.error:
            raise JevError(self.error)
        score = self.high if FIXTURE in json.dumps(turn) else 0.1
        return LeakJudgement(score, ((FIXTURE, score),), "jev-fake")

    def close(self):
        self.closed += 1


def judged_review(tmp_path, monkeypatch, commands, *, judge, **changes):
    """A one-point, one-candidate review with a leak judge, the grade naming a test
    patch that holds ``FIXTURE``; ``commands`` are the candidate's successive
    proposals, the last repeated."""
    root = tmp_path / "artifacts"
    parent_run(root / "job-1" / "attempt-1", monkeypatch)
    patch = tmp_path / "grade" / "test.patch"
    patch.parent.mkdir(parents=True)
    patch.write_text(f"+++ b/tests/test_convert.py\n+    assert convert('{FIXTURE}') == 'ok'\n")
    spec = review_spec(root, points=1, candidates=1, **changes)
    spec["parents"][0]["grade"]["test_patch"] = str(patch)
    queue = list(commands)

    def propose(call):
        command = queue.pop(0) if len(queue) > 1 else queue[0]
        if isinstance(command, BaseException):
            raise command
        return command

    fake = FakeReviewModel(proposals=propose)
    result = run_review(spec, tmp_path / "review", transport_factory=fake.factory,
                        judge_factory=lambda config: judge)
    plan = json.loads((tmp_path / "review" / "plan-point1-candidate1.json").read_text())
    return result, fake, plan


def test_a_flagged_turn_goes_back_to_the_reviewer_without_naming_what_was_flagged(tmp_path, monkeypatch):
    judge = FakeJudge()
    result, fake, plan = judged_review(tmp_path, monkeypatch, [LEAKING, "pytest -q"], judge=judge,
                                       leak_judge={"resamples": 2}, max_attempts=1)
    (candidate,) = result["points"][0]["candidates"]
    assert FIXTURE not in json.dumps(candidate["assistant_turn"]) and "pytest -q" in json.dumps(candidate)
    assert candidate["leak_judge"]["leak"] is False and candidate["leak_judge"]["leak_terms"] == []
    assert candidate["leak_judge"]["jev"]["score"] == pytest.approx(0.1)
    # One rejection, not counted against max_attempts (1): the second request is the rewrite.
    rejected, taken = plan["review_attempts"]
    assert rejected["status"] == "turn_rejected" and taken["status"] == "validated"
    assert rejected["turn_checks"][0]["leak"] is True and plan["turn_rejections"] == 1
    assert plan["selected_branches"][0]["turn_check"]["leak"] is False
    # The reviewer is told which branch and why, never the flagged span: the only
    # copy of it the rewrite sees is in its own earlier reply.
    feedback = rejected["feedback"]
    assert "branch 1" in feedback and "could not know" in feedback and FIXTURE not in feedback
    prompts = [prompt for call, prompt in fake.requests if call == "point1-candidate1"]
    assert len(prompts) == 2 and FIXTURE not in prompts[0]
    assert FIXTURE not in prompts[1].split("## Validation feedback")[1]
    # The judge read the test patch the grade applied, the verdict as the analyst
    # read it, the task statement and h_s; it was closed after the call.
    seen = judge.seen[0]
    assert FIXTURE in seen["hidden"].test_patch and "t::a: FAILED" in seen["hidden"].verifier_output
    assert seen["task_statement"] == "Fix the failing test." and "echo early" in seen["visible"]
    assert judge.closed == 1


def test_a_turn_still_flagged_after_its_resamples_is_taken_and_marked(tmp_path, monkeypatch):
    result, _, plan = judged_review(tmp_path, monkeypatch, [LEAKING], judge=FakeJudge(),
                                    leak_judge={"resamples": 1})
    (candidate,) = result["points"][0]["candidates"]
    assert FIXTURE in json.dumps(candidate["assistant_turn"])
    assert candidate["leak_judge"]["leak"] is True and candidate["leak_judge"]["jev"]["leak"] is True
    assert [a["status"] for a in plan["review_attempts"]] == ["turn_rejected", "validated"]
    assert plan["turn_rejections"] == 1 and plan["turn_resamples"] == 1


def test_a_hidden_test_name_is_flagged_too(tmp_path, monkeypatch):
    result, _, plan = judged_review(tmp_path, monkeypatch, ["pytest -q -k test_secret_behaviour"],
                                    judge=FakeJudge(), leak_judge={"resamples": 0})
    (candidate,) = result["points"][0]["candidates"]
    assert candidate["leak_judge"]["leak"] is True and candidate["leak_judge"]["jev"]["leak"] is False
    assert candidate["leak_judge"]["leak_terms"] == ["test_secret_behaviour"] == candidate["leak_terms"]
    assert [a["status"] for a in plan["review_attempts"]] == ["validated"]


def test_a_judge_error_is_unknown_not_clean_and_sends_nothing_back(tmp_path, monkeypatch):
    result, _, plan = judged_review(tmp_path, monkeypatch, ["pytest -q"], judge=FakeJudge(error="HTTP 503"),
                                    leak_judge={})
    (candidate,) = result["points"][0]["candidates"]
    assert candidate["leak_judge"]["leak"] is None and "HTTP 503" in candidate["leak_judge"]["jev"]["error"]
    assert [a["status"] for a in plan["review_attempts"]] == ["validated"]


def test_without_a_leak_judge_nothing_is_checked_and_candidates_keep_their_shape(tmp_path, monkeypatch):
    root = tmp_path / "artifacts"
    parent_run(root / "job-1" / "attempt-1", monkeypatch)
    result = run_review(review_spec(root, points=1, candidates=1), tmp_path / "review",
                        transport_factory=FakeReviewModel(proposals=LEAKING).factory,
                        judge_factory=lambda config: pytest.fail("no judge without leak_judge"))
    (candidate,) = result["points"][0]["candidates"]
    assert set(candidate) == {"assistant_turn", "why", "multiplicity", "leak_terms"}
    plan = json.loads((tmp_path / "review" / "plan-point1-candidate1.json").read_text())
    assert "turn_resamples" not in plan and "turn_check" not in plan["selected_branches"][0]


def test_leak_judge_settings_are_checked_and_the_key_is_read_on_the_worker(tmp_path, monkeypatch):
    base = review_spec(tmp_path)
    for bad in ({"threshold": 0}, {"threshold": 1.5}, {"resamples": -1}, {"resamples": 1.0},
                {"model": ""}, {"api_key_env": "NOT A NAME"}, {"other": 1}, ["threshold"]):
        with pytest.raises(ValueError, match="leak_judge"):
            ReviewSpec(**{**base, "leak_judge": bad}).validate()
    ReviewSpec(**{**base, "leak_judge": {"threshold": 0.8, "model": "jev-1.13.0", "resamples": 3,
                                         "api_key_env": "TS_KEY"}}).validate()
    monkeypatch.delenv("TS_KEY", raising=False)
    with pytest.raises(ValueError, match=r"\$TS_KEY"):
        review_module.jev_judge({"api_key_env": "TS_KEY"})
    monkeypatch.setenv("TS_KEY", "k")
    judge = review_module.jev_judge({"api_key_env": "TS_KEY", "threshold": 0.8})
    assert judge.key == "k" and judge.threshold == 0.8
    # Bounded to fit inside the review's own deadline.
    assert judge.timeout_s * judge.max_attempts <= 120
    judge.close()


@pytest.mark.parametrize("rewrite", [RuntimeError("endpoint gone"), ""], ids=["request fails", "rewrite invalid"])
def test_a_rewrite_that_yields_no_plan_falls_back_to_the_flagged_turn(tmp_path, monkeypatch, rewrite):
    result, _, plan = judged_review(tmp_path, monkeypatch, [LEAKING, rewrite], judge=FakeJudge(),
                                    leak_judge={"resamples": 2}, max_attempts=1)
    (candidate,) = result["points"][0]["candidates"]
    assert FIXTURE in json.dumps(candidate["assistant_turn"]), "the flagged turn, not nothing"
    assert candidate["leak_judge"]["leak"] is True
    assert plan["taken_flagged"]["attempt"] == 1 and "validation_error" not in plan
    assert plan["review_attempts"][0]["status"] == "turn_rejected"


def test_an_empty_leak_judge_config_takes_the_default_resamples(tmp_path, monkeypatch):
    _, _, plan = judged_review(tmp_path, monkeypatch, [LEAKING], judge=FakeJudge(), leak_judge={})
    assert plan["turn_resamples"] == review_module.LEAK_RESAMPLES == 2
    assert [a["status"] for a in plan["review_attempts"]] == ["turn_rejected", "turn_rejected", "validated"]


class BrokenJudge(FakeJudge):
    def judge(self, turn, **kwargs):
        raise TypeError("float() argument must be a string or a real number, not 'NoneType'")


def test_a_judge_reply_of_the_wrong_shape_is_unchecked_not_fatal(tmp_path, monkeypatch):
    result, _, _ = judged_review(tmp_path, monkeypatch, [LEAKING], judge=BrokenJudge(), leak_judge={})
    (candidate,) = result["points"][0]["candidates"]
    assert candidate["leak_judge"]["leak"] is None and "TypeError" in candidate["leak_judge"]["jev"]["error"]


def test_a_missing_typesafe_key_fails_the_review_before_any_model_call(tmp_path, monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    spec = review_spec(tmp_path, leak_judge={})
    with pytest.raises(ValueError, match=r"\$TYPESAFE_API_KEY"):
        run_review(spec, tmp_path / "review", transport_factory=lambda *a, **k: pytest.fail("no model call"))

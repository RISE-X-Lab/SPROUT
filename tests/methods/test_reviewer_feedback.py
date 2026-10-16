import json

import pytest

from sprout.harness_agents.mini_swe.history import validate_assistant_turn
from sprout.methods.branch_plan import ReviewerPlanError, ReviewResponseTruncated, review_with_feedback
from tests.methods.test_branch_selection import direction, exercise


def test_runner_corrects_invalid_selection_before_execution(tmp_path, monkeypatch):
    _, trials, _, prompts = exercise(
        tmp_path, monkeypatch,
        [{"branches": [direction(99)]}, {"branches": [direction(2)]},
         {"branches": []}], reviewer_max_attempts=3)
    assert len(trials) == 1
    assert trials[0]["context"]["snapshot_id"] == "parent-snap-2"
    assert "no eligible checkpoint" in prompts[1]
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert [a["status"] for a in saved["review_attempts"]] == ["validation_error", "validated"]
    assert "validation_error" not in saved


def test_nested_json_error_returned_verbatim_and_response_preserved():
    def turn(arguments):
        return {"role": "assistant", "content": "Inspect the match.",
                "tool_calls": [{"id": "review-1", "type": "function",
                                "function": {"name": "bash", "arguments": arguments}}]}
    bad = json.dumps(turn('{"command":"grep a\\|b file"}'))
    good = json.dumps(turn(json.dumps({"command": r"grep a\|b file"})))
    replies, prompts, saved = iter([bad, good]), [], {}

    def request(prompt):
        prompts.append(prompt)
        return next(replies)

    def validate(plan):
        try:
            return validate_assistant_turn(plan)
        except ValueError as error:
            raise ReviewerPlanError(str(error)) from error

    result = review_with_feedback(request, "original instructions", json.loads, validate, record=saved)
    assert result[1] == json.loads(good)
    error = saved["review_attempts"][0]["validation_error"]
    assert "Invalid \\escape" in error and "tool_calls[0]" in error
    assert error in prompts[1] and bad in prompts[1]
    assert saved["review_attempts"][0]["response"] == bad


@pytest.mark.parametrize("limit", [1, 3])
@pytest.mark.parametrize("failure", ["bad_json", "truncated"])
def test_exhaustion_keeps_every_reply(failure, limit):
    saved, persisted = {}, []

    def request(_):
        if failure == "truncated":
            raise ReviewResponseTruncated('{"branches":[')
        return "{bad json"

    result = review_with_feedback(
        request, "prompt", json.loads, lambda p: p, max_attempts=limit, record=saved,
        persist=lambda data: persisted.append(json.loads(json.dumps(data))))
    assert result is None
    assert len(saved["review_attempts"]) == limit, "a truncated reply spends the same correction limit"
    assert all(a["status"] == "validation_error" for a in saved["review_attempts"])
    assert persisted[-1] == saved
    if failure == "truncated":
        assert all(a["finish_reason"] == "length" for a in saved["review_attempts"])


def test_outer_json_corrected_and_empty_plan_accepted():
    replies = iter(["not JSON", '{"branches": []}'])
    saved = {}
    result = review_with_feedback(lambda _: next(replies), "prompt", json.loads,
                                  lambda p: p["branches"], record=saved)
    assert result == ({"branches": []}, [])
    assert len(saved["review_attempts"]) == 2


@pytest.mark.parametrize("partial", ['{"branches":[' + "\t \r\n" * 20_000, '{"branches": []}'])
def test_token_truncation_is_retried_without_echoing_or_accepting_partial_output(partial):
    saved, prompts, parsed = {}, [], []
    good = '{"branches": [{"command": "printf \'a  b\\\\n\'"}]}'

    def request(prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise ReviewResponseTruncated(partial)
        return good

    def parse(text):
        parsed.append(text)
        return json.loads(text)

    result = review_with_feedback(request, "original instructions", parse, lambda p: p, record=saved)
    assert result == (json.loads(good), json.loads(good))
    assert parsed == [good], "even syntactically valid JSON with finish_reason=length is incomplete"
    assert partial not in prompts[1]
    assert "original instructions" in prompts[1] and "finish_reason=length" in prompts[1]
    assert len(prompts[1]) < 1500
    first = saved["review_attempts"][0]
    assert first["response"] == partial and first["finish_reason"] == "length"
    assert [a["status"] for a in saved["review_attempts"]] == ["validation_error", "validated"]


@pytest.mark.parametrize("where", ["request", "state"])
def test_non_model_failure_not_retried(where):
    saved = {}

    def fail(_):
        raise ValueError("unavailable native history" if where == "state" else "authentication")

    result = review_with_feedback(
        fail if where == "request" else lambda _: "{}",
        "prompt", json.loads, fail, record=saved)
    assert result is None
    assert len(saved["review_attempts"]) == 1
    assert saved["review_attempts"][0]["status"] == where + "_error"


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_limit_fails_before_request(limit):
    with pytest.raises(ValueError, match="positive integer"):
        review_with_feedback(lambda _: pytest.fail("unexpected request"),
                             "", json.loads, lambda p: p, max_attempts=limit)

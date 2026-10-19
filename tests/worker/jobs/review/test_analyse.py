"""The analyst's report on one failed attempt: its prompt, its budget, and the loop that checks the reply."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.jobs.review.branch_plan import PromptRejected, ReviewResponseTruncated
from sprout.worker.jobs.review.branching import Attempt, Grade, analyse, outcome_from_journal
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS
from sprout.worker.jobs.review.structured_review import ANALYST_SCHEMA
from sprout.worker.run.journal import JournalWriter
from tests.worker.jobs.review.plain_prompts import plain_prompts
from tests.worker.jobs.review.recorded import record_attempt

REPORT = {
    "failure_reason": "r",
    "lesson": "l",
    "salvage": "s",
    "branch_candidates": [{"step": 2, "why": "w"}],
}
VERDICT = (
    "official Harbor reward=0.0; resolved=False\n\nPatch (1 lines):\n+x\n\n\nVerifier output:\nFAILED t::a"
)
TRANSCRIPT = (
    "    (agent said: Inspect the files.)\n"
    '[1] shell({"command": "echo early > answer"})  -> ok\n'
    '[2] shell({"command": "cat answer"})  -> early\n'
    "    (agent said: Inspect the files.)\n"
    '[3] shell({"command": "echo late > answer"})  -> ok\n'
    "    (agent said: Inspect the files.)\n"
    '[4] shell({"command": "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"})  -> ok'
)


class Analyst:
    """A plain transport: the next of ``replies`` answers each request. An error among them is raised."""

    def __init__(self, *replies: str | Exception) -> None:
        self.replies, self.prompts = list(replies), []

    def __call__(self, model: str, prompt: str) -> str:
        self.prompts.append(prompt)
        reply = self.replies.pop(0)
        if isinstance(reply, Exception):
            raise reply
        return reply


class StructuredAnalyst(Analyst):
    """A transport to an endpoint that takes JSON schemas. It records each schema and its name."""

    supports_json_schema = True

    def __init__(self, *replies: str | Exception, overhead: int) -> None:
        super().__init__(*replies)
        self.overhead, self.schemas = overhead, []

    def json(self, model: str, prompt: str, *, schema: dict[str, Any], name: str) -> str:
        self.schemas.append((schema, name))
        return super().__call__(model, prompt)

    def json_overhead(self, schema: dict[str, Any], name: str) -> int:
        return self.overhead

    def __call__(self, model: str, prompt: str) -> str:
        raise AssertionError("a structured analyst asks through json")


def failed(journal: Path, grade: Grade | None = None) -> Attempt:
    return Attempt(
        "parent",
        outcome_from_journal(journal),
        grade or Grade(reward=0.0, patch="+x\n", detail="FAILED t::a"),
    )


def long_journal(path: Path, *, steps: int, output_chars: int = 4000) -> Path:
    """A journal of ``steps`` tool steps, each with an output of ``output_chars``."""
    with JournalWriter(path) as journal:
        for step in range(1, steps + 1):
            journal.emit(
                "tool.started", call_id="c%d" % step, name="shell", args={"command": "step-%d" % step}
            )
            journal.emit(
                "tool.finished",
                call_id="c%d" % step,
                status="ok",
                output=("o%d " % step) * (output_chars // 5),
            )
    return path


def test_the_analyst_reads_the_problem_the_verdict_the_transcript_and_the_steps_it_may_choose(tmp_path):
    analyst = Analyst(json.dumps(REPORT))
    attempt = failed(record_attempt(tmp_path / "parent"))

    analyse(
        analyst,
        "fixture",
        "Fix it.",
        attempt,
        {4: None, 2: None, 9: None},
        3,
        100_000,
        prompts=plain_prompts(tmp_path / "p.yaml"),
    )

    assert analyst.prompts == [
        f"PROBLEM Fix it.\nVERDICT {VERDICT}\nTRANSCRIPT {TRANSCRIPT}\nSTEPS [2, 4] OF 1..4 AT MOST 3"
    ]


def test_a_valid_report_comes_back_with_the_last_step_and_how_its_prompt_was_fitted(tmp_path):
    analyst = Analyst(json.dumps(REPORT))
    attempt = failed(record_attempt(tmp_path / "parent"))

    report = analyse(
        analyst,
        "fixture",
        "Fix it.",
        attempt,
        {2: None},
        3,
        100_000,
        prompts=plain_prompts(tmp_path / "p.yaml"),
    )

    assert report == {
        **REPORT,
        "steps": 4,
        "prompt_budget": {
            "budget": 400_000,
            "chars": len(analyst.prompts[0]),
            "transcript_chars": len(TRANSCRIPT),
            "verdict_chars": len(VERDICT),
            "within_budget": True,
            "halvings": 0,
        },
    }


@pytest.mark.parametrize(
    "reply, error",
    [
        (json.dumps({"failure_reason": "r", "branch_candidates": []}), "'lesson' is a required property"),
        ("no json here", "analyst returned no JSON: 'no json here'"),
        (
            json.dumps({**REPORT, "branch_candidates": [{"step": "two", "why": "w"}]}),
            "'two' is not of type 'integer'",
        ),
        (
            json.dumps({**REPORT, "extra": 1}),
            "Additional properties are not allowed ('extra' was unexpected)",
        ),
    ],
    ids=["a-missing-field", "no-json", "a-step-not-an-integer", "an-extra-field"],
)
def test_an_invalid_report_goes_back_with_the_validators_error_and_is_never_repaired(tmp_path, reply, error):
    analyst = Analyst(reply, json.dumps(REPORT))

    report = analyse(
        analyst,
        "fixture",
        "Fix it.",
        failed(record_attempt(tmp_path / "parent")),
        {2: None},
        3,
        100_000,
        prompts=plain_prompts(tmp_path / "p.yaml"),
    )

    assert analyst.prompts[1] == analyst.prompts[0] + "\nREJECTED " + error
    assert {key: report[key] for key in REPORT} == REPORT


def test_an_analyst_without_a_valid_report_in_its_attempts_fails_the_analysis(tmp_path):
    analyst = Analyst("no json here", "still none")

    with pytest.raises(
        ValueError, match="analyst report invalid after 2 attempts: analyst returned no JSON: 'still none'"
    ):
        analyse(
            analyst,
            "fixture",
            "Fix it.",
            failed(record_attempt(tmp_path / "parent")),
            {2: None},
            3,
            100_000,
            max_attempts=2,
        )

    assert len(analyst.prompts) == 2


def test_a_report_cut_at_the_token_limit_spends_an_attempt_and_is_not_echoed(tmp_path):
    partial = '{"failure_reason":' + "\t \r\n" * 20_000
    analyst = Analyst(ReviewResponseTruncated(partial), json.dumps(REPORT))

    analyse(
        analyst,
        "fixture",
        "Fix it.",
        failed(record_attempt(tmp_path / "parent")),
        {2: None},
        3,
        100_000,
        prompts=plain_prompts(tmp_path / "p.yaml"),
    )

    assert analyst.prompts[1] == analyst.prompts[0] + "\nREJECTED " + str(ReviewResponseTruncated(partial))


def test_an_attempt_without_tool_steps_gets_a_fixed_report_without_a_request(tmp_path):
    analyst = Analyst()

    report = analyse(
        analyst,
        "fixture",
        "Fix it.",
        failed(long_journal(tmp_path / "trajectory.jsonl", steps=0)),
        {},
        3,
        100_000,
    )

    assert (report, analyst.prompts) == (
        {
            "failure_reason": "no tool steps recorded",
            "steps": 0,
            "lesson": "",
            "salvage": "nothing",
            "branch_candidates": [],
        },
        [],
    )


def test_a_long_attempt_loses_the_middle_of_its_transcript_and_leaves_room_for_one_feedback(tmp_path):
    attempt = failed(
        long_journal(tmp_path / "trajectory.jsonl", steps=300),
        Grade(reward=0.0, patch="+x\n" * 2000, detail="FAILED"),
    )
    analyst = Analyst("no json here", json.dumps(REPORT))

    report = analyse(analyst, "fixture", "Fix it.", attempt, {}, 3, 100_000, prompt_chars=120_000)

    first, retry = analyst.prompts
    assert "middle elided" in first and '[300] shell({"command": "step-300"})' in first
    assert "+x\n" * 2000 in first, "the verdict is whole while the transcript has room to give"
    assert len(first) < len(retry) <= 120_000 and report["prompt_budget"]["within_budget"]


def test_a_verdict_that_leaves_the_transcript_under_a_quarter_of_the_budget_is_clipped_first(tmp_path):
    """The transcript is the evidence. The verdict keeps its tail, where the verifier's summary is."""
    grade = Grade(reward=0.0, patch="+line\n" * 7000, detail="checks " * 1500 + "FAILED test_x")
    attempt = failed(long_journal(tmp_path / "trajectory.jsonl", steps=40), grade)
    analyst = Analyst(json.dumps(REPORT))

    report = analyse(analyst, "fixture", "Fix it.", attempt, {}, 3, 100_000, prompt_chars=60_000)

    sizing, prompt = report["prompt_budget"], analyst.prompts[0]
    assert sizing["transcript_chars"] >= 60_000 // 4
    assert sizing["verdict_chars"] < len(attempt.verdict_text())
    assert len(prompt) <= 60_000
    assert prompt.count("FAILED test_x") == 1 and "chars elided]… " in prompt


def test_a_refused_request_far_below_its_budget_is_sent_again_below_its_own_size(tmp_path):
    """Half the budget can still hold the prompt that the endpoint refused."""
    analyst = Analyst(PromptRejected("HTTP 400"), json.dumps(REPORT))
    attempt = failed(long_journal(tmp_path / "trajectory.jsonl", steps=40))

    report = analyse(analyst, "fixture", "Fix it.", attempt, {}, 3, 100_000, prompt_chars=400_000)

    first, second = analyst.prompts
    assert (len(first) < 200_000, report["prompt_budget"]["budget"], len(second) < len(first)) == (
        True,
        len(first) - 1,
        True,
    )


def test_a_refused_request_is_sent_again_with_half_the_budget_and_below_the_refused_size(tmp_path):
    analyst = Analyst(PromptRejected("HTTP 400"), json.dumps(REPORT))

    report = analyse(
        analyst,
        "fixture",
        "Fix it.",
        failed(long_journal(tmp_path / "trajectory.jsonl", steps=300)),
        {},
        3,
        100_000,
        prompt_chars=200_000,
    )

    first, second = analyst.prompts
    assert len(second) <= 100_000 < len(first) <= 200_000
    sizing = report["prompt_budget"]
    assert (sizing["budget"], sizing["chars"], sizing["halvings"]) == (100_000, len(second), 1)


def test_a_request_refused_after_two_halvings_fails_the_analysis(tmp_path):
    analyst = Analyst(PromptRejected("HTTP 400"), PromptRejected("HTTP 400"), PromptRejected("HTTP 400"))

    with pytest.raises(PromptRejected):
        analyse(
            analyst,
            "fixture",
            "Fix it.",
            failed(long_journal(tmp_path / "trajectory.jsonl", steps=300)),
            {},
            3,
            100_000,
            prompt_chars=200_000,
        )

    first, second, third = (len(prompt) for prompt in analyst.prompts)
    assert (first <= 200_000, second <= 100_000, third <= 50_000) == (True, True, True)


def test_a_structured_analyst_asks_for_the_report_schema_and_reserves_its_size_from_the_budget(tmp_path):
    analyst = StructuredAnalyst(json.dumps(REPORT), overhead=1000)

    report = analyse(
        analyst,
        "fixture",
        "Fix it.",
        failed(record_attempt(tmp_path / "parent")),
        {2: None},
        3,
        100_000,
        prompt_chars=200_000,
    )

    assert analyst.schemas == [(ANALYST_SCHEMA, "sprout_analysis")]
    sizing = report["prompt_budget"]
    assert (sizing["budget"], sizing["request_budget"], sizing["structured_output_chars"]) == (
        199_000,
        200_000,
        1000,
    )


def test_the_packaged_analyst_prompt_asks_for_exactly_the_fields_that_the_report_schema_requires():
    """A field that the schema requires and the prompt never asks for fails every analysis."""
    prompt = DEFAULT_PROMPTS.analyst_case.format(
        problem="p", verdict="v", transcript="t", checkpoint_steps="[2]", lo=1, hi=2, candidate_limit=2
    )

    example = json.loads(prompt.rsplit("Return ONLY a JSON object, no prose:", 1)[1].replace("<int>", "1"))

    candidate_schema = ANALYST_SCHEMA["properties"]["branch_candidates"]["items"]
    assert (set(example), set(example["branch_candidates"][0])) == (
        set(ANALYST_SCHEMA["required"]),
        set(candidate_schema["required"]),
    )

"""What the analyst reads of a finished attempt: its transcript, its verdict and its outcome."""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout.contract.messages import mark_hint
from sprout.worker.jobs.review.branching import (
    ELISION,
    Attempt,
    Grade,
    branch_count_rule,
    branch_run_name,
    extract_json,
    outcome_from_journal,
    render_transcript,
    require_mini_parent,
)
from sprout.worker.run.journal import JournalWriter
from tests.worker.jobs.review.recorded import record_attempt, turn


def journal_of(path: Path, *records: dict[str, object]) -> Path:
    """A journal at ``path`` with ``records``, each a ``type`` and its payload."""
    with JournalWriter(path) as journal:
        for record in records:
            journal.emit(**record)
    return path


def tool_step(call_id: str, command: str, output: str) -> tuple[dict[str, object], dict[str, object]]:
    return (
        {"event_type": "tool.started", "call_id": call_id, "name": "shell", "args": {"command": command}},
        {
            "event_type": "tool.finished",
            "call_id": call_id,
            "name": "shell",
            "status": "ok",
            "output": output,
        },
    )


def test_the_transcript_has_one_line_per_tool_step_numbered_as_the_checkpoints_number_them(tmp_path):
    journal = record_attempt(
        tmp_path, [turn("ls", "cat answer", outputs=("", "early\nlate")), turn("pytest -q")]
    )

    assert render_transcript(journal) == (
        "    (agent said: Inspect the files.)\n"
        '[1] shell({"command": "ls"})  -> ok\n'
        '[2] shell({"command": "cat answer"})  -> early late\n'
        "    (agent said: Inspect the files.)\n"
        '[3] shell({"command": "pytest -q"})  -> ok',
        1,
        3,
    )


def test_the_transcript_keeps_the_agents_words_but_not_the_hint_injected_into_its_prompt(tmp_path):
    journal = journal_of(
        tmp_path / "trajectory.jsonl",
        {"event_type": "fork.origin", "actor_hint": "INJECTED NOTE"},
        {"event_type": "run.started", "slot": "mini-swe-agent", "task_prompt": mark_hint("INJECTED NOTE")},
        {"event_type": "agent.message", "text": "I read the reminder about the parser."},
        *tool_step("c1", "ls", "files"),
    )

    assert render_transcript(journal) == (
        "    (agent said: I read the reminder about the parser.)\n"
        + '[1] shell({"command": "ls"})  -> files',
        1,
        1,
    )


@pytest.mark.parametrize(
    "output, shown",
    [
        ("H" * 4000 + "T" * 2000, "H" * 4000 + "T" * 2000),
        ("H" * 4000 + "x" + "T" * 2000, "H" * 4000 + " …[1 chars elided]… " + "T" * 2000),
    ],
    ids=["at-the-6000-cap", "one-over-the-cap"],
)
def test_a_long_tool_result_keeps_its_head_and_its_tail_where_a_test_run_prints_its_verdict(
    tmp_path, output, shown
):
    journal = journal_of(tmp_path / "trajectory.jsonl", *tool_step("c1", "pytest", output))

    body, _, _ = render_transcript(journal)

    assert body == '[1] shell({"command": "pytest"})  -> ' + shown


def test_a_long_agent_message_and_long_arguments_are_clipped_to_their_caps(tmp_path):
    command = "a" * 3986  # with '{"command": ""}', the arguments are 4001 characters
    journal = journal_of(
        tmp_path / "trajectory.jsonl",
        {"event_type": "agent.message", "text": "M" * 1333 + "x" + "N" * 667},
        *tool_step("c1", command, "ok"),
    )

    body, _, _ = render_transcript(journal)

    assert body == (
        "    (agent said: " + "M" * 1333 + " …[1 chars elided]… " + "N" * 667 + ")\n"
        '[1] shell({"command": "' + "a" * 2653 + " …[1 chars elided]… " + "a" * 1332 + '"})  -> ok'
    )


@pytest.mark.parametrize(
    "token_budget, char_budget, length",
    [(1000, None, 3200 + len(ELISION)), (10**6, 5000, 5000)],
    ids=["tokens-at-3.2-characters", "an-exact-character-budget"],
)
def test_a_transcript_over_budget_loses_its_middle_and_keeps_a_third_before_it(
    tmp_path, token_budget, char_budget, length
):
    """The late steps hold the failure under diagnosis, so the tail gets two thirds."""
    steps = [record for n in range(1, 401) for record in tool_step("c%d" % n, "step-%d" % n, "o" * 100)]
    journal = journal_of(tmp_path / "trajectory.jsonl", *steps)

    body, first, last = render_transcript(journal, token_budget=token_budget, char_budget=char_budget)

    assert (len(body), body.index(ELISION), body.count(ELISION)) == (length, (length - len(ELISION)) // 3, 1)
    assert (
        body[:4],
        body.endswith('[400] shell({"command": "step-400"})  -> ' + "o" * 100),
        first,
        last,
    ) == (
        "[1] ",
        True,
        1,
        400,
    )


@pytest.mark.parametrize(
    "grade, summary",
    [
        (Grade(resolved=False, reward=0.0), "official Harbor reward=0.0; resolved=False"),
        (Grade(resolved=False, reward=0.0, error="verifier crashed"), "GRADING ERROR: verifier crashed"),
    ],
)
def test_a_grade_summary_shows_a_grading_error_instead_of_the_reward(grade, summary):
    assert grade.summary() == summary


def test_the_verdict_shows_the_summary_the_patch_and_the_verifier_output_each_within_its_cap(tmp_path):
    grade = Grade(
        resolved=False, reward=0.0, patch="+x\n" + "p" * 40_000, detail="FAILED t::a" + "d" * 20_000
    )
    attempt = Attempt("parent", outcome_from_journal(record_attempt(tmp_path)), grade)

    assert attempt.verdict_text() == (
        "official Harbor reward=0.0; resolved=False\n\nPatch (1 lines):\n"
        + "+x\n"
        + "p" * 39_997
        + "\n\nVerifier output:\nFAILED t::a"
        + "d" * 19_989
    )


def test_the_outcome_of_a_finished_attempt_is_read_back_from_its_journal(tmp_path):
    journal = record_attempt(tmp_path, status="error", failed_steps=(3,), final_text="Gave up.")

    outcome = outcome_from_journal(journal, run_id="root-1")

    assert (outcome.run_id, outcome.journal_path, outcome.status, outcome.final_text) == (
        "root-1",
        journal,
        "error",
        "Gave up.",
    )
    assert (outcome.usage, outcome.checkpoints, outcome.error) == ({"prompt_tokens": 11}, 2, None)


def test_an_attempt_that_never_finished_has_an_unknown_status(tmp_path):
    journal = journal_of(
        tmp_path / "trajectory.jsonl", {"event_type": "run.started", "slot": "mini-swe-agent"}
    )

    assert outcome_from_journal(journal).status == "unknown"


@pytest.mark.parametrize(
    "text, value",
    [
        ('Here:\n```json\n{"a": 1}\n```\nand {"b": 2}', {"a": 1}),
        ('Sure. {"a": {"b": 2}} Done.', {"a": {"b": 2}}),
        ('{"hint": "first line\nsecond line"}', {"hint": "first line\nsecond line"}),
    ],
    ids=["the-first-fenced-block", "from-the-first-brace-to-the-last", "a-literal-newline-in-a-string"],
)
def test_the_json_of_a_reply_is_its_first_fenced_object_else_its_outermost_braces(text, value):
    assert extract_json(text) == value


@pytest.mark.parametrize("text", ["no json here", "} reversed {"])
def test_a_reply_without_an_object_is_refused(text):
    with pytest.raises(ValueError, match="analyst returned no JSON"):
        extract_json(text)


@pytest.mark.parametrize(
    "round_no, index, name, run_name",
    [
        (1, 1, "Fix Parser!", "r1b1-fix-parser-"),
        (2, 3, None, "r2b3-b"),
        (1, 2, "a-very-long-name-for-a-repair-direction", "r1b2-a-very-long-name-for-a-r"),
    ],
    ids=["slugged", "unnamed", "cut-at-24"],
)
def test_a_branch_run_is_named_by_its_round_its_place_and_a_slug_of_its_direction(
    round_no, index, name, run_name
):
    assert branch_run_name(round_no, index, name) == run_name


def test_an_unknown_branch_count_mode_is_refused():
    with pytest.raises(ValueError, match="unknown branch count mode: 'exact'"):
        branch_count_rule("exact", 2)


@pytest.mark.parametrize(
    "slots",
    [["claude-code"], ["mini-swe-agent", "claude-code"], []],
    ids=["another-agent", "a-mixed-journal", "no-run"],
)
def test_only_a_journal_whose_every_run_is_mini_is_a_parent_to_branch(tmp_path, slots):
    journal = journal_of(
        tmp_path / "trajectory.jsonl", *({"event_type": "run.started", "slot": slot} for slot in slots)
    )

    with pytest.raises(ValueError, match="only mini-swe-agent parent trajectories"):
        require_mini_parent(journal)


def test_a_mini_journal_is_a_parent_to_branch(tmp_path):
    require_mini_parent(record_attempt(tmp_path))

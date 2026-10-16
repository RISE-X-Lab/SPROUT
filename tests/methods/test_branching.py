"""What the analyst actually gets to see."""
import json
from string import Formatter

import pytest

from sprout.worker.jobs.review import branching
from sprout.worker.jobs.review.branching import _clip, render_transcript
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS


def _journal(tmp_path, records):
    path = tmp_path / "j.jsonl"
    with open(path, "w") as fh:
        for i, r in enumerate(records, 1):
            fh.write(json.dumps(dict(r, seq=i, run_id="r", agent_id="a")) + "\n")
    return path


def test_a_long_tool_result_keeps_its_tail():
    """THE defect this budget exists to fix. Tool results on a real run have a
    median of ~1.9k characters and a max of ~17k; the old 300-character head-only
    cap fed the analyst a test run's banner and threw away the assertion that
    explains the failure -- which is the one thing it needs to diagnose."""
    body = "banner " * 2000 + "ASSERTION FAILED: the answer"
    clipped = _clip(body, 300)
    assert len(clipped) < len(body)
    assert "ASSERTION FAILED: the answer" in clipped, "the verdict is at the END"
    assert clipped.startswith("banner"), "and the head still identifies it"
    assert _clip("small", 300) == "small", "below the limit nothing is touched"


def test_step_numbers_match_the_snapshot_map(tmp_path):
    """The analyst names a step and the branch-point index looks it up: both count one per
    exec call, in order. Two counting schemes would put every branch at the
    wrong snapshot."""
    path = _journal(tmp_path, [
        {"type": "run.started"},
        {"type": "tool.started", "name": "shell", "args": {"command": "a"}},
        {"type": "tool.finished", "name": "shell", "output": "out-a"},
        {"type": "checkpoint.captured", "step": 1, "snapshot_id": "s1"},
        {"type": "tool.started", "name": "shell", "args": {"command": "b"}},
        {"type": "tool.finished", "name": "shell", "output": "OK"},
        {"type": "checkpoint.captured", "step": 2, "snapshot_id": "s2"},
    ])
    body, lo, hi = render_transcript(path)
    assert (lo, hi) == (1, 2)
    assert body.startswith("[1] shell(")
    assert '[2] shell({"command": "b"})' in body
    assert "out-a" in body and "OK" in body


def test_the_token_budget_elides_the_middle_not_the_ends(tmp_path):
    """When even the generous budget overflows, the early steps (what was
    understood) and the late ones (the failure) are what must survive."""
    records = []
    for i in range(400):
        records.append({"type": "tool.started", "name": "shell",
                        "args": {"command": "step-%d" % i}})
        records.append({"type": "tool.finished", "name": "shell",
                        "output": "x" * 4000})
    path = _journal(tmp_path, records)
    body, _, hi = render_transcript(path, token_budget=2000)
    assert hi == 400, "the step COUNT is still the truth"
    assert "middle elided" in body
    assert "step-0" in body and "step-399" in body


# --- the verdict the analyst reads ------------------------------------------------
def test_verdict_text_carries_the_patch_and_the_verifiers_words():
    """The verdict is what the analyst sees; the diff and the verifier's output are
    the two facts it needs, and an empty grade must still render."""
    from sprout.worker.jobs.review.branching import Attempt, Grade

    grade = Grade(resolved=False, reward=0.0, patch="+x\n", detail="FAILED tests/test_a.py::test_a")
    text = Attempt("r1b1", outcome=None, grade=grade, hint="try X").verdict_text()
    assert "reward=0.0" in text and "+x" in text and "test_a" in text
    assert Attempt("p", None, Grade()).verdict_text()
    assert "GRADING ERROR: boom" in Grade(error="boom").summary()


@pytest.mark.parametrize("template,input_fields,output_fields", [
    (DEFAULT_PROMPTS.analyst_case,
     {"problem", "verdict", "transcript", "lo", "hi", "checkpoint_steps", "candidate_limit"},
     {"failure_reason", "lesson", "salvage", "branch_candidates"}),
    (DEFAULT_PROMPTS.reviewer_user_hint,
     {"problem", "reports", "count_rule"},
     {"synthesis", "branches"}),
])
def test_analysis_prompts_preserve_fields_and_separate_private_evidence(
        template, input_fields, output_fields):
    fields = {entry[1] for entry in Formatter().parse(template) if entry[1]}
    assert fields == input_fields
    values = {field: "INPUT_%s" % field.upper() for field in input_fields}
    rendered = template.format(**values)
    assert all(value in rendered for value in values.values())
    schema = json.loads(rendered.rsplit("Return ONLY a JSON object, no prose:", 1)[1]
                        .replace("<int>", "1"))
    assert set(schema) == output_fields
    compact = " ".join(rendered.split())
    assert "private diagnostic evidence" in compact
    assert "natural and effective continuation" in compact


def test_analyst_translates_private_evidence_to_accessible_repair_directions():
    """Instruction coverage, not a claim that a model always follows it."""
    prompt = " ".join(DEFAULT_PROMPTS.analyst_case.split())
    assert "## Translate evidence into repair directions" in prompt
    assert "A path in verifier output is not evidence that the actor can open it" in prompt
    assert "In every returned field" in prompt
    assert "omit verifier-only filenames, test names/IDs and grader paths" in prompt
    assert "task or retained prefix establishes it is available" in prompt
    assert "Do not ask the actor to read, locate or recreate a hidden test" in prompt
    assert "compliance_test.go" not in DEFAULT_PROMPTS.analyst_case


def test_analyst_uncertainty_still_produces_a_testable_direction():
    prompt = " ".join(DEFAULT_PROMPTS.analyst_case.split())
    assert "strongest supported hypothesis" in prompt
    assert "what would support or rule it out" in prompt
    assert "Do not merely add 'possibly' to an invented explanation" in prompt
    assert "do not guess how an unseen test constructs its fixtures" in prompt
    assert "If the cause is established, state the correction directly" in prompt
    assert "caller-owned or handler-owned prefix" in prompt


def test_analysis_prompts_anchor_guidance_to_the_retained_prefix():
    analyst = " ".join(DEFAULT_PROMPTS.analyst_case.split())
    reviewer = " ".join(DEFAULT_PROMPTS.reviewer_user_hint.split())
    assert "Separate observed facts from inferences" in analyst
    assert "failure_reason and lesson are controller-only reports" in analyst
    assert "state after each candidate step" in analyst
    assert "LATER IS BETTER" in analyst
    assert "For EACH branch, choose its own base attempt and branch_step" in reviewer
    assert "Positions do NOT need to be distinct" in reviewer
    assert "state AFTER branch_step" in reviewer
    assert "already present or already verified" in reviewer
    assert "Prefer 2-5 concise sentences" in reviewer
    assert "through its own tools" in reviewer
    assert "not just the wording" in reviewer


def test_reviewer_prompt_includes_continuity_usefulness_and_no_source_narration():
    reviewer = " ".join(DEFAULT_PROMPTS.reviewer_user_hint.split())
    assert "delivered VERBATIM" in reviewer
    assert "mentally remove the hint" in reviewer
    assert "Do not make a precise diagnosis vague" in reviewer
    assert "Do not fabricate the agent's reasoning" in reviewer
    assert "do not include test names or IDs" in reviewer
    assert "pass/fail counts, scores, raw verifier output" in reviewer
    assert "Do not label the text as a reminder" in reviewer
    assert "useful diagnosis and behavioral constraints are still intact" in reviewer


def test_analyst_transcript_omits_injected_note_without_rewriting_agent_speech(tmp_path):
    """The agent's own words are kept -- its stated intent is how you tell a wrong
    turn from a wrong result -- but the note we injected is not."""
    path = tmp_path / "branch.jsonl"
    records = [
        {"type": "fork.origin", "actor_hint": "INJECTED_NOTE"},
        {"type": "run.started", "task_prompt": "INJECTED_NOTE"},
        {"type": "agent.message", "text": "ACTUAL_RECORDED_SPEECH about a reminder"},
        {"type": "tool.started", "name": "shell", "args": {"command": "ls"}},
        {"type": "tool.finished", "output": "OBSERVED_FILES"},
    ]
    path.write_text("\n".join(json.dumps(record) for record in records) + "\n")
    text, lo, hi = branching.render_transcript(path)
    assert "INJECTED_NOTE" not in text
    assert "ACTUAL_RECORDED_SPEECH about a reminder" in text
    assert "OBSERVED_FILES" in text
    assert (lo, hi) == (1, 1)

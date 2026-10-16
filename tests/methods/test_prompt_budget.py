"""The analyst and reviewer requests are held to a character budget.

INCIDENT: in round 2 of one task the reviewer request -- a 136-step parent plus
four graded branches, each repeating the parent's prefix -- reached 3.7M
characters (about a million tokens); the endpoint answered 400 Bad Request and
the whole task was lost as an infrastructure error. Round-1 requests of 746k
characters had succeeded. Here the budget holds, the reduction order is the
documented one, the plan records what was dropped, and a refused request is
retried smaller instead of ending the round.
"""

import json
from pathlib import Path

import httpx
import pytest

from sprout.worker.jobs.review import branching
from sprout.worker.jobs.review.branch_plan import PromptRejected
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS
from sprout.worker.jobs.review.branching import (
    DEFAULT_PROMPT_CHARS, FLOOR, STAGES, Attempt, Grade,
    Window, analyse, branch_count_rule, fit_review_prompt, reduce_row, review, window_messages,
)

TOOLS = [{"type": "function", "function": {"name": "bash", "parameters": {
    "type": "object", "properties": {"command": {"type": "string"}}, "required": ["command"]}}}]


def history(turns, *, result_chars=3000, reasoning_chars=7000):
    """A native history: system and task prompts, then ``turns`` closed turns of
    one assistant message (with provider reasoning) and one tool result each."""
    messages = [{"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": "Please solve this issue: " + "task " * 300}]
    counts = {}
    for step in range(1, turns + 1):
        messages.append({"role": "assistant", "content": "Step %d: inspect." % step,
                         "reasoning_content": ("think-%d " % step) * (reasoning_chars // 8),
                         "tool_calls": [{"id": "c%d" % step, "type": "function", "function": {
                             "name": "bash", "arguments": json.dumps({"command": "cmd %d" % step})}}]})
        messages.append({"role": "tool", "tool_call_id": "c%d" % step,
                         "content": json.dumps({"returncode": 0, "output": ("out-%d " % step) * (result_chars // 6)})})
        counts[str(step)] = len(messages)
    return {"messages": messages, "prefix_message_counts": counts, "tools": TOOLS,
            "workspace": {"cwd": "/app", "repository_dir": "/app"}}


def row(name, round_no, turns, candidates, **changes):
    value = {"failure_reason": "The boundary case is unhandled.", "lesson": "Handle the empty input.",
             "salvage": "The parser.", "steps": turns,
             "branch_candidates": [{"step": step, "why": "useful state"} for step in candidates],
             "name": name, "round": round_no, "grade": "official Harbor reward=0.0; resolved=False",
             "available_steps": list(range(1, turns + 1)), "assistant_turn_given": None,
             "native_history": history(turns)}
    value.update(changes)
    return value


REPORT_FIELDS = ("failure_reason", "lesson", "salvage", "steps", "branch_candidates", "name", "round",
                 "grade", "available_steps", "assistant_turn_given")


def render(rows):
    return DEFAULT_PROMPTS.reviewer_assistant_turn.format(
        problem="Fix it.", reports=json.dumps(rows, indent=1, ensure_ascii=False),
        count_rule=branch_count_rule("adaptive", 3))


def fit(rows, budget):
    return fit_review_prompt(DEFAULT_PROMPTS.reviewer_assistant_turn, problem="Fix it.", rows=rows,
                             count_rule=branch_count_rule("adaptive", 3), budget=budget)


def size_at(rows, stage):
    return len(render([reduce_row(r, stage.parent if r["round"] == 0 else stage.branches, stage.clip)[0]
                       for r in rows]))


def positions(reduced, original):
    """The 1-based position in ``original`` of every shown message, marker ranges as tuples."""
    out = []
    for message in reduced:
        if message["role"] == "omitted":
            out.append(tuple(message["positions"]))
        else:
            out.append(original.index(message) + 1)
    return out


def test_a_request_within_budget_is_unchanged_byte_for_byte():
    rows = [row("parent", 0, 6, [4]), row("r1b1-x", 1, 8, [7])]
    prompt, record = fit(rows, 10 ** 7)
    assert prompt == render(rows)
    assert record == {"budget": 10 ** 7, "chars_before": len(prompt), "chars_after": len(prompt),
                      "within_budget": True, "stage": None, "attempts": {}}


def test_reduction_order_is_branches_then_parent_then_clipping_then_the_reports_alone():
    rows = [row("parent", 0, 60, [50, 30]), row("r1b1-a", 1, 70, [66]), row("r1b2-b", 1, 72, [70, 65])]
    sizes = [size_at(rows, stage) for stage in STAGES]
    full = len(render(rows))
    assert full > sizes[0] > sizes[1] > sizes[2] > sizes[3] > sizes[4] > sizes[5], sizes

    # Stage 1: the branches are windowed, the parent is whole.
    prompt, record = fit(rows, sizes[0] + 10)
    assert record["stage"] == STAGES[0].name and record["within_budget"] and len(prompt) <= sizes[0] + 10
    reduced = json.loads(prompt.split("## Every attempt so far\n", 1)[1].split("\n\n## Branch count", 1)[0])
    assert reduced[0]["native_history"] == rows[0]["native_history"]
    assert all(m["role"] == "omitted" for r in reduced[1:] for m in r["native_history"]["messages"][2:3])
    assert set(record["attempts"]) == {"r1b1-a", "r1b2-b"}

    # Stage 2: the parent too; the analyst reports are first class at every stage.
    prompt, record = fit(rows, sizes[1] + 10)
    assert record["stage"] == STAGES[1].name and record["within_budget"]
    reduced = json.loads(prompt.split("## Every attempt so far\n", 1)[1].split("\n\n## Branch count", 1)[0])
    parent = record["attempts"]["parent"]
    assert parent["messages"] == 122 and parent["omitted"] + parent["retained"] == 122 and parent["omitted"] > 60
    for before, after in zip(rows, reduced):
        assert {k: after[k] for k in REPORT_FIELDS} == {k: before[k] for k in REPORT_FIELDS}
        assert after["native_history"]["prefix_message_counts"] == before["native_history"]["prefix_message_counts"]
        assert after["native_history"]["tools"] == TOOLS and after["native_history"]["workspace"] == {"cwd": "/app", "repository_dir": "/app"}

    # Then clipping: windows unchanged, long fields kept head and tail.
    prompt, record = fit(rows, sizes[3] + 10)
    assert record["stage"] == STAGES[3].name and record["within_budget"]
    reduced = json.loads(prompt.split("## Every attempt so far\n", 1)[1].split("\n\n## Branch count", 1)[0])
    shown = [m for m in reduced[0]["native_history"]["messages"] if m["role"] != "omitted"]
    assert all(len(m["content"]) <= 2000 + 40 for m in shown if m["role"] == "tool")
    assert all("chars elided" in m["reasoning_content"] for m in shown if m["role"] == "assistant")
    assert record["attempts"]["parent"]["clipped"] > 0

    # The floor: the reports alone -- every native history is one marker that
    # stands for the whole of it, and says the history ends there.
    prompt, record = fit(rows, 1000)
    assert record["stage"] == FLOOR and record["within_budget"] is False
    reduced = json.loads(prompt.split("## Every attempt so far\n", 1)[1].split("\n\n## Branch count", 1)[0])
    for before, after in zip(rows, reduced):
        messages = after["native_history"]["messages"]
        total = len(before["native_history"]["messages"])
        assert [m["role"] for m in messages] == ["omitted"]
        assert messages[0]["omitted_messages"] == total and messages[0]["positions"] == [1, total]
        assert "through its end" in messages[0]["content"] and "next message" not in messages[0]["content"]
        assert messages == [branching.omission_marker(1, total, total)]
        assert {k: after[k] for k in REPORT_FIELDS} == {k: before[k] for k in REPORT_FIELDS}
        assert after["native_history"]["prefix_message_counts"] == before["native_history"]["prefix_message_counts"]
    assert all(stats == {"messages": 122 if name == "parent" else len(history(70 if name == "r1b1-a" else 72)["messages"]),
                         "omitted": stats["messages"], "retained": 0}
               for name, stats in record["attempts"].items())


@pytest.mark.parametrize("tail", [4, 0])
def test_the_marker_names_the_omitted_positions_and_the_prefix_counts_stay_exact(tail):
    """A marker in the middle names the position of the message after it; one
    that stands for the tail (no tail kept) must not claim a message follows it."""
    native = history(40)
    messages, cuts = native["messages"], native["prefix_message_counts"]
    reduced, omitted = window_messages(messages, [cuts["20"]], Window(before=6, after=2, tail=tail))
    where = positions(reduced, messages)
    # The system and task prompts, six messages up to the cut and two after it,
    # the last ``tail``; one marker per gap, each saying exactly which positions.
    cut = cuts["20"]
    assert where == ([1, 2, (3, cut - 6)] + list(range(cut - 5, cut + 3))
                     + [(cut + 3, 82 - tail)] + list(range(83 - tail, 83)))
    assert omitted == 82 - 2 - 8 - tail
    markers = [m for m in reduced if m["role"] == "omitted"]
    for marker in markers:
        first, last = marker["positions"]
        assert marker["omitted_messages"] == last - first + 1
        assert "prefix_message_counts count the full history and are still exact" in marker["content"]
        if last == len(messages):
            assert ("positions %d-%d of the full history, through its end" % (first, last)) in marker["content"]
            assert "next message" not in marker["content"]
        else:
            assert ("positions %d-%d of the full history" % (first, last)) in marker["content"]
            assert ("the next message is position %d." % (last + 1)) in marker["content"]
    assert omitted == sum(m["omitted_messages"] for m in markers) == len(messages) - (len(reduced) - len(markers))
    # What a branch at step 20 retains ends exactly at the cut, and that is shown.
    assert reduced[where.index(cut)] == messages[cut - 1] and messages[cut - 1]["role"] == "tool"
    assert native["prefix_message_counts"]["20"] == cut


def test_clipped_messages_keep_their_newlines():
    """The reviewer reads tool results, reasoning and arguments as the actor saw
    them; clipping takes the middle, not the line structure."""
    from sprout.worker.jobs.review.branching import clip_messages

    result = json.dumps({"returncode": 0, "output": "\n".join("line %d" % i for i in range(400))}, indent=2)
    messages = [{"role": "assistant", "content": "x", "reasoning_content": "first\n" * 500,
                 "tool_calls": [{"id": "c1", "type": "function", "function": {
                     "name": "bash", "arguments": json.dumps({"command": "cat <<'EOF' > f\n" + "l\n" * 600 + "EOF"})}}]},
                {"role": "tool", "tool_call_id": "c1", "content": result}]
    clipped, count = clip_messages(messages, 1000)
    assert count == 3
    tool = clipped[1]["content"]
    assert tool.startswith('{\n  "returncode": 0,\n  "output": "line 0\\nline 1') and tool.endswith('line 399"\n}')
    assert "chars elided" in tool and len(tool) <= 1000 + 40
    assert clipped[0]["reasoning_content"].startswith("first\nfirst\n") and "chars elided" in clipped[0]["reasoning_content"]
    arguments = clipped[0]["tool_calls"][0]["function"]["arguments"]
    assert arguments.startswith('{"command": "cat <<\'EOF\' > f\\nl\\nl') and arguments.endswith('EOF"}')
    assert messages[1]["content"] == result, "the input is not modified"


def test_overlapping_windows_merge_and_a_history_shorter_than_the_window_is_untouched():
    native = history(5)
    reduced, omitted = window_messages(native["messages"], [native["prefix_message_counts"]["3"]], Window(12, 4, 8))
    assert reduced == native["messages"] and omitted == 0
    native = history(30)
    reduced, _ = window_messages(native["messages"], [20, 22, 24], Window(4, 2, 2))
    assert len([m for m in reduced if m["role"] == "omitted"]) == 2, "adjacent windows share one gap"


def test_both_reviewer_prompts_explain_the_marker_and_keep_their_instructions():
    for template in (DEFAULT_PROMPTS.reviewer_assistant_turn, DEFAULT_PROMPTS.reviewer_point):
        text = " ".join(template.split())
        assert 'role "omitted" is a controller marker' in text
        assert "prefix_message_counts count the" in text.lower() or "prefix_message_counts count the full" in text
        # _clip keeps head and tail: the marker is in the middle, not at the end.
        assert 'Text containing "…[N chars elided]…" had its middle clipped by the controller' in text
        assert 'Text ending in "…[N chars elided]…"' not in text
    assistant = " ".join(DEFAULT_PROMPTS.reviewer_assistant_turn.split())
    assert "take ONLY the first native_history.prefix_message_counts[step] messages" in assistant
    assert "Return exactly one fenced code block labelled branch-plan" in assistant


# --- through review(): the plan record and the 400 retry ---------------------------

def _review_setup(tmp_path, monkeypatch, *, turns=(80, 90, 90)):
    from tests.methods.test_branch_selection import write_journal

    attempts = []
    for index, steps in enumerate(turns):
        name = "parent" if index == 0 else "r1b%d-lead" % index
        journal = write_journal(tmp_path / "trials" / name / "agent" / "trajectory.jsonl", steps=min(steps, 5))
        attempt = Attempt(name, branching.outcome_from_journal(journal, run_id=name), Grade(patch="x"),
                          round_no=0 if index == 0 else 1)
        attempts.append(attempt)
    histories = {a.name: history(steps) for a, steps in zip(attempts, turns)}
    # Eligibility uses write_journal's real native transcript; the histories
    # supplied to prompt budgeting below are deliberately much larger.
    monkeypatch.setattr(branching, "conversation_restore", lambda path, step, session=None: "cut-%d" % step)
    monkeypatch.setattr(branching, "reviewer_context", lambda journal, points: {
        **histories[Path(journal).parent.parent.name],
        "prefix_message_counts": {str(s): histories[Path(journal).parent.parent.name]["prefix_message_counts"][str(s)]
                                  for s in points}})
    monkeypatch.setattr(branching, "selected_prefix", lambda *a: [])
    cases = {a.name: {"failure_reason": "r", "lesson": "l", "salvage": "s", "steps": 5,
                      "branch_candidates": [{"step": 3, "why": "w"}],
                      "prompt_budget": {"budget": 1, "chars": 1}} for a in attempts}
    return attempts, cases


def plan_text(step=3):
    plan = {"synthesis": "Restart there.", "branches": [{"name": "restart", "base": "parent", "branch_step": step,
                                                         "why": "useful"}]}
    return "```branch-plan\n" + json.dumps(plan) + "\n```"


def test_structured_reviewer_uses_raw_json_and_reserves_schema_budget(tmp_path, monkeypatch):
    attempts, cases = _review_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(branching, "actor_tools_at", lambda *args: TOOLS)
    calls = []

    class Transport:
        supports_json_schema = True

        def json_overhead(self, schema, name):
            return 1000

        def json(self, model, prompt, *, schema, name):
            calls.append((prompt, schema, name))
            return json.dumps({"synthesis": "Inspect.", "branches": [{
                "name": "inspect", "base": "parent", "branch_step": 3, "why": "Check.",
                "content": "I will inspect the current code.", "commands": ['printf "quoted"'],
            }]})

        def __call__(self, *args):
            raise AssertionError("Structured reviewer used the unstructured path")

    choices = review(Transport(), "m", "Fix it.", attempts, cases, round_no=1, width=1,
                     plan_path=tmp_path / "plan.json", guidance_mode="assistant-turn",
                     count_mode="fixed", prompt_chars=200_000)
    assert len(choices) == 1 and len(calls) == 1
    assert calls[0][2] == "sprout_branches"
    assert calls[0][1]["properties"]["branches"]["minItems"] == 1
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["prompt_budget"]["budget"] == 199_000
    assert len(calls[0][0]) + plan["structured_output"]["overhead_chars"] <= 200_000
    assert plan["selected_branches"][0]["branch_step"] == 3
    assert plan["structured_review"] is True, "the plan says this review was schema-constrained"


def test_an_assistant_turn_review_through_a_plain_transport_stays_fenced_and_says_so(tmp_path, monkeypatch):
    """Without ``supports_json_schema`` (the default ReviewTransport, --structured-review
    absent) the reviewer answers in the labelled fence as before, no schema budget is
    reserved, and the plan records ``structured_review: false``."""
    attempts, cases = _review_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(branching, "actor_tools_at", lambda *args: TOOLS)
    calls = []

    class Transport:
        def __call__(self, model, prompt):
            calls.append(prompt)
            plan = {"synthesis": "Inspect.", "branches": [{
                "name": "inspect", "base": "parent", "branch_step": 3, "why": "Check.",
                "content": "I will inspect the current code.", "commands": ['printf "quoted"'],
            }]}
            return "```branch-plan\n" + json.dumps(plan) + "\n```"

        def json(self, *args, **kwargs):
            raise AssertionError("A plain transport must not be asked for json_schema output")

    choices = review(Transport(), "m", "Fix it.", attempts, cases, round_no=1, width=1,
                     plan_path=tmp_path / "plan.json", guidance_mode="assistant-turn",
                     count_mode="fixed", prompt_chars=200_000)
    assert len(choices) == 1 and len(calls) == 1
    assert "labelled branch-plan" in calls[0]
    plan = json.loads((tmp_path / "plan.json").read_text())
    assert plan["structured_review"] is False and "structured_output" not in plan
    assert plan["prompt_budget"]["budget"] == 200_000, "no schema overhead reserved"
    assert plan["selected_branches"][0]["branch_step"] == 3


def test_review_sends_the_given_prompts_and_records_which_file_they_came_from(tmp_path, monkeypatch):
    """The reviewer's template and count rule are the ones in the Prompts passed in --
    the packaged file by default, an edited copy under --prompts -- and the plan names
    the file and its sha256, so a run says which prompts produced it."""
    import yaml
    from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH, load_prompts

    attempts, cases = _review_setup(tmp_path, monkeypatch)
    monkeypatch.setattr(branching, "actor_tools_at", lambda *args: TOOLS)
    calls = []

    class Transport:
        supports_json_schema = True

        def json_overhead(self, schema, name):
            return 1000

        def json(self, model, prompt, *, schema, name):
            calls.append(prompt)
            return json.dumps({"synthesis": "Inspect.", "branches": [{
                "name": "inspect", "base": "parent", "branch_step": 3, "why": "Check.",
                "content": "I will inspect the current code.", "commands": ['printf "quoted"'],
            }]})

    heading = "## Probe first, then fix, then verify"
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["reviewer"]["assistant_turn"] = data["reviewer"]["assistant_turn"].replace(heading, "## EDITED SECTION")
    data["reviewer"]["count_rule"]["fixed"] = "EXACTLY {count}, EDITED."
    (tmp_path / "edited.yaml").write_text(yaml.safe_dump(data))
    edited = load_prompts(tmp_path / "edited.yaml")

    common = dict(round_no=1, width=1, guidance_mode="assistant-turn", count_mode="fixed", prompt_chars=200_000)
    review(Transport(), "m", "Fix it.", attempts, cases, plan_path=tmp_path / "default.json", **common)
    review(Transport(), "m", "Fix it.", attempts, cases, plan_path=tmp_path / "edited.json", prompts=edited, **common)
    default, custom = calls
    assert heading in default and "## EDITED SECTION" not in default
    assert "Return exactly 1 branches. This is a required count." in default
    assert "## EDITED SECTION" in custom and heading not in custom
    assert "EXACTLY 1, EDITED." in custom
    assert json.loads((tmp_path / "default.json").read_text())["prompts"] == {
        "path": str(DEFAULT_PROMPTS_PATH), "sha256": DEFAULT_PROMPTS.sha256}
    assert json.loads((tmp_path / "edited.json").read_text())["prompts"] == {
        "path": str(tmp_path / "edited.yaml"), "sha256": edited.sha256}
    assert edited.sha256 != DEFAULT_PROMPTS.sha256


def test_review_records_the_budget_in_the_plan_and_hides_the_analysts_sizing(tmp_path, monkeypatch):
    attempts, cases = _review_setup(tmp_path, monkeypatch)
    prompts = []

    def transport(model, prompt):
        prompts.append(prompt)
        return plan_text()

    choices = review(transport, "m", "Fix it.", attempts, cases, round_no=2, width=3,
                     plan_path=tmp_path / "plan-round2.json", guidance_mode="none", prompt_chars=200_000)
    assert len(choices) == 1 and len(prompts) == 1 and len(prompts[0]) <= 200_000
    plan = json.loads((tmp_path / "plan-round2.json").read_text())
    budget = plan["prompt_budget"]
    assert budget["budget"] == 200_000 and budget["within_budget"] and budget["halvings"] == 0
    assert budget["chars_before"] > 200_000 >= budget["chars_after"] == len(prompts[0])
    assert budget["stage"] in {stage.name for stage in STAGES}
    assert set(budget["attempts"]) <= {a.name for a in attempts} and budget["attempts"]
    assert plan["review_attempts"][0]["prompt_chars"] == len(prompts[0])
    assert budget["refused"] == []
    assert plan["reports"]["parent"]["prompt_budget"] == {"budget": 1, "chars": 1}, "the analyst's sizing stays in the plan"
    assert "prompt_budget" not in prompts[0].split("## Every attempt so far", 1)[1]
    assert plan["selected_branches"][0]["branch_step"] == 3


def test_a_refused_request_is_retried_with_the_budget_halved_and_every_try_recorded(tmp_path, monkeypatch):
    attempts, cases = _review_setup(tmp_path, monkeypatch)
    prompts = []

    def transport(model, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise PromptRejected("model endpoint refused the request (HTTP 400 Bad Request): too long")
        return plan_text()

    choices = review(transport, "m", "Fix it.", attempts, cases, round_no=2, width=3,
                     plan_path=tmp_path / "plan-round2.json", guidance_mode="none", prompt_chars=400_000)
    assert len(choices) == 1 and len(prompts) == 2
    assert len(prompts[1]) < len(prompts[0]) <= 400_000
    plan = json.loads((tmp_path / "plan-round2.json").read_text())
    assert [a["status"] for a in plan["review_attempts"]] == ["request_rejected", "validated"]
    assert [a["prompt_chars"] for a in plan["review_attempts"]] == [len(prompts[0]), len(prompts[1])]
    assert "HTTP 400" in plan["review_attempts"][0]["request_error"]
    # Halved, and below what was refused: a coarse stage never resends the same prompt.
    assert plan["prompt_budget"]["budget"] == min(200_000, len(prompts[0]) - 1)
    assert plan["prompt_budget"]["halvings"] == 1 and len(prompts[1]) <= plan["prompt_budget"]["budget"]
    assert plan["prompt_budget"]["chars_after"] == len(prompts[1])
    # The refused fit is kept beside the one sent: its budget, stage and what each attempt lost.
    refused = plan["prompt_budget"]["refused"]
    assert len(refused) == 1 and refused[0]["budget"] == 400_000 and refused[0]["chars_after"] == len(prompts[0])
    assert refused[0]["stage"] in {stage.name for stage in STAGES} and refused[0]["stage"] != plan["prompt_budget"]["stage"]
    assert refused[0]["attempts"] and refused[0]["within_budget"] and "refused" not in refused[0]
    assert "validation_error" not in plan


def test_two_halvings_then_the_round_gives_up_with_the_failure_recorded(tmp_path, monkeypatch):
    attempts, cases = _review_setup(tmp_path, monkeypatch)
    prompts = []

    def transport(model, prompt):
        prompts.append(prompt)
        raise PromptRejected("model endpoint refused the request (HTTP 400 Bad Request)")

    choices = review(transport, "m", "Fix it.", attempts, cases, round_no=2, width=3,
                     plan_path=tmp_path / "plan-round2.json", guidance_mode="none", prompt_chars=400_000)
    assert choices == [] and len(prompts) == 3
    assert len(prompts[0]) > len(prompts[1]) > len(prompts[2])
    plan = json.loads((tmp_path / "plan-round2.json").read_text())
    assert [a["status"] for a in plan["review_attempts"]] == ["request_rejected", "request_rejected", "request_error"]
    assert plan["validation_error"].startswith("reviewer failed: model endpoint refused")
    assert plan["prompt_budget"]["budget"] <= 100_000 and plan["prompt_budget"]["halvings"] == 2
    assert [f["chars_after"] for f in plan["prompt_budget"]["refused"]] == [len(prompts[0]), len(prompts[1])]
    assert [f["budget"] for f in plan["prompt_budget"]["refused"]] == [400_000, min(200_000, len(prompts[0]) - 1)]


@pytest.mark.parametrize("budget,statuses", [
    # 100k fits at the narrowed-windows stage; halved, only the floor fits; its
    # refusal leaves nothing smaller to send.
    (100_000, ["request_rejected", "request_error"]),
    # Already at the floor on the first try: a refusal ends the round at once.
    (50_000, ["request_error"]),
])
def test_at_the_floor_a_refusal_ends_the_round_without_another_try(tmp_path, monkeypatch, budget, statuses):
    attempts, cases = _review_setup(tmp_path, monkeypatch, turns=(8, 8, 8))
    prompts = []

    def transport(model, prompt):
        prompts.append(prompt)
        raise PromptRejected("refused")

    review(transport, "m", "Fix it.", attempts, cases, round_no=2, width=3,
           plan_path=tmp_path / "plan-round2.json", guidance_mode="none", prompt_chars=budget)
    plan = json.loads((tmp_path / "plan-round2.json").read_text())
    assert plan["prompt_budget"]["stage"] == FLOOR
    assert [a["status"] for a in plan["review_attempts"]] == statuses
    assert len(prompts) == len(statuses) and plan["prompt_budget"]["halvings"] == len(statuses) - 1
    floor = json.loads(prompts[-1].split("## Every attempt so far\n", 1)[1].split("\n\n## Branch count", 1)[0])
    assert all([m["role"] for m in r["native_history"]["messages"]] == ["omitted"] for r in floor)


def test_a_corrections_feedback_is_kept_when_the_retried_prompt_shrinks(tmp_path, monkeypatch):
    """A validation retry appends the rejected response; a refusal of THAT request
    shrinks the base prompt and keeps the feedback, so the reviewer still sees
    what to correct."""
    attempts, cases = _review_setup(tmp_path, monkeypatch)
    prompts = []

    def transport(model, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            return plan_text(step=99)
        if len(prompts) == 2:
            raise PromptRejected("refused")
        return plan_text()

    choices = review(transport, "m", "Fix it.", attempts, cases, round_no=2, width=3,
                     plan_path=tmp_path / "plan-round2.json", guidance_mode="none",
                     reviewer_max_attempts=3, prompt_chars=400_000)
    assert len(choices) == 1 and len(prompts) == 3
    assert "## Validation feedback" in prompts[1] and "## Validation feedback" in prompts[2]
    assert "no eligible checkpoint" in prompts[2]
    plan = json.loads((tmp_path / "plan-round2.json").read_text())
    assert [a["status"] for a in plan["review_attempts"]] == ["validation_error", "request_rejected", "validated"]


# --- the analyst ------------------------------------------------------------------

def _long_journal(tmp_path, steps=300, output_chars=4000):
    path = tmp_path / "trials" / "parent" / "agent" / "trajectory.jsonl"
    path.parent.mkdir(parents=True)
    records = [{"type": "run.started", "run_id": "parent", "slot": "mini-swe-agent"}]
    for step in range(1, steps + 1):
        records += [{"type": "tool.started", "step": step, "call_id": "c%d" % step, "turn_id": "t%d" % step,
                     "name": "shell", "args": {"command": "step-%d" % step}},
                    {"type": "tool.finished", "call_id": "c%d" % step, "turn_id": "t%d" % step,
                     "status": "ok", "output": ("o%d " % step) * (output_chars // 5)},
                    {"type": "model.turn.completed", "turn_id": "t%d" % step, "call_ids": ["c%d" % step], "step": step},
                    {"type": "checkpoint.captured", "step": step, "turn_id": "t%d" % step,
                     "snapshot_id": "snap-%d" % step, "session_ckpt": "ses-1", "reason": "captured"}]
    records.append({"type": "run.finished", "run_id": "parent", "status": "completed", "usage": {}})
    for seq, record in enumerate(records):
        record.update(v=2, seq=seq)
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    return path


def _report():
    return json.dumps({"failure_reason": "r", "lesson": "l", "salvage": "s",
                       "branch_candidates": [{"step": 2, "why": "w"}]})


def test_the_analyst_request_is_held_to_the_budget_with_the_transcript_elided_first(tmp_path):
    journal = _long_journal(tmp_path)
    attempt = Attempt("parent", branching.outcome_from_journal(journal), Grade(patch="+x\n" * 2000, detail="FAILED"))
    prompts = []

    def transport(model, prompt):
        prompts.append(prompt)
        return _report()

    points = {s: None for s in range(1, 301)}
    report = analyse(transport, "m", "Fix it.", attempt, points, 3, 100_000, prompt_chars=120_000)
    assert len(prompts) == 1 and len(prompts[0]) <= 120_000
    assert "middle elided" in prompts[0] and "step-300" in prompts[0], "the failure is at the end"
    assert "+x" in prompts[0], "the verdict is kept whole while the transcript has room to give"
    assert report["steps"] == 300
    assert report["prompt_budget"]["budget"] == 120_000 and report["prompt_budget"]["halvings"] == 0
    assert report["prompt_budget"]["chars"] == len(prompts[0]) and report["prompt_budget"]["within_budget"]
    # A correction's feedback fits inside the budget too.
    replies = iter(["not json", _report()])
    prompts.clear()
    analyse(lambda m, p: (prompts.append(p), next(replies))[1], "m", "Fix it.", attempt, points, 3, 100_000,
            prompt_chars=120_000)
    assert len(prompts) == 2 and all(len(p) <= 120_000 for p in prompts)


def test_a_huge_verdict_is_clipped_so_the_transcript_keeps_a_quarter_of_the_budget(tmp_path):
    journal = _long_journal(tmp_path, steps=40)
    attempt = Attempt("parent", branching.outcome_from_journal(journal),
                      Grade(patch="+line\n" * 7000, detail="FAILED test_x " * 1500))
    prompts = []
    analyse(lambda m, p: (prompts.append(p), _report())[1], "m", "Fix it.", attempt, {}, 3, 100_000,
            prompt_chars=60_000)
    prompt = prompts[0]
    assert len(prompt) <= 60_000
    transcript = prompt.split('("[N] tool(args) -> result")\n', 1)[1].split("\n\n## Recorded snapshot", 1)[0]
    assert len(transcript) >= 60_000 // 4 - len(branching.ELISION)
    assert "chars elided" in prompt.split("## Private grading evidence", 1)[1].split("## The attempt", 1)[0]
    assert "FAILED test_x" in prompt, "the verdict keeps its tail"


@pytest.mark.parametrize("recovers", [True, False])
def test_truncated_analyst_replies_use_the_existing_correction_budget(tmp_path, recovers):
    from sprout.worker.jobs.review.branch_plan import ReviewResponseTruncated

    journal = _long_journal(tmp_path, steps=3)
    attempt = Attempt("parent", branching.outcome_from_journal(journal), Grade(patch="x"))
    prompts = []
    partial = '{"failure_reason":' + "\t \r\n" * 20_000

    def transport(model, prompt):
        prompts.append(prompt)
        if recovers and len(prompts) == 2:
            return _report()
        raise ReviewResponseTruncated(partial)

    if recovers:
        report = analyse(transport, "m", "Fix it.", attempt, {}, 3, 100_000, max_attempts=2)
        assert report["failure_reason"] == "r"
    else:
        with pytest.raises(ValueError, match="invalid after 2 attempts.*finish_reason=length"):
            analyse(transport, "m", "Fix it.", attempt, {}, 3, 100_000, max_attempts=2)
    assert len(prompts) == 2
    assert partial not in prompts[1] and "finish_reason=length" in prompts[1]


def test_a_refused_analyst_request_is_retried_smaller_then_given_up(tmp_path):
    journal = _long_journal(tmp_path)
    attempt = Attempt("parent", branching.outcome_from_journal(journal), Grade(patch="x"))
    prompts = []

    def once(model, prompt):
        prompts.append(prompt)
        if len(prompts) == 1:
            raise PromptRejected("refused")
        return _report()

    report = analyse(once, "m", "Fix it.", attempt, {}, 3, 100_000, prompt_chars=200_000)
    assert len(prompts) == 2 and len(prompts[1]) <= 100_000 < len(prompts[0]) <= 200_000
    assert report["prompt_budget"]["budget"] == 100_000 and report["prompt_budget"]["halvings"] == 1
    assert report["prompt_budget"]["chars"] == len(prompts[1])

    def always(model, prompt):
        prompts.append(prompt)
        raise PromptRejected("refused")

    prompts.clear()
    with pytest.raises(PromptRejected):
        analyse(always, "m", "Fix it.", attempt, {}, 3, 100_000, prompt_chars=200_000)
    assert [len(p) <= b for p, b in zip(prompts, (200_000, 100_000, 50_000))] == [True, True, True]
    assert len(prompts) == 3


# --- the transport and the wiring -----------------------------------------------------

def test_review_transport_turns_a_400_into_prompt_rejected_and_records_it(tmp_path, monkeypatch):
    from sprout.worker.jobs.review.transport import ReviewTransport

    transport = ReviewTransport(tmp_path, model="fixture", model_endpoint="http://127.0.0.1:18252",
                                api_key_env="TEST_KEY", max_output_tokens=1000)
    monkeypatch.setenv("TEST_KEY", "local-test")
    statuses = iter([400] + [500] * 4)
    monkeypatch.setattr("sprout.worker.jobs.review.transport.time.sleep", lambda s: None)

    def post(url, **kwargs):
        return httpx.Response(next(statuses), text="context length exceeded",
                              request=httpx.Request("POST", url))

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", post)
    with pytest.raises(PromptRejected, match="HTTP 400.*context length exceeded"):
        transport("fixture", "prompt")
    receipt = json.loads((tmp_path / "review-001.json").read_text())
    assert receipt["error"].startswith("PromptRejected:")
    assert len(list(tmp_path.glob("review-*.json"))) == 1, "a refused request is not retried by the transport"
    with pytest.raises(httpx.HTTPStatusError):
        transport("fixture", "prompt")
    assert len(list(tmp_path.glob("review-*.json"))) == 5, "a 500 is an outage: three more tries, then it propagates"


def test_evaluate_passes_the_budget_to_the_analyst_and_the_reviewer(tmp_path, monkeypatch):
    pytest.importorskip("harbor")
    from tests.methods.test_branch_selection import direction, exercise

    exercise(tmp_path, monkeypatch, [{"branches": [direction(2)]}])
    plan = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert plan["prompt_budget"]["budget"] == 400_000 and plan["prompt_budget"]["within_budget"]
    assert plan["reports"]["parent"]["prompt_budget"]["budget"] == 400_000
    assert plan["review_attempts"][0]["prompt_chars"] == len(plan["review_attempts"][0]["prompt"])


def test_evaluate_option_has_the_documented_default_and_a_floor(tmp_path):
    pytest.importorskip("harbor")
    from sprout.eval import evaluate

    runtime = tmp_path / "runtime"
    runtime.touch()
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "m", "--model-endpoint", "http://127.0.0.1:1", "--model-key-env", "KEY",
            "--runtime-bin", str(runtime), "--output", str(tmp_path / "out"), "--volatile-ok"]
    assert evaluate.parse_args(argv).prompt_budget_chars == DEFAULT_PROMPT_CHARS == 400_000
    assert evaluate.parse_args(argv + ["--prompt-budget-chars", "250000"]).prompt_budget_chars == 250_000
    with pytest.raises(SystemExit):
        evaluate.parse_args(argv + ["--prompt-budget-chars", "1000"])


def test_the_qwen_runner_threads_the_budget_to_every_evaluate_command(tmp_path, capsys):
    from examples.qwen import run

    (tmp_path / "tasks" / "t").mkdir(parents=True)
    (tmp_path / "tasks" / "t" / "task.toml").write_text("[task]\nname='org/t'\n")
    argv = ["--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path / "tasks"), "--task", "t",
            "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "runtime"),
            "--model-endpoint", "http://localhost:9000", "--baseline-root", str(tmp_path / "base"), "--plan"]
    assert run.parse_args(argv).prompt_budget_chars == DEFAULT_PROMPT_CHARS
    assert run.main(argv + ["--prompt-budget-chars", "300000"]) == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["manifest"]["prompt_budget_chars"] == 300_000
    for command in printed["example_commands"].values():
        assert command[command.index("--prompt-budget-chars") + 1] == "300000"
    with pytest.raises(SystemExit):
        run.parse_args(argv + ["--prompt-budget-chars", "100"])

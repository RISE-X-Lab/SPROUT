from copy import deepcopy
import json

import pytest

from sprout.contract.mini_history import validate_assistant_turn
from sprout.methods.branch_plan import review_with_feedback
from sprout.methods.prompts import DEFAULT_PROMPTS
from sprout.methods.structured_review import extract_assistant_turn_plan, response_text


def plan(*commands, content="I will inspect the implementation."):
    """What the prompt asks the reviewer for: a branch's text and its bash commands."""
    return {
        "synthesis": "Check the implementation",
        "branches": [{
            "name": "inspect", "base": "parent", "branch_step": 6, "why": "Inspect visible code",
            "content": content, "commands": list(commands) or ["pwd"],
        }],
    }


def nested_plan(command):
    """The wire form the controller builds, as a reviewer used to write it."""
    return {
        "synthesis": "Check the implementation",
        "branches": [{
            "name": "inspect", "base": "parent", "branch_step": 6, "why": "Inspect visible code",
            "assistant_turn": {
                "role": "assistant", "content": "I will inspect the implementation.",
                "tool_calls": [{
                    "id": "fresh-call", "type": "function",
                    "function": {"name": "bash", "arguments": {"command": command}},
                }],
            },
        }],
    }


def fenced(value):
    return "```branch-plan\n" + json.dumps(value, ensure_ascii=False) + "\n```"


def reply(content):
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


def prompt(reports="[]"):
    return DEFAULT_PROMPTS.reviewer_assistant_turn.format(
        problem="Fix the parser", reports=reports, count_rule="At most 4 branches.")


def test_commands_become_bash_calls_with_controller_ids_in_order():
    """The reviewer closes one string per command; the envelope, the ids and the
    argument serialization are the controller's, and the actor sees an ordinary
    assistant message that validate_assistant_turn accepts. The ids name the
    round, the branch and the call, so no two branches or rounds share one."""
    command = '''python - <<'PY'\nprint("a \\"quoted\\" value", r"\\n", "$(literal)", "中文")\nPY\n'''
    wire = plan("cd /app && ls", command, "pytest -q")
    wire["branches"].append({**wire["branches"][0], "name": "other", "commands": ["pwd", "id"]})
    before = deepcopy(wire)
    decoded = extract_assistant_turn_plan(response_text(reply(fenced(wire))), round_no=2)
    branch, _ = decoded["branches"]
    assert set(branch) == {"name", "base", "branch_step", "why", "assistant_turn"}, \
        "content and commands are folded into the assistant turn"
    turn = branch["assistant_turn"]
    assert turn["role"] == "assistant" and turn["content"] == "I will inspect the implementation."
    assert all(c["type"] == "function" and c["function"]["name"] == "bash" for c in turn["tool_calls"])
    assert [json.loads(c["function"]["arguments"]) for c in turn["tool_calls"]] == [
        {"command": "cd /app && ls"}, {"command": command}, {"command": "pytest -q"}]
    assert wire == before
    ids = [c["id"] for b in decoded["branches"] for c in b["assistant_turn"]["tool_calls"]]
    assert ids == ["reviewer-r2-b1-c1", "reviewer-r2-b1-c2", "reviewer-r2-b1-c3", "reviewer-r2-b2-c1", "reviewer-r2-b2-c2"]
    earlier = [c["id"] for b in extract_assistant_turn_plan(fenced(wire), round_no=1)["branches"] for c in b["assistant_turn"]["tool_calls"]]
    assert not set(ids) & set(earlier), "a round-2 branch's history may hold round-1 calls; none is reused"
    history = [{"tool_calls": [{"id": identifier}]} for identifier in earlier]
    for b in decoded["branches"]:
        validate_assistant_turn(b["assistant_turn"], history=history)


def test_the_nested_wire_form_is_still_accepted():
    """Plans recorded before the flat format, and a model that still writes the
    envelope, parse as before: arguments serialized exactly once."""
    command = "grep -rn 'x' . | head"
    decoded = extract_assistant_turn_plan(fenced(nested_plan(command)))
    turn = decoded["branches"][0]["assistant_turn"]
    validate_assistant_turn(turn)
    assert turn["tool_calls"][0]["id"] == "fresh-call"
    assert json.loads(turn["tool_calls"][0]["function"]["arguments"]) == {"command": command}


@pytest.mark.parametrize("form,mutation,field", [
    (plan, lambda p: p["branches"][0].update(commands=[]), "commands"),
    (plan, lambda p: p["branches"][0].update(commands=["ls", 3]), "commands.1"),
    (plan, lambda p: p["branches"][0].update(commands="ls"), "commands"),
    (plan, lambda p: p["branches"][0].pop("content"), "content"),
    (plan, lambda p: p["branches"][0].pop("commands"), "commands"),
    (plan, lambda p: (p["branches"][0].pop("content"), p["branches"][0].pop("commands")), "needs content and commands"),
    (plan, lambda p: p["branches"][0].update(content=["not", "text"]), "content"),
    (plan, lambda p: p["branches"][0].update(hint="unexpected"), "hint"),
    (plan, lambda p: p["branches"][0].update(timeout=30), "timeout"),
    (plan, lambda p: p["branches"][0].update(branch_step=True), "branch_step"),
    (plan, lambda p: p["branches"][0].pop("name"), "name"),
    (plan, lambda p: p.pop("synthesis"), "synthesis"),
    (plan, lambda p: p["branches"][0].update(assistant_turn={"role": "assistant"}), "assistant_turn"),
    # the nested form is checked field by field here too, while the field can still be named to the reviewer
    (nested_plan, lambda p: p["branches"][0]["assistant_turn"]["tool_calls"][0]["function"].update(name="shell"), "function.name"),
    (nested_plan, lambda p: p["branches"][0]["assistant_turn"]["tool_calls"][0]["function"]["arguments"].update(timeout=30), "timeout"),
    (nested_plan, lambda p: p["branches"][0]["assistant_turn"]["tool_calls"][0]["function"].update(arguments='{"command":"pwd"}'),
     "function.arguments"),
    (nested_plan, lambda p: p["branches"][0].update(hint="unexpected"), "hint"),
])
def test_invalid_wire_fields_are_rejected_without_filling_or_coercion(form, mutation, field):
    value = form("pwd")
    mutation(value)
    before = deepcopy(value)
    with pytest.raises(ValueError, match="Structured branch-plan") as caught:
        extract_assistant_turn_plan(fenced(value))
    assert field in str(caught.value), "the error names the field so the reviewer can correct it"
    assert value == before


def test_blank_commands_and_content_are_rejected():
    with pytest.raises(ValueError, match=r"branches\.0\.commands\.1.*blank"):
        extract_assistant_turn_plan(fenced(plan("ls", "   ")))
    with pytest.raises(ValueError, match=r"branches\.0\.content.*blank"):
        extract_assistant_turn_plan(fenced(plan("ls", content=" \n")))


@pytest.mark.parametrize("value,field", [
    (plan("pytest -q", "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"), r"branches\.0\.assistant_turn\.tool_calls\.1"),
    (plan("pytest -q && echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"), r"branches\.0\.assistant_turn\.tool_calls\.0"),
    (nested_plan("echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"), r"branches\.0\.assistant_turn\.tool_calls\.0"),
])
def test_an_inserted_turn_that_submits_is_rejected(value, field):
    """A repair turn that ends with mini's submit command leaves the policy no turn
    to sample after it (6 of 8 repairs in one pydantic search did): the plan goes
    back to the reviewer with the call and the sentinel named, and the prompt says
    not to submit."""
    with pytest.raises(ValueError, match=field + r".*must not submit \(COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT\)"):
        extract_assistant_turn_plan(fenced(value))
    assert "Never submit in this turn" in prompt()


def test_prompt_asks_for_the_flat_form_the_parser_accepts():
    """The prompt sent is the prompt recorded: its example must be a plan the
    parser accepts, and it must not ask for the envelope the controller builds."""
    history = json.dumps([{"native_history": {"messages": [{"role": "user", "content": "task"}]}}])
    rendered = prompt(history)
    assert history in rendered
    assert "Return exactly one fenced code block labelled branch-plan" in rendered
    assert rendered.count("```branch-plan") == 1
    for key in ("synthesis", "branches", "name", "base", "branch_step", "why", "content", "commands"):
        assert '"' + key + '"' in rendered
    for envelope in ('"tool_calls"', '"id":', '"type": "function"', '"arguments"', '"assistant_turn"'):
        assert envelope not in rendered, "the reviewer is not asked to write the tool-call envelope"
    assert "Do not write tool_calls, ids, type or" in rendered
    example = rendered.rsplit("```branch-plan\n", 1)[1].split("\n```", 1)[0]
    parsed = extract_assistant_turn_plan("```branch-plan\n" + example.replace("<int>", "1") + "\n```", round_no=1)
    assert [json.loads(c["function"]["arguments"])["command"] for c in parsed["branches"][0]["assistant_turn"]["tool_calls"]] == [
        "<bash command>", "<next bash command>"]


def test_real_feedback_loop_keeps_response_and_names_the_flat_form():
    original = prompt()
    invalid = plan("pwd")
    invalid["branches"][0]["commands"] = "pwd"
    command = "printf '%s\\n' 'a \"quoted\" value'"
    replies = [fenced(invalid), fenced(plan(command))]
    requests, validated = [], []

    def request(text):
        requests.append(text)
        return response_text(reply(replies[len(requests) - 1]))

    def validate(parsed):
        turn = parsed["branches"][0]["assistant_turn"]
        validate_assistant_turn(turn)
        validated.append(turn)
        return parsed["branches"]

    record = {}
    result = review_with_feedback(request, original, extract_assistant_turn_plan, validate,
                                  max_attempts=3, record=record,
                                  format_hint=DEFAULT_PROMPTS.reviewer_assistant_format_hint)
    assert result is not None and len(requests) == 2 and len(validated) == 1
    assert replies[0] in requests[1]
    feedback = requests[1].split("## Validation feedback\n", 1)[1]
    assert "commands" in feedback and "branches.0.commands" in feedback
    assert "do not write tool_calls, ids or function envelopes" in feedback
    assert "assistant_turn.function.arguments" not in feedback
    assert json.loads(validated[0]["tool_calls"][0]["function"]["arguments"]) == {"command": command}
    assert [row["status"] for row in record["review_attempts"]] == ["validation_error", "validated"]


INVALID_JSON = '```branch-plan\n{"synthesis":"x" "branches":[]}\n```'
PROSE_AROUND_FENCE = "A short explanation.\n\n" + fenced(plan("pwd")) + "\nEnd."
TOOL_REPLY = {"choices": [{"message": {"tool_calls": [{
    "type": "function", "function": {"name": "bash", "arguments": '{"command":"pwd"}'},
}]}}]}


@pytest.mark.parametrize("value,text,error", [
    (reply(INVALID_JSON), INVALID_JSON, "Invalid JSON"),
    (reply(PROSE_AROUND_FENCE), PROSE_AROUND_FENCE, None),
    (TOOL_REPLY, "", "found 0"),
], ids=["invalid-json", "prose-around-fence", "tool-reply"])
def test_reply_text_is_passed_through_and_never_wrapped_repaired_or_adopted(value, text, error):
    """response_text hands the reply on as written -- broken JSON is not repaired,
    prose around a fence is not wrapped again, a tool call is not adopted as a
    plan -- and the parser alone decides what it accepts."""
    assert response_text(value) == text
    if error is None:
        assert extract_assistant_turn_plan(text)["branches"][0]["name"] == "inspect"
    else:
        with pytest.raises(ValueError, match=error):
            extract_assistant_turn_plan(text)


def test_a_bare_json_plan_is_rejected_unless_json_only_is_asked_for():
    """The wrapper's default is the fenced parser with no fallback: a valid plan
    written without its fence is rejected, and response_text does not wrap it.
    Only the structured (JSON-schema) reviewer path opts in with json_only=True
    (branching.py). Other envelope faults -- prose around bare JSON, two fences,
    an unclosed block -- are tests/methods/test_branch_plan.py's."""
    text = json.dumps(plan("pwd"))
    assert response_text(reply(text)) == text
    with pytest.raises(ValueError, match="found 0"):
        extract_assistant_turn_plan(text)
    assert extract_assistant_turn_plan(text, json_only=True)["branches"][0]["name"] == "inspect"

import json

import pytest

from sprout.methods.branch_plan import extract_branch_plan


def fenced(body, label="branch-plan"):
    return f"```{label}\n{body}\n```"


PLAN = {"synthesis": "Check the boundary.", "branches": [
    {"name": "edge", "base": "parent", "branch_step": 2,
     "why": "Retain existing work.", "hint": 'Keep {"x": "}"} and `code` unchanged.'}]}


@pytest.mark.parametrize("text,expected", [
    pytest.param(
        '{"command":"grep example","working_dir":"/app"}\n'
        + fenced('{"branches": [{"name": "wrong-example"}]}', "json") + "\n"
        + "The actual plan follows.\n" + fenced(json.dumps(PLAN)) + "\n"
        + '{"branches": [{"name": "wrong-tail"}]}',
        PLAN, id="ignores-commands-examples-and-prose-outside-its-block"),
    pytest.param(
        '  ```branch-plan \r\n{"synthesis":"No useful branch.","branches":[]}\r\n  ``` \r\n',
        {"synthesis": "No useful branch.", "branches": []}, id="crlf-indent-trailing-space"),
    pytest.param(
        '```branch-plan\n{"branches":[{"hint":"first line\nsecond line"}]}\n```',
        {"branches": [{"hint": "first line\nsecond line"}]}, id="literal-newline-in-hint"),
])
def test_plan_is_read_from_its_labelled_block(text, expected):
    assert extract_branch_plan(text) == expected


@pytest.mark.parametrize("text,expected_error", [
    ('{"branches":[]}', "found 0"),
    (fenced('{"branches":[]}', "json"), "found 0"),
    (fenced('{"branches":[]}', "branch-plan-example"), "found 0"),
    ("An inline ```branch-plan marker is not a block.", "found 0"),
    pytest.param('> ```branch-plan\n> {"branches": []}\n> ```\n', "found 0",
                 id="blockquote-is-not-a-block"),
    ('```branch-plan\n{"branches":[]}', "not closed"),
    (fenced('{"branches":['), "Invalid JSON"),
    (fenced('{"command":"grep example"}'), "branches list"),
    (fenced('[]'), "branches list"),
    (fenced('null'), "branches list"),
    (fenced('{"branches":{}}'), "branches list"),
    (fenced('{"branches":[]} {"branches":[]}'), "Invalid JSON"),
    (fenced('{"branches":[]}') + "\n" + fenced('{"branches":[]}'), "found 2"),
    (fenced('{"branches":[]}') + '\n```branch-plan\n{"branches":[]}', "found 2"),
])
def test_bad_envelopes_fail_without_falling_back_to_another_json(text, expected_error):
    with pytest.raises(ValueError, match=expected_error):
        extract_branch_plan(text)


def test_plan_tag_inside_an_example_fence_is_not_a_second_plan():
    example = '````text\n```branch-plan\n{"branches": [{"name":"example"}]}\n```\n````\n'
    assert extract_branch_plan(example + fenced('{"branches":[]}')) == {"branches": []}
    with pytest.raises(ValueError, match="found 0"):
        extract_branch_plan(example)

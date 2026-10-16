"""The leak gate: hidden test names a reviewer-written turn mentions that h_s never showed."""

import json

import pytest

from sprout.worker.jobs.review.leak.terms import hidden_terms, hidden_terms_of, leak_terms, turn_text, visible_text

TARGET = "tests/test_hidden.py::test_secret_behaviour"
TEST_FILE = "tests/test_hidden.py"


def call(identifier, command):
    return {"id": identifier, "type": "function",
            "function": {"name": "bash", "arguments": json.dumps({"command": command})}}


def turn(*commands, content="I'll run the relevant check.", **extra):
    return {"role": "assistant", "content": content,
            "tool_calls": [call(f"c{n}", command) for n, command in enumerate(commands)], **extra}


def history(output):
    """h_s: the task, one inspection and the tool output it produced."""
    return [{"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Fix the failing behaviour in src/module.py."},
            {"role": "assistant", "content": "Look first.", "reasoning_content": "Inspect the module.",
             "tool_calls": [call("h1", "pytest -q")]},
            {"role": "tool", "tool_call_id": "h1", "content": output}]


def test_a_hidden_test_name_is_a_leak_only_while_h_s_has_not_shown_it():
    hidden = hidden_terms(fail_to_pass=[TARGET], test_patch_paths=[TEST_FILE])
    assert hidden == {TARGET, "test_hidden", "test_secret_behaviour", TEST_FILE, "test_hidden.py"}
    repair = turn("pytest -q -k test_secret_behaviour")
    unseen = visible_text(history("1 failed, 3 passed"))
    assert leak_terms(repair, hidden, unseen) == ["test_secret_behaviour"]
    # The actor's own test run printed the name, so the turn shows nothing new.
    seen = visible_text(history("FAILED src/test_module.py::test_secret_behaviour - AssertionError"))
    assert leak_terms(repair, hidden, seen) == []
    assert leak_terms(turn("git diff"), hidden, unseen) == []


def test_a_hidden_test_path_in_reasoning_is_a_leak_and_so_is_its_basename():
    hidden = hidden_terms(fail_to_pass=[], test_patch_paths=[TEST_FILE])
    visible = visible_text(history("1 failed"))
    # The form a repair continuation is trained on: content moved to reasoning_content.
    reasoned = turn("ls", content="", reasoning_content=f"The checks for this live in {TEST_FILE}.")
    assert leak_terms(reasoned, hidden, visible) == ["test_hidden.py", TEST_FILE]
    assert leak_terms(turn("cat test_hidden.py"), hidden, visible) == ["test_hidden.py"]


def test_a_call_shows_its_raw_arguments_and_the_command_they_decode_to():
    hidden = hidden_terms(fail_to_pass=[], test_patch_paths=["tests/test_café.py"])
    repair = turn("pytest tests/test_café.py")
    raw = repair["tool_calls"][0]["function"]["arguments"]
    assert "tests/test_café.py" not in raw, "the raw arguments escape it"
    assert raw in turn_text(repair) and "tests/test_café.py" in turn_text(repair)
    assert leak_terms(repair, hidden, visible_text(history("ok"))) == ["test_café.py", "tests/test_café.py"]
    # A history that ran the path in a command has shown it, escaped or not.
    assert leak_terms(repair, hidden, visible_text([turn("cat tests/test_café.py")])) == []
    # So has a call the schema check rejected: its arguments are not JSON, the raw text is what it showed.
    broken = {"role": "assistant", "content": "", "tool_calls": [{
        "id": "b1", "type": "function",
        "function": {"name": "bash", "arguments": '{"command": "cat tests/test_café.py"'}}]}
    assert leak_terms(repair, hidden, visible_text([broken])) == []


def test_a_test_id_printed_by_the_actors_own_run_is_visible_through_the_escaped_observation():
    target = "tests/test_café.py::test_parse[<lambda>-it's]"
    hidden = hidden_terms(fail_to_pass=[target], test_patch_paths=[])
    # How mini renders an observation: JSON through Jinja's tojson, which also escapes <, >, & and '.
    observation = json.dumps({"returncode": 1, "output": f"FAILED {target}\n"})
    for character, escape in (("<", "\\u003c"), (">", "\\u003e"), ("&", "\\u0026"), ("'", "\\u0027")):
        observation = observation.replace(character, escape)
    assert target not in observation
    repair = turn(f'pytest -q "{target}"')
    assert target in leak_terms(repair, hidden, visible_text(history("1 failed")))
    assert leak_terms(repair, hidden, visible_text(history(observation))) == []


def test_generic_names_are_never_terms_and_an_empty_vocabulary_flags_nothing():
    terms = hidden_terms(fail_to_pass=["tests", "py", "test_x"],
                         test_patch_paths=["tests", "py", "conftest.py", "tests/conftest.py",
                                           "src/pkg/__init__.py", "a/b.py"])
    assert not {"tests", "py", "conftest.py", "__init__.py", "test_x", "b.py"} & terms
    assert terms == {"tests/conftest.py", "src/pkg/__init__.py", "a/b.py"}, "a path long enough is still a term"
    assert hidden_terms(fail_to_pass=["tests/test_x.py::test_case[12345678]"], test_patch_paths=[]) == {
        "tests/test_x.py::test_case[12345678]", "test_case"}, "an all-digit token is not a term"
    everything = turn("cat tests/conftest.py", content="Check conftest.py and the tests in py files.")
    assert leak_terms(everything, frozenset(), "") == []
    assert leak_terms(everything, hidden_terms(fail_to_pass=[], test_patch_paths=[]), "") == []


def test_hidden_terms_of_reads_only_the_two_grade_facts():
    assert hidden_terms_of({"resolved": False, "progress": 0.0}) is None
    assert hidden_terms_of({"resolved": False, "fail_to_pass": None, "test_patch_paths": None}) is None
    assert hidden_terms_of({"resolved": False, "fail_to_pass": [], "test_patch_paths": []}) == frozenset()
    assert hidden_terms_of({"resolved": False, "test_patch_paths": [TEST_FILE]}) == {TEST_FILE, "test_hidden.py"}
    # PASS_TO_PASS tests appear in test_statuses, and touched_test_paths is the
    # patch's own business: neither adds a term.
    grade = {"resolved": False, "fail_to_pass": [TARGET], "touched_test_paths": ["tests/test_visible.py"],
             "test_statuses": {"tests/test_visible.py::test_passing_before": "PASSED", TARGET: "FAILED"}}
    assert hidden_terms_of(grade) == hidden_terms(fail_to_pass=[TARGET], test_patch_paths=None)
    with pytest.raises(ValueError, match="fail_to_pass must be a list of strings"):
        hidden_terms_of({"resolved": False, "fail_to_pass": json.dumps([TARGET])})
    with pytest.raises(ValueError, match="test_patch_paths must be a list of strings"):
        hidden_terms_of({"resolved": False, "test_patch_paths": [TEST_FILE, None]})

"""``leak_terms``: the hidden test names that a reviewer-written turn gives away, and h_s never showed."""

from __future__ import annotations

import json
from typing import Any

import pytest

from sprout.worker.jobs.review.leak.terms import hidden_terms, hidden_terms_of, leak_terms, visible_text

TARGET = "tests/test_hidden.py::test_secret_behaviour"
TEST_FILE = "tests/test_hidden.py"


def call(call_id: str, command: str) -> dict[str, Any]:
    return {
        "id": call_id,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def turn(*commands: str, content: str = "I'll run the relevant check.", **extra: Any) -> dict[str, Any]:
    calls = [call("c%d" % n, command) for n, command in enumerate(commands, 1)]
    return {"role": "assistant", "content": content, "tool_calls": calls, **extra}


def history(output: str) -> list[dict[str, Any]]:
    """h_s: the task, one test run, and the observation that the run gave, as mini renders it."""
    return [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Fix the failing behaviour in src/module.py."},
        {"role": "assistant", "content": "Run the tests.", "tool_calls": [call("h1", "pytest -q")]},
        {"role": "tool", "tool_call_id": "h1", "content": json.dumps({"returncode": 1, "output": output})},
    ]


def test_the_vocabulary_is_each_path_its_basename_each_test_id_and_its_long_identifier_tokens():
    terms = hidden_terms(fail_to_pass=[TARGET], test_patch_paths=[TEST_FILE])

    assert terms == {TARGET, "test_secret_behaviour", "test_hidden", TEST_FILE, "test_hidden.py"}


@pytest.mark.parametrize(
    "paths, terms",
    [
        (["a/bcd"], set()),
        (["a/bcde"], {"a/bcde"}),
        (["a/b.py"], {"a/b.py"}),
        (["tests/conftest.py"], {"tests/conftest.py"}),
        (["conftest.py"], set()),
        (["src/pkg/__init__.py"], {"src/pkg/__init__.py"}),
        (["tests/test_x.py"], {"tests/test_x.py", "test_x.py"}),
    ],
    ids=[
        "5-chars",
        "6-chars",
        "short-basename",
        "generic-basename",
        "generic-name",
        "generic-init",
        "long-basename",
    ],
)
def test_a_path_counts_from_6_characters_and_its_basename_only_when_as_long_and_not_generic(paths, terms):
    assert hidden_terms(fail_to_pass=None, test_patch_paths=paths) == terms


@pytest.mark.parametrize(
    "names, terms",
    [
        (["t::abcd"], set()),
        (["t::abcde"], {"t::abcde"}),
        (["t::abcdefg"], {"t::abcdefg"}),
        (["t::abcdefgh"], {"t::abcdefgh", "abcdefgh"}),
        (["  t::abcdefgh  "], {"t::abcdefgh", "abcdefgh"}),
        (["t::case[12345678]"], {"t::case[12345678]"}),
    ],
    ids=["7-char-name", "8-char-name", "7-char-token", "8-char-token", "padded-name", "all-digit-token"],
)
def test_a_test_id_and_each_of_its_tokens_count_from_8_characters_but_not_all_digits(names, terms):
    assert hidden_terms(fail_to_pass=names, test_patch_paths=None) == terms


@pytest.mark.parametrize(
    "facts, error",
    [
        ({"fail_to_pass": json.dumps([TARGET])}, "fail_to_pass must be a list of strings"),
        ({"test_patch_paths": [TEST_FILE, None]}, "test_patch_paths must be a list of strings"),
        ({"test_patch_paths": {TEST_FILE: 1}}, "test_patch_paths must be a list of strings"),
    ],
    ids=["a-json-string", "a-null-item", "a-mapping"],
)
def test_a_fact_that_is_not_a_list_of_strings_is_refused_rather_than_read_as_clean(facts, error):
    """A list sent as a JSON string would split into characters, none of them a term."""
    with pytest.raises(ValueError, match=error):
        hidden_terms_of({"resolved": False, **facts})


@pytest.mark.parametrize(
    "grade, terms",
    [
        ({"resolved": False, "progress": 0.0}, None),
        ({"resolved": False, "fail_to_pass": None, "test_patch_paths": None}, None),
        ({"resolved": False, "fail_to_pass": [], "test_patch_paths": []}, frozenset()),
        ({"resolved": False, "test_patch_paths": [TEST_FILE]}, {TEST_FILE, "test_hidden.py"}),
        (
            {
                "resolved": False,
                "fail_to_pass": [TARGET],
                "touched_test_paths": ["tests/test_visible.py"],
                "test_statuses": {"tests/test_visible.py::test_passing_before": "PASSED", TARGET: "FAILED"},
            },
            {TARGET, "test_secret_behaviour", "test_hidden"},
        ),
    ],
    ids=["no-fact", "null-facts", "empty-facts", "paths-only", "statuses-and-touched-paths-add-nothing"],
)
def test_a_grades_vocabulary_comes_from_its_two_hidden_facts_and_is_none_without_them(grade, terms):
    assert hidden_terms_of(grade) == terms


@pytest.mark.parametrize(
    "output, found",
    [
        ("1 failed, 3 passed", ["test_secret_behaviour"]),
        ("FAILED src/test_module.py::test_secret_behaviour - AssertionError", []),
    ],
    ids=["h_s-never-showed-it", "the-actors-own-run-printed-it"],
)
def test_a_hidden_name_in_the_turn_is_a_leak_only_while_h_s_has_not_shown_it(output, found):
    hidden = hidden_terms(fail_to_pass=[TARGET], test_patch_paths=[TEST_FILE])

    assert (
        leak_terms(turn("pytest -q -k test_secret_behaviour"), hidden, visible_text(history(output))) == found
    )


def test_a_hidden_path_in_the_reasoning_is_a_leak_and_the_terms_are_sorted():
    """A repair continuation is trained with its text moved to ``reasoning_content``."""
    hidden = hidden_terms(fail_to_pass=[], test_patch_paths=[TEST_FILE])
    reasoned = turn("ls", content="", reasoning_content=f"The checks for this live in {TEST_FILE}.")

    assert leak_terms(reasoned, hidden, visible_text(history("1 failed"))) == ["test_hidden.py", TEST_FILE]


def test_a_name_that_json_escapes_in_the_raw_arguments_is_found_in_the_decoded_command():
    hidden = hidden_terms(fail_to_pass=[], test_patch_paths=["tests/test_café.py"])

    assert leak_terms(turn("pytest tests/test_café.py"), hidden, visible_text(history("ok"))) == [
        "test_café.py",
        "tests/test_café.py",
    ]


@pytest.mark.parametrize(
    "arguments",
    [json.dumps({"command": "cat tests/test_café.py"}), '{"command": "cat tests/test_café.py"'],
    ids=["escaped-json", "broken-json-shows-its-raw-text"],
)
def test_a_name_that_an_earlier_call_of_h_s_ran_is_visible(arguments):
    hidden = hidden_terms(fail_to_pass=[], test_patch_paths=["tests/test_café.py"])
    earlier = {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {"id": "e1", "type": "function", "function": {"name": "bash", "arguments": arguments}}
        ],
    }

    assert leak_terms(turn("pytest tests/test_café.py"), hidden, visible_text([earlier])) == []


def test_a_test_id_that_the_actors_run_printed_is_visible_through_the_escaped_observation():
    """mini renders an observation through Jinja's tojson, which escapes ``<``, ``>``, ``&`` and ``'``."""
    target = "tests/test_café.py::test_parse[<lambda>-it's]"
    observation = json.dumps({"returncode": 1, "output": f"FAILED {target}\n"})
    observation = observation.replace("<", "\\u003c").replace(">", "\\u003e").replace("'", "\\u0027")
    shown = [{"role": "tool", "tool_call_id": "h1", "content": observation}]

    assert (
        leak_terms(
            turn(f'pytest -q "{target}"'),
            hidden_terms(fail_to_pass=[target], test_patch_paths=[]),
            visible_text(shown),
        )
        == []
    )


def test_an_empty_vocabulary_flags_nothing():
    everything = turn("cat tests/conftest.py", content="Check conftest.py and test_secret_behaviour.")

    assert leak_terms(everything, frozenset(), "") == []

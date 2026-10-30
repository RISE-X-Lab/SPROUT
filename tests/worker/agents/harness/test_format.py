"""The strict check of a raw Qwen3.5 turn: Arctic-SWE's 13 malformed turns, the harness's preferences,
the canonical rendering, and the raw text that the server's log-probs carry."""

from __future__ import annotations

import pytest

from sprout.worker.agents.harness.qwen_format import (
    Call,
    FormatViolation,
    Turn,
    check,
    parse,
    raw_completion,
    render,
    structured_completion,
)

CALL = (
    "<tool_call>\n<function=execute_bash>\n<parameter=command>\nls\n</parameter>\n</function>\n</tool_call>"
)
VALID = "I will list the files.\n</think>\n\n" + CALL
LS = Turn(reasoning="I will list the files.", content="", calls=(Call("execute_bash", (("command", "ls"),)),))

#: Each malformed turn of Arctic-SWE's Table 6, built from one valid ``execute_bash(ls)`` turn.
TABLE_6 = {
    "no </think>": ("I will list the files.\n\n" + CALL, "unclosed_reasoning"),
    "second </think>": ("I will list the files.\n</think>\n\n</think>\n\n" + CALL, "repeated_think_tag"),
    "extra <think>": ("<think>\nI will list the files.\n</think>\n\n" + CALL, "repeated_think_tag"),
    "call inside reasoning": ("I will list the files.\n" + CALL + "\n</think>\n\n", "call_in_reasoning"),
    "text after </tool_call>": (VALID + "\nThat should list them.", "text_after_call"),
    "unclosed <tool_call>": (VALID.removesuffix("\n</tool_call>"), "unclosed_tool_call"),
    "nested <tool_call>": (
        "I will list the files.\n</think>\n\n<tool_call>\n" + CALL + "\n</tool_call>",
        "nested_tool_call",
    ),
    "no </parameter>": (VALID.replace("ls\n</parameter>\n", "ls\n"), "unclosed_parameter"),
    "duplicate parameter": (
        VALID.replace(
            "<parameter=command>\nls\n</parameter>\n",
            "<parameter=command>\nls\n</parameter>\n<parameter=command>\nrm -rf build\n</parameter>\n",
        ),
        "duplicate_parameter",
    ),
    "no </function>": (VALID.replace("</function>\n", ""), "unclosed_function"),
    "no > after <function=": (
        VALID.replace("<function=execute_bash>", "<function=execute_bash"),
        "malformed_tag",
    ),
    "two functions in one call": (
        VALID.replace(
            "</function>\n</tool_call>",
            "</function>\n<function=execute_bash>\n<parameter=command>\npwd\n</parameter>\n</function>\n</tool_call>",
        ),
        "multiple_functions",
    ),
    "no <function=": (
        "I will list the files.\n</think>\n\n<tool_call>\n<parameter=command>\nls\n</parameter>\n"
        "</tool_call>",
        "missing_function",
    ),
}


def test_a_valid_turn_parses_to_its_reasoning_and_its_one_call() -> None:
    assert check(VALID, finish_reason="stop") == LS


@pytest.mark.parametrize("flaw", sorted(TABLE_6))
def test_each_malformed_turn_of_table_6_is_rejected(flaw: str) -> None:
    raw, code = TABLE_6[flaw]

    with pytest.raises(FormatViolation) as rejected:
        check(raw, finish_reason="stop")

    assert rejected.value.code == code
    assert rejected.value.message


@pytest.mark.parametrize(
    ("raw", "code"),
    [
        ("\n</think>\n\n" + CALL, "empty_reasoning"),
        ("I will list the files.\n</think>\n\nLet me run ls.\n\n" + CALL, "prose_before_call"),
        (VALID + "\n" + CALL, "multiple_calls"),
        ("I am done.\n</think>\n\nThe answer is 42.", "missing_tool_call"),
    ],
    ids=["empty reasoning", "prose before the call", "two calls", "no call"],
)
def test_the_harness_preferences_reject_a_well_formed_turn_too(raw: str, code: str) -> None:
    with pytest.raises(FormatViolation) as rejected:
        check(raw, finish_reason="stop")

    assert rejected.value.code == code


def test_a_response_cut_off_by_the_token_limit_is_rejected_whatever_it_holds() -> None:
    with pytest.raises(FormatViolation) as rejected:
        check(VALID, finish_reason="length")

    assert rejected.value.code == "response_truncated"


def test_parameters_keep_their_order_and_their_inner_newlines() -> None:
    raw = (
        "Edit it.\n</think>\n\n<tool_call>\n<function=str_replace>\n"
        "<parameter=file_path>\na.py\n</parameter>\n"
        "<parameter=old_string>\ndef f():\n    pass\n</parameter>\n"
        "<parameter=new_string>\nx\n\n</parameter>\n"
        "</function>\n</tool_call>"
    )

    turn = parse(raw)

    assert turn.calls[0].parameters == (
        ("file_path", "a.py"),
        ("old_string", "def f():\n    pass"),
        ("new_string", "x\n"),
    )


def test_render_writes_the_chat_templates_text_so_a_canonical_turn_round_trips() -> None:
    assert render(LS) == VALID
    assert render(parse(VALID)) == VALID


def test_a_turn_with_extra_whitespace_parses_but_does_not_round_trip() -> None:
    raw = "I will list the files.\n\n</think>\n\n" + CALL

    assert parse(raw) == LS
    assert render(parse(raw)) != raw


def test_raw_completion_joins_the_token_bytes_and_drops_the_end_token() -> None:
    # The three bytes of 你 arrive in two tokens, so neither token's text is valid alone.
    tokens = [
        {"token": "ab", "bytes": [97, 98]},
        {"token": "�", "bytes": [0xE4, 0xBD]},
        {"token": "�", "bytes": [0xA0]},
        {"token": "<|im_end|>", "bytes": list(b"<|im_end|>")},
    ]

    assert raw_completion({"logprobs": {"content": tokens}}) == "ab你"


def test_raw_completion_is_none_without_log_probs() -> None:
    assert raw_completion({"message": {"content": "x"}}) is None
    assert raw_completion({"logprobs": None}) is None


def test_structured_completion_rebuilds_the_canonical_text_of_a_parsed_response() -> None:
    message = {
        "content": "",
        "reasoning_content": "I will list the files.",
        "tool_calls": [
            {
                "id": "c",
                "type": "function",
                "function": {"name": "execute_bash", "arguments": '{"command": "ls"}'},
            }
        ],
    }

    assert structured_completion(message) == VALID

"""The training-message format of an exported attempt: hints, cleanup, tools, failure kinds and truncation."""

from __future__ import annotations

import re
from copy import deepcopy
from typing import Any

import pytest

from sprout.contract.messages import (
    branch_origin,
    clean_messages,
    export_tools,
    failure_kind,
    is_truncated_result,
    mark_hint,
)
from tests.contract.builders import bash_call, bash_tools


def conversation(*opening: dict[str, Any]) -> list[dict[str, Any]]:
    """``opening``, then one closed bash turn: the shortest history that has training output."""
    return [
        *opening,
        {"role": "assistant", "content": "", "tool_calls": [bash_call("ls", "c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": "README\n"},
    ]


def test_a_hint_is_marked_with_the_training_hint_tags():
    """Stored native histories hold these tags, and a later export strips them again."""
    assert mark_hint("Try the parser.") == "<sprout_training_hint>Try the parser.</sprout_training_hint>"


@pytest.mark.parametrize("hint", ["see <sprout_training_hint>", "</sprout_training_hint> here"])
def test_a_hint_that_holds_a_tag_is_not_marked(hint):
    with pytest.raises(ValueError, match="^Branch hint contains reserved delimiters$"):
        mark_hint(hint)


@pytest.mark.parametrize(
    "content, trained",
    [
        pytest.param("Fix the bug.\n" + mark_hint("Look at parse()."), "Fix the bug.\n", id="after-the-task"),
        pytest.param("a" + mark_hint("x") + "b" + mark_hint("y") + "c", "abc", id="text-between-two-hints"),
        pytest.param(
            "Fix " + mark_hint("first line\nsecond line") + "it.", "Fix it.", id="hint-of-two-lines"
        ),
    ],
)
@pytest.mark.parametrize("role", ["system", "user"])
def test_the_actor_reads_a_hint_and_training_does_not(role, content, trained):
    messages = conversation({"role": role, "content": content})

    cleaned = clean_messages(messages)

    assert cleaned == conversation({"role": role, "content": trained})


@pytest.mark.parametrize(
    "content",
    [mark_hint("Try the parser.") + mark_hint("Then the lexer."), "\n" + mark_hint("Try the parser.") + "\n"],
    ids=["two-hints", "hint-between-newlines"],
)
def test_a_message_that_held_only_hints_is_left_out(content):
    messages = conversation({"role": "user", "content": "Fix the bug."}, {"role": "user", "content": content})

    cleaned = clean_messages(messages)

    assert cleaned == conversation({"role": "user", "content": "Fix the bug."})


def test_a_blank_message_without_a_hint_is_kept():
    messages = conversation({"role": "user", "content": ""})

    cleaned = clean_messages(messages)

    assert cleaned == conversation({"role": "user", "content": ""})


@pytest.mark.parametrize(
    "assistant_content, tool_content",
    [(mark_hint("literal assistant output"), "notes\n"), ("", mark_hint("literal tool output"))],
    ids=["assistant", "tool"],
)
def test_text_like_a_hint_in_an_assistant_or_tool_message_is_trained_on(assistant_content, tool_content):
    messages = [
        {"role": "user", "content": "Fix the bug."},
        {"role": "assistant", "content": assistant_content, "tool_calls": [bash_call("cat notes", "c1")]},
        {"role": "tool", "tool_call_id": "c1", "content": tool_content},
    ]

    cleaned = clean_messages(messages)

    assert cleaned == messages


def test_a_developer_message_is_trained_as_a_system_message():
    messages = conversation({"role": "developer", "content": "You are mini."})

    cleaned = clean_messages(messages)

    assert cleaned == conversation({"role": "system", "content": "You are mini."})


def test_cleaning_leaves_the_given_messages_as_they_were():
    messages = conversation({"role": "developer", "content": "Fix it." + mark_hint("Try the parser.")})
    before = deepcopy(messages)

    clean_messages(messages)

    assert messages == before


@pytest.mark.parametrize(
    "messages, error",
    [
        pytest.param(
            conversation({"role": "function", "content": "x"}),
            "Unsupported message role 'function'",
            id="role",
        ),
        pytest.param(
            conversation({"role": "user", "content": [{"type": "text", "text": "Fix it."}]}),
            "Training export requires text content",
            id="content-parts",
        ),
        pytest.param(
            conversation({"role": "user", "content": "Fix it. <sprout_training_hint>Try"}),
            "Incomplete hint boundary",
            id="open-tag",
        ),
        pytest.param(
            conversation({"role": "system", "content": "Try</sprout_training_hint> it."}),
            "Incomplete hint boundary",
            id="close-tag",
        ),
        pytest.param(
            [
                {"role": "assistant", "content": "", "tool_calls": [bash_call("ls", "c1")]},
                {"role": "tool", "tool_call_id": "c1", "content": "README\n"},
                {"role": "assistant", "content": "", "tool_calls": [bash_call("pwd", "c1")]},
                {"role": "tool", "tool_call_id": "c1", "content": "/testbed\n"},
            ],
            "Duplicate tool call",
            id="call-id-twice",
        ),
        pytest.param(
            conversation({"role": "tool", "tool_call_id": "c0", "content": "early"}),
            "Unmatched tool result",
            id="result-before-call",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Fix the bug."},
                {
                    "role": "assistant",
                    "content": "",
                    "tool_calls": [bash_call("ls", "c1"), bash_call("pwd", "c2")],
                },
                {"role": "tool", "tool_call_id": "c1", "content": "README\n"},
            ],
            "Native history has incomplete tool calls",
            id="call-without-result",
        ),
        pytest.param(
            [
                {"role": "user", "content": "Fix the bug."},
                {"role": "user", "content": mark_hint("Try the parser.")},
            ],
            "Native history has no assistant output",
            id="no-assistant",
        ),
    ],
)
def test_a_history_that_is_not_a_training_conversation_is_refused(messages, error):
    with pytest.raises(ValueError, match="^%s$" % re.escape(error)):
        clean_messages(messages)


def test_the_exported_tools_are_the_offered_schemas_once_for_each_name():
    bash = bash_tools()[0]
    submit = {"type": "function", "function": {"name": "submit", "parameters": {"type": "object"}}}
    events = [
        {"type": "rollout.model_tools", "shape": "chat/completions", "tools": [bash]},
        {"type": "rollout.model_tools", "shape": "chat/completions", "tools": [bash, submit]},
    ]

    tools = export_tools(events)

    assert tools == [bash, submit]


def test_a_tool_schema_that_changed_during_the_trajectory_is_refused():
    events = [
        {"type": "rollout.model_tools", "shape": "chat/completions", "tools": bash_tools()},
        {"type": "rollout.model_tools", "shape": "chat/completions", "tools": bash_tools(maxLength=10)},
    ]

    with pytest.raises(ValueError, match="^Tool schema changed during the trajectory$"):
        export_tools(events)


@pytest.mark.parametrize(
    "event",
    [
        {"type": "rollout.model_tools", "shape": "responses", "tools": bash_tools()},
        {
            "type": "rollout.model_tools",
            "shape": "chat/completions",
            "tools": [{**bash_tools()[0], "type": "custom"}],
        },
    ],
    ids=["responses-shape", "custom-tool"],
)
def test_a_tool_that_is_not_a_chat_completions_function_is_refused(event):
    with pytest.raises(ValueError, match="^Unsupported native tool schema for training$"):
        export_tools([event])


@pytest.mark.parametrize(
    "result, kind",
    [
        pytest.param({"status": "completed", "error": "HTTP 502 Bad Gateway"}, None, id="completed"),
        pytest.param(
            {"status": "error", "error": "Frozen dataset checksum changed", "failure_kind": "infrastructure"},
            "infrastructure",
            id="checksum-changed",
        ),
        pytest.param(
            {"status": "timeout", "error": "HTTP 504 during exhausted budget"}, "actor", id="timeout"
        ),
        pytest.param({"status": "step_limit", "error": "connection reset"}, "actor", id="step-limit-status"),
        pytest.param({"status": "cost_limit", "error": "connection reset"}, "actor", id="cost-limit-status"),
        pytest.param(
            {"status": "budget_exhausted", "error": "connection reset"}, "actor", id="budget-status"
        ),
        pytest.param(
            {"status": "error", "error": "run budget exhausted: spent $1.0000 of $1.0000 after HTTP 503"},
            "actor",
            id="budget-exhausted",
        ),
        pytest.param({"status": "error", "error": "Cost limit reached (503)"}, "actor", id="cost-limit"),
        pytest.param({"status": "error", "error": "Step limit reached after 504"}, "actor", id="step-limit"),
        pytest.param(
            {"status": "error", "error": "Maximum turns reached at HTTP 502"}, "actor", id="maximum-turns"
        ),
        pytest.param({"status": "error", "error": "HTTP 502 Bad Gateway"}, "infrastructure", id="502"),
        pytest.param({"status": "error", "error": "503 Service Unavailable"}, "infrastructure", id="503"),
        pytest.param({"status": "error", "error": "upstream answered 504"}, "infrastructure", id="504"),
        pytest.param(
            {"status": "error", "error": "TransportClosedError"}, "infrastructure", id="transport-closed"
        ),
        pytest.param(
            {"status": "error", "error": "ConnectionError: Connection reset by peer"},
            "infrastructure",
            id="connection-reset",
        ),
        pytest.param(
            {"status": "error", "error": "Connection refused"}, "infrastructure", id="connection-refused"
        ),
        pytest.param(
            {"status": "error", "error": "model stream stalled"}, "infrastructure", id="stream-stalled"
        ),
        pytest.param(
            {"status": "error", "error": "execution uncertain at step 3 (call c1)"},
            "infrastructure",
            id="execution-uncertain",
        ),
        pytest.param(
            {"status": "error", "error": "ToolInfraError: tool execution did not settle: timeout"},
            "infrastructure",
            id="did-not-settle",
        ),
        pytest.param(
            {"status": "error", "error": "final snapshot failed"}, "infrastructure", id="final-snapshot"
        ),
        pytest.param({"status": "error", "error": "httpx.ReadTimeout"}, "infrastructure", id="read-timeout"),
        pytest.param(
            {"status": "error", "error": "listening on port 5020"}, "actor", id="code-inside-a-number"
        ),
        pytest.param(
            {"status": "error", "error": "No patch produced", "failure_kind": "infrastructure"},
            "infrastructure",
            id="own-kind",
        ),
        pytest.param({"status": "error", "failure_kind": None}, "actor", id="no-error"),
    ],
)
def test_a_failure_is_of_the_kind_that_its_status_and_error_say(result, kind):
    """Each marker is part of an error text that a Sprout module, a client library or a gateway writes."""
    assert failure_kind(result) == kind


def truncated(**changes: Any) -> dict[str, Any]:
    """An attempt that its turn limit ended, with a final snapshot, and ``changes``."""
    return {
        "status": "truncated",
        "stop_reason": "max_turns_reached",
        "final_snapshot_id": "snapshot-7",
        "failure_kind": "actor",
        **changes,
    }


@pytest.mark.parametrize(
    "result, expected",
    [
        pytest.param(truncated(), True, id="turn-limit"),
        pytest.param(truncated(stop_reason="timeout"), True, id="time-limit"),
        pytest.param(truncated(stop_reason="interrupted", failure_kind=None), True, id="salvaged"),
        pytest.param(truncated(status="timeout"), False, id="not-truncated"),
        pytest.param(truncated(stop_reason="context_exhausted"), False, id="other-stop"),
        pytest.param(truncated(final_snapshot_id=""), False, id="empty-snapshot"),
        pytest.param(truncated(final_snapshot_id=None), False, id="no-snapshot"),
        pytest.param(truncated(final_snapshot_id=7), False, id="snapshot-not-text"),
        pytest.param(truncated(failure_kind="infrastructure"), False, id="infrastructure"),
    ],
)
def test_only_a_truncated_attempt_with_a_verified_cutoff_is_a_truncated_result(result, expected):
    assert is_truncated_result(result) is expected


@pytest.mark.parametrize(
    "point, origin",
    [
        pytest.param(
            {"id": "point-1", "job_id": "job-9", "tool_depth": 4, "message_step": 2},
            {"job_id": "job-9", "point_id": "point-1", "tool_depth": 4},
            id="with-job",
        ),
        pytest.param(
            {"id": "point-1", "tool_depth": 4},
            {"job_id": None, "point_id": "point-1", "tool_depth": 4},
            id="no-job",
        ),
    ],
)
def test_the_training_origin_of_a_branch_names_its_point(point, origin):
    assert branch_origin(point) == origin

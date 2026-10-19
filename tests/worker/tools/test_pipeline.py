"""The interceptor pipeline: the order of the interceptors, their verdicts, their fail modes and the executor.

The interceptors and the executor here are small classes that record what they see. No sandbox runs.
"""

from __future__ import annotations

import logging
from typing import Any

import pytest

from sprout.worker.tools.pipeline import (
    CallContext,
    Continue,
    Reject,
    Rewrite,
    ToolInterceptor,
    ToolPipeline,
    Verdict,
)
from sprout.worker.tools.result import ToolResult


class Tap(ToolInterceptor):
    """Records the tool and the args of each call that it sees, and lets the call pass."""

    def __init__(self, label: str, seen: list[tuple[Any, ...]], tools: set[str] | str = "*") -> None:
        self.label, self.seen, self.tools = label, seen, tools

    def before(self, ctx: CallContext) -> Verdict:
        self.seen.append((self.label, ctx.tool_name, dict(ctx.args)))
        return Continue()


class Refuse(ToolInterceptor):
    """Rejects every call with ``message``."""

    def __init__(self, message: str) -> None:
        self.message = message

    def before(self, ctx: CallContext) -> Verdict:
        return Reject(self.message)


class Prefix(ToolInterceptor):
    """Rewrites the command of each call to start with ``prefix``."""

    def __init__(self, prefix: str) -> None:
        self.prefix = prefix

    def before(self, ctx: CallContext) -> Verdict:
        return Rewrite({**ctx.args, "command": self.prefix + ctx.args["command"]})


class Crash(ToolInterceptor):
    """A ``before`` that raises, or that returns something that is not a verdict."""

    def __init__(self, fail_mode: str, answer: object) -> None:
        self.fail_mode, self.answer = fail_mode, answer

    def before(self, ctx: CallContext) -> Any:
        if isinstance(self.answer, Exception):
            raise self.answer
        return self.answer


def recording_executor(seen: list[tuple[Any, ...]]) -> Any:
    """An executor that records each call that reaches it and answers ``ran``."""

    def execute(tool: str, args: dict[str, Any]) -> ToolResult:
        seen.append(("executor", tool, dict(args)))
        return ToolResult(success=True, output="ran")

    return execute


def call(tool: str = "shell", **args: Any) -> CallContext:
    return CallContext(agent_id="agent", sandbox_id="sandbox", tool_name=tool, args=args)


def test_each_interceptor_sees_the_call_in_list_order_then_the_executor_runs_it():
    seen: list[tuple[Any, ...]] = []
    pipeline = ToolPipeline([Tap("first", seen), Tap("second", seen)])

    result = pipeline.execute(call(command="ls"), recording_executor(seen))

    assert result == ToolResult(success=True, output="ran")
    assert seen == [
        ("first", "shell", {"command": "ls"}),
        ("second", "shell", {"command": "ls"}),
        ("executor", "shell", {"command": "ls"}),
    ]


def test_a_pipeline_without_interceptors_runs_the_call_as_it_was_asked():
    """``ToolAccess`` builds this pipeline when a run gives none."""
    seen: list[tuple[Any, ...]] = []

    result = ToolPipeline().execute(call(command="ls"), recording_executor(seen))

    assert (result, seen) == (
        ToolResult(success=True, output="ran"),
        [("executor", "shell", {"command": "ls"})],
    )


def test_a_rejection_stops_the_chain_and_fails_the_call_with_its_message():
    """The interceptors after the rejecting one and the executor never see the call."""
    seen: list[tuple[Any, ...]] = []
    pipeline = ToolPipeline([Tap("before", seen), Refuse("budget exhausted"), Tap("after", seen)])

    result = pipeline.execute(call(command="ls"), recording_executor(seen))

    assert result == ToolResult(success=False, output="budget exhausted", error="rejected by Refuse")
    assert seen == [("before", "shell", {"command": "ls"})]


def test_a_rewrite_reaches_the_later_interceptors_and_the_executor_but_not_the_callers_context():
    seen: list[tuple[Any, ...]] = []
    asked = call(command="pytest")
    pipeline = ToolPipeline(
        [Tap("outer", seen), Prefix("timeout 300 "), Tap("middle", seen), Prefix("nice "), Tap("inner", seen)]
    )

    pipeline.execute(asked, recording_executor(seen))

    assert seen == [
        ("outer", "shell", {"command": "pytest"}),
        ("middle", "shell", {"command": "timeout 300 pytest"}),
        ("inner", "shell", {"command": "nice timeout 300 pytest"}),
        ("executor", "shell", {"command": "nice timeout 300 pytest"}),
    ]
    assert asked.args == {"command": "pytest"}


@pytest.mark.parametrize(
    "answer, message",
    [
        (RuntimeError("boom"), "interceptor Crash failed closed: boom"),
        (None, "interceptor Crash failed closed: before() returned None, expected a Verdict"),
        ("allow", "interceptor Crash failed closed: before() returned 'allow', expected a Verdict"),
    ],
    ids=["raises", "returns-none", "returns-a-string"],
)
def test_an_interceptor_that_fails_closed_rejects_the_call_when_it_crashes(answer, message):
    """Returning something that is not a verdict counts as a crash, ``None`` included."""
    seen: list[tuple[Any, ...]] = []

    result = ToolPipeline([Crash("closed", answer)]).execute(call(command="ls"), recording_executor(seen))

    assert result == ToolResult(success=False, output=message, error="rejected by Crash")
    assert seen == []


@pytest.mark.parametrize(
    "answer, reason",
    [(RuntimeError("boom"), "boom"), (None, "before() returned None, expected a Verdict")],
    ids=["raises", "returns-none"],
)
def test_an_interceptor_that_fails_open_logs_a_warning_and_lets_the_call_through_unchanged(
    caplog, answer, reason
):
    seen: list[tuple[Any, ...]] = []

    with caplog.at_level(logging.WARNING, logger="sprout.pipeline"):
        result = ToolPipeline([Crash("open", answer)]).execute(call(command="ls"), recording_executor(seen))

    assert result == ToolResult(success=True, output="ran")
    assert seen == [("executor", "shell", {"command": "ls"})]
    assert caplog.messages == ["interceptor Crash before() failed open (%s); continuing" % reason]


@pytest.mark.parametrize(
    "tool, expected",
    [
        ("shell", [("shell-only", "shell", {"command": "ls"}), ("executor", "shell", {"command": "ls"})]),
        ("grep_files", [("executor", "grep_files", {"command": "ls"})]),
    ],
)
def test_an_interceptor_sees_only_the_calls_of_its_tools(tool, expected):
    seen: list[tuple[Any, ...]] = []
    pipeline = ToolPipeline([Tap("shell-only", seen, tools={"shell"})])

    pipeline.execute(call(tool, command="ls"), recording_executor(seen))

    assert seen == expected


def test_an_executor_that_raises_gives_a_failed_result_instead_of_an_exception():
    def lost(tool: str, args: dict[str, Any]) -> ToolResult:
        raise ConnectionError("connection lost")

    result = ToolPipeline([]).execute(call(command="ls"), lost)

    assert result == ToolResult(success=False, output="", error="connection lost")

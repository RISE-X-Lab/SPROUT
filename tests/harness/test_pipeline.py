"""Unit tests for the tool interceptor pipeline (harness.tools.pipeline).

No sandbox, no model calls -- the interceptors are exercised against plain
functions.

Covered:
- befores run in list order; a Reject stops the chain and skips the executor
- Rewrite composition + context immutability
- per-interceptor fail-open vs fail-closed
- tools filtering, inner-exception conversion
"""

from __future__ import annotations

import pytest

from sprout.harness.tools.pipeline import (
    CallContext,
    Continue,
    Reject,
    Rewrite,
    ToolInterceptor,
    ToolPipeline,
)
from sprout.harness_core.result import ToolResult

def _ok(output: str = "ok") -> ToolResult:
    return ToolResult(success=True, output=output)


def _inner_recorder(log: list):
    def inner(tool: str, args: dict) -> ToolResult:
        log.append(("inner", tool, dict(args)))
        return _ok("ran")
    return inner


def _ctx(tool: str = "shell", args: dict | None = None, agent: str = "A") -> CallContext:
    return CallContext(agent_id=agent, sandbox_id="sb-1", tool_name=tool,
                       args=dict(args or {}))


class Tap(ToolInterceptor):
    """Records what it saw and passes everything through."""

    def __init__(self, label: str, log: list, tools="*"):
        self.label = label
        self.log = log
        self.tools = tools

    def before(self, ctx: CallContext):
        self.log.append((self.label, "before", ctx.tool_name, dict(ctx.args)))
        return Continue()


# --------------------------------------------------------------------------- #
#  Ordering and short-circuits
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("labels", [[], ["a", "b"]])    # [] is ToolAccess's default
def test_befores_run_in_list_order_then_the_executor(labels):
    log = []
    result = ToolPipeline([Tap(label, log) for label in labels]) \
        .execute(_ctx("shell", {"command": "ls"}), _inner_recorder(log))
    assert result.success and result.output == "ran"
    assert log == [(label, "before", "shell", {"command": "ls"}) for label in labels] \
        + [("inner", "shell", {"command": "ls"})]


def test_a_reject_stops_the_chain_where_it_happens():
    """Interceptors before the rejecter saw the call; those after it never do.
    The message is the failed result's output; the error names the rejecter."""
    log = []

    class Rejecter(ToolInterceptor):
        def before(self, ctx):
            log.append(("rejecter", "before"))
            return Reject("nope")

    result = ToolPipeline([Tap("outer", log), Rejecter(), Tap("inner", log)]) \
        .execute(_ctx(), _inner_recorder(log))
    assert not result.success
    assert result.output == "nope"
    assert result.error == "rejected by Rejecter"
    assert [e[0] for e in log] == ["outer", "rejecter"]    # inner never ran


# --------------------------------------------------------------------------- #
#  Rewrite
# --------------------------------------------------------------------------- #

class AddKey(ToolInterceptor):
    def __init__(self, key: str, value):
        self.key, self.value = key, value

    def before(self, ctx):
        return Rewrite({**ctx.args, self.key: self.value})


def test_rewrites_compose_deeper_interceptors_see_them_and_the_caller_keeps_its_context():
    """Each rewrite builds on the one before it and everything deeper in sees
    it. The caller's context still says what the agent asked for: ToolAccess
    journals that, and the executor gets the rewritten command."""
    log = []
    original = _ctx("shell", {"command": "pytest -x"})

    class AddTimeout(ToolInterceptor):
        def before(self, ctx):
            return Rewrite({**ctx.args, "command": "timeout 300 " + ctx.args["command"]})

    result = ToolPipeline([Tap("outer", log), AddKey("a", 1), Tap("between", log),
                           AddTimeout(), Tap("deeper", log)]) \
        .execute(original, _inner_recorder(log))
    assert result.success
    assert log == [
        ("outer", "before", "shell", {"command": "pytest -x"}),
        ("between", "before", "shell", {"command": "pytest -x", "a": 1}),
        ("deeper", "before", "shell", {"command": "timeout 300 pytest -x", "a": 1}),
        ("inner", "shell", {"command": "timeout 300 pytest -x", "a": 1}),
    ]
    assert original.args == {"command": "pytest -x"}       # contexts are immutable


# --------------------------------------------------------------------------- #
#  Fail-open vs fail-closed
# --------------------------------------------------------------------------- #

class _BeforeBoom(ToolInterceptor):
    def __init__(self, fail_mode: str):
        self.fail_mode = fail_mode

    def before(self, ctx):
        raise RuntimeError("boom")


class _BeforeJunk(ToolInterceptor):
    """Returns something that is not a Verdict: a fault, not a Continue."""

    def __init__(self, fail_mode: str):
        self.fail_mode = fail_mode

    def before(self, ctx):
        return "not a verdict"


def test_fail_open_before_logs_and_continues(caplog):
    log = []
    with caplog.at_level("WARNING", logger="sprout.pipeline"):
        result = ToolPipeline([_BeforeBoom("open")]).execute(_ctx(), _inner_recorder(log))
    assert result.success and log                          # call went through
    assert any("failed open" in r.message for r in caplog.records)


@pytest.mark.parametrize("fault,fail_mode", [
    pytest.param(_BeforeBoom, "closed", id="raises-closed"),
    pytest.param(_BeforeJunk, "closed", id="junk-closed"),
    pytest.param(_BeforeJunk, "open", id="junk-open"),    # raises-open: the test above
])
def test_a_crashing_or_junk_before_honours_its_fail_mode(fault, fail_mode, caplog):
    """Raising and returning a non-Verdict are the same fault; fail_mode decides."""
    log = []
    with caplog.at_level("WARNING", logger="sprout.pipeline"):
        result = ToolPipeline([fault(fail_mode)]).execute(_ctx(), _inner_recorder(log))
    if fail_mode == "closed":
        assert not result.success
        assert result.error == f"rejected by {fault.__name__}"
        assert "failed closed" in result.output
        assert log == []                                   # inner never ran
    else:
        assert result.success and log                      # call went through
        assert any("failed open" in r.message for r in caplog.records)


# --------------------------------------------------------------------------- #
#  Tools filtering + inner failures
# --------------------------------------------------------------------------- #

def test_tools_filter_skips_non_matching_calls():
    log = []
    pipe = ToolPipeline([Tap("shell_only", log, tools={"shell"})])
    pipe.execute(_ctx("other", {"arg": 1}), _inner_recorder(log))
    assert [e[0] for e in log] == ["inner"]                # the hook did not run
    pipe.execute(_ctx("shell", {"command": "ls"}), _inner_recorder(log))
    assert ("shell_only", "before", "shell", {"command": "ls"}) in log


def test_inner_exception_becomes_failed_result():
    def broken(tool, args):
        raise RuntimeError("connection lost")

    result = ToolPipeline([Tap("audit", [])]).execute(_ctx(), broken)
    assert not result.success
    assert result.error == "connection lost"

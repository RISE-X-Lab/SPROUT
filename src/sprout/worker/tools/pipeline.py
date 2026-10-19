"""The tool-call interceptor pipeline: an ordered chain of ``ToolInterceptor`` in front of one executor.

The run's rules for tool calls, such as rollout limits, checkpoint tracking and a task's shell
context, are interceptors. For each call, the ``before`` of each interceptor that applies runs in
list order, and returns a verdict:

- ``Continue`` lets the call pass unchanged.
- ``Rewrite(new_args)`` replaces the args that the later interceptors and the executor see. The
  ``CallContext`` that the caller gave does not change, so it still says what the agent asked for.
- ``Reject(message)`` stops the chain. The later interceptors and the executor do not see the call,
  and it fails with ``message``.

An interceptor with ``fail_mode="closed"`` that raises rejects the call. One with
``fail_mode="open"`` logs a warning and lets the call continue unchanged. An executor
exception gives a failed ``ToolResult``. The pipeline never raises. It holds no state other than
its interceptors, so an interceptor keeps its own state and its own locks.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Any, Literal

from sprout.worker.tools.result import ToolResult

logger = logging.getLogger("sprout.pipeline")  # unconfigured -> WARNING+ to stderr

#: The inner executor: runs the tool named by its first argument with the args of the second.
Executor = Callable[[str, dict[str, Any]], ToolResult]


# --- call context and verdicts ------------------------------------------------


@dataclass(frozen=True)
class CallContext:
    """One tool call travelling through the pipeline."""

    agent_id: str
    sandbox_id: str
    tool_name: str
    args: dict[str, Any]


@dataclass(frozen=True)
class Continue:
    """Let the call proceed unchanged.

    An interceptor that only observes calls returns it. ``None`` is not a verdict, and counts as a crash.
    """


@dataclass(frozen=True)
class Reject:
    """Refuse the call. The call fails, with ``message`` as the result's output.

    ``success`` is False, because the model must see that its call did not happen.
    """

    message: str


@dataclass(frozen=True)
class Rewrite:
    """Let the call proceed with ``new_args`` instead of ``ctx.args``.

    An interceptor returns a ``Rewrite`` and does not edit ``ctx.args`` in place, although that dict is
    writable. The interceptors before it, and the caller, still see what the agent asked for."""

    new_args: dict[str, Any]


#: What an interceptor decides about a call: let it through, change it, or refuse it.
type Verdict = Continue | Reject | Rewrite
_VERDICT_TYPES = (Continue, Reject, Rewrite)


# --- interceptor base class ---------------------------------------------------


class ToolInterceptor:
    """One rule on the tool-call path. A subclass overrides ``before``, and can set the class attributes.

    ``before`` must return a ``Verdict``. Another return value counts as a crash. The pipeline does not
    serialize calls, so an interceptor that keeps state guards that state itself.
    """

    #: The tools whose calls this interceptor sees, or ``"*"`` for every tool.
    tools: set[str] | Literal["*"] = "*"
    #: What a crash in ``before`` does: ``"closed"`` rejects the call, ``"open"`` lets it continue.
    #: A safety rule fails closed. An observer fails open.
    fail_mode: Literal["open", "closed"] = "open"

    @property
    def name(self) -> str:
        """The interceptor's name in log lines and rejections: its class name."""
        return type(self).__name__

    def applies_to(self, tool_name: str) -> bool:
        """Whether this interceptor sees the calls of ``tool_name``."""
        return self.tools == "*" or tool_name in self.tools

    def before(self, ctx: CallContext) -> Verdict:
        """The verdict on ``ctx``, before the call reaches the executor. The default is ``Continue``."""
        return Continue()


# --- pipeline -----------------------------------------------------------------


class ToolPipeline:
    """An ordered chain of interceptors in front of one executor.

    The order matters: an earlier interceptor decides what a later one sees.
    """

    def __init__(self, interceptors: list[ToolInterceptor] | None = None) -> None:
        self._interceptors: tuple[ToolInterceptor, ...] = tuple(interceptors or ())

    @property
    def interceptors(self) -> tuple[ToolInterceptor, ...]:
        """The interceptors, in order."""
        return self._interceptors

    def execute(self, ctx: CallContext, inner: Executor) -> ToolResult:
        """Run ``ctx`` through the interceptors that apply to its tool, then ``inner``, and return the result.

        Never raises. A rejection, or a failure of ``inner``, gives a failed ``ToolResult``.
        """
        current = ctx
        for interceptor in self._interceptors:
            if not interceptor.applies_to(current.tool_name):
                continue
            verdict = self._safe_before(interceptor, current)
            if isinstance(verdict, Reject):
                return _rejection(interceptor, verdict.message)
            if isinstance(verdict, Rewrite):
                current = replace(current, args=dict(verdict.new_args))
        return self._run_inner(current, inner)

    # --- guarded hook invocation ---

    @staticmethod
    def _safe_before(interceptor: ToolInterceptor, ctx: CallContext) -> Verdict:
        """The verdict of ``interceptor.before(ctx)``, or for a crash the verdict of its ``fail_mode``."""
        try:
            verdict = interceptor.before(ctx)
            if isinstance(verdict, _VERDICT_TYPES):
                return verdict
            raise TypeError(f"before() returned {verdict!r}, expected a Verdict")
        except Exception as exc:
            if interceptor.fail_mode == "closed":
                return Reject(f"interceptor {interceptor.name} failed closed: {exc}")
            logger.warning(
                "interceptor %s before() failed open (%s); continuing", interceptor.name, exc, exc_info=True
            )
            return Continue()

    @staticmethod
    def _run_inner(ctx: CallContext, inner: Executor) -> ToolResult:
        """The result of ``inner``, or a failed result for its exception.

        The agent then sees a failure, where a crash would end the run."""
        try:
            return inner(ctx.tool_name, ctx.args)
        except Exception as exc:
            logger.warning("inner executor failed for %s: %s", ctx.tool_name, exc, exc_info=True)
            return ToolResult(success=False, output="", error=str(exc))


def _rejection(interceptor: ToolInterceptor, message: str) -> ToolResult:
    return ToolResult(success=False, output=message, error=f"rejected by {interceptor.name}")

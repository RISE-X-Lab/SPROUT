"""The tool-call interceptor pipeline.

Governance of the tool-call path (checkpoint tracking, rollout limits, the
Harbor execution context) is expressed as an ordered chain of
``ToolInterceptor``s in front of one inner executor::

    executor: (tool_name: str, args: dict) -> ToolResult

Semantics
---------
Each interceptor's ``before`` runs in list order and returns a verdict:

- ``Continue`` lets the call pass unchanged.
- ``Rewrite(new_args)`` replaces the args seen by deeper interceptors and by
  the inner executor. Contexts are immutable: each rewrite derives a new
  ``CallContext``, so the context the caller handed in still says what the
  agent asked for.
- ``Reject(message)`` stops the chain: deeper interceptors never see the call,
  the executor does not run, and the message becomes a failed ``ToolResult``.

Fail-safety is per interceptor. ``fail_mode="closed"`` turns an interceptor
crash into a rejection (safety interceptors); ``fail_mode="open"`` logs to
stderr and passes the call through unchanged (observability interceptors). An
inner-executor exception becomes a failed ``ToolResult``. Every failure
surfaces as a result: the pipeline never raises.

The framework is stateless: the pipeline holds only its interceptor list;
per-call state lives in locals, durable state (locks, counters) inside
interceptors -- the pipeline itself is never a concurrency bottleneck.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, replace
from typing import Callable, Literal, Union

from sprout.harness.core.result import ToolResult

logger = logging.getLogger("sprout.pipeline")  # unconfigured -> WARNING+ to stderr

Executor = Callable[[str, dict], ToolResult]


# --------------------------------------------------------------------------- #
#  Call context and verdicts
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class CallContext:
    """One tool call travelling through the pipeline."""
    agent_id: str
    sandbox_id: str
    tool_name: str
    args: dict


#  The verdicts are the things an interceptor can do about a call: let it
#  through, change it, or refuse it.


@dataclass(frozen=True)
class Continue:
    """Let the call proceed unchanged.

    An observer needs a way to say "I looked and I am not interfering", distinct
    from having no opinion. Audit and metering interceptors return only this.
    """


@dataclass(frozen=True)
class Reject:
    """Refuse the call; ``message`` becomes the failed result's output.

    Forces ``success=False``, which is the point: the model must see that its
    call did not happen.
    """
    message: str


@dataclass(frozen=True)
class Rewrite:
    """Let the call proceed with ``new_args`` instead of ``ctx.args``.

    A verdict rather than an in-place edit of ``ctx.args`` (``CallContext`` is
    frozen, but that only binds the fields; the dict inside is writable): the
    pipeline derives a fresh context for everything deeper in, and the context
    the caller handed in is never changed under it. Adding a working directory
    to a shell command is invisible to the interceptors above -- they still see
    what the agent asked for.
    """
    new_args: dict


Verdict = Union[Continue, Reject, Rewrite]
_VERDICT_TYPES = (Continue, Reject, Rewrite)


# --------------------------------------------------------------------------- #
#  Interceptor base class
# --------------------------------------------------------------------------- #

class ToolInterceptor:
    """One interceptor on the tool-call path. Subclass and override ``before``."""

    tools: "set[str] | Literal['*']" = "*"  # which tools this interceptor sees
    fail_mode: Literal["open", "closed"] = "open"

    @property
    def name(self) -> str:
        return type(self).__name__

    def applies_to(self, tool_name: str) -> bool:
        return self.tools == "*" or tool_name in self.tools

    def before(self, ctx: CallContext) -> Verdict:
        """Runs before the call reaches the runtime. Returns a ``Verdict``."""
        return Continue()


# --------------------------------------------------------------------------- #
#  Pipeline
# --------------------------------------------------------------------------- #

class ToolPipeline:
    """Ordered interceptor chain in front of one executor. Order is semantics."""

    def __init__(self, interceptors: "list[ToolInterceptor] | None" = None) -> None:
        self._interceptors: tuple[ToolInterceptor, ...] = tuple(interceptors or ())

    @property
    def interceptors(self) -> tuple[ToolInterceptor, ...]:
        return self._interceptors

    def execute(self, ctx: CallContext, inner: Executor) -> ToolResult:
        """Run one call through the chain: befores in order, then the executor.
        Never raises -- every failure surfaces as a ``ToolResult``."""
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

    # -- guarded hook invocation --------------------------------------------- #

    @staticmethod
    def _safe_before(interceptor: ToolInterceptor, ctx: CallContext) -> Verdict:
        """Run ``before``; a crash rejects (fail-closed) or logs and lets the
        call continue (fail-open)."""
        try:
            verdict = interceptor.before(ctx)
            if isinstance(verdict, _VERDICT_TYPES):
                return verdict
            raise TypeError(f"before() returned {verdict!r}, expected a Verdict")
        except Exception as exc:
            if interceptor.fail_mode == "closed":
                return Reject(f"interceptor {interceptor.name} failed closed: {exc}")
            logger.warning("interceptor %s before() failed open (%s); continuing",
                           interceptor.name, exc, exc_info=True)
            return Continue()

    @staticmethod
    def _run_inner(ctx: CallContext, inner: Executor) -> ToolResult:
        """Execute the tool; an exception becomes a failed result, so the
        agent sees a failure where a crash would have ended the run."""
        try:
            return inner(ctx.tool_name, ctx.args)
        except Exception as exc:
            logger.warning("inner executor failed for %s: %s",
                           ctx.tool_name, exc, exc_info=True)
            return ToolResult(success=False, output="", error=str(exc))


def _rejection(interceptor: ToolInterceptor, message: str) -> ToolResult:
    return ToolResult(success=False, output=message,
                      error=f"rejected by {interceptor.name}")

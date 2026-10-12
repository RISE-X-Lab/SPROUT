"""Which tool calls could have changed the environment.

The interceptor half of checkpointing: it answers "did anything since the last
snapshot possibly mutate", which the :class:`Checkpointer` reads to decide
whether a turn needs a capture of its own. *When* to snapshot is a separate
decision and lives in ``harness/checkpoint/checkpointer.py``.

The runtime serves one tool, ``shell``, and a command's effect cannot be read
off its text (``ls`` reads, ``ls > out`` writes), so every executed call counts
as a mutation. A wrong "clean" verdict would silently lose a step's state; a
wrong "dirty" one costs a cheap disk capture.

Mounted outermost, so it also sees calls the inner interceptors reject: a
rejected call changes nothing, but over-counting is cheaper than reasoning about
which rejections are total.
"""

from __future__ import annotations

from sprout.harness.tools.pipeline import CallContext, Continue, ToolInterceptor, Verdict

__all__ = ["MutationTracker"]


class MutationTracker(ToolInterceptor):
    """Flags whether any call since the last checkpoint could have mutated."""

    fail_mode = "open"

    def __init__(self) -> None:
        self._dirty = False

    @property
    def dirty(self) -> bool:
        return self._dirty

    def clear(self) -> None:
        self._dirty = False

    def before(self, ctx: CallContext) -> Verdict:
        self._dirty = True
        return Continue()

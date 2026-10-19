"""Whether the tool calls since the last snapshot could have changed the environment.

``MutationTracker`` is the interceptor half of checkpointing. A step whose calls could not have
changed the environment reuses the previous snapshot. ``sprout.worker.checkpoint.checkpointer``
decides when to take a snapshot.

Every ``shell`` call counts, because the effect of a command is not in its text: ``ls`` reads, and
``ls > out`` writes. A search, a fetch or a ``text_editor view`` does not count. A wrong "clean"
loses the state of a step, and a wrong "dirty" costs one cheap capture. The tracker marks each call
that reaches it, so a call that a deeper interceptor rejects still counts.
"""

from __future__ import annotations

from typing import Any

from sprout.worker.tools.pipeline import CallContext, Continue, ToolInterceptor, Verdict

__all__ = ["MutationTracker", "call_mutates", "READ_ONLY_TOOLS"]


#: The tools that cannot change the environment, because they only read, search or fetch.
#: Every tool that is not listed counts as a mutation.
READ_ONLY_TOOLS = frozenset({"grep_files", "web_fetch", "web_search"})

#: ``text_editor`` sub-commands that only read.
READ_ONLY_EDITOR_COMMANDS = frozenset({"view"})


def call_mutates(tool_name: str, args: dict[str, Any]) -> bool:
    """Whether a call of ``tool_name`` with ``args`` could have changed the environment."""
    if tool_name in READ_ONLY_TOOLS:
        return False
    if tool_name == "text_editor":
        return str(args.get("command", "")) not in READ_ONLY_EDITOR_COMMANDS
    return True


class MutationTracker(ToolInterceptor):
    """Records whether any call since the last ``clear`` could have changed the environment.

    It never rejects or rewrites a call.
    """

    fail_mode = "open"

    def __init__(self) -> None:
        self._dirty = False

    @property
    def dirty(self) -> bool:
        """Whether a call since the last ``clear`` could have changed the environment."""
        return self._dirty

    def clear(self) -> None:
        """Mark the environment as unchanged, for example after a snapshot of it."""
        self._dirty = False

    def before(self, ctx: CallContext) -> Verdict:
        if call_mutates(ctx.tool_name, ctx.args):
            self._dirty = True
        return Continue()

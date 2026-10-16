"""Which tool calls could have changed the environment.

The interceptor half of checkpointing: a step whose calls could not have mutated
reuses the previous snapshot (*when* to snapshot is ``checkpoint/checkpointer.py``).
``shell`` always counts, since a command's effect is not in its text (``ls`` reads,
``ls > out`` writes); a search, a fetch or a ``text_editor view`` does not. A wrong
"clean" silently loses a step's state, a wrong "dirty" costs a cheap capture. It is
mounted outermost, so a call an inner interceptor rejects still counts.
"""

from __future__ import annotations

from sprout.harness_tools.pipeline import CallContext, Continue, ToolInterceptor, Verdict

__all__ = ["MutationTracker", "call_mutates", "READ_ONLY_TOOLS"]


#: Tools that cannot change the environment: they read, search, or fetch.
#: Anything not listed is assumed to mutate.
READ_ONLY_TOOLS = frozenset({"grep_files", "web_fetch", "web_search"})

#: ``text_editor`` sub-commands that only read.
READ_ONLY_EDITOR_COMMANDS = frozenset({"view"})


def call_mutates(tool_name: str, args: dict) -> bool:
    """Whether a tool call could have changed the environment."""
    if tool_name in READ_ONLY_TOOLS:
        return False
    if tool_name == "text_editor":
        return str(args.get("command", "")) not in READ_ONLY_EDITOR_COMMANDS
    return True


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
        if call_mutates(ctx.tool_name, ctx.args):
            self._dirty = True
        return Continue()

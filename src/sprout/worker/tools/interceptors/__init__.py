"""The interceptors that ship with Sprout, one module each, and the registry of the declarable ones.

- ``mutation``: ``MutationTracker`` records whether a call could have changed the environment.
- ``shell_context``: ``ShellContext`` runs each shell call in the task's working directory, with the
  task's environment and as the task's user.

A run declares its interceptors as data, in ``RunSpec.interceptors``: ``[{"name": ..., **kwargs}]``.
So a benchmark or a queue chooses the rules, and no code names a class. ``INTERCEPTORS`` holds the
declarable interceptors by name. It does not hold the mutation tracker, because checkpointing needs
one on every run.
"""

from collections.abc import Callable, Iterable

from sprout.worker.tools.pipeline import ToolInterceptor

from .mutation import MutationTracker
from .shell_context import ShellContext, shell_command

INTERCEPTORS: dict[str, Callable[..., ToolInterceptor]] = {"shell_context": ShellContext}


def build_interceptors(specs: Iterable[object] | None) -> list[ToolInterceptor]:
    """The interceptors that ``specs`` declares, built in order. ``None`` declares none.

    Raises ``ValueError`` if a declaration is not ``{"name": ..., **kwargs}``, or if its name is not
    in ``INTERCEPTORS``. The interceptor's constructor raises for keyword arguments that it refuses.
    """
    built = []
    for spec in specs or ():
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise ValueError('an interceptor is declared as {"name": ..., **kwargs}')
        kwargs = {key: value for key, value in spec.items() if key != "name"}
        try:
            factory = INTERCEPTORS[spec["name"]]
        except KeyError:
            raise ValueError(
                "unknown interceptor %r; known: %s" % (spec["name"], ", ".join(sorted(INTERCEPTORS)))
            ) from None
        built.append(factory(**kwargs))
    return built


__all__ = ["MutationTracker", "ShellContext", "shell_command", "INTERCEPTORS", "build_interceptors"]

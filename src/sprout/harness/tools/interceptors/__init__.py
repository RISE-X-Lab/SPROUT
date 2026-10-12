"""The interceptors that ship with Sprout, one module each.

    mutation.py       did anything since the last snapshot possibly change the env
    shell_context.py  run each shell call under the task's cwd, env and user

The mutation tracker is part of checkpointing and is always mounted. The
shell context is declared by a run (``RunSpec.interceptors``:
``[{"name": ..., **kwargs}]``) and built here by name, so a benchmark or a
queue says which rules apply without any code naming a class.
"""

from .mutation import MutationTracker
from .shell_context import ShellContext, shell_command

INTERCEPTORS = {"shell_context": ShellContext}


def build_interceptors(specs) -> list:
    """Instantiate declared interceptors, in order. An unknown name is an error."""
    built = []
    for spec in specs or ():
        if not isinstance(spec, dict) or not isinstance(spec.get("name"), str):
            raise ValueError("an interceptor is declared as {\"name\": ..., **kwargs}")
        kwargs = {key: value for key, value in spec.items() if key != "name"}
        try:
            factory = INTERCEPTORS[spec["name"]]
        except KeyError:
            raise ValueError("unknown interceptor %r; known: %s"
                             % (spec["name"], ", ".join(sorted(INTERCEPTORS)))) from None
        built.append(factory(**kwargs))
    return built


__all__ = ["MutationTracker", "ShellContext", "shell_command",
           "INTERCEPTORS", "build_interceptors"]

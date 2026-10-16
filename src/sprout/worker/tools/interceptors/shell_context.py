"""Run every shell call under the task's working directory, environment and user.

A Harbor task declares how commands run (``workdir``, ``env``, ``user``, and the
image's own ``WorkingDir``/``Env``/``User``); the agent's tool only supplies a
command. Rewriting the call here keeps that policy on the pipeline that serves
the calls, where the mutation tracker and the checkpoint boundary already are.
"""

from __future__ import annotations

import shlex

from sprout.worker.tools.pipeline import Rewrite, ToolInterceptor


def shell_command(command: str, cwd: str, env: dict[str, str], user: str | int | None) -> str:
    """``command`` as one shell line: ``cd`` + ``env`` + bash, optionally as ``user``.

    Arguments are quoted, never interpolated, so an environment value cannot
    become a command.
    """
    inner = f"cd {shlex.quote(cwd)} && exec /bin/bash -c {shlex.quote(command)}"
    invocation = ["env", *[f"{key}={value}" for key, value in env.items()], "/bin/bash", "-c", inner]
    wrapped = shlex.join(invocation)
    if user not in (None, "", "root", "0", 0):
        account, separator, group = str(user).partition(":")
        username = f'"$(getent passwd {shlex.quote(account)} | cut -d: -f1)"'
        prefix = f"runuser -u {username}"
        if separator:
            prefix += f' -g "$(getent group {shlex.quote(group)} | cut -d: -f1)"'
        wrapped = prefix + " -- " + wrapped
    return wrapped


class ShellContext(ToolInterceptor):
    name = "shell_context"
    tools = {"shell"}
    fail_mode = "closed"

    def __init__(self, cwd: str, env: dict[str, str] | None = None, user: str | int | None = None):
        if not isinstance(cwd, str) or not cwd:
            raise ValueError("shell_context needs the task working directory")
        self.cwd = cwd
        self.env = dict(env or {})
        self.user = user

    def before(self, context):
        arguments = dict(context.args)
        arguments["command"] = shell_command(
            arguments["command"], arguments.get("working_dir") or self.cwd, self.env, self.user
        )
        arguments["working_dir"] = "/"
        return Rewrite(arguments)

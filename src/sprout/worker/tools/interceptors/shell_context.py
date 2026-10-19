"""Run every shell call in the task's working directory, with its environment and as its user.

A Harbor task declares how its commands run: ``workdir``, ``env``, ``user``, and the image's own
``WorkingDir``, ``Env`` and ``User``. The agent's tool supplies only a command. ``ShellContext``
rewrites the call, so the policy stays on the pipeline that serves the calls, with the mutation
tracker and the checkpoint boundary.
"""

from __future__ import annotations

import shlex

from sprout.worker.tools.pipeline import CallContext, Rewrite, ToolInterceptor


def shell_command(command: str, cwd: str, env: dict[str, str], user: str | int | None) -> str:
    """One shell line that runs ``command`` with bash in ``cwd``, with ``env``, and as ``user`` if given.

    The line quotes every argument, so an environment value cannot become a command. ``user`` is a
    name or a uid, with an optional ``:group``. ``None``, ``""``, ``root`` and ``0`` run as the shell's
    own user."""
    inner = f"cd {shlex.quote(cwd)} && exec /bin/bash -c {shlex.quote(command)}"
    invocation = ["env", *[f"{key}={value}" for key, value in env.items()], "/bin/bash", "-c", inner]
    wrapped = shlex.join(invocation)
    if user not in (None, "", "root", "0", 0):
        account, separator, group = str(user).partition(":")
        # runuser looks a user up by name only, so getent turns a uid (and a gid below) into its name.
        username = f'"$(getent passwd {shlex.quote(account)} | cut -d: -f1)"'
        prefix = f"runuser -u {username}"
        if separator:
            prefix += f' -g "$(getent group {shlex.quote(group)} | cut -d: -f1)"'
        wrapped = prefix + " -- " + wrapped
    return wrapped


class ShellContext(ToolInterceptor):
    """Rewrites each ``shell`` call to run in ``cwd``, with ``env`` and as ``user``.

    The call's own ``working_dir`` wins over ``cwd``. Raises ``ValueError`` if ``cwd`` is empty.
    A call that the interceptor cannot rewrite is rejected (``fail_mode`` is ``closed``).
    """

    name = "shell_context"
    tools = {"shell"}
    fail_mode = "closed"

    def __init__(self, cwd: str, env: dict[str, str] | None = None, user: str | int | None = None) -> None:
        if not isinstance(cwd, str) or not cwd:
            raise ValueError("shell_context needs the task working directory")
        self.cwd = cwd
        self.env = dict(env or {})
        self.user = user

    def before(self, context: CallContext) -> Rewrite:
        arguments = dict(context.args)
        arguments["command"] = shell_command(
            arguments["command"], arguments.get("working_dir") or self.cwd, self.env, self.user
        )
        arguments["working_dir"] = "/"
        return Rewrite(arguments)

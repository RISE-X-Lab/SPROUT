"""Running the agent's commands, ``solve.sh`` and ``test.sh`` as the image's ``USER``, as Harbor runs them.

The runtime runs every command as root. A backend section that turns ``image_user`` on runs
these three kinds of command as the image's ``USER`` instead, or as root when the image names
none. With ``image_env`` as well, each of them gets the image's ``ENV`` again after the switch
of user. Every other command, such as a grade's preparation of ``/logs``, still runs as root.
``docs/RUNSTORE.md`` describes both options.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sprout.contract.backend import backend_name
from sprout.worker.tools.interceptors import shell_command

if TYPE_CHECKING:
    from sprout.worker.sandbox.session import SandboxSession


@dataclass(frozen=True)
class ImageShell:
    """The user that the image's commands run as, the ENV that they get and where they run."""

    #: The image's ``USER``, else ``root``.
    user: str
    #: The image's ENV, set on each command. Empty when ``image_env`` is off.
    env: dict[str, str]
    #: The image's ``WorkingDir``, else ``/``: where a command runs when its caller names no directory.
    workdir: str

    def command(self, command: str, *, working_dir: str | None) -> str:
        """``command`` as one shell line: as ``user``, with ``env``, in ``working_dir`` or ``workdir``."""
        return shell_command(command, working_dir or self.workdir, self.env, self.user)


def image_shell(session: SandboxSession, *, backend: dict[str, Any], image: str | None) -> ImageShell | None:
    """The shell of the image of ``session``'s sandbox. None when ``backend`` leaves ``image_user`` off.

    ``image`` is the OCI reference of that image, which a microVM needs to read the image's config
    (``SandboxSession.image_config``). Raises ``RuntimeError`` or ``TemplateError`` when it cannot be read.
    """
    section = backend.get(backend_name(backend)) or {}
    if not section.get("image_user"):
        return None
    config = session.image_config(image)
    env = dict(entry.split("=", 1) for entry in config.get("Env") or [] if "=" in entry)
    return ImageShell(
        user=config.get("User") or "root",
        env=env if section.get("image_env") else {},
        workdir=config.get("WorkingDir") or "/",
    )

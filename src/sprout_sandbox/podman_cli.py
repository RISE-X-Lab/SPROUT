"""The podman executable that every entry point runs, and one podman or btrfs command run with a timeout.

A command that fails raises ``PodmanError``, with a message that names the command.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

#: The environment variable that names the podman to run when the caller gives no ``podman_bin``.
#: Its value is a path (``~`` is expanded) or a bare name that PATH resolves, such as ``podman``.
PODMAN_BIN_ENV = "SPROUT_PODMAN_BIN"
#: Where ``ops/setup_podman.sh`` installs podman 6, relative to the home directory: a
#: static build, off PATH so a distribution podman is left alone.
INSTALLED_PODMAN = (".local", "opt", "podman", "bin", "podman")


def _installed_podman() -> Path:
    return Path.home().joinpath(*INSTALLED_PODMAN)


def default_podman_bin() -> str:
    """The podman that every entry point runs when the caller names none.

    It is ``$SPROUT_PODMAN_BIN`` if that is not empty, with ``~`` expanded. Else it is the podman that
    ``ops/setup_podman.sh`` installed, if that is executable. Else it is ``podman`` on PATH.
    """
    chosen = os.environ.get(PODMAN_BIN_ENV)
    if chosen:
        return os.path.expanduser(chosen)
    installed = _installed_podman()
    if installed.is_file() and os.access(installed, os.X_OK):
        return str(installed)
    return "podman"


class PodmanError(RuntimeError):
    """An operation of the podman sandbox backend failed. The message says what failed.

    A failed podman or btrfs command gives its command words, its exit code and its stderr.
    """


async def run_command(argv: list[str], *, timeout: float, cwd: str | Path | None = None) -> str:
    """Run ``argv`` in ``cwd`` and return its stdout. Raise ``PodmanError`` if it fails or times out.

    A command that times out is killed. ``cwd=None`` keeps the caller's directory. For podman, avoid a
    checkout: conmon inherits the directory, and writes an ``oom`` file there on a memory-cgroup kill.
    """
    process = await asyncio.create_subprocess_exec(
        *argv,
        stdin=asyncio.subprocess.DEVNULL,
        cwd=None if cwd is None else str(cwd),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        stdout, stderr = await asyncio.wait_for(process.communicate(), timeout)
    except TimeoutError:
        process.kill()
        await process.wait()
        raise PodmanError("%s did not finish within %.0f s" % (_words(argv), timeout)) from None
    if process.returncode != 0:
        raise PodmanError(
            "%s failed (exit %s): %s"
            % (_words(argv), process.returncode, stderr.decode(errors="replace").strip() or "(no stderr)")
        )
    return stdout.decode(errors="replace")


async def remove_container(podman_bin: str, container: str, *, timeout: float, cwd: str | Path) -> None:
    """Remove ``container`` with ``podman rm -f``. A container that does not exist is not an error."""
    try:
        await run_command([podman_bin, "rm", "-f", container], timeout=timeout, cwd=cwd)
    except PodmanError as error:
        text = str(error).lower()
        if "no such container" in text or "no container with" in text:
            return
        raise


def _words(argv: list[str]) -> str:
    """The tool name and up to two command words of ``argv``, such as ``podman image inspect``."""
    words = [Path(argv[0]).name]
    for arg in argv[1:3]:
        if arg.startswith("-"):
            break
        words.append(arg)
    return " ".join(words)

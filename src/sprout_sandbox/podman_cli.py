"""Which podman executable every entry point runs, and how one podman or btrfs command runs:
with a timeout, never from the caller's directory, failing as a ``PodmanError`` that names it.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

#: Names the podman executable every entry point runs when no ``podman_bin`` is
#: given; ``python -m sprout.services env`` exports it for an experiment shell. The
#: bare name ``podman`` means the one on PATH (what the fake-tool tests set).
PODMAN_BIN_ENV = "SPROUT_PODMAN_BIN"
#: Where ``ops/setup_podman.sh`` installs podman 6, relative to the home directory: a
#: static build, off PATH so a distribution podman is left alone.
#: ``sprout.services.spec.installed_podman`` names the same path (a test keeps them in step).
INSTALLED_PODMAN = (".local", "opt", "podman", "bin", "podman")


def installed_podman() -> Path:
    return Path.home().joinpath(*INSTALLED_PODMAN)


def default_podman_bin() -> str:
    """``$SPROUT_PODMAN_BIN`` when set (``~`` expanded), else the podman
    ``ops/setup_podman.sh`` installed when it is executable, else ``podman`` on PATH.

    One rule for every entry point (the pool, ``image_info``, the reaper, the Run
    Store worker and its children, the live tests), so two podmans never share a
    store by accident.
    """
    chosen = os.environ.get(PODMAN_BIN_ENV)
    if chosen:
        return os.path.expanduser(chosen)
    installed = installed_podman()
    if installed.is_file() and os.access(installed, os.X_OK):
        return str(installed)
    return "podman"


class PodmanError(RuntimeError):
    """A podman or btrfs command failed; the message carries its stderr."""


async def run_command(argv: list[str], *, timeout: float, cwd: str | Path | None = None) -> str:
    """Run one command; its stdout, or ``PodmanError`` with its stderr.

    ``cwd`` is where the command runs. It matters for podman: conmon, the
    monitor podman leaves behind each container, inherits it and writes an
    empty file named ``oom`` THERE when any process in the container is killed
    by the memory cgroup -- a marker that once landed in a run's source
    checkout and failed every later trial's clean-tree check. The pool
    therefore never runs podman from the caller's directory.
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
        raise PodmanError("%s did not finish within %.0f s" % (_words(argv), timeout))
    if process.returncode != 0:
        raise PodmanError(
            "%s failed (exit %d): %s"
            % (_words(argv), process.returncode, stderr.decode(errors="replace").strip() or "(no stderr)")
        )
    return stdout.decode(errors="replace")


async def remove_container(podman_bin: str, container: str, *, timeout: float, cwd: str | Path) -> None:
    """``podman rm -f``; a container that is already gone is not an error."""
    try:
        await run_command([podman_bin, "rm", "-f", container], timeout=timeout, cwd=cwd)
    except PodmanError as error:
        text = str(error).lower()
        if "no such container" in text or "no container with" in text:
            return
        raise


def _words(argv: list[str]) -> str:
    """``podman image inspect`` / ``btrfs subvolume delete``: the tool and its command words."""
    words = [Path(argv[0]).name]
    for arg in argv[1:3]:
        if arg.startswith("-"):
            break
        words.append(arg)
    return " ".join(words)

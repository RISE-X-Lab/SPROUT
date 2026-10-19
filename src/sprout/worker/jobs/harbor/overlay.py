"""A Harbor task's Dockerfile as an overlay on an image, and its replay in a sandbox.

Harbor builds an image from a Dockerfile, such as the separate verifier's ``tests/Dockerfile``.
Sprout starts sandboxes from OCI images only. So it starts a sandbox from the FROM image, and
``replay`` repeats the COPY and RUN steps in it, without a registry or a build. ``parse_overlay``
refuses each instruction that a replay cannot reproduce exactly, such as ENV, WORKDIR, USER or a
second stage. A replayed environment therefore never differs from the one the task author built.
"""

from __future__ import annotations

import re
import shlex
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from sprout.worker.jobs.harbor.files import run, upload_dir, upload_file
from sprout.worker.tools.interceptors import shell_command

if TYPE_CHECKING:
    from sprout.worker.sandbox.session import SandboxSession

_INSTRUCTION = re.compile(r"^([A-Za-z]+)\s+(.*)$")


@dataclass(frozen=True)
class ImageOverlay:
    """A Dockerfile that only adds files and commands on top of the image ``base``."""

    base: str
    #: Each COPY as ``(the host source, the destination in the sandbox)``.
    copies: tuple[tuple[Path, str], ...]
    #: Each RUN command, in Dockerfile order.
    runs: tuple[str, ...]


def parse_overlay(path: Path) -> ImageOverlay:
    """The overlay of the Dockerfile at ``path``. Raises ValueError when a sandbox cannot replay it."""
    base: str | None = None
    copies: list[tuple[Path, str]] = []
    runs: list[str] = []
    for line in _dockerfile_lines(path.read_text()):
        match = _INSTRUCTION.match(line)
        word, rest = (match.group(1).upper(), match.group(2)) if match else (line.upper(), "")
        if word == "FROM":
            if base is not None:
                raise ValueError(f"{path}: multi-stage Dockerfiles are not replayed")
            base = rest.split()[0]
        elif word == "COPY":
            parts = shlex.split(rest)
            if len(parts) != 2 or parts[0].startswith("--"):
                raise ValueError(f"{path}: only `COPY <src> <dst>` is replayed, got {line!r}")
            source = path.parent / parts[0]
            if not source.exists():
                raise ValueError(f"{path}: COPY source {parts[0]} is missing")
            copies.append((source, parts[1]))
        elif word == "RUN":
            runs.append(rest)
        else:
            raise ValueError(
                f"{path}: {word} is not replayed in a sandbox; refusing to "
                f"verify in an environment that differs from the task's"
            )
    if base is None:
        raise ValueError(f"{path}: no FROM")
    return ImageOverlay(base, tuple(copies), tuple(runs))


def overlay_for(environment_dir: Path, docker_image: str | None) -> ImageOverlay | None:
    """The overlay of the directory's Dockerfile, or None when a sandbox has nothing of it to replay.

    With a ``docker_image``, a Dockerfile FROM another image built that image (None), and one FROM it
    must parse (ValueError). Without one, an unparsable Dockerfile gives None (``docs/HARBOR_PLUGIN.md``).
    """
    path = environment_dir / "Dockerfile"
    if not path.is_file():
        return None
    if docker_image and _dockerfile_base(path) != docker_image:
        return None
    try:
        return parse_overlay(path)
    except ValueError:
        if docker_image:
            raise
        return None


def replay(
    session: SandboxSession, overlay: ImageOverlay, *, cwd: str, env: dict[str, str], timeout_s: int
) -> None:
    """Put the overlay's COPY sources in the sandbox of ``session``, then run its RUN steps there.

    Each RUN step runs as root in ``cwd`` with ``env``, within ``timeout_s``. Raises
    RuntimeError when a file does not move or a step does not exit 0.
    """
    for source, destination in overlay.copies:
        if source.is_dir():
            upload_dir(session, source, destination)
        else:
            upload_file(session, source, destination)
    for command in overlay.runs:
        code, timed_out, output = run(session, shell_command(command, cwd, env, "root"), timeout_s=timeout_s)
        if timed_out or code != 0:
            raise RuntimeError(f"overlay RUN failed ({command}): {output}")


def _dockerfile_base(path: Path) -> str | None:
    """The image of the first FROM of the Dockerfile at ``path``, or None without one."""
    for line in _dockerfile_lines(path.read_text()):
        word, rest = line.split(None, 1) if " " in line else (line, "")
        if word.upper() == "FROM":
            return rest.split()[0]
    return None


def _dockerfile_lines(text: str) -> list[str]:
    """The instructions of a Dockerfile, one for each line, with continuations joined and no comments."""
    lines: list[str] = []
    pending = ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append((pending + stripped).strip())
        pending = ""
    if pending:
        lines.append(pending.strip())
    return lines

"""Commands, files and directories in a sandbox, driven through its session.

A file moves through the session's file channel (``SandboxSession.upload_file`` and
``download_file``). A directory moves as one gzip tarball, which the sandbox holds under
``/tmp`` while it moves. No file travels inside a command, so the 128 KiB that Linux allows
one command argument does not limit a file. Each function raises RuntimeError when the
sandbox refuses or a file does not move.
"""

from __future__ import annotations

import shlex
import tarfile
import tempfile
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING
from uuid import uuid4

from sprout.worker.sandbox.commands import checked

if TYPE_CHECKING:
    from sprout.worker.sandbox.session import SandboxSession

#: The seconds that a file-moving command (mkdir, tar, rm) may take. Arbitrary.
_MOVE_TIMEOUT_S = 600
#: The bytes of each output stream that a command may return. ``sprout.eval.harbor`` uses the same.
_MAX_OUTPUT_BYTES = 16 * 1024 * 1024


def run(
    session: SandboxSession, command: str, *, timeout_s: float, working_dir: str | None = None
) -> tuple[int | None, bool, str]:
    """Run a shell command, and return its exit code, whether it timed out, and its output.

    The output is the last 4000 characters of stderr, or of stdout when stderr is empty. The
    exit code is None when a timeout left none. Raises RuntimeError when the runtime refuses
    the command or does not know whether the command finished."""
    timeout = max(int(timeout_s), 1)
    arguments: dict[str, str | int] = {
        "command": command,
        "timeout": timeout,
        "max_output_bytes": _MAX_OUTPUT_BYTES,
    }
    if working_dir is not None:
        arguments["working_dir"] = working_dir
    result = session.execute("shell", arguments, timeout=timeout + 60)
    outcome = result.outcome
    if outcome is None:
        if not result.success:
            raise RuntimeError(result.error or result.output or "the sandbox refused the command")
        return 0, False, result.output[-4000:]
    if outcome.running or (outcome.exit_code is None and not outcome.timed_out):
        raise RuntimeError("the command did not settle: %s" % command[:200])
    return outcome.exit_code, bool(outcome.timed_out), (outcome.stderr or outcome.stdout)[-4000:]


def upload_file(session: SandboxSession, source: Path, destination: str) -> None:
    """Copy the host file ``source`` to ``destination`` in the sandbox, making its directory."""
    checked(session, "mkdir -p " + shlex.quote(str(PurePosixPath(destination).parent)), _MOVE_TIMEOUT_S)
    if not session.upload_file(source, destination):
        raise RuntimeError(f"could not upload {source} to {destination}: {session.transfer_error}")


def download_file(session: SandboxSession, source: str, destination: Path) -> None:
    """Copy the sandbox file ``source`` to the host file ``destination``, making its directory."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not session.download_file(source, destination):
        raise RuntimeError(f"could not download {source}: {session.transfer_error}")


def upload_dir(session: SandboxSession, source: Path, destination: str) -> None:
    """Copy the contents of the host directory ``source`` into ``destination`` in the sandbox."""
    remote = _staged()
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "files.tar.gz"
        with tarfile.open(archive, "w:gz") as stream:
            stream.add(source, arcname=".")
        try:
            upload_file(session, archive, remote)
            target = shlex.quote(destination)
            checked(session, f"mkdir -p {target} && tar -xzf {remote} -C {target}", _MOVE_TIMEOUT_S)
        finally:
            checked(session, f"rm -f {remote}", _MOVE_TIMEOUT_S)


def download_dir(session: SandboxSession, source: str, destination: Path) -> None:
    """Copy the contents of the sandbox directory ``source`` into the host directory ``destination``."""
    remote = _staged()
    with tempfile.TemporaryDirectory() as temporary:
        archive = Path(temporary) / "files.tar.gz"
        try:
            checked(session, f"tar -czf {remote} -C {shlex.quote(source)} .", _MOVE_TIMEOUT_S)
            download_file(session, remote, archive)
            destination.mkdir(parents=True, exist_ok=True)
            with tarfile.open(archive, "r:gz") as stream:
                stream.extractall(destination, filter="data")
        finally:
            checked(session, f"rm -f {remote}", _MOVE_TIMEOUT_S)


def _staged() -> str:
    return f"/tmp/sprout-harbor-{uuid4().hex}.tar.gz"

"""Settled shell commands over anything with ``execute()``: parse the runtime's
shell result and decide whether the command finished, timed out or failed."""

from __future__ import annotations

import json
from typing import Any


def shell_outcome(session: Any, command: str, timeout: int = 120) -> tuple[str, str, int | None, bool]:
    """``(stdout, stderr, exit_code, timed_out)`` of a command that finished or ran out of time.

    For a caller to whom a timeout is an answer rather than a fault: a grader
    whose tests hang scores them, it does not retry them. A timed-out command
    reports the exit code the runtime saw (137, its process group killed), or
    ``None`` when it did not say. Raises when the outcome is unknown: the
    command still running, a stream truncated, or neither an exit code nor a
    timeout.
    """
    result = session.execute("shell", {"command": command, "timeout": timeout}, timeout=timeout + 60)
    outcome = getattr(result, "outcome", None)
    if outcome is not None:
        if outcome.running or outcome.truncated or (outcome.exit_code is None and not outcome.timed_out):
            raise RuntimeError("Sandbox command incomplete or truncated")
        return outcome.stdout, outcome.stderr, outcome.exit_code, bool(outcome.timed_out)
    try:
        payload = json.loads(result.output)
    except (ValueError, TypeError):
        payload = None
    if isinstance(payload, dict) and "stdout" in payload:
        code, timed_out = payload.get("exit_code"), bool(payload.get("timed_out"))
        if (
            payload.get("running")
            or payload.get("stdout_truncated")
            or payload.get("stderr_truncated")
            or (code is None and not timed_out)
        ):
            raise RuntimeError("Sandbox command incomplete or truncated")
        return (
            str(payload["stdout"]),
            str(payload.get("stderr") or ""),
            None if code is None else int(code),
            timed_out,
        )
    if not result.success:
        raise RuntimeError(result.error or result.output or "Sandbox tool execution failed")
    return result.output, "", 0, False


def shell(session: Any, command: str, timeout: int = 120) -> tuple[str, str, int]:
    """``(stdout, stderr, exit_code)`` of a settled command; a timeout raises."""
    stdout, stderr, code, timed_out = shell_outcome(session, command, timeout)
    if timed_out:
        raise RuntimeError("Sandbox command timed out after %d s" % timeout)
    return stdout, stderr, code


def checked(session: Any, command: str, timeout: int = 120) -> str:
    """stdout of a command that must exit 0."""
    stdout, stderr, code = shell(session, command, timeout)
    if code != 0:
        raise RuntimeError(f"Sandbox command exited {code}: {(stderr or stdout)[-2000:]}")
    return stdout

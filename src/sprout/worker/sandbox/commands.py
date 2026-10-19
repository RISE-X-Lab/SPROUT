"""Shell commands whose outcome is settled, run through any session that has ``execute()``.

Each function makes one runtime ``shell`` call. It raises ``RuntimeError`` when the call failed or
the outcome of the command is unknown.
"""

from __future__ import annotations

from typing import Any, Protocol

from sprout.worker.tools.result import ToolResult


class _Session(Protocol):
    """A session that runs one tool call and returns its result, such as ``SandboxSession``."""

    def execute(self, tool_name: str, args: dict[str, Any], /, timeout: float) -> ToolResult: ...


def shell_outcome(session: _Session, command: str, timeout: int = 120) -> tuple[str, str, int | None, bool]:
    """``(stdout, stderr, exit_code, timed_out)`` of a command that finished or ran out of time.

    For a caller to whom a timeout is an answer: a grader scores hung tests, it does not retry them. A timeout
    gives the runtime's exit code (137 when it killed the process group), or None. Raises ``RuntimeError`` if
    the call failed, the command still runs, a stream was clipped, or there is no exit code and no timeout."""
    result = session.execute("shell", {"command": command, "timeout": timeout}, timeout=timeout + 60)
    outcome = result.outcome
    if outcome is not None:
        if outcome.running or outcome.truncated or (outcome.exit_code is None and not outcome.timed_out):
            raise RuntimeError("Sandbox command incomplete or truncated")
        return outcome.stdout, outcome.stderr, outcome.exit_code, bool(outcome.timed_out)
    if not result.success:
        raise RuntimeError(result.error or result.output or "Sandbox tool execution failed")
    # The runtime sends bare stdout only for exit 0 without stderr. The SDK already read any outcome,
    # so JSON that the command printed stays its stdout.
    return result.output, "", 0, False


def shell(session: _Session, command: str, timeout: int = 120) -> tuple[str, str, int]:
    """``(stdout, stderr, exit_code)`` of a settled command. A timeout raises ``RuntimeError``."""
    stdout, stderr, code, timed_out = shell_outcome(session, command, timeout)
    if timed_out:
        raise RuntimeError("Sandbox command timed out after %d s" % timeout)
    # shell_outcome raises for a command without an exit code that did not time out.
    assert code is not None
    return stdout, stderr, code


def checked(session: _Session, command: str, timeout: int = 120) -> str:
    """The stdout of a command that must exit 0. Any other exit raises ``RuntimeError``."""
    stdout, stderr, code = shell(session, command, timeout)
    if code != 0:
        raise RuntimeError(f"Sandbox command exited {code}: {(stderr or stdout)[-2000:]}")
    return stdout

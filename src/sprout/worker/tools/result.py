"""The result of one tool call, ``ToolResult``, and what a command produced, ``CommandOutcome``.

Every layer runs a tool through ``executor(tool_name, args) -> ToolResult``, so these types belong
to the execution plane, not to a benchmark.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from sprout_sandbox.result import ToolResult as SdkToolResult


@dataclass
class ToolResult:
    """Result of one tool call. ``output`` and ``error`` are alternatives, not two views of one message.

    ``output`` is what the tool produced. ``error`` means that the tool refused to produce anything, for
    example for bad arguments, no sandbox or a transport failure. A command that exits non-zero has its
    output, no error, and ``success`` False."""

    success: bool
    output: str
    error: str | None = None
    #: What running a command produced. None for a plain success, a refusal, or a tool that runs no command.
    outcome: CommandOutcome | None = None

    @classmethod
    def from_sdk(cls, result: SdkToolResult) -> ToolResult:
        """The ``ToolResult`` of a ``sprout_sandbox.ToolResult``.

        The SDK gives one text and an ``is_error`` flag, so the text of a failed call is ``output``, not
        ``error``. The command's outcome stays structured, because only the agent runtime renders it."""
        return cls(success=not result.is_error, output=result.output, outcome=CommandOutcome.from_sdk(result))


@dataclass
class CommandOutcome:
    """What running a command produced: the runtime's ``CommandOutcome`` (``runtime/tools/boundedlog.go``).

    ``shell`` reports one when bare stdout is not enough: a non-zero exit, stderr, a clipped stream or a
    timeout. ``exit_code`` is None when the runtime did not say. Test it with ``is None``, because 0 is real.
    """

    exit_code: int | None = None
    stdout: str = ""
    stderr: str = ""
    running: bool = False
    timed_out: bool = False
    stdout_bytes: int = 0
    stderr_bytes: int = 0
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def truncated(self) -> bool:
        """Whether the runtime dropped part of either stream."""
        return self.stdout_truncated or self.stderr_truncated

    @classmethod
    def from_sdk(cls, result: SdkToolResult) -> CommandOutcome | None:
        """The outcome fields of a ``sprout_sandbox.ToolResult``, or None when it has no command outcome."""
        if getattr(result, "stdout", None) is None:
            return None
        return cls(
            exit_code=result.exit_code,
            stdout=result.stdout or "",
            stderr=result.stderr or "",
            running=result.running,
            timed_out=result.timed_out,
            stdout_bytes=result.stdout_bytes or 0,
            stderr_bytes=result.stderr_bytes or 0,
            stdout_truncated=result.stdout_truncated,
            stderr_truncated=result.stderr_truncated,
        )

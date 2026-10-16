from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: Fields of the runtime's CommandOutcome schema (runtime/tools/boundedlog.go),
#: the shape `shell` answers in when there is more to say than bare stdout.
_OUTCOME_FIELDS = frozenset(
    {
        "stdout",
        "stderr",
        "exit_code",
        "running",
        "timed_out",
        "stdout_bytes",
        "stderr_bytes",
        "stdout_truncated",
        "stderr_truncated",
        "max_per_stream",
    }
)


@dataclass
class ToolResult:
    output: str
    is_error: bool

    # -- command outcome ----------------------------------------------------- #
    # Set when the tool ran a command and had something to report beyond its
    # stdout: a non-zero exit, output on stderr, a clipped stream, a process
    # still running. A plain success leaves them None/False and puts the output
    # in `output` alone -- there is nothing else to say about `echo hello`.
    #
    # `exit_code` is None when unknown: no command was run, or one is still
    # running. Test it with `is None`, since 0 is a real answer.
    exit_code: int | None = None
    stdout: str | None = None
    stderr: str | None = None
    running: bool = False
    timed_out: bool = False
    stdout_bytes: int | None = None
    stderr_bytes: int | None = None
    stdout_truncated: bool = False
    stderr_truncated: bool = False

    @property
    def truncated(self) -> bool:
        """Whether the runtime dropped part of either stream."""
        return self.stdout_truncated or self.stderr_truncated

    @classmethod
    def from_response(cls, result: dict[str, Any]) -> ToolResult:
        """Build from one JSON-RPC ``tools/call`` result.

        The one parser for the payload: a hand-rolled copy of this drifted the
        moment the shape grew a field.

        A text slot holding a CommandOutcome is unpacked into fields; any other
        text is passed through as ``output``. Detection is by shape rather than
        by tool name, so a command that prints JSON is not mistaken for one.
        """
        content = result.get("content") or []
        text = content[0].get("text", "") if content else ""
        self = cls(output=text, is_error=result.get("isError", False))
        outcome = _parse_outcome(text)
        if outcome is None:
            return self
        self.exit_code = outcome.get("exit_code")
        self.stdout = outcome.get("stdout", "")
        self.stderr = outcome.get("stderr", "")
        self.running = bool(outcome.get("running"))
        self.timed_out = bool(outcome.get("timed_out"))
        self.stdout_bytes = outcome.get("stdout_bytes")
        self.stderr_bytes = outcome.get("stderr_bytes")
        self.stdout_truncated = bool(outcome.get("stdout_truncated"))
        self.stderr_truncated = bool(outcome.get("stderr_truncated"))
        return self


def _parse_outcome(text: str) -> dict[str, Any] | None:
    """The CommandOutcome in ``text``, or None if it is not one.

    Requires both streams and a recognised field set: a JSON document a *command*
    happened to print must not be mistaken for the runtime's own report.
    """
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if "stdout" not in payload or "stderr" not in payload:
        return None
    if not payload.keys() <= _OUTCOME_FIELDS:
        return None
    return payload

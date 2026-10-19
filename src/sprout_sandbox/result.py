"""The result of one tool call, parsed from the runtime's JSON-RPC reply."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: The JSON fields of the runtime's ``CommandOutcome`` (``runtime/tools/boundedlog.go``).
#: Keep the two sets equal: a reply with a field outside this set is read as plain output.
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
    """The text that one tool call returned, and whether the call failed.

    The other fields hold the command outcome, when the runtime reported more than a plain stdout.
    Then ``output`` is the outcome's JSON text. Otherwise every outcome field is ``None`` or false.
    """

    output: str
    is_error: bool
    #: The command's exit code, or ``None`` if the runtime did not report one.
    #: Test it with ``is None``, because ``0`` is a real exit code.
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
        """The result in the ``result`` object of one JSON-RPC ``tools/call`` reply.

        ``output`` is the first text slot, or ``""`` if there is none. A text with the shape of a
        ``CommandOutcome`` also fills the outcome fields. Other JSON that a command printed does not.
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
    """The ``CommandOutcome`` that ``text`` encodes, or ``None`` if it encodes none.

    Both streams and no unknown field are required, so that a JSON document that a command
    printed is not taken for the runtime's own report.
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

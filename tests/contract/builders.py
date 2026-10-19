"""Creation functions for the values that the contract tests pass around: assistant turns and tool lists."""

from __future__ import annotations

import json
from typing import Any

#: A reviewer's text that a test does not read.
REVIEWER_TEXT = "The existing branch suggests checking this boundary."


def bash_call(command: str = "printf check", identifier: str = "reviewer-call-1") -> dict[str, Any]:
    """A chat/completions call of the ``bash`` tool with ``command``."""
    return {
        "id": identifier,
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }


def assistant_turn(
    command: str = "printf check", identifier: str = "reviewer-call-1", *, content: str = REVIEWER_TEXT
) -> dict[str, Any]:
    """An assistant turn, in the form that its author writes, with one ``bash`` call of ``command``."""
    return {"role": "assistant", "content": content, "tool_calls": [bash_call(command, identifier)]}


def bash_tools(**command: Any) -> list[dict[str, Any]]:
    """The chat/completions ``tools`` list of mini 2.4.6, with ``command`` added to the command's schema."""
    return [
        {
            "type": "function",
            "function": {
                "name": "bash",
                "description": "Execute a bash command",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "command": {"type": "string", "description": "The bash command to execute", **command}
                    },
                    "required": ["command"],
                    "additionalProperties": False,
                },
            },
        }
    ]

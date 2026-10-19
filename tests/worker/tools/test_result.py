"""``ToolResult.from_sdk``: the result of one tool call, from the reply that the SDK parsed.

Each reply is the JSON-RPC ``result`` that the runtime sends, parsed by ``sprout_sandbox``.
"""

from __future__ import annotations

import json

from sprout.worker.tools.result import CommandOutcome, ToolResult
from sprout_sandbox import ToolResult as SdkToolResult


def runtime_reply(text: str, *, is_error: bool) -> SdkToolResult:
    return SdkToolResult.from_response({"content": [{"type": "text", "text": text}], "isError": is_error})


def test_a_failed_command_keeps_its_output_as_output_and_its_outcome_structured():
    """``error`` is for a tool that refused to run. A command that ran and failed has output instead."""
    text = json.dumps({"stdout": "", "stderr": "boom\n", "exit_code": 2, "stderr_bytes": 5})

    result = ToolResult.from_sdk(runtime_reply(text, is_error=True))

    assert result == ToolResult(
        success=False,
        output=text,
        error=None,
        outcome=CommandOutcome(exit_code=2, stdout="", stderr="boom\n", stdout_bytes=0, stderr_bytes=5),
    )


def test_a_reply_of_bare_stdout_has_no_outcome():
    """The runtime sends bare stdout for a command that exited 0 with no stderr."""
    result = ToolResult.from_sdk(runtime_reply("hello\n", is_error=False))

    assert result == ToolResult(success=True, output="hello\n", error=None, outcome=None)

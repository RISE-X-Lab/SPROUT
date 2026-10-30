"""The harness's system and task messages for each tool set."""

from __future__ import annotations

from sprout.worker.agents.harness.prompts import system_message, task_message
from sprout.worker.agents.harness.tools import OUTPUT_LIMIT, TOOLSETS

DEFAULT = frozenset(TOOLSETS["default"])
FULL = frozenset(TOOLSETS["full"])
#: The full tool set's system message, as the overfit run of 2026-10-30 trained on it.
FULL_SYSTEM = """You are an expert software engineer. You solve tasks in a sandboxed Linux machine, and \
you act only through your tools.

Every response has two parts: first your reasoning, then exactly one tool call. Put all of your \
thinking in the reasoning block. You see the tool's result before your next response.

Format rule: after </think>, write the tool call and nothing else. Any text between </think> and \
<tool_call> is rejected, and the call does not run.

<workspace>
Working directory: /testbed
Repository root: /testbed
Each execute_bash call runs in a new shell in the working directory, so cd and export do \
not carry over to the next call.
</workspace>"""


def test_the_full_tool_set_keeps_the_system_message_of_the_first_overfit_run() -> None:
    assert system_message(cwd="/testbed", repository_dir="/testbed", tools=FULL) == FULL_SYSTEM


def test_without_read_file_the_system_message_says_how_to_read_files_with_commands() -> None:
    message = system_message(cwd="/testbed", repository_dir="/testbed")

    assert "grep -n" in message and "sed -n" in message
    assert "%d characters" % OUTPUT_LIMIT in message
    assert "read_file" not in message
    assert message.endswith("</workspace>")


def test_the_task_message_names_only_the_edit_tools_offered() -> None:
    assert "Use str_replace for edits, and write_file for new files." in task_message("Fix it.", FULL)
    assert "write_file" not in task_message("Fix it.") and "heredoc" in task_message("Fix it.", DEFAULT)

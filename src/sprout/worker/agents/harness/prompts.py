"""The harness's prompts: the system message, the task message, and the feedback on a rejected turn.

The task message follows mini-swe-agent-plus's SWE-bench template, rewritten for the harness's tools.
Like Arctic-SWE's harness, it asks the agent to run a relevant test before it submits.
"""

from __future__ import annotations

from sprout.worker.agents.harness.tools import OUTPUT_LIMIT, TOOLSETS

SYSTEM = """You are an expert software engineer. You solve tasks in a sandboxed Linux machine, and you \
act only through your tools.

Every response has two parts: first your reasoning, then exactly one tool call. Put all of your \
thinking in the reasoning block. You see the tool's result before your next response.

Format rule: after </think>, write the tool call and nothing else. Any text between </think> and \
<tool_call> is rejected, and the call does not run.

<workspace>
Working directory: {cwd}
{repository}Each execute_bash call runs in a new shell in the working directory, so cd and export do \
not carry over to the next call.{reading}
</workspace>"""

#: How to read files when no tool reads them: the tool sets without read_file add it to the workspace.
READING = (
    """
Read files with execute_bash too: find code with `grep -rn PATTERN DIR`, and print the lines you need \
with `sed -n 'START,ENDp' FILE` or `grep -n -A 20 PATTERN FILE`. Output past %d characters keeps only \
its beginning and its end, so print line ranges rather than whole long files."""
    % OUTPUT_LIMIT
)

TASK = """<task>
{task}
</task>

Make the changes to the non-test files in the repository that resolve the task, in a way that is \
general and consistent with the codebase. Do not modify the existing tests, and do not change \
configuration files such as pyproject.toml or setup.cfg unless the task asks for it.

A good way to work:
1. Find and read the code that the task concerns.
2. Reproduce the problem with a small script, and run it.
3. Edit the source code to fix it. {edit_hint}
4. Run your script again, and run the existing tests that cover the code you changed.
5. Think about edge cases, and check them.
6. When the fix is complete and tested, call submit. You cannot act after submit."""

FEEDBACK = """Your last response was rejected and nothing ran: {message}

Respond again: reason first, then make exactly one tool call in the required format."""

REPEAT_NOTE = (
    "\n\nNote: you made this exact call before and nothing has changed since. If it did not help then, "
    "try a different approach."
)


def system_message(
    *, cwd: str, repository_dir: str | None, tools: frozenset[str] = frozenset(TOOLSETS["default"])
) -> str:
    """The system message for a workspace in ``cwd``, in the repository ``repository_dir`` if any, with
    the ``tools`` offered."""
    repository = "Repository root: %s\n" % repository_dir if repository_dir else ""
    reading = "" if "read_file" in tools else READING
    return SYSTEM.format(cwd=cwd, repository=repository, reading=reading)


def task_message(task: str, tools: frozenset[str] = frozenset(TOOLSETS["default"])) -> str:
    """The first user message: the task and how to work on it with the ``tools`` offered."""
    edit_hint = "Use str_replace for edits, and %s for new files." % (
        "write_file" if "write_file" in tools else "execute_bash (for example a heredoc)"
    )
    return TASK.format(task=task.strip(), edit_hint=edit_hint)


def feedback_message(message: str) -> str:
    """The user message that answers a rejected turn in evaluation mode."""
    return FEEDBACK.format(message=message)

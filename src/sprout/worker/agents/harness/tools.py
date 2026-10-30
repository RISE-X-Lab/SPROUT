"""The harness's tools: their schemas, the runtime call that runs each, and the result that the model reads.

Each model call is one ``ToolAccess`` call, so the journal keeps one step per call. ``execute_bash``
runs as the runtime's ``shell``; ``read_file``, ``write_file`` and ``str_replace`` as its ``text_editor``
``view``, ``write`` and ``str_replace``. ``submit`` runs nothing in the sandbox. Results say what
happened and, after a failure, what to do next.
"""

from __future__ import annotations

import posixpath
import re
import shlex
from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from sprout.worker.agents.harness.qwen_format import Call, FormatViolation

if TYPE_CHECKING:
    from sprout.worker.sandbox.image_shell import ImageShell
    from sprout.worker.tools.result import ToolResult

#: Seconds a command runs before the runtime kills it, unless the call asks for another limit.
COMMAND_TIMEOUT_S = 60
#: The longest limit that a call may ask for.
MAX_COMMAND_TIMEOUT_S = 600
#: Characters of command output that a result keeps: the first and the last half of them.
OUTPUT_LIMIT = 10_000
#: Lines that ``read_file`` reads when the call does not say, and the characters it returns at most.
READ_LINES = 400
READ_CHARS = 20_000
#: Variables that keep commands from paging, prompting or drawing progress bars.
COMMAND_ENV = {
    "PAGER": "cat",
    "MANPAGER": "cat",
    "LESS": "-R",
    "PIP_PROGRESS_BAR": "off",
    "TQDM_DISABLE": "1",
}

_PATH = {"type": "string", "description": "An absolute path, or a path relative to the working directory."}
SCHEMAS: dict[str, dict[str, Any]] = {
    "execute_bash": {
        "description": (
            "Run a bash command in the task's sandbox and get its exit code and output. Each call "
            "starts a new shell in the working directory, so cd and export do not carry over to the "
            "next call: chain commands with && instead. Commands must not wait for input. Output "
            "longer than %d characters keeps only its beginning and its end." % OUTPUT_LIMIT
        ),
        "properties": {
            "command": {"type": "string", "description": "The bash command to run."},
            "timeout": {
                "type": "integer",
                "description": "Seconds before the command is killed: default %d, at most %d."
                % (COMMAND_TIMEOUT_S, MAX_COMMAND_TIMEOUT_S),
            },
        },
        "required": ["command"],
    },
    "read_file": {
        "description": (
            "Read a text file with numbered lines: limit lines from line offset. The result says "
            "whether the file goes on and the offset to read from next."
        ),
        "properties": {
            "file_path": _PATH,
            "offset": {"type": "integer", "description": "The first line to read, from 1. Default 1."},
            "limit": {"type": "integer", "description": "How many lines to read. Default %d." % READ_LINES},
        },
        "required": ["file_path"],
    },
    "write_file": {
        "description": (
            "Create a file, or overwrite a whole file, with exactly content. Missing directories are "
            "created. To change part of an existing file, use str_replace."
        ),
        "properties": {
            "file_path": _PATH,
            "content": {"type": "string", "description": "The complete new text of the file."},
        },
        "required": ["file_path", "content"],
    },
    "str_replace": {
        "description": (
            "Replace old_string in a file with new_string. old_string must match the file exactly, "
            "whitespace and indentation included, and occur exactly once: include enough surrounding "
            "lines to make it unique. An empty new_string deletes old_string. The result shows the "
            "edited lines with their numbers."
        ),
        "properties": {
            "file_path": _PATH,
            "old_string": {"type": "string", "description": "The exact text to replace."},
            "new_string": {"type": "string", "description": "The text to put in its place."},
        },
        "required": ["file_path", "old_string", "new_string"],
    },
    "submit": {
        "description": (
            "Finish the task. Call it once your change is complete and checked, for example with the "
            "relevant tests. The repository is graded as it is, and nothing runs after this call."
        ),
        "properties": {},
        "required": [],
    },
}

#: The tool sets that a run can offer. ``default`` is the harness of the Arctic-SWE paper: a shell, the
#: string-replace editor and submit. ``full`` adds read_file and write_file, as the first overfit run had.
TOOLSETS: dict[str, tuple[str, ...]] = {
    "default": ("execute_bash", "str_replace", "submit"),
    "full": ("execute_bash", "read_file", "write_file", "str_replace", "submit"),
}


def tool_schemas(toolset: str) -> list[dict[str, Any]]:
    """The Chat Completions ``tools`` of ``toolset``. Raises ``ValueError`` for an unknown tool set."""
    if toolset not in TOOLSETS:
        raise ValueError("Unknown harness toolset %r; choose one of %s" % (toolset, ", ".join(TOOLSETS)))
    return [
        {
            "type": "function",
            "function": {
                "name": name,
                "description": SCHEMAS[name]["description"],
                "parameters": {
                    "type": "object",
                    "properties": deepcopy(SCHEMAS[name]["properties"]),
                    "required": list(SCHEMAS[name]["required"]),
                    "additionalProperties": False,
                },
            },
        }
        for name in TOOLSETS[toolset]
    ]


def bind(call: Call, tools: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The arguments of ``call``, typed by its tool's schema. Raises ``FormatViolation`` if they break it."""
    names = [tool["function"]["name"] for tool in tools]
    if call.name not in names:
        raise FormatViolation(
            "unknown_tool", "There is no tool named %r. The tools are %s." % (call.name, ", ".join(names))
        )
    parameters = tools[names.index(call.name)]["function"]["parameters"]
    properties: Mapping[str, Any] = parameters["properties"]
    arguments: dict[str, Any] = {}
    for key, value in call.parameters:
        if key not in properties:
            accepted = ", ".join(properties) or "no parameters"
            raise FormatViolation(
                "unknown_parameter", "%s has no parameter %r; it takes %s." % (call.name, key, accepted)
            )
        if properties[key]["type"] == "integer":
            if not re.fullmatch(r"-?(0|[1-9][0-9]*)", value):
                raise FormatViolation(
                    "invalid_parameter",
                    "%s's %r must be a whole number, not %r." % (call.name, key, value[:80]),
                )
            arguments[key] = int(value)
        else:
            arguments[key] = value
    missing = [key for key in parameters["required"] if key not in arguments]
    if missing:
        raise FormatViolation(
            "missing_parameter",
            "%s needs the parameter %s." % (call.name, " and ".join(repr(k) for k in missing)),
        )
    return arguments


@dataclass(frozen=True)
class RuntimeCall:
    """The runtime tool and arguments that run one model call, and what its result needs to know."""

    tool: str
    args: dict[str, Any]
    path: str | None = None
    first_line: int = 1
    lines: int = 0
    timeout_s: int = 0


def runtime_call(
    name: str, arguments: Mapping[str, Any], *, cwd: str, remaining_s: float, image_shell: ImageShell | None
) -> RuntimeCall:
    """The runtime call of the model call ``name(arguments)`` in ``cwd``, within ``remaining_s`` seconds."""
    if name == "execute_bash":
        timeout = max(1, min(int(arguments.get("timeout", COMMAND_TIMEOUT_S)), MAX_COMMAND_TIMEOUT_S))
        timeout = max(1, min(timeout, int(remaining_s)))
        prefix = " ".join(shlex.quote(f"{key}={value}") for key, value in COMMAND_ENV.items())
        line = f"env {prefix} bash -c {shlex.quote(arguments['command'])}"
        if image_shell is not None:
            line = image_shell.command(line, working_dir=cwd)
        return RuntimeCall(
            "shell", {"command": line, "working_dir": cwd, "timeout": timeout}, timeout_s=timeout
        )
    path = resolve(arguments.get("file_path", ""), cwd)
    if name == "read_file":
        first = max(1, int(arguments.get("offset", 1)))
        count = max(1, int(arguments.get("limit", READ_LINES)))
        # Two lines more than asked tell whether the file goes on, past the empty "line" that
        # view shows after a final newline.
        view = {"command": "view", "path": path, "view_range": [first, first + count + 1]}
        return RuntimeCall("text_editor", view, path=path, first_line=first, lines=count)
    if name == "write_file":
        return RuntimeCall(
            "text_editor", {"command": "write", "path": path, "file_text": arguments["content"]}, path
        )
    if name == "str_replace":
        replace = {
            "command": "str_replace",
            "path": path,
            "old_str": arguments["old_string"],
            "new_str": arguments["new_string"],
        }
        return RuntimeCall("text_editor", replace, path)
    raise ValueError("%s runs no runtime tool" % name)


def resolve(path: str, cwd: str) -> str:
    """``path`` as an absolute path: a relative one is taken from ``cwd``."""
    if not path:
        return ""
    return posixpath.normpath(path if path.startswith("/") else posixpath.join(cwd, path))


def observation(
    name: str, call: RuntimeCall, result: ToolResult, *, elapsed_s: float, tools: frozenset[str] = frozenset()
) -> str:
    """The text that the model reads as the result of ``call``. Its hints name only the ``tools`` offered."""
    if name == "execute_bash":
        return _command_result(call, result, elapsed_s)
    text = result.error or result.output
    if not call.path:
        return "file_path is empty: give the path of a file."
    if name == "read_file":
        return _read_result(call, result.success, text)
    if name == "write_file":
        return text if result.success else "Could not write %s: %s" % (call.path, _missing(text, call.path))
    return text if result.success else _replace_failure(text, call.path, tools)


def _command_result(call: RuntimeCall, result: ToolResult, elapsed_s: float) -> str:
    outcome = result.outcome
    if outcome is None:
        return "The command did not run: %s" % (result.error or result.output or "no reason given")
    if outcome.timed_out:
        head = "The command timed out after %ds and was killed." % call.timeout_s
        hint = "\nGive it a longer timeout (at most %ds), or make it do less." % MAX_COMMAND_TIMEOUT_S
    else:
        head, hint = "Exit code: %s (%.1fs)" % (outcome.exit_code, elapsed_s), ""
    clipped = " The sandbox kept only part of the output." if outcome.truncated else ""
    if outcome.stdout and outcome.stderr:
        body = "Stdout:\n%s\nStderr:\n%s" % (
            _bounded(outcome.stdout, OUTPUT_LIMIT // 2),
            _bounded(outcome.stderr, OUTPUT_LIMIT // 2),
        )
    elif outcome.stdout or outcome.stderr:
        body = "Output:\n" + _bounded(outcome.stdout or outcome.stderr, OUTPUT_LIMIT)
    else:
        body = "(no output)"
    return head + clipped + "\n" + body + hint


def _bounded(text: str, limit: int) -> str:
    """``text``, or its first and last ``limit // 2`` characters and how many were left out."""
    if len(text) <= limit:
        return text
    half = limit // 2
    return "%s\n[... %d characters omitted ...]\n%s" % (text[:half], len(text) - 2 * half, text[-half:])


_VIEW_LINE = re.compile(r" *(\d+) \| ")


def _read_result(call: RuntimeCall, success: bool, text: str) -> str:
    """The numbered lines that ``view`` returned, with where they are in the file and what comes next."""
    if not success:
        return "Could not read %s: %s" % (call.path, _missing(text, call.path))
    lines = [line for line in text.split("\n") if _VIEW_LINE.match(line)]
    if len(lines) <= call.lines + 1 and lines and _VIEW_LINE.sub("", lines[-1], count=1) == "":
        lines.pop()  # The view reached the end: its last "line" is the one after the final newline.
    more = len(lines) > call.lines
    kept: list[str] = []
    size = 0
    for line in lines[: call.lines]:
        if size + len(line) > READ_CHARS:
            more = True
            if not kept:  # One line longer than the whole budget: show its start.
                kept.append(line[:READ_CHARS] + " [... line cut at %d characters]" % READ_CHARS)
            break
        kept.append(line)
        size += len(line) + 1
    if not kept:
        if call.first_line == 1:
            return "%s is empty." % call.path
        return "%s has no line %d: the file is shorter." % (call.path, call.first_line)
    last = call.first_line + len(kept) - 1
    where = "the file goes on: read from offset=%d" % (last + 1) if more else "end of file"
    return "%s, lines %d-%d (%s):\n%s" % (call.path, call.first_line, last, where, "\n".join(kept))


def _missing(text: str, path: str | None) -> str:
    """``text``, or what to do about a path that does not exist or is a directory."""
    if "no such file or directory" in text:
        return "it does not exist. Check the path, for example with `ls` in execute_bash."
    if "is a directory" in text:
        return "it is a directory. List it with execute_bash, for example `ls -la %s`." % path
    return text


def _replace_failure(text: str, path: str, tools: frozenset[str]) -> str:
    """Why ``str_replace`` changed nothing, and what to do instead with the ``tools`` offered."""
    reader = "read_file" if "read_file" in tools else "execute_bash (for example `sed -n '10,40p' FILE`)"
    writer = "write_file" if "write_file" in tools else "execute_bash (for example a heredoc)"
    if text == "No match found":
        reason = (
            "old_string was not found in %s. It must match the file exactly, whitespace and indentation "
            "included: show the lines with %s and copy them." % (path, reader)
        )
    elif text.startswith("Multiple matches"):
        count = re.search(r"\((\d+)\)", text)
        lines = re.search(r"lines? ([0-9, and]+)", text)
        reason = (
            "old_string occurs %s times in %s%s. Include more surrounding lines so that it matches once."
            % (
                count.group(1) if count else "several",
                path,
                " (starting at lines %s)" % lines.group(1) if lines else "",
            )
        )
    elif text.startswith("old_str and new_str are identical"):
        reason = "old_string and new_string are identical."
    elif text.startswith("old_str is required"):
        reason = "old_string is empty. To create or overwrite a whole file, use %s." % writer
    elif "no such file or directory" in text:
        reason = "%s does not exist. Create it with %s." % (path, writer)
    else:
        reason = text
    return "No change made: " + reason

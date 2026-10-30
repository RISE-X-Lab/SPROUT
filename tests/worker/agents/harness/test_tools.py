"""The harness's tools: their schemas, the typed arguments of a parsed call, the runtime call of each
tool, and the result that the model reads."""

from __future__ import annotations

import shlex

import pytest

from sprout.worker.agents.harness.qwen_format import Call, FormatViolation
from sprout.worker.agents.harness.tools import (
    OUTPUT_LIMIT,
    READ_CHARS,
    RuntimeCall,
    bind,
    observation,
    runtime_call,
    tool_schemas,
)
from sprout.worker.tools.result import CommandOutcome, ToolResult

TOOLS = tool_schemas("full")


def view(*lines: str, first: int = 1) -> str:
    """What the runtime's ``text_editor view`` returns for ``lines`` from line ``first``."""
    return "".join("  %6d | %s\n" % (first + i, line) for i, line in enumerate(lines))


def test_the_default_toolset_is_the_papers_three_tools_and_full_adds_the_file_tools() -> None:
    names = [tool["function"]["name"] for tool in TOOLS]

    assert [t["function"]["name"] for t in tool_schemas("default")] == [
        "execute_bash",
        "str_replace",
        "submit",
    ]
    assert names == ["execute_bash", "read_file", "write_file", "str_replace", "submit"]
    assert all(tool["function"]["parameters"]["additionalProperties"] is False for tool in TOOLS)
    with pytest.raises(ValueError, match="Unknown harness toolset"):
        tool_schemas("all")


def test_bind_types_each_argument_by_the_schema() -> None:
    call = Call("read_file", (("file_path", "a.py"), ("offset", "10"), ("limit", "-1")))

    assert bind(call, TOOLS) == {"file_path": "a.py", "offset": 10, "limit": -1}


@pytest.mark.parametrize(
    ("call", "code"),
    [
        (Call("bash", (("command", "ls"),)), "unknown_tool"),
        (Call("execute_bash", (("cmd", "ls"),)), "unknown_parameter"),
        (Call("execute_bash", ()), "missing_parameter"),
        (Call("execute_bash", (("command", "ls"), ("timeout", "010"))), "invalid_parameter"),
        (Call("read_file", (("file_path", "a"), ("offset", "1.5"))), "invalid_parameter"),
        (Call("submit", (("reason", "done"),)), "unknown_parameter"),
    ],
    ids=[
        "unknown tool",
        "unknown parameter",
        "missing parameter",
        "leading zero",
        "not a whole number",
        "submit args",
    ],
)
def test_a_call_that_breaks_its_schema_is_a_format_violation(call: Call, code: str) -> None:
    with pytest.raises(FormatViolation) as rejected:
        bind(call, TOOLS)

    assert rejected.value.code == code


def test_a_command_runs_in_a_fresh_noninteractive_shell_within_its_timeout() -> None:
    call = runtime_call(
        "execute_bash", {"command": "ls -la"}, cwd="/testbed", remaining_s=900, image_shell=None
    )

    assert call.tool == "shell"
    assert call.args["working_dir"] == "/testbed"
    assert call.args["timeout"] == 60
    assert call.args["command"].startswith("env ") and call.args["command"].endswith(
        "bash -c " + shlex.quote("ls -la")
    )
    assert "PAGER=cat" in call.args["command"]


def test_a_command_timeout_is_capped_at_the_maximum_and_at_the_time_left() -> None:
    long = runtime_call(
        "execute_bash", {"command": "x", "timeout": 5000}, cwd="/", remaining_s=900, image_shell=None
    )
    late = runtime_call(
        "execute_bash", {"command": "x", "timeout": 300}, cwd="/", remaining_s=12.5, image_shell=None
    )

    assert long.args["timeout"] == 600
    assert late.args["timeout"] == 12


def test_a_command_runs_as_the_image_user_when_the_run_has_an_image_shell() -> None:
    class Shell:
        def command(self, line: str, *, working_dir: str) -> str:
            return "runuser -u app -- " + line

    call = runtime_call("execute_bash", {"command": "id"}, cwd="/app", remaining_s=60, image_shell=Shell())  # type: ignore[arg-type]

    assert call.args["command"].startswith("runuser -u app -- env ")


def test_file_tools_resolve_relative_paths_and_read_two_lines_more_than_asked() -> None:
    read = runtime_call(
        "read_file",
        {"file_path": "src/../a.py", "offset": 5, "limit": 10},
        cwd="/testbed",
        remaining_s=60,
        image_shell=None,
    )
    write = runtime_call(
        "write_file",
        {"file_path": "/tmp/x", "content": "y"},
        cwd="/testbed",
        remaining_s=60,
        image_shell=None,
    )
    edit = runtime_call(
        "str_replace",
        {"file_path": "a.py", "old_string": "o", "new_string": "n"},
        cwd="/testbed",
        remaining_s=60,
        image_shell=None,
    )

    assert read.args == {"command": "view", "path": "/testbed/a.py", "view_range": [5, 16]}
    assert write.args == {"command": "write", "path": "/tmp/x", "file_text": "y"}
    assert edit.args == {"command": "str_replace", "path": "/testbed/a.py", "old_str": "o", "new_str": "n"}


def command(outcome: CommandOutcome, *, timeout: int = 60) -> str:
    call = RuntimeCall("shell", {}, timeout_s=timeout)
    return observation(
        "execute_bash", call, ToolResult(outcome.exit_code == 0, "", outcome=outcome), elapsed_s=1.24
    )


def test_a_command_result_names_its_exit_code_duration_and_streams() -> None:
    assert command(CommandOutcome(exit_code=0, stdout="a.py\n")) == "Exit code: 0 (1.2s)\nOutput:\na.py\n"
    assert command(CommandOutcome(exit_code=1, stdout="out\n", stderr="err\n")) == (
        "Exit code: 1 (1.2s)\nStdout:\nout\n\nStderr:\nerr\n"
    )
    assert command(CommandOutcome(exit_code=0)) == "Exit code: 0 (1.2s)\n(no output)"


def test_a_long_output_keeps_its_beginning_and_end_and_counts_the_rest() -> None:
    text = "a" * OUTPUT_LIMIT + "b" * 100 + "c" * OUTPUT_LIMIT

    result = command(CommandOutcome(exit_code=0, stdout=text))

    assert "[... %d characters omitted ...]" % (len(text) - OUTPUT_LIMIT) in result
    assert result.startswith("Exit code: 0 (1.2s)\nOutput:\naaa") and result.endswith("ccc")


def test_a_timed_out_command_says_so_and_how_to_give_it_longer() -> None:
    result = command(CommandOutcome(exit_code=None, stdout="partial\n", timed_out=True), timeout=60)

    assert result.startswith("The command timed out after 60s and was killed.\nOutput:\npartial\n")
    assert "at most 600s" in result


def read(text: str, *, success: bool = True, first: int = 1, lines: int = 3) -> str:
    call = RuntimeCall("text_editor", {}, path="/testbed/a.py", first_line=first, lines=lines)
    return observation("read_file", call, ToolResult(success, text), elapsed_s=0)


def test_read_file_says_where_the_lines_are_and_where_to_read_next() -> None:
    assert read(view("one", "two", "three", "four")) == (
        "/testbed/a.py, lines 1-3 (the file goes on: read from offset=4):\n"
        + view("one", "two", "three").rstrip("\n")
    )
    assert read(view("one", "two", ""), lines=3) == (
        "/testbed/a.py, lines 1-2 (end of file):\n" + view("one", "two").rstrip("\n")
    )
    assert read("", first=50) == "/testbed/a.py has no line 50: the file is shorter."


def test_read_file_ending_exactly_at_the_files_last_line_is_the_end_of_file() -> None:
    # A three-line file with its final newline: view shows an empty fourth "line".
    assert read(view("one", "two", "three", ""), lines=3) == (
        "/testbed/a.py, lines 1-3 (end of file):\n" + view("one", "two", "three").rstrip("\n")
    )
    # A real empty line before more text is a line of the file.
    assert "(the file goes on: read from offset=4)" in read(view("one", "two", "three", "", "five"), lines=3)
    assert read(view(""), lines=3) == "/testbed/a.py is empty."


def test_read_file_cuts_one_line_that_is_longer_than_its_budget() -> None:
    result = read(view("x" * (READ_CHARS * 2)), lines=3)

    assert len(result) < READ_CHARS + 200 and "[... line cut at %d characters]" % READ_CHARS in result
    assert "(the file goes on: read from offset=2)" in result


def test_read_file_explains_a_missing_path_and_a_directory() -> None:
    assert "does not exist" in read("open /testbed/a.py: no such file or directory", success=False)
    assert "is a directory" in read("read /testbed/a.py: is a directory", success=False)


def edit(text: str, *, success: bool, tools: frozenset[str] = frozenset({"read_file", "write_file"})) -> str:
    call = RuntimeCall("text_editor", {}, path="/testbed/a.py")
    return observation("str_replace", call, ToolResult(success, text), elapsed_s=0, tools=tools)


def test_str_replace_passes_the_runtimes_snippet_through_and_explains_each_failure() -> None:
    assert edit("Edited /testbed/a.py: replaced line 2 (1 line) with 1 line:\n", success=True).startswith(
        "Edited"
    )
    assert "must match the file exactly" in edit("No match found", success=False)
    assert "occurs 2 times in /testbed/a.py (starting at lines 1 and 2)" in edit(
        "Multiple matches (2), starting at lines 1 and 2. old_str must be unique.", success=False
    )
    assert edit("old_str and new_str are identical; nothing to change", success=False) == (
        "No change made: old_string and new_string are identical."
    )
    assert "Create it with write_file" in edit("open /testbed/a.py: no such file or directory", success=False)


def test_failure_hints_name_only_the_tools_of_the_default_toolset() -> None:
    default = frozenset({"execute_bash", "str_replace", "submit"})

    missing = edit("open /testbed/a.py: no such file or directory", success=False, tools=default)
    absent = edit("No match found", success=False, tools=default)

    assert "write_file" not in missing and "execute_bash" in missing
    assert "read_file" not in absent and "sed -n" in absent

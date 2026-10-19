"""The interceptors that ship with Sprout, and the registry that builds a run's declared interceptors.

``ShellContext`` is checked by running its rewritten command in a real ``/bin/sh``, as the runtime
runs each ``shell`` call. The command line with a user is checked as text, because ``runuser`` needs
root.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from sprout.worker.tools.interceptors import MutationTracker, ShellContext, build_interceptors
from sprout.worker.tools.pipeline import CallContext, Reject, ToolInterceptor, ToolPipeline, Verdict
from sprout.worker.tools.result import ToolResult


def call(tool: str, **args: Any) -> CallContext:
    return CallContext(agent_id="agent", sandbox_id="sandbox", tool_name=tool, args=args)


def run_in_sh(tool: str, args: dict[str, Any]) -> ToolResult:
    """Run a ``shell`` call the way the runtime does: ``sh -c`` in the call's ``working_dir``."""
    done = subprocess.run(
        ["/bin/sh", "-c", args["command"]],
        cwd=args["working_dir"],
        capture_output=True,
        text=True,
        timeout=30,
    )
    return ToolResult(success=done.returncode == 0, output=done.stdout + done.stderr)


def received(tool: str, args: dict[str, Any]) -> ToolResult:
    """An executor that answers with the args that reached it."""
    return ToolResult(success=True, output=repr(args))


class RefuseAll(ToolInterceptor):
    def before(self, ctx: CallContext) -> Verdict:
        return Reject("refused")


# --- the registry --------------------------------------------------------------


def test_declared_interceptors_are_built_in_order_with_their_arguments():
    built = build_interceptors(
        [
            {"name": "shell_context", "cwd": "/app", "env": {"A": "1"}, "user": "nobody"},
            {"name": "shell_context", "cwd": "/srv"},
        ]
    )

    assert [(type(i), i.cwd, i.env, i.user) for i in built] == [
        (ShellContext, "/app", {"A": "1"}, "nobody"),
        (ShellContext, "/srv", {}, None),
    ]


def test_a_run_that_declares_nothing_gets_no_interceptor():
    assert build_interceptors(None) == []


@pytest.mark.parametrize(
    "spec", ["shell_context", {"cwd": "/app"}, {"name": 3}], ids=["bare-name", "no-name", "name-not-a-string"]
)
def test_a_declaration_that_is_not_a_named_mapping_is_refused(spec):
    with pytest.raises(ValueError, match=r'^an interceptor is declared as \{"name": \.\.\., \*\*kwargs\}$'):
        build_interceptors([spec])


def test_an_unknown_name_is_refused_with_the_names_that_a_run_can_declare():
    """The mutation tracker is not among them: the runner adds the one that the checkpointer reads."""
    with pytest.raises(ValueError, match=r"^unknown interceptor 'mutation'; known: shell_context$"):
        build_interceptors([{"name": "mutation"}])


def test_a_shell_context_without_a_working_directory_is_refused():
    with pytest.raises(ValueError, match="^shell_context needs the task working directory$"):
        build_interceptors([{"name": "shell_context", "cwd": ""}])


# --- ShellContext ----------------------------------------------------------------


def test_a_shell_call_runs_in_the_task_directory_with_the_task_environment(tmp_path):
    """A value with a space and a ``;`` stays one value: it does not become a second command."""
    pipeline = ToolPipeline([ShellContext(cwd=str(tmp_path), env={"GREETING": "a b; echo injected"})])

    result = pipeline.execute(call("shell", command='pwd; printf "%s\\n" "$GREETING"'), run_in_sh)

    assert result == ToolResult(success=True, output="%s\na b; echo injected\n" % tmp_path)


def test_the_calls_own_working_directory_wins_over_the_task_directory(tmp_path):
    (tmp_path / "repo").mkdir()
    pipeline = ToolPipeline([ShellContext(cwd=str(tmp_path))])

    result = pipeline.execute(call("shell", command="pwd", working_dir=str(tmp_path / "repo")), run_in_sh)

    assert result == ToolResult(success=True, output="%s\n" % (tmp_path / "repo"))


#: ``id -un`` in ``/app``, as the shell's own user.
IN_APP = """env /bin/bash -c 'cd /app && exec /bin/bash -c '"'"'id -un'"'"''"""


@pytest.mark.parametrize(
    "user, command",
    [
        ("app", 'runuser -u "$(getent passwd app | cut -d: -f1)" -- ' + IN_APP),
        (1000, 'runuser -u "$(getent passwd 1000 | cut -d: -f1)" -- ' + IN_APP),
        (
            "1000:1001",
            'runuser -u "$(getent passwd 1000 | cut -d: -f1)" -g "$(getent group 1001 | cut -d: -f1)" -- '
            + IN_APP,
        ),
        (None, IN_APP),
        ("root", IN_APP),
        (0, IN_APP),
    ],
    ids=["name", "uid", "uid-and-gid", "none", "root", "uid-0"],
)
def test_a_task_user_other_than_root_runs_the_command_through_runuser(user, command):
    """``runuser`` looks a user up by name only, so a uid or a gid goes through ``getent`` first."""
    pipeline = ToolPipeline([ShellContext(cwd="/app", user=user)])

    result = pipeline.execute(call("shell", command="id -un"), received)

    assert result.output == repr({"command": command, "working_dir": "/"})


def test_a_call_of_another_tool_passes_unchanged():
    pipeline = ToolPipeline([ShellContext(cwd="/app", user="app")])

    result = pipeline.execute(call("text_editor", command="view", path="notes.txt"), received)

    assert result.output == repr({"command": "view", "path": "notes.txt"})


def test_a_shell_call_that_cannot_be_rewritten_is_rejected():
    """A call that escaped the task's user and directory would run as root in ``/``."""
    pipeline = ToolPipeline([ShellContext(cwd="/app", user="app")])

    result = pipeline.execute(call("shell", working_dir="/tmp"), received)

    assert result == ToolResult(
        success=False,
        output="interceptor shell_context failed closed: 'command'",
        error="rejected by shell_context",
    )


# --- MutationTracker --------------------------------------------------------------


@pytest.mark.parametrize(
    "tool, args, dirty",
    [
        ("shell", {"command": "ls"}, True),
        ("text_editor", {"command": "str_replace", "path": "/a", "old_str": "x", "new_str": "y"}, True),
        ("text_editor", {"path": "/a"}, True),
        ("text_editor", {"command": "view", "path": "/a"}, False),
        ("grep_files", {"pattern": "x"}, False),
        ("web_fetch", {"url": "https://example.com"}, False),
        ("web_search", {"query": "x"}, False),
        ("a_tool_added_later", {}, True),
    ],
    ids=[
        "shell",
        "editor-edit",
        "editor-no-command",
        "editor-view",
        "grep",
        "fetch",
        "search",
        "unknown-tool",
    ],
)
def test_a_call_counts_as_a_change_unless_its_tool_only_reads(tool, args, dirty):
    """Any ``shell`` command counts, because ``ls`` reads and ``ls > out`` writes."""
    tracker = MutationTracker()

    ToolPipeline([tracker]).execute(call(tool, **args), received)

    assert tracker.dirty is dirty


def test_after_a_clear_a_call_that_only_reads_leaves_the_environment_unchanged():
    tracker = MutationTracker()
    pipeline = ToolPipeline([tracker])
    pipeline.execute(call("shell", command="touch a"), received)

    tracker.clear()
    pipeline.execute(call("grep_files", pattern="a"), received)

    assert tracker.dirty is False


def test_a_call_that_a_later_rule_rejects_still_counts_as_a_change():
    """A wrong "unchanged" loses the state of a step. A wrong "changed" costs one capture."""
    tracker = MutationTracker()

    result = ToolPipeline([tracker, RefuseAll()]).execute(call("shell", command="rm -rf build"), received)

    assert (result.success, tracker.dirty) == (False, True)

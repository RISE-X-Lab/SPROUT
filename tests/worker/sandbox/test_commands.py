"""``shell_outcome``, ``shell`` and ``checked``: one command through a session, with its outcome settled.

Each command runs through a real ``SandboxSession`` on a ``StubPool``. Its runtime answers with the
JSON-RPC result that the Go runtime sends, parsed by ``sprout_sandbox``.
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from sprout.worker.sandbox.commands import checked, shell, shell_outcome
from sprout.worker.sandbox.session import OWNER_AGENT_ID, SandboxSession
from sprout_sandbox import ToolResult as SdkToolResult
from tests.worker.sandbox.pools import StubPool, StubRuntime, created


def answering(
    use_pool: Any, command: str, text: str, *, is_error: bool = False
) -> tuple[SandboxSession, StubRuntime]:
    """A session whose runtime answers ``command`` with the reply text ``text``."""
    reply = SdkToolResult.from_response({"content": [{"type": "text", "text": text}], "isError": is_error})
    runtime = StubRuntime(answers={command: reply})
    use_pool(StubPool(runtimes={"sandbox-1": runtime}))
    return created(), runtime


def outcome(**fields: Any) -> str:
    """The text of the runtime's ``CommandOutcome`` for a command."""
    return json.dumps({"stdout": "", "stderr": "", **fields})


def test_the_command_runs_with_its_timeout(use_pool):
    session, runtime = answering(use_pool, "pytest", "ok\n")

    shell_outcome(session, "pytest", 5)

    assert runtime.calls == [("shell", {"command": "pytest", "timeout": 5}, OWNER_AGENT_ID)]


def test_a_command_that_exited_non_zero_reads_as_its_streams_and_exit_code(use_pool):
    session, _ = answering(use_pool, "make", outcome(stderr="boom\n", exit_code=2), is_error=True)

    assert (shell_outcome(session, "make"), shell(session, "make")) == (
        ("", "boom\n", 2, False),
        ("", "boom\n", 2),
    )


def test_the_bare_stdout_of_a_command_reads_as_exit_0(use_pool):
    """The runtime sends bare stdout for exit 0 with no stderr."""
    session, _ = answering(use_pool, "ls", "a.txt\n")

    assert shell_outcome(session, "ls") == ("a.txt\n", "", 0, False)


@pytest.mark.parametrize(
    "text, expected",
    [
        (
            outcome(stdout="collected 3 items\n", exit_code=137, timed_out=True),
            ("collected 3 items\n", "", 137, True),
        ),
        (outcome(stdout="collected 3 items\n", timed_out=True), ("collected 3 items\n", "", None, True)),
    ],
    ids=["killed-with-exit-code", "without-exit-code"],
)
def test_shell_outcome_answers_a_timeout(use_pool, text, expected):
    """A grader scores the tests that hung. It does not run them again."""
    session, _ = answering(use_pool, "pytest", text, is_error=True)

    assert shell_outcome(session, "pytest", 5) == expected


def test_shell_raises_for_a_timeout(use_pool):
    session, _ = answering(use_pool, "pytest", outcome(exit_code=137, timed_out=True), is_error=True)

    with pytest.raises(RuntimeError, match="^Sandbox command timed out after 5 s$"):
        shell(session, "pytest", 5)


@pytest.mark.parametrize(
    "text",
    [
        outcome(exit_code=0, stdout_truncated=True),
        outcome(exit_code=1, stderr_truncated=True),
        outcome(),
    ],
    ids=["stdout-clipped", "stderr-clipped", "no-exit-code"],
)
def test_an_outcome_that_is_not_known_in_full_raises(use_pool, text):
    session, _ = answering(use_pool, "pytest", text, is_error=True)

    with pytest.raises(RuntimeError, match="^Sandbox command incomplete or truncated$"):
        shell_outcome(session, "pytest")


def test_a_call_that_failed_raises_with_its_error(use_pool):
    use_pool(
        StubPool(runtimes={"sandbox-1": StubRuntime(answers={"ls": ConnectionResetError("reset by peer")})})
    )
    session = created()

    with pytest.raises(RuntimeError, match="^reset by peer$"):
        shell_outcome(session, "ls")


def test_checked_gives_the_stdout_of_a_command_that_exited_0(use_pool):
    session, _ = answering(use_pool, "pwd -P", "/testbed\n")

    assert checked(session, "pwd -P") == "/testbed\n"


@pytest.mark.parametrize(
    "streams, message",
    [
        ({"stdout": "out", "stderr": "boom"}, "Sandbox command exited 2: boom"),
        ({"stdout": "out", "stderr": ""}, "Sandbox command exited 2: out"),
        ({"stdout": "", "stderr": "x" * 500 + "y" * 2000}, "Sandbox command exited 2: " + "y" * 2000),
    ],
    ids=["stderr", "stdout-without-stderr", "the-last-2000-characters"],
)
def test_checked_raises_for_another_exit_with_the_end_of_what_the_command_said(use_pool, streams, message):
    session, _ = answering(use_pool, "make", json.dumps({**streams, "exit_code": 2}), is_error=True)

    with pytest.raises(RuntimeError) as raised:
        checked(session, "make")

    assert str(raised.value) == message


def test_a_command_whose_stdout_is_a_json_object_with_a_stdout_key_reads_as_its_stdout(use_pool):
    """Such as ``cat`` of a JSON report. The runtime sent bare stdout, so the command exited 0."""
    session, _ = answering(use_pool, "cat report.json", '{"stdout": "x", "passed": 3}')

    assert shell(session, "cat report.json") == ('{"stdout": "x", "passed": 3}', "", 0)

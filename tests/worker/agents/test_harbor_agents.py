"""The ``oracle`` and ``nop`` agents act in the sandbox the way Harbor's agents of the same names act.

Each slot runs through a real ``ToolAccess``. A stub sandbox session behind it answers each call and
records it, because the calls that reach the sandbox are what Harbor compatibility means.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.harbor_task import HarborTask, load_task
from sprout.worker.agents.nop import NopSlot
from sprout.worker.agents.oracle import OracleSlot
from sprout.worker.agents.slot import SlotResult, TaskSpec
from sprout.worker.run.journal import JournalWriter
from sprout.worker.sandbox.image_shell import ImageShell
from sprout.worker.tools.access import ToolAccess
from sprout.worker.tools.result import CommandOutcome, ToolResult

#: The line that Harbor's oracle agent runs, as root.
SOLVE = (
    "chmod +x /solution/solve.sh && mkdir -p /logs/agent && "
    "(DEBIAN_FRONTEND=noninteractive /solution/solve.sh) > /logs/agent/oracle.txt 2>&1"
)
#: What root runs first when the image's user runs ``solve.sh``.
PREPARE = "chmod +x /solution/solve.sh && mkdir -p /logs/agent && chmod 777 /logs/agent"
#: What the image's user runs.
RUN = "(DEBIAN_FRONTEND=noninteractive /solution/solve.sh) > /logs/agent/oracle.txt 2>&1"

Answer = Callable[[str, dict[str, Any]], ToolResult]


def exits(code: int, *, timed_out: bool = False) -> ToolResult:
    """The result of a shell command that exited with ``code``."""
    return ToolResult(code == 0, "", outcome=CommandOutcome(exit_code=code, timed_out=timed_out))


def shell_gives(result: ToolResult) -> Answer:
    """An answer that writes each file, and gives ``result`` for each shell command."""
    return lambda name, args: ToolResult(True, "File written") if name == "text_editor" else result


written = shell_gives(exits(0))


class StubSession:
    """A sandbox session that answers each call with ``answer`` and records ``(tool, args)``."""

    sandbox_id = "sandbox-fixture"

    def __init__(self, answer: Answer = written) -> None:
        self.answer = answer
        self.calls: list[tuple[str, dict[str, Any]]] = []

    def call(self, name: str, args: dict[str, Any], timeout: float = 300) -> ToolResult:
        self.calls.append((name, dict(args)))
        return self.answer(name, args)


def harbor_task(
    root: Path, *, solution: dict[str, bytes], agent_timeout_s: int | None = 60, workdir: str = "/app"
) -> HarborTask:
    """A Harbor task in ``root`` whose ``solution/`` holds ``solution``, by relative path."""
    agent = "[agent]\ntimeout_sec = %d\n" % agent_timeout_s if agent_timeout_s is not None else ""
    (root / "tests").mkdir(parents=True)
    (root / "task.toml").write_text(
        'schema_version = "1.4"\n%s[environment]\ndocker_image = "image:fixture"\nworkdir = "%s"\n'
        % (agent, workdir)
    )
    (root / "instruction.md").write_text("Write the answer.\n")
    (root / "tests" / "test.sh").write_text("exit 0\n")
    for name, data in solution.items():
        (root / "solution" / name).parent.mkdir(parents=True, exist_ok=True)
        (root / "solution" / name).write_bytes(data)
    return load_task(root)


def run_oracle(
    tmp_path: Path,
    *,
    session: StubSession,
    solution: dict[str, bytes] | None = None,
    timeout_s: float = 3600.0,
    image_shell: ImageShell | None = None,
    **task_options: Any,
) -> SlotResult:
    """Run the oracle of a task with ``solution`` (by default one ``solve.sh``) on ``session``."""
    task = harbor_task(
        tmp_path / "task",
        solution=solution or {"solve.sh": b"echo ready > /app/answer.txt\n"},
        **task_options,
    )
    with JournalWriter(tmp_path / "journal.jsonl") as journal:
        tools = ToolAccess(session, journal)
        spec = TaskSpec(prompt="", cwd="/app", timeout_s=timeout_s, image_shell=image_shell)
        return OracleSlot(task).run(spec, journal, tools)


def test_the_oracle_writes_each_solution_file_under_slash_solution_then_runs_solve_sh(tmp_path):
    session = StubSession()
    solution = {"solve.sh": b"python3 helper/fix.py\n", "helper/fix.py": b"print('fixed')\n"}

    run_oracle(tmp_path, session=session, solution=solution)

    assert session.calls == [
        (
            "text_editor",
            {"command": "write", "path": "/solution/helper/fix.py", "file_text": "print('fixed')\n"},
        ),
        (
            "text_editor",
            {"command": "write", "path": "/solution/solve.sh", "file_text": "python3 helper/fix.py\n"},
        ),
        ("shell", {"command": SOLVE, "timeout": 60, "working_dir": "/app"}),
    ]


@pytest.mark.parametrize("code", [0, 3])
def test_the_oracle_completes_whatever_solve_sh_exits_with(tmp_path, code):
    """The grade, not the exit code, decides the reward."""
    session = StubSession(shell_gives(exits(code)))

    result = run_oracle(tmp_path, session=session)

    assert result == SlotResult(status="completed", final_text="solve.sh exited %d" % code)


def test_a_solve_sh_that_outruns_its_timeout_times_the_run_out(tmp_path):
    session = StubSession(shell_gives(exits(137, timed_out=True)))

    result = run_oracle(tmp_path, session=session)

    assert result == SlotResult(status="timeout", error="solve.sh did not finish within 60 s")


@pytest.mark.parametrize(
    "agent_timeout_s, timeout_s, expected",
    [(60, 3600.0, 60), (60, 30.9, 30), (None, 30.0, 30), (60, 0.5, 1)],
    ids=["agent-limit", "run-limit", "no-agent-limit", "at-least-one-second"],
)
def test_solve_sh_gets_the_smaller_of_the_agent_and_run_timeouts_in_whole_seconds(
    tmp_path, agent_timeout_s, timeout_s, expected
):
    session = StubSession()

    run_oracle(tmp_path, session=session, agent_timeout_s=agent_timeout_s, timeout_s=timeout_s)

    assert session.calls[-1] == ("shell", {"command": SOLVE, "timeout": expected, "working_dir": "/app"})


def test_with_the_images_user_root_prepares_solve_sh_and_the_user_runs_it(tmp_path):
    session = StubSession()
    shell = ImageShell(user="app", env={"LANG": "C.UTF-8"}, workdir="/")

    run_oracle(tmp_path, session=session, image_shell=shell)

    assert session.calls[1:] == [
        ("shell", {"command": PREPARE, "timeout": 300}),
        ("shell", {"command": shell.command(RUN, working_dir="/app"), "timeout": 60, "working_dir": "/app"}),
    ]


def test_a_failed_preparation_raises_before_solve_sh_runs(tmp_path):
    session = StubSession(shell_gives(exits(1)))
    shell = ImageShell(user="app", env={}, workdir="/")

    with pytest.raises(RuntimeError, match="^could not prepare solve.sh"):
        run_oracle(tmp_path, session=session, image_shell=shell)

    assert session.calls[1:] == [("shell", {"command": PREPARE, "timeout": 300})]


def test_a_task_without_solve_sh_raises_before_any_tool_call(tmp_path):
    session = StubSession()

    with pytest.raises(ValueError, match="solve.sh: missing, and the oracle runs it$"):
        run_oracle(tmp_path, session=session, solution={"README": b"no script\n"})

    assert session.calls == []


def test_a_solution_file_that_is_not_utf8_text_raises(tmp_path):
    session = StubSession()

    with pytest.raises(ValueError, match="data.bin: the oracle writes text files only$"):
        run_oracle(tmp_path, session=session, solution={"solve.sh": b"true\n", "data.bin": b"\xff\xfe"})


def test_a_write_that_the_sandbox_refuses_raises(tmp_path):
    session = StubSession(lambda name, args: ToolResult(False, "", error="read-only file system"))

    with pytest.raises(RuntimeError, match="^could not write /solution/solve.sh: read-only file system$"):
        run_oracle(tmp_path, session=session)


def test_a_solve_sh_that_did_not_start_raises(tmp_path):
    """A shell result without a command outcome means that ``solve.sh`` never ran."""
    session = StubSession(shell_gives(ToolResult(False, "", "no shell")))

    with pytest.raises(RuntimeError, match="^solve.sh did not run: no shell$"):
        run_oracle(tmp_path, session=session)


def test_the_nop_agent_completes_without_a_tool_call(tmp_path):
    """A nop grade measures what the verifier gives an untouched sandbox."""
    session = StubSession()

    with JournalWriter(tmp_path / "journal.jsonl") as journal:
        result = NopSlot().run(
            TaskSpec(prompt="Write the answer.", cwd="/app"), journal, ToolAccess(session, journal)
        )

    assert (result, session.calls) == (SlotResult(status="completed"), [])

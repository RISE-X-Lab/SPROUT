"""The ``oracle`` agent: a Harbor task's reference solution, run the way Harbor's ``oracle`` agent runs it.

``OracleSlot`` writes each file of the task's ``solution/`` to ``/solution`` in the sandbox. Then it
runs ``/solution/solve.sh`` with its output in ``/logs/agent/oracle.txt``, within the task's agent
timeout and the run's ``timeout_s``. With the run's image shell (``TaskSpec.image_shell``), root
prepares ``solve.sh`` and ``/logs/agent`` first, and the image's user runs ``solve.sh``. Every effect
goes through the run's ``ToolAccess``, as one turn. The run completes whatever ``solve.sh`` exits
with, because the grade decides the reward.
"""

from __future__ import annotations

import shlex
from pathlib import Path
from typing import TYPE_CHECKING

from sprout.contract.harbor_task import HarborTask
from sprout.worker.agents.slot import AgentSlot, SlotResult, TaskSpec
from sprout.worker.run.journal import JournalWriter
from sprout.worker.sandbox.image_shell import ImageShell

if TYPE_CHECKING:
    from sprout.worker.tools.access import ToolAccess
    from sprout.worker.tools.result import ToolResult

#: Where Harbor puts a task's ``solution/`` in the sandbox.
_SOLUTION_DIR = "/solution"
#: Where Harbor's oracle writes the output of ``solve.sh``.
_ORACLE_LOG = "/logs/agent/oracle.txt"
_TURN = "oracle"
#: How Harbor's oracle runs ``solve.sh``: executable, non-interactive, with its output in ``_ORACLE_LOG``.
_SOLVE = "chmod +x {script} && mkdir -p {logs} && (DEBIAN_FRONTEND=noninteractive {script}) > {log} 2>&1"
# ``_SOLVE`` for the image's user: root runs ``_PREPARE``, and then the user runs ``_RUN``.
# ``_PREPARE`` leaves ``/logs/agent`` writable to all, as Harbor's environment does.
_PREPARE = "chmod +x {script} && mkdir -p {logs} && chmod 777 {logs}"
_RUN = "(DEBIAN_FRONTEND=noninteractive {script}) > {log} 2>&1"
_PATHS = {
    "script": shlex.quote(f"{_SOLUTION_DIR}/solve.sh"),
    "logs": shlex.quote(str(Path(_ORACLE_LOG).parent)),
    "log": shlex.quote(_ORACLE_LOG),
}
#: The seconds that the preparation may take. Arbitrary.
_PREPARE_TIMEOUT_S = 300


class OracleSlot(AgentSlot):
    """Runs the ``solution/solve.sh`` of one Harbor task."""

    name = "oracle"

    def __init__(self, task: HarborTask) -> None:
        self._task = task

    def run(self, task: TaskSpec, journal: JournalWriter, tools: ToolAccess) -> SlotResult:
        """Run the task's ``solve.sh``: ``timeout`` when it outruns its timeout, else ``completed``.

        Raises ``ValueError`` when ``solve.sh`` is missing or a solution file is not UTF-8 text.
        Raises ``RuntimeError`` when a write, the preparation or the start of ``solve.sh`` fails.
        """
        solve = self._task.solution_dir / "solve.sh"
        if not solve.is_file():
            raise ValueError(f"{solve}: missing, and the oracle runs it")
        for index, path in enumerate(sorted(p for p in self._task.solution_dir.rglob("*") if p.is_file())):
            _write(tools, path=path, root=self._task.solution_dir, call_id=f"oracle-write-{index}")
        timeout_s = int(min(self._task.agent_timeout_s or task.timeout_s, task.timeout_s))
        if task.image_shell is not None:
            _prepare(tools)
        command = _solve_command(task.image_shell, workdir=self._task.workdir)
        solve_timeout_s = max(timeout_s, 1)
        arguments: dict[str, str | int] = {"command": command, "timeout": solve_timeout_s}
        if self._task.workdir is not None:
            arguments["working_dir"] = self._task.workdir
        result = tools.call("shell", arguments, call_id="oracle-solve", turn_id=_TURN)
        tools.close_turn(_TURN)
        return _outcome(result, timeout_s=solve_timeout_s)


def _write(tools: ToolAccess, *, path: Path, root: Path, call_id: str) -> None:
    """Write the solution file ``path`` to the same place under ``_SOLUTION_DIR``.

    Raises ``ValueError`` for a file that is not UTF-8 text, and ``RuntimeError`` when the runtime
    refuses the write.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{path}: the oracle writes text files only") from error
    destination = f"{_SOLUTION_DIR}/{path.relative_to(root).as_posix()}"
    result = tools.call(
        "text_editor",
        {"command": "write", "path": destination, "file_text": text},
        call_id=call_id,
        turn_id=_TURN,
    )
    if not result.success:
        raise RuntimeError(f"could not write {destination}: {result.error or result.output}")


def _prepare(tools: ToolAccess) -> None:
    """Make ``solve.sh`` executable and ``/logs/agent`` writable, as root.

    Raises ``RuntimeError`` when that fails.
    """
    arguments = {"command": _PREPARE.format(**_PATHS), "timeout": _PREPARE_TIMEOUT_S}
    result = tools.call("shell", arguments, call_id="oracle-prepare", turn_id=_TURN)
    if result.outcome is None or result.outcome.exit_code != 0:
        raise RuntimeError(f"could not prepare solve.sh: {result.error or result.output}")


def _solve_command(shell: ImageShell | None, *, workdir: str | None) -> str:
    """The command that runs ``solve.sh``: ``_SOLVE`` as root without ``shell``, else ``_RUN`` as its user."""
    if shell is None:
        return _SOLVE.format(**_PATHS)
    return shell.command(_RUN.format(**_PATHS), working_dir=workdir)


def _outcome(result: ToolResult, *, timeout_s: int) -> SlotResult:
    """The run's result: ``timeout`` when ``solve.sh`` outran ``timeout_s``, else ``completed``.

    Raises ``RuntimeError`` when ``result`` has no command outcome, because ``solve.sh`` did not run.
    """
    outcome = result.outcome
    if outcome is None:
        raise RuntimeError(f"solve.sh did not run: {result.error or result.output}")
    if outcome.timed_out:
        return SlotResult(status="timeout", error=f"solve.sh did not finish within {timeout_s} s")
    return SlotResult(status="completed", final_text=f"solve.sh exited {outcome.exit_code}")

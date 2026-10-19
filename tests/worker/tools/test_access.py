"""``ToolAccess``: the agent's checked and journaled tool calls, per-turn checkpoints and the final snapshot.

``FakeSandbox`` stands in for a ``SandboxSession``. A ``write <name>`` command adds a file to its list,
and a snapshot keeps a copy of the list. The journal, the pipeline, the mutation tracker, the
checkpointer and the run control are the real ones.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.checkpoint.checkpointer import Checkpointer
from sprout.worker.run.control import RunAborted, RunControl
from sprout.worker.run.journal import JournalWriter, read_journal
from sprout.worker.tools.access import ToolAccess, ToolInfraError
from sprout.worker.tools.interceptors import MutationTracker, ShellContext
from sprout.worker.tools.pipeline import CallContext, Reject, ToolInterceptor, ToolPipeline, Verdict
from sprout.worker.tools.result import CommandOutcome, ToolResult
from sprout.worker.tools.schema import runtime_tools
from sprout_sandbox import Snapshot

#: The fields that ``JournalWriter`` adds to each event.
ENVELOPE = {"v", "ts", "seq", "run_id", "agent_id", "sandbox_id"}


@dataclass
class FakeSandbox:
    """Runs each call on a list of files, and keeps a copy of the list for each snapshot."""

    sandbox_id: str = "sandbox-1"
    #: The answer to a command, or the exception that its call raises. Other calls succeed with no output.
    answers: dict[str, ToolResult | Exception] = field(default_factory=dict)
    can_snapshot: bool = True
    #: The ``disk_only`` values whose snapshots fail.
    failing_snapshots: set[bool] = field(default_factory=set)
    #: What each snapshot reports as ``delta_empty``.
    delta_empty: bool | None = None
    files: list[str] = field(default_factory=list)
    #: ``(tool, args, timeout)`` of each call, in call order.
    calls: list[tuple[str, dict[str, Any], float]] = field(default_factory=list)
    #: The files and ``disk_only`` of each snapshot, by id.
    snapshots: dict[str, tuple[tuple[str, ...], bool]] = field(default_factory=dict)

    def supports_snapshot(self) -> bool:
        return self.can_snapshot

    def call(self, tool: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        self.calls.append((tool, dict(args), timeout))
        answer = self.answers.get(args.get("command", ""))
        if isinstance(answer, Exception):
            raise answer
        if answer is not None:
            return answer
        if tool == "shell" and args["command"].startswith("write "):
            self.files.append(args["command"].removeprefix("write "))
        return ToolResult(success=True, output="")

    def snapshot(self, name: str | None = None, disk_only: bool = True) -> Snapshot | None:
        if disk_only in self.failing_snapshots:
            return None
        snapshot_id = "snap-%d" % (len(self.snapshots) + 1)
        self.snapshots[snapshot_id] = (tuple(self.files), disk_only)
        return Snapshot(snapshot_id, disk_only=disk_only, delta_empty=self.delta_empty)

    def swap_sandbox(self, snapshot: Snapshot) -> bool:
        return False


@dataclass
class Rollout:
    """One rollout's tool access, with its sandbox, journal and run control."""

    access: ToolAccess
    sandbox: FakeSandbox
    journal: JournalWriter
    control: RunControl

    def events(self, *types: str) -> list[dict[str, Any]]:
        """The journal's events of ``types``, in order, without the envelope fields."""
        records = read_journal(self.journal.path)
        return [{k: v for k, v in r.items() if k not in ENVELOPE} for r in records if r["type"] in types]

    def values(self, event_type: str, key: str) -> list[Any]:
        """The ``key`` of each journal event of ``event_type``, in order."""
        return [event[key] for event in self.events(event_type)]

    def unsettled_call(self, command: str) -> None:
        """A ``shell`` call of turn ``t1`` whose outcome is unknown."""
        with pytest.raises(ToolInfraError):
            self.access.call("shell", {"command": command}, call_id="c1", turn_id="t1")

    def turn(self, turn_id: str, *commands: str, close: bool = True) -> list[ToolResult]:
        """One model response: a ``shell`` call for each command, then the agent closes the turn."""
        results = [
            self.access.call("shell", {"command": command}, call_id="%s-%d" % (turn_id, n), turn_id=turn_id)
            for n, command in enumerate(commands)
        ]
        if close:
            self.access.close_turn(turn_id)
        return results


def rollout(
    tmp_path: Path,
    sandbox: FakeSandbox | None = None,
    *,
    interceptors: list[ToolInterceptor] | None = None,
    checkpoints: bool = True,
    stoppable: bool = True,
) -> Rollout:
    """A rollout wired as the runner wires one: the mutation tracker first, then ``interceptors``."""
    sandbox = sandbox or FakeSandbox()
    journal = JournalWriter(tmp_path / "journal.jsonl")
    control = RunControl()
    tracker = MutationTracker()
    access = ToolAccess(
        sandbox,
        journal,
        pipeline=ToolPipeline([tracker, *(interceptors or [])]),
        control=control if stoppable else None,
        checkpointer=Checkpointer(session=sandbox, tracker=tracker) if checkpoints else None,
    )
    journal.emit("session.ref", native_session_id="native-1")
    return Rollout(access=access, sandbox=sandbox, journal=journal, control=control)


@pytest.fixture
def make_rollout(tmp_path: Path) -> Iterator[Callable[..., Rollout]]:
    """``rollout`` in ``tmp_path``. Each journal that it opens is closed after the test."""
    made: list[Rollout] = []

    def make(sandbox: FakeSandbox | None = None, **options: Any) -> Rollout:
        made.append(rollout(tmp_path, sandbox, **options))
        return made[-1]

    yield make
    for each in made:
        each.journal.close()


@pytest.fixture
def run(make_rollout: Callable[..., Rollout]) -> Rollout:
    return make_rollout()


class RefuseAll(ToolInterceptor):
    def before(self, ctx: CallContext) -> Verdict:
        return Reject("budget exhausted")


def failed_command(exit_code: int | None, **flags: bool) -> ToolResult:
    """The result of a command that exited ``exit_code``, or that the runtime did not settle."""
    return ToolResult(success=False, output="", outcome=CommandOutcome(exit_code=exit_code, **flags))


# --- the agent's functions -------------------------------------------------------


def test_the_agent_is_offered_the_runtimes_own_declaration(run):
    assert run.access.tools == runtime_tools()


def test_a_call_runs_on_the_sandbox_and_is_journaled_when_it_starts_and_when_it_finishes(run):
    """A failed command is an observation: its result has the command's outcome and no error."""
    run.sandbox.answers["make"] = ToolResult(
        success=False, output="build failed", outcome=CommandOutcome(exit_code=2, stderr="build failed")
    )

    result = run.access.call("shell", {"command": "make"}, call_id="c1", turn_id="t1")

    assert result == run.sandbox.answers["make"]
    assert run.events("tool.started", "tool.finished") == [
        {
            "type": "tool.started",
            "call_id": "c1",
            "turn_id": "t1",
            "name": "shell",
            "args": {"command": "make"},
            "step": 1,
        },
        {
            "type": "tool.finished",
            "call_id": "c1",
            "turn_id": "t1",
            "name": "shell",
            "status": "error",
            "output": "build failed",
            "outcome": {
                "exit_code": 2,
                "stdout": "",
                "stderr": "build failed",
                "running": False,
                "timed_out": False,
                "stdout_bytes": 0,
                "stderr_bytes": 0,
                "stdout_truncated": False,
                "stderr_truncated": False,
            },
        },
    ]


@pytest.mark.parametrize(
    "args, timeout",
    [({"command": "ls", "timeout": 5}, 65.0), ({"command": "ls"}, 360.0)],
    ids=["own", "default"],
)
def test_the_sandbox_gets_the_commands_timeout_plus_a_margin_for_the_round_trip(run, args, timeout):
    """Without its own timeout, a command gets the runtime's default of 300 s."""
    run.access.call("shell", args, call_id="c1", turn_id="t1")

    assert run.sandbox.calls == [("shell", args, timeout)]


@pytest.mark.parametrize(
    "tool, answer, outcome",
    [
        ("shell", ToolResult(True, "café\n"), CommandOutcome(exit_code=0, stdout="café\n", stdout_bytes=6)),
        ("shell", ToolResult(False, "", error="no route"), None),
        ("text_editor", ToolResult(True, "written"), None),
    ],
    ids=["shell-bare-stdout", "shell-failure", "other-tool"],
)
def test_the_bare_stdout_of_a_shell_success_reads_as_exit_0(make_rollout, tool, answer, outcome):
    """The runtime sends bare stdout for exit 0 with no stderr. The byte count is of UTF-8."""
    sandbox = FakeSandbox(answers={"x": answer})
    access = make_rollout(sandbox).access

    result = access.call(tool, {"command": "x"}, call_id="c1", turn_id="t1")

    assert result.outcome == outcome


@pytest.mark.parametrize(
    "name, args, refusal",
    [
        (
            "shell",
            {"command": "ls", "max_length": 20},
            "shell does not take max_length; it accepts command, max_output_bytes, timeout, working_dir",
        ),
        ("read_file", {"path": "/a"}, "unknown tool: read_file"),
    ],
    ids=["unknown-argument", "unknown-tool"],
)
def test_a_tool_or_an_argument_that_the_runtime_does_not_take_is_refused_without_running(
    run, name, args, refusal
):
    """The runtime ignores an argument that it does not take, so a model would believe in a bound it lacks."""
    result = run.access.call(name, args, call_id="c1", turn_id="t1")

    assert result == ToolResult(success=False, output="", error=refusal)
    assert run.sandbox.calls == []
    assert run.events("tool.rejected", "tool.started") == [
        {
            "type": "tool.rejected",
            "call_id": "c1",
            "turn_id": "t1",
            "name": name,
            "executed": False,
            "source": "schema",
            "feedback": refusal,
        }
    ]


def test_an_interceptor_refusal_reaches_the_agent_as_a_failed_result_and_nothing_runs(make_rollout):
    run = make_rollout(interceptors=[RefuseAll()])

    result = run.access.call("shell", {"command": "ls"}, call_id="c1", turn_id="t1")

    assert result == ToolResult(success=False, output="budget exhausted", error="rejected by RefuseAll")
    assert run.sandbox.calls == []


def test_the_sandbox_runs_the_rewritten_call_and_the_journal_keeps_what_the_agent_asked(make_rollout):
    run = make_rollout(interceptors=[ShellContext(cwd="/app")])

    run.access.call("shell", {"command": "ls"}, call_id="c1", turn_id="t1")

    assert run.sandbox.calls == [
        (
            "shell",
            {"command": "env /bin/bash -c 'cd /app && exec /bin/bash -c ls'", "working_dir": "/"},
            360.0,
        )
    ]
    assert run.events("tool.started")[0]["args"] == {"command": "ls"}


@pytest.mark.parametrize("call_id, turn_id", [("", "t1"), ("c1", "")], ids=["no-call-id", "no-turn-id"])
def test_a_call_needs_a_call_id_and_a_turn_id(run, call_id, turn_id):
    with pytest.raises(ValueError, match="^tool calls need a"):
        run.access.call("shell", {"command": "ls"}, call_id=call_id, turn_id=turn_id)

    assert run.sandbox.calls == []


def test_a_call_id_that_an_earlier_call_used_is_refused(run):
    run.access.call("shell", {"command": "ls"}, call_id="c1", turn_id="t1")

    with pytest.raises(ValueError, match="^tool calls need a unique call_id$"):
        run.access.call("shell", {"command": "pwd"}, call_id="c1", turn_id="t2")


def test_a_stopped_rollout_runs_no_further_call(run):
    run.control.request_stop("deadline passed")

    with pytest.raises(RunAborted, match="^deadline passed$"):
        run.access.call("shell", {"command": "ls"}, call_id="c1", turn_id="t1")

    assert run.sandbox.calls == []


def test_no_call_runs_after_finalize(run):
    run.access.finalize()

    with pytest.raises(ToolInfraError, match="^tool access is closed$"):
        run.access.call("shell", {"command": "ls"}, call_id="c1", turn_id="t1")


def test_calls_from_two_threads_run_one_at_a_time(run):
    """The first call holds the sandbox until the second thread has asked. The second must wait for it."""
    first_inside, second_asked, second_inside = threading.Event(), threading.Event(), threading.Event()
    order: list[str] = []

    def first_call(tool: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        order.append("first starts")
        first_inside.set()
        second_asked.wait(10)
        # Fails the test if the second call reaches the sandbox while this one runs.
        second_inside.wait(0.2)
        order.append("first ends")
        return ToolResult(True, "")

    def second_call(tool: str, args: dict[str, Any], timeout: float = 300.0) -> ToolResult:
        second_inside.set()
        order.append("second runs")
        return ToolResult(True, "")

    calls = iter([first_call, second_call])
    run.sandbox.call = lambda tool, args, timeout=300.0: next(calls)(tool, args, timeout)
    first = threading.Thread(
        target=run.access.call, args=("shell", {"command": "a"}), kwargs={"call_id": "c1", "turn_id": "t1"}
    )
    first.start()
    first_inside.wait(10)

    second = threading.Thread(
        target=run.access.call, args=("shell", {"command": "b"}), kwargs={"call_id": "c2", "turn_id": "t1"}
    )
    second.start()
    second_asked.set()
    first.join(10)
    second.join(10)

    assert order == ["first starts", "first ends", "second runs"]


# --- calls whose outcome is unknown -------------------------------------------------------


@pytest.mark.parametrize(
    "answer, detail",
    [
        (
            TimeoutError("read timed out"),
            {"kind": "transport_exception", "exception": "TimeoutError", "message": "read timed out"},
        ),
        (
            failed_command(None, running=True),
            {"kind": "runtime_still_running", "running": True, "timed_out": False, "exit_code": None},
        ),
        (
            failed_command(None, timed_out=True),
            {"kind": "runtime_timeout", "running": False, "timed_out": True, "exit_code": None},
        ),
    ],
    ids=["transport-failure", "still-running", "timed-out-without-exit-code"],
)
def test_a_call_whose_outcome_is_unknown_stops_the_rollout(make_rollout, answer, detail):
    """Nothing after such a call pairs with a known filesystem."""
    run = make_rollout(FakeSandbox(answers={"pytest": answer}))
    run.turn("t1", "write a")

    with pytest.raises(ToolInfraError, match="^tool execution did not settle: "):
        run.access.call("shell", {"command": "pytest"}, call_id="c2", turn_id="t2")

    assert run.events("execution.uncertain") == [
        {"type": "execution.uncertain", "call_id": "c2", "step": 2, **detail}
    ]
    assert run.control.reason == "execution uncertain at step 2 (call c2)"


@pytest.mark.parametrize(
    "answer",
    [failed_command(3), failed_command(137, timed_out=True)],
    ids=["non-zero-exit", "timed-out-with-exit-code"],
)
def test_a_command_that_failed_or_timed_out_with_an_exit_code_is_an_observation(make_rollout, answer):
    """A grader scores a hung test as failed. A timeout that the runtime settled does not stop the rollout."""
    run = make_rollout(FakeSandbox(answers={"pytest": answer}))

    result = run.access.call("shell", {"command": "pytest"}, call_id="c1", turn_id="t1")

    assert (result, run.control.reason, run.events("execution.uncertain")) == (answer, None, [])


def test_after_a_call_whose_outcome_is_unknown_each_call_is_refused_without_running(make_rollout):
    """Also without a run control whose stop request would refuse the call first."""
    run = make_rollout(FakeSandbox(answers={"pytest": TimeoutError("read timed out")}), stoppable=False)
    run.unsettled_call("pytest")

    with pytest.raises(
        ToolInfraError, match="^an earlier call did not settle; the sandbox state is unknown$"
    ):
        run.access.call("shell", {"command": "ls"}, call_id="c2", turn_id="t2")

    assert run.sandbox.calls == [("shell", {"command": "pytest"}, 360.0)]


# --- per-turn checkpoints -------------------------------------------------------------


def test_a_turn_is_captured_when_the_next_turn_starts_and_before_its_first_call_runs(run):
    """Only a model request ran in between, so the snapshot holds the disk as the turn left it."""
    run.turn("t1", "write a", "write b")

    run.turn("t2", "write c")

    assert run.events("checkpoint.captured") == [
        {
            "type": "checkpoint.captured",
            "step": 2,
            "turn_id": "t1",
            "turn_closed": True,
            "call_ids": ["t1-0", "t1-1"],
            "snapshot_id": "snap-1",
            "session_ckpt": "native-1",
            "captured": True,
            "reason": "captured",
            "disk_only": True,
            "delta_empty": None,
            "reboarded": False,
        }
    ]
    assert run.sandbox.snapshots["snap-1"] == (("a", "b"), True)


def turn_capture(**fields: Any) -> dict[str, Any]:
    """The journal event of the capture of the closed turn ``t2``, step 2, as of ``snap-1``."""
    event = {
        "type": "checkpoint.captured",
        "step": 2,
        "turn_id": "t2",
        "turn_closed": True,
        "call_ids": ["t2-0"],
        "snapshot_id": "snap-1",
        "session_ckpt": "native-1",
        "captured": True,
        "reason": "captured",
        "disk_only": True,
        "delta_empty": None,
        "reboarded": False,
    }
    return {**event, **fields}


def test_a_turn_that_only_read_is_journaled_as_a_reuse_of_the_last_snapshot(run):
    """The runner's tracker feeds the checkpointer, so a search costs no capture."""
    run.turn("t1", "write a")
    run.access.call("grep_files", {"pattern": "a"}, call_id="t2-0", turn_id="t2")
    run.access.close_turn("t2")

    run.turn("t3", "write b")

    assert run.events("checkpoint.captured") == [
        turn_capture(step=1, turn_id="t1", call_ids=["t1-0"]),
        turn_capture(captured=False, reason="clean"),
    ]


def test_a_failed_capture_is_journaled_as_failed_with_the_last_snapshot(run):
    run.turn("t1", "write a")
    run.turn("t2", "write b")
    run.sandbox.failing_snapshots = {True}

    run.turn("t3", "write c")

    assert run.events("checkpoint.captured")[-1] == turn_capture(captured=False, reason="failed")


@pytest.mark.parametrize("delta_empty", [True, False])
def test_whether_a_snapshot_holds_no_change_reaches_the_journal(make_rollout, delta_empty):
    """A snapshot that holds no change costs less, so the journal says which one it is."""
    run = make_rollout(FakeSandbox(delta_empty=delta_empty))
    run.turn("t1", "write a")
    run.turn("t2", "write b")

    run.access.finalize()

    assert run.values("checkpoint.captured", "delta_empty") == [delta_empty, delta_empty]


def test_a_turn_whose_every_call_was_refused_is_not_captured(run):
    """It has no step that a checkpoint could name."""
    run.access.call("read_file", {"path": "/a"}, call_id="t1-0", turn_id="t1")

    run.turn("t2", "write a")

    assert run.events("checkpoint.captured") == []


def test_without_a_checkpointer_no_turn_is_captured_and_the_final_snapshot_is_still_taken(make_rollout):
    run = make_rollout(checkpoints=False)
    run.turn("t1", "write a")
    run.turn("t2", "write b")

    final = run.access.finalize()

    assert run.events("checkpoint.captured") == [{"type": "checkpoint.captured", **final}]


def test_a_turn_cannot_be_closed_before_a_call_of_it_ran(run):
    with pytest.raises(ValueError, match="^close_turn names a turn with no executed calls$"):
        run.access.close_turn("t1")


def test_a_turn_cannot_be_closed_after_the_next_turn_started(run):
    run.turn("t1", "ls", close=False)
    run.turn("t2", "ls", close=False)

    with pytest.raises(ValueError, match="^close_turn names a turn with no executed calls$"):
        run.access.close_turn("t1")


# --- the end of the rollout -------------------------------------------------------------


def final_record(**fields: Any) -> dict[str, Any]:
    """The final checkpoint record of a rollout whose last call is step 1 of the closed turn ``t1``."""
    record = {
        "step": 1,
        "turn_id": "t1",
        "turn_closed": True,
        "snapshot_id": "snap-1",
        "session_ckpt": "native-1",
        "captured": True,
        "reason": "captured",
        "disk_only": True,
        "final": True,
        "delta_empty": None,
    }
    return {**record, **fields}


def test_finalize_takes_the_final_disk_snapshot_of_the_last_turn_and_journals_it(run):
    """The last turn is open here: the snapshot is graded, but it is no branch point."""
    run.turn("t1", "write a")
    run.turn("t2", "write b", close=False)

    final = run.access.finalize()

    assert final == final_record(step=2, turn_id="t2", turn_closed=False, snapshot_id="snap-2")
    assert run.events("checkpoint.captured")[-1] == {"type": "checkpoint.captured", **final}
    assert run.sandbox.snapshots["snap-2"] == (("a", "b"), True)


def test_a_rollout_without_a_call_ends_with_a_final_snapshot_at_step_0(run):
    """An agent such as ``nop`` calls no tool."""
    final = run.access.finalize()

    assert final == final_record(step=0, turn_id=None, turn_closed=False)


def test_a_failed_final_snapshot_is_journaled_and_gives_no_record(make_rollout):
    run = make_rollout(FakeSandbox(failing_snapshots={True}))
    run.turn("t1", "write a")

    final = run.access.finalize()

    assert final is None
    assert run.events("checkpoint.captured") == [
        {"type": "checkpoint.captured", **final_record(snapshot_id=None, captured=False, reason="failed")}
    ]


def test_a_sandbox_that_cannot_snapshot_gives_no_final_record(make_rollout):
    run = make_rollout(FakeSandbox(can_snapshot=False))
    run.turn("t1", "write a")

    final = run.access.finalize()

    assert (final, run.events("checkpoint.captured")) == (None, [])


def test_the_final_snapshot_after_a_call_whose_outcome_is_unknown_says_so(make_rollout):
    run = make_rollout(FakeSandbox(answers={"pytest": TimeoutError("read timed out")}), stoppable=False)
    run.unsettled_call("pytest")

    final = run.access.finalize()

    assert final == final_record(turn_closed=False, uncertain=True)


def test_a_memory_snapshot_follows_the_final_disk_snapshot_when_asked(run):
    """A grader can need the sandbox's processes. The journal keeps the memory snapshot apart from
    the disk record, which stays the branch point."""
    run.turn("t1", "write a")

    final = run.access.finalize(memory=True)

    assert final == final_record(memory_snapshot_id="snap-2")
    assert run.sandbox.snapshots == {"snap-1": (("a",), True), "snap-2": (("a",), False)}
    assert run.events("checkpoint.captured", "snapshot.memory") == [
        {"type": "checkpoint.captured", **final_record()},
        {
            "type": "snapshot.memory",
            "step": 1,
            "turn_id": "t1",
            "snapshot_id": "snap-2",
            "captured": True,
            "after": "snap-1",
        },
    ]


def test_a_failed_memory_snapshot_is_journaled_and_the_disk_record_stands(make_rollout):
    run = make_rollout(FakeSandbox(failing_snapshots={False}))
    run.turn("t1", "write a")

    final = run.access.finalize(memory=True)

    assert final == final_record(memory_snapshot_id=None)
    assert run.events("snapshot.memory") == [
        {
            "type": "snapshot.memory",
            "step": 1,
            "turn_id": "t1",
            "snapshot_id": None,
            "captured": False,
            "after": "snap-1",
        }
    ]

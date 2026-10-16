"""ToolAccess: the agent's functions on its sandbox, and the per-turn checkpoints.

A fake session stands in for the sandbox: it records the commands it ran and
what the filesystem held at each snapshot. Everything else -- the schema check,
the pipeline, the checkpointer, the journal -- is the real code.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from sprout_sandbox.pool import Snapshot
from sprout.harness_checkpoint.checkpointer import Checkpointer
from sprout.harness_checkpoint.rollback import branch_checkpoints, uncaptured_turns
from sprout.harness.core.control import RunAborted, RunControl
from sprout.harness.core.journal import JournalWriter, read_journal
from sprout.harness.core.result import CommandOutcome, ToolResult
from sprout.harness.tools.access import ToolAccess, ToolInfraError
from sprout.harness.tools.interceptors import MutationTracker
from sprout.harness.tools.pipeline import Reject, ToolInterceptor, ToolPipeline


class Session:
    sandbox_id = "sandbox"

    def __init__(self):
        self.state, self.snapshots, self.ran = [], {}, []
        self.fail_snapshot = False
        self.raise_on = None
        self.unsettled_on = None   # (command, the CommandOutcome it returns without settling)
        self.delta_empty = None

    def supports_snapshot(self):
        return True

    def call(self, name, args, timeout=300):
        command = args.get("command", "")
        self.ran.append(command if name == "shell" else "%s %s" % (name, command))
        if command == self.raise_on:
            raise TimeoutError("synthetic transport failure")
        if self.unsettled_on and command == self.unsettled_on[0]:
            return ToolResult(False, "", outcome=self.unsettled_on[1])
        if name == "shell" and command.startswith("write "):
            self.state.append(command[6:])
        code = 3 if command == "fail" else 0
        return ToolResult(code == 0, "", outcome=CommandOutcome(exit_code=code, stdout="ok"))

    def snapshot(self, disk_only=True, name=None):
        if self.fail_snapshot:
            return None
        sid = "snap-%d" % (len(self.snapshots) + 1)
        self.snapshots[sid] = (list(self.state), disk_only)
        return Snapshot(sid, delta_empty=self.delta_empty)


@pytest.fixture
def env(tmp_path):
    journal = JournalWriter(tmp_path / "trajectory.jsonl")
    session = Session()
    tracker = MutationTracker()
    control = RunControl()
    access = ToolAccess(session, journal, pipeline=ToolPipeline([tracker]), control=control,
                        checkpointer=Checkpointer(session=session, tracker=tracker))
    journal.emit("session.ref", native_session_id="native")
    yield SimpleNamespace(journal=journal, session=session, access=access, control=control,
                          path=journal.path)
    journal.close()


def turn(env, turn_id, *commands, close=True):
    """One model response: its calls, then the agent closes the turn. A call is a
    shell command line, or a ``(tool, args)`` pair for another runtime tool."""
    results = []
    for index, command in enumerate(commands):
        name, args = command if isinstance(command, tuple) else (
            "shell", {"command": command, "working_dir": "/", "timeout": 5})
        results.append(env.access.call(name, args, call_id="%s-%d" % (turn_id, index),
                                       turn_id=turn_id))
    if close:
        calls = ["%s-%d" % (turn_id, i) for i in range(len(commands))]
        env.journal.emit("model.turn.completed", turn_id=turn_id, call_ids=calls,
                         step=env.journal.tool_calls()[-1]["step"])
        env.access.close_turn(turn_id)
    return results


def captures(env):
    return [r for r in read_journal(env.path) if r["type"] == "checkpoint.captured"]


def test_a_turn_is_captured_when_the_next_turn_starts_and_the_last_by_the_final_snapshot(env):
    turn(env, "t1", "write a", "write b")
    assert captures(env) == [], "nothing is captured until the next turn begins"
    turn(env, "t2", "write c")
    final = env.access.finalize()
    records = captures(env)
    assert [(r["step"], r["turn_id"], r.get("final", False)) for r in records] == \
        [(2, "t1", False), (3, "t2", True)]
    assert env.session.snapshots[records[0]["snapshot_id"]] == (["a", "b"], True)
    assert env.session.snapshots[final["snapshot_id"]] == (["a", "b", "c"], True), \
        "the final snapshot is disk-only unless the benchmark asks for memory"
    assert "memory_snapshot_id" not in final
    points = branch_checkpoints(env.path)
    assert set(points) == {2, 3} and points[3].final and points[2].session_ckpt == "native"
    assert points[2].snapshot_id == records[0]["snapshot_id"]
    assert uncaptured_turns(env.path) == []


def test_a_failed_capture_is_recorded_and_is_never_a_branch_point(env):
    turn(env, "t1", "write a")
    env.session.fail_snapshot = True
    turn(env, "t2", "write b")
    env.session.fail_snapshot = False
    turn(env, "t3", "write c")
    env.access.finalize()
    by_turn = {r["turn_id"]: r for r in captures(env)}
    assert by_turn["t1"]["reason"] == "failed"
    assert set(branch_checkpoints(env.path)) == {2, 3}


def test_a_turn_that_only_read_reuses_the_previous_snapshot_and_stays_a_branch_point(env):
    """A search, a fetch or a ``text_editor view`` says by name that it changed
    nothing; an edit does not, and neither can any shell command."""
    turn(env, "t1", "write a")
    turn(env, "t2", ("grep_files", {"pattern": "a"}), ("text_editor", {"command": "view", "path": "/a"}),
         ("web_fetch", {"url": "http://example.com"}), ("web_search", {"query": "a"}))
    turn(env, "t3", ("text_editor", {"command": "write", "path": "/b", "file_text": "b"}))
    turn(env, "t4", "ls")
    by_turn = {r["turn_id"]: r for r in captures(env)}
    assert by_turn["t2"]["reason"] == "clean" and by_turn["t2"]["captured"] is False
    assert by_turn["t2"]["snapshot_id"] == by_turn["t1"]["snapshot_id"]
    assert by_turn["t3"]["reason"] == "captured"
    assert by_turn["t3"]["snapshot_id"] != by_turn["t1"]["snapshot_id"]
    points = branch_checkpoints(env.path)
    assert set(points) == {1, 5, 6} and points[5].snapshot_id == by_turn["t1"]["snapshot_id"]


def test_an_unclosed_last_turn_is_graded_but_not_branched_from(env):
    turn(env, "t1", "write a")
    turn(env, "t2", "write b", close=False)   # e.g. the deadline hit mid-response
    final = env.access.finalize()
    assert final["turn_id"] == "t2" and final["turn_closed"] is False
    assert env.session.snapshots[final["snapshot_id"]][0] == ["a", "b"]
    assert set(branch_checkpoints(env.path)) == {1}


@pytest.mark.parametrize("kind,running,timed_out", [
    ("transport_exception", None, None),      # the transport raised: no outcome came back at all
    ("runtime_still_running", True, False),   # the runtime answered, but the command is still going
    ("runtime_timeout", False, True),         # ... or was killed without an exit code
])
def test_uncertain_execution_stops_the_rollout_and_refuses_further_calls(env, kind, running, timed_out):
    """A command whose outcome is unknown, however it got that way, is journaled
    as uncertain and ends the rollout; a timeout WITH an exit code is settled
    (next test)."""
    if kind == "transport_exception":
        env.session.raise_on = "hang"
    else:
        env.session.unsettled_on = ("hang", CommandOutcome(running=running, timed_out=timed_out,
                                                           exit_code=None))
    with pytest.raises(ToolInfraError):
        turn(env, "t1", "write a", "hang")
    assert env.control.reason and "execution uncertain" in env.control.reason
    uncertain = [r for r in read_journal(env.path) if r["type"] == "execution.uncertain"]
    assert uncertain[0]["kind"] == kind and uncertain[0]["step"] == 2
    with pytest.raises((ToolInfraError, RunAborted)):
        turn(env, "t2", "write b")
    assert env.session.ran == ["write a", "hang"]
    final = env.access.finalize()
    assert final["uncertain"] and branch_checkpoints(env.path) == {}


def test_a_failing_command_or_a_settled_timeout_is_an_observation(env):
    [failed] = turn(env, "t1", "fail")
    assert failed.outcome.exit_code == 3 and env.control.reason is None
    env.session.call = lambda name, args, timeout=300: ToolResult(False, "", outcome=CommandOutcome(
        timed_out=True, exit_code=137, stdout="partial"))
    [timed_out] = turn(env, "t2", "slow")
    assert timed_out.outcome.timed_out and env.control.reason is None


@pytest.mark.parametrize("name,args,mentions", [
    ("shell", {"command": "ls", "max_length": 20}, ["max_length", "command"]),
    ("read_file", {"path": "/a"}, ["unknown tool"]),
])
def test_an_argument_the_runtime_does_not_take_is_refused_without_running(env, name, args, mentions):
    """The Go tools read arguments by name and ignore the rest, so an unchecked
    `max_length` would run unbounded while the model believed it had a bound;
    a tool the runtime does not have is refused the same way."""
    result = env.access.call(name, args, call_id="c1", turn_id="t1")
    assert not result.success and all(text in result.error for text in mentions)
    assert env.session.ran == []
    rejected = [r for r in read_journal(env.path) if r["type"] == "tool.rejected"]
    assert rejected[0]["executed"] is False and rejected[0]["source"] == "schema"


def test_an_interceptor_refusal_reaches_the_agent_as_a_failed_result(env):
    class Refuse(ToolInterceptor):
        def before(self, ctx):
            return Reject("budget exhausted")

    env.access.pipeline = ToolPipeline([Refuse()])
    [result] = turn(env, "t1", "write a")
    assert not result.success and "budget exhausted" in result.output + (result.error or "")
    assert env.session.ran == []


def test_call_ids_are_unique_and_tools_close_at_finalize(env):
    turn(env, "t1", "write a")
    with pytest.raises(ValueError, match="unique"):
        env.access.call("shell", {"command": "ls", "working_dir": "/", "timeout": 5},
                        call_id="t1-0", turn_id="t2")
    with pytest.raises(ValueError, match="close_turn"):
        env.access.close_turn("t9")
    env.access.finalize()
    with pytest.raises(ToolInfraError, match="closed"):
        turn(env, "t3", "write b")


def test_a_memory_snapshot_follows_the_final_disk_one_when_asked(env):
    """A grader that runs inside the VM may need its processes; the disk record
    stays the branch point, the memory one is journaled apart from it."""
    turn(env, "t1", "write a")
    final = env.access.finalize(memory=True)
    disk, memory = final["snapshot_id"], final["memory_snapshot_id"]
    assert env.session.snapshots[disk] == (["a"], True)
    assert env.session.snapshots[memory] == (["a"], False)
    records = read_journal(env.path)
    [mem] = [r for r in records if r["type"] == "snapshot.memory"]
    assert mem["snapshot_id"] == memory and mem["after"] == disk and mem["captured"] is True
    assert [r["snapshot_id"] for r in records if r["type"] == "checkpoint.captured" and r.get("final")] == [disk]
    assert branch_checkpoints(env.path)[1].snapshot_id == disk, "memory snapshots are never branch points"


def test_a_failed_memory_snapshot_is_recorded_but_the_disk_record_stands(env):
    turn(env, "t1", "write a")
    disk_snapshot = env.session.snapshot

    def flaky(disk_only=True, name=None):
        return disk_snapshot(disk_only=disk_only, name=name) if disk_only else None

    env.session.snapshot = flaky
    final = env.access.finalize(memory=True)
    assert final["snapshot_id"] and final["memory_snapshot_id"] is None
    [mem] = [r for r in read_journal(env.path) if r["type"] == "snapshot.memory"]
    assert mem["captured"] is False and mem["snapshot_id"] is None


@pytest.mark.parametrize("flag", [True, False, None])
def test_the_delta_empty_measurement_reaches_the_journal(env, flag):
    env.session.delta_empty = flag
    turn(env, "t1", "write a")
    turn(env, "t2", "write b")
    assert captures(env)[0]["delta_empty"] is flag


def test_skipped_and_failed_captures_do_not_inherit_a_previous_measurement():
    tracker = MutationTracker()
    session = SimpleNamespace(supports_snapshot=lambda: True,
                              snapshot=lambda **kwargs: Snapshot("s", delta_empty=True))
    checkpointer = Checkpointer(session, tracker=tracker)
    assert checkpointer.after_step(1).delta_empty is True
    clean = checkpointer.after_step(2)
    assert clean.reason == "clean" and clean.delta_empty is None
    tracker.before(SimpleNamespace(tool_name="shell", args={"command": "write"}))
    session.snapshot = lambda **kwargs: None
    failed = checkpointer.after_step(3)
    assert failed.reason == "failed" and failed.delta_empty is None

# ruff: noqa: E402
"""Single2 execution, exact rollback, and accounting across native branches."""

import asyncio
import json
import subprocess
import threading
import time
from uuid import uuid4

import pytest

pytest.importorskip("opencollab")
from opencollab.adapters.llm.types import LLMResponse, Usage
from opencollab.application.session_run import GenerationTimeoutError

from sprout.contract.agents.history import load_prefix, training_messages, validate_assistant_turn
from sprout.worker.agents.history import actor_tools_at, available_branch_points, reference_at
from sprout.worker.agents.opencollab.environment import SproutEnvironment
from sprout.worker.agents.opencollab.slot import OpenCollabSlot
from sprout.worker.agents.slot import TaskSpec
from sprout.worker.checkpoint.checkpointer import Checkpointer
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter, read_journal
from sprout.worker.tools.access import ToolAccess
from sprout.worker.tools.interceptors import MutationTracker
from sprout.worker.tools.pipeline import ToolPipeline
from sprout.worker.tools.result import CommandOutcome, ToolResult
from sprout_sandbox.pool import Snapshot


def call(identifier, name, **args):
    return {"id": identifier, "type": "function", "function": {"name": name, "arguments": json.dumps(args)}}


class Model:
    def context_window(self):
        return 128000

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []
        self.closed = False

    async def complete(self, messages, **kwargs):
        self.requests.append((json.loads(json.dumps(messages)), kwargs))
        return next(self.responses)

    async def close(self):
        self.closed = True


def response(*calls, content="Testing a concrete change."):
    return LLMResponse(
        content=content,
        tool_calls=list(calls),
        finish_reason="tool_calls" if calls else "stop",
        usage=Usage(
            input_tokens=10, output_tokens=5, raw_usage={"prompt_tokens": 10, "completion_tokens": 5}
        ),
    )


class Sandbox:
    """Only a controlled test fixture; does not claim live VM verification."""

    def __init__(self, root):
        self.root = root
        self.sandbox_id = uuid4().hex
        self.snapshots = {}
        self.failure = False

    def supports_snapshot(self):
        return True

    def call(self, name, args, timeout):
        if self.failure:
            raise ConnectionError("synthetic lost execution transport")
        result = subprocess.run(
            args["command"],
            shell=True,
            cwd=args["working_dir"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
        return ToolResult(
            result.returncode == 0,
            result.stdout + result.stderr,
            outcome=CommandOutcome(exit_code=result.returncode, stdout=result.stdout, stderr=result.stderr),
        )

    def snapshot(self, **kwargs):
        identifier = str(uuid4())
        self.snapshots[identifier] = {
            path.name: path.read_bytes() for path in self.root.iterdir() if path.is_file()
        }
        return Snapshot(identifier)


def run(
    root,
    logdir,
    model,
    *,
    prefix=None,
    reviewer=None,
    point_only=False,
    sandbox=None,
    timeout=30,
    protocol="single2",
):
    logdir.mkdir()
    sandbox = sandbox or Sandbox(root)
    with JournalWriter(logdir / "trajectory.jsonl") as journal:
        control, tracker = RunControl(), MutationTracker()
        tools = ToolAccess(
            sandbox,
            journal,
            control=control,
            pipeline=ToolPipeline([tracker]),
            checkpointer=Checkpointer(session=sandbox, tracker=tracker),
        )
        extra = {
            "mini_workspace": {"cwd": str(root), "repository_dir": str(root)},
            "native_home": str(logdir / "native-home"),
            "opencollab_llm": model,
            "opencollab_tool_protocol": protocol,
        }
        if prefix:
            extra["native_prefix"] = prefix
        if reviewer:
            extra["assistant_turn"] = reviewer
        if point_only:
            extra["resume_without_hint"] = True
        task = TaskSpec(
            "Fix the fixture.",
            str(logdir),
            model="fixture-model",
            timeout_s=timeout,
            control=control,
            extra=extra,
        )
        result = OpenCollabSlot().run(task, journal, tools)
        tools.finalize()
    return result, sandbox, logdir / "trajectory.jsonl"


def test_native_six_tools_closed_turns_and_reviewer_branch(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    model = Model(
        [
            response(
                call("a", "file_write", path="fixture.txt", content="parent", mode="create"),
                call("b", "file_read", path="fixture.txt"),
            ),
            response(call("c", "bash", command="printf later > fixture.txt")),
            response(content="Done."),
        ]
    )
    result, sandbox, journal = run(root, tmp_path / "parent", model)
    assert result.status == "completed", result.error
    assert result.usage["input_tokens"] == 30
    assert model.closed
    assert [schema["function"]["name"] for schema in model.requests[0][1]["tools"]] == [
        "bash",
        "file_read",
        "file_write",
        "apply_patch",
        "git_diff",
        "grep",
    ]
    assert model.requests[0][1]["reasoning_effort"] == "high"
    points = available_branch_points(journal)
    assert len(points) == 2
    step, point = min(points.items())
    prefix = reference_at(journal, step, point.session_ckpt)
    entries = load_prefix(prefix)
    retained = training_messages(entries)
    assert retained[-1]["role"] == "tool"
    assert "parent" in retained[-1]["content"]
    assert not any("printf later" in str(message) for message in retained)
    # Restore this test fixture's disk BEFORE continuing from the exact state.
    for path in root.iterdir():
        if path.is_file():
            path.unlink()
    for name, contents in sandbox.snapshots[point.snapshot_id].items():
        (root / name).write_bytes(contents)
    branch_model = Model([response(content="Branch done.")])
    reviewer = {
        "role": "assistant",
        "content": "I will verify a revised fix.",
        "tool_calls": [call("review", "file_write", path="fixture.txt", content="branch", mode="create")],
    }
    branch, _, child_journal = run(root, tmp_path / "child", branch_model, prefix=prefix, reviewer=reviewer)
    assert branch.status == "completed", branch.error
    assert (root / "fixture.txt").read_text() == "branch"
    assert branch.usage["input_tokens"] == 10  # inherited and reviewer tokens are not recharged
    child_messages = branch_model.requests[0][0]
    assert child_messages[: len(retained)] == retained
    # Single2 adds its own native progress steering; branching must not add
    # another task prompt or reviewer instruction as a synthetic user message.
    new_users = [message for message in child_messages[len(retained) :] if message["role"] == "user"]
    assert all(message["content"].startswith("[Progress:") for message in new_users)
    assert len(available_branch_points(child_journal)) == 1
    validate_assistant_turn(
        reviewer, history=retained, tools=actor_tools_at(journal, step), slot="opencollab-base"
    )


def test_transport_failure_cannot_be_reported_completed(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    sandbox = Sandbox(root)
    sandbox.failure = True
    result, _, journal = run(
        root,
        tmp_path / "broken",
        Model([response(call("a", "bash", command="true")), response(content="Done.")]),
        sandbox=sandbox,
    )
    assert result.status == "error"
    assert not available_branch_points(journal)
    rows = read_journal(journal)
    assert any(row["type"] == "execution.uncertain" for row in rows)
    bridge = next(row for row in rows if row["type"] == "opencollab.bridge.failed")
    assert bridge["cause_type"] == "ToolInfraError"
    assert "did not settle" in bridge["cause"]
    uncertain = next(row for row in rows if row["type"] == "execution.uncertain")
    assert uncertain["exception"] == "ConnectionError"
    assert uncertain["message"] == "synthetic lost execution transport"
    finished = next(row for row in rows if row["type"] == "run.finished")
    assert finished["stop_reason"] == "infrastructure_error"


@pytest.mark.parametrize(
    "returncode,stdout,truncated,error",
    [
        (0, "Unicode: 中文\n", False, None),
        (1, "", False, FileNotFoundError),
        (0, "partial", True, OSError),
    ],
)
def test_full_file_read_uses_owned_transport(tmp_path, returncode, stdout, truncated, error):
    """New Docker private methods must not bypass the SPROUT environment port."""
    from opencollab.adapters.env import ExecResult

    env = SproutEnvironment(None, str(tmp_path), RunControl(), time.monotonic() + 10)
    calls = []

    async def execute(command, timeout=120):
        calls.append(command)
        return ExecResult(returncode, stdout, "missing", stdout_truncated=truncated)

    env.exec_cmd = execute
    path = "/workspace/a'b.txt"
    if error is None:
        assert asyncio.run(env.read_file(path)) == stdout
    else:
        with pytest.raises(error):
            asyncio.run(env.read_file(path))
    import shlex

    assert calls == [f"cat -- {shlex.quote(path)}"]


def test_cancelled_command_is_settled_without_becoming_transport_failure(tmp_path):
    """OpenCollab may cancel a tool deadline while its synchronous worker is
    queued.  The adapter must let that owned call settle before returning the
    cancellation, otherwise it runs behind a later checkpoint."""
    settled = threading.Event()

    class SlowTools:
        def call(self, _name, _args, *, call_id, turn_id):
            assert call_id and turn_id
            time.sleep(0.05)
            settled.set()
            return ToolResult(True, "done", outcome=CommandOutcome(exit_code=0, stdout="done"))

    async def scenario():
        env = SproutEnvironment(SlowTools(), str(tmp_path), RunControl(), time.monotonic() + 10)
        env.turn_id = "turn"
        task = asyncio.create_task(env.exec_cmd("true"))
        await asyncio.sleep(0.005)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return env

    env = asyncio.run(scenario())
    assert settled.is_set()
    assert env.failure is None
    assert len(env.call_ids) == 1


def test_controller_stop_after_settled_command_is_not_transport_failure(tmp_path):
    control = RunControl()

    class StoppingTools:
        def call(self, _name, _args, *, call_id, turn_id):
            assert call_id and turn_id
            control.request_stop("deadline", stop_reason="timeout")
            return ToolResult(True, "done", outcome=CommandOutcome(exit_code=0, stdout="done"))

    async def scenario():
        env = SproutEnvironment(StoppingTools(), str(tmp_path), control, time.monotonic() + 10)
        env.turn_id = "turn"
        with pytest.raises(Exception, match="deadline"):
            await env.exec_cmd("true")
        return env

    env = asyncio.run(scenario())
    assert env.failure is None
    assert len(env.call_ids) == 1


def test_prefix_checksum_and_invalid_reviewer_arguments(tmp_path):
    import hashlib

    from opencollab.bootstrap.agent_profiles import resolve_agent_profile

    from sprout.worker.agents.opencollab.history import History

    path = tmp_path / "native.jsonl"
    history = History(path, "session", {"cwd": "/fixture"})
    history.append({"type": "oc.turn", "session_snapshot": {}, "runtime_call_ids": []})
    history.close()
    reference = {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    assert load_prefix(reference)
    with path.open("ab") as stream:
        stream.write(b" \n")
    with pytest.raises(ValueError, match="checksum"):
        load_prefix(reference)
    schemas = [tool.to_openai_schema() for tool in resolve_agent_profile("base").resolve_tools("coding")]
    invalid = {
        "role": "assistant",
        "content": "Check a hypothesis.",
        "tool_calls": [call("a", "bash", command="true", invented=True)],
    }
    with pytest.raises(ValueError, match="recorded tool schema"):
        validate_assistant_turn(invalid, tools=schemas, slot="opencollab-base")


def test_native_context_shaping_preserves_actual_model_context(tmp_path):
    from sprout.worker.agents.opencollab.history import History

    path = tmp_path / "shaped.jsonl"
    history = History(path, "native", {"cwd": "/fixture"})
    initial = [{"role": "system", "content": "System."}, {"role": "user", "content": "Original task."}]
    history.persist_messages(initial)
    shaped = [{"role": "system", "content": "System."}, {"role": "user", "content": "Shaped native task."}]
    history.persist_messages(shaped)
    history.persist_messages(shaped[:1])
    history.close()
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert training_messages(rows) == shaped[:1]


def test_slow_generation_has_no_native_per_call_stop(tmp_path, monkeypatch):
    import asyncio

    from sprout.worker.agents.opencollab import slot

    build = slot.build_session
    sessions = []
    agents = []

    def capture(**kwargs):
        agents.append(kwargs["agent"])
        session = build(**kwargs)
        sessions.append(session)
        return session

    class SlowModel(Model):
        async def complete(self, messages, **kwargs):
            await asyncio.sleep(0.08)
            return response(content="Done.")

    monkeypatch.setattr(slot, "build_session", capture)
    root = tmp_path / "repo"
    root.mkdir()
    result, _, _ = run(root, tmp_path / "slow", SlowModel([]), timeout=1)
    assert sessions[0].runner._per_call_timeout is None
    assert all(tool.disable_outer_timeout for tool in agents[0].tools)
    assert result.status == "completed"


@pytest.mark.parametrize(
    "exception", [TimeoutError("provider timed out"), GenerationTimeoutError("synthetic generation deadline")]
)
def test_inner_timeout_is_infrastructure_error_not_actor_deadline(tmp_path, exception):
    class FailedModel(Model):
        async def complete(self, messages, **kwargs):
            raise exception

    root = tmp_path / "repo"
    root.mkdir()
    result, _, journal = run(root, tmp_path / "failed", FailedModel([]))
    assert result.status == "error"
    assert type(exception).__name__ in result.error
    assert "wall-time exhausted" not in result.error
    finished = next(r for r in read_journal(journal) if r["type"] == "run.finished")
    assert finished["termination_kind"] == "infrastructure_error"
    assert finished["elapsed_s"] < finished["actor_timeout_s"]


def test_real_actor_deadline_still_stops_a_pending_generation(tmp_path):
    import asyncio

    class HungModel(Model):
        async def complete(self, messages, **kwargs):
            await asyncio.Event().wait()

    root = tmp_path / "repo"
    root.mkdir()
    result, _, journal = run(root, tmp_path / "deadline", HungModel([]), timeout=0.08)
    assert result.status == "timeout"
    finished = next(r for r in read_journal(journal) if r["type"] == "run.finished")
    assert finished["termination_kind"] == "actor_deadline"
    assert finished["elapsed_s"] >= 0.08


def test_native_compaction_uses_loop_owned_provider_clients(tmp_path):
    import asyncio

    from opencollab.application.compaction_summary import run_coro_blocking

    from sprout.worker.agents.opencollab.slot import RecordedModel

    created = []

    class LoopOwnedModel(Model):
        def __init__(self):
            super().__init__([])
            self.loop = asyncio.get_running_loop()
            created.append(self)

        async def complete(self, messages, **kwargs):
            assert asyncio.get_running_loop() is self.loop
            return response(content="<summary>Retained progress.</summary>")

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            assert asyncio.get_running_loop() is self.loop
            await self.close()

    async def exercise(journal):
        primary = LoopOwnedModel()
        model = RecordedModel(primary, journal, RunControl(), client_factory=LoopOwnedModel)
        await model.complete([])
        for _ in range(2):
            run_coro_blocking(lambda: model.complete([]))
        await model.complete([])
        assert model.usage.input_tokens == 40
        assert not primary.closed
        await primary.close()

    with JournalWriter(tmp_path / "compaction.jsonl") as journal:
        asyncio.run(exercise(journal))
    assert len(created) == 3 and all(client.closed for client in created)

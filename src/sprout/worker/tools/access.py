"""``ToolAccess``: the functions that an agent calls on its sandbox, and their per-turn checkpoints.

An agent gets the runtime's tools as functions. ``ToolAccess.tools`` is their declaration
(``runtime/schema/tools.json``). ``call`` checks a call against that declaration, runs it through
the run's interceptor pipeline, and journals it as ``tool.started`` and ``tool.finished``. The agent
decides how to offer the functions to its model, for example as mini's ``bash``. ``call`` is the
agent's only path into the sandbox.

Checkpoints are per model turn. The agent closes a turn with ``close_turn`` after it records every
observation of the turn's response. The first call of the next turn takes the turn's snapshot first.
Only a model request ran in between, so the disk is the turn's end state. ``finalize`` takes the
rollout's final disk snapshot, which is the last turn's branch point. A benchmark whose grader needs
the sandbox's processes alive asks for a memory snapshot too. That snapshot is journaled separately,
and is never a branch point.

A call whose outcome is unknown stops the rollout, because nothing after it pairs with a known
filesystem. Examples are a transport failure, and a command that still runs or that timed out
without an exit code.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from sprout.worker.run.events import CHECKPOINT_CAPTURED, SESSION_REF, SNAPSHOT_MEMORY
from sprout.worker.tools.pipeline import CallContext, ToolPipeline
from sprout.worker.tools.result import CommandOutcome, ToolResult
from sprout.worker.tools.schema import runtime_tools

if TYPE_CHECKING:
    from sprout.worker.checkpoint.checkpointer import Checkpointer
    from sprout.worker.run.journal import JournalWriter
    from sprout.worker.sandbox.session import SandboxSession

#: Seconds allowed beyond a command's own timeout for the round trip.
TRANSPORT_MARGIN_S = 60.0


class ToolInfraError(RuntimeError):
    """Tool access is closed, or whether a call ran is unknown. The rollout cannot go on."""


class _Control(Protocol):
    """The run's stop requests, such as its ``RunControl``."""

    def raise_if_stopped(self) -> None: ...

    def request_stop(self, reason: str) -> None: ...


@dataclass
class _Turn:
    turn_id: str
    call_ids: list[str] = field(default_factory=list)
    steps: list[int] = field(default_factory=list)
    closed: bool = False


class ToolAccess:
    """The run's tool functions on one sandbox: checked, journaled, and checkpointed per turn.

    One call runs at a time. After ``finalize``, or after a call whose outcome is unknown, each call
    raises ``ToolInfraError``.
    """

    def __init__(
        self,
        session: SandboxSession,
        journal: JournalWriter,
        *,
        tools: list[dict[str, Any]] | None = None,
        pipeline: ToolPipeline | None = None,
        control: _Control | None = None,
        checkpointer: Checkpointer | None = None,
        agent_id: str = "agent",
    ) -> None:
        self.session = session
        self.journal = journal
        self.pipeline = pipeline or ToolPipeline([])
        self.control = control
        #: None turns off the per-turn checkpoints. ``finalize`` still takes the final snapshot.
        self.checkpointer = checkpointer
        self.agent_id = agent_id
        #: The declarations of the tools that the agent may call, by default the runtime's.
        self.tools = list(tools if tools is not None else runtime_tools())
        self._accepts = {t["name"]: set(t["inputSchema"].get("properties", {})) for t in self.tools}
        #: The agent's native session id from the journal's latest ``session.ref`` event, or None.
        self.session_ref: str | None = None
        self._lock = threading.Lock()
        self._turn: _Turn | None = None
        self._calls: set[str] = set()
        self._uncertain: dict[str, object] | None = None
        self._closed = False
        journal.subscribe(self._on_event)

    def _on_event(self, record: dict[str, Any]) -> None:
        if record.get("type") == SESSION_REF and record.get("native_session_id"):
            self.session_ref = record["native_session_id"]

    # --- the agent's functions ----------------------------------------------
    def call(self, name: str, args: dict[str, Any], *, call_id: str, turn_id: str) -> ToolResult:
        """Run tool ``name`` with ``args`` as a call of turn ``turn_id``, and return its result.

        A tool or an argument that the declaration does not list gives a failed result. Raises ``ValueError``
        for an empty or reused ``call_id`` or an empty ``turn_id``. Raises ``ToolInfraError`` when access is
        closed, or when the outcome of this call or an earlier one is unknown."""
        with self._lock:
            if self._closed:
                raise ToolInfraError("tool access is closed")
            if self._uncertain is not None:
                raise ToolInfraError("an earlier call did not settle; the sandbox state is unknown")
            if self.control is not None:
                self.control.raise_if_stopped()
            if not isinstance(call_id, str) or not call_id or call_id in self._calls:
                raise ValueError("tool calls need a unique call_id")
            if not isinstance(turn_id, str) or not turn_id:
                raise ValueError("tool calls need a turn_id")
            turn = self._begin_turn(turn_id)
            message = self._check(name, args)
            if message is not None:
                self.journal.emit(
                    "tool.rejected",
                    call_id=call_id,
                    turn_id=turn_id,
                    name=name,
                    executed=False,
                    source="schema",
                    feedback=message,
                )
                return ToolResult(success=False, output="", error=message)
            tool, routed = name, dict(args)
            self._calls.add(call_id)
            started = self.journal.emit(
                "tool.started", call_id=call_id, turn_id=turn_id, name=tool, args=routed
            )
            turn.call_ids.append(call_id)
            turn.steps.append(started["step"])
            # The pipeline turns an executor exception into a failed result. ``failure`` keeps the
            # exception, so that a transport failure is not taken for a refusal.
            failure: list[Exception] = []

            def execute(runtime_tool: str, runtime_args: dict[str, Any]) -> ToolResult:
                # 300 s is the runtime's default command timeout.
                timeout = float(runtime_args.get("timeout") or 300) + TRANSPORT_MARGIN_S
                try:
                    return _settled(
                        runtime_tool, self.session.call(runtime_tool, runtime_args, timeout=timeout)
                    )
                except Exception as error:
                    failure.append(error)
                    raise

            ctx = CallContext(
                agent_id=self.agent_id, sandbox_id=self.session.sandbox_id, tool_name=tool, args=dict(routed)
            )
            result = self.pipeline.execute(ctx, execute)
            self.journal.emit(
                "tool.finished",
                call_id=call_id,
                turn_id=turn_id,
                name=tool,
                status="ok" if result.success else "error",
                output=result.output or result.error or "",
                outcome=asdict(result.outcome) if result.outcome else None,
            )
            detail = _uncertainty(failure, result)
            if detail is not None:
                self._uncertain = {"call_id": call_id, "step": started["step"], **detail}
                self.journal.emit("execution.uncertain", **self._uncertain)
                if self.control is not None:
                    self.control.request_stop(
                        "execution uncertain at step %s (call %s)" % (started["step"], call_id)
                    )
                raise ToolInfraError("tool execution did not settle: %s" % detail)
            return result

    def _check(self, name: str, args: dict[str, Any]) -> str | None:
        """Why the runtime would not do what this call asks, or None."""
        accepted = self._accepts.get(name)
        if accepted is None:
            return "unknown tool: %s" % name
        extra = sorted(set(args) - accepted)
        if extra:
            return "%s does not take %s; it accepts %s" % (
                name,
                ", ".join(extra),
                ", ".join(sorted(accepted)),
            )
        return None

    def close_turn(self, turn_id: str) -> None:
        """Mark turn ``turn_id`` closed: the agent recorded every observation of its response.

        A closed turn can be a branch point. Raises ``ValueError`` unless ``turn_id`` is the turn of the
        latest call."""
        with self._lock:
            if self._turn is None or self._turn.turn_id != turn_id:
                raise ValueError("close_turn names a turn with no executed calls")
            self._turn.closed = True

    # --- end of the rollout ---------------------------------------------------
    def finalize(self, *, memory: bool = False) -> dict[str, Any] | None:
        """Stop accepting calls, take the rollout's final disk snapshot, and return its journal record.

        None when the backend cannot snapshot or the capture failed. The caller decides what that means.
        With ``memory``, a full snapshot follows. Its id is the record's ``memory_snapshot_id``, None when
        it failed, because the disk snapshot is the durable copy."""
        with self._lock:
            self._closed = True
            if not self.session.supports_snapshot():
                return None
            snapshot = self.session.snapshot(disk_only=True)
            turn = self._turn
            record = {
                "step": turn.steps[-1] if turn and turn.steps else 0,
                "turn_id": turn.turn_id if turn else None,
                "turn_closed": bool(turn and turn.closed),
                "snapshot_id": getattr(snapshot, "id", None),
                "session_ckpt": self.session_ref,
                "captured": snapshot is not None,
                "reason": "captured" if snapshot else "failed",
                "disk_only": True,
                "final": True,
                "delta_empty": getattr(snapshot, "delta_empty", None),
            }
            if self._uncertain is not None:
                record["uncertain"] = True
            self.journal.emit(CHECKPOINT_CAPTURED, **record)
            if snapshot is None:
                return None
            if memory:
                full = self.session.snapshot(disk_only=False)
                record["memory_snapshot_id"] = getattr(full, "id", None)
                self.journal.emit(
                    SNAPSHOT_MEMORY,
                    step=record["step"],
                    turn_id=record["turn_id"],
                    snapshot_id=record["memory_snapshot_id"],
                    captured=full is not None,
                    after=record["snapshot_id"],
                )
            return record

    # --- per-turn checkpoints --------------------------------------------------
    def _begin_turn(self, turn_id: str) -> _Turn:
        """The turn ``turn_id``. A new turn first captures the checkpoint of the turn before it."""
        if self._turn is not None and self._turn.turn_id != turn_id:
            self._capture(self._turn)
            self._turn = None
        if self._turn is None:
            self._turn = _Turn(turn_id)
        return self._turn

    def _capture(self, turn: _Turn) -> None:
        if self.checkpointer is None or not turn.steps:
            return
        step = turn.steps[-1]
        record = self.checkpointer.after_step(step)
        if record is None:
            return
        self.journal.emit(
            CHECKPOINT_CAPTURED,
            step=step,
            turn_id=turn.turn_id,
            turn_closed=turn.closed,
            call_ids=list(turn.call_ids),
            snapshot_id=record.snapshot_id,
            session_ckpt=self.session_ref,
            captured=record.captured,
            reason=record.reason,
            disk_only=record.disk_only,
            delta_empty=record.delta_empty,
            reboarded=record.reboarded,
        )


def _settled(tool: str, result: ToolResult) -> ToolResult:
    """``result``, with an outcome added when it is the bare stdout of a ``shell`` success.

    The runtime sends bare stdout for a command that exited 0 with no stderr and no clipped stream."""
    if tool == "shell" and result.success and result.outcome is None:
        result.outcome = CommandOutcome(
            exit_code=0, stdout=result.output, stdout_bytes=len(result.output.encode("utf-8"))
        )
    return result


def _uncertainty(failure: list[Exception], result: ToolResult) -> dict[str, object] | None:
    """Why the outcome of a call is unknown, as journal fields, or None when it is known."""
    if failure:
        error = failure[0]
        return {
            "kind": "transport_exception",
            "exception": type(error).__name__,
            "message": str(error)[:1000],
        }
    outcome = result.outcome
    if outcome is not None and (outcome.running or (outcome.timed_out and outcome.exit_code is None)):
        return {
            "kind": "runtime_timeout" if outcome.timed_out else "runtime_still_running",
            "running": outcome.running,
            "timed_out": outcome.timed_out,
            "exit_code": outcome.exit_code,
        }
    return None

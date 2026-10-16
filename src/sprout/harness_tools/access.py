"""ToolAccess: the functions an agent calls on its sandbox.

An agent is handed the runtime's tools as functions (``tools`` is their
declaration, ``runtime/schema/tools.json``). ``call`` checks a call against that
declaration, runs it through the run's interceptor pipeline, and journals it as
``tool.started`` / ``tool.finished``. How the agent exposes these functions to
its model -- mini's ``bash``, an MCP server, anything else -- is the agent's
business; this is the one path into the sandbox.

Checkpoints are per model turn. The agent closes a turn with ``close_turn``
once every observation of its response is recorded. The turn's snapshot is
taken lazily, when the next turn's first call arrives: nothing but a model
request ran in between, so the disk is exactly the turn's end state, and the
last turn never pays for a capture of its own -- ``finalize`` takes the
rollout's final disk snapshot, which becomes that turn's branch point. A
benchmark whose grader needs the VM's processes alive (a server the task
started) asks for a memory snapshot on top; it is journaled separately and is
never a branch point.

A call whose outcome is unknown (transport failure, a command still running or
timed out without an exit code) stops the rollout: nothing after it can be
paired with a known filesystem.
"""

from __future__ import annotations

import threading
from dataclasses import asdict, dataclass, field
from typing import Any

from sprout.harness_core.events import CHECKPOINT_CAPTURED, SESSION_REF, SNAPSHOT_MEMORY
from sprout.harness_core.result import CommandOutcome, ToolResult
from sprout.harness_tools.pipeline import CallContext, ToolPipeline
from sprout.harness_tools.schema import runtime_tools

#: Seconds allowed beyond a command's own timeout for the round trip.
TRANSPORT_MARGIN_S = 60.0


class ToolInfraError(RuntimeError):
    """Whether the call ran is unknown, or tools are closed; the rollout cannot go on."""


@dataclass
class _Turn:
    turn_id: str
    call_ids: list = field(default_factory=list)
    steps: list = field(default_factory=list)
    closed: bool = False


class ToolAccess:
    def __init__(
        self,
        session: Any,
        journal: Any,
        *,
        tools: list | None = None,
        pipeline: ToolPipeline | None = None,
        control: Any = None,
        checkpointer: Any = None,
        agent_id: str = "agent",
    ) -> None:
        self.session = session
        self.journal = journal
        self.pipeline = pipeline or ToolPipeline([])
        self.control = control
        #: None disables per-turn checkpoints; ``finalize`` still snapshots.
        self.checkpointer = checkpointer
        self.agent_id = agent_id
        self.tools = list(tools if tools is not None else runtime_tools())
        self._accepts = {t["name"]: set(t["inputSchema"].get("properties", {})) for t in self.tools}
        self.session_ref: str | None = None
        self._lock = threading.Lock()
        self._turn: _Turn | None = None
        self._calls: set = set()
        self._uncertain: dict | None = None
        self._closed = False
        journal.subscribe(self._on_event)

    def _on_event(self, record: dict) -> None:
        if record.get("type") == SESSION_REF and record.get("native_session_id"):
            self.session_ref = record["native_session_id"]

    # --- the agent's functions ----------------------------------------------
    def call(self, name: str, args: dict, *, call_id: str, turn_id: str) -> ToolResult:
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
            self._begin_turn(turn_id)
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
            self._turn.call_ids.append(call_id)
            self._turn.steps.append(started["step"])
            failure: list = []

            def execute(runtime_tool: str, runtime_args: dict) -> ToolResult:
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

    def _check(self, name: str, args: dict) -> str | None:
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
        """Every observation of this turn's response is recorded; it may be branched from."""
        with self._lock:
            if self._turn is None or self._turn.turn_id != turn_id:
                raise ValueError("close_turn names a turn with no executed calls")
            self._turn.closed = True

    # --- end of the rollout ---------------------------------------------------
    def finalize(self, *, memory: bool = False) -> dict | None:
        """Take the rollout's final disk snapshot and stop accepting calls.

        Returns the journaled record, or None when the backend cannot snapshot
        or the capture failed (the caller decides what a missing final snapshot
        means). With ``memory``, a full snapshot follows and its id is returned
        as ``memory_snapshot_id`` (None when it failed: the disk record is the
        durable copy, the memory one an extra the benchmark asked for).
        """
        with self._lock:
            self._closed = True
            if not getattr(self.session, "supports_snapshot", lambda: False)():
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
    def _begin_turn(self, turn_id: str) -> None:
        if self._turn is not None and self._turn.turn_id != turn_id:
            self._capture(self._turn)
            self._turn = None
        if self._turn is None:
            self._turn = _Turn(turn_id)

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
    """The runtime encodes an unclipped, exit-0, stderr-free command as bare
    stdout. Decode that one documented variant so every settled command has an
    outcome."""
    if tool == "shell" and result.success and result.outcome is None:
        result.outcome = CommandOutcome(
            exit_code=0, stdout=result.output, stdout_bytes=len(result.output.encode("utf-8"))
        )
    return result


def _uncertainty(failure: list, result: ToolResult) -> dict | None:
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

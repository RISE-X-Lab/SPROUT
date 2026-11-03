"""One run, start to finish: the sandbox, the gateway, the tools, the slot, and the teardown.

The runner owns the sandbox and hands the agent a ``ToolAccess``, the functions that it may call
on that sandbox. So every tool call crosses one boundary that the runner controls, where per-turn
checkpoints, budgets and rollout limits apply. Every run ends with one disk snapshot, and the
runner never destroys a sandbox before that snapshot exists.

The runner, not the slot, is responsible for these:

- ordering: the sandbox exists before the agent can name it, and the gateway exists before the
  agent reads its ``base_url``. Either one too late is silently wrong, not an error.
- teardown: the runner releases each resource that it acquired, in reverse, also when the slot
  raises. A killed process releases nothing, so each sandbox is also claimed in the resource
  ledger, which ``python -m sprout.worker reap`` reads. Neither one is enough alone.
- accounting: the run's status, usage and resources go to the journal, so a run that dies still
  leaves a record.
"""

from __future__ import annotations

from collections.abc import Callable
from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sprout.contract.backend import backend_name, snapshot_store
from sprout.contract.harbor_task import load_task
from sprout.contract.run_spec import RunSpec
from sprout.worker.agents.mini_swe.slot import MiniSweSlot
from sprout.worker.agents.nop import NopSlot
from sprout.worker.agents.oracle import OracleSlot
from sprout.worker.agents.slot import AgentSlot, SlotResult, TaskSpec
from sprout.worker.run.events import SANDBOX_CREATED
from sprout.worker.run.journal import JournalWriter, new_run_id, read_journal
from sprout.worker.run.resources import ResourceLedger, RunClaim
from sprout.worker.sandbox.image_shell import image_shell

if TYPE_CHECKING:
    from sprout.worker.gateway import GatewayServer
    from sprout.worker.run.control import RunControl
    from sprout.worker.run.rollout import RolloutControls
    from sprout.worker.tools.access import ToolAccess


@dataclass
class OwnedSandbox:
    """A sandbox that the runner holds. The runner does not destroy a ``keep`` one."""

    # Any: a SandboxSession, or a stand-in that has only part of its interface.
    session: Any
    keep: bool = False
    #: Recorded at creation, because teardown runs before the outcome is assembled, and a
    #: destroyed session answers "unknown". A re-board moves it to the replacement sandbox.
    sandbox_id: str = ""

    def destroy(self) -> None:
        self.session.destroy()

    def release(self) -> None:
        """Destroy the sandbox unless it is kept, when wiring fails halfway and nothing can reach it."""
        if not self.keep:
            self.destroy()


@dataclass
class RunOutcome:
    """What one run produced. ``ok`` is True when ``status`` is ``completed``."""

    run_id: str
    journal_path: Path
    status: str
    final_text: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    native_session_id: str | None = None
    sandbox_id: str | None = None
    checkpoints: int = 0
    #: The rollout's final disk snapshot: the durable record and the last branch point.
    final_snapshot_id: str | None = None
    #: The full snapshot taken after it when the run asked for one.
    final_memory_snapshot_id: str | None = None
    error: str | None = None
    stop_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class Runner:
    """Runs one task per call of ``run``.

    It keeps no registry of live agents, because no step of a run needs to know what else runs.
    ``on_event`` hears of the sandbox and the gateway of each run when they start."""

    def __init__(
        self,
        *,
        out_dir: str | Path = "runs",
        ledger: ResourceLedger | None = None,
        on_event: Callable[[str, dict[str, Any]], None] | None = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.ledger = ledger
        self.on_event = on_event or (lambda kind, payload: None)

    # --- the sequence ------------------------------------------------------
    def run(self, spec: RunSpec) -> RunOutcome:
        """Run ``spec``, and return its outcome.

        An error of the wiring or of the slot goes to the outcome and the journal. It is not raised."""
        run_id = spec.run_id or new_run_id()
        journal_path = Path(spec.journal_path or self.out_dir / ("%s.jsonl" % run_id))
        extra = dict(spec.extra)

        # Acquired in order, released in reverse.
        provisioned: OwnedSandbox | None = None
        detach: Callable[[], None] | None = None
        gateway: GatewayServer | None = None
        tools: ToolAccess | None = None
        final: dict[str, Any] | None = None
        rollout_controls: RolloutControls | None = None
        result: SlotResult | None = None
        error: str | None = None
        control: RunControl | None = None

        # ExitStack so the ledger's run context (which marks the run done even on
        # exception) is optional without duplicating the body under an if.
        with ExitStack() as stack:
            journal = stack.enter_context(JournalWriter(journal_path, run_id=run_id, agent_id=spec.agent_id))
            if spec.origin:
                # First, before anything subscribes or runs: whatever else this
                # journal ends up holding, it says where the run came from.
                journal.emit("fork.origin", **spec.origin)
            claim = stack.enter_context(self.ledger.run(run_id)) if self.ledger is not None else None
            try:
                slot = _agent_slot(spec)
                from sprout.worker.run.control import RunControl

                control = RunControl()
                if extra.get("rollout_contract") is not None:
                    from sprout.worker.run.rollout import RolloutControls

                    rollout_controls = RolloutControls(extra["rollout_contract"], journal, control)
                    rollout_controls.start()
                provisioned = self._wire_sandbox(spec, claim)
                detach = self._follow_reboards(spec, provisioned, claim)
                _emit_sandbox_created(spec=spec, journal=journal, sandbox_id=provisioned.sandbox_id)
                if spec.agent == "mini":
                    if spec.slot == "opencollab-base":
                        from sprout.worker.agents.opencollab.workspace import resolve_workspace
                    else:
                        from sprout.worker.agents.mini_swe.sandbox import resolve_workspace

                    extra = resolve_workspace(provisioned.session, extra, journal)
                tools = self._tool_access(spec, journal, provisioned, control, rollout_controls)
                task = _task_spec(spec=spec, extra=extra, control=control, session=provisioned.session)
                task.extra["rollout_controls"] = rollout_controls
                gateway = self._wire_gateway(spec, journal, task, run_id, controls=rollout_controls)
                result = slot.run(task, journal, tools)
                error = _stop_error(control=control, result=result)
                if error is not None:
                    journal.emit("run.finished", status="error", error=error)
            except Exception as exc:  # noqa: BLE001 - a run reports, it does not raise
                error = "%s: %s" % (type(exc).__name__, exc)
                journal.emit("run.finished", status="error", error=error)
            finally:
                if rollout_controls is not None:
                    rollout_controls.finish()
                # The runner makes the tools after the sandbox, so ``provisioned`` is set when ``tools`` is.
                if tools is not None and provisioned is not None:
                    final, failure = self._finalize(spec, tools, provisioned, claim, journal)
                    if failure:
                        error = failure
                        journal.emit("run.finished", status="error", error=error)
                if detach is not None:
                    detach()
                self._teardown(spec, gateway, provisioned, claim)

        return _outcome(
            run_id=run_id,
            journal_path=journal_path,
            result=result,
            error=error,
            sandbox_id=(provisioned.sandbox_id if provisioned else None),
            final=final,
            stop_reason=(control.stop_reason if control is not None else None),
            checkpoints=(_checkpoint_count(journal_path) if tools is not None else 0),
        )

    # --- steps -------------------------------------------------------------
    def _wire_sandbox(self, spec: RunSpec, claim: RunClaim | None) -> OwnedSandbox:
        """Create the run's sandbox, on the agent network of its Harbor task if it names one, and claim it.

        Every tool call, snapshot and the teardown go through the returned session. Raises ``ValueError``
        without ``spec.sandbox_image``, and ``RuntimeError`` if the backend cannot create the sandbox."""
        from sprout.worker.sandbox.backends import with_egress, with_sandbox_budget
        from sprout.worker.sandbox.session import SandboxSession

        if not spec.sandbox_image:
            raise ValueError("RunSpec.sandbox_image is required: the orchestrator owns the sandbox")
        backend = with_sandbox_budget(spec.backend, spec.timeout_s)
        if spec.harbor_task is not None:
            # Not the request's to set: the worker profile picks the backend whose section holds it.
            backend = with_egress(backend, allow=load_task(Path(spec.harbor_task)).agent_network == "public")
        session = SandboxSession(backend=backend, quiet=True)
        if not session.create(spec.sandbox_image, spec.sandbox_resources):
            # The quiet session prints no reason, so the exception carries it. getattr: a stand-in
            # session can record no reason.
            reason = getattr(session, "create_error", "")
            raise RuntimeError(
                "could not create a sandbox from %s%s"
                % (spec.sandbox_image, ": %s" % reason if reason else "")
            )
        owned = OwnedSandbox(session=session, keep=spec.keep_sandbox, sandbox_id=session.sandbox_id)
        try:
            if claim is not None:
                # Claimed before the run uses it, because a killed process releases nothing. The reaper
                # frees it through the claim's backend, and finds a podman rootfs in the claim's store.
                claim.sandbox(
                    session.sandbox_id,
                    keep=spec.keep_sandbox,
                    backend=backend_name(spec.backend),
                    store=snapshot_store(spec.backend),
                )
            self.on_event("sandbox", {"sandbox_id": session.sandbox_id})
            return owned
        except Exception:
            # The sandbox exists, but nothing can reach it. Release it, or it leaks behind the exception.
            owned.release()
            raise

    def _follow_reboards(
        self, spec: RunSpec, owned: OwnedSandbox, claim: RunClaim | None
    ) -> Callable[[], None]:
        """Keep ``owned.sandbox_id`` and the ledger's claim on the live sandbox across a re-board.

        A re-board moves the session onto a sandbox started from a compacted snapshot. Returns a function
        that detaches the listener, because a lent session outlives the run. A stand-in session without
        ``on_swap`` never re-boards."""
        listeners = getattr(owned.session, "on_swap", None)
        if not isinstance(listeners, list):
            return lambda: None

        def follow(replacement: object) -> None:
            # The session calls this before it destroys the old sandbox. A process killed afterwards
            # leaves a replacement that the reaper can attribute.
            owned.sandbox_id = owned.session.sandbox_id
            if claim is not None:
                claim.sandbox(
                    owned.sandbox_id,
                    keep=owned.keep,
                    backend=backend_name(spec.backend),
                    store=snapshot_store(spec.backend),
                )

        def detach() -> None:
            if follow in listeners:
                listeners.remove(follow)

        listeners.append(follow)
        return detach

    def _tool_access(
        self,
        spec: RunSpec,
        journal: JournalWriter,
        owned: OwnedSandbox,
        control: RunControl,
        rollout_controls: RolloutControls | None = None,
    ) -> ToolAccess:
        """The functions the agent may call on its sandbox.

        One pipeline serves every call: rollout limits first, then the mutation
        tracker the checkpointer reads, then the rules the run declares.
        """
        from sprout.worker.checkpoint.checkpointer import Checkpointer
        from sprout.worker.tools.access import ToolAccess
        from sprout.worker.tools.interceptors import MutationTracker, build_interceptors
        from sprout.worker.tools.pipeline import ToolPipeline

        tracker = MutationTracker()
        interceptors = [
            *([rollout_controls] if rollout_controls is not None else []),
            tracker,
            *build_interceptors(spec.interceptors),
        ]
        checkpointer = (
            Checkpointer(session=owned.session, tracker=tracker, disk_only=True)
            if spec.checkpoint_enabled
            else None
        )
        return ToolAccess(
            owned.session,
            journal,
            pipeline=ToolPipeline(interceptors),
            control=control,
            checkpointer=checkpointer,
            agent_id=spec.agent_id,
        )

    def _finalize(
        self,
        spec: RunSpec,
        tools: ToolAccess,
        owned: OwnedSandbox,
        claim: RunClaim | None,
        journal: JournalWriter,
    ) -> tuple[dict[str, Any] | None, str | None]:
        """Take the rollout's final disk snapshot, and a memory one if asked. Returns ``(record, error)``.

        If the snapshot fails on a backend that can snapshot, the sandbox is kept, in the journal and the
        ledger, because it holds the only copy of the rollout's result."""
        try:
            record = tools.finalize(memory=spec.final_memory_snapshot)
        except Exception as exc:  # noqa: BLE001
            record, reason = None, "%s: %s" % (type(exc).__name__, exc)
        else:
            reason = "snapshot failed"
        supported = owned.session.supports_snapshot()
        if record is not None or not supported:
            return record, None
        owned.keep = True
        if claim is not None:
            claim.sandbox(
                owned.sandbox_id,
                keep=True,
                backend=backend_name(spec.backend),
                store=snapshot_store(spec.backend),
            )
        journal.emit("sandbox.retained", sandbox_id=owned.sandbox_id, reason="snapshot_pending", error=reason)
        return None, "Final snapshot failed (%s); sandbox %s retained" % (reason, owned.sandbox_id)

    def _wire_gateway(
        self,
        spec: RunSpec,
        journal: JournalWriter,
        task: TaskSpec,
        run_id: str,
        *,
        controls: RolloutControls | None,
    ) -> GatewayServer | None:
        """Start a gateway and point the agent at it, if the run needs one, and return the gateway.

        With rollout controls, the gateway routes to the contract's endpoint and applies its limits to each
        request. A ``budget_usd`` without a contract also starts one, with no route. It refuses every request,
        so the agent cannot reach the provider directly, past the ceiling."""
        if controls is None and not spec.budget_usd:
            return None
        from sprout.worker.gateway import GatewayServer, RoutingTable

        table = RoutingTable()
        if controls is not None:
            from sprout.worker.gateway.routing import ModelRoute

            route = ModelRoute(
                base_url=controls.base_url,
                upstream_model=controls.contract.get("model"),
                api_key_env=controls.contract.get("api_key_env"),
            )
            # An endpoint in the request must never inherit another provider's
            # credential. Only the explicitly configured rollout key may travel.
            table = RoutingTable({"default": route})
            # The run fails here, before its agent starts, when the route's key variable is unset.
            route.resolve_key()
        gateway = GatewayServer(table, journal=journal, request_policy=controls).start()
        token = table.mint(spec.agent_id, run_id=run_id, budget_usd=spec.budget_usd)
        task.env.update(gateway.env_for(token))
        self.on_event("gateway", {"url": gateway.base_url, "budget_usd": spec.budget_usd})
        return gateway

    def _teardown(
        self,
        spec: RunSpec,
        gateway: GatewayServer | None,
        provisioned: OwnedSandbox | None,
        claim: RunClaim | None,
    ) -> None:
        """Stop the gateway, then destroy the sandbox unless it is kept. A failed step skips no other."""
        if gateway is not None:
            try:
                gateway.stop()
            except Exception:  # noqa: BLE001
                pass
        if provisioned is not None and not (spec.keep_sandbox or provisioned.keep):
            try:
                provisioned.destroy()
                if claim is not None:
                    claim.released("sandbox", provisioned.sandbox_id)
            except Exception:  # noqa: BLE001 - `python -m sprout.worker reap` is the backstop
                pass


def _emit_sandbox_created(*, spec: RunSpec, journal: JournalWriter, sandbox_id: str) -> None:
    """Journal the run's sandbox ``sandbox_id``, with its backend, its image and its snapshot store."""
    # The ``sweep`` command reads this record later. A podman run's snapshot ids are subvolume
    # names in its store, meaningless to AgentENV, so the record also names the store.
    store = snapshot_store(spec.backend)
    journal.emit(
        SANDBOX_CREATED,
        sandbox_id=sandbox_id,
        backend=backend_name(spec.backend),
        image=spec.sandbox_image,
        **({"store": store} if store else {}),
    )


def _task_spec(*, spec: RunSpec, extra: dict[str, Any], control: RunControl, session: Any) -> TaskSpec:
    """The task that the slot runs, with a shell on the run's image in ``session``."""
    return TaskSpec(
        prompt=spec.prompt,
        cwd=spec.cwd,
        model=spec.model,
        timeout_s=spec.timeout_s,
        extra=extra,
        control=control,
        image_shell=image_shell(session, backend=spec.backend, image=_image_reference(spec)),
    )


def _stop_error(*, control: RunControl, result: SlotResult | None) -> str | None:
    """The run's error from a stop request, or None.

    A slot that ended in ``timeout`` after a stop for a gradable reason ends without an error."""
    from sprout.worker.run.control import GRADABLE_STOP_REASONS

    if control.reason is not None and not (
        result is not None and result.status == "timeout" and control.stop_reason in GRADABLE_STOP_REASONS
    ):
        return control.reason
    return None


def _checkpoint_count(journal_path: Path) -> int:
    """The number of snapshots that the ``checkpoint.captured`` records name, without the final one."""
    return len(
        [
            r
            for r in read_journal(journal_path)
            if r.get("type") == "checkpoint.captured" and r.get("snapshot_id") and not r.get("final")
        ]
    )


def _outcome(
    *,
    run_id: str,
    journal_path: Path,
    result: SlotResult | None,
    error: str | None,
    sandbox_id: str | None,
    final: dict[str, Any] | None,
    stop_reason: str | None,
    checkpoints: int,
) -> RunOutcome:
    """The outcome of a run. With an ``error``, the status is ``error``, whatever the slot returned."""
    return RunOutcome(
        run_id=run_id,
        journal_path=journal_path,
        status=(result.status if result and not error else "error"),
        final_text=(result.final_text if result else ""),
        usage=(result.usage if result else {}),
        native_session_id=(result.native_session_id if result else None),
        sandbox_id=sandbox_id,
        checkpoints=checkpoints,
        final_snapshot_id=(final or {}).get("snapshot_id"),
        final_memory_snapshot_id=(final or {}).get("memory_snapshot_id"),
        error=error or (result.error if result else None),
        stop_reason=stop_reason or ("timeout" if result and result.status == "timeout" else None),
    )


def _image_reference(spec: RunSpec) -> str | None:
    """The OCI reference of the run's image: its Harbor task's, which a branch from a snapshot names too."""
    if spec.harbor_task is not None:
        return load_task(Path(spec.harbor_task)).docker_image
    return spec.sandbox_image


def _agent_slot(spec: RunSpec) -> AgentSlot:
    """The slot that runs ``spec.agent``. Raises ValueError for a slot or an agent that no slot runs."""
    if spec.slot not in {"mini-swe-agent", "opencollab-base"}:
        raise ValueError("unknown slot %r; the only slot is mini-swe-agent" % spec.slot)
    if spec.agent == "mini":
        if spec.slot == "opencollab-base":
            from sprout.worker.agents.opencollab.slot import OpenCollabSlot

            return OpenCollabSlot()
        return MiniSweSlot()
    if spec.agent == "nop":
        return NopSlot()
    if spec.agent != "oracle":
        raise ValueError("unknown agent %r; the agents are mini, oracle and nop" % spec.agent)
    if spec.harbor_task is None:
        raise ValueError("an oracle run needs the harbor_task whose solution it runs")
    return OracleSlot(load_task(Path(spec.harbor_task)))

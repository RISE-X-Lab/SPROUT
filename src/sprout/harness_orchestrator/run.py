"""One run, start to finish.

This is the code the CLI used to hold inline: provision a sandbox, mount a
gateway, install the checkpoint bridge, drive the slot, and tear all of it down
whichever way the run ends. It lives here because it is the *shape* of a run
rather than a way of invoking one -- ``harness run``, the eval loops, Run Store
workers and the Harbor plugin all need the same sequence, and a copy in each is
a copy that drifts.

The orchestrator always owns the sandbox and hands the agent a ToolAccess --
the functions it may call on that sandbox -- so every tool call crosses one
boundary it controls: that is where per-turn checkpoints, budgets and rollout
limits are enforced. Every run ends with one disk snapshot, and a sandbox is
never destroyed before that snapshot exists.

What the orchestrator owns (and a slot deliberately does not):

**Ordering.** The sandbox exists before the agent can name it, and the gateway
exists before the agent reads its ``base_url``. Each of those is a "too late is
silently wrong" rather than an error.

**Teardown.** Every resource acquired here is released here, in reverse, in a
``finally`` -- including when the slot raises. This is best-effort by nature: a
killed *process* releases nothing, which is why acquisitions are also written to
the resource ledger for ``harness reap`` to reclaim. Guaranteed cleanup lives in
two places on purpose, and neither is sufficient alone.

**Accounting.** The run's status, usage and resources go to the journal, not to a
return value only: a run that dies still leaves a record.
"""

from __future__ import annotations

from contextlib import ExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

from sprout.harness_agents.mini_swe.slot import MiniSweSlot
from sprout.harness_core.events import SANDBOX_CREATED
from sprout.harness_core.journal import JournalWriter, new_run_id, read_journal
from sprout.harness_core.slot import SlotResult, TaskSpec
from sprout.harness_orchestrator.resources import ResourceLedger
from sprout.harness_sandbox.backends import backend_name, snapshot_store


@dataclass
class RunSpec:
    """Everything one run needs; ``harness run`` sets the subset a person types."""

    prompt: str
    #: The agent. mini-swe-agent is the only one; any other name is refused
    #: before a sandbox is made.
    slot: str = "mini-swe-agent"
    cwd: str = "."
    model: Optional[str] = None
    timeout_s: float = 3600.0
    agent_id: str = "agent"
    run_id: Optional[str] = None
    journal_path: Optional[Union[str, Path]] = None

    # --- sandbox ---
    #: Where the sandbox comes from (``harness_sandbox/backends.py``):
    #: ``{"backend": "microvm", "microvm": {...}}`` for an AgentENV microVM, or
    #: ``{"backend": "podman", "podman": {"store": ..., "runtime_bin": ...}}``
    #: for a rootless container whose rootfs is a subvolume of the store. Empty
    #: means podman, the default backend, whose section must still name the
    #: store and the runtime binary.
    backend: Dict[str, Any] = field(default_factory=dict)
    #: The sprout-runtime binary named on the command line (``harness run
    #: --runtime-bin``) or in a Run Store job's ``run_defaults``. Accepted for
    #: that contract (``runstore/specs.py`` validates job specs against these
    #: fields) and not otherwise consumed: the binary a backend actually uses is
    #: the one in its own section -- ``backend["podman"]["runtime_bin"]`` is
    #: mounted into every container, ``backend["microvm"]["runtime_bin"]`` builds
    #: the per-image template (``harness_sandbox/templates.py``).
    runtime_bin: Optional[str] = None
    #: The image (or snapshot) the sandbox is created from. Required.
    sandbox_image: Optional[str] = None
    #: Shape for that sandbox, ``{"cpu": 2, "memory_mb": 8192}``. A task that
    #: declares its needs must get them; the backend's default (2 CPUs, 1 GB)
    #: OOMs a build-heavy task instead of failing loudly. None = default.
    sandbox_resources: Optional[Dict[str, Any]] = None
    #: Keep the sandbox after the run -- grading and extraction need it alive.
    keep_sandbox: bool = False
    #: Rules mounted on the serving pipeline after the mutation tracker, by name
    #: (``harness_tools.interceptors.INTERCEPTORS``): ``[{"name": ..., **kwargs}]``.
    interceptors: List[Dict[str, Any]] = field(default_factory=list)

    # --- gateway ---
    #: USD ceiling the gateway enforces on this run's model traffic. Set by the
    #: RL contract and by Run Store continuations, never by hand: the gateway
    #: starts for a run with rollout controls, and a budget on its own has no
    #: upstream to route to.
    budget_usd: Optional[float] = None

    # --- checkpoints ---
    #: One disk snapshot per closed model turn. Off for trajectories that are
    #: never branched; the final disk snapshot is taken either way.
    checkpoint_enabled: bool = True
    #: Also take a full (memory) snapshot at the end, for a grader that needs
    #: the VM's processes alive. The benchmark decides; it may release it after
    #: grading.
    final_memory_snapshot: bool = False

    # --- lineage ---
    #: Lineage of a branched run, emitted as the journal's FIRST event
    #: (``fork.origin``): parent run id, branch step, the pair it grew from.
    #: In the journal rather than a sidecar because the journal is the record a
    #: killed run leaves behind -- a branch whose ancestry lives anywhere else
    #: is unattributable exactly when attribution matters. The retained history
    #: itself travels as ``extra["native_prefix"]``.
    origin: Optional[Dict[str, Any]] = None

    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OwnedSandbox:
    """A sandbox this orchestrator created (or was lent, ``keep=True``)."""

    session: Any
    keep: bool = False
    #: Recorded at creation, NOT read from the session on demand. Teardown runs
    #: before the outcome is assembled, and a destroyed session answers "unknown".
    #: A re-board moves it to the replacement (``Orchestrator._follow_reboards``).
    sandbox_id: str = ""

    def destroy(self) -> None:
        self.session.destroy()

    def release(self) -> None:
        """Give it back when wiring fails halfway: nothing can reach it."""
        if not self.keep:
            self.destroy()


@dataclass
class RunOutcome:
    run_id: str
    journal_path: Path
    status: str
    final_text: str = ""
    usage: Dict[str, Any] = field(default_factory=dict)
    native_session_id: Optional[str] = None
    sandbox_id: Optional[str] = None
    checkpoints: int = 0
    #: The rollout's final disk snapshot: the durable record and the last branch point.
    final_snapshot_id: Optional[str] = None
    #: The full snapshot taken after it when the run asked for one.
    final_memory_snapshot_id: Optional[str] = None
    error: Optional[str] = None
    stop_reason: str | None = None

    @property
    def ok(self) -> bool:
        return self.status == "completed"


class Orchestrator:
    """Runs one task at a time; ``batch`` puts many of these in flight.

    Deliberately not a registry of live agents: nothing here needs to ask "what
    else is running", and inventing that before a subagent exists would be a
    state machine with one state. When spawning arrives, this is where it goes --
    the sequence below is already what a child would need.
    """

    def __init__(
        self,
        *,
        out_dir: Union[str, Path] = "runs",
        ledger: Optional[ResourceLedger] = None,
        on_event: Optional[Callable[[str, dict], None]] = None,
    ) -> None:
        self.out_dir = Path(out_dir)
        self.ledger = ledger
        self.on_event = on_event or (lambda kind, payload: None)

    # --- the sequence ------------------------------------------------------
    def run(self, spec: RunSpec) -> RunOutcome:
        run_id = spec.run_id or new_run_id()
        journal_path = Path(spec.journal_path or self.out_dir / ("%s.jsonl" % run_id))
        slot = MiniSweSlot()
        extra = dict(spec.extra)

        # Acquired in order, released in reverse.
        provisioned = None
        detach = None
        gateway = None
        tools = None
        final = None
        rollout_controls = None
        result: Optional[SlotResult] = None
        error: Optional[str] = None
        control = None

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
                if spec.slot != "mini-swe-agent":
                    raise ValueError("unknown slot %r; the only slot is mini-swe-agent" % spec.slot)
                from sprout.harness_core.control import GRADABLE_STOP_REASONS, RunControl

                control = RunControl()
                if extra.get("rollout_contract") is not None:
                    from sprout.harness_orchestrator.rollout import RolloutControls

                    rollout_controls = RolloutControls(extra["rollout_contract"], journal, control)
                    rollout_controls.start()
                provisioned = self._wire_sandbox(spec, claim)
                detach = self._follow_reboards(spec, provisioned, claim)
                # On the record with its backend, and with the store when the
                # snapshots are files on this host: the journal is what `sweep`
                # reads later, and a podman run's snapshot ids are subvolume
                # names in that store, meaningless to AgentENV.
                store = snapshot_store(spec.backend)
                journal.emit(
                    SANDBOX_CREATED,
                    sandbox_id=provisioned.sandbox_id,
                    backend=backend_name(spec.backend),
                    image=spec.sandbox_image,
                    **({"store": store} if store else {}),
                )
                from sprout.harness_agents.mini_swe.sandbox import resolve_workspace

                extra = resolve_workspace(provisioned.session, extra, journal)
                tools = self._tool_access(spec, journal, provisioned, control, rollout_controls)
                task = TaskSpec(
                    prompt=spec.prompt,
                    cwd=spec.cwd,
                    model=spec.model,
                    timeout_s=spec.timeout_s,
                    extra=extra,
                    control=control,
                )
                task._rollout_controls = rollout_controls
                gateway = self._wire_gateway(spec, journal, task, run_id)
                result = slot.run(task, journal, tools)
                if control.reason is not None and not (
                    result is not None
                    and result.status == "timeout"
                    and control.stop_reason in GRADABLE_STOP_REASONS
                ):
                    error = control.reason
                    journal.emit("run.finished", status="error", error=error)
            except Exception as exc:  # noqa: BLE001 - a run reports, it does not raise
                error = "%s: %s" % (type(exc).__name__, exc)
                journal.emit("run.finished", status="error", error=error)
            finally:
                if rollout_controls is not None:
                    rollout_controls.finish()
                if tools is not None:
                    final, failure = self._finalize(spec, tools, provisioned, claim, journal)
                    if failure:
                        error = failure
                        journal.emit("run.finished", status="error", error=error)
                if detach is not None:
                    detach()
                self._teardown(spec, gateway, provisioned, claim)

        records = [
            r
            for r in (read_journal(journal_path) if tools is not None else ())
            if r.get("type") == "checkpoint.captured" and r.get("snapshot_id") and not r.get("final")
        ]
        return RunOutcome(
            run_id=run_id,
            journal_path=journal_path,
            status=(result.status if result and not error else "error"),
            final_text=(result.final_text if result else ""),
            usage=(result.usage if result else {}),
            native_session_id=(result.native_session_id if result else None),
            sandbox_id=(provisioned.sandbox_id if provisioned else None),
            checkpoints=len(records),
            final_snapshot_id=(final or {}).get("snapshot_id"),
            final_memory_snapshot_id=(final or {}).get("memory_snapshot_id"),
            error=error or (result.error if result else None),
            stop_reason=(
                getattr(control, "stop_reason", None)
                or ("timeout" if result and result.status == "timeout" else None)
            ),
        )

    # --- steps -------------------------------------------------------------
    def _wire_sandbox(self, spec: RunSpec, claim) -> "OwnedSandbox":
        """Create the sandbox. The session stays here: every tool call, snapshot
        and the final teardown go through this one handle."""
        from sprout.harness_sandbox.backends import with_sandbox_budget
        from sprout.harness_sandbox.session import SandboxSession

        if not spec.sandbox_image:
            raise ValueError("RunSpec.sandbox_image is required: the orchestrator owns the sandbox")
        session = SandboxSession(backend=with_sandbox_budget(spec.backend, spec.timeout_s), quiet=True)
        if not session.create(spec.sandbox_image, spec.sandbox_resources):
            # The session is quiet here (its progress lines are not this run's
            # output), so the reason has to travel in the exception or it is lost.
            # getattr: any session-shaped object may be substituted here, and not
            # every one records a reason.
            reason = getattr(session, "create_error", "")
            raise RuntimeError(
                "could not create a sandbox from %s%s"
                % (spec.sandbox_image, ": %s" % reason if reason else "")
            )
        owned = OwnedSandbox(session=session, keep=spec.keep_sandbox, sandbox_id=session.sandbox_id)
        try:
            if claim is not None:
                # Claimed before the run does anything with it: a process killed
                # mid-run releases nothing, and `harness reap` reads this -- with
                # the backend it frees it through and, for podman, the store
                # that holds its rootfs once the container is gone.
                claim.sandbox(
                    session.sandbox_id,
                    keep=spec.keep_sandbox,
                    backend=backend_name(spec.backend),
                    store=snapshot_store(spec.backend),
                )
            self.on_event("sandbox", {"sandbox_id": session.sandbox_id})
            return owned
        except Exception:
            # The sandbox exists but nothing can reach it; do not leak it while
            # the caller sees only an exception.
            owned.release()
            raise

    def _follow_reboards(self, spec: RunSpec, owned: "OwnedSandbox", claim) -> Callable[[], None]:
        """Keep the run's record on the live sandbox across a re-board.

        A checkpoint that finds the chain compacted moves the session onto a
        sandbox started from the compact snapshot, and the session destroys
        the old one. Its listeners hear of the swap before that: the ledger
        claims the replacement there, so a process killed afterwards leaves a
        sandbox ``harness reap`` can attribute, and ``owned.sandbox_id`` --
        what the retained claim, the release and the outcome name -- moves to
        it. Returns what detaches the listener, since a lent session outlives
        the run; a stand-in session with no listeners never swaps.
        """
        listeners = getattr(owned.session, "on_swap", None)
        if not isinstance(listeners, list):
            return lambda: None

        def follow(replacement) -> None:
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

    def _tool_access(self, spec: RunSpec, journal, owned: "OwnedSandbox", control, rollout_controls=None):
        """The functions the agent may call on its sandbox.

        One pipeline serves every call: rollout limits first, then the mutation
        tracker the checkpointer reads, then the rules the run declares.
        """
        from sprout.harness_checkpoint.checkpointer import Checkpointer
        from sprout.harness_tools.access import ToolAccess
        from sprout.harness_tools.interceptors import MutationTracker, build_interceptors
        from sprout.harness_tools.pipeline import ToolPipeline

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

    def _finalize(self, spec: RunSpec, tools, owned: "OwnedSandbox", claim, journal):
        """The rollout's final disk snapshot (plus memory when asked); (record, error).

        A sandbox is never destroyed before the disk snapshot exists: when it fails,
        the sandbox is kept -- on the record, for the ledger and ``reap`` -- since
        it holds the only copy of the rollout's result.
        """
        try:
            record = tools.finalize(memory=spec.final_memory_snapshot)
        except Exception as exc:  # noqa: BLE001
            record, reason = None, "%s: %s" % (type(exc).__name__, exc)
        else:
            reason = "snapshot failed"
        supported = getattr(owned.session, "supports_snapshot", lambda: False)()
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

    def _wire_gateway(self, spec: RunSpec, journal, task: TaskSpec, run_id: str):
        """Start a gateway and point the agent at it, when this run needs one.

        A run with rollout controls gets one: its model endpoint, credential
        and model name come from the contract, and the gateway applies the
        contract's sampling, turn and deadline limits to every request. A
        ``budget_usd`` without a contract also starts one -- with no upstream
        route, so every request is refused rather than letting the agent reach
        the provider directly, past the ceiling.
        """
        controls = getattr(task, "_rollout_controls", None)
        if controls is None and not spec.budget_usd:
            return None
        from sprout.harness_gateway import GatewayServer, RoutingTable

        table = RoutingTable()
        kwargs = {}
        if controls is not None:
            from sprout.harness_gateway.routing import ModelRoute

            route = ModelRoute(
                base_url=controls.base_url,
                upstream_model=controls.contract.get("model"),
                api_key_env=controls.contract.get("api_key_env"),
            )
            # An endpoint in the request must never inherit another provider's
            # credential. Only the explicitly configured rollout key may travel.
            table = RoutingTable({"default": route})
            kwargs["request_policy"] = controls
        gateway = GatewayServer(table, journal=journal, **kwargs).start()
        token = table.mint(spec.agent_id, run_id=run_id, budget_usd=spec.budget_usd)
        task.env.update(gateway.env_for(token))
        self.on_event("gateway", {"url": gateway.base_url, "budget_usd": spec.budget_usd})
        return gateway

    def _teardown(self, spec: RunSpec, gateway, provisioned, claim) -> None:
        """Release in reverse. Each step is independent: one failure must not
        strand the others, which is why they are separately guarded."""
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
            except Exception:  # noqa: BLE001 - `harness reap` is the backstop
                pass

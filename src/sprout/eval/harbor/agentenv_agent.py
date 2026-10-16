"""Sprout actor and per-tool checkpoints on a Harbor-owned sandbox, microVM or podman."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from dataclasses import asdict

from harbor.agents.base import BaseAgent
from harbor.models.agent.context import AgentContext

from sprout.contract.run_spec import RunSpec
from sprout.eval.harbor.environment import SandboxEnvironment
from sprout.worker.checkpoint.rollback import uncaptured_turns
from sprout.worker.run.atif import export_file
from sprout.worker.run.runner import Runner, OwnedSandbox


class AgentENVOrchestrator(Runner):
    def __init__(self, environment: SandboxEnvironment, model_env: dict, **kwargs):
        super().__init__(**kwargs)
        self.environment = environment
        self.model_env = model_env
        self.cancelled = threading.Event()
        self.cancel_stop_reason = None

    def cancel(self, *, stop_reason=None) -> None:
        if stop_reason is not None:
            self.cancel_stop_reason = stop_reason
        self.cancelled.set()
        if self.environment.actor_control is not None:
            self.environment.actor_control.request_stop(
                "Harbor cancelled the actor", stop_reason=self.cancel_stop_reason
            )

    def _wire_sandbox(self, spec, claim):
        # Harbor owns this sandbox's lifetime; the orchestrator only borrows it.
        return OwnedSandbox(
            session=self.environment.session, keep=True, sandbox_id=self.environment.session.sandbox_id
        )

    def _wire_gateway(self, spec, journal, task, run_id):
        task.env.update(self.model_env)
        self.environment.actor_control = task.control
        if self.cancelled.is_set():
            self.cancel()
        return super()._wire_gateway(spec, journal, task, run_id)


class AgentENVAgent(BaseAgent):
    """A Sprout actor on a Harbor-owned sandbox; subclasses choose the agent.

    The environment is any :class:`SandboxEnvironment` -- a microVM or a podman
    container; the agent reads only what that class promises. Subclasses
    provide ``name()``, ``version()`` and ``_make_spec()``. Everything else --
    sharing the environment's session with the orchestrator, Harbor deadline
    handling and the checks on what the rollout left -- is common to every
    agent.

    The orchestrator snapshots each closed turn (disk only) and ends the rollout
    with one more disk snapshot -- and a memory snapshot when the benchmark asks
    (``final_memory_snapshot``), for a grader that needs the task's processes
    alive. A VM whose final disk snapshot failed is retained, never destroyed:
    that snapshot is the only durable copy of the rollout's result.
    """

    async def setup(self, environment: SandboxEnvironment) -> None:
        if not isinstance(environment, SandboxEnvironment) or environment.session is None:
            raise TypeError(f"{type(self).__name__} requires a started SandboxEnvironment")
        if self.mcp_servers:
            raise ValueError("Task MCP service networking is not implemented for Sprout sandboxes")
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    def _instruction(self, instruction: str, environment: SandboxEnvironment) -> str:
        """The task text plus this agent's description of its tools."""
        raise NotImplementedError

    def _make_spec(self, prompt: str, workspace, journal, environment: SandboxEnvironment) -> RunSpec:
        raise NotImplementedError

    async def run(self, instruction: str, environment: SandboxEnvironment, context: AgentContext) -> None:
        actor_started = time.monotonic()
        journal = self.logs_dir / "trajectory.jsonl"
        if journal.exists():
            raise FileExistsError(f"Refusing to overwrite {journal}")
        workspace = self.logs_dir / "actor-workspace"
        workspace.mkdir(parents=True, exist_ok=False)
        prompt = self._instruction(instruction, environment)
        if self.skills_dir:
            prompt += f"\nTask skills are available inside the sandbox at {self.skills_dir}."
        spec = self._make_spec(prompt, workspace, journal, environment)
        orchestrator = AgentENVOrchestrator(environment, self.extra_env, out_dir=self.logs_dir)
        previous_listeners = list(environment.session.on_swap)
        pending = asyncio.create_task(asyncio.to_thread(orchestrator.run, spec))
        context.metadata = {
            "environment_owner": "harbor",
            "backend": environment.type(),
            "checkpoint_mode": environment.checkpoint_mode,
            "checkpoint_capture_enabled": spec.checkpoint_enabled,
            "sandbox_id": environment.session.sandbox_id,
            "journal": str(journal),
        }
        harbor_deadline = False
        try:
            try:
                result = await asyncio.shield(pending)
            except asyncio.CancelledError:
                # Harbor starts its timer immediately before entering run().
                # Preserve ordinary cancellation, but finalize deadline-limited
                # work before Harbor proceeds to its official verifier.
                harbor_deadline = time.monotonic() - actor_started >= max(0, spec.timeout_s - 1)
                orchestrator.cancel(stop_reason="timeout" if harbor_deadline else None)
                result = await asyncio.shield(pending)
                if not harbor_deadline:
                    raise
            context.n_input_tokens = result.usage.get("input_tokens")
            context.n_cache_tokens = result.usage.get("cached_input_tokens")
            context.n_output_tokens = result.usage.get("output_tokens")
            context.cost_usd = result.usage.get("cost_usd")
            context.metadata.update(
                actor_status=result.status,
                checkpoints=result.checkpoints,
                native_session_id=result.native_session_id,
            )
            (self.logs_dir / "execution.json").write_text(json.dumps(asdict(result), default=str, indent=2))
            # First: without the final snapshot the VM is the only copy of the
            # rollout, and Harbor's teardown must not destroy it.
            if result.final_snapshot_id is None:
                environment.retain_reason = "snapshot_pending"
                raise RuntimeError(result.error or "Final actor snapshot failed; sandbox retained")
            # An actor deadline is a valid, judgeable failed rollout rather than
            # an infrastructure failure. Preserve its final filesystem and let
            # Harbor's verifier score it.
            if result.status not in {"completed", "timeout"}:
                raise RuntimeError(result.error or f"Actor {result.status}")
            missing = uncaptured_turns(journal) if spec.checkpoint_enabled else []
            if missing:
                raise RuntimeError(f"Closed turns lack snapshots: {missing}")
            context.metadata["final_snapshot_id"] = result.final_snapshot_id
            context.metadata["final_memory_snapshot_id"] = result.final_memory_snapshot_id
            (self.logs_dir / "snapshot.json").write_text(json.dumps(context.metadata, indent=2))
            if harbor_deadline:
                raise asyncio.CancelledError
        finally:
            environment.actor_control = None
            environment.session.on_swap[:] = previous_listeners
            if journal.exists():
                (self.logs_dir / "trajectory.json").write_text(json.dumps(export_file(journal), indent=2))

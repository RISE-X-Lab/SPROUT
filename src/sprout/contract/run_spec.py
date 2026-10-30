"""What a rollout request asks a worker to run: the fields of one run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

#: The longest time, in seconds, that a queued run keeps its sandbox after it ends (ADR-0004).
KEEP_SANDBOX_S = 1200

#: The slot of each agent that a run can name.
AGENT_SLOTS = {
    "mini": "mini-swe-agent",
    "oracle": "mini-swe-agent",
    "nop": "mini-swe-agent",
    "harness": "sprout-harness",
}


@dataclass
class RunSpec:
    """Everything one run needs. ``python -m sprout.worker run`` sets the fields that a person types."""

    prompt: str
    #: The harness that the agent runs in: ``mini-swe-agent`` for ``mini``, ``oracle`` and ``nop``,
    #: ``sprout-harness`` for ``harness`` (``AGENT_SLOTS``). The runner refuses another pair.
    slot: str = "mini-swe-agent"
    #: What the run does in its sandbox: ``mini`` and ``harness`` run the model in ``slot``,
    #: ``oracle`` runs the ``solution/solve.sh`` of ``harbor_task``, ``nop`` does nothing.
    agent: str = "mini"
    #: The Harbor task directory that the run comes from. The sandbox starts on the task's agent
    #: network policy. An ``oracle`` run requires it.
    harbor_task: str | None = None
    cwd: str = "."
    model: str | None = None
    timeout_s: float = 3600.0
    agent_id: str = "agent"
    run_id: str | None = None
    journal_path: str | Path | None = None

    # --- sandbox ---

    #: The sandbox's backend config, ``{"backend": <name>, <name>: {...}}`` (``worker/sandbox/backends.py``).
    #: Empty means podman, the default backend, which still needs its store and runtime binary.
    backend: dict[str, Any] = field(default_factory=dict)
    #: Lets a spec name the runtime binary, as ``worker run --runtime-bin`` and ``run_defaults`` do.
    #: No backend reads it: each one reads the ``runtime_bin`` of its own section of ``backend``.
    runtime_bin: str | None = None
    #: The image (or snapshot) the sandbox is created from. Required.
    sandbox_image: str | None = None
    #: The shape of the sandbox, such as ``{"cpu": 2, "memory_mb": 8192}``. ``None`` takes the backend's
    #: default, and a default of 2 CPUs and 1 GB makes a build-heavy task run out of memory.
    sandbox_resources: dict[str, Any] | None = None
    #: Keep the sandbox alive after the run, for a grade that runs in it or for
    #: an inspection. A queued run keeps it for at most ``KEEP_SANDBOX_S``.
    keep_sandbox: bool = False
    #: Interceptors that the tool pipeline runs after the mutation tracker, as ``[{"name": ..., **kwargs}]``
    #: with names from ``worker.tools.interceptors.INTERCEPTORS``.
    interceptors: list[dict[str, Any]] = field(default_factory=list)

    # --- gateway ---

    #: The USD ceiling that the gateway enforces on the run's model traffic. Without a rollout contract,
    #: the gateway has no upstream route, and it refuses each model request.
    budget_usd: float | None = None

    # --- checkpoints ---

    #: Take a disk snapshot after each closed model turn. Turn it off for a run that is never
    #: branched. The run takes its final disk snapshot either way.
    checkpoint_enabled: bool = True
    #: Also take a memory snapshot at the end, for a grader that needs the VM's processes alive.
    #: A podman run gets none, because podman snapshots are disk-only.
    final_memory_snapshot: bool = False

    # --- lineage ---

    #: The lineage of a branched run, such as its recovery point. The journal's first event,
    #: ``fork.origin``, records it, so that the journal of a killed branch still names its lineage.
    origin: dict[str, Any] | None = None

    #: Inputs of the agent and the worker, such as ``rollout_contract``, and the retained history of a
    #: branch (``native_prefix``).
    extra: dict[str, Any] = field(default_factory=dict)

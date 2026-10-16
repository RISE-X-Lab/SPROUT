"""What a rollout request asks a worker to run: the fields of one run."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class RunSpec:
    """Everything one run needs; ``harness run`` sets the subset a person types."""

    prompt: str
    #: The agent. mini-swe-agent is the only one; any other name is refused
    #: before a sandbox is made.
    slot: str = "mini-swe-agent"
    cwd: str = "."
    model: str | None = None
    timeout_s: float = 3600.0
    agent_id: str = "agent"
    run_id: str | None = None
    journal_path: str | Path | None = None

    # --- sandbox ---
    #: Where the sandbox comes from (``harness_sandbox/backends.py``):
    #: ``{"backend": "microvm", "microvm": {...}}`` for an AgentENV microVM, or
    #: ``{"backend": "podman", "podman": {"store": ..., "runtime_bin": ...}}``
    #: for a rootless container whose rootfs is a subvolume of the store. Empty
    #: means podman, the default backend, whose section must still name the
    #: store and the runtime binary.
    backend: dict[str, Any] = field(default_factory=dict)
    #: The sprout-runtime binary named on the command line (``harness run
    #: --runtime-bin``) or in a Run Store job's ``run_defaults``. Accepted for
    #: that contract (``runstore/specs.py`` validates job specs against these
    #: fields) and not otherwise consumed: the binary a backend actually uses is
    #: the one in its own section -- ``backend["podman"]["runtime_bin"]`` is
    #: mounted into every container, ``backend["microvm"]["runtime_bin"]`` builds
    #: the per-image template (``harness_sandbox/templates.py``).
    runtime_bin: str | None = None
    #: The image (or snapshot) the sandbox is created from. Required.
    sandbox_image: str | None = None
    #: Shape for that sandbox, ``{"cpu": 2, "memory_mb": 8192}``. A task that
    #: declares its needs must get them; the backend's default (2 CPUs, 1 GB)
    #: OOMs a build-heavy task instead of failing loudly. None = default.
    sandbox_resources: dict[str, Any] | None = None
    #: Keep the sandbox after the run -- grading and extraction need it alive.
    keep_sandbox: bool = False
    #: Rules mounted on the serving pipeline after the mutation tracker, by name
    #: (``harness_tools.interceptors.INTERCEPTORS``): ``[{"name": ..., **kwargs}]``.
    interceptors: list[dict[str, Any]] = field(default_factory=list)

    # --- gateway ---
    #: USD ceiling the gateway enforces on this run's model traffic. Set by the
    #: RL contract and by Run Store continuations, never by hand: the gateway
    #: starts for a run with rollout controls, and a budget on its own has no
    #: upstream to route to.
    budget_usd: float | None = None

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
    origin: dict[str, Any] | None = None

    extra: dict[str, Any] = field(default_factory=dict)

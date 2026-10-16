"""Restore an exact mini history and sandbox snapshot inside a Harbor trial."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from sprout.contract.continuation import Continuation, prepare
from sprout.contract.mini_history import load_prefix
from sprout.harbor_plugin.agentenv import AgentENVEnvironment
from sprout.harbor_plugin.agentenv_mini import AgentENVMini
from sprout.harbor_plugin.podman import PodmanEnvironment
from sprout.harness_agents.mini_swe.history import actor_tools_at, available_branch_points, reference_at

#: A snapshot id as a backend names one: AgentENV's UUID, or a podman subvolume
#: name (``<sandbox id>-<counter>[-<name>]``). Nothing a path could be built
#: from: no separators, no leading dot.
_SNAPSHOT_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]*")


def continuation_of(context: dict) -> Continuation:
    """The continuation a recorded branch context asks for.

    The context keeps the reviewer's hint beside any guidance, so only a
    ``user-hint`` branch hands it on.
    """
    guidance = context.get("branch_guidance")
    return Continuation(
        guidance,
        assistant_turn=context.get("assistant_turn"),
        hint=context.get("hint") if guidance == "user-hint" else None,
    )


def read_branch_context(path: str | Path) -> dict:
    context = json.loads(Path(path).read_text())
    continuation_of(context)
    if not _SNAPSHOT_ID.fullmatch(str(context.get("snapshot_id") or "")):
        raise ValueError("A branch requires an explicit snapshot id")
    if context.get("checkpoint_mode") != "disk_only":
        raise ValueError("Harbor branch requires a parent with per-call snapshots")
    if "@sha256:" not in context.get("base_image", ""):
        raise ValueError("The original task image digest must be pinned")
    if not isinstance(context.get("image_config"), dict):
        raise ValueError("The original task image config must be recorded")
    if type(context.get("step")) is not int or context["step"] < 1:
        raise ValueError("A branch needs a positive parent tool step")
    if not isinstance(context.get("native_session_id"), str) or not context["native_session_id"]:
        raise ValueError("A branch needs the parent native session ID")
    if not isinstance(context.get("conversation_cut"), str) or not context["conversation_cut"]:
        raise ValueError("A branch needs the exact native cut")
    parent = Path(context.get("parent_journal", ""))
    if not parent.is_file() or hashlib.sha256(parent.read_bytes()).hexdigest() != context.get(
        "parent_journal_sha256"
    ):
        raise ValueError("Parent journal is missing or changed")
    if not Path(context.get("task_dir", "")).is_dir():
        raise ValueError("A branch needs the Harbor task directory it belongs to")
    return context


class SnapshotRestore:
    """Boot Harbor's task environment from a selected snapshot.

    Mixed into a backend's environment class ahead of it (``class
    SnapshotEnvironment(SnapshotRestore, AgentENVEnvironment)``): the snapshot
    id replaces the image, and the session creates the sandbox from it the way
    it would from an image -- what differs per backend stays in the backend
    class. Harbor builds a trial's separate verifier from the same environment
    configuration, so the class also serves as the verifier environment. The
    two are told apart by the directory Harbor hands over: the agent
    environment is built from the task's ``environment/``, the verifier from
    its ``tests/``. Only the former restores the snapshot; the verifier boots
    the pristine task image and replays ``tests/Dockerfile`` like the base class.
    """

    def __init__(self, *args, branch_context: str, **kwargs):
        self.branch_context_path = Path(branch_context)
        self.branch_context = read_branch_context(self.branch_context_path)
        super().__init__(*args, **kwargs)
        if self.checkpoint_mode != self.branch_context["checkpoint_mode"]:
            raise ValueError("Branch environment checkpoint mode differs from its parent")

    @property
    def restores_snapshot(self) -> bool:
        task_dir = Path(self.branch_context["task_dir"]).resolve()
        return Path(self.environment_dir).resolve() == task_dir / "environment"

    async def _prepare_image(self, force_build: bool):
        if not self.restores_snapshot:
            return await super()._prepare_image(force_build)
        if force_build:
            raise ValueError("A branch must restore its snapshot, not rebuild the task image")
        return self.branch_context["snapshot_id"], self.branch_context["image_config"]

    async def _populate(self):
        # Restored task files are already in the snapshot. A fresh upload would
        # overwrite edits made before the selected checkpoint.
        if not self.restores_snapshot:
            await super()._populate()

    async def start(self, force_build: bool = False) -> None:
        await super().start(force_build=force_build)
        if not self.restores_snapshot:
            return
        receipt = {
            "snapshot_id": self.branch_context["snapshot_id"],
            "sandbox_id": self.session.sandbox_id,
            "base_image": self.branch_context["base_image"],
            "branch_context": str(self.branch_context_path),
            "checkpoint_mode": self.checkpoint_mode,
        }
        (self.trial_paths.trial_dir / "restored-state.json").write_text(json.dumps(receipt, indent=2))


class SnapshotEnvironment(SnapshotRestore, AgentENVEnvironment):
    """A branch on a microVM: the AgentENV snapshot is the image."""


class PodmanSnapshotEnvironment(SnapshotRestore, PodmanEnvironment):
    """A branch on podman: the read-only subvolume the id names is the image.

    Snapshots live in one store, so the branch runs with the ``store`` its
    parent recorded; the session's podman backend gives the new container a
    writable snapshot of the subvolume and the task's shape again.
    """


class BranchMini(AgentENVMini):
    """Continue from an exact point, with assistant-turn or neutral guidance."""

    def __init__(self, *args, branch_context: str, **kwargs):
        self.branch_context_path = Path(branch_context)
        self.branch_context = read_branch_context(self.branch_context_path)
        super().__init__(*args, **kwargs)

    @staticmethod
    def name() -> str:
        return "sprout-agentenv-mini-branch"

    def _make_spec(self, prompt, workspace, journal, environment):
        context = self.branch_context
        if (
            not isinstance(environment, SnapshotRestore)
            or environment.branch_context_path != self.branch_context_path
        ):
            raise ValueError("Agent and environment must use the same branch context")
        parent = Path(context["parent_journal"])
        if hashlib.sha256(parent.read_bytes()).hexdigest() != context["parent_journal_sha256"]:
            raise ValueError("Parent journal changed after branch selection")
        point = available_branch_points(parent).get(context["step"])
        if (
            point is None
            or point.snapshot_id != context["snapshot_id"]
            or point.session_ckpt != context["native_session_id"]
        ):
            raise ValueError("Branch snapshot/session does not match the parent journal")
        reference = reference_at(parent, context["step"], context["native_session_id"])
        if reference is None or reference.get("cut") != context["conversation_cut"]:
            raise ValueError("Branch native prefix does not match the selected cut")
        continuation = continuation_of(context)
        tools = actor_tools_at(parent, context["step"]) if continuation.guidance == "assistant-turn" else None
        note, options = prepare(
            continuation,
            load_prefix(reference),
            tools=tools,
            commit_required=bool(context.get("commit_required")),
        )
        spec = super()._make_spec(note, workspace, journal, environment)
        spec.checkpoint_enabled = bool(context.get("capture_checkpoints", True))
        spec.extra["native_prefix"] = reference
        spec.extra.update(options)
        spec.origin = {
            "parent_journal": str(parent),
            "branch_step": context["step"],
            "snapshot_id": context["snapshot_id"],
            "conversation_cut": context["conversation_cut"],
            **continuation.origin(),
            "parent_journal_sha256": context["parent_journal_sha256"],
        }
        return spec

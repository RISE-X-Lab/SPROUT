"""Per-step environment checkpoints for a rollout.

An RL rollout that can restart from any step needs, for each step, a snapshot
of the environment as it stood *after* that step. Two observations make that
cheap enough to do on every turn:

**Only mutations matter.** The runtime's shell is stateless (every call is a
fresh ``sh -c`` with an explicit working directory), so nothing carries across
steps except the filesystem. :class:`MutationTracker` records whether any call
since the last capture could have changed it: every ``shell`` call does,
because a command's effect is not in its text, while a search, a fetch or a
``text_editor view`` does not. A call that one of the run's declared rules then
refused counts too: the tracker is mounted ahead of them, and over-counting
costs one cheap capture. A step whose calls could not have changed anything is
recorded as ``clean`` and reuses the last snapshot id; the map from step to
snapshot stays complete either way.

**A checkpoint need not include memory.** Replay restores the disk and
re-feeds the transcript, so the VM's memory image is dead weight -- hence
``disk_only`` snapshots. The cost of one is roughly the bytes the step wrote.

One requirement comes with ``disk_only``: a sandbox created from such a
snapshot **cold-boots**, so its runtime is only there if the microVM template
declares a startup command that launches it (AgentENV re-runs a snapshot's
startup command after a cold boot; a template captured from a hand-started
process carries none). Without one, re-boarding and replaying land on
sandboxes whose runtime is missing; build the template through AgentENV's
template API with ``startCmd`` set (``aenv snapshot create`` records none).
Sprout cannot repair that from outside -- it reaches a sandbox only through
its runtime -- so :meth:`~harness_sandbox.session.SandboxSession.swap_sandbox`
probes a replacement before adopting it and keeps the current sandbox when the
probe fails. (A podman container has no template: the pool mounts the runtime
into every container it starts, so a restore always has one.)

The remaining subtlety is layer accounting, and it is the microVM backend's
alone. Each capture there adds a layer to the sandbox's chain; the server
compacts the chain once it crosses a configured budget, and a compacted capture
is visible here as the layer count *dropping*. The running sandbox's own layer
stack is never compacted, so from then on every capture would re-compact the
whole chain -- unless the session continues on a sandbox started from that
compact snapshot. That is what :meth:`Checkpointer.after_step` does when it
sees the count drop: it re-boards. Watching for the drop rather than counting
steps means the policy follows whatever trigger the server is configured with
(layer count, chain size, or both) with nothing to keep in sync. A podman
snapshot is a btrfs subvolume: nothing stacks, so it reports no layer count
(``rootfs_layers`` is ``None``) and none of this applies to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

from sprout.harness.tools.interceptors.mutation import MutationTracker

#: Re-boarding resets the *suffix* of the layer chain but ratchets the inherited
#: prefix up by one layer per cycle -- captures can never compact layers the
#: sandbox did not produce, so the prefix only grows, and at AgentENV's
#: 255-layer stack cap the lineage dies (~20k captures from a fresh start).
#: Squashing is the repository-side merge that CAN fold the prefix; doing it at
#: a re-board whose target is already this deep resets the ratchet and removes
#: the ceiling on trajectory length. Costs one repo-side merge of the whole
#: chain (chain bytes / store write bandwidth) plus a duplicate of that data
#: until the old layers are reclaimed.
SQUASH_LINEAGE_AT = 128


def _failed_to_grow(previous: Optional[int], current: Optional[int]) -> bool:
    return previous is not None and current is not None and current <= previous


@dataclass
class CheckpointRecord:
    """One step's checkpoint, as recorded in the trajectory."""

    turn: int
    #: None only for a failed capture with no earlier snapshot to fall back to.
    snapshot_id: Optional[str]
    #: False when this step did not produce its own snapshot.
    captured: bool
    #: Why, when it did not. "clean" means nothing could have changed and the
    #: previous snapshot already is this step's state -- free and expected.
    #: "failed" means a capture was attempted and did not happen, which is an
    #: alarm: checkpointing has stopped working. Recording only `captured`
    #: conflated the two, and a run whose captures were all failing looked
    #: exactly like a run with a very clean workload.
    reason: str = "captured"
    disk_only: bool = False
    #: Set when this checkpoint triggered continuing on a new sandbox.
    reboarded: bool = False
    delta_empty: Optional[bool] = None


@dataclass
class Checkpointer:
    """Drives per-step checkpoints and layer-chain upkeep for one episode.

    ``session`` is a :class:`~harness_sandbox.session.SandboxSession`; only its
    ``snapshot`` / ``squash_snapshot`` / ``swap_sandbox`` / ``supports_snapshot``
    surface is used, so a test double is a small object.
    """

    session: object
    tracker: Optional[MutationTracker] = None
    disk_only: bool = True

    latest_snapshot_id: Optional[str] = None
    _failed_captures: int = 0
    _previous_layers: Optional[int] = None
    _consecutive_compactions: int = 0
    _warned_budget: bool = False

    def enabled(self) -> bool:
        return bool(getattr(self.session, "supports_snapshot", lambda: False)())

    def after_step(self, turn: int) -> Optional[CheckpointRecord]:
        """Checkpoint the environment as it stands after step ``turn``.

        Returns the record, or ``None`` when checkpointing is unavailable and
        nothing was recorded. A re-board happens on one trigger: the server
        compacted the chain, seen as a layer count that failed to grow.
        """
        if not self.enabled():
            return None

        mutated = self.tracker is None or self.tracker.dirty
        if not mutated and self.latest_snapshot_id:
            # Nothing could have changed: the previous snapshot is this step's
            # state, so record the mapping without paying for a capture.
            return CheckpointRecord(
                turn=turn, snapshot_id=self.latest_snapshot_id,
                captured=False, reason="clean", disk_only=self.disk_only)

        snapshot = self.session.snapshot(disk_only=self.disk_only)
        if snapshot is None:
            # Capture failed (or the backend declined). Fall back to the last
            # good snapshot so the map stays complete and monotonic -- but say
            # so, because a run that has silently stopped checkpointing is
            # indistinguishable from one with nothing to capture.
            self._failed_captures += 1
            if self._failed_captures in (1, 10, 100) or (
                    self._failed_captures % 250 == 0):
                logging.getLogger(__name__).warning(
                    "checkpoint capture has failed %d time(s); steps are "
                    "mapping to an older snapshot and replay from them will "
                    "be wrong", self._failed_captures)
            return CheckpointRecord(
                turn=turn, snapshot_id=self.latest_snapshot_id,
                captured=False, reason="failed", disk_only=self.disk_only)

        if self.tracker is not None:
            self.tracker.clear()
        self.latest_snapshot_id = snapshot.id
        record = CheckpointRecord(
            turn=turn, snapshot_id=snapshot.id, captured=True,
            disk_only=self.disk_only,
            delta_empty=getattr(snapshot, "delta_empty", None),
        )

        layers = snapshot.rootfs_layers
        if layers is not None:
            # Every uncompacted capture adds exactly one layer to the rootfs
            # chain, so a count that FAILED TO GROW means the server compacted
            # it and this snapshot is its compact base: continue from it, or
            # every later capture re-compacts. `<=` rather than `<`: a chain
            # whose per-cycle suffix merges into a single layer compacts at the
            # same count it had before -- seen live as a constant 13 with a
            # re-compaction every step and a drop that never came.
            compacted = _failed_to_grow(self._previous_layers, layers)
            if compacted:
                target = snapshot
                if self._lineage_needs_squash(snapshot):
                    # A failed squash falls back to the plain re-board: the
                    # ratchet keeps walking and the next re-board retries.
                    target = self._squashed(snapshot) or snapshot
                record.reboarded = bool(self.session.swap_sandbox(target))
            self._previous_layers = None if record.reboarded else layers

            # A budget smaller than a single step's writes degrades silently:
            # every capture compacts, and with re-boarding that means a swap
            # every other step. It still works -- say so once, loudly, because
            # the fix is a config value, not code.
            self._consecutive_compactions = (
                self._consecutive_compactions + 1 if compacted else 0)
            if self._consecutive_compactions >= 3 and not self._warned_budget:
                self._warned_budget = True
                logging.getLogger(__name__).warning(
                    "every recent capture compacted the layer chain: the "
                    "server's compaction budget (max_chain_size_mib) is "
                    "likely smaller than a single step's writes for this "
                    "workload; raise it to amortize compaction over more "
                    "steps")

        return record

    def _squashed(self, snapshot):
        """The flattened copy of ``snapshot``, or ``None`` when squashing changed nothing."""
        squash = getattr(self.session, "squash_snapshot", None)
        if not callable(squash):
            return None
        squashed = squash(snapshot)
        squashed_id = getattr(squashed, "id", None)
        if not squashed_id or squashed_id == snapshot.id:
            return None
        return squashed

    def _lineage_needs_squash(self, snapshot) -> bool:
        """Whether a re-board target's inherited prefix has grown deep.

        A just-compacted snapshot is prefix + one merged layer, so its layer
        count IS the lineage depth.
        """
        if not callable(getattr(self.session, "squash_snapshot", None)):
            return False
        return (snapshot.rootfs_layers or 0) > SQUASH_LINEAGE_AT

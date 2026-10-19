"""Per-step environment checkpoints for one rollout: ``Checkpointer``.

``Checkpointer.after_step`` records, for each step, a snapshot of the environment as it stands
after that step. The map from step to snapshot is complete:

- If the run's ``MutationTracker`` saw no call that can change the filesystem, the step reuses
  the last snapshot, as ``clean``.
- A failed capture also reuses the last snapshot, and its record says ``failed``.

Snapshots are disk-only by default. A sandbox started from one cold-boots, so a microVM template
must declare a startup command that launches the runtime.

On the microVM backend, each capture adds a layer to the sandbox's snapshot chain. When the server
compacts the chain, the session continues on a sandbox started from the compact snapshot. A podman
snapshot reports no layer count, so this never applies to it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Protocol

from sprout.worker.tools.interceptors.mutation import MutationTracker

if TYPE_CHECKING:
    from sprout_sandbox import Snapshot

#: Lineage depth, in layers, above which a re-board squashes its target first. AgentENV caps a
#: layer stack at 255, and each re-board adds a layer to the prefix that no capture can compact.
SQUASH_LINEAGE_AT = 128


def _failed_to_grow(previous: int | None, current: int | None) -> bool:
    """Whether the layer count stayed or dropped between two captures.

    An equal count is a compaction too: a chain whose per-cycle suffix merges into one layer
    compacts at the count that it had before.
    """
    return previous is not None and current is not None and current <= previous


class _SnapshotSession(Protocol):
    """The part of a ``SandboxSession`` that a ``Checkpointer`` calls.

    ``supports_snapshot`` and ``squash_snapshot`` are optional. Without them, the session has no
    snapshots and nothing to squash.
    """

    def snapshot(self, *, disk_only: bool) -> Snapshot | None: ...

    def swap_sandbox(self, snapshot: Snapshot) -> bool: ...


@dataclass
class CheckpointRecord:
    """One step's checkpoint, as recorded in the trajectory."""

    turn: int
    #: None only for a failed capture with no earlier snapshot to fall back to.
    snapshot_id: str | None
    #: False when this step did not produce its own snapshot.
    captured: bool
    #: ``captured``, ``clean`` (nothing could have changed, so the previous snapshot holds) or
    #: ``failed`` (a capture was attempted and did not happen: checkpointing has stopped working).
    reason: str = "captured"
    disk_only: bool = False
    #: Set when this checkpoint triggered continuing on a new sandbox.
    reboarded: bool = False
    delta_empty: bool | None = None


@dataclass
class Checkpointer:
    """Takes the per-step checkpoints of one episode, and re-boards it after a chain compaction.

    ``session`` needs ``snapshot`` and ``swap_sandbox``. Its ``supports_snapshot`` and
    ``squash_snapshot`` are optional, so a test double can be a small object.
    """

    session: _SnapshotSession
    #: The run's mutation tracker. ``None`` captures after every step.
    tracker: MutationTracker | None = None
    disk_only: bool = True

    latest_snapshot_id: str | None = None
    _failed_captures: int = 0
    _previous_layers: int | None = None
    _consecutive_compactions: int = 0
    _warned_budget: bool = False

    def enabled(self) -> bool:
        """Whether the session can take snapshots."""
        return bool(getattr(self.session, "supports_snapshot", lambda: False)())

    def after_step(self, turn: int) -> CheckpointRecord | None:
        """Checkpoint the environment as it stands after step ``turn``, and return the record.

        Returns ``None`` if the session cannot take snapshots. If the server has compacted the layer
        chain, the session continues on a sandbox started from the compact snapshot (``reboarded``).
        """
        if not self.enabled():
            return None

        mutated = self.tracker is None or self.tracker.dirty
        if not mutated and self.latest_snapshot_id:
            # Nothing could have changed: the previous snapshot is this step's
            # state, so record the mapping without paying for a capture.
            return CheckpointRecord(
                turn=turn,
                snapshot_id=self.latest_snapshot_id,
                captured=False,
                reason="clean",
                disk_only=self.disk_only,
            )

        snapshot = self.session.snapshot(disk_only=self.disk_only)
        if snapshot is None:
            # The record falls back to the last good snapshot, so the map stays complete and monotonic.
            # Without the warning, a run that stopped checkpointing looks like a run with nothing to capture.
            self._failed_captures += 1
            if self._failed_captures in (1, 10, 100) or (self._failed_captures % 250 == 0):
                logging.getLogger(__name__).warning(
                    "checkpoint capture has failed %d time(s); steps are "
                    "mapping to an older snapshot and replay from them will "
                    "be wrong",
                    self._failed_captures,
                )
            return CheckpointRecord(
                turn=turn,
                snapshot_id=self.latest_snapshot_id,
                captured=False,
                reason="failed",
                disk_only=self.disk_only,
            )

        if self.tracker is not None:
            self.tracker.clear()
        self.latest_snapshot_id = snapshot.id
        record = CheckpointRecord(
            turn=turn,
            snapshot_id=snapshot.id,
            captured=True,
            disk_only=self.disk_only,
            # A session double can return a snapshot without delta_empty.
            delta_empty=getattr(snapshot, "delta_empty", None),
        )

        layers = snapshot.rootfs_layers
        if layers is not None:
            # Each uncompacted capture adds one layer, so a count that fails to grow means that the server
            # compacted the chain. Without a re-board, each later capture compacts the whole chain again.
            compared = self._previous_layers is not None
            compacted = _failed_to_grow(self._previous_layers, layers)
            if compacted:
                target = snapshot
                if self._lineage_needs_squash(snapshot):
                    # A failed squash falls back to a plain re-board. The lineage grows one layer,
                    # and the next re-board tries the squash again.
                    target = self._squashed(snapshot) or snapshot
                record.reboarded = bool(self.session.swap_sandbox(target))
            self._previous_layers = None if record.reboarded else layers

            # A compaction budget below one step's writes compacts at each capture, and the run swaps
            # sandboxes every other step. The fix is a server config value, so warn once.
            if compacted:
                self._consecutive_compactions += 1
            elif compared:
                # The capture after a re-board has no count to compare, so it does not end the row.
                self._consecutive_compactions = 0
            if self._consecutive_compactions >= 3 and not self._warned_budget:
                self._warned_budget = True
                logging.getLogger(__name__).warning(
                    "every recent capture compacted the layer chain: the "
                    "server's compaction budget (max_chain_size_mib) is "
                    "likely smaller than a single step's writes for this "
                    "workload; raise it to amortize compaction over more "
                    "steps"
                )

        return record

    def _squashed(self, snapshot: Snapshot) -> Snapshot | None:
        """The flattened copy of ``snapshot``, or ``None`` if the session cannot squash or nothing changed."""
        squash = getattr(self.session, "squash_snapshot", None)
        if not callable(squash):
            return None
        squashed: Snapshot | None = squash(snapshot)
        squashed_id = getattr(squashed, "id", None)
        if not squashed_id or squashed_id == snapshot.id:
            return None
        return squashed

    def _lineage_needs_squash(self, snapshot: Snapshot) -> bool:
        """Whether the session can squash, and a re-board target is deeper than ``SQUASH_LINEAGE_AT``.

        A snapshot that was just compacted is the inherited prefix plus one merged layer, so its layer
        count is the lineage depth.
        """
        if not callable(getattr(self.session, "squash_snapshot", None)):
            return False
        return (snapshot.rootfs_layers or 0) > SQUASH_LINEAGE_AT

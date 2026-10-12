"""Checkpointer layer-chain upkeep: when a step's checkpoint re-boards the run.

One trigger: the server compacted the chain (the layer count failed to grow), so
the run re-boards onto the compact snapshot, squashed first when the lineage is
deeper than SQUASH_LINEAGE_AT. That is the microVM backend's accounting; a
podman snapshot is a btrfs subvolume, reports no layer count, and never
re-boards (the last case).

A fake session hands out snapshots with the layer counts a case dictates and
records what was squashed and swapped; everything else is the real Checkpointer.
"""

from __future__ import annotations

import pytest

from sprout_sandbox.pool import Snapshot
from sprout.harness.checkpoint.checkpointer import SQUASH_LINEAGE_AT, Checkpointer


class Session:
    """A microVM-shaped session: snapshots, squashes and swaps."""

    def __init__(self, layer_counts, flatten=True):
        self._layers = iter(layer_counts)
        self.flatten = flatten
        self.count = 0
        self.squashed: list[str] = []
        self.swapped: list[str] = []

    def supports_snapshot(self):
        return True

    def snapshot(self, disk_only=True, name=None):
        self.count += 1
        return Snapshot("snap-%d" % self.count, rootfs_layers=next(self._layers),
                        disk_only=disk_only)

    def squash_snapshot(self, snapshot, name=None):
        self.squashed.append(snapshot.id)
        if not self.flatten:
            return snapshot
        return Snapshot(snapshot.id + "-flat", rootfs_layers=1, disk_only=True)

    def swap_sandbox(self, snapshot):
        self.swapped.append(snapshot.id)
        return True


def run(session, steps):
    checkpointer = Checkpointer(session=session)
    records = [checkpointer.after_step(turn) for turn in range(1, steps + 1)]
    return checkpointer, records


# --------------------------------------------------------------------------- #
#  The compaction rule, as one decision table (the podman row last)
# --------------------------------------------------------------------------- #

DEEP = SQUASH_LINEAGE_AT + 10  # a lineage past the squash threshold


@pytest.mark.parametrize(
    "layer_counts, flatten, reboarded, squashed, swapped",
    [
        pytest.param(
            # Two hundred layers, each capture one more: no compaction happened,
            # so nothing is squashed or swapped -- not even once the chain is
            # deeper than SQUASH_LINEAGE_AT, since depth alone triggers nothing.
            range(1, 201), True, [False] * 200, [], [],
            id="growing-chain"),
        pytest.param(
            # The server compacted (an equal count, not a drop: seen live as a
            # constant 13): continue on the compact snapshot, with no squash
            # because the lineage is shallow. A re-board forgets the previous
            # count, so the fourth capture's repeated 13 is one step of grace,
            # not another compaction.
            [12, 13, 13, 13], True, [False, False, True, False], [], ["snap-3"],
            id="shallow-compaction"),
        pytest.param(
            # A deep lineage is squashed first, and the FLATTENED copy is what
            # the run re-boards onto.
            [DEEP - 1, DEEP, DEEP], True, [False, False, True], ["snap-3"], ["snap-3-flat"],
            id="deep-compaction-squashed"),
        pytest.param(
            # A squash that hands back its input is no squash: the plain
            # re-board onto the compact snapshot still happens.
            [DEEP, DEEP], False, [False, True], ["snap-2"], ["snap-2"],
            id="failed-squash-falls-back"),
        pytest.param(
            # The podman pool's snapshots: btrfs subvolumes, nothing stacks, no
            # count. Every capture is recorded and none re-boards, whatever
            # came before.
            [None] * 5, True, [False] * 5, [], [],
            id="podman-no-layer-count"),
    ])
def test_the_compaction_rule(layer_counts, flatten, reboarded, squashed, swapped):
    """One step per dictated layer count; which steps re-board, what was
    squashed on the way and what the session ended up swapped onto."""
    session = Session(layer_counts, flatten=flatten)
    _, records = run(session, len(layer_counts))
    assert all(r.captured for r in records)
    assert [r.reboarded for r in records] == reboarded
    assert session.squashed == squashed and session.swapped == swapped

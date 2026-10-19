"""The per-step checkpoints of a rollout: when a step gets a snapshot, and when a compaction re-boards it.

The sessions here hand out snapshots with the layer counts that a case gives, and record what they
squash and swap, because a squash and a swap are the effects of a re-board on the session.
"""

from __future__ import annotations

import logging

import pytest

from sprout.worker.checkpoint.checkpointer import SQUASH_LINEAGE_AT, Checkpointer, CheckpointRecord
from sprout.worker.tools.interceptors.mutation import MutationTracker
from sprout.worker.tools.pipeline import CallContext
from sprout_sandbox.pool import Snapshot


class Session:
    """A session that snapshots. Capture N has ``layer_counts[N-1]`` layers, or fails for N in ``failing``."""

    def __init__(
        self, layer_counts: list[int | None] | None = None, *, failing: tuple[int, ...] = ()
    ) -> None:
        self.layer_counts = list(layer_counts or [])
        self.failing = set(failing)
        self.captures = 0
        self.squashed: list[str] = []
        self.swapped: list[str] = []

    def supports_snapshot(self) -> bool:
        return True

    def snapshot(self, *, disk_only: bool) -> Snapshot | None:
        self.captures += 1
        if self.captures in self.failing:
            return None
        layers = self.layer_counts[self.captures - 1] if self.layer_counts else None
        return Snapshot("snapshot-%d" % self.captures, rootfs_layers=layers, disk_only=disk_only)

    def swap_sandbox(self, snapshot: Snapshot) -> bool:
        self.swapped.append(snapshot.id)
        return True


class SquashingSession(Session):
    """A microVM session: its squash returns the flattened copy ``<id>-flat``."""

    def squash_snapshot(self, snapshot: Snapshot) -> Snapshot:
        self.squashed.append(snapshot.id)
        return Snapshot(snapshot.id + "-flat", rootfs_layers=1, disk_only=True)


class UnchangedSquashSession(SquashingSession):
    """A session whose squash fails, and so returns its input."""

    def squash_snapshot(self, snapshot: Snapshot) -> Snapshot:
        self.squashed.append(snapshot.id)
        return snapshot


class StuckSession(Session):
    """A session whose re-board fails: the replacement sandbox has no runtime."""

    def swap_sandbox(self, snapshot: Snapshot) -> bool:
        self.swapped.append(snapshot.id)
        return False


class NoSnapshotSession:
    """A session of a backend that cannot snapshot."""

    def __init__(self) -> None:
        self.captures = 0

    def supports_snapshot(self) -> bool:
        return False

    def snapshot(self, *, disk_only: bool) -> Snapshot | None:
        self.captures += 1
        return Snapshot("snapshot-%d" % self.captures)

    def swap_sandbox(self, snapshot: Snapshot) -> bool:
        return False


def steps(checkpointer: Checkpointer, count: int) -> list[CheckpointRecord | None]:
    """The records of steps 1 to ``count``, checkpointed in order."""
    return [checkpointer.after_step(step) for step in range(1, count + 1)]


def shell_call(tracker: MutationTracker) -> None:
    """Pass one ``shell`` call through ``tracker``, as the run's pipeline does."""
    tracker.before(
        CallContext(agent_id="agent", sandbox_id="sandbox-1", tool_name="shell", args={"command": "ls"})
    )


def test_a_session_that_cannot_snapshot_gets_no_checkpoint_and_no_capture():
    session = NoSnapshotSession()

    record = Checkpointer(session=session).after_step(1)

    assert (record, session.captures) == (None, 0)


def test_a_step_whose_calls_could_not_change_the_environment_reuses_the_last_snapshot():
    session, tracker = Session(), MutationTracker()
    checkpointer = Checkpointer(session=session, tracker=tracker)
    shell_call(tracker)
    checkpointer.after_step(1)

    record = checkpointer.after_step(2)

    assert (record, session.captures) == (
        CheckpointRecord(turn=2, snapshot_id="snapshot-1", captured=False, reason="clean", disk_only=True),
        1,
    )


def test_the_first_step_is_captured_also_when_nothing_could_have_changed():
    session = Session()

    record = Checkpointer(session=session, tracker=MutationTracker()).after_step(1)

    assert record == CheckpointRecord(turn=1, snapshot_id="snapshot-1", captured=True, disk_only=True)


def test_without_a_tracker_every_step_is_captured():
    session = Session()
    checkpointer = Checkpointer(session=session)

    records = steps(checkpointer, 2)

    assert [(record.snapshot_id, record.reason) for record in records] == [
        ("snapshot-1", "captured"),
        ("snapshot-2", "captured"),
    ]


def test_a_failed_capture_maps_the_step_to_the_last_snapshot_and_says_that_it_failed():
    session, tracker = Session(failing=(2,)), MutationTracker()
    checkpointer = Checkpointer(session=session, tracker=tracker)
    shell_call(tracker)
    checkpointer.after_step(1)
    shell_call(tracker)

    record = checkpointer.after_step(2)

    assert record == CheckpointRecord(
        turn=2, snapshot_id="snapshot-1", captured=False, reason="failed", disk_only=True
    )


def test_the_step_after_a_failed_capture_is_captured_without_a_new_change():
    """A failed capture leaves the change of its step unrecorded, so the next step must not be clean."""
    session, tracker = Session(failing=(2,)), MutationTracker()
    checkpointer = Checkpointer(session=session, tracker=tracker)
    shell_call(tracker)
    checkpointer.after_step(1)
    shell_call(tracker)
    checkpointer.after_step(2)

    record = checkpointer.after_step(3)

    assert (record.snapshot_id, record.reason) == ("snapshot-3", "captured")


def test_a_failed_capture_is_logged_as_a_warning(caplog):
    """Without the warning, a run that stopped checkpointing looks like a run with nothing to capture."""
    checkpointer = Checkpointer(session=Session(failing=(1,)))

    with caplog.at_level(logging.WARNING, logger="sprout.worker.checkpoint.checkpointer"):
        checkpointer.after_step(1)

    assert caplog.messages == [
        "checkpoint capture has failed 1 time(s); steps are mapping to an older snapshot "
        "and replay from them will be wrong"
    ]


DEEP = SQUASH_LINEAGE_AT + 10


@pytest.mark.parametrize(
    "session,reboarded,squashed,swapped",
    [
        # The count grows at each capture, also past the squash depth: no compaction.
        pytest.param(
            SquashingSession([DEEP, DEEP + 1, DEEP + 2]), [False, False, False], [], [], id="growing-chain"
        ),
        # A live run showed a constant count, not a drop, at each compaction. A re-board forgets
        # the count, so the capture after it is not a compaction.
        pytest.param(
            SquashingSession([12, 13, 13, 13]),
            [False, False, True, False],
            [],
            ["snapshot-3"],
            id="equal-count",
        ),
        pytest.param(
            SquashingSession([12, 13, 5]), [False, False, True], [], ["snapshot-3"], id="dropped-count"
        ),
        pytest.param(
            SquashingSession([SQUASH_LINEAGE_AT, SQUASH_LINEAGE_AT]),
            [False, True],
            [],
            ["snapshot-2"],
            id="lineage-at-the-squash-depth",
        ),
        pytest.param(
            SquashingSession([SQUASH_LINEAGE_AT + 1, SQUASH_LINEAGE_AT + 1]),
            [False, True],
            ["snapshot-2"],
            ["snapshot-2-flat"],
            id="lineage-past-the-squash-depth",
        ),
        pytest.param(
            UnchangedSquashSession([DEEP, DEEP]),
            [False, True],
            ["snapshot-2"],
            ["snapshot-2"],
            id="squash-fails",
        ),
        pytest.param(Session([DEEP, DEEP]), [False, True], [], ["snapshot-2"], id="session-cannot-squash"),
        # A podman snapshot is a btrfs subvolume: it reports no count, and nothing stacks.
        pytest.param(
            SquashingSession([None, None, None]), [False, False, False], [], [], id="no-layer-count"
        ),
    ],
)
def test_a_compacted_chain_re_boards_the_run_onto_the_compact_snapshot_squashed_when_deep(
    session, reboarded, squashed, swapped
):
    checkpointer = Checkpointer(session=session)

    records = steps(checkpointer, len(session.layer_counts))

    assert ([record.reboarded for record in records], session.squashed, session.swapped) == (
        reboarded,
        squashed,
        swapped,
    )


def test_a_failed_re_board_is_tried_again_at_the_next_compaction():
    session = StuckSession([12, 12, 12])
    checkpointer = Checkpointer(session=session)

    records = steps(checkpointer, 3)

    assert ([record.reboarded for record in records], session.swapped) == (
        [False, False, False],
        ["snapshot-2", "snapshot-3"],
    )


def test_the_third_compaction_in_a_row_warns_about_the_compaction_budget(caplog):
    checkpointer = Checkpointer(session=StuckSession([12, 12, 12, 12]))

    with caplog.at_level(logging.WARNING, logger="sprout.worker.checkpoint.checkpointer"):
        steps(checkpointer, 4)

    assert caplog.messages == [
        "every recent capture compacted the layer chain: the server's compaction budget "
        "(max_chain_size_mib) is likely smaller than a single step's writes for this workload; "
        "raise it to amortize compaction over more steps"
    ]


def test_the_compaction_budget_warning_is_given_once_per_run(caplog):
    checkpointer = Checkpointer(session=StuckSession([12, 12, 12, 12, 12, 12]))

    with caplog.at_level(logging.WARNING, logger="sprout.worker.checkpoint.checkpointer"):
        steps(checkpointer, 6)

    assert len(caplog.messages) == 1


def test_a_run_that_re_boards_at_every_other_step_warns_about_the_compaction_budget(caplog):
    """Such a run swaps sandboxes every other step, and the checkpointer warns once."""
    checkpointer = Checkpointer(session=Session([13, 13, 13, 13, 13, 13, 13]))

    with caplog.at_level(logging.WARNING, logger="sprout.worker.checkpoint.checkpointer"):
        records = steps(checkpointer, 7)

    assert [record.reboarded for record in records] == [False, True, False, True, False, True, False]
    assert len(caplog.messages) == 1

"""``show``, ``atif`` and ``sweep`` of ``python -m sprout.worker``: the commands that read a run's journal.

A journal is written here as JSON lines. ``sweep`` deletes podman snapshots from a store that the fake
``btrfs`` and ``findmnt`` of ``tests/support`` accept, and microVM snapshots from a stand-in AgentENV.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.cli import main
from tests.support.http import serving
from tests.worker.agentenv import AgentEnv


def journal(path: Path, *events: dict[str, Any]) -> Path:
    """A journal at ``path`` of ``events``, one JSON line each, numbered from 1 as ``seq``."""
    path.write_text("".join(json.dumps({"seq": seq, **event}) + "\n" for seq, event in enumerate(events, 1)))
    return path


def lines(path: Path) -> list[dict[str, Any]]:
    """Each line of the journal at ``path``, parsed. Raises ValueError for a line that is not whole JSON."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_show_counts_the_events_of_each_type_most_frequent_first(tmp_path, capsys):
    path = journal(
        tmp_path / "run.jsonl",
        {"type": "run.started"},
        {"type": "agent.message", "text": "one"},
        {"type": "tool.started", "call_id": "call-1"},
        {"type": "agent.message", "text": "two"},
        {"type": "tool.started", "call_id": "call-2"},
        {"type": "agent.message", "text": "three"},
    )

    code = main(["show", str(path)])

    assert (code, capsys.readouterr().out) == (
        0,
        "events   6\n"
        "  agent.message            3\n"
        "  tool.started             2\n"
        "  run.started              1\n",
    )


def test_atif_with_an_output_file_writes_the_document_that_it_prints_without_one(tmp_path, capsys):
    path = journal(
        tmp_path / "run.jsonl",
        {"type": "run.started", "run_id": "run-1", "slot": "mini-swe-agent", "model": "my-model"},
        {"type": "agent.message", "text": "I will read the tests."},
        {"type": "turn.completed"},
        {"type": "agent.message", "text": "The tests pass."},
        {"type": "turn.completed"},
    )
    main(["atif", str(path)])
    printed = json.loads(capsys.readouterr().out)
    output = tmp_path / "run.atif.json"

    code = main(["atif", str(path), "-o", str(output)])

    assert (code, capsys.readouterr().out) == (0, "wrote %s (2 steps)\n" % output)
    assert json.loads(output.read_text()) == printed


def podman_journal(path: Path, store: Path) -> Path:
    """The journal of a podman run on ``store`` that took snap-1 and snap-2, and recorded snap-1 twice."""
    return journal(
        path,
        {"type": "sandbox.created", "sandbox_id": "sprout-1", "backend": "podman", "store": str(store)},
        {"type": "checkpoint.captured", "step": 1, "snapshot_id": "snap-1"},
        {"type": "checkpoint.captured", "step": 2, "snapshot_id": "snap-2"},
        {"type": "checkpoint.captured", "step": 2, "snapshot_id": "snap-1"},
    )


def snapshots(store: Path, *names: str) -> None:
    """Make the snapshot subvolumes ``names`` in ``store``, as the fake btrfs keeps them: directories."""
    for name in names:
        (store / "snapshots" / name).mkdir(parents=True)


def remaining_snapshots(store: Path) -> list[str]:
    """The names of the snapshots that ``store`` holds, sorted."""
    return sorted(path.name for path in (store / "snapshots").iterdir())


def swept(*, deleted: int, failed: int, seq: int = 5, backend: str = "podman") -> dict[str, Any]:
    """The marker that a sweep appends to a journal whose last ``seq`` was ``seq - 1``."""
    marker = {"v": 2, "type": "snapshots.swept", "seq": seq, "run_id": "sweep", "agent_id": "sweep"}
    return {**marker, "backend": backend, "deleted": deleted, "failed": failed}


def test_a_sweep_without_yes_deletes_nothing_and_says_what_it_would_delete(
    tmp_path, fake_btrfs_store, capsys
):
    path = podman_journal(tmp_path / "run.jsonl", fake_btrfs_store)
    snapshots(fake_btrfs_store, "snap-1", "snap-2")
    before = path.read_bytes()

    code = main(["sweep", str(path)])

    assert (code, capsys.readouterr().out) == (
        0,
        "%s: would delete 2 podman snapshot(s)  (dry run; pass --yes)\n" % path,
    )
    assert (remaining_snapshots(fake_btrfs_store), path.read_bytes()) == (["snap-1", "snap-2"], before)


@pytest.mark.parametrize(
    "present", [["snap-1", "snap-2"], ["snap-1"]], ids=["both present", "one gone already"]
)
def test_a_sweep_deletes_each_snapshot_that_its_journal_records_once_and_marks_the_journal_swept(
    tmp_path, fake_btrfs_store, capsys, present
):
    path = podman_journal(tmp_path / "run.jsonl", fake_btrfs_store)
    snapshots(fake_btrfs_store, *present)

    code = main(["sweep", str(path), "--yes"])

    assert (code, capsys.readouterr().out) == (0, "%s: deleted 2/2 snapshot(s)\n" % path)
    assert (remaining_snapshots(fake_btrfs_store), lines(path)[-1]) == ([], swept(deleted=2, failed=0))


def test_a_journal_that_was_swept_once_is_skipped_by_a_later_sweep(
    tmp_path, fake_btrfs_store, fake_btrfs, capsys
):
    path = podman_journal(tmp_path / "run.jsonl", fake_btrfs_store)
    snapshots(fake_btrfs_store, "snap-1", "snap-2")
    main(["sweep", str(path), "--yes"])
    capsys.readouterr()
    after_first = (path.read_bytes(), len(fake_btrfs.calls()))

    code = main(["sweep", str(path), "--yes"])

    assert (code, capsys.readouterr().out) == (0, "%s: already swept, skipping\n" % path)
    assert (path.read_bytes(), len(fake_btrfs.calls())) == after_first


def test_a_sweep_moves_a_torn_last_line_into_its_marker_so_that_the_journal_stays_whole_lines(
    tmp_path, fake_btrfs_store
):
    """A killed run can leave a partial last record. Each reader of the journal parses whole lines."""
    path = podman_journal(tmp_path / "run.jsonl", fake_btrfs_store)
    complete = lines(path)
    torn = '{"seq": 5, "type": "agent.message", "text": "I will'
    with path.open("a") as stream:
        stream.write(torn)

    main(["sweep", str(path), "--yes"])

    assert lines(path) == [*complete, {**swept(deleted=2, failed=0), "torn_tail": torn}]


def test_a_torn_last_line_that_ends_inside_a_multibyte_character_leaves_the_journal_whole_lines(
    tmp_path, fake_btrfs_store
):
    """Journals hold UTF-8 text, so a write that a kill cut short can end inside a character."""
    path = podman_journal(tmp_path / "run.jsonl", fake_btrfs_store)
    complete = lines(path)
    with path.open("ab") as stream:
        stream.write('{"seq": 5, "type": "agent.message", "text": "\u4e2d'.encode()[:-1])

    main(["sweep", str(path), "--yes"])

    assert lines(path)[:-1] == complete


def test_a_missing_journal_exits_2_and_the_other_journals_are_still_swept(tmp_path, fake_btrfs_store, capsys):
    path = podman_journal(tmp_path / "run.jsonl", fake_btrfs_store)
    missing = tmp_path / "missing.jsonl"

    code = main(["sweep", str(missing), str(path), "--yes"])

    assert (code, capsys.readouterr().err, lines(path)[-1]) == (
        2,
        "no such journal: %s\n" % missing,
        swept(deleted=2, failed=0),
    )


def test_a_snapshot_that_cannot_be_deleted_is_counted_and_reported_and_the_sweep_exits_1(
    tmp_path, fake_btrfs_store, fake_btrfs, capsys
):
    path = podman_journal(tmp_path / "run.jsonl", fake_btrfs_store)
    snapshots(fake_btrfs_store, "snap-1", "snap-2")
    fake_btrfs.fail("delete")

    code = main(["sweep", str(path), "--yes"])

    output = capsys.readouterr()
    assert (code, output.out, lines(path)[-1]) == (
        1,
        "%s: deleted 0/2 snapshot(s), 2 failed\n" % path,
        swept(deleted=0, failed=2),
    )
    assert output.err == (
        "snap-1: btrfs subvolume delete failed (exit 1): ERROR: simulated delete failure\n"
        "snap-2: btrfs subvolume delete failed (exit 1): ERROR: simulated delete failure\n"
    )


def test_the_store_flag_names_the_store_of_a_journal_whose_store_has_moved(tmp_path, fake_btrfs_store):
    path = podman_journal(tmp_path / "run.jsonl", tmp_path / "moved-away")
    snapshots(fake_btrfs_store, "snap-1", "snap-2")

    code = main(["sweep", str(path), "--yes", "--store", str(fake_btrfs_store)])

    assert (code, remaining_snapshots(fake_btrfs_store)) == (0, [])


@pytest.mark.parametrize(
    "created, reason",
    [
        (
            {"type": "sandbox.created", "sandbox_id": "sandbox-1", "backend": "docker"},
            "a docker run; no way to delete its snapshots is known here",
        ),
        (
            {"type": "sandbox.created", "sandbox_id": "sprout-1", "backend": "podman"},
            "a podman run whose journal does not name its store; pass --store <dir> "
            "(the store its subvolumes live in)",
        ),
        (
            {
                "type": "sandbox.created",
                "sandbox_id": "sprout-1",
                "backend": "podman",
                "store": "/path/to/store",
            },
            "a podman run: store does not exist: /path/to/store (create one with "
            "sudo ops/btrfs_store.sh <mount point> <image file> <size>)",
        ),
    ],
    ids=[
        "a backend without snapshot deletion",
        "a podman journal without a store",
        "a store that is not one",
    ],
)
def test_a_journal_whose_snapshots_cannot_be_deleted_is_left_unmarked_with_the_reason(
    tmp_path, fake_btrfs_store, capsys, created, reason
):
    path = journal(tmp_path / "run.jsonl", created, {"type": "checkpoint.captured", "snapshot_id": "snap-1"})
    before = path.read_bytes()

    code = main(["sweep", str(path), "--yes"])

    assert (code, capsys.readouterr().err, path.read_bytes()) == (1, "%s: %s\n" % (path, reason), before)


def test_a_journal_that_names_no_backend_has_its_snapshots_deleted_from_agentenv(
    tmp_path, monkeypatch, capsys
):
    """A run without a sandbox.created backend is a microVM run, from before podman was the default."""
    path = journal(
        tmp_path / "run.jsonl",
        {"type": "checkpoint.captured", "snapshot_id": "snap-1"},
        {"type": "checkpoint.captured", "snapshot_id": "snap-2"},
    )
    agentenv = AgentEnv(delete=404)
    monkeypatch.setenv("AENV_API_KEY", "aenv-key")

    with serving(agentenv) as url:
        monkeypatch.setenv("AENV_SERVER_URL", url)
        code = main(["sweep", str(path), "--yes"])

    assert (code, agentenv.requests, lines(path)[-1]) == (
        0,
        [("DELETE", "/snapshots/snap-1", "aenv-key"), ("DELETE", "/snapshots/snap-2", "aenv-key")],
        swept(deleted=2, failed=0, seq=3, backend="microvm"),
    )

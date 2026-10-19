"""``python -m ops.report snapshots``: each snapshot that the journals name, what selected it, its state."""

from __future__ import annotations

import json

from ops.report import cli
from tests.ops.trees import (
    actor,
    artifact_journal,
    attempt_row,
    fork_origin,
    numbered,
    selected_branch,
    sha256,
    write_attempt,
    write_journal,
    write_manifest,
    write_plan,
    write_summary,
)


def snapshots(capsys, *argv):
    assert cli.main(["snapshots", *map(str, argv), "--format", "json"]) == 0
    return json.loads(capsys.readouterr().out)


def statuses(journal):
    """Each record of a journal's report as (snapshot id, what selected it, its state in the store)."""
    return [(record["snapshot_id"], record["selected"], record["store"]) for record in journal["records"]]


def journals(document, *keys):
    """Each journal of a snapshots report as a tuple of ``keys``."""
    return [tuple(journal[key] for key in keys) for journal in document["journals"]]


def store_with(tmp_path, *snapshot_ids):
    """A podman store whose ``snapshots/`` holds a directory for each of ``snapshot_ids``."""
    store = tmp_path / "store"
    (store / "snapshots").mkdir(parents=True)
    for snapshot_id in snapshot_ids:
        (store / "snapshots" / snapshot_id).mkdir()
    return store


def sprout_output(tmp_path):
    """A sprout evaluation through the Run Store whose plan selected the parent's snapshot after step 2."""
    parent_output = tmp_path / "baseline" / "attempt-1"
    parent = artifact_journal(tmp_path / "artifacts", "job-0", turns=numbered(3), sandbox="par")
    branch = artifact_journal(tmp_path / "artifacts", "job-3", turns=[actor("echo done")], sandbox="brn")
    output = tmp_path / "sprout"
    rows = [
        attempt_row("parent", journal=parent, resolved=False),
        attempt_row("r1b1-a", journal=branch, resolved=True, round_no=1),
    ]
    write_summary(output, attempts=rows, parent_output=str(parent_output))
    write_plan(
        output,
        1,
        selected_branches=[selected_branch("r1b1-a", step=2, snapshot_id="par-0002", commands=("ls",))],
    )
    return output


def test_each_snapshot_is_present_or_missing_in_the_store(tmp_path, capsys):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(3), sandbox="par")
    store = store_with(tmp_path, "par-0001", "par-0002")

    document = snapshots(capsys, journal, "--store", store)

    assert statuses(document["journals"][0]) == [
        ("par-0001", [], "present (not a subvolume)"),
        ("par-0002", [], "present (not a subvolume)"),
        ("par-0003", [], "missing"),
    ]


def test_a_snapshot_that_a_plan_selected_is_marked_with_its_branch(tmp_path, capsys):
    output = sprout_output(tmp_path)

    document = snapshots(capsys, output)

    assert statuses(document["journals"][0]) == [
        ("par-0001", [], "not checked"),
        ("par-0002", ["r1b1-a, parent@2"], "not checked"),
        ("par-0003", [], "not checked"),
    ]


def test_the_journals_of_an_evaluation_through_the_run_store_are_those_its_summary_names(tmp_path, capsys):
    """Every journal is in the worker's artifacts. Only the reused parent is noted as outside the output."""
    output = sprout_output(tmp_path)

    document = snapshots(capsys, output)

    assert journals(document, "run_id", "note") == [
        ("attempt-job-0", "recorded parent, outside this output"),
        ("attempt-job-3", None),
    ]


def test_a_branch_that_no_plan_selected_marks_the_snapshot_it_forked_from(tmp_path, capsys):
    output = tmp_path / "out"
    parent = write_journal(
        output / "trials" / "t-parent" / "agent" / "trajectory.jsonl", turns=numbered(3), sandbox="par"
    )
    origin = fork_origin(parent, step=2, snapshot_id="par-0002", sha=sha256(parent))
    write_journal(
        output / "trials" / "t-b1" / "agent" / "trajectory.jsonl",
        run_id="t-b1",
        turns=[actor("ls -la")],
        sandbox="b1",
        origin=origin,
    )
    write_summary(output, attempts=[])

    document = snapshots(capsys, output)

    assert (statuses(document["journals"][1]), document["journals"][0]["note"]) == (
        [
            ("par-0001", [], "not checked"),
            ("par-0002", ["forked: t-b1"], "not checked"),
            ("par-0003", [], "not checked"),
        ],
        "branch of t-parent@2 from par-0002",
    )


def test_every_snapshot_of_a_swept_journal_is_swept(tmp_path, capsys):
    swept = {"type": "snapshots.swept", "run_id": "sweep", "backend": "podman", "deleted": 2, "failed": 0}
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2), sandbox="par", tail=(swept,))
    store = store_with(tmp_path, "par-0001", "par-0002")

    document = snapshots(capsys, journal, "--store", store)

    assert statuses(document["journals"][0]) == [("par-0001", [], "swept"), ("par-0002", [], "swept")]


def test_a_released_memory_snapshot_is_released(tmp_path, capsys):
    memory = {"type": "snapshot.memory", "step": 1, "turn_id": "t1", "snapshot_id": "mem-1", "captured": True}
    released = {"type": "snapshot.released", "snapshot_id": "mem-1", "kind": "memory", "reason": "resolved"}
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1), sandbox="par", tail=(memory, released))

    document = snapshots(capsys, journal, "--store", store_with(tmp_path, "par-0001"))

    assert statuses(document["journals"][0]) == [
        ("par-0001", [], "present (not a subvolume)"),
        ("mem-1", [], "released"),
    ]


def test_microvm_snapshots_are_not_checked(tmp_path, capsys):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1), sandbox="vm", backend="microvm")

    document = snapshots(capsys, journal, "--store", store_with(tmp_path, "vm-0001"))

    assert (document["journals"][0]["store_note"], statuses(document["journals"][0])) == (
        "not checked (AgentENV is not queried in this version)",
        [("vm-0001", [], "not checked")],
    )


def test_a_store_without_a_snapshots_directory_is_not_checked_and_warned_about(tmp_path, capsys):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1), sandbox="par")
    (tmp_path / "empty").mkdir()

    document = snapshots(capsys, journal, "--store", tmp_path / "empty")

    assert (document["warnings"], document["journals"][0]["store_note"]) == (
        ["store %s has no snapshots/ directory" % (tmp_path / "empty")],
        "not checked: store %s has no snapshots/ directory" % (tmp_path / "empty"),
    )


def test_the_journal_records_the_store_when_no_store_is_given(tmp_path, capsys):
    store = store_with(tmp_path, "par-0001")
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1), sandbox="par", store=str(store))

    document = snapshots(capsys, journal)

    assert (document["store"], document["store_source"], statuses(document["journals"][0])) == (
        str(store),
        "sandbox.created",
        [("par-0001", [], "present (not a subvolume)")],
    )


def test_task_narrows_an_experiment_to_the_journals_of_one_task(tmp_path, capsys):
    root = tmp_path / "exp" / "baseline"
    write_manifest(root, method="baseline", tasks=["alpha", "beta"])
    write_attempt(root / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False)
    write_attempt(root / "tasks" / "beta", 1, artifacts=tmp_path / "artifacts", resolved=True)

    document = snapshots(capsys, root.parent, "--task", "beta")

    assert journals(document, "run_id") == [("attempt-job-beta-1",)]


def test_the_text_report_counts_the_records_of_each_journal_and_of_all_journals(tmp_path, capsys):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(3), sandbox="par")
    store = store_with(tmp_path, "par-0001")

    assert cli.main(["snapshots", str(journal), "--store", str(store)]) == 0

    assert capsys.readouterr().out.splitlines()[-3:] == [
        "   3 records · 3 captured (3 ids) · 0 clean · 0 failed · 3 branch points · 0 selected"
        " · store: 1 present, 2 missing",
        "",
        "totals  1 journals · 3 records · 3 snapshot ids · 1 present · 2 missing · 0 not checked"
        " · 0 selected by plans · 0 swept · 0 released",
    ]

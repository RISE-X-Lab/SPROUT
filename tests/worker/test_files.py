"""``write_json`` and ``read_json``: the receipts that an attempt and its worker leave for each other."""

from __future__ import annotations

from sprout.worker.files import read_json, write_json


def test_a_receipt_replaces_the_one_before_it_in_a_directory_that_the_write_creates(tmp_path):
    path = tmp_path / "job" / "attempt" / "outcome.json"
    write_json(path, {"status": "error", "error": "the first outcome"})

    write_json(path, {"status": "completed"})

    assert (read_json(path), sorted(item.name for item in path.parent.iterdir())) == (
        {"status": "completed"},
        ["outcome.json"],
    )

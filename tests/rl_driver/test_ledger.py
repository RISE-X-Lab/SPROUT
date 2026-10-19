"""The ledger keeps one record for each group id, and a release or a restart does not lose it."""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout.rl_driver.ledger import Conflict, Ledger


def request(prompt: str = "Fix the parser") -> dict:
    return {"rollout_job_id": "group-1", "samples": [{"sample_slot_id": "slot-0", "prompt": prompt}]}


def test_a_repeated_request_returns_the_stored_group_and_keeps_its_document(tmp_path):
    """A resubmission must not reset the document, which names the jobs already submitted."""
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.create("group-1", request(), {"status": "running", "job_id": "job-0"})

    row = ledger.create("group-1", request(), {"status": "queued", "job_id": None})

    assert row["document"] == {"status": "running", "job_id": "job-0"}


def test_a_request_with_its_keys_in_another_order_is_the_same_request(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.create("group-1", {"a": 1, "b": {"c": 2, "d": 3}}, {"status": "queued"})

    row = ledger.create("group-1", {"b": {"d": 3, "c": 2}, "a": 1}, {"status": "queued"})

    assert row["request"] == {"a": 1, "b": {"c": 2, "d": 3}}


def test_a_different_request_under_a_stored_id_is_a_conflict(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.create("group-1", request("Fix the parser"), {"status": "queued"})

    with pytest.raises(Conflict, match="rollout_job_id already names a different request"):
        ledger.create("group-1", request("Fix the lexer"), {"status": "queued"})

    assert ledger.get("group-1")["request"] == request("Fix the parser")


@pytest.mark.parametrize("operation", ["get", "release"])
def test_an_unknown_group_raises_key_error(tmp_path, operation):
    ledger = Ledger(tmp_path / "ledger.sqlite3")

    with pytest.raises(KeyError):
        getattr(ledger, operation)("group-1")


def test_the_active_groups_are_those_not_terminal_oldest_first(tmp_path, clock):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    clock.now = 100.0
    ledger.create("group-b", request(), {"status": "queued"})
    clock.now = 200.0
    ledger.create("group-a", request(), {"status": "queued"})
    clock.now = 300.0
    ledger.create("group-c", request(), {"status": "queued"})
    ledger.save("group-c", {"status": "completed"}, terminal=True)

    active = ledger.active()

    assert active == ["group-b", "group-a"]


def test_saving_a_document_keeps_a_cancellation_requested_meanwhile(tmp_path):
    """A poll that saves the document it read must not undo a release made after the read."""
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.create("group-1", request(), {"status": "queued"})
    ledger.release("group-1")

    ledger.save("group-1", {"status": "running"})

    row = ledger.get("group-1")
    assert (row["document"], row["cancel_requested"]) == ({"status": "running"}, True)


def test_releasing_an_active_group_requests_its_cancellation(tmp_path):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.create("group-1", request(), {"status": "running"})

    ledger.release("group-1")

    row = ledger.get("group-1")
    assert (row["cancel_requested"], row["acknowledged_at"]) == (True, None)


def test_releasing_a_terminal_group_records_the_time_of_the_first_release_only(tmp_path, clock):
    ledger = Ledger(tmp_path / "ledger.sqlite3")
    ledger.create("group-1", request(), {"status": "running"})
    ledger.save("group-1", {"status": "completed"}, terminal=True)
    clock.now = 100.0
    ledger.release("group-1")
    clock.now = 200.0

    ledger.release("group-1")

    row = ledger.get("group-1")
    assert (row["cancel_requested"], row["acknowledged_at"]) == (False, 100.0)


def test_a_new_ledger_on_the_same_file_has_the_groups_of_the_first(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    Ledger(path).create("group-1", request(), {"status": "running"})

    row = Ledger(path).get("group-1")

    assert (row["request"], row["document"]) == (request(), {"status": "running"})


def test_a_ledger_in_a_directory_that_does_not_exist_creates_the_directory(tmp_path):
    path = tmp_path / "state" / "rl-driver" / "ledger.sqlite3"

    Ledger(path).create("group-1", request(), {"status": "queued"})

    assert Ledger(path).active() == ["group-1"]


def test_a_ledger_stays_bound_to_the_run_store_of_its_first_use(tmp_path):
    path = tmp_path / "ledger.sqlite3"
    Ledger(path).bind("http://runstore-a.test")
    Ledger(path).bind("http://runstore-a.test")

    with pytest.raises(Conflict, match="This ledger belongs to a different Run Store URL"):
        Ledger(path).bind("http://runstore-b.test")


def test_a_second_owner_is_refused_at_once_while_the_first_holds_the_ledger(tmp_path):
    path: Path = tmp_path / "ledger.sqlite3"

    with Ledger(path).owner():
        with pytest.raises(Conflict, match="Another driver already owns this ledger"):
            with Ledger(path).owner():
                pytest.fail("a second owner held the ledger")

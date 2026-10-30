"""What the Run Store keeps for each attempt: its journal of events and its frozen payload.

These tests use the PostgreSQL test database (``run_store``), and skip without it.
"""

from __future__ import annotations

from typing import Any

import pytest

from sprout.runstore.payload import PayloadUnavailable
from sprout.runstore.store import Conflict, Store
from tests.runstore.builders import job, payload_for
from tests.support.runstore import registered


def claimed_job(store: Store, kind: str = "rollout", **changes: Any) -> dict[str, Any]:
    """A job of ``kind`` that ``worker-1`` claimed, with ``changes`` over its spec."""
    registered(store, "worker-1")
    store.submit(job(kind, **changes), "key-1")
    claimed = store.claim("worker-1")
    assert claimed is not None
    return claimed


def event(seq: int, **fields: Any) -> dict[str, Any]:
    return {"seq": seq, "type": "tool.started", "call_id": "call-%d" % seq, **fields}


# The journal


def test_the_journal_returns_events_in_seq_order_after_a_cursor_and_up_to_a_limit(run_store):
    claimed = claimed_job(run_store)
    run_store.append_events(claimed["id"], claimed["lease_token"], [event(3), event(1), event(2)])

    whole, page = (
        run_store.events(claimed["active_attempt"]),
        run_store.events(claimed["active_attempt"], after=1, limit=1),
    )

    assert (whole, page) == ([event(1), event(2), event(3)], [event(2)])


def test_appending_the_same_events_again_changes_nothing(run_store):
    """A worker that lost its reply to an append sends the batch again."""
    claimed = claimed_job(run_store)
    run_store.append_events(claimed["id"], claimed["lease_token"], [event(1), event(2)])

    run_store.append_events(claimed["id"], claimed["lease_token"], [event(1), event(2)])

    assert run_store.events(claimed["active_attempt"]) == [event(1), event(2)]


def test_an_event_that_rewrites_a_seq_is_a_conflict_and_the_first_event_stays(run_store):
    claimed = claimed_job(run_store)
    run_store.append_events(claimed["id"], claimed["lease_token"], [event(1)])

    with pytest.raises(Conflict, match="Journal sequence was rewritten"):
        run_store.append_events(claimed["id"], claimed["lease_token"], [event(1, call_id="changed")])

    assert run_store.events(claimed["active_attempt"]) == [event(1)]


def test_an_event_with_a_nul_character_is_kept_with_a_replacement_character_and_appends_again(run_store):
    """PostgreSQL cannot hold U+0000, which a command's binary output can carry: ``cat`` of a LAS file."""
    claimed = claimed_job(run_store)
    binary = event(1, stdout="LASF\x00\x01", **{"key\x00": "value"})
    run_store.append_events(claimed["id"], claimed["lease_token"], [binary])

    run_store.append_events(claimed["id"], claimed["lease_token"], [binary])

    assert run_store.events(claimed["active_attempt"]) == [
        event(1, stdout="LASF\ufffd\x01", **{"key\ufffd": "value"})
    ]


@pytest.mark.parametrize(
    "bad",
    [{"seq": 0}, {"seq": -1}, {"seq": True}, {"seq": "1"}, {"seq": 1.0}, {"type": "tool.started"}],
    ids=["zero", "negative", "boolean", "text", "float", "missing"],
)
def test_a_batch_with_an_event_whose_seq_is_not_a_positive_integer_appends_nothing(run_store, bad):
    claimed = claimed_job(run_store)

    with pytest.raises(ValueError, match="Journal sequence must be positive"):
        run_store.append_events(claimed["id"], claimed["lease_token"], [event(1), bad])

    assert run_store.events(claimed["active_attempt"]) == []


# The frozen payload


def test_a_frozen_payload_reads_back_from_another_store_as_freezing_returned_it(run_store):
    claimed = claimed_job(run_store)

    frozen = run_store.freeze_payload(claimed["id"], claimed["lease_token"], payload_for(claimed))

    assert (frozen, Store(run_store.dsn).payload(claimed["id"], claimed["active_attempt"])) == (
        payload_for(claimed),
        payload_for(claimed),
    )


def test_a_payload_whose_numbers_the_database_rewrites_still_passes_its_check(run_store):
    """The database stores 1e20 as 100000000000000000000. The digest must be of the stored form."""
    claimed = claimed_job(run_store)
    spec = {"prompt": "Write the answer.", "timeout_s": 1e20}

    frozen = run_store.freeze_payload(
        claimed["id"], claimed["lease_token"], payload_for(claimed, effective_spec=spec)
    )

    assert run_store.payload(claimed["id"], claimed["active_attempt"]) == frozen


def test_freezing_the_same_payload_again_returns_it(run_store):
    claimed = claimed_job(run_store)
    first = run_store.freeze_payload(claimed["id"], claimed["lease_token"], payload_for(claimed))

    again = run_store.freeze_payload(claimed["id"], claimed["lease_token"], payload_for(claimed))

    assert again == first


def test_freezing_another_payload_for_the_attempt_is_a_conflict(run_store):
    claimed = claimed_job(run_store)
    run_store.freeze_payload(claimed["id"], claimed["lease_token"], payload_for(claimed))

    with pytest.raises(Conflict, match="Attempt payload is already frozen"):
        run_store.freeze_payload(
            claimed["id"], claimed["lease_token"], payload_for(claimed, recovery={"point_id": "point-1"})
        )


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"kind": "grade"}, "Attempt payload kind differs from job"),
        ({"attempt_id": "another-attempt"}, "Attempt payload identity mismatch"),
        ({"profile_config": {"worker_env": {"MODEL_API_KEY": "sk-live"}}}, r"Use a worker-side \$env"),
    ],
    ids=["another-kind", "another-attempt", "inline-credential"],
)
def test_a_refused_payload_is_not_frozen(run_store, changes, error):
    claimed = claimed_job(run_store)

    with pytest.raises(ValueError, match=error):
        run_store.freeze_payload(claimed["id"], claimed["lease_token"], payload_for(claimed, **changes))

    with pytest.raises(PayloadUnavailable, match="Attempt has no frozen DB payload"):
        run_store.payload(claimed["id"], claimed["active_attempt"])


def test_a_frozen_payload_keeps_an_env_reference_unresolved(run_store, monkeypatch):
    """The value of a credential stays in the worker's environment, out of the database."""
    monkeypatch.setenv("SPROUT_MODEL_API_KEY", "sk-never-stored")
    claimed = claimed_job(run_store)
    referenced = payload_for(
        claimed, profile_config={"worker_env": {"MODEL_API_KEY": {"$env": "SPROUT_MODEL_API_KEY"}}}
    )

    run_store.freeze_payload(claimed["id"], claimed["lease_token"], referenced)

    assert run_store.attempts(claimed["id"])[0]["payload"] == referenced


def test_a_payload_whose_stored_digest_no_longer_matches_is_unavailable(run_store):
    claimed = claimed_job(run_store)
    run_store.freeze_payload(claimed["id"], claimed["lease_token"], payload_for(claimed))
    with run_store.transaction() as cursor:
        cursor.execute(
            "UPDATE rs_attempts SET payload_hash='corrupt' WHERE id=%s", (claimed["active_attempt"],)
        )

    with pytest.raises(PayloadUnavailable, match="Attempt payload checksum mismatch"):
        run_store.payload(claimed["id"], claimed["active_attempt"])


def test_the_payload_of_an_attempt_of_another_job_raises_key_error(run_store):
    claimed = claimed_job(run_store)
    run_store.submit(job("rollout", prompt="Another task."), "key-2")
    registered(run_store, "worker-2")
    other = run_store.claim("worker-2")
    run_store.freeze_payload(other["id"], other["lease_token"], payload_for(other))

    with pytest.raises(KeyError, match=other["active_attempt"]):
        run_store.payload(claimed["id"], other["active_attempt"])

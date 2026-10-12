"""Real PostgreSQL concurrency and fencing, not a SQLite approximation."""

from concurrent.futures import ThreadPoolExecutor
import os
import time
from uuid import uuid4

import pytest

from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Conflict, Fenced, Store


@pytest.fixture
def store():
    dsn = os.environ.get("SPROUT_RUNSTORE_TEST_DSN")
    if not dsn:
        pytest.skip("Set SPROUT_RUNSTORE_TEST_DSN to an isolated PostgreSQL database")
    pytest.importorskip("psycopg2")
    schema = "fixture_" + uuid4().hex
    admin = Store(dsn)
    with admin.transaction() as cursor:
        cursor.execute(f"CREATE SCHEMA {schema}")
    instance = Store(dsn + f" options='-c search_path={schema}'")
    instance.initialize()
    try:
        yield instance
    finally:
        with admin.transaction() as cursor:
            cursor.execute(f"DROP SCHEMA {schema} CASCADE")


def request(**changes):
    return JobSpec("rollout", {"prompt": "fixture", **changes})


def test_idempotent_submit_and_restart(store):
    job = store.submit(request(), "fixture")
    assert Store(store.dsn).submit(request(), "fixture")["id"] == job["id"]
    with pytest.raises(Conflict):
        store.submit(request(prompt="different"), "fixture")


def test_concurrent_claimers_claim_each_job_once(store):
    jobs = [store.submit(request(), str(index))["id"] for index in range(24)]
    with ThreadPoolExecutor(max_workers=24) as pool:
        claims = list(pool.map(lambda number: store.claim(f"worker-{number}"), range(32)))
    assert sorted(claim["id"] for claim in claims if claim) == sorted(jobs)
    assert store.claim("late-worker") is None


def test_expiry_quarantines_instead_of_replaying_uncertain_execution(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker", lease_s=0.02)
    time.sleep(0.04)
    with pytest.raises(Fenced):
        store.heartbeat(job["id"], claim["lease_token"])
    assert store.expire() == [job["id"]]
    assert store.get(job["id"])["state"] == "quarantined"
    assert store.claim("other-worker") is None
    with pytest.raises(Fenced):
        store.finish(job["id"], claim["lease_token"], {"ok": True})


def test_result_is_durable_and_old_attempt_cannot_overwrite(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    store.finish(job["id"], claim["lease_token"], {"final_text": "fixture result"})
    assert Store(store.dsn).get(job["id"])["result"]["final_text"] == "fixture result"
    with pytest.raises(Fenced):
        store.finish(job["id"], claim["lease_token"], {"final_text": "stale"})


def test_journal_append_is_idempotent_but_not_rewritable(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    event = {"seq": 1, "type": "tool.started", "call_id": "first"}
    store.append_events(job["id"], claim["lease_token"], [event, event])
    assert store.events(claim["active_attempt"]) == [event]
    with pytest.raises(Conflict):
        store.append_events(job["id"], claim["lease_token"], [{**event, "call_id": "changed"}])


def test_actor_retry_needs_verified_pair_and_preserves_cap(store):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    result = {"failure_kind": "infrastructure", "error": "transport"}
    with pytest.raises(ValueError, match="verified continuation"):
        store.finish(job["id"], claim["lease_token"], result, state="failed", retry=True)
    for number in range(3):
        store.finish(job["id"], claim["lease_token"], result, state="failed", retry=True,
                     recovery={"point_id": "paired", "timeout_s": 10})
        if number < 2:
            with store.transaction() as cursor:
                cursor.execute("UPDATE rs_jobs SET ready_at=clock_timestamp() WHERE id=%s", (job["id"],))
            claim = store.claim("worker")
    assert store.get(job["id"])["state"] == "failed"
    assert len(store.attempts(job["id"])) == 3


def test_a_stop_cancels_a_queued_job_and_reaches_a_running_one_at_its_heartbeat(store):
    queued = store.submit(request(prompt="queued"), "stop-queued")
    assert store.request_stop(queued["id"])["state"] == "cancelled"
    running = store.submit(request(prompt="running"), "stop-running")
    claim = store.claim("worker")
    assert claim["id"] == running["id"] and store.heartbeat(running["id"], claim["lease_token"]) is False
    assert store.request_stop(running["id"])["state"] == "running", "only the worker can say it stopped"
    first = store.get(running["id"])["stop_requested_at"]
    store.request_stop(running["id"])
    assert store.get(running["id"])["stop_requested_at"] == first, "asking twice is asking once"
    assert store.heartbeat(running["id"], claim["lease_token"]) is True
    # Even an infrastructure failure of a stopped job is final: nobody wants it run again.
    store.finish(running["id"], claim["lease_token"], {"failure_kind": "infrastructure", "error": "x"},
                 state="failed", retry=True, recovery={"point_id": "paired", "timeout_s": 10})
    assert store.get(running["id"])["state"] == "failed"
    assert store.request_stop(running["id"])["state"] == "failed", "a finished job is left as it is"


@pytest.mark.parametrize("updates", [
    {"session": "not-serializable-session"}, {"journal_path": "/tmp/old.jsonl"},
    {"slot": "codex-cli"}, {"cwd": "/some/host/repo"},
    {"backend": {"api_key": "should-not-be-persisted"}}, {"timeout_s": float("nan")},
])
def test_unsupported_or_unsafe_runs_rejected(updates):
    with pytest.raises(ValueError):
        request(**updates).validate()

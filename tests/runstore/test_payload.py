import json
import os
import threading
import time

import pytest

from sprout.runstore.payload import (PayloadUnavailable, encode_payload, receive_payload,
                              send_payload)
from sprout.runstore.specs import digest
from sprout.runstore.store import Conflict, Fenced, Store
from tests.runstore.test_store import request, store


def envelope(claim: dict) -> dict:
    return {"version": 1, "job_id": claim["id"], "attempt_id": claim["active_attempt"],
            "kind": claim["kind"], "effective_spec": claim["request"]["spec"],
            "profile_config": {}, "recovery": None}


def test_payload_is_frozen_durable_and_separate_from_heartbeat_metadata(store):
    job = store.submit(request(timeout_s=1e20), "payload")
    claim = store.claim("worker")
    payload = envelope(claim)
    frozen = store.freeze_payload(job["id"], claim["lease_token"], payload)
    assert Store(store.dsn).payload(job["id"], claim["active_attempt"]) == frozen
    assert store.freeze_payload(job["id"], claim["lease_token"], payload) == frozen
    with pytest.raises(Conflict, match="already frozen"):
        store.freeze_payload(job["id"], claim["lease_token"], {**payload, "recovery": {"snapshot_id": "changed"}})
    store.heartbeat(job["id"], claim["lease_token"], execution={"payload": {"not": "authoritative"}})
    assert store.payload(job["id"], claim["active_attempt"]) == frozen
    store.finish(job["id"], claim["lease_token"], {"status": "completed"})
    with pytest.raises(Fenced):
        store.freeze_payload(job["id"], claim["lease_token"], payload)


@pytest.mark.parametrize("updates", [
    {"attempt_id": "another-attempt"}, {"job_id": "another-job"}, {"kind": "grade"},
    {"profile_config": {"worker_env": {"UPSTREAM_API_KEY": "not-for-db"}}},
])
def test_payload_rejects_identity_and_inline_credentials(store, updates):
    store.submit(request(), "payload")
    claim = store.claim("worker")
    with pytest.raises(ValueError):
        store.freeze_payload(claim["id"], claim["lease_token"], {**envelope(claim), **updates})
    assert store.attempts(claim["id"])[0]["payload"] is None


def test_payload_keeps_env_references_unresolved(store, monkeypatch):
    monkeypatch.setenv("FIXTURE_API_KEY", "never-persist-this")
    store.submit(request(), "payload")
    claim = store.claim("worker")
    payload = envelope(claim)
    payload["profile_config"] = {"worker_env": {"UPSTREAM_API_KEY": {"$env": "FIXTURE_API_KEY"}}}
    assert store.freeze_payload(claim["id"], claim["lease_token"], payload) == payload
    assert "never-persist-this" not in json.dumps(store.attempts(claim["id"])[0]["payload"])


def test_existing_schema_upgrade_preserves_legacy_records(store):
    store.submit(request(), "legacy")
    claim = store.claim("old-worker")
    store.heartbeat(claim["id"], claim["lease_token"], execution={"directory": "retained"})
    store.finish(claim["id"], claim["lease_token"], {"status": "completed", "text": "retained"})
    with store.transaction() as cursor:
        cursor.execute("ALTER TABLE rs_attempts DROP COLUMN payload, DROP COLUMN payload_hash")
    store.initialize()
    store.initialize()
    attempt = store.attempts(claim["id"])[0]
    assert attempt["execution"]["directory"] == "retained"
    assert attempt["result"]["text"] == "retained"
    with pytest.raises(PayloadUnavailable, match="no frozen DB payload"):
        store.payload(claim["id"], claim["active_attempt"])


def test_payload_checksum_corruption_is_detected(store):
    store.submit(request(), "payload")
    claim = store.claim("worker")
    store.freeze_payload(claim["id"], claim["lease_token"], envelope(claim))
    with store.transaction() as cursor:
        cursor.execute("UPDATE rs_attempts SET payload_hash='corrupt' WHERE id=%s", (claim["active_attempt"],))
    with pytest.raises(PayloadUnavailable, match="checksum"):
        store.payload(claim["id"], claim["active_attempt"])


def test_large_unicode_payload_roundtrips_over_pipe():
    payload = {"version": 1, "job_id": "job", "attempt_id": "attempt", "kind": "rollout",
               "effective_spec": {"prompt": "测试" * 50000}, "profile_config": {}, "recovery": None}
    data = encode_payload(payload, "job", "attempt")
    reader, writer = os.pipe()
    errors = []

    def send():
        try:
            send_payload(os.fdopen(writer, "wb", buffering=0), data, timeout_s=5)
        except Exception as error:
            errors.append(error)

    thread = threading.Thread(target=send)
    thread.start()
    try:
        assert receive_payload(reader, digest(payload), "job", "attempt", timeout_s=5) == payload
    finally:
        os.close(reader)
        thread.join(timeout=6)
    assert not thread.is_alive() and not errors


@pytest.mark.parametrize("data", [b"", b'{"version":', b"[]", b"null", b"{}", b"\xff"])
def test_invalid_or_truncated_payload_fails_closed(data):
    reader, writer = os.pipe()
    os.write(writer, data)
    os.close(writer)
    try:
        with pytest.raises(ValueError):
            receive_payload(reader, "expected", "job", "attempt", timeout_s=1)
    finally:
        os.close(reader)


def test_payload_handoff_has_bounded_wait_when_child_does_not_read():
    reader, writer = os.pipe()
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError):
            send_payload(os.fdopen(writer, "wb", buffering=0), b"x" * 1024 * 1024, timeout_s=0.05)
        assert time.monotonic() - started < 2
    finally:
        os.close(reader)


def test_child_wait_for_payload_is_bounded():
    reader, writer = os.pipe()
    try:
        with pytest.raises(TimeoutError):
            receive_payload(reader, "expected", "job", "attempt", timeout_s=0.05)
    finally:
        os.close(reader)
        os.close(writer)


def test_payload_size_limit(monkeypatch):
    monkeypatch.setattr("sprout.runstore.payload.MAX_PAYLOAD_BYTES", 8)
    reader, writer = os.pipe()
    os.write(writer, b"x" * 9)
    os.close(writer)
    try:
        with pytest.raises(ValueError, match="exceeds"):
            receive_payload(reader, "expected", "job", "attempt")
    finally:
        os.close(reader)

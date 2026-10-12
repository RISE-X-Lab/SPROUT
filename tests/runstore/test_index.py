import hashlib

from sprout.runstore.index import Index
from sprout.harness.agents.mini_swe.history import NativePoint
from tests.runstore.test_store import request, store


def test_later_boundary_invalidation_revokes_published_recovery(store, tmp_path):
    job = store.submit(request(), "fixture")
    claim = store.claim("worker")
    prefix = tmp_path / "native.jsonl"
    prefix.write_bytes(b'{}\n')
    native = {"path": str(prefix), "byte_length": 3,
              "sha256": hashlib.sha256(prefix.read_bytes()).hexdigest()}
    events = [{"type": "tool.started", "call_id": "call", "name": "shell", "args": {}},
              {"type": "tool.finished", "call_id": "call", "output": "response"}]
    index = Index(store, lambda point: True)
    index.project(job["id"], claim["lease_token"], {}, events, [NativePoint(1, 1, "snapshot", native)])
    assert index.points(claim["active_attempt"])[0]["valid"]
    index.project(job["id"], claim["lease_token"], {}, events, [])
    assert not index.points(claim["active_attempt"])[0]["valid"]

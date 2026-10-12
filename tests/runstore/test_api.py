import pytest

from sprout.runstore.api import create_app
from tests.runstore.test_store import store


def test_api_auth_validation_status_and_durable_result(store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    app = create_app(store, "fixture-token")
    with TestClient(app) as client:
        assert client.get("/v1/jobs").status_code == 401
        client.headers["Authorization"] = "Bearer fixture-token"
        assert client.post("/v1/jobs", json={"kind": "rollout", "spec": {"prompt": "fixture"}}).status_code == 422
        created = client.post("/v1/jobs", headers={"Idempotency-Key": "fixture"},
                              json={"kind": "rollout", "spec": {"prompt": "fixture"}})
        assert created.status_code == 202, created.text
        job_id = created.json()["id"]
        assert client.get(f"/v1/jobs/{job_id}").json()["state"] == "queued"
        claim = store.claim("fixture-worker")
        store.finish(job_id, claim["lease_token"], {"final_text": "durable"})
        for repeat in range(2):
            result = client.get(f"/v1/jobs/{job_id}/result").json()
            assert result["ready"] and result["result"]["final_text"] == "durable"
        assert len(client.get(f"/v1/jobs/{job_id}/attempts").json()) == 1


def test_http_submission_freezes_profile_defaults(store):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient
    from sprout.runstore.specs import digest

    profile = {"python": "/usr/bin/python3.11", "run_defaults": {"slot": "mini-swe-agent"}}
    with TestClient(create_app(store, "fixture", profiles={"default": profile})) as client:
        response = client.post("/v1/jobs", headers={"Authorization": "Bearer fixture", "Idempotency-Key": "profile"},
                               json={"kind": "rollout", "spec": {"prompt": "fixture"}})
        assert response.status_code == 202, response.text
        request = response.json()["request"]
        assert request["spec"]["slot"] == "mini-swe-agent"
        assert request["profile_hash"] == digest(profile)

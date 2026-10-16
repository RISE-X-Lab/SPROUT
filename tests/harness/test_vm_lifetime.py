"""The common rollout entry must propagate its budget before creating the VM."""

from copy import deepcopy
from types import SimpleNamespace

import pytest

from sprout.harness_sandbox.backends import BackendError
from sprout.harness_orchestrator.run import Orchestrator, RunSpec


@pytest.fixture
def recorded_session(monkeypatch):
    observed = {}

    def create(image, resources=None):
        observed["image"] = image
        observed["resources"] = resources
        return True

    session = SimpleNamespace(create=create, sandbox_id="fixture-vm", supports_snapshot=lambda: False)

    def make_session(**kwargs):
        observed["constructor"] = kwargs
        return session

    monkeypatch.setattr("sprout.harness_sandbox.session.SandboxSession", make_session)
    return observed


@pytest.mark.parametrize("budget,explicit,expected", [
    (1800, None, 2400), (1800, 600, 2400), (1800, 7200, 7200),
    (3600, None, 4200), (10800, 600, 11400),
    (1.2, None, 602), (125.5, None, 726),
])
def test_owned_rollout_passes_longer_lifetime_without_changing_budget(
        recorded_session, budget, explicit, expected):
    backend = {"backend": "microvm", "microvm": {"server_url": "http://fixture", "allow_internet": False}}
    if explicit is not None:
        backend["microvm"]["sandbox_ttl"] = explicit
    before = deepcopy(backend)
    spec = RunSpec(prompt="fixture", sandbox_image="ready-template", backend=backend,
                   timeout_s=budget, sandbox_resources={"cpu": 2, "memory_mb": 12288})
    Orchestrator()._wire_sandbox(spec, None)
    effective = recorded_session["constructor"]["backend"]
    assert effective["microvm"]["sandbox_ttl"] == expected
    assert effective["microvm"]["allow_internet"] is False
    assert effective["microvm"]["server_url"] == "http://fixture"
    assert spec.timeout_s == budget and spec.backend == before
    assert recorded_session["image"] == "ready-template"
    assert recorded_session["resources"] == spec.sandbox_resources


def test_an_empty_backend_is_podman_which_takes_no_lease(recorded_session):
    """A bare RunSpec names no backend; that is podman, whose containers live
    until they are removed, so no lease is added -- only a named microVM gets one."""
    spec = RunSpec(prompt="fixture", sandbox_image="image", backend={}, timeout_s=1800)
    Orchestrator()._wire_sandbox(spec, None)
    assert recorded_session["constructor"]["backend"] == {}
    assert spec.backend == {}
    spec = RunSpec(prompt="fixture", sandbox_image="image", backend={"backend": "microvm"}, timeout_s=1800)
    Orchestrator()._wire_sandbox(spec, None)
    assert recorded_session["constructor"]["backend"] == {"backend": "microvm", "microvm": {"sandbox_ttl": 2400}}


@pytest.mark.parametrize("budget", [None, 0, -1, True, float("inf"), float("nan")])
def test_invalid_vm_budget_is_rejected_before_allocation(recorded_session, budget):
    spec = RunSpec(prompt="fixture", sandbox_image="image", timeout_s=budget, backend={"backend": "microvm"})
    with pytest.raises(BackendError, match="operation budget"):
        Orchestrator()._wire_sandbox(spec, None)
    assert recorded_session == {}


def test_longer_vm_lifetime_reaches_agentenv_request(recorded_session):
    import asyncio
    import json

    import httpx

    from sprout.harness_sandbox.backends import build_pool

    spec = RunSpec(prompt="fixture", sandbox_image="ready-template", timeout_s=1800,
                   backend={"backend": "microvm", "microvm": {"server_url": "http://fixture"}})
    Orchestrator()._wire_sandbox(spec, None)
    requests = []

    def respond(request):
        if request.method == "POST" and request.url.path == "/sandboxes":
            requests.append(json.loads(request.content))
            return httpx.Response(200, json={"sandboxID": "fixture-vm", "templateID": "ready-template"})
        assert request.method == "DELETE"
        return httpx.Response(204)

    async def create():
        pool = build_pool(recorded_session["constructor"]["backend"])
        await pool._client.aclose()
        pool._client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
        try:
            await pool.spawn("ready-template")
        finally:
            await pool.close()

    asyncio.run(create())
    assert len(requests) == 1
    assert requests[0]["timeout"] == 2400
    assert requests[0]["autoPause"] is True
    assert requests[0]["templateID"] == "ready-template"

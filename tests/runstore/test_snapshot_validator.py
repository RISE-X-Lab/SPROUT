"""Whether the snapshot of a recovery point still exists, on the backend of its job's profile.

These tests use the PostgreSQL test database (``run_store``), and skip without it. A podman snapshot is a
directory in a store under ``tmp_path``. A microVM snapshot is a record of a stub AgentENV on a local port,
and, with a ``snapshot_repository``, a directory in it.
"""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from sprout.runstore.config import snapshot_validator
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from tests.runstore.builders import rollout_spec
from tests.support.http import serving
from tests.support.runstore import registered


def podman_store(directory: Path, *snapshots: str) -> Path:
    """A podman store in ``directory`` that holds ``snapshots``."""
    for snapshot_id in snapshots:
        (directory / "snapshots" / snapshot_id).mkdir(parents=True)
    return directory


def podman_backend(store: Path) -> dict[str, Any]:
    return {"backend": "podman", "podman": {"store": str(store), "runtime_bin": "/path/to/sprout-runtime"}}


def job_of_profile(
    store: Store, profile: dict[str, Any], spec: dict[str, Any] | None = None
) -> tuple[str, dict]:
    """The id of a rollout of the profile ``mini``, and a config in which ``mini`` is ``profile``."""
    submitted = store.submit(JobSpec("rollout", spec or rollout_spec(), profile="mini"), "key-1")
    return submitted["id"], {"artifact_root": "/path/to/artifacts", "profiles": {"mini": profile}}


# Podman


@pytest.mark.parametrize(
    "snapshot_id,valid", [("sprout-1-1", True), ("sprout-1-2", False)], ids=["present", "gone"]
)
def test_a_podman_point_is_valid_while_the_store_of_its_profile_holds_its_snapshot(
    run_store, tmp_path, snapshot_id, valid
):
    store = podman_store(tmp_path / "store", "sprout-1-1")
    job_id, config = job_of_profile(run_store, {"run_defaults": {"backend": podman_backend(store)}})

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": snapshot_id}) is valid


def test_the_backend_of_the_request_wins_over_the_backend_of_its_profile(run_store, tmp_path):
    profile_store = podman_store(tmp_path / "profile-store")
    request_store = podman_store(tmp_path / "request-store", "sprout-1-1")
    job_id, config = job_of_profile(
        run_store,
        {"run_defaults": {"backend": podman_backend(profile_store)}},
        rollout_spec(backend={"podman": {"store": str(request_store)}}),
    )

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "sprout-1-1"}) is True


@pytest.mark.parametrize("backend", [{}, {"backend": "podman"}], ids=["no-backend", "podman-without-a-store"])
def test_a_podman_point_without_a_named_store_is_looked_for_in_the_default_store(
    run_store, tmp_path, monkeypatch, backend
):
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(podman_store(tmp_path / "default-store", "sprout-1-1")))
    job_id, config = job_of_profile(run_store, {"run_defaults": {"backend": backend}})

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "sprout-1-1"}) is True


def test_a_point_without_its_job_id_is_checked_on_the_job_of_its_attempt(run_store, tmp_path):
    """``Index.points`` gives the rows of the points of an attempt, which name the attempt and not the job."""
    store = podman_store(tmp_path / "store", "sprout-1-1")
    _, config = job_of_profile(run_store, {"run_defaults": {"backend": podman_backend(store)}})
    registered(run_store, "worker-1")
    claimed = run_store.claim("worker-1")

    valid = snapshot_validator(run_store, config)(
        {"attempt_id": claimed["active_attempt"], "snapshot_id": "sprout-1-1"}
    )

    assert valid is True


def test_a_point_of_an_unknown_attempt_raises_key_error(run_store, tmp_path):
    _, config = job_of_profile(run_store, {"run_defaults": {"backend": podman_backend(tmp_path)}})

    with pytest.raises(KeyError, match="no-such-attempt"):
        snapshot_validator(run_store, config)({"attempt_id": "no-such-attempt", "snapshot_id": "sprout-1-1"})


# MicroVMs


@dataclass
class AgentEnv:
    """A stub AgentENV: ``answers`` maps a snapshot id to the status and the body that it answers."""

    url: str
    answers: dict[str, tuple[int, object]] = field(default_factory=dict)
    #: The ``X-API-Key`` of each request, in arrival order.
    api_keys: list[str | None] = field(default_factory=list)


@pytest.fixture
def agentenv() -> Iterator[AgentEnv]:
    pytest.importorskip("starlette", reason="install the runstore extra: pip install -e '.[runstore]'")
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse, PlainTextResponse
    from starlette.routing import Route

    stub = AgentEnv(url="")

    async def snapshot(request: Any) -> Any:
        stub.api_keys.append(request.headers.get("x-api-key"))
        status, body = stub.answers.get(request.path_params["snapshot_id"], (404, {"error": "not found"}))
        return JSONResponse(body, status) if isinstance(body, dict) else PlainTextResponse(str(body), status)

    with serving(Starlette(routes=[Route("/snapshots/{snapshot_id}", snapshot)])) as url:
        stub.url = url
        yield stub


def microvm_profile(server_url: str | None, *, backend: str = "microvm", **profile: Any) -> dict[str, Any]:
    """A profile whose ``backend`` names AgentENV at ``server_url``, with the key in ``$AENV_API_KEY``."""
    microvm = {"api_key": {"$env": "AENV_API_KEY"}, "server_url": server_url}
    return {"run_defaults": {"backend": {"backend": backend, "microvm": microvm}}, **profile}


@pytest.mark.parametrize(
    "answer,valid",
    [
        ((200, {"snapshotID": "snap-1"}), True),
        ((200, {"snapshotID": "snap-2"}), False),
        ((404, {"snapshotID": "snap-1"}), False),
        ((200, "snapshot snap-1"), False),
    ],
    ids=["known", "another-record", "not-found", "not-json"],
)
def test_a_microvm_point_is_valid_while_agentenv_answers_with_its_record(
    run_store, agentenv, monkeypatch, answer, valid
):
    monkeypatch.setenv("AENV_API_KEY", "aenv-key")
    agentenv.answers["snap-1"] = answer
    job_id, config = job_of_profile(run_store, microvm_profile(agentenv.url))

    checked = snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "snap-1"})

    assert (checked, agentenv.api_keys) == (valid, ["aenv-key"])


def test_a_microvm_point_is_not_valid_while_agentenv_does_not_answer(run_store, monkeypatch):
    monkeypatch.setenv("AENV_API_KEY", "aenv-key")
    job_id, config = job_of_profile(run_store, microvm_profile("http://127.0.0.1:9"))

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "snap-1"}) is False


def test_a_microvm_point_without_an_agentenv_url_is_never_valid(run_store, monkeypatch):
    monkeypatch.setenv("AENV_API_KEY", "aenv-key")
    job_id, config = job_of_profile(run_store, microvm_profile(None))

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "snap-1"}) is False


def test_only_a_microvm_backend_asks_agentenv_for_a_snapshot(run_store, agentenv, monkeypatch):
    """Another backend keeps no snapshot that a branch can restore, even with a microvm section."""
    monkeypatch.setenv("AENV_API_KEY", "aenv-key")
    agentenv.answers["snap-1"] = (200, {"snapshotID": "snap-1"})
    job_id, config = job_of_profile(run_store, microvm_profile(agentenv.url, backend="docker"))

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "snap-1"}) is False


def repository_snapshot(
    repository: Path, snapshot_id: str, manifest: dict[str, Any], files: dict[str, bytes]
) -> None:
    """A snapshot in the on-disk ``repository`` of AgentENV: its manifest, and ``files`` beside it."""
    directory = repository / "snapshots" / snapshot_id
    directory.mkdir(parents=True)
    (directory / "firecracker-manifest.json").write_text(json.dumps(manifest))
    for name, data in files.items():
        (directory / name).write_bytes(data)


COMPLETE = {"version": 1, "rootfs": {"virtualSize": 1024}}


@pytest.mark.parametrize(
    "manifest,files,valid",
    [
        (COMPLETE, {"commit": b"c"}, True),
        ({**COMPLETE, "vmState": {}}, {"commit": b"c", "vm_state.bin": b"m"}, True),
        ({**COMPLETE, "version": 2}, {"commit": b"c"}, False),
        ({"version": 1, "rootfs": {}}, {"commit": b"c"}, False),
        (COMPLETE, {"commit": b""}, False),
        (COMPLETE, {}, False),
        ({**COMPLETE, "vmState": {}}, {"commit": b"c", "vm_state.bin": b""}, False),
        ({**COMPLETE, "vmState": {}}, {"commit": b"c"}, False),
    ],
    ids=[
        "complete",
        "complete-with-memory",
        "another-version",
        "no-rootfs-size",
        "empty-commit",
        "no-commit",
        "empty-memory",
        "no-memory",
    ],
)
def test_a_microvm_point_needs_a_complete_snapshot_in_the_repository_of_its_profile(
    run_store, agentenv, monkeypatch, tmp_path, manifest, files, valid
):
    monkeypatch.setenv("AENV_API_KEY", "aenv-key")
    agentenv.answers["snap-1"] = (200, {"snapshotID": "snap-1"})
    repository_snapshot(tmp_path / "repository", "snap-1", manifest, files)
    profile = microvm_profile(agentenv.url, snapshot_repository=str(tmp_path / "repository"))
    job_id, config = job_of_profile(run_store, profile)

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "snap-1"}) is valid


def test_a_microvm_point_without_a_manifest_in_the_repository_is_not_valid(
    run_store, agentenv, monkeypatch, tmp_path
):
    monkeypatch.setenv("AENV_API_KEY", "aenv-key")
    agentenv.answers["snap-1"] = (200, {"snapshotID": "snap-1"})
    (tmp_path / "repository" / "snapshots").mkdir(parents=True)
    job_id, config = job_of_profile(
        run_store, microvm_profile(agentenv.url, snapshot_repository=str(tmp_path / "repository"))
    )

    assert snapshot_validator(run_store, config)({"job_id": job_id, "snapshot_id": "snap-1"}) is False

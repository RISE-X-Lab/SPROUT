"""The Run Store on podman, its default backend: which jobs a worker takes, how
it establishes that an attempt's sandboxes are gone, and when a recovery
point's snapshot still exists. The podman and btrfs CLIs are the fakes from
tests/conftest.py; no container runs."""

import json
from pathlib import Path

import pytest

from sprout.runstore.config import podman_snapshot_present, snapshot_validator
from sprout.worker.attempts import OWNED_BACKENDS, Worker, owned_backend
from tests.conftest import fake_networks, recorded_calls


def podman_backend(store: Path) -> dict:
    return {"backend": "podman", "podman": {"store": str(store), "runtime_bin": "/opt/sprout-runtime"}}


def leaked(store: Path, name: str) -> dict:
    """A sandbox the child never destroyed: container, network, rootfs and socket directory."""
    (store / "sandboxes" / name).mkdir(parents=True)
    (store / "sandboxes" / (name + ".json")).write_text("{}")
    sockets = store / "sockets" / name
    sockets.mkdir(parents=True)
    return {"name": name, "store": str(store), "sockets": str(sockets), "created": 1700000000}


def attempt(tmp_path: Path, *rows: dict) -> Path:
    directory = tmp_path / "attempt"
    directory.mkdir()
    (directory / "resources.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    return directory


def claim(name: str, **fields) -> dict:
    return {"event": "claim", "run_id": "run", "kind": "sandbox", "id": name, **fields}


def worker(tmp_path: Path) -> Worker:
    # Cleanup and the backend check never reach the database.
    return Worker(object(), {"artifact_root": str(tmp_path / "artifacts"), "profiles": {}})


@pytest.mark.parametrize("backend, expected", [
    ({}, "podman"),
    ({"podman": {"store": "/s"}}, "podman"),
    ({"backend": "podman"}, "podman"),
    ({"backend": "microvm"}, "microvm"),
    ({"backend": "MicroVM"}, "microvm"),
])
def test_queued_jobs_run_on_podman_unless_the_profile_names_microvm(backend, expected):
    assert OWNED_BACKENDS == ("podman", "microvm")
    assert owned_backend("rollout", {"sandbox_image": "image", "backend": backend}) == expected
    assert owned_backend("grade", {"backend": backend}) == expected


def test_a_backend_the_worker_cannot_clean_up_after_is_refused():
    with pytest.raises(ValueError, match="not 'docker'"):
        owned_backend("rollout", {"sandbox_image": "image", "backend": {"backend": "docker"}})
    with pytest.raises(ValueError, match="sandbox_image"):
        owned_backend("rollout", {"backend": {"backend": "podman"}})


def test_a_sandbox_the_child_destroyed_is_only_looked_at(tmp_path, store, fake_podman):
    directory = attempt(tmp_path, claim("sprout-000000000001", backend="podman", store=str(store)),
                        {"event": "release", "run_id": "run", "kind": "sandbox", "id": "sprout-000000000001"})
    assert worker(tmp_path)._cleanup(directory, {"backend": podman_backend(store)}) is True
    assert recorded_calls(tmp_path, "podman") == [
        ["container", "exists", "sprout-000000000001"], ["network", "exists", "sprout-000000000001"]]


def test_a_leaked_sandbox_is_removed_whole_before_the_attempt_counts_as_clean(
        tmp_path, store, fake_podman, monkeypatch):
    """A child the watchdog killed leaves its container running: the worker
    removes it the way `harness reap` does, then checks that nothing is left."""
    name = "sprout-000000000002"
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", json.dumps([leaked(store, name)]))
    monkeypatch.setenv("FAKE_PODMAN_NETWORKS", json.dumps([
        {"name": name, "store": str(store), "created": "2023-11-14T22:13:20.123456789Z"}]))
    directory = attempt(tmp_path, claim(name, backend="podman", store=str(store)))
    assert worker(tmp_path)._cleanup(directory, {"backend": podman_backend(store)}) is True
    podman = recorded_calls(tmp_path, "podman")
    assert ["rm", "-f", name] in podman and ["network", "rm", name] in podman
    assert podman[-2:] == [["container", "exists", name], ["network", "exists", name]], "looked at again after removal"
    assert not (store / "sandboxes" / name).exists() and not (store / "sockets" / name).exists()
    assert fake_networks(tmp_path) == {}


def test_a_grader_claim_without_a_store_is_looked_for_in_the_jobs_store(tmp_path, store, fake_podman, monkeypatch):
    name = "sprout-000000000003"
    (store / "sandboxes" / name).mkdir(parents=True)  # a rootfs whose container is gone
    directory = attempt(tmp_path, {"event": "allocation_started", "run_id": "grader", "image": "task"},
                        claim(name, run_id="grader"))
    assert worker(tmp_path)._cleanup(directory, {"backend": podman_backend(store)}) is True
    assert not (store / "sandboxes" / name).exists()


def test_a_sandbox_that_cannot_be_removed_quarantines_the_attempt(tmp_path, store, fake_podman, monkeypatch):
    name = "sprout-000000000004"
    monkeypatch.setenv("FAKE_PODMAN_CONTAINERS", json.dumps([leaked(store, name)]))
    monkeypatch.setenv("FAKE_PODMAN_FAIL", "rm")
    directory = attempt(tmp_path, claim(name, backend="podman", store=str(store)))
    assert worker(tmp_path)._cleanup(directory, {"backend": podman_backend(store)}) is False
    assert (store / "sandboxes" / name).exists(), "nothing claimed to be reclaimed that was not"


def test_a_podman_that_cannot_answer_is_not_taken_for_nothing_left(tmp_path, store):
    backend = {"backend": "podman", "podman": {"store": str(store), "podman_bin": str(tmp_path / "no-podman")}}
    directory = attempt(tmp_path, claim("sprout-000000000005", backend="podman", store=str(store)))
    assert worker(tmp_path)._cleanup(directory, {"backend": backend}) is False


def test_an_unknown_allocation_still_quarantines_on_podman(tmp_path, store, fake_podman):
    directory = attempt(tmp_path, {"event": "allocation_started", "run_id": "grader", "image": "task"},
                        {"event": "allocation_unknown", "run_id": "grader", "image": "task"})
    assert worker(tmp_path)._cleanup(directory, {"backend": podman_backend(store)}) is False


def test_a_podman_point_is_available_while_its_snapshot_subvolume_exists(tmp_path):
    store = tmp_path / "store"
    (store / "snapshots" / "sprout-000000000006-3").mkdir(parents=True)
    assert podman_snapshot_present(str(store), "sprout-000000000006-3") is True
    assert podman_snapshot_present(str(store), "sprout-000000000006-4") is False
    for unsafe in ("", ".", "..", "../snapshots/sprout-000000000006-3", "a/b", None, 7):
        assert podman_snapshot_present(str(store), unsafe) is False


def test_the_snapshot_validator_asks_the_store_the_jobs_profile_names(tmp_path):
    store = tmp_path / "store"
    (store / "snapshots" / "sprout-000000000007-1").mkdir(parents=True)

    class Jobs:
        def get(self, job_id):
            return {"request": {"profile": "mini", "spec": {"prompt": "x"}}}

    config = {"profiles": {"mini": {"run_defaults": {"backend": podman_backend(store)}}}}
    valid = snapshot_validator(Jobs(), config)
    assert valid({"job_id": "job", "snapshot_id": "sprout-000000000007-1"}) is True
    assert valid({"job_id": "job", "snapshot_id": "sprout-000000000007-2"}) is False
    config["profiles"]["mini"]["run_defaults"]["backend"] = {"podman": {"store": str(store)}}
    assert valid({"job_id": "job", "snapshot_id": "sprout-000000000007-1"}) is True, "podman when unnamed"

"""Same-host worker profiles; credential values stay out of database requests."""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import Any

from sprout.contract.backend import backend_name, default_podman_store, snapshot_store
from sprout.runstore.specs import no_credentials


def resolve(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$env"}:
            return os.environ[value["$env"]]
        return {key: resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item) for item in value]
    return value


def referenced_env(value: Any) -> set[str]:
    if isinstance(value, dict):
        if set(value) == {"$env"}:
            return {value["$env"]}
        return set().union(*(referenced_env(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(referenced_env(item) for item in value))
    return set()


def merge(base: dict, updates: dict) -> dict:
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def read_config(path: Path) -> dict:
    body = json.loads(path.read_text())
    no_credentials(body)
    if not isinstance(body.get("profiles"), dict) or not body.get("artifact_root"):
        raise ValueError("Worker config requires profiles and artifact_root")
    return body


def podman_snapshot_present(store: str, snapshot_id: Any) -> bool:
    """Whether the podman store still holds a snapshot: the read-only subvolume
    ``snapshots/<id>``, checked by directory listing as ``inspect snapshots``
    does (``btrfs subvolume show`` needs root)."""
    if (
        not isinstance(snapshot_id, str)
        or snapshot_id in {"", ".", ".."}
        or any(character in snapshot_id for character in "/\\")
    ):
        return False
    return (Path(store) / "snapshots" / snapshot_id).is_dir()


def snapshot_validator(store: Any, config: dict) -> Callable[[dict], bool]:
    def validate(point: dict) -> bool:
        import httpx

        job_id = point.get("job_id")
        if job_id is None:
            with store.transaction() as cursor:
                cursor.execute("SELECT job_id FROM rs_attempts WHERE id=%s", (point["attempt_id"],))
                job_id = cursor.fetchone()["job_id"]
        request = store.get(job_id)["request"]
        profile = config["profiles"][request["profile"]]
        spec = merge(profile.get("run_defaults", {}), request["spec"])
        backend = resolve(spec.get("backend", {}))
        if backend_name(backend) == "podman":
            return podman_snapshot_present(
                snapshot_store(backend) or default_podman_store(), point["snapshot_id"]
            )
        microvm = backend.get("microvm", {})
        url = microvm.get("server_url")
        if backend_name(backend) != "microvm" or not url:
            return False
        repository = profile.get("snapshot_repository")
        if repository:
            if not point["snapshot_id"] or any(character in point["snapshot_id"] for character in "/\\"):
                return False
            directory = Path(repository) / "snapshots" / point["snapshot_id"]
            try:
                manifest = json.loads((directory / "firecracker-manifest.json").read_text())
                if (
                    manifest.get("version") != 1
                    or not manifest.get("rootfs", {}).get("virtualSize")
                    or (directory / "commit").stat().st_size <= 0
                ):
                    return False
                if "vmState" in manifest and (directory / "vm_state.bin").stat().st_size <= 0:
                    return False
            except (OSError, ValueError, TypeError):
                return False
        try:
            response = httpx.get(
                url.rstrip("/") + "/snapshots/" + point["snapshot_id"],
                headers={"X-API-Key": microvm.get("api_key", "")},
                timeout=5,
            )
            metadata = response.json()
            return (
                response.status_code == 200
                and isinstance(metadata, dict)
                and metadata.get("snapshotID") == point["snapshot_id"]
            )
        except (httpx.HTTPError, ValueError):
            return False

    return validate

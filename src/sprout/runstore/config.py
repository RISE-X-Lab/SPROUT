"""The Run Store's config file: the artifact root and the worker profiles of one host.

A profile names each credential as an ``{"$env": NAME}`` reference. ``read_config`` refuses an
inline credential, so no credential value reaches a request in the database. ``resolve`` reads
the referenced variables where a value is needed. ``snapshot_validator`` checks the snapshot of a
recovery point on the backend of its job's profile. docs/RUNSTORE.md ("Start") describes the file.
"""

from __future__ import annotations

import json
import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sprout.contract.backend import backend_name, default_podman_store, snapshot_store
from sprout.runstore.specs import no_credentials

if TYPE_CHECKING:
    from sprout.runstore.store import Store


def resolve(value: Any) -> Any:
    """The JSON ``value`` with each ``{"$env": NAME}`` replaced by ``$NAME``. Raises KeyError if unset."""
    if isinstance(value, dict):
        if set(value) == {"$env"}:
            return os.environ[value["$env"]]
        return {key: resolve(item) for key, item in value.items()}
    if isinstance(value, list):
        return [resolve(item) for item in value]
    return value


def referenced_env(value: Any) -> set[str]:
    """The names of the variables that the ``{"$env": NAME}`` references in the JSON ``value`` name."""
    if isinstance(value, dict):
        if set(value) == {"$env"}:
            return {value["$env"]}
        return set().union(*(referenced_env(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(referenced_env(item) for item in value))
    return set()


def merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    """``updates`` over ``base``, merged key by key where both values are dicts. Neither input changes."""
    result = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = merge(result[key], value)
        else:
            result[key] = value
    return result


def read_config(path: Path) -> dict[str, Any]:
    """The config in ``path``.

    Raises ValueError if it holds an inline credential, or lacks ``profiles`` or ``artifact_root``.
    """
    body: dict[str, Any] = json.loads(path.read_text())
    no_credentials(body)
    if not isinstance(body.get("profiles"), dict) or not body.get("artifact_root"):
        raise ValueError("Worker config requires profiles and artifact_root")
    return body


def podman_snapshot_present(store: str, snapshot_id: object) -> bool:
    """Whether the podman ``store`` still holds the snapshot ``snapshot_id``.

    The snapshot is the subvolume ``snapshots/<id>``. An id that is not a plain directory name gives False.
    """
    if (
        not isinstance(snapshot_id, str)
        or snapshot_id in {"", ".", ".."}
        or any(character in snapshot_id for character in "/\\")
    ):
        return False
    # A directory check, as ``python -m ops.report snapshots`` makes. ``btrfs subvolume show`` needs root.
    return (Path(store) / "snapshots" / snapshot_id).is_dir()


def snapshot_validator(store: Store, config: dict[str, Any]) -> Callable[[dict[str, Any]], bool]:
    """A check of whether the snapshot of a recovery point still exists on its job's backend, for ``Index``.

    The job's profile in ``config`` names the backend. A podman snapshot must be in the store, a microVM
    one known to AgentENV and complete in any ``snapshot_repository``. Other backends give False.
    """

    def validate(point: dict[str, Any]) -> bool:
        import httpx

        job_id = point.get("job_id")
        if job_id is None:
            with store.transaction() as cursor:
                cursor.execute("SELECT job_id FROM rs_attempts WHERE id=%s", (point["attempt_id"],))
                attempt = cursor.fetchone()
            if attempt is None:
                raise KeyError(point["attempt_id"])
            job_id = attempt["job_id"]
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

"""Run Store jobs in the states in which a worker meets them: claimed, with recovery points, left running.

Each function takes a Run Store from the ``run_store`` fixture. A claimed job has a frozen payload. The
attempt directory of a job is ``<artifact root>/<job id>/<attempt id>``, as the worker makes it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any
from uuid import uuid4

from sprout.contract.mini_history import NativePoint
from sprout.runstore.index import Index
from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from sprout.worker.files import write_json
from tests.support.runstore import registered

#: The worker that ran each attempt before the one under test.
OLD_WORKER = "old-worker"


def worker_config(root: Path, **profile: Any) -> dict[str, Any]:
    """A Run Store config whose artifact root is ``root``, with ``profile`` as the ``default`` profile."""
    return {"artifact_root": str(root), "profiles": {"default": profile}}


def rollout(**spec: Any) -> JobSpec:
    """A rollout request of the ``default`` profile. ``spec`` adds to a prompt and a sandbox image."""
    return JobSpec("rollout", {"prompt": "fix the failing test", "sandbox_image": "image", **spec})


def claimed(store: Store, request: JobSpec, *, by: str = OLD_WORKER) -> dict[str, Any]:
    """The job of ``request``, claimed by the worker ``by``, with its payload frozen from its spec.

    ``by`` registers first, on the host ``host-1``.
    """
    registered(store, by)
    job = store.submit(request, uuid4().hex)
    claim = store.claim(by)
    assert claim is not None and claim["id"] == job["id"]
    envelope = {
        "version": 1,
        "job_id": claim["id"],
        "attempt_id": claim["active_attempt"],
        "kind": request.kind,
        "effective_spec": request.spec,
        "profile_config": {},
        "recovery": None,
    }
    store.freeze_payload(claim["id"], claim["lease_token"], envelope)
    return claim


def attempt_directory(root: Path, job: dict[str, Any]) -> Path:
    """The directory of the job's active attempt under the artifact root ``root``."""
    return root / job["id"] / job["active_attempt"]


def left_running(
    store: Store, root: Path, request: JobSpec, *, outcome: dict[str, Any] | None, by: str = OLD_WORKER
) -> dict[str, Any]:
    """The job of ``request``, as the worker ``by`` left it running when it stopped.

    Its attempt wrote ``outcome`` as ``outcome.json`` under the artifact root ``root``, unless None.
    """
    job = claimed(store, request, by=by)
    if outcome is not None:
        write_json(attempt_directory(root, job) / "outcome.json", outcome)
    return job


def _native_prefix(path: Path) -> dict[str, Any]:
    """Write a native history prefix of mini-swe-agent to ``path``, and return the reference that names it."""
    data = b'{"role": "system", "content": "You are a helpful assistant."}\n'
    path.write_bytes(data)
    return {
        "path": str(path),
        "byte_length": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "slot": "mini-swe-agent",
    }


def publish_points(store: Store, job: dict[str, Any], *, snapshots: list[str], prefix: Path) -> None:
    """Publish a recovery point for each of ``snapshots``, after one tool call each, for the job's attempt."""
    calls = [
        {"seq": step, "type": "tool.started", "call_id": "call-%d" % step, "name": "bash", "args": {}}
        for step in range(1, len(snapshots) + 1)
    ]
    native = _native_prefix(prefix)
    points = [NativePoint(step, step, snapshot, native) for step, snapshot in enumerate(snapshots, 1)]
    Index(store).project(job["id"], job["lease_token"], {"context": {}}, calls, points)

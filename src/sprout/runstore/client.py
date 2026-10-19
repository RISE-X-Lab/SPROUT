"""A client of the Run Store's HTTP API (docs/RUNSTORE.md, "Submit and observe").

Reading a result does not consume it: a finished job's result stays in the Run Store and reads the
same each time. Each method raises ``httpx.HTTPStatusError`` when the API refuses the request.
"""

from __future__ import annotations

import time
from typing import Any

import httpx

from sprout.runstore.specs import JobSpec


class Client:
    """A connection to the Run Store at ``url``, authenticated by the bearer ``token``."""

    def __init__(self, url: str, token: str) -> None:
        self.http = httpx.Client(
            base_url=url.rstrip("/"), headers={"Authorization": "Bearer " + token}, timeout=30
        )

    def close(self) -> None:
        self.http.close()

    def submit(self, job: JobSpec, idempotency_key: str) -> str:
        """Submit ``job`` under ``idempotency_key``, and return the job's id.

        The same job under the same key returns the same id. Another job under that key gets the
        status 409.
        """
        response = self.http.post(
            "/v1/jobs", json=job.validate(), headers={"Idempotency-Key": idempotency_key}
        )
        response.raise_for_status()
        job_id: str = response.json()["id"]
        return job_id

    def get(self, job_id: str) -> dict[str, Any]:
        """The job ``job_id``: its request, lease, state, result and error."""
        response = self.http.get(f"/v1/jobs/{job_id}")
        response.raise_for_status()
        job: dict[str, Any] = response.json()
        return job

    def get_result(self, job_id: str) -> dict[str, Any]:
        """The job's ``state``, ``result`` and ``error``, and whether it is ``ready`` (finished)."""
        response = self.http.get(f"/v1/jobs/{job_id}/result")
        response.raise_for_status()
        result: dict[str, Any] = response.json()
        return result

    def recovery_points(self, job_id: str) -> list[dict[str, Any]]:
        """The recovery points of the job's active attempt, each with whether it is ``available``."""
        response = self.http.get(f"/v1/jobs/{job_id}/recovery-points")
        response.raise_for_status()
        points: list[dict[str, Any]] = response.json()
        return points

    def branch(
        self,
        job_id: str,
        point_id: str,
        *,
        idempotency_key: str,
        continuation: dict[str, Any],
        budget: dict[str, Any] | None = None,
        **overrides: object,
    ) -> str:
        """Submit a branch of ``job_id`` from its recovery point ``point_id``, and return the branch's id.

        ``continuation`` is what the actor is given after the restored history. ``budget`` sets the
        branch's own ``max_turns`` and ``deadline_at``. ``overrides`` can set only ``BRANCH_OVERRIDES``.
        """
        body = {"point_id": point_id, "continuation": continuation, "overrides": overrides}
        if budget is not None:
            body["budget"] = budget
        response = self.http.post(
            f"/v1/jobs/{job_id}/branch", json=body, headers={"Idempotency-Key": idempotency_key}
        )
        response.raise_for_status()
        branch_id: str = response.json()["id"]
        return branch_id

    def wait(self, job_id: str, timeout_s: float = 3600, interval_s: float = 1) -> dict[str, Any]:
        """Poll ``get_result`` until the job's result is ready, and return it.

        Raises TimeoutError after ``timeout_s`` seconds. The timeout does not cancel the job.
        """
        deadline = time.monotonic() + timeout_s
        while True:
            result = self.get_result(job_id)
            if result["ready"]:
                return result
            if time.monotonic() >= deadline:
                raise TimeoutError("Client wait timed out; job remains queued/running")
            time.sleep(min(interval_s, max(0, deadline - time.monotonic())))

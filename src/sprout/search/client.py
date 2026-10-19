"""The Run Store calls a coordinator makes, over HTTP only.

Execution state and retries belong to the Run Store and its workers. Each call raises
``httpx.HTTPError`` when the request fails or the Run Store answers with an error status.
"""

from typing import Any
from urllib.parse import quote

from sprout.runstore.client import Client


def _segment(value: str) -> str:
    return quote(value, safe="")


class RunStoreClient(Client):
    """The Run Store ``Client``, plus a stop request and the reads of an attempt's tools and points."""

    def stop(self, job_id: str) -> dict[str, Any]:
        """Cancel a queued job or stop a running one (``Store.request_stop``), and return the job."""
        response = self.http.post(f"/v1/jobs/{_segment(job_id)}/stop")
        response.raise_for_status()
        job: dict[str, Any] = response.json()
        return job

    def tools(self, job_id: str, *, attempt_id: str, after: int = 0) -> list[dict[str, Any]]:
        """One page of the attempt's tool calls whose ``depth`` is above ``after``, in depth order."""
        response = self.http.get(
            f"/v1/jobs/{_segment(job_id)}/tools",
            params={
                "attempt_id": attempt_id,
                "after": after,
            },
        )
        response.raise_for_status()
        page: list[dict[str, Any]] = response.json()
        return page

    def points(self, job_id: str, *, attempt_id: str) -> list[dict[str, Any]]:
        """The recovery points of one attempt of the job, in message order, each with its ``available``."""
        response = self.http.get(
            f"/v1/jobs/{_segment(job_id)}/recovery-points", params={"attempt_id": attempt_id}
        )
        response.raise_for_status()
        points: list[dict[str, Any]] = response.json()
        return points

    def all_tools(self, job_id: str, *, attempt_id: str) -> list[dict[str, Any]]:
        """Every tool call of the attempt, in depth order. Raises ValueError when a page does not advance."""
        result: list[dict[str, Any]] = []
        after = 0
        while True:
            page = self.tools(job_id, attempt_id=attempt_id, after=after)
            if not page:
                return result
            cursor = page[-1]["depth"]
            if cursor <= after:
                raise ValueError("Run Store tool cursor did not advance")
            result.extend(page)
            after = cursor

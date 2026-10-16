"""HTTP only: execution state and retries belong to Run Store workers."""

from urllib.parse import quote

from sprout.runstore.client import Client


def segment(value: str) -> str:
    return quote(value, safe="")


class RunStoreClient(Client):
    def stop(self, job_id: str) -> dict:
        """Cancel a queued job or stop a running one (``Store.request_stop``)."""
        response = self.http.post(f"/v1/jobs/{segment(job_id)}/stop")
        response.raise_for_status()
        return response.json()

    def tools(self, job_id: str, *, attempt_id: str, after: int = 0) -> list[dict]:
        response = self.http.get(
            f"/v1/jobs/{segment(job_id)}/tools",
            params={
                "attempt_id": attempt_id,
                "after": after,
            },
        )
        response.raise_for_status()
        return response.json()

    def points(self, job_id: str, *, attempt_id: str) -> list[dict]:
        response = self.http.get(
            f"/v1/jobs/{segment(job_id)}/recovery-points", params={"attempt_id": attempt_id}
        )
        response.raise_for_status()
        return response.json()

    def all_tools(self, job_id: str, *, attempt_id: str) -> list[dict]:
        result = []
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

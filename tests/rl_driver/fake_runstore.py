"""A fake Run Store: the routes of its HTTP API that a ``RunStoreClient`` calls, in memory, for one test.

``FakeRunStore.client()`` sends its requests to the fake through an ``httpx.MockTransport``. The fake
checks each submission with the Run Store's own ``JobSpec``, and it keeps one job for each idempotency
key, as ``sprout.runstore.api`` does. Job ids are ``job-0``, ``job-1`` and so on, in submission order.
The attempt of ``job-N`` is ``attempt-job-N``. The test plays the worker: it starts and finishes each
job, and it publishes the tool calls and the recovery points of an attempt. ``test_runstore_fidelity.py``
runs the same driver on the real Run Store.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from typing import Any

import httpx

from sprout.contract.continuation import Continuation
from sprout.runstore.specs import BRANCH_OVERRIDES, JobSpec, canonical, validate_branch_budget
from sprout.search.client import RunStoreClient

#: The base URL of the fake. No request leaves the process.
URL = "http://runstore.fake"
#: The most rows that one page of ``/tools`` holds, the Run Store API's default ``limit``.
TOOLS_PAGE = 1000


class FakeRunStore:
    """Jobs, tool calls and recovery points in memory, behind the Run Store's HTTP routes."""

    def __init__(self) -> None:
        #: Each job by its id, as ``GET /v1/jobs/{id}`` returns it.
        self.jobs: dict[str, dict[str, Any]] = {}
        #: The path and the idempotency key of each POST that submits a job, repeats included.
        self.posts: list[tuple[str, str]] = []
        #: The parent job id and the JSON body of each valid ``POST /v1/jobs/{id}/branch``, repeats included.
        self.branch_posts: list[tuple[str, dict[str, Any]]] = []
        self._keys: dict[str, str] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}
        self._points: dict[str, list[dict[str, Any]]] = {}
        self._lose_next_reply = False
        self._down = False
        self._before_points_reply: Callable[[], None] | None = None

    def client(self) -> RunStoreClient:
        """A client of this fake. The caller closes it."""
        client = RunStoreClient(URL, "fake-runstore-token")
        client.http.close()
        client.http = httpx.Client(
            base_url=URL, headers=client.http.headers, transport=httpx.MockTransport(self._handle)
        )
        return client

    def ids(self, kind: str) -> list[str]:
        """The ids of the jobs of ``kind``, in submission order."""
        return [job_id for job_id, job in self.jobs.items() if job["kind"] == kind]

    def request(self, job_id: str) -> dict[str, Any]:
        """The request body that the fake stored for the job, as the real Run Store stores it."""
        request: dict[str, Any] = self.jobs[job_id]["request"]
        return request

    def stop_requested(self, job_id: str) -> bool:
        """Whether a client asked the job to stop while it ran."""
        return self.jobs[job_id]["stop_requested_at"] is not None

    def start(self, job_id: str) -> None:
        """Lease the job to a worker: it runs its first attempt."""
        self.jobs[job_id].update(state="running", phase="claimed", active_attempt="attempt-" + job_id)

    def finish(
        self,
        job_id: str,
        *,
        state: str = "succeeded",
        result: dict[str, Any] | None = None,
        error: str | None = None,
    ) -> None:
        """End the job's attempt in ``state``: succeeded or failed."""
        self.jobs[job_id].update(
            state=state, phase=state, active_attempt="attempt-" + job_id, result=result, error=error
        )

    def publish(self, job_id: str, *, tools: list[dict[str, Any]], points: list[dict[str, Any]]) -> None:
        """Index the tool calls and the recovery points of the job's attempt.

        Each point gives its ``id``, ``tool_depth``, ``message_step``, ``snapshot_id`` and ``available``.
        """
        self._tools[job_id] = list(tools)
        self._points[job_id] = [{"job_id": job_id, **point} for point in points]

    def lose_next_reply(self) -> None:
        """Accept the next submission, then fail its reply, as a connection that drops after the commit."""
        self._lose_next_reply = True

    def go_down(self) -> None:
        """Refuse every later connection."""
        self._down = True

    def before_points_reply(self, action: Callable[[], None]) -> None:
        """Run ``action`` each time a client reads recovery points, before the reply."""
        self._before_points_reply = action

    def _handle(self, request: httpx.Request) -> httpx.Response:
        if self._down:
            raise httpx.ConnectError("The fake Run Store is down", request=request)
        parts = request.url.path.strip("/").split("/")
        body = json.loads(request.content) if request.content else None
        try:
            return self._route(request, parts=parts, body=body)
        except KeyError:
            return httpx.Response(404, json={"detail": "Reference not found"})
        except (ValueError, TypeError) as error:
            return httpx.Response(422, json={"detail": str(error)})

    def _route(self, request: httpx.Request, *, parts: list[str], body: Any) -> httpx.Response:
        params = request.url.params
        if request.method == "POST" and parts == ["v1", "jobs"]:
            return self._submit(request, JobSpec.from_dict(body).validate())
        job = self.jobs[parts[2]]
        route = (request.method, parts[3] if len(parts) > 3 else "")
        if route == ("GET", ""):
            return httpx.Response(200, json=job)
        if route == ("POST", "stop"):
            if job["state"] == "queued":
                job.update(state="cancelled", phase="cancelled")
            elif job["state"] == "running":
                # The real Run Store records the time of the first request. A client reads whether it is set.
                job["stop_requested_at"] = job["stop_requested_at"] or "2026-01-01T00:00:00+00:00"
            return httpx.Response(200, json=job)
        if route == ("POST", "branch"):
            return self._branch(request, job=job, body=body)
        if params.get("attempt_id", job["active_attempt"]) != job["active_attempt"]:
            raise KeyError(params["attempt_id"])
        if route == ("GET", "tools"):
            after = int(params.get("after", 0))
            rows = [row for row in self._tools.get(job["id"], []) if row["depth"] > after]
            return httpx.Response(200, json=rows[:TOOLS_PAGE])
        if route == ("GET", "recovery-points"):
            if self._before_points_reply is not None:
                self._before_points_reply()
            return httpx.Response(200, json=self._points.get(job["id"], []))
        raise AssertionError(f"The fake Run Store has no route {request.method} {request.url.path}")

    def _branch(self, request: httpx.Request, *, job: dict[str, Any], body: Any) -> httpx.Response:
        # The checks of sprout.runstore.api's branch route that a driver can trip.
        if not isinstance(body, dict) or set(body) - {"point_id", "continuation", "overrides", "budget"}:
            raise ValueError("A branch names point_id and continuation, and optionally overrides and budget")
        point = {p["id"]: p for p in self._points.get(job["id"], [])}[body["point_id"]]
        if not point["available"]:
            return httpx.Response(409, json={"detail": "Recovery point is not available"})
        if set(body.get("overrides", {})) - BRANCH_OVERRIDES:
            raise ValueError("A branch can change only " + ", ".join(sorted(BRANCH_OVERRIDES)))
        parent = JobSpec.from_dict(job["request"])
        if "budget" in body and parent.spec.get("extra", {}).get("rollout_contract") is None:
            raise ValueError("A branch budget bounds a rollout contract; the parent run has none")
        branch = replace(
            parent,
            spec={**parent.spec, **body.get("overrides", {})},
            parent_point=point["id"],
            continuation=Continuation.from_dict(body.get("continuation")).to_dict(),
            branch_budget=validate_branch_budget(body["budget"]) if "budget" in body else None,
        )
        self.branch_posts.append((job["id"], body))
        return self._submit(request, branch.validate())

    def _submit(self, request: httpx.Request, body: dict[str, Any]) -> httpx.Response:
        key = request.headers["Idempotency-Key"]
        self.posts.append((request.url.path, key))
        if key not in self._keys:
            job_id = f"job-{len(self.jobs)}"
            self._keys[key] = job_id
            self.jobs[job_id] = {
                "id": job_id,
                "kind": body["kind"],
                "idempotency_key": key,
                "request": body,
                "state": "queued",
                "phase": "queued",
                "active_attempt": None,
                "result": None,
                "error": None,
                "stop_requested_at": None,
            }
        job = self.jobs[self._keys[key]]
        if canonical(job["request"]) != canonical(body):
            return httpx.Response(409, json={"detail": "Idempotency key already names a different request"})
        if self._lose_next_reply:
            self._lose_next_reply = False
            raise httpx.ReadError("The reply was lost after the Run Store accepted the job", request=request)
        return httpx.Response(202, json=job)

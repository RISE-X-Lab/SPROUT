"""A Run Store API in memory: the routes that ``RunStoreClient`` calls, for the tests of ``sprout.search``.

``FakeRunStore`` answers its ``client`` through an ``httpx.MockTransport``, so no socket is opened.
It keeps one job for each idempotency key, numbers the jobs ``job-0``, ``job-1`` and so on in the
order of submission, and runs none of them. A test plays the worker with ``run``, ``finish`` and
``finish_rollout``. An answer carries the fields of the real API's answer that a coordinator reads.
``test_fake_runstore.py`` checks these answers against the real API on PostgreSQL.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass, replace
from typing import Any

import httpx

from sprout.contract.continuation import Continuation
from sprout.runstore.specs import JobSpec, canonical, validate_branch_budget
from sprout.search.client import RunStoreClient

URL = "http://runstore"
#: What the fake records as the time of a stop request for a running job.
STOP_REQUESTED_AT = "2026-01-01T00:00:00+00:00"


@dataclass(frozen=True)
class _Fault:
    method: str
    route: str
    status: int | None
    lose_reply: bool = False
    #: How many matching requests the fault lets pass first.
    after: int = 0


class FakeRunStore:
    """The jobs, tool calls and recovery points of one Run Store, and the requests that reached it.

    A route is ``jobs`` (``POST /v1/jobs``), ``job`` (``GET /v1/jobs/{id}``), ``branch``, ``stop``,
    ``tools`` or ``recovery-points``. An unknown job id gets a 404, as from the real API.
    """

    def __init__(self) -> None:
        self.client = RunStoreClient(URL, "fixture")
        self.client.http.close()
        self.client.http = httpx.Client(base_url=URL, transport=httpx.MockTransport(self._answer))
        #: The method and the route of each request, in arrival order, the refused ones included.
        self.requests: list[tuple[str, str]] = []
        #: Each exception that the fake raised while it answered a request: a defect of the fake.
        self.defects: list[str] = []
        self._jobs: dict[str, dict[str, Any]] = {}
        #: The parent job id (None for ``POST /v1/jobs``) and the body that made each job.
        self._posted: dict[str, tuple[str | None, dict[str, Any]]] = {}
        #: The spec that each job runs. A branch runs the spec of its parent, with its overrides.
        self._specs: dict[str, dict[str, Any]] = {}
        self._keys: dict[str, str] = {}
        self._tools: dict[str, list[dict[str, Any]]] = {}
        self._points: dict[str, list[dict[str, Any]]] = {}
        self._faults: list[_Fault] = []
        self._hooks: list[tuple[str, str, Callable[[], object]]] = []

    def run(self, job_id: str) -> None:
        """Claim the job for a worker: it runs as its attempt ``attempt-<job id>``."""
        self._jobs[job_id].update(state="running", phase="running", active_attempt="attempt-" + job_id)

    def finish(self, job_id: str, result: dict[str, Any], *, state: str = "succeeded") -> None:
        """End the job's attempt with ``result``, in ``state``: succeeded or failed."""
        job = self._jobs[job_id]
        job.update(state=state, phase=state, result=deepcopy(result), error=result.get("error"))
        job["active_attempt"] = job["active_attempt"] or "attempt-" + job_id

    def record(
        self, job_id: str, *, tools: list[dict[str, Any]], points: list[dict[str, Any]] | None = None
    ) -> None:
        """Index the tool calls and publish the recovery points of the job's active attempt."""
        attempt = self._jobs[job_id]["active_attempt"]
        assert attempt, "a job records tool calls only after a worker claimed it"
        self._tools[attempt] = deepcopy(tools)
        self._points[attempt] = deepcopy(points or [])

    def finish_rollout(self, job_id: str, result: dict[str, Any]) -> None:
        """Finish a rollout with ``result``, three tool calls and a recovery point after each of the last two.

        The points are ``point(job_id, 2)`` and ``point(job_id, 3)``. The second one is at the final
        tool boundary."""
        self.finish(job_id, result)
        self.record(job_id, tools=complete_tools(3), points=[point(job_id, 2), point(job_id, 3)])

    def fail_next(self, method: str, route: str, *, status: int | None = None, after: int = 0) -> None:
        """Answer the next ``method`` request to ``route`` after ``after`` others with ``status``. Without
        ``status``, the client gets a ``httpx.ConnectError``. Either way, the request changes nothing."""
        self._faults.append(_Fault(method, route, status, after=after))

    def lose_next_reply(self, *, after: int = 0) -> None:
        """Accept the next submission or branch after ``after`` others, then lose the reply: the client gets
        a ``httpx.ReadError``."""
        self._faults.append(_Fault("POST", "", None, lose_reply=True, after=after))

    def before_next(self, method: str, route: str, action: Callable[[], object]) -> None:
        """Call ``action`` when the next ``method`` request to ``route`` arrives, before the answer."""
        self._hooks.append((method, route, action))

    def job(self, job_id: str) -> dict[str, Any]:
        return deepcopy(self._jobs[job_id])

    def job_ids(self, kind: str) -> list[str]:
        """The ids of the jobs of ``kind``, in the order of submission. A branch is a ``rollout``."""
        return [job_id for job_id, job in self._jobs.items() if job["kind"] == kind]

    def submitted(self, kind: str) -> list[dict[str, Any]]:
        """The body of each accepted ``POST /v1/jobs`` of ``kind``, in the order of submission."""
        return [
            deepcopy(body)
            for parent, body in self._posted.values()
            if parent is None and body["kind"] == kind
        ]

    def branches(self) -> list[tuple[str, dict[str, Any]]]:
        """The parent job id and the body of each accepted branch, in the order of submission."""
        return [(parent, deepcopy(body)) for parent, body in self._posted.values() if parent is not None]

    def keys(self) -> list[str]:
        """The idempotency key of each job, in the order of submission."""
        return [job["idempotency_key"] for job in self._jobs.values()]

    def _answer(self, request: httpx.Request) -> httpx.Response:
        route, job_id = _route(request.url.path)
        self.requests.append((request.method, route))
        hook = next((hook for hook in self._hooks if hook[:2] == (request.method, route)), None)
        if hook is not None:
            self._hooks.remove(hook)
            hook[2]()
        fault = self._take_fault(request.method, route)
        if fault is not None and not fault.lose_reply:
            if fault.status is None:
                raise httpx.ConnectError("the fake Run Store is unreachable", request=request)
            return httpx.Response(fault.status, json={"detail": "the fake Run Store refused it"})
        if job_id is not None and job_id not in self._jobs:
            return httpx.Response(404, json={"detail": "Reference not found"})
        try:
            response = self._dispatch(request, route=route, job_id=job_id)
        except Exception as error:
            # The coordinator keeps an error of its client as a last_error and goes on, so a defect of
            # the fake would pass unseen. The fixture fails the test on it.
            self.defects.append("%s %s: %r" % (request.method, request.url.path, error))
            raise
        if fault is not None:
            raise httpx.ReadError("the reply of the fake Run Store was lost", request=request)
        return response

    def _take_fault(self, method: str, route: str) -> _Fault | None:
        for index, fault in enumerate(self._faults):
            if fault.method == method and (
                fault.route == route or (fault.lose_reply and route in {"jobs", "branch"})
            ):
                if fault.after:
                    self._faults[index] = replace(fault, after=fault.after - 1)
                    return None
                del self._faults[index]
                return fault
        return None

    def _dispatch(self, request: httpx.Request, *, route: str, job_id: str | None) -> httpx.Response:
        if (request.method, route) == ("POST", "jobs"):
            return self._submit(request, parent=None)
        if job_id is None:
            return httpx.Response(404, json={"detail": "Not Found"})
        if (request.method, route) == ("POST", "branch"):
            return self._branch(request, job_id=job_id)
        if (request.method, route) == ("POST", "stop"):
            return httpx.Response(200, json=self._stop(job_id))
        if (request.method, route) == ("GET", "job"):
            return httpx.Response(200, json=self._jobs[job_id])
        if request.method == "GET" and route in {"tools", "recovery-points"}:
            return self._attempt_rows(request, route=route, job_id=job_id)
        return httpx.Response(404, json={"detail": "Not Found"})

    def _submit(self, request: httpx.Request, *, parent: str | None) -> httpx.Response:
        body = json.loads(request.content)
        key = request.headers.get("Idempotency-Key", "")
        try:
            kind = "rollout" if parent is not None else JobSpec.from_dict(body).kind
        except (TypeError, ValueError) as error:
            return httpx.Response(422, json={"detail": str(error)})
        if key in self._keys:
            job_id = self._keys[key]
            if canonical(self._posted[job_id]) != canonical((parent, body)):
                return httpx.Response(
                    409, json={"detail": "Idempotency key already names a different request"}
                )
            return httpx.Response(202, json=self._jobs[job_id])
        job_id = "job-%d" % len(self._jobs)
        self._jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "state": "queued",
            "phase": "queued",
            "idempotency_key": key,
            "active_attempt": None,
            "result": None,
            "error": None,
            "stop_requested_at": None,
        }
        self._posted[job_id] = (parent, body)
        self._specs[job_id] = (
            {**self._specs[parent], **body["overrides"]} if parent is not None else body["spec"]
        )
        self._keys[key] = job_id
        return httpx.Response(202, json=self._jobs[job_id])

    def _branch(self, request: httpx.Request, *, job_id: str) -> httpx.Response:
        body = json.loads(request.content)
        owners = [attempt for attempt, points in self._points.items() if body["point_id"] in _ids(points)]
        if not owners:
            return httpx.Response(404, json={"detail": "Reference not found"})
        if owners != [self._jobs[job_id]["active_attempt"]]:
            return httpx.Response(422, json={"detail": "Branch point does not belong to this rollout"})
        try:
            Continuation.from_dict(body["continuation"])
            if "budget" in body:
                validate_branch_budget(body["budget"])
                if "rollout_contract" not in self._specs[job_id].get("extra", {}):
                    raise ValueError("A branch budget bounds a rollout contract; the parent run has none")
        except ValueError as error:
            return httpx.Response(422, json={"detail": str(error)})
        return self._submit(request, parent=job_id)

    def _stop(self, job_id: str) -> dict[str, Any]:
        job = self._jobs[job_id]
        if job["state"] == "queued":
            job.update(state="cancelled", phase="cancelled")
        elif job["state"] == "running":
            job["stop_requested_at"] = job["stop_requested_at"] or STOP_REQUESTED_AT
        return job

    def _attempt_rows(self, request: httpx.Request, *, route: str, job_id: str) -> httpx.Response:
        attempt = request.url.params.get("attempt_id") or self._jobs[job_id]["active_attempt"]
        if attempt != self._jobs[job_id]["active_attempt"]:
            return httpx.Response(404, json={"detail": "Reference not found"})
        if route == "recovery-points":
            return httpx.Response(200, json=self._points.get(attempt, []))
        after = int(request.url.params.get("after", 0))
        return httpx.Response(200, json=[row for row in self._tools.get(attempt, []) if row["depth"] > after])


def _route(path: str) -> tuple[str, str | None]:
    """The route of a request path, and the job id in the path or None."""
    parts = path.strip("/").split("/")
    if parts == ["v1", "jobs"]:
        return "jobs", None
    if len(parts) == 3:
        return "job", parts[2]
    if len(parts) == 4:
        return parts[3], parts[2]
    return path, None


def _ids(points: list[dict[str, Any]]) -> set[str]:
    return {point["id"] for point in points}


def complete_tools(depth: int) -> list[dict[str, Any]]:
    """The tool calls ``1..depth`` of an attempt, each with its response."""
    return [{"depth": n, "call_id": "call-%d" % n, "response": {"output": "ok"}} for n in range(1, depth + 1)]


def point(job_id: str, depth: int, *, available: bool = True) -> dict[str, Any]:
    """The recovery point of ``job_id`` after tool call ``depth``, at message step ``depth - 1``.

    Its id is ``<job id>-step<depth>``. Its snapshot id has ``depth`` letters, ``AB`` at depth 2."""
    return {
        "id": "%s-step%d" % (job_id, depth),
        "tool_depth": depth,
        "message_step": depth - 1,
        "snapshot_id": "ABCDEFGHIJ"[:depth],
        "available": available,
    }

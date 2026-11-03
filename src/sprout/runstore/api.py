"""The Run Store's HTTP API (docs/RUNSTORE.md, "Submit and observe").

Every route needs the bearer token. Reading a result does not consume it. A ``Conflict`` answers
409, a ``ValueError`` 422 and a ``KeyError`` 404.
"""

from __future__ import annotations

import secrets
from dataclasses import replace
from typing import TYPE_CHECKING, Any

from sprout.contract.agents.history import load_prefix
from sprout.contract.continuation import Continuation, prepare
from sprout.runstore.config import merge
from sprout.runstore.index import Index
from sprout.runstore.specs import (
    BRANCH_OVERRIDES,
    JobSpec,
    defaults_key,
    validate_branch_budget,
    validate_continuation,
)
from sprout.runstore.store import Conflict, Store

if TYPE_CHECKING:
    from collections.abc import Callable

    from fastapi import FastAPI, Request


def create_app(
    store: Store, token: str, *, index: Index | None = None, profiles: dict[str, dict[str, Any]] | None = None
) -> FastAPI:
    """The API over ``store``, for clients that send ``token``. Raises ValueError if ``token`` is empty.

    With ``profiles``, a submission must name one of them, and its spec takes that profile's defaults.
    Without ``index``, no recovery point is available, so the API accepts no branch.
    """
    from fastapi import Depends, FastAPI, Header, HTTPException

    if not token:
        raise ValueError("A control-plane bearer token is required")
    index = index or Index(store)

    def authenticate(authorization: str = Header(default="")) -> None:
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Invalid control-plane token")

    app = FastAPI(dependencies=[Depends(authenticate)])
    _add_error_handlers(app)

    def submit(body: dict[str, Any], key: str) -> dict[str, Any]:
        return _submit(store=store, index=index, profiles=profiles, body=body, key=key)

    # Each route sets response_model=None, so that FastAPI ignores its return annotation. A response
    # model would write each UTC timestamp of a row as ...Z instead of ...+00:00.
    _add_job_routes(app, store=store, submit=submit)
    _add_attempt_routes(app, store=store, index=index)
    _add_control_routes(app, store=store, index=index, submit=submit)
    return app


def _add_error_handlers(app: FastAPI) -> None:
    from fastapi.responses import JSONResponse

    @app.exception_handler(Conflict)
    async def conflict_handler(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def validation_handler(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def missing_handler(request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": "Reference not found"})


def _submit(
    *, store: Store, index: Index, profiles: dict[str, dict[str, Any]] | None, body: dict[str, Any], key: str
) -> dict[str, Any]:
    """Queue the request ``body`` under the idempotency ``key``, and return its job."""
    try:
        spec = JobSpec.from_dict(body)
    except TypeError as error:
        raise ValueError("Invalid JobSpec fields") from error
    if profiles is not None and spec.profile not in profiles:
        raise ValueError("Unknown worker profile")
    if profiles is not None:
        defaults = profiles[spec.profile].get(defaults_key(spec.kind), {})
        spec = replace(spec, spec=merge(defaults, spec.spec))
    if spec.parent_point:
        point = index.get_point(spec.parent_point)
        if not index.valid(point):
            raise Conflict("Recovery point no longer has both native prefix and snapshot")
        parent = store.get(point["job_id"])["request"]
        validate_continuation(parent, spec.validate())
        if spec.kind != "rollout" or spec.spec.get("slot", "mini-swe-agent") != point["native"]["slot"]:
            raise ValueError("Continuation must use its native history slot")
        if spec.profile != parent["profile"]:
            raise ValueError("v1 continuation must use the source execution profile")
        # An inserted assistant turn must fit the history that it continues (fresh call ids).
        # Mini checks it again against the tools that it offers.
        continuation = Continuation.from_dict(spec.continuation)
        prefix = load_prefix(point["native"]) if continuation.guidance == "assistant-turn" else ()
        prepare(continuation, prefix)
    return store.submit(spec, key)


def _add_job_routes(
    app: FastAPI, *, store: Store, submit: Callable[[dict[str, Any], str], dict[str, Any]]
) -> None:
    """Add the routes that submit jobs and read a job, its result and its attempts."""
    from fastapi import Header

    @app.post("/v1/jobs", status_code=202, response_model=None)
    def submit_job(body: dict[str, Any], idempotency_key: str = Header(default="")) -> dict[str, Any]:
        return submit(body, idempotency_key)

    @app.get("/v1/jobs", response_model=None)
    def list_jobs(state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        return store.list_jobs(state, limit)

    @app.get("/v1/jobs/{job_id}", response_model=None)
    def get_job(job_id: str) -> dict[str, Any]:
        return store.get(job_id)

    @app.get("/v1/jobs/{job_id}/result", response_model=None)
    def get_result(job_id: str) -> dict[str, Any]:
        job = store.get(job_id)
        return {
            "job_id": job_id,
            "state": job["state"],
            "result": job["result"],
            "error": job["error"],
            "ready": job["state"] in {"succeeded", "failed", "cancelled"},
        }

    @app.get("/v1/jobs/{job_id}/attempts", response_model=None)
    def attempts(job_id: str) -> list[dict[str, Any]]:
        store.get(job_id)
        return store.attempts(job_id)


def _add_attempt_routes(app: FastAPI, *, store: Store, index: Index) -> None:
    """Add the routes that read the events, tool calls and recovery points of an attempt."""

    def selected(job_id: str, attempt_id: str | None) -> str:
        """``attempt_id``, or the job's active attempt when it is None. Raises KeyError if not the job's."""
        job = store.get(job_id)
        identifier = attempt_id or job["active_attempt"]
        if identifier not in {attempt["id"] for attempt in store.attempts(job_id)}:
            raise KeyError(identifier)
        return identifier

    @app.get("/v1/jobs/{job_id}/events", response_model=None)
    def events(
        job_id: str, attempt_id: str | None = None, after: int = 0, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return store.events(selected(job_id, attempt_id), after, limit)

    @app.get("/v1/jobs/{job_id}/tools", response_model=None)
    def tools(
        job_id: str, attempt_id: str | None = None, after: int = 0, limit: int = 1000
    ) -> list[dict[str, Any]]:
        return index.tools(selected(job_id, attempt_id), after, limit)

    @app.get("/v1/jobs/{job_id}/recovery-points", response_model=None)
    def recovery_points(job_id: str, attempt_id: str | None = None) -> list[dict[str, Any]]:
        return [
            {**point, "available": index.valid(point)} for point in index.points(selected(job_id, attempt_id))
        ]


def _add_control_routes(
    app: FastAPI, *, store: Store, index: Index, submit: Callable[[dict[str, Any], str], dict[str, Any]]
) -> None:
    """Add the routes that branch, cancel and stop a job."""
    from fastapi import Header, HTTPException

    @app.post("/v1/jobs/{job_id}/branch", status_code=202, response_model=None)
    def branch(
        job_id: str, body: dict[str, Any], idempotency_key: str = Header(default="")
    ) -> dict[str, Any]:
        if not isinstance(body, dict) or set(body) - {"point_id", "continuation", "overrides", "budget"}:
            raise ValueError("A branch names point_id and continuation, and optionally overrides and budget")
        parent = JobSpec.from_dict(store.get(job_id)["request"])
        point = index.get_point(body["point_id"])
        if point["job_id"] != job_id or parent.kind != "rollout":
            raise ValueError("Branch point does not belong to this rollout")
        overrides = body.get("overrides", {})
        if not isinstance(overrides, dict) or set(overrides) - BRANCH_OVERRIDES:
            raise ValueError(
                "A branch can change only %s; what the actor is given next is its continuation"
                % ", ".join(sorted(BRANCH_OVERRIDES))
            )
        continuation = Continuation.from_dict(body.get("continuation"))
        budget = validate_branch_budget(body["budget"]) if "budget" in body else None
        if budget is not None and parent.spec.get("extra", {}).get("rollout_contract") is None:
            raise ValueError("A branch budget bounds a rollout contract; the parent run has none")
        request = replace(
            parent,
            spec={**parent.spec, **overrides},
            parent_point=point["id"],
            continuation=continuation.to_dict(),
            branch_budget=budget,
        )
        return submit(request.validate(), idempotency_key)

    @app.post("/v1/jobs/{job_id}/cancel", response_model=None)
    def cancel(job_id: str) -> dict[str, Any]:
        store.cancel_queued(job_id)
        return store.get(job_id)

    @app.post("/v1/jobs/{job_id}/stop", response_model=None)
    def stop(job_id: str) -> dict[str, Any]:
        try:
            return store.request_stop(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

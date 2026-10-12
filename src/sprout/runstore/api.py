"""Authenticated control-plane API. Result polling never consumes the result."""

from __future__ import annotations

from dataclasses import replace
import secrets

from sprout.harness.agents.mini_swe.continuation import Continuation, prepare
from sprout.harness.agents.mini_swe.history import load_prefix
from sprout.runstore.index import Index
from sprout.runstore.config import merge
from sprout.runstore.specs import (BRANCH_OVERRIDES, JobSpec, defaults_key, digest, validate_branch_budget,
                                   validate_continuation)
from sprout.runstore.store import Conflict, Store


def create_app(store: Store, token: str, *, index: Index | None = None, profiles: dict | None = None):
    from fastapi import Depends, FastAPI, Header, HTTPException
    from fastapi.responses import JSONResponse

    if not token:
        raise ValueError("A control-plane bearer token is required")
    index = index or Index(store)

    def authenticate(authorization: str = Header(default="")) -> None:
        if not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Invalid control-plane token")

    app = FastAPI(dependencies=[Depends(authenticate)])

    @app.exception_handler(Conflict)
    async def conflict_handler(request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    async def validation_handler(request, error):
        return JSONResponse(status_code=422, content={"detail": str(error)})

    @app.exception_handler(KeyError)
    async def missing_handler(request, error):
        return JSONResponse(status_code=404, content={"detail": "Reference not found"})

    def submit(body: dict, key: str) -> dict:
        try:
            spec = JobSpec.from_dict(body)
        except TypeError as error:
            raise ValueError("Invalid JobSpec fields") from error
        if profiles is not None and spec.profile not in profiles:
            raise ValueError("Unknown worker profile")
        if profiles is not None:
            fingerprint = digest(profiles[spec.profile])
            if spec.profile_hash not in (None, fingerprint):
                raise Conflict("Worker profile changed")
            defaults = profiles[spec.profile].get(defaults_key(spec.kind), {})
            spec = replace(spec, spec=merge(defaults, spec.spec), profile_hash=fingerprint)
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
            # An inserted assistant turn must fit the history it continues
            # (fresh call ids); mini checks it again against the tools it offers.
            continuation = Continuation.from_dict(spec.continuation)
            prefix = load_prefix(point["native"]) if continuation.guidance == "assistant-turn" else ()
            prepare(continuation, prefix)
        return store.submit(spec, key)

    @app.post("/v1/jobs", status_code=202)
    def submit_job(body: dict, idempotency_key: str = Header(default="")):
        return submit(body, idempotency_key)

    @app.get("/v1/jobs")
    def list_jobs(state: str | None = None, limit: int = 100):
        return store.list_jobs(state, limit)

    @app.get("/v1/jobs/{job_id}")
    def get_job(job_id: str):
        return store.get(job_id)

    @app.get("/v1/jobs/{job_id}/result")
    def get_result(job_id: str):
        job = store.get(job_id)
        return {"job_id": job_id, "state": job["state"], "result": job["result"],
                "error": job["error"], "ready": job["state"] in {"succeeded", "failed", "cancelled"}}

    @app.get("/v1/jobs/{job_id}/attempts")
    def attempts(job_id: str):
        store.get(job_id)
        return store.attempts(job_id)

    def selected(job_id: str, attempt_id: str | None) -> str:
        job = store.get(job_id)
        identifier = attempt_id or job["active_attempt"]
        if identifier not in {attempt["id"] for attempt in store.attempts(job_id)}:
            raise KeyError(identifier)
        return identifier

    @app.get("/v1/jobs/{job_id}/events")
    def events(job_id: str, attempt_id: str | None = None, after: int = 0, limit: int = 1000):
        return store.events(selected(job_id, attempt_id), after, limit)

    @app.get("/v1/jobs/{job_id}/tools")
    def tools(job_id: str, attempt_id: str | None = None, after: int = 0, limit: int = 1000):
        return index.tools(selected(job_id, attempt_id), after, limit)

    @app.get("/v1/jobs/{job_id}/recovery-points")
    def recovery_points(job_id: str, attempt_id: str | None = None):
        return [{**point, "available": index.valid(point)}
                for point in index.points(selected(job_id, attempt_id))]

    @app.post("/v1/jobs/{job_id}/branch", status_code=202)
    def branch(job_id: str, body: dict, idempotency_key: str = Header(default="")):
        if not isinstance(body, dict) or set(body) - {"point_id", "continuation", "overrides", "budget"}:
            raise ValueError("A branch names point_id and continuation, and optionally overrides and budget")
        parent = JobSpec.from_dict(store.get(job_id)["request"])
        point = index.get_point(body["point_id"])
        if point["job_id"] != job_id or parent.kind != "rollout":
            raise ValueError("Branch point does not belong to this rollout")
        overrides = body.get("overrides", {})
        if not isinstance(overrides, dict) or set(overrides) - BRANCH_OVERRIDES:
            raise ValueError("A branch can change only %s; what the actor is given next is its continuation"
                             % ", ".join(sorted(BRANCH_OVERRIDES)))
        continuation = Continuation.from_dict(body.get("continuation"))
        budget = validate_branch_budget(body["budget"]) if "budget" in body else None
        if budget is not None and parent.spec.get("extra", {}).get("rollout_contract") is None:
            raise ValueError("A branch budget bounds a rollout contract; the parent run has none")
        request = replace(parent, spec={**parent.spec, **overrides}, parent_point=point["id"],
                          continuation=continuation.to_dict(), branch_budget=budget)
        return submit(request.validate(), idempotency_key)

    @app.post("/v1/jobs/{job_id}/cancel")
    def cancel(job_id: str):
        store.cancel_queued(job_id)
        return store.get(job_id)

    @app.post("/v1/jobs/{job_id}/stop")
    def stop(job_id: str):
        try:
            return store.request_stop(job_id)
        except KeyError:
            raise HTTPException(404, "Unknown job") from None

    return app

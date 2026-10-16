"""Independent HTTP service on the original rollout-driver port, 11001."""

import logging
import secrets
import threading
from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException
from fastapi.responses import JSONResponse

from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Conflict

LOG = logging.getLogger(__name__)
DEFAULT_PORT = 11001


def create_app(
    driver: Driver, token: str | None, *, poll_interval_s: float = 1, background: bool = True, miles=None
) -> FastAPI:
    if token == "":
        raise ValueError("Use a nonempty token or None for trusted local access")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    def poll():
        while not driver.stopping.is_set():
            driver.wakeup.clear()
            try:
                driver.tick()
            except Exception:
                LOG.exception("Driver polling failed; saved submission intents are retained")
            driver.wakeup.wait(poll_interval_s)

    @asynccontextmanager
    async def lifespan(_app):
        with driver.ledger.owner():
            driver.stopping.clear()
            thread = threading.Thread(target=poll, daemon=True, name="rl-driver-poll") if background else None
            if thread:
                thread.start()
            try:
                yield
            finally:
                driver.stopping.set()
                driver.wakeup.set()
                if thread:
                    thread.join(timeout=45)
                    if thread.is_alive():
                        raise RuntimeError(
                            "Driver polling did not stop; submission intents remain in the ledger"
                        )

    def authenticate(authorization: str = Header(default="")):
        if token is not None and not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Invalid driver token")

    app = FastAPI(
        title="Sprout RL driver", version="0.2.0", dependencies=[Depends(authenticate)], lifespan=lifespan
    )

    @app.exception_handler(Conflict)
    async def conflict(_request, error):
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    async def invalid(_request, error):
        return JSONResponse(
            status_code=400 if _request.url.path.startswith("/rollout-") else 422,
            content={"detail": str(error)},
        )

    @app.exception_handler(KeyError)
    async def missing(_request, error):
        return JSONResponse(
            status_code=404, content={"detail": "Group, sample or execution reference not found"}
        )

    @app.exception_handler(httpx.HTTPError)
    async def upstream(_request, error):
        return JSONResponse(
            status_code=502,
            content={"detail": "Run Store query failed; retry without resubmitting execution"},
        )

    @app.get("/health")
    def health():
        return {"role": "driver", "miles_configured": miles is not None}

    @app.post("/rollout-groups", status_code=202)
    def submit_miles(body: dict):
        if miles is None:
            raise HTTPException(503, "Configure the miles section before submitting")
        return miles.submit(body)

    @app.get("/rollout-groups/{group_id:path}")
    def get_miles(group_id: str):
        if miles is None:
            raise HTTPException(503, "Miles adapter is not configured")
        return miles.get(group_id)

    @app.delete("/rollout-groups/{group_id:path}")
    def release_miles(group_id: str):
        if miles is None:
            raise HTTPException(503, "Miles adapter is not configured")
        return miles.release(group_id)

    @app.get("/miles-executions/{group_id:path}")
    def miles_execution(group_id: str):
        if miles is None:
            raise HTTPException(503, "Miles adapter is not configured")
        return miles.execution(group_id)

    @app.post("/execution-groups", status_code=202)
    def submit(body: dict):
        view = driver.submit(body)
        return {key: view[key] for key in ("rollout_job_id", "status")}

    @app.get("/execution-groups/{group_id}")
    @app.get("/execution-groups/{group_id}/result")
    def get(group_id: str):
        return driver.get(group_id)

    @app.delete("/execution-groups/{group_id}")
    def release(group_id: str):
        acknowledgement = driver.release(group_id)
        return JSONResponse(
            acknowledgement, status_code=202 if acknowledgement["status"] == "cancelling" else 200
        )

    @app.post("/execution-groups/{group_id}/samples/{sample_id}/reconcile-submission")
    def reconcile_submission(group_id: str, sample_id: str, body: dict):
        if set(body) != {"phase", "job_id"}:
            raise ValueError("Reconciliation requires phase and job_id")
        from sprout.rl_driver.specs import identifier

        return driver.reconcile_submission(
            group_id, sample_id, body["phase"], identifier(body["job_id"], "job_id")
        )

    return app

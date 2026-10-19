"""The RL driver's HTTP API, on port ``DEFAULT_PORT`` unless the caller chooses another.

``create_app`` builds the app. While the app runs, it holds the driver's ledger and can poll
the driver in a background thread. ``docs/RL_DRIVER.md`` lists the routes and the error codes.
"""

import logging
import secrets
import threading
from collections.abc import AsyncIterator, Callable
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from typing import Any

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse

from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Conflict
from sprout.rl_driver.messages import MessageAdapter

LOG = logging.getLogger(__name__)
DEFAULT_PORT = 11001


def create_app(
    driver: Driver,
    token: str | None,
    *,
    poll_interval_s: float = 1,
    background: bool = True,
    miles: MessageAdapter | None = None,
) -> FastAPI:
    """The driver's FastAPI app. Each request must carry ``Bearer <token>``, unless ``token`` is None.

    The app holds the ledger (``Ledger.owner``) while it runs. With ``background``, a thread ticks the
    driver every ``poll_interval_s`` seconds. Without ``miles``, the Miles routes answer 503.
    """
    if token == "":
        raise ValueError("Use a nonempty token or None for trusted local access")
    if poll_interval_s <= 0:
        raise ValueError("poll_interval_s must be positive")

    def authenticate(authorization: str = Header(default="")) -> None:
        if token is not None and not secrets.compare_digest(authorization, "Bearer " + token):
            raise HTTPException(401, "Invalid driver token")

    app = FastAPI(
        title="Sprout RL driver",
        version="0.2.0",
        dependencies=[Depends(authenticate)],
        lifespan=_lifespan(driver=driver, poll_interval_s=poll_interval_s, background=background),
    )
    _add_error_handlers(app)

    # response_model=None: a return annotation would make FastAPI validate and reserialize each body.
    @app.get("/health", response_model=None)
    def health() -> dict[str, Any]:
        return {"role": "driver", "miles_configured": miles is not None}

    _add_miles_routes(app, miles=miles)
    _add_execution_routes(app, driver=driver)
    return app


def _poll(*, driver: Driver, poll_interval_s: float) -> None:
    """Tick ``driver`` every ``poll_interval_s`` seconds, or sooner on a wakeup, until it stops."""
    while not driver.stopping.is_set():
        driver.wakeup.clear()
        try:
            driver.tick()
        except Exception:
            LOG.exception("Driver polling failed; saved submission intents are retained")
        driver.wakeup.wait(poll_interval_s)


def _lifespan(
    *, driver: Driver, poll_interval_s: float, background: bool
) -> Callable[[FastAPI], AbstractAsyncContextManager[None]]:
    """The app's lifespan: it holds the driver's ledger and, with ``background``, runs ``_poll``."""

    @asynccontextmanager
    async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
        with driver.ledger.owner():
            driver.stopping.clear()
            thread: threading.Thread | None = None
            if background:
                thread = threading.Thread(
                    target=_poll,
                    kwargs={"driver": driver, "poll_interval_s": poll_interval_s},
                    daemon=True,
                    name="rl-driver-poll",
                )
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

    return lifespan


def _add_error_handlers(app: FastAPI) -> None:
    @app.exception_handler(Conflict)
    async def conflict(_request: Request, error: Conflict) -> JSONResponse:
        return JSONResponse(status_code=409, content={"detail": str(error)})

    @app.exception_handler(ValueError)
    @app.exception_handler(TypeError)
    async def invalid(_request: Request, error: Exception) -> JSONResponse:
        return JSONResponse(
            status_code=400 if _request.url.path.startswith("/rollout-") else 422,
            content={"detail": str(error)},
        )

    @app.exception_handler(KeyError)
    async def missing(_request: Request, error: KeyError) -> JSONResponse:
        return JSONResponse(
            status_code=404, content={"detail": "Group, sample or execution reference not found"}
        )

    @app.exception_handler(httpx.HTTPError)
    async def upstream(_request: Request, error: httpx.HTTPError) -> JSONResponse:
        return JSONResponse(
            status_code=502,
            content={"detail": "Run Store query failed; retry without resubmitting execution"},
        )


def _add_miles_routes(app: FastAPI, *, miles: MessageAdapter | None) -> None:
    """Add the routes of the Miles rollout groups. Each answers 503 if ``miles`` is None."""

    @app.post("/rollout-groups", status_code=202, response_model=None)
    def submit_miles(body: dict[str, Any]) -> dict[str, Any]:
        if miles is None:
            raise HTTPException(503, "Configure the miles section before submitting")
        return miles.submit(body)

    @app.get("/rollout-groups/{group_id:path}", response_model=None)
    def get_miles(group_id: str) -> dict[str, Any]:
        if miles is None:
            raise HTTPException(503, "Miles adapter is not configured")
        return miles.get(group_id)

    @app.delete("/rollout-groups/{group_id:path}", response_model=None)
    def release_miles(group_id: str) -> dict[str, Any]:
        if miles is None:
            raise HTTPException(503, "Miles adapter is not configured")
        return miles.release(group_id)

    @app.get("/miles-executions/{group_id:path}", response_model=None)
    def miles_execution(group_id: str) -> dict[str, Any]:
        if miles is None:
            raise HTTPException(503, "Miles adapter is not configured")
        return miles.execution(group_id)


def _add_execution_routes(app: FastAPI, *, driver: Driver) -> None:
    """Add the routes of the execution groups, which ``driver`` runs."""

    @app.post("/execution-groups", status_code=202, response_model=None)
    def submit(body: dict[str, Any]) -> dict[str, Any]:
        view = driver.submit(body)
        return {key: view[key] for key in ("rollout_job_id", "status")}

    @app.get("/execution-groups/{group_id}", response_model=None)
    @app.get("/execution-groups/{group_id}/result", response_model=None)
    def get(group_id: str) -> dict[str, Any]:
        return driver.get(group_id)

    @app.delete("/execution-groups/{group_id}")
    def release(group_id: str) -> JSONResponse:
        acknowledgement = driver.release(group_id)
        return JSONResponse(
            acknowledgement, status_code=202 if acknowledgement["status"] == "cancelling" else 200
        )

    @app.post("/execution-groups/{group_id}/samples/{sample_id}/reconcile-submission", response_model=None)
    def reconcile_submission(group_id: str, sample_id: str, body: dict[str, Any]) -> dict[str, Any]:
        if set(body) != {"phase", "job_id"}:
            raise ValueError("Reconciliation requires phase and job_id")
        from sprout.search.group import identifier

        return driver.reconcile_submission(
            group_id, sample_id, body["phase"], identifier(body["job_id"], "job_id")
        )

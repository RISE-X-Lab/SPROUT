"""A stand-in AgentENV for ``tests.support.http.serving``: fixed answers, and a record of each request."""

from __future__ import annotations

from typing import Any


class AgentEnv:
    """An ASGI app that answers each request of a method with a fixed status and an empty body.

    ``delete`` and ``get`` are the statuses of the DELETE and GET requests."""

    def __init__(self, *, delete: int, get: int = 404) -> None:
        self.statuses = {"DELETE": delete, "GET": get}
        #: The method, the path and the X-API-Key header of each request, in arrival order.
        self.requests: list[tuple[str, str, str]] = []

    async def __call__(self, scope: dict[str, Any], receive: Any, send: Any) -> None:
        if scope["type"] == "lifespan":
            await receive()
            await send({"type": "lifespan.startup.complete"})
            await receive()
            await send({"type": "lifespan.shutdown.complete"})
            return
        key = dict(scope["headers"]).get(b"x-api-key", b"").decode()
        self.requests.append((scope["method"], scope["path"], key))
        await send({"type": "http.response.start", "status": self.statuses[scope["method"]], "headers": []})
        await send({"type": "http.response.body", "body": b""})

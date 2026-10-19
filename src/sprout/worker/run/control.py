"""The out-of-band stop of one rollout, which any thread can request.

A safety failure stops the rollout through :class:`RunControl`, so that the stop does not
depend on the model's cooperation.
"""

import asyncio
import threading
from collections.abc import Callable
from typing import Any

import httpx


class RunAborted(RuntimeError):
    """A rollout operation ended because the rollout was stopped. The message is the reason."""


#: Stop reasons that end a rollout with a workspace the verifier still grades:
#: a spent wall-time or turn budget, or a conversation past the model's context.
GRADABLE_STOP_REASONS = frozenset({"timeout", "max_turns_reached", "context_exhausted"})


class RunControl:
    """The stop request of one rollout. Any thread can request the stop and observe it.

    Only the first :meth:`request_stop` counts. A subscribed callback runs once: on the thread
    that requests the stop, or at once in :meth:`subscribe` if the stop came first.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._reason: str | None = None
        self._stop_reason: str | None = None
        self._callbacks: list[Callable[[], object]] = []

    @property
    def reason(self) -> str | None:
        """Why the rollout was stopped, or None while it is not stopped."""
        with self._lock:
            return self._reason

    @property
    def stop_reason(self) -> str | None:
        """The ``stop_reason`` of the stop request, such as ``timeout``, or None."""
        with self._lock:
            return self._stop_reason

    def request_stop(self, reason: str, *, stop_reason: str | None = None) -> None:
        """Stop the rollout for ``reason``, and run each subscribed callback. A later request does nothing."""
        with self._lock:
            if self._reason is not None:
                return
            self._reason = reason
            self._stop_reason = stop_reason
            callbacks = list(self._callbacks)
        for callback in callbacks:
            try:
                callback()
            except RuntimeError:
                pass  # a driver that has already closed its loop is stopped

    def subscribe(self, callback: Callable[[], object]) -> Callable[[], None]:
        """Run ``callback`` when the rollout stops, and return the function that unsubscribes it."""
        with self._lock:
            self._callbacks.append(callback)
            already_stopped = self._reason is not None
        if already_stopped:
            callback()

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def raise_if_stopped(self) -> None:
        """Raise :class:`RunAborted` with the reason if the rollout was stopped."""
        if self.reason is not None:
            raise RunAborted(self.reason)


def post(url: str, *, timeout_s: float, control: RunControl | None = None, **kwargs: Any) -> httpx.Response:
    """POST to ``url`` with the ``httpx`` request arguments ``kwargs``, and return the response.

    The request fails after ``timeout_s`` seconds. A stop on ``control`` ends it at once with
    :class:`RunAborted`, also while the server sends nothing. The calling thread must not run an
    event loop."""

    async def request() -> httpx.Response:
        if control:
            control.raise_if_stopped()
        loop = asyncio.get_running_loop()
        owner = asyncio.current_task()
        assert owner is not None, "asyncio.run runs this coroutine as a task"
        unsubscribe = (
            control.subscribe(lambda: loop.call_soon_threadsafe(owner.cancel)) if control else lambda: None
        )
        try:
            async with httpx.AsyncClient(timeout=timeout_s) as client:
                return await asyncio.wait_for(client.post(url, **kwargs), timeout_s)
        except asyncio.CancelledError:
            if control and control.reason:
                raise RunAborted(control.reason) from None
            raise
        finally:
            unsubscribe()

    return asyncio.run(request())

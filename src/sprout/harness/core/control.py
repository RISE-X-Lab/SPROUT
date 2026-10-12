"""Thread-safe, out-of-band termination for a single rollout."""

import asyncio
import threading

import httpx


class RunAborted(RuntimeError):
    pass


#: Stop reasons that end a rollout with a workspace the verifier still grades:
#: a spent wall-time or turn budget, or a conversation past the model's context.
GRADABLE_STOP_REASONS = frozenset({"timeout", "max_turns_reached", "context_exhausted"})


class RunControl:
    """A safety failure must reach the driver, not rely on model cooperation."""

    def __init__(self):
        self._lock = threading.Lock()
        self._reason = None
        self._stop_reason = None
        self._callbacks = []

    @property
    def reason(self):
        with self._lock:
            return self._reason

    @property
    def stop_reason(self):
        with self._lock:
            return self._stop_reason

    def request_stop(self, reason, *, stop_reason=None):
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

    def subscribe(self, callback):
        with self._lock:
            self._callbacks.append(callback)
            already_stopped = self._reason is not None
        if already_stopped:
            callback()

        def unsubscribe():
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)
        return unsubscribe

    def raise_if_stopped(self):
        if self.reason is not None:
            raise RunAborted(self.reason)


def post(url: str, *, timeout_s: float, control: RunControl | None = None, **kwargs) -> httpx.Response:
    async def request():
        if control:
            control.raise_if_stopped()
        loop = asyncio.get_running_loop()
        owner = asyncio.current_task()
        unsubscribe = (control.subscribe(lambda: loop.call_soon_threadsafe(owner.cancel))
                       if control else lambda: None)
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

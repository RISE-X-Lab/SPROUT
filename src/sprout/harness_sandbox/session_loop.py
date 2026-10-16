"""One private event loop that synchronous callers on any thread drive one at a
time, and the policy for an operation interrupted while it ran: cancel it with a
grace period, and quarantine the loop until it settles."""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable


class SessionLoop:
    """The event loop a session drives all of its sandbox I/O through."""

    def __init__(self, warn: Callable[[str], None]):
        self._warn = warn
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        self.unsettled: asyncio.Future | None = None
        self.cancel_grace_seconds = 5.0

    def get(self) -> asyncio.AbstractEventLoop:
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def drive(self, coroutine):
        """One owner for the session loop, even across checkpoint workers."""
        with self._lock:
            unsettled = self.unsettled
            if unsettled is not None and not unsettled.done():
                if asyncio.isfuture(coroutine):
                    coroutine.cancel()
                else:
                    coroutine.close()
                raise RuntimeError("Previous sandbox operation has not settled; session is quarantined")
            if unsettled is not None:
                try:
                    unsettled.result()
                except BaseException:
                    pass
            self.unsettled = None
            loop = self.get()
            task = asyncio.ensure_future(coroutine, loop=loop)
            try:
                return loop.run_until_complete(task)
            except BaseException:
                if not task.done():
                    task.cancel()
                    try:
                        loop.run_until_complete(asyncio.wait({task}, timeout=self.cancel_grace_seconds))
                    except BaseException as cleanup_error:
                        self._warn("Interrupted operation cleanup failed: %s" % cleanup_error)
                if task.done():
                    try:
                        task.result()
                    except BaseException:
                        pass
                else:
                    self.unsettled = task
                    self._warn("Interrupted sandbox operation did not settle; quarantining session")
                raise

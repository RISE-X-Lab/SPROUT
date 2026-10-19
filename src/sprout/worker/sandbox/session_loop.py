"""A private event loop that synchronous callers on any thread drive, one call at a time.

An operation that is interrupted while it runs is cancelled, with a grace period to finish. If it
does not settle in that period, the loop is quarantined: each later call raises until it settles.
"""

from __future__ import annotations

import asyncio
import threading
from collections.abc import Callable, Coroutine
from typing import Any, TypeVar

_T = TypeVar("_T")


class SessionLoop:
    """The event loop that a session drives all of its sandbox I/O through."""

    def __init__(self, warn: Callable[[str], None]) -> None:
        self._warn = warn
        self._loop: asyncio.AbstractEventLoop | None = None
        self._lock = threading.RLock()
        #: The interrupted operation that did not settle in its grace period, or None.
        #: Its result type is the type of whatever operation it was.
        self.unsettled: asyncio.Future[Any] | None = None
        #: Seconds that an interrupted operation gets to finish after its cancellation.
        self.cancel_grace_seconds = 5.0

    def get(self) -> asyncio.AbstractEventLoop:
        """The loop, new if there was none or the last one was closed."""
        if self._loop is None or self._loop.is_closed():
            self._loop = asyncio.new_event_loop()
        return self._loop

    def drive(self, coroutine: Coroutine[Any, Any, _T] | asyncio.Future[_T]) -> _T:
        """Run ``coroutine`` on the loop, and return its result. Callers on other threads wait for their turn.

        An interruption, such as ``KeyboardInterrupt``, cancels the operation and is raised again. While an
        earlier interrupted operation has not settled, raises ``RuntimeError``, and ``coroutine`` never runs.
        """
        with self._lock:
            unsettled = self.unsettled
            if unsettled is not None and not unsettled.done():
                if asyncio.isfuture(coroutine):
                    coroutine.cancel()
                else:
                    coroutine.close()
                raise RuntimeError("Previous sandbox operation has not settled; session is quarantined")
            if unsettled is not None:
                # Read the settled result, so that asyncio does not log a never-retrieved exception.
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

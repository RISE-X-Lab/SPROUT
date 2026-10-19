"""Wait for a condition with a deadline, instead of a sleep of a fixed length."""

from __future__ import annotations

import time
from collections.abc import Callable


def wait_until(
    condition: Callable[[], object], *, timeout_s: float = 15.0, what: str = "the condition"
) -> None:
    """Return once ``condition()`` is true. Raise ``AssertionError``, naming ``what``, after ``timeout_s``.

    It calls ``condition`` about every 20 ms until the condition holds or the deadline passes."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "%s did not hold within %.1f s" % (what, timeout_s)
        time.sleep(0.02)

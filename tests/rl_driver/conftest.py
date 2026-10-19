"""The fixtures of the RL driver tests: a fake Run Store and a clock that the test sets."""

from __future__ import annotations

import time

import pytest

from tests.rl_driver.fake_runstore import FakeRunStore

#: The Unix time at which the ``clock`` fixture starts.
NOW = 1_800_000_000.0


class Clock:
    """``time.time`` for one test: it returns ``now`` until the test sets another."""

    def __init__(self, now: float) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def fake_run_store() -> FakeRunStore:
    """A new, empty ``FakeRunStore``."""
    return FakeRunStore()


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """Freeze ``time.time`` at ``NOW`` for the test. The test moves the time by setting ``clock.now``."""
    clock = Clock(NOW)
    monkeypatch.setattr(time, "time", clock)
    return clock

"""Fixtures of the sandbox tests: a stub pool that each session builds, and a scripted AgentENV API."""

from __future__ import annotations

from collections.abc import Callable, Iterator

import pytest

from tests.worker.sandbox.agentenv import AgentEnvStub, agentenv
from tests.worker.sandbox.pools import StubPool


@pytest.fixture
def use_pool(monkeypatch: pytest.MonkeyPatch) -> Callable[[StubPool], StubPool]:
    """Make each ``SandboxSession`` build the given pool, whatever its backend section names."""

    def install(pool: StubPool) -> StubPool:
        monkeypatch.setattr("sprout.worker.sandbox.session.build_pool", lambda backend: pool)
        return pool

    return install


@pytest.fixture
def agentenv_stub() -> Iterator[AgentEnvStub]:
    """A scripted AgentENV API on a local port (``tests/worker/sandbox/agentenv.py``)."""
    with agentenv() as stub:
        yield stub

"""Fixtures of the tests of ``sprout.search``: a fake Run Store, a group store in ``tmp_path``, a clock."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

import pytest

from sprout.eval.groups import FileGroups
from tests.search.documents import Clock
from tests.search.fake_runstore import FakeRunStore


@pytest.fixture
def fake_run_store() -> Iterator[FakeRunStore]:
    """A ``FakeRunStore`` with no job. After the test, it closes its client and fails on a defect."""
    fake = FakeRunStore()
    yield fake
    fake.client.close()
    assert fake.defects == [], "the fake Run Store raised while it answered"


@pytest.fixture
def groups(tmp_path: Path) -> FileGroups:
    """A real ``GroupStore``: one JSON file for each group, in ``tmp_path``."""
    return FileGroups(tmp_path / "groups")


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> Clock:
    """``time.time`` for the test: 1000.0 until the test sets ``clock.now``."""
    fixed = Clock(1000.0)
    monkeypatch.setattr(time, "time", fixed)
    return fixed

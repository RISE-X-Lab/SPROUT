"""Each test gives the process back the umask and the environment that it found."""

from __future__ import annotations

import os

from tests.support.process import kept_process_state


def current_umask() -> int:
    umask = os.umask(0o022)
    os.umask(umask)
    return umask


def test_a_umask_set_in_the_block_is_undone_when_the_block_exits():
    """A leaked 077 made each base that a later test extracted unreadable to an image's non-root user."""
    os.umask(0o022)

    with kept_process_state():
        os.umask(0o077)

    assert current_umask() == 0o022


def test_variables_set_changed_or_removed_in_the_block_are_undone_when_the_block_exits(monkeypatch):
    monkeypatch.setenv("SPROUT_SUPPORT_CHANGED", "before")
    monkeypatch.setenv("SPROUT_SUPPORT_REMOVED", "before")
    monkeypatch.delenv("SPROUT_SUPPORT_ADDED", raising=False)
    before = dict(os.environ)

    with kept_process_state():
        os.environ["SPROUT_SUPPORT_CHANGED"] = "inside"
        os.environ["SPROUT_SUPPORT_ADDED"] = "inside"
        del os.environ["SPROUT_SUPPORT_REMOVED"]

    assert dict(os.environ) == before


def test_every_test_runs_in_a_kept_process_state(request):
    assert "restore_process_state" in request.fixturenames

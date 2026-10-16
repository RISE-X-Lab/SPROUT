"""`harness run`: what is refused before a sandbox exists.

Each refusal here is a pair of flags that would otherwise be accepted and do
something other than what was typed. Nothing is created: the checks run before
the RunSpec is built.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

import sprout.worker.cli as cli

RUN = ["run", "--sandbox-image", "docker.io/library/python:3.11-slim", "--cwd", "/tmp", "fix it"]


PODMAN = ["--backend", "podman", "--runtime-bin", "runtime/sprout-runtime", "--store", "/mnt/store"]


def test_podman_without_a_runtime_binary_is_refused():
    with pytest.raises(SystemExit, match="needs --runtime-bin"):
        cli.main([*RUN, "--backend", "podman", "--store", "/mnt/store"])


def test_podman_is_the_default_and_takes_the_default_store(monkeypatch, tmp_path):
    """No --backend is podman, and no --store is the default store
    ($SPROUT_PODMAN_STORE, else ~/sprout-btrfs); nothing here is created, the
    orchestrator is a stand-in that records what it was handed."""
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(tmp_path / "store"))
    handed = []

    class Recording:
        def __init__(self, **kwargs):
            pass

        def run(self, spec):
            handed.append(spec)
            return SimpleNamespace(run_id="r", journal_path="j", status="completed", native_session_id=None,
                                   usage=None, checkpoints=0, error=None, final_text="", ok=True)

    monkeypatch.setattr(cli, "Runner", Recording)
    assert cli.main([*RUN, "--runtime-bin", "runtime/sprout-runtime"]) == 0
    assert handed[0].backend == {"backend": "podman", "podman": {"store": str(tmp_path / "store"),
                                                                 "runtime_bin": "runtime/sprout-runtime"}}


def test_podman_with_a_memory_snapshot_is_refused():
    """Otherwise a no-op: the pool declines, the quiet session swallows the
    warning, the journal says `snapshot.memory captured=false`, the run exits 0
    and nothing reaches the person who typed the flag."""
    with pytest.raises(SystemExit, match="podman snapshots are disk-only"):
        cli.main([*RUN, *PODMAN, "--final-memory-snapshot"])

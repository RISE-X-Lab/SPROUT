"""``ops/bench_snapshots.py``: the snapshot timings, and the list of snapshots that the caller must delete."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from ops import bench_snapshots
from ops.bench_snapshots import bench


@dataclass(frozen=True)
class Result:
    is_error: bool
    exit_code: int | None
    output: str


@dataclass(frozen=True)
class Snapshot:
    id: str


class FakeSandbox:
    """A sandbox whose shell runs nothing. A command that contains ``fail_on`` fails with exit 1."""

    def __init__(self, fail_on: str | None = None) -> None:
        self.fail_on = fail_on

    async def call(self, tool, *, command, timeout):
        failed = self.fail_on is not None and self.fail_on in command
        return Result(
            is_error=False, exit_code=1 if failed else 0, output="No space left on device" if failed else ""
        )


class FakePool:
    """A pool whose snapshots are numbered ``snap-1``, ``snap-2`` and so on."""

    def __init__(self) -> None:
        self.taken = 0

    async def snapshot(self, sandbox, *, disk_only):
        self.taken += 1
        return Snapshot("snap-%d" % self.taken)


def steps(samples):
    """Each sample's step and whether its snapshot was disk-only, in order."""
    return [(sample["step"], sample["disk_only"]) for sample in samples]


def test_bench_times_each_step_repeat_times_and_drops_the_warm_up():
    samples = asyncio.run(bench(FakePool(), FakeSandbox(), 2, False, []))

    assert steps(samples) == [
        ("nothing written", True),
        ("nothing written", True),
        ("4 KiB written", True),
        ("4 KiB written", True),
        ("100 MiB written", True),
        ("100 MiB written", True),
    ]


def test_a_full_bench_times_memory_snapshots_on_a_fresh_guest_and_after_the_writes():
    samples = asyncio.run(bench(FakePool(), FakeSandbox(), 1, True, []))

    assert steps(samples) == [
        ("full (memory too), fresh guest", False),
        ("nothing written", True),
        ("4 KiB written", True),
        ("100 MiB written", True),
        ("full (memory too), after 1 GiB written", False),
    ]


def test_bench_lists_each_snapshot_it_took_before_a_step_failed():
    """The caller deletes every snapshot in the list, also after the failure."""
    taken = []

    with pytest.raises(RuntimeError, match="No space left on device"):
        asyncio.run(bench(FakePool(), FakeSandbox(fail_on="blob-"), 2, False, taken))

    assert taken == ["snap-1", "snap-2", "snap-3", "snap-4", "snap-5"]


def test_the_podman_backend_needs_a_store(capsys):
    with pytest.raises(SystemExit) as stop:
        bench_snapshots.main(
            ["--backend", "podman", "--image", "registry.example/a:1", "--runtime-bin", "/path/to/runtime"]
        )

    assert (stop.value.code, capsys.readouterr().err.splitlines()[-1].split(": ", 1)[1]) == (
        2,
        "error: --backend podman needs --store",
    )

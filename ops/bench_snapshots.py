#!/usr/bin/env python3
"""Time one snapshot on either backend, measured the same way on both.

The script measures the cost of a snapshot. It starts one sandbox from the image
and times ``snapshot()``, the call that the checkpointer makes once per model turn, from the
client. It times the call after three kinds of step: nothing written since the last
snapshot, a 4 KiB file written, and 100 MiB of new, incompressible data written. On
``microvm`` it also times full snapshots (disk plus memory), which ``--final-memory-snapshot``
takes once per rollout. Podman snapshots are disk-only. Before it exits, the script deletes
the snapshots that it took and then the sandbox.

    python ops/bench_snapshots.py --backend podman --store ~/sprout-btrfs \\
        --runtime-bin runtime/sprout-runtime --image <image>
    python ops/bench_snapshots.py --backend microvm \\
        --runtime-bin runtime/sprout-runtime --image <image>  # AENV_SERVER_URL, AENV_API_KEY

The figures are for an idle host. Under the load of a batch, the same call waits for the
disk writes of the other sandboxes.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from sprout.worker.run.reap import AgentEnvClient
from sprout.worker.sandbox.backends import resolve_microvm_endpoint
from sprout.worker.sandbox.templates import TemplateBuilder
from sprout_sandbox import MicroVMPool, PodmanPool

WORKDIR = "/var/tmp/sprout-bench"
STEPS = (
    ("nothing written", None),
    ("4 KiB written", "head -c 4096 /dev/urandom > %s/edit-{i}" % WORKDIR),
    ("100 MiB written", "head -c 104857600 /dev/urandom > %s/blob-{i}" % WORKDIR),
)
RESOURCES = {"cpu": 2, "memory_mb": 8192}


async def shell(sandbox, command: str) -> None:
    result = await sandbox.call("shell", command=command, timeout=120)
    if result.is_error or result.exit_code not in (None, 0):
        raise RuntimeError(f"{command!r} failed in the sandbox: {result.output[:500]}")


async def ready(sandbox, seconds: float = 120) -> None:
    """Wait until the runtime in ``sandbox`` answers, for at most ``seconds``.

    A new sandbox can still be booting. After the deadline, the last error propagates.
    """
    deadline = time.monotonic() + seconds
    while True:
        try:
            return await shell(sandbox, "mkdir -p %s" % WORKDIR)
        except Exception:  # noqa: BLE001 - not up yet
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(1)


async def bench(pool, sandbox, repeat: int, full: bool, taken: list[str]) -> list[dict]:
    """The timed samples, without the warm-up.

    Each snapshot id goes into ``taken`` when the snapshot exists. After a failure,
    ``taken`` still lists each snapshot that the caller must delete.
    """
    samples = []

    async def timed(step: str, disk_only: bool = True) -> None:
        started = time.perf_counter()
        snapshot = await pool.snapshot(sandbox, disk_only=disk_only)
        samples.append({"step": step, "disk_only": disk_only, "seconds": time.perf_counter() - started})
        taken.append(snapshot.id)

    async def full_snapshots(when: str) -> None:
        for _ in range(min(repeat, 3)):
            await shell(sandbox, STEPS[1][1].format(i=f"full-{len(taken)}"))
            await timed(f"full (memory too), {when}", disk_only=False)

    await timed("warm-up")
    # A full snapshot stores the memory that the guest has touched. So it is timed on a
    # fresh guest and again after the 100 MiB steps have filled the page cache.
    if full:
        await full_snapshots("fresh guest")
    for step, command in STEPS:
        for i in range(repeat):
            if command:
                await shell(sandbox, command.format(i=i))
            await timed(step)
    if full:
        await full_snapshots("after 1 GiB written")
    return [s for s in samples if s["step"] != "warm-up"]


async def run(args) -> list[dict]:
    if args.backend == "podman":
        pool = PodmanPool(args.store, args.runtime_bin)
        sandbox = await pool.spawn_from_image(args.image, resources=RESOURCES)
        delete = pool.delete_snapshot
    else:
        server_url, api_key = resolve_microvm_endpoint({})
        template = TemplateBuilder(
            server_url=server_url, api_key=api_key, runtime_bin=Path(args.runtime_bin)
        ).template_for(args.image, RESOURCES)
        pool = MicroVMPool(server_url, api_key=api_key, sandbox_ttl=1800)
        sandbox = await pool.spawn(template)
        client = AgentEnvClient(server_url, api_key)

        async def delete(snapshot_id: str) -> None:
            if not client.delete_snapshot(snapshot_id):
                raise RuntimeError(f"could not delete snapshot {snapshot_id}")

    taken: list[str] = []
    try:
        await ready(sandbox)
        samples = await bench(pool, sandbox, args.repeat, args.backend == "microvm", taken)
    finally:
        for snapshot_id in taken:
            await delete(snapshot_id)
        await pool.destroy(sandbox)
        await pool.close()
    return samples


def table(samples: list[dict], backend: str) -> str:
    rows = [f"| {backend} | step | n | median | min | max |", "|---|---|---|---|---|---|"]
    for step in dict.fromkeys(s["step"] for s in samples):
        x = [s["seconds"] * 1000 for s in samples if s["step"] == step]
        rows.append(
            f"| | {step} | {len(x)} | {statistics.median(x):.0f} ms | {min(x):.0f} ms | {max(x):.0f} ms |"
        )
    return "\n".join(rows)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--backend", choices=("microvm", "podman"), required=True)
    parser.add_argument("--image", required=True, help="an OCI image reference, e.g. a task's docker_image")
    parser.add_argument("--runtime-bin", required=True)
    parser.add_argument("--store", help="podman: the btrfs store")
    parser.add_argument("--repeat", type=int, default=10, help="snapshots per kind of step")
    parser.add_argument("--output", type=Path, help="also write every sample here, as JSON")
    args = parser.parse_args(argv)
    if args.backend == "podman" and not args.store:
        parser.error("--backend podman needs --store")
    samples = asyncio.run(run(args))
    if args.output:
        args.output.write_text(
            json.dumps({"backend": args.backend, "image": args.image, "samples": samples}, indent=1)
        )
    print(table(samples, args.backend))
    return 0


if __name__ == "__main__":
    sys.exit(main())

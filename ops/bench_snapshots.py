#!/usr/bin/env python3
"""How long one snapshot takes, on either backend, measured the same way.

The number in the README's snapshot table. One sandbox starts from the image
and the script times ``snapshot()`` -- the call the checkpointer makes once per
model turn -- from the client, after three kinds of step: nothing written since
the last snapshot, a 4 KiB file written, and 100 MiB of new, incompressible
data written. On ``microvm`` it also times full snapshots (disk plus memory),
which ``--final-memory-snapshot`` takes once per rollout; podman snapshots are
disk-only. Everything the script creates (the snapshots, then the sandbox) is
deleted before it exits.

    python ops/bench_snapshots.py --backend podman --store ~/sprout-btrfs \\
        --runtime-bin runtime/sprout-runtime --image <image>
    python ops/bench_snapshots.py --backend microvm \\
        --runtime-bin runtime/sprout-runtime --image <image>  # AENV_SERVER_URL, AENV_API_KEY

An idle-host figure: under a batch's load the same call queues behind the
other sandboxes' disk writes, which is what ``inspect`` timings of a real run
show.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import statistics
import sys
import time
from pathlib import Path

from sprout.harness_orchestrator.reap import AgentEnvClient
from sprout.harness_sandbox.backends import resolve_microvm_endpoint
from sprout.harness_sandbox.templates import TemplateBuilder
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
    """Until the runtime answers; a freshly started sandbox may still be booting."""
    deadline = time.monotonic() + seconds
    while True:
        try:
            return await shell(sandbox, "mkdir -p %s" % WORKDIR)
        except Exception:  # noqa: BLE001 - not up yet
            if time.monotonic() > deadline:
                raise
            await asyncio.sleep(1)


async def bench(pool, sandbox, repeat: int, full: bool, taken: list[str]) -> list[dict]:
    """The samples; every snapshot id goes into ``taken`` as it is made, so a
    failure halfway still leaves the caller a complete list to delete."""
    samples = []

    async def timed(step: str, disk_only: bool = True) -> None:
        started = time.perf_counter()
        snapshot = await pool.snapshot(sandbox, disk_only=disk_only)
        samples.append({"step": step, "disk_only": disk_only, "seconds": time.perf_counter() - started})
        taken.append(snapshot.id)

    async def full_snapshots(when: str) -> None:
        for i in range(min(repeat, 3)):
            await shell(sandbox, STEPS[1][1].format(i=f"full-{len(taken)}"))
            await timed(f"full (memory too), {when}", disk_only=False)

    await timed("warm-up")
    # What a full snapshot stores grows with the memory the guest has touched,
    # so it is timed on a fresh guest and again once the 100 MiB steps have
    # filled the page cache.
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

#!/usr/bin/env python3
"""Pull every task image into the backend's cache BEFORE an eval runs.

Why this exists: a full launch once lost 49 of its first 125 instances to
Docker Hub's anonymous rate limit -- eight workers pulling uncached images in
parallel burned the pull budget in minutes, and each failure cost an agent slot
that then had nothing to grade. Pulling is a separate resource from running, so
it gets a separate, gentler phase: convert everything first, then let the eval
hit only the local cache.

Two backends, two caches:

* ``--backend podman`` (default; ``--store`` names the store, else
  ``$SPROUT_PODMAN_STORE`` or ``~/sprout-btrfs``): ``PodmanPool.prepare`` pulls
  the image into rootless podman's storage and extracts its rootfs into the
  store's ``bases/`` as a read-only subvolume -- what the first spawn of an image
  does otherwise inside the first trial of a task, at seconds for a slim image
  and about half a minute for a 3 GB one. No container starts.
* ``--backend microvm``: AgentENV has no prefetch endpoint, and conversion
  happens on the cold-start path. So "pulling" an image means cold-starting a
  throwaway microVM from it and destroying it immediately. A cached image
  cold-starts in seconds, which makes this naturally idempotent.

Images come from Harbor task directories (TB2.1, HARD-51, DeepSWE): every
``task.toml`` under the given roots contributes its ``[environment].docker_image``.

    python ops/prepull_images.py --tasks-dir ~/sprout-data/datasets/deep-swe/tasks --workers 4
    python ops/prepull_images.py --backend microvm --tasks-dir <tb21> --tasks-dir <hard51> --workers 2

State lands in ``--state`` (one line per image, ok or error, with the backend),
so a rate-limit failure late in the run costs a retry of THAT image, not the
batch. Records of another backend in the same file are ignored.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
import tomllib
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from sprout.contract.backend import default_podman_store
from sprout.worker.sandbox.session import SandboxSession
from sprout_sandbox import PodmanPool

BACKENDS = ("microvm", "podman")


def task_images(roots: list[Path]) -> list[str]:
    images = set()
    for root in roots:
        for path in root.expanduser().rglob("task.toml"):
            image = tomllib.loads(path.read_text()).get("environment", {}).get("docker_image")
            if image:
                images.add(image)
    return sorted(images)


def done_images(state: Path, backend: str) -> set:
    """Images the state file records as pulled for ``backend`` (older records name none: microvm)."""
    done = set()
    if state.exists():
        for line in state.read_text().splitlines():
            record = json.loads(line)
            if record.get("ok") and record.get("backend", "microvm") == backend:
                done.add(record["image"])
    return done


def pull_one_microvm(image: str, runtime_bin: str) -> dict:
    """Cold-start and destroy. The boot is the price of the conversion."""
    started = time.time()
    session = SandboxSession(
        quiet=True,
        backend={
            "backend": "microvm",
            "microvm": {"from_image": True, "runtime_bin": runtime_bin},
        },
    )
    try:
        if session.create(image):
            return {
                "image": image,
                "backend": "microvm",
                "ok": True,
                "seconds": round(time.time() - started, 1),
            }
        return {
            "image": image,
            "backend": "microvm",
            "ok": False,
            "error": (session.create_error or "unknown")[:300],
        }
    finally:
        try:
            session.destroy()
        except Exception:
            pass


def pull_one_podman(image: str, runtime_bin: str, store: str) -> dict:
    """Pull and extract the base into the store; no container is started."""
    started = time.time()

    async def prepare() -> Path:
        pool = PodmanPool(store, runtime_bin)
        try:
            return await pool.prepare(image)
        finally:
            await pool.close()

    try:
        base = asyncio.run(prepare())
    except Exception as error:  # noqa: BLE001 - one image's failure is one record
        return {"image": image, "backend": "podman", "ok": False, "error": str(error)[:300]}
    return {
        "image": image,
        "backend": "podman",
        "ok": True,
        "seconds": round(time.time() - started, 1),
        "base": str(base),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--tasks-dir",
        type=Path,
        action="append",
        required=True,
        help="a directory of Harbor tasks; repeat for several datasets",
    )
    parser.add_argument("--state", type=Path, default=Path("runs/prepull/prepull.jsonl"))
    # 2 on purpose: enough to overlap conversion (CPU) with pulling (network)
    # without recreating the burst that got the anonymous run throttled.
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--runtime-bin", default="runtime/sprout-runtime")
    parser.add_argument(
        "--backend",
        choices=BACKENDS,
        default="podman",
        help="whose cache to fill: rootless podman's plus the btrfs store's bases "
        "(podman, the default) or AgentENV's (microvm)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="podman: the btrfs store (default $SPROUT_PODMAN_STORE, else ~/sprout-btrfs)",
    )
    args = parser.parse_args()
    if args.backend == "podman" and not args.store:
        args.store = Path(default_podman_store())

    images = task_images(args.tasks_dir)
    done = done_images(args.state, args.backend)
    todo = [i for i in images if i not in done]
    print(
        "images %d total, %d already pulled for %s, %d to pull"
        % (len(images), len(done), args.backend, len(todo)),
        flush=True,
    )
    args.state.parent.mkdir(parents=True, exist_ok=True)

    runtime_bin = str(Path(args.runtime_bin).resolve())
    if args.backend == "podman":
        store = str(args.store.expanduser().resolve())

        def pull_one(image: str) -> dict:
            return pull_one_podman(image, runtime_bin, store)
    else:

        def pull_one(image: str) -> dict:
            return pull_one_microvm(image, runtime_bin)

    failures = []
    with (
        args.state.open("a", encoding="utf-8") as state,
        ThreadPoolExecutor(max_workers=args.workers) as pool,
    ):
        futures = {pool.submit(pull_one, image): image for image in todo}
        for n, future in enumerate(as_completed(futures), 1):
            record = future.result()
            state.write(json.dumps(record) + "\n")
            state.flush()
            if record["ok"]:
                print(
                    "[%d/%d] ok    %-70s %ss" % (n, len(todo), record["image"], record["seconds"]), flush=True
                )
            else:
                failures.append(record)
                print(
                    "[%d/%d] FAIL  %-70s %s" % (n, len(todo), record["image"], record["error"][:120]),
                    flush=True,
                )

    print("\ndone: %d ok, %d failed" % (len(todo) - len(failures), len(failures)), flush=True)
    if failures:
        print("re-run this script to retry the failures (state is per-image).")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

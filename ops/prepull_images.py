#!/usr/bin/env python3
"""Pull every task image into the backend's cache before an evaluation runs.

The evaluation then reads only the local cache. Pulling is a separate phase with few
workers, because many parallel pulls exceed a registry's anonymous rate limit.

* ``--backend podman`` (the default): ``PodmanPool.prepare`` pulls each image into rootless
  podman's storage and extracts its rootfs into the store's ``bases/``. No container starts.
  ``--store`` names the store, else ``$SPROUT_PODMAN_STORE``, else ``~/sprout-btrfs``.
* ``--backend microvm``: AgentENV has no prefetch endpoint and converts an image when it
  cold-starts a microVM from it. So the script cold-starts a microVM and destroys it.

Each ``task.toml`` under each ``--tasks-dir`` gives its ``[environment].docker_image``.
Each pull appends one JSON line to ``--state``: the image, ok or the error, and the backend.
A re-run pulls only the images that have no ok line for the same backend.

    python ops/prepull_images.py --tasks-dir ~/sprout-data/datasets/deep-swe/tasks --workers 4
    python ops/prepull_images.py --backend microvm --tasks-dir <tb21> --tasks-dir <hard51> --workers 2
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
    """The images that ``state`` records as pulled for ``backend``. A record without one counts as microvm."""
    done = set()
    if state.exists():
        for line in state.read_text().splitlines():
            record = json.loads(line)
            if record.get("ok") and record.get("backend", "microvm") == backend:
                done.add(record["image"])
    return done


def pull_one_microvm(image: str, runtime_bin: str) -> dict:
    """Cold-start a microVM from ``image``, destroy it, and return the state record.

    The cold start makes AgentENV convert the image.
    """
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
    """Pull ``image``, extract its base into ``store``, and return the state record.

    No container starts. A failure returns a record with ``ok`` false, not an exception.
    """
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
    # 2 overlaps conversion (CPU) with pulling (network). Eight parallel pulls of uncached
    # images exceeded Docker Hub's anonymous rate limit and failed 49 of 125 instances.
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

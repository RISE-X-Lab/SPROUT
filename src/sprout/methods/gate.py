"""The grader gate: the task's oracle must score 1 and doing nothing must score 0.

Not a test of the benchmark's grader -- of *our transport around it*: the
sandbox environment (a rootless podman container on the btrfs store, or with
``--backend microvm`` an AgentENV microVM),
the replayed tests/Dockerfile, artifact collection, the separate verifier
sandbox and reading the reward. Each check is a real Harbor trial on the same
environment class the evaluations use, with Harbor's own ``oracle`` agent (runs
``solution/solve.sh``) or ``nop`` agent (touches nothing).

    python -m sprout.methods.gate --tasks-dir ~/datasets/deep-swe/tasks --output runs/gate \\
        --runtime-bin runtime/sprout-runtime --workers 8                  # every task, resumable
    python -m sprout.methods.gate ... --task abs-module-cache-flags --mode oracle
    python -m sprout.methods.gate ... --backend microvm      # on the microVM service instead

Both outcomes are required. A grader that passes the oracle but also passes
nothing is measuring nothing; one that fails the oracle is broken plumbing.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

from harbor.models.task.task import Task
from harbor.trial.trial import Trial

from sprout.methods.evaluate import ENVIRONMENTS, trial_config
from sprout.worker.sandbox.backends import default_podman_store

MODES = ("oracle", "nop")


def check(args, task_dir: Path, mode: str) -> dict:
    """One trial; a missing result is an error, never a silent 0."""
    task = Task(task_dir)
    started = time.time()
    record = {"task": task_dir.name, "mode": mode, "expected": 1.0 if mode == "oracle" else 0.0}
    # Each task's trials in a directory of its own: a trial's name keeps only
    # the first 36 characters of the task's, and tasks that share them
    # (SWE-bench Pro's eight openlibrary instances) wrote one trial directory.
    task_args = {**vars(args), "task_dir": task_dir, "output": Path(args.output) / task_dir.name}
    config = trial_config(SimpleNamespace(**task_args), task, name=mode, agent=mode)
    directory = config.trials_dir / config.trial_name
    try:
        result = directory / "result.json"
        if result.exists() and json.loads(result.read_text()).get("exception_info"):
            # A trial that died of infrastructure is retried, not remembered.
            shutil.rmtree(directory)
        if not result.exists():
            trial = asyncio.run(Trial.create(config))
            asyncio.run(trial.run())
        body = json.loads(result.read_text())
        if body.get("exception_info"):
            raise RuntimeError(str(body["exception_info"])[:1000])
        reward = ((body.get("verifier_result") or {}).get("rewards") or {}).get("reward")
        record.update(reward=reward, ok=reward == record["expected"], trial=str(directory))
    except Exception as exc:  # noqa: BLE001 - one check must not sink the batch
        record.update(ok=False, error=f"{type(exc).__name__}: {exc}")
    record["seconds"] = round(time.time() - started, 1)
    return record


def check_task(args, task_dir: Path, modes: list[str]) -> list[dict]:
    # Modes run in sequence for one task: two microVM trials on the same image
    # at once each try to build the same per-image template, and the second
    # fails; two podman trials would both extract the image's rootfs.
    return [check(args, task_dir, mode) for mode in modes]


def already_ok(log_path: Path) -> set:
    done = set()
    if log_path.exists():
        for line in log_path.read_text().split("\n"):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("ok"):
                done.add((record["task"], record["mode"]))
    return done


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--selection", type=Path, help="JSON list of task directory names")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="both", choices=["both", *MODES])
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--runtime-bin", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=tuple(ENVIRONMENTS),
        default="podman",
        help="the sandboxes every check runs in: podman (rootless containers on btrfs "
        "subvolumes of --store, the default) or microvm (AgentENV)",
    )
    parser.add_argument(
        "--store",
        type=Path,
        help="podman: the btrfs store (default $SPROUT_PODMAN_STORE, else ~/sprout-btrfs)",
    )
    parser.add_argument("--server-url", help="microvm: the AgentENV server")
    parser.add_argument("--api-key-file", type=Path, help="microvm: the AgentENV key")
    parser.add_argument("--sandbox-ttl", type=int, default=36000, help="microvm: the VM lease")
    args = parser.parse_args(argv)
    if args.backend == "podman" and not args.store:
        args.store = Path(default_podman_store())
    args.dataset_digest = "gate"
    args.checkpoint_mode = "none"
    names = list(args.task)
    if args.selection:
        names += json.loads(args.selection.read_text())
    if not names:
        names = sorted(p.name for p in args.tasks_dir.iterdir() if (p / "task.toml").is_file())
    modes = list(MODES) if args.mode == "both" else [args.mode]
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "gate.jsonl"
    done = already_ok(log_path)
    jobs = [(args.tasks_dir / name, [m for m in modes if (name, m) not in done]) for name in names]
    jobs = [(task_dir, ms) for task_dir, ms in jobs if ms]
    total = sum(len(ms) for _, ms in jobs)
    print(f"{len(names)} task(s) x {'/'.join(modes)}: {total} check(s) to run, {len(done)} already ok")

    from concurrent.futures import ThreadPoolExecutor, as_completed

    failures, n = [], 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool, log_path.open("a", encoding="utf-8") as log:
        futures = [pool.submit(check_task, args, task_dir, ms) for task_dir, ms in jobs]
        for future in as_completed(futures):
            for record in future.result():
                n += 1
                log.write(json.dumps(record, ensure_ascii=False) + "\n")
                log.flush()
                print(
                    "[%d/%d] %s %-7s %-45s %s"
                    % (
                        n,
                        total,
                        "ok  " if record.get("ok") else "FAIL",
                        record["mode"],
                        record["task"],
                        record.get("error") or "reward=%s" % record.get("reward"),
                    ),
                    flush=True,
                )
                if not record.get("ok"):
                    failures.append(record)
    done = already_ok(log_path)
    summary = {
        "tasks": len(names),
        "modes": modes,
        "ok": sorted("%s/%s" % key for key in done if key[0] in set(names)),
        "failures": failures,
    }
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\nGATE: {len(summary['ok'])}/{len(names) * len(modes)} checks ok; "
        f"{len(failures)} failed this run -> {args.output}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

"""The task gate: a task's oracle must score 1 and doing nothing must score 0 (ADR-0002).

For each task the gate runs an ``oracle`` rollout, which runs the task's
``solution/solve.sh``, and then a ``nop`` rollout, which does nothing. The task's own
verifier grades each. Every rollout and grade is a Run Store job that the search
coordinator (``sprout.search``) submits and follows, so a worker must serve the
profiles. A check passes only when the verifier wrote the reward that the check
expects. The gate tests the transport around the grader: the sandbox of the agent
phase, the artifacts, the verifier's sandbox and the reward file. As in Harbor,
``solve.sh`` and ``test.sh`` run as the image's USER with its ENV, unless
``--no-image-user`` or ``--no-image-env``.

    python -m sprout.eval.gate --tasks-dir ~/datasets/deep-swe/tasks --output runs/gate --workers 8
    python -m sprout.eval.gate ... --task abs-module-cache-flags --mode oracle

``gate.jsonl`` gets a line for each check. A rerun into the same output skips the checks that
passed and runs the others again. ``summary.json`` names the tasks that failed, and
``groups/<run id>/`` keeps each run's group documents. The checks of a task carry one affinity
key, and a check whose host is lost runs again under the task's next key (docs/EVALUATION.md).
"""

from __future__ import annotations

import argparse
import json
import os
import time
import uuid
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

from sprout.contract.harbor_task import load_task
from sprout.eval.groups import DEFAULT_RUNSTORE_URL, Groups
from sprout.eval.jobs import (
    ImageOptions,
    add_image_flags,
    check_environment,
    grade_template,
    group_request,
    image_options,
    rollout_job,
    run_sample,
)
from sprout.search.client import RunStoreClient
from sprout.search.group import Submitter
from sprout.search.rerun import RUNS_AGAIN, again_key, lost

#: What each check's verifier must score.
EXPECTED = {"oracle": 1.0, "nop": 0.0}
MODES = tuple(EXPECTED)


def check_record(document: dict[str, Any], *, task: str, mode: str) -> dict[str, Any]:
    """The ``gate.jsonl`` line of one check, from its settled group document.

    A check whose job lost its data with its host has the ``reason`` ``data_lost``."""
    (sample,) = document["samples"]
    actor, grade = sample["actor"], sample["grade"]
    record: dict[str, Any] = {
        "task": task,
        "mode": mode,
        "expected": EXPECTED[mode],
        "affinity": document["affinity"],
        "rollout_job": actor["job_id"],
        "grade_job": grade["job_id"],
    }
    if lost(document):
        record["reason"] = "data_lost"
    error: str | None
    if actor["state"] != "succeeded":
        error = f"the rollout {actor['state']}: {actor.get('error')}"
    elif grade["state"] != "succeeded":
        error = f"the grade {grade['state']}: {grade.get('error')}"
    else:
        result = grade.get("result") or {}
        record.update(reward=result.get("reward"), reason=result.get("reason"))
        # A reward of 0 without the verifier's word is no evidence that nothing scores 0.
        error = None if record["reason"] is None else f"the verifier wrote no reward: {record['reason']}"
    record["ok"] = error is None and record.get("reward") == EXPECTED[mode]
    return {**record, "error": error} if error else record


def check_request(
    task_dir: Path,
    *,
    mode: str,
    group_id: str,
    run_id: str,
    profile: str,
    grade_profile: str,
    image: ImageOptions,
) -> dict[str, Any]:
    """The group request ``group_id`` of one check: the ``mode`` agent's rollout on the task, and its grade.

    Raises ValueError for a task that the reader refuses or that a rollout cannot run.
    """
    task = load_task(task_dir)
    check_environment(task)
    job = rollout_job(task, agent=mode, profile=profile, checkpoint=False, image=image)
    grade = grade_template(task, profile=grade_profile, image=image)
    return group_request(
        group_id,
        run_id=run_id,
        context={"task": task.name, "gate": mode},
        samples=[run_sample(mode, job=job, grade=grade)],
    )


@dataclass
class _Checks:
    """The checks of one task in one gate run. A check that runs again changes ``modes`` and ``keys``."""

    task_dir: Path
    #: The modes still to run, in order.
    modes: deque[str]
    #: The affinity key of each run of the task's checks: the first, then one for each run again.
    keys: list[str] = field(default_factory=lambda: ["gate-" + uuid.uuid4().hex])

    def group_id(self, mode: str) -> str:
        """The group id of the check ``mode``: ``<task>-<mode>``, with ``-r<n>`` in the n-th run again."""
        runs = len(self.keys) - 1
        return f"{self.task_dir.name}-{mode}" + (f"-r{runs}" if runs else "")


class Gate:
    """The checks of one gate run: each task's modes in turn, ``workers`` tasks at a time.

    ``report`` gets each check's ``gate.jsonl`` record as soon as the check settles. A check whose job
    lost its data runs again first, under the task's next key, unless the task ran again ``RUNS_AGAIN`` times.
    """

    def __init__(
        self,
        *,
        groups: Groups,
        run_id: str,
        profile: str,
        grade_profile: str,
        image: ImageOptions,
        workers: int,
        report: Callable[[dict[str, Any]], None],
    ) -> None:
        self._groups = groups
        self._run_id = run_id
        self._profile = profile
        self._grade_profile = grade_profile
        self._image = image
        self._workers = workers
        self._report = report

    def run(self, checks: list[tuple[Path, list[str]]]) -> None:
        """Run ``(task directory, modes)`` checks until every one of them has a record."""
        pending = deque(_Checks(task_dir, deque(modes)) for task_dir, modes in checks)
        # Group id -> the checks of the group's task, and the group's mode.
        active: dict[str, tuple[_Checks, str]] = {}
        while pending or active:
            while pending and len(active) < self._workers:
                self._start(pending.popleft(), active=active)
            self._groups.tick()
            for group_id, (task, mode) in list(active.items()):
                document = self._groups.document(group_id)
                if document["ready"]:
                    del active[group_id]
                    self._settled(task, mode=mode, document=document)
                    self._start(task, active=active)
            if active:
                self._groups.pause()

    def _settled(self, task: _Checks, *, mode: str, document: dict[str, Any]) -> None:
        """Report the check, or put it first among the task's modes to run again under the next key."""
        if lost(document) and len(task.keys) <= RUNS_AGAIN:
            task.keys.append(again_key(task.keys[0], runs=len(task.keys)))
            task.modes.appendleft(mode)
            return
        self._report(check_record(document, task=task.task_dir.name, mode=mode))

    def _start(self, task: _Checks, *, active: dict[str, tuple[_Checks, str]]) -> None:
        """Submit the task's next mode. A mode that cannot be submitted is recorded as failed at once."""
        while task.modes:
            mode = task.modes.popleft()
            try:
                request = check_request(
                    task.task_dir,
                    mode=mode,
                    group_id=task.group_id(mode),
                    run_id=self._run_id,
                    profile=self._profile,
                    grade_profile=self._grade_profile,
                    image=self._image,
                )
                group_id = self._groups.start(request, affinity=task.keys[-1])
            except (ValueError, OSError) as error:
                self._report(
                    {
                        "task": task.task_dir.name,
                        "mode": mode,
                        "expected": EXPECTED[mode],
                        "ok": False,
                        "error": f"{type(error).__name__}: {error}",
                    }
                )
                continue
            active[group_id] = (task, mode)
            return


def already_ok(log_path: Path) -> set[tuple[str, str]]:
    """The (task, mode) checks that a line of ``gate.jsonl`` records as passed."""
    done: set[tuple[str, str]] = set()
    if log_path.exists():
        for line in log_path.read_text().split("\n"):
            try:
                record = json.loads(line)
            except ValueError:
                continue
            if isinstance(record, dict) and record.get("ok"):
                done.add((record["task"], record["mode"]))
    return done


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """The options of a gate run, from ``argv``. Exits with status 2 for a value that no run can use."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--tasks-dir", type=Path, required=True)
    parser.add_argument("--task", action="append", default=[])
    parser.add_argument("--selection", type=Path, help="JSON list of task directory names")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--mode", default="both", choices=["both", *MODES])
    parser.add_argument(
        "--workers", type=int, default=8, help="how many tasks are checked at once (default 8)"
    )
    parser.add_argument("--runstore-url", default=DEFAULT_RUNSTORE_URL, help="the Run Store's HTTP API")
    parser.add_argument(
        "--runstore-token-env",
        default="SPROUT_RUNSTORE_TOKEN",
        help="environment variable holding the Run Store's bearer token",
    )
    parser.add_argument("--profile", default="mini", help="the worker profile of every rollout")
    parser.add_argument("--grade-profile", default="grade", help="the worker profile of every grade")
    parser.add_argument(
        "--poll-interval", type=float, default=5.0, help="seconds between two looks at the Run Store's jobs"
    )
    add_image_flags(parser)
    args = parser.parse_args(argv)
    if args.workers < 1 or args.poll_interval <= 0:
        parser.error("--workers must be at least 1 and --poll-interval positive")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    """Gate the tasks that the command line ``argv`` selects, and return the exit status of ``run_gate``.

    Raises ValueError when the variable of ``--runstore-token-env`` holds no token.
    """
    args = parse_args(argv)
    token = os.environ.get(args.runstore_token_env, "").strip()
    if not token:
        raise ValueError(f"${args.runstore_token_env} holds no Run Store token")
    client = RunStoreClient(args.runstore_url, token)
    try:
        return run_gate(args, client=client, pause=lambda: time.sleep(args.poll_interval))
    finally:
        client.close()


def run_gate(args: argparse.Namespace, *, client: RunStoreClient, pause: Callable[[], None]) -> int:
    """Gate the tasks that ``args`` (``parse_args``) selects. Returns 0 unless a check of this run failed.

    ``pause`` runs between two looks at the Run Store's jobs.
    """
    names = list(args.task)
    if args.selection:
        names += json.loads(args.selection.read_text())
    if not names:
        names = sorted(p.name for p in args.tasks_dir.iterdir() if (p / "task.toml").is_file())
    modes = list(MODES) if args.mode == "both" else [args.mode]
    args.output.mkdir(parents=True, exist_ok=True)
    log_path = args.output / "gate.jsonl"
    done = already_ok(log_path)
    checks = [(args.tasks_dir / name, [mode for mode in modes if (name, mode) not in done]) for name in names]
    checks = [(task_dir, ms) for task_dir, ms in checks if ms]
    total = sum(len(ms) for _, ms in checks)
    print(f"{len(names)} task(s) x {'/'.join(modes)}: {total} check(s) to run, {len(done)} already ok")
    # Each run submits its own jobs: a check that failed before runs again, never reads the old job.
    run_id = uuid.uuid4().hex
    groups = Groups(
        client=client,
        directory=args.output / "groups" / run_id,
        submitter=Submitter(context_key="gate", key_prefix=f"gate/{run_id}/"),
        pause=pause,
    )
    with log_path.open("a", encoding="utf-8") as stream:
        log = _Log(stream, total=total)
        gate = Gate(
            groups=groups,
            run_id=run_id,
            profile=args.profile,
            grade_profile=args.grade_profile,
            image=image_options(args),
            workers=args.workers,
            report=log,
        )
        gate.run(checks)
    return _summarize(args.output, names=names, modes=modes, failures=log.failures, log_path=log_path)


class _Log:
    """``gate.jsonl``, open for appending: each record is written, printed and, when it failed, kept."""

    def __init__(self, stream: TextIO, *, total: int) -> None:
        self._stream, self._total, self._count = stream, total, 0
        self.failures: list[dict[str, Any]] = []

    def __call__(self, record: dict[str, Any]) -> None:
        self._count += 1
        self._stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self._stream.flush()
        if not record["ok"]:
            self.failures.append(record)
        print(
            "[%d/%d] %s %-7s %-45s %s"
            % (
                self._count,
                self._total,
                "ok  " if record["ok"] else "FAIL",
                record["mode"],
                record["task"],
                record.get("error") or "reward=%s" % record.get("reward"),
            ),
            flush=True,
        )


def _summarize(
    output: Path, *, names: list[str], modes: list[str], failures: list[dict[str, Any]], log_path: Path
) -> int:
    done = already_ok(log_path)
    ok = sorted("%s/%s" % key for key in done if key[0] in set(names))
    failed_tasks = sorted({name for name in names for mode in modes if (name, mode) not in done})
    summary = {
        "tasks": len(names),
        "modes": modes,
        "ok": ok,
        "failed_tasks": failed_tasks,
        "failures": failures,
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(
        f"\nGATE: {len(ok)}/{len(names) * len(modes)} checks ok; {len(failures)} failed this run -> {output}"
    )
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())

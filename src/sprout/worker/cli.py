"""``python -m sprout.worker``: run Run Store jobs on this host, or one agent run, and clean up.

``start`` checks the profiles, then claims and runs Run Store jobs, under the worker id
``<host>/<name>``. ``run`` runs mini-swe-agent once, in a rootless podman container (the
default) or an AgentENV microVM. ``show`` summarizes a journal and ``atif`` exports it.
``sweep`` deletes the snapshots that a named journal records, ``reap`` reclaims the
sandboxes of killed runs, and ``doctor`` checks this host for the podman backend.

    python -m sprout.worker run --model <model> \
        --sandbox-image docker.io/library/python:3.11-slim --runtime-bin runtime/sprout-runtime \
        --cwd /tmp --journal runs/one.jsonl "fix the failing test in /testbed"
    python -m sprout.worker show runs/<id>.jsonl
    python -m sprout.worker sweep runs/<id>.jsonl      # a dry run without --yes
    python -m sprout.worker reap --dry-run
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from threading import Event
from typing import TYPE_CHECKING, Any

from sprout.contract.backend import default_podman_store
from sprout.contract.run_spec import RunSpec
from sprout.contract.storage import volatile_reason
from sprout.worker.doctor import CHECKOUT_RUNTIME, host_report
from sprout.worker.run.atif import export_file
from sprout.worker.run.journal import read_journal
from sprout.worker.run.reap import AgentEnvClient, PodmanClient, Reaper, parse_duration
from sprout.worker.run.resources import ResourceLedger
from sprout.worker.run.runner import Runner
from sprout.worker.run.sweep import journal_snapshots, snapshot_deleter, sweep
from sprout_sandbox.podman_cli import default_podman_bin

if TYPE_CHECKING:
    from sprout.runstore.store import Store


def _backend_section(name: str, runtime_bin: str | None, store: str | None = None) -> dict[str, Any]:
    """The backend config of a run, from the CLI flags.

    The config holds no AgentENV URL or key. The microVM pool reads ``AENV_SERVER_URL`` and
    ``AENV_API_KEY`` itself, so a key never lands in shell history.
    """
    # podman spawns from an image and from a snapshot the same way, so it needs no from_image.
    if name == "podman":
        return {"backend": name, name: {"store": store, "runtime_bin": runtime_bin}}
    # A name on the command line is an OCI image reference, and only the cold-start path takes one.
    section: dict[str, Any] = {"from_image": True}
    # With a runtime binary, the pool builds a per-image template instead. A cold-started plain
    # image comes up without the runtime and answers each tool call with a 502.
    if runtime_bin:
        section["runtime_bin"] = runtime_bin
    return {"backend": name, name: section}


def _cmd_run(args: argparse.Namespace) -> int:
    """Run one task from the CLI flags through :class:`Runner`, and print the outcome.

    Returns 0 if the run succeeded, else 1.
    """
    if args.backend == "podman" and not args.runtime_bin:
        raise SystemExit(
            "--backend podman needs --runtime-bin: the sprout-runtime binary is mounted into every container"
        )
    if args.backend == "podman" and not args.store:
        args.store = default_podman_store()
    if args.backend == "podman" and args.final_memory_snapshot:
        # Without this refusal, the pool declines the snapshot in silence: the journal says
        # `snapshot.memory captured=false`, and the run exits 0.
        raise SystemExit(
            "--final-memory-snapshot needs a backend with memory snapshots; podman snapshots are disk-only"
        )
    spec = RunSpec(
        prompt=args.prompt,
        slot=args.slot,
        agent="harness" if args.slot == "sprout-harness" else "mini",
        cwd=args.cwd,
        model=args.model,
        timeout_s=args.timeout,
        agent_id=args.agent_id,
        run_id=args.run_id,
        journal_path=args.journal,  # guarded below: journals must survive a reboot
        sandbox_image=args.sandbox_image,
        keep_sandbox=args.keep_sandbox,
        backend=_backend_section(args.backend, args.runtime_bin, args.store),
        runtime_bin=args.runtime_bin,
        final_memory_snapshot=args.final_memory_snapshot,
        extra=json.loads(args.extra) if args.extra else {},
    )

    def report(kind: str, payload: dict[str, Any]) -> None:
        if kind == "sandbox":
            print("sandbox     %s (bound)" % payload["sandbox_id"])
        elif kind == "gateway":
            print("gateway     %s (budget %s)" % (payload["url"], payload.get("budget_usd") or "none"))

    reason = volatile_reason(args.journal) if args.journal else None
    if reason and not args.volatile_ok:
        raise SystemExit("refusing: %s (pass --volatile-ok to override)" % reason)

    # The ledger records each sandbox that this run acquires. `python -m sprout.worker reap`
    # reads it to reclaim what a killed process could not release.
    ledger = ResourceLedger()
    outcome = Runner(ledger=ledger, on_event=report).run(spec)

    print("run_id      %s" % outcome.run_id)
    print("journal     %s" % outcome.journal_path)
    print("status      %s" % outcome.status)
    if outcome.native_session_id:
        print("session     %s" % outcome.native_session_id)
    if outcome.usage:
        print("usage       %s" % json.dumps(outcome.usage))
    if outcome.checkpoints:
        print("checkpoints %d" % outcome.checkpoints)
    if outcome.error:
        print("error       %s" % outcome.error, file=sys.stderr)
    if outcome.final_text:
        print("---")
        print(outcome.final_text)
    return 0 if outcome.ok else 1


def _cmd_show(args: argparse.Namespace) -> int:
    records = read_journal(args.journal)
    counts = Counter(r.get("type") for r in records)
    print("events   %d" % len(records))
    for etype, count in counts.most_common():
        print("  %-24s %d" % (etype, count))
    return 0


def _cmd_atif(args: argparse.Namespace) -> int:
    document = export_file(args.journal)
    payload = json.dumps(document, indent=2, ensure_ascii=False)
    if args.output:
        Path(args.output).write_text(payload, encoding="utf-8")
        print("wrote %s (%d steps)" % (args.output, len(document["steps"])))
    else:
        print(payload)
    return 0


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Delete the snapshots that each named journal records. Without ``--yes``, a dry run.

    A sweep appends a ``snapshots.swept`` line to the journal, and a later sweep skips it. The
    journal's ``sandbox.created`` names the backend and, for podman, the store.
    """
    exit_code = 0
    for journal_path in args.journals:
        path = Path(journal_path)
        if not path.exists():
            print("no such journal: %s" % path, file=sys.stderr)
            exit_code = 2
            continue
        snapshots = journal_snapshots(path)
        if snapshots.swept:
            print("%s: already swept, skipping" % path)
            continue
        delete = snapshot_deleter(snapshots.backend, args.store or snapshots.store)
        if isinstance(delete, str):
            print("%s: %s" % (path, delete), file=sys.stderr)
            exit_code = 1
            continue
        count, backend = len(snapshots.ids), snapshots.backend
        if not args.yes:
            print("%s: would delete %d %s snapshot(s)  (dry run; pass --yes)" % (path, count, backend))
            continue
        deleted, failed = sweep(snapshots, delete)
        print(
            "%s: deleted %d/%d snapshot(s)%s"
            % (path, deleted, count, ", %d failed" % failed if failed else "")
        )
        if failed:
            exit_code = 1
    return exit_code


def _cmd_reap(args: argparse.Namespace) -> int:
    ledger = ResourceLedger(args.ledger) if args.ledger else ResourceLedger()

    older_than = parse_duration(args.older_than) if args.older_than else None
    if args.include_unknown and older_than is None:
        print("refusing --include-unknown without --older-than", file=sys.stderr)
        return 2

    # A claim of a backend without a client here is listed as foreign and left alone. The podman
    # client also searches the stores of the claims and of --store for a rootfs without a container.
    stores = {resource.store for resource in ledger.resources() if resource.store} | set(args.store or [])
    reaper = Reaper({"microvm": AgentEnvClient(), "podman": PodmanClient(stores=stores)}, ledger)
    plan = reaper.plan(include_unknown=args.include_unknown, older_than=older_than)

    for sandbox_id in plan.foreign:
        print("  %-9s %s  %s" % ("foreign", sandbox_id, plan.reasons.get(sandbox_id, "")))
    if not plan.total():
        print("nothing to reap (%d kept)" % len(plan.kept))
        return 0
    for sandbox_id in plan.sandboxes:
        print(
            "  %-9s %-8s %s  %s"
            % ("sandbox", plan.backends[sandbox_id], sandbox_id, plan.reasons.get(sandbox_id, ""))
        )
    if args.dry_run:
        print("\n%d sandbox(es) would be freed (dry run)" % plan.total())
        return 0

    done = reaper.apply(plan)
    print("\nfreed %d sandbox(es)" % len(done["sandboxes"]))
    if done["failed"]:
        print("failed: %s" % ", ".join(done["failed"]), file=sys.stderr)
        return 1
    return 0


def _cmd_doctor(args: argparse.Namespace) -> int:
    """Print every check of this host, one line each or one JSON object. Returns 1 if a check fails."""
    report = host_report(
        podman_bin=args.podman_bin or default_podman_bin(),
        store=args.store or default_podman_store(),
        runtime_bin=args.runtime_bin or CHECKOUT_RUNTIME,
    )
    if args.json:
        print(json.dumps(report.to_dict(), indent=2))
    else:
        for name, check in report.checks.items():
            print("%-5s %-8s %s" % (check.status, name, check.detail))
    return 0 if report.ok else 1


def _positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def _store_and_config(config_path: Path) -> tuple[Store, dict[str, Any]]:
    """The Run Store and its config. Exits with the reason if the artifact root is in a volatile directory."""
    from sprout.runstore.config import read_config
    from sprout.runstore.store import Store

    config = read_config(config_path)
    reason = volatile_reason(config["artifact_root"])
    if reason:
        raise SystemExit(reason)
    return Store(os.environ["SPROUT_RUNSTORE_DSN"]), config


def _worker_name(value: str) -> str:
    if not value or "/" in value:
        raise argparse.ArgumentTypeError("must be nonempty and without a slash")
    return value


def _cmd_start(args: argparse.Namespace) -> int:
    """Claim and run Run Store jobs on this machine until SIGTERM or SIGINT, or one job with --once.

    Exits with the reason, before it registers, if a profile fails ``checked_grades``. Exits with the
    reason if the worker is live in another session.
    """
    from sprout.runstore.store import Conflict
    from sprout.worker.attempts import Worker, local_worker_id
    from sprout.worker.manager import WorkerManager
    from sprout.worker.profiles import checked_grades

    store, config = _store_and_config(args.config)
    try:
        grades = checked_grades(config["profiles"])
    except ValueError as error:
        raise SystemExit(str(error)) from None
    stop = Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    worker = Worker(store, config, worker_id=local_worker_id(args.worker_name), grades=grades)
    manager = WorkerManager(worker, concurrency=args.concurrency, kept=worker.kept)
    try:
        manager.run(stop=stop, once=args.once)
    except Conflict as error:
        raise SystemExit(str(error)) from None
    return 0


def _add_start_options(start: argparse.ArgumentParser) -> None:
    start.add_argument(
        "--config", type=Path, required=True, help="the Run Store config (artifact root, profiles)"
    )
    start.add_argument(
        "--concurrency", type=_positive_int, default=1, help="maximum concurrent attempts (default: 1)"
    )
    start.add_argument("--once", action="store_true", help="run one claimed job, then exit")
    start.add_argument(
        "--worker-name",
        type=_worker_name,
        default="default",
        help="the worker's name on this host; its id is <host>/<name> (default: default). A worker "
        "that starts under its id again first settles the jobs that it left running",
    )


def _add_run_options(run: argparse.ArgumentParser) -> None:
    run.add_argument("prompt")
    run.add_argument(
        "--slot",
        choices=("mini-swe-agent", "sprout-harness"),
        default="mini-swe-agent",
        help="the agent: mini-swe-agent (the default) or sprout-harness",
    )
    run.add_argument("--cwd", default=".")
    run.add_argument("--model")
    run.add_argument("--timeout", type=float, default=3600.0)
    run.add_argument("--journal")
    run.add_argument(
        "--volatile-ok",
        action="store_true",
        help="allow a journal under /tmp and friends -- refused by "
        "default, journals are the run's only record",
    )
    run.add_argument("--run-id")
    run.add_argument("--agent-id", default="agent")
    run.add_argument(
        "--sandbox-image",
        required=True,
        help="create a sandbox from this image (or snapshot); this process owns it "
        "and hands the agent its tools",
    )
    run.add_argument(
        "--backend",
        choices=("microvm", "podman"),
        default="podman",
        help="where a sandbox this process creates comes from: podman (rootless "
        "containers on btrfs subvolumes of --store; disk-only snapshots, no "
        "memory snapshots; the default) or microvm (AgentENV, reads "
        "AENV_SERVER_URL / AENV_API_KEY)",
    )
    run.add_argument(
        "--runtime-bin",
        default=None,
        help="the sprout-runtime binary: built into a per-image template (microvm) "
        "or mounted into every container (podman, where it is required)",
    )
    run.add_argument(
        "--store",
        default=None,
        help="podman: the directory on a btrfs filesystem the sandboxes live in (made "
        "once with sudo ops/btrfs_store.sh; default $SPROUT_PODMAN_STORE, else "
        "~/sprout-btrfs)",
    )
    run.add_argument(
        "--final-memory-snapshot",
        action="store_true",
        help="also take a full (memory) snapshot at the end, for a grader that needs "
        "the sandbox's processes alive; the final disk snapshot is taken either way "
        "(microvm only: podman snapshots are disk-only)",
    )
    run.add_argument(
        "--keep-sandbox",
        action="store_true",
        help="do not destroy a provisioned sandbox on exit (grading, inspection)",
    )
    run.add_argument("--extra", help="JSON dict of slot-specific options")


def _add_sweep_options(sweep: argparse.ArgumentParser) -> None:
    sweep.add_argument(
        "journals",
        nargs="+",
        help="journal file(s) of the run(s) whose snapshots die; naming them is the consent",
    )
    sweep.add_argument("--yes", action="store_true", help="actually delete (default is a dry run)")
    sweep.add_argument(
        "--store",
        default=None,
        help="podman journals only: the btrfs store holding the snapshots, when "
        "the journal does not name it or the store has moved",
    )


def _add_reap_options(reap: argparse.ArgumentParser) -> None:
    reap.add_argument("--ledger", default=None, help="path to resources.jsonl")
    reap.add_argument("--dry-run", action="store_true")
    reap.add_argument(
        "--include-unknown",
        action="store_true",
        help="also consider backend sandboxes the ledger never saw (needs care)",
    )
    reap.add_argument("--older-than", help="with --include-unknown, e.g. 24h")
    reap.add_argument(
        "--store",
        action="append",
        default=None,
        help="podman: a btrfs store to look in for a rootfs whose container is gone "
        "(a destroy that failed half-way); repeatable, added to the stores the "
        "ledger's claims name",
    )


def _add_doctor_options(doctor: argparse.ArgumentParser) -> None:
    doctor.add_argument(
        "--json",
        action="store_true",
        help="print one JSON object: each check, and the podman and store it checked",
    )
    doctor.add_argument(
        "--podman-bin",
        default=None,
        help="the podman to check (default: the one every run uses, $SPROUT_PODMAN_BIN, else the podman "
        "ops/setup_podman.sh installed, else podman on PATH)",
    )
    doctor.add_argument(
        "--store", default=None, help="the store to check (default $SPROUT_PODMAN_STORE, else ~/sprout-btrfs)"
    )
    doctor.add_argument(
        "--runtime-bin",
        type=Path,
        default=None,
        help="the sprout-runtime binary to check (default: runtime/sprout-runtime of this checkout)",
    )


def main(argv: Sequence[str] | None = None) -> int:
    """Run the worker CLI on ``argv`` (default ``sys.argv[1:]``), and return the exit status."""
    parser = argparse.ArgumentParser(prog="worker", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    start = sub.add_parser("start", help="claim and run Run Store jobs on this machine")
    _add_start_options(start)
    start.set_defaults(func=_cmd_start)

    run = sub.add_parser("run", help="run a task through mini-swe-agent")
    _add_run_options(run)
    run.set_defaults(func=_cmd_run)

    show = sub.add_parser("show", help="summarize a journal")
    show.add_argument("journal")
    show.set_defaults(func=_cmd_show)

    atif = sub.add_parser("atif", help="export a journal as ATIF v1.8")
    atif.add_argument("journal")
    atif.add_argument("-o", "--output")
    atif.set_defaults(func=_cmd_atif)

    sweep = sub.add_parser(
        "sweep", help="delete the snapshots a named journal records (dry-run without --yes)"
    )
    _add_sweep_options(sweep)
    sweep.set_defaults(func=_cmd_sweep)

    reap = sub.add_parser(
        "reap", help="reclaim sandboxes left by dead runs (microVMs and podman containers alike)"
    )
    _add_reap_options(reap)
    reap.set_defaults(func=_cmd_reap)

    doctor = sub.add_parser(
        "doctor", help="check this host for the podman backend; exit 1 when a check fails"
    )
    _add_doctor_options(doctor)
    doctor.set_defaults(func=_cmd_doctor)

    args = parser.parse_args(argv)
    status: int = args.func(args)
    return status


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

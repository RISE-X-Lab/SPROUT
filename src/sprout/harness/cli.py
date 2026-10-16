"""``python -m sprout.harness`` — run mini-swe-agent once, inspect a journal, export ATIF.

    # mini-swe-agent in a rootless podman container on this machine (snapshots
    # are btrfs subvolumes of the store, ~/sprout-btrfs unless --store names one)
    python -m sprout.harness run --model <model> \
        --sandbox-image docker.io/library/python:3.11-slim --runtime-bin runtime/sprout-runtime \
        --cwd /tmp --journal runs/one.jsonl "fix the failing test in /testbed"

    # the same on a fresh microVM (AgentENV, from AENV_SERVER_URL / AENV_API_KEY)
    python -m sprout.harness run --backend microvm --model <model> \
        --sandbox-image python:3.11-slim --runtime-bin runtime/sprout-runtime \
        --cwd /tmp --journal runs/one.jsonl "..."

    python -m sprout.harness show   runs/<id>.jsonl        # event summary
    python -m sprout.harness atif   runs/<id>.jsonl -o t.json

    # delete the snapshots a named run's journal records (dry run without --yes);
    # AgentENV records or the podman store's subvolumes, as the journal says
    python -m sprout.harness sweep runs/<id>.jsonl

    # reclaim sandboxes left behind by killed runs: microVMs and podman containers
    python -m sprout.harness reap --dry-run

"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from collections import Counter
from collections.abc import Callable
from pathlib import Path

from sprout.harness.atif import export_file
from sprout.harness_core.events import SANDBOX_CREATED
from sprout.harness_core.journal import journal_events, read_journal, volatile_reason
from sprout.harness_orchestrator import (
    AgentEnvClient,
    Orchestrator,
    PodmanClient,
    Reaper,
    ResourceLedger,
    RunSpec,
    parse_duration,
)
from sprout.harness_sandbox.backends import default_podman_store
from sprout_sandbox import BtrfsStore


def _backend_section(name: str, runtime_bin: str | None, store: str | None = None) -> dict:
    """The backend config from CLI flags.

    ``server_url`` and ``api_key`` are left out on purpose: ``build_pool`` reads
    AENV_SERVER_URL / AENV_API_KEY itself, so a key never has to be typed on a
    command line where it would land in shell history.

    ``from_image`` is set for microvm because a name given on a command line is
    an OCI image reference and only the cold-start path accepts one. With
    ``--runtime-bin`` a per-image template is built instead, which is what a
    plain image needs: the backend runs no startup command for one, so a
    cold-started image comes up without the runtime and answers every tool call
    with a 502.

    podman has one spawn path for images and snapshots alike, so it needs no
    such flag; it needs the store its subvolumes live in and the runtime
    binary, which is mounted into every container.
    """
    if name == "podman":
        return {"backend": name, name: {"store": store, "runtime_bin": runtime_bin}}
    section: dict = {"from_image": True}
    if runtime_bin:
        section["runtime_bin"] = runtime_bin
    return {"backend": name, name: section}


def _cmd_run(args: argparse.Namespace) -> int:
    """Translate flags into a RunSpec and let the orchestrator drive.

    The sequence (provision, gateway, checkpoints, teardown) used to be inline
    here, which meant nothing else could reuse it. This function is now argument
    parsing and printing.
    """
    if args.backend == "podman" and not args.runtime_bin:
        raise SystemExit(
            "--backend podman needs --runtime-bin: the sprout-runtime binary is mounted into every container"
        )
    if args.backend == "podman" and not args.store:
        args.store = default_podman_store()
    if args.backend == "podman" and args.final_memory_snapshot:
        # Otherwise accepted and silently a no-op: the pool declines, the quiet
        # session swallows the warning, the journal says `snapshot.memory
        # captured=false` and the run exits 0.
        raise SystemExit(
            "--final-memory-snapshot needs a backend with memory snapshots; podman snapshots are disk-only"
        )
    spec = RunSpec(
        prompt=args.prompt,
        slot=args.slot,
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

    def report(kind: str, payload: dict) -> None:
        if kind == "sandbox":
            print("sandbox     %s (bound)" % payload["sandbox_id"])
        elif kind == "gateway":
            print("gateway     %s (budget %s)" % (payload["url"], payload.get("budget_usd") or "none"))

    # Every sandbox this run acquires: `harness reap` reads the ledger to reclaim
    # what a killed process could not release.
    reason = volatile_reason(args.journal) if args.journal else None
    if reason and not args.volatile_ok:
        raise SystemExit("refusing: %s (pass --volatile-ok to override)" % reason)

    ledger = ResourceLedger()
    outcome = Orchestrator(ledger=ledger, on_event=report).run(spec)

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


def _snapshot_deleter(backend: str, store: str | None) -> Callable[[str], bool] | str:
    """How a journal's snapshots die, by the backend that took them -- or why they cannot.

    AgentENV records are deleted through its API; a podman run's snapshots are
    read-only subvolumes of the store its journal names (``--store`` overrides a
    moved one), deleted through :class:`BtrfsStore` with no pool, runtime binary
    or container involved. Either way a snapshot that is already gone counts as
    deleted, as AgentENV's 404 does.
    """
    if backend == "microvm":
        return AgentEnvClient().delete_snapshot
    if backend != "podman":
        return "a %s run; no way to delete its snapshots is known here" % backend
    if not store:
        return (
            "a podman run whose journal does not name its store; pass --store <dir> "
            "(the store its subvolumes live in)"
        )
    try:
        btrfs = BtrfsStore(store)
    except ValueError as error:
        return "a podman run: %s" % error

    def delete(snapshot_id: str) -> bool:
        if not (btrfs.snapshots / snapshot_id).is_dir():
            return True
        try:
            asyncio.run(btrfs.delete_snapshot(snapshot_id))
        except Exception as error:  # noqa: BLE001 - reported per snapshot, never raised
            print("%s: %s" % (snapshot_id, error), file=sys.stderr)
            return False
        return True

    return delete


def _cmd_sweep(args: argparse.Namespace) -> int:
    """Delete the snapshots a NAMED run's journal records. Explicitly, only.

    Dry-run by default; ``--yes`` deletes. The journal path is the consent
    token: nothing sweeps a run you did not name, and no lifecycle hook calls
    this. After a real sweep the journal gains a ``snapshots.swept`` line, so
    "this run's snapshots are gone" is itself on the record. The journal's
    ``sandbox.created`` says which backend took the snapshots, and for podman
    which store holds them.
    """
    exit_code = 0
    for journal_path in args.journals:
        path = Path(journal_path)
        if not path.exists():
            print("no such journal: %s" % path, file=sys.stderr)
            exit_code = 2
            continue
        ids, last_seq, already, backend, store = [], 0, False, "microvm", None
        # A run that was killed leaves half a record at the end of its journal.
        # The snapshots the complete records name are no less real for it; the
        # fragment moves into the swept marker below, so the journal stays a
        # file of whole lines (a marker glued to the fragment would turn it
        # into one corrupt line that every reader then trips on).
        data = path.read_bytes()
        torn = (
            data[data.rfind(b"\n") + 1 :].decode("utf-8", errors="replace")
            if data and not data.endswith(b"\n")
            else ""
        )
        for record in journal_events(path):
            last_seq = max(last_seq, record.get("seq") or 0)
            if record.get("type") == SANDBOX_CREATED and record.get("backend"):
                backend, store = record["backend"], record.get("store")
            if record.get("type") == "checkpoint.captured" and record.get("snapshot_id"):
                if record["snapshot_id"] not in ids:
                    ids.append(record["snapshot_id"])
            if record.get("type") == "snapshots.swept":
                already = True
        if already:
            print("%s: already swept, skipping" % path)
            continue
        delete = _snapshot_deleter(backend, args.store or store)
        if isinstance(delete, str):
            print("%s: %s" % (path, delete), file=sys.stderr)
            exit_code = 1
            continue
        if not args.yes:
            print("%s: would delete %d %s snapshot(s)  (dry run; pass --yes)" % (path, len(ids), backend))
            continue
        deleted = failed = 0
        for snapshot_id in ids:
            if delete(snapshot_id):
                deleted += 1
            else:
                failed += 1
        marker = {
            "v": 2,
            "type": "snapshots.swept",
            "seq": last_seq + 1,
            "run_id": "sweep",
            "agent_id": "sweep",
            "backend": backend,
            "deleted": deleted,
            "failed": failed,
        }
        if torn:
            marker["torn_tail"] = torn
            with path.open("r+b") as fh:
                fh.truncate(len(data) - len(torn.encode("utf-8", errors="replace")))
        with path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(marker) + "\n")
        print(
            "%s: deleted %d/%d snapshot(s)%s"
            % (path, deleted, len(ids), ", %d failed" % failed if failed else "")
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

    # One client per backend a ledger claim can name; a claim from any other is
    # listed as foreign and left alone. The podman client also looks into every
    # store the ledger's claims name (and --store) for a rootfs whose container
    # is already gone.
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


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="harness", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run a task through mini-swe-agent")
    run.add_argument("prompt")
    run.add_argument(
        "--slot",
        choices=("mini-swe-agent",),
        default="mini-swe-agent",
        help="the agent; mini-swe-agent is the only one",
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
    sweep.set_defaults(func=_cmd_sweep)

    reap = sub.add_parser(
        "reap", help="reclaim sandboxes left by dead runs (microVMs and podman containers alike)"
    )
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
    reap.set_defaults(func=_cmd_reap)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

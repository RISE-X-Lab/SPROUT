"""``python -m sprout.inspect``: argument parsing, dispatch and exit codes.

Exit 0 whenever the target was read, partial, errored, live or swept included
(the output says what is missing); 2 when the target does not exist or is none
of the recognised kinds. Nothing here writes under a target: ``-o`` is the only
output file.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import sprout.inspect
from sprout.inspect import export, views
from sprout.inspect.layout import KINDS, detect

#: What detect() looks for, in its order.
NOT_A_ROOT = (
    "not a run root: %s (looked for baseline/ or sprout/ with a readable manifest-*.json, one here, "
    "gate.jsonl, summary.json with attempts, a trial's config.json + agent/, a task's *.task.json, "
    "*.stage.json or baseline/attempt-*.stage.json, or a .jsonl journal)"
)
FORMAT_HELP = "text for a terminal (default); json is dataclasses.asdict of the same model"


def _dump(model) -> str:
    return json.dumps(
        asdict(model) if not isinstance(model, (dict, list)) else model,
        indent=2,
        ensure_ascii=False,
        default=str,
    )


def _write(text: str, output: str | None) -> None:
    if output:
        Path(output).write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        print("wrote %s (%d lines)" % (output, text.count("\n") + (not text.endswith("\n"))))
    else:
        sys.stdout.write(text if text.endswith("\n") else text + "\n")


def _target(path: str, accepted: tuple[str, ...], command: str) -> tuple[Path, str] | None:
    target = Path(path).expanduser()
    kind = detect(target)
    if kind is None:
        print(NOT_A_ROOT % target, file=sys.stderr)
        return None
    if kind not in accepted:
        print(
            "%s: %s is a %s; this command reads %s (try `%s`)"
            % (command, target, kind, ", ".join(accepted), views.SUGGESTED_COMMAND.get(kind, "list")),
            file=sys.stderr,
        )
        return None
    return target.resolve(), kind


def _cmd_runs(args: argparse.Namespace) -> int:
    found = _target(args.root, ("experiment", "method", "gate"), "runs")
    if found is None:
        return 2
    report = views.runs_report(*found)
    _write(_dump(report) if args.format == "json" else views.render_runs(report), None)
    return 0


def _cmd_task(args: argparse.Namespace) -> int:
    found = _target(args.root, ("experiment", "method", "task", "evaluate"), "task")
    if found is None:
        return 2
    root, kind = found
    if kind in ("experiment", "method") and not args.task:
        print("task: an experiment or method root needs the task name (ROOT TASK)", file=sys.stderr)
        return 2
    try:
        tree = views.task_tree(
            root, kind, args.task, [Path(p).expanduser().resolve() for p in args.with_dirs or []]
        )
    except LookupError as exc:  # the name is in no manifest of this root
        print("task: %s" % exc, file=sys.stderr)
        return 2
    if args.format == "json":
        text = _dump(tree)
    elif args.format == "mermaid":
        text = views.render_mermaid(tree)
    else:
        text = views.render_task(tree)
    _write(text, None)
    return 0


def _cmd_export(args: argparse.Namespace) -> int:
    found = _target(args.journal, ("journal", "trial"), "export")
    if found is None:
        return 2
    path, kind = found
    if kind == "trial":
        path = path / "agent" / "trajectory.jsonl"
        if not path.is_file():
            print("export: %s has no agent/trajectory.jsonl" % found[0], file=sys.stderr)
            return 2
    chain = export.lineage(path, prefix=not args.no_prefix)
    if args.format == "json":
        text = _dump(export.render_json(chain, max_output_chars=args.max_output_chars))
    else:
        text = export.render_markdown(chain, max_output_chars=args.max_output_chars)
    _write(text, args.output)
    return 0


def _cmd_snapshots(args: argparse.Namespace) -> int:
    found = _target(args.target, KINDS, "snapshots")
    if found is None:
        return 2
    report = views.snapshots_report(*found, store=args.store, task=args.task)
    _write(
        _dump(views.snapshots_json(report)) if args.format == "json" else views.render_snapshots(report), None
    )
    return 0


def _cmd_list(args: argparse.Namespace) -> int:
    directory = Path(args.dir).expanduser()
    if not directory.is_dir():
        print("list: %s is not a directory" % directory, file=sys.stderr)
        return 2
    rows = views.list_rows(directory.resolve())
    _write(_dump(rows) if args.format == "json" else views.render_list(rows), None)
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m sprout.inspect",
        description=sprout.inspect.__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    runs = sub.add_parser("runs", help="progress and results of an experiment or method root")
    runs.add_argument("root", help="an experiment root (baseline/, sprout/), a method root or a gate root")
    runs.add_argument("--format", choices=("text", "json"), default="text", help=FORMAT_HELP)
    runs.set_defaults(func=_cmd_runs)

    task = sub.add_parser("task", help="one task's fork tree: baseline attempts, rounds, branches")
    task.add_argument(
        "root", help="an experiment or method root (with TASK), a task directory or an evaluate output"
    )
    task.add_argument("task", nargs="?", help="the task directory name")
    task.add_argument(
        "--with",
        dest="with_dirs",
        action="append",
        metavar="EVAL_DIR",
        help="attach an evaluate output that branched from this task's attempt-1 (repeatable)",
    )
    task.add_argument(
        "--format",
        choices=("text", "mermaid", "json"),
        default="text",
        help="text is the tree (default), mermaid a graph LR with the branches hanging off their "
        "steps, json dataclasses.asdict of the same model",
    )
    task.set_defaults(func=_cmd_task)

    exp = sub.add_parser("export", help="one trajectory as markdown or json, stitched onto its parent")
    exp.add_argument("journal", help="a journal (.jsonl) or a Harbor trial directory")
    exp.add_argument(
        "--format",
        choices=("md", "json"),
        default="md",
        help="md is markdown by turn (default); json the same segments, calls and boundaries",
    )
    exp.add_argument(
        "--no-prefix",
        action="store_true",
        help="render the branch alone; the boundary block still says where it came from",
    )
    exp.add_argument("-o", "--output", help="write here instead of stdout")
    exp.add_argument(
        "--max-output-chars",
        type=int,
        default=1500,
        help="chars of each tool result to keep (head 2/3, tail 1/3); 0 = unlimited",
    )
    exp.set_defaults(func=_cmd_export)

    snaps = sub.add_parser("snapshots", help="every snapshot the journals name, and whether the store has it")
    snaps.add_argument("target", help="a journal, trial, evaluate output, task, method or experiment root")
    snaps.add_argument("--store", help="the podman btrfs store, when the journal names none or it moved")
    snaps.add_argument("--task", help="experiment/method root: only this task")
    snaps.add_argument("--format", choices=("text", "json"), default="text", help=FORMAT_HELP)
    snaps.set_defaults(func=_cmd_snapshots)

    lst = sub.add_parser("list", help="every run root under a directory, with its state")
    lst.add_argument("dir", help="a directory of run roots (each root, child or grandchild is listed)")
    lst.add_argument("--format", choices=("text", "json"), default="text", help=FORMAT_HELP)
    lst.set_defaults(func=_cmd_list)

    args = parser.parse_args(argv)
    return args.func(args)

"""Run fixed-four baseline and up to seven continuations from its first failure."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import tomllib

from sprout.harness.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S
from sprout.harness_gateway.auth import bearer_key
from sprout.harness_sandbox.backends import default_podman_store
from sprout.methods.branching import DEFAULT_PROMPT_CHARS, MAX_ROLLOUTS
from sprout.methods.leak_judge import KEY_ENV as LEAK_KEY_ENV, THRESHOLD as LEAK_THRESHOLD
from sprout.methods.prompts import load_prompts

ROOT = Path(__file__).resolve().parents[2]
METHODS = ("baseline", "sprout", "bpo", "shepherd")
#: The actor's reasoning_effort on every stage, and the analyst and reviewer's unless
#: --review-effort says otherwise.
REASONING_EFFORT = "high"
#: The protocol allows 8 rollouts per method, which its default 4-then-3 schedule is
#: derived from; a --sprout-rounds sweep may go up to this many, and a wider schedule
#: is spelled out with --branches (up to methods.branching's MAX_ROLLOUTS).
MAX_SCHEDULED_ROLLOUTS = 17


def write(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def read(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def attempts(output: Path) -> list[dict]:
    body = read(output / "summary.json")
    rows = body.get("attempts", [])
    if not rows or body.get("complete") is False or body.get("branch_schedule_complete") is False:
        raise ValueError("Incomplete evaluation")
    for row in rows:
        grade = row.get("grade") or {}
        if (row.get("status") not in (None, "completed", "timeout")
                or row.get("grading_error") or row.get("verifier_artifact_error")
                or grade.get("error") or grade.get("verifier_artifact_error")):
            raise ValueError("Actor or verifier infrastructure failure")
        if type(row.get("resolved")) is not bool:
            raise ValueError("Missing binary verdict")
    return rows


def task_digest(tasks_dir: Path, names: list[str]) -> str:
    h = hashlib.sha256()
    for name in sorted(names):
        directory = tasks_dir / name
        if not (directory / "task.toml").is_file():
            raise FileNotFoundError(directory / "task.toml")
        for path in sorted(directory.rglob("*")):
            if path.is_file() and ".git" not in path.parts and "__pycache__" not in path.parts:
                h.update(path.relative_to(tasks_dir).as_posix().encode())
                h.update(b"\0")
                h.update(path.read_bytes())
    return "sha256:" + h.hexdigest()


def snapshot_ids(output: Path) -> set[str]:
    ids = set()
    for path in output.rglob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").split("\n"):
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and row.get("type") == "checkpoint.captured" and row.get("snapshot_id"):
                ids.add(row["snapshot_id"])
    for path in output.rglob("snapshot.json"):
        value = read(path).get("final_snapshot_id")
        if value:
            ids.add(value)
    return ids


#: Where a backend keeps the snapshots a retained parent's branch points name.
SNAPSHOT_STORAGE = {
    "microvm": "AgentENV content-addressed shared base plus disk deltas; native transcript stored once",
    "podman": "read-only btrfs snapshots of the sandbox's rootfs in the store; native transcript stored once",
}


def retain_parent(output: Path) -> None:
    from dataclasses import asdict
    from sprout.harness.agents.mini_swe.history import available_branch_points, reference_at
    from sprout.harness_core.journal import read_journal
    from sprout.methods.evaluate import parent_journal

    journal = parent_journal(output)
    points = available_branch_points(journal)
    if not points:
        raise ValueError("Initial failure has no restorable assistant-turn checkpoint")
    records = []
    for step, point in sorted(points.items()):
        native = reference_at(journal, step, point.session_ckpt)
        if native is None:
            raise ValueError(f"Missing exact native prefix at step {step}")
        records.append({"checkpoint": asdict(point), "native": native})
    created = next(row for row in read_journal(journal) if row.get("type") == "sandbox.created")
    write(output / "retained-checkpoints.json", {
        "schema_version": 1, "retention": "keep-for-future-policies",
        "mode": "disk_only", "journal": str(journal),
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "points": records, "snapshot_ids": sorted(snapshot_ids(output)),
        "backend": created.get("backend"), "store": created.get("store"),
        "storage": SNAPSHOT_STORAGE[created.get("backend")],
        "restore_limitations": "No RAM, live processes, tmpfs or external services; use full snapshots for such tasks",
    })


#: Which datasets grade inside the agent's VM and so may need its processes alive
#: for a re-grade: TerminalBench's verifier is shared, DeepSWE's and SWE-bench
#: Pro's run on a collected patch in a fresh VM, where memory is worthless.
MEMORY_SNAPSHOT_DEFAULTS = {"terminalbench21": "on-failure", "deepswe": "never",
                            "swebenchpro-v2-hard": "never"}


def memory_snapshot_policy(args) -> str:
    """``auto`` resolves by dataset -- and to ``never`` on podman, whose
    snapshots are disk-only (an explicit other choice there is refused at parsing)."""
    chosen = getattr(args, "final_memory_snapshot", "auto")
    if chosen != "auto":
        return chosen
    return "never" if getattr(args, "backend", "podman") == "podman" else MEMORY_SNAPSHOT_DEFAULTS[args.dataset]


def branch_widths(args) -> list[int]:
    """Branches per sprout round. ``--branches`` lists them outright;
    ``--sprout-rounds R --sprout-branches W`` asks for R rounds of W branches each
    (a round-scaling sweep); by default the protocol's schedule, derived from
    ``--max-rollouts`` (4 then 3 for 8). parse_args refuses the first two together."""
    if getattr(args, "branches", None):
        return list(args.branches)
    rounds = getattr(args, "sprout_rounds", None)
    if rounds:
        return [args.sprout_branches] * rounds
    widths = [min(4, args.max_rollouts - 1)]
    if args.max_rollouts > 5:
        widths.append(args.max_rollouts - 5)
    return widths


#: The analyst and reviewer's own settings, each an option of methods.evaluate with
#: the same name; absent, the evaluator uses the actor's model, endpoint, key and effort.
REVIEW_OPTIONS = ("review_model", "review_endpoint", "review_key_env", "review_effort")


def command(args, task: str, output: Path, method: str,
            parent: Path | None = None, checkpoint: bool = True) -> list[str]:
    """One ``methods.evaluate`` invocation: the same runner for every dataset and method."""
    widths = branch_widths(args)
    cmd = [sys.executable, "-u", "-m", "sprout.methods.evaluate",
           "--task-dir", str(args.tasks_dir / task), "--dataset-digest", args.digest,
           "--preparation-manifest", str(args.preparation), "--output", str(output),
           "--method", method, "--model", args.model, "--model-endpoint", args.model_endpoint,
           "--model-key-env", args.model_key_env, "--runtime-bin", str(args.runtime_bin),
           "--reasoning-effort", REASONING_EFFORT, "--max-turns", str(args.max_turns),
           "--max-output-tokens", str(args.max_output_tokens),
           "--review-output-tokens", str(getattr(args, "review_output_tokens", None) or args.max_output_tokens),
           "--prompt-budget-chars", str(args.prompt_budget_chars),
           "--command-timeout", str(args.command_timeout),
           "--review-timeout", str(args.review_timeout),
           "--meta-max-tokens", str(args.max_output_tokens),
           "--max-rollouts", str(args.max_rollouts),
           "--branch-count-mode", getattr(args, "branch_count_mode", "adaptive"),
           "--rounds", str(len(widths) if method == "sprout" else 0),
           "--branches", *map(str, widths),
           "--checkpoint-mode", "disk_only" if checkpoint else "none",
           "--final-memory-snapshot", memory_snapshot_policy(args),
           "--backend", args.backend]
    if getattr(args, "prompts", None):
        # Absent, the evaluator reads its packaged prompts.yaml; either way its
        # summary records the file's sha256, as the manifest does.
        cmd += ["--prompts", str(args.prompts)]
    if getattr(args, "structured_review", False):
        cmd.append("--structured-review")
    if args.backend == "podman":
        cmd += ["--store", str(args.store)]
    else:
        cmd += ["--server-url", args.server_url]
        if args.api_key_file:
            cmd += ["--api-key-file", str(args.api_key_file)]
    if method == "sprout":
        # Only sprout has an analyst and a reviewer to configure or to judge.
        for option in REVIEW_OPTIONS:
            value = getattr(args, option, None)
            if value:
                cmd += ["--" + option.replace("_", "-"), value]
        if getattr(args, "leak_judge", False):
            cmd += ["--leak-judge", "--leak-judge-resamples", str(args.leak_judge_resamples),
                    "--leak-judge-threshold", str(args.leak_judge_threshold)]
    if parent:
        cmd += ["--parent-output", str(parent)]
    return cmd


def stage(args, task: str, output: Path, method: str,
          parent: Path | None = None, checkpoint: bool = True) -> list[dict]:
    receipt_path = output.parent / (output.name + ".stage.json")
    if receipt_path.exists():
        receipt = read(receipt_path)
        if receipt.get("complete"):
            return attempts(output)
        raise RuntimeError(f"Incomplete stage retained at {output}; choose a fresh experiment root to retry")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    receipt = {"task": task, "method": method, "started_at": time.time(),
               "complete": False, "output": str(output), "source_commit": args.source_commit}
    write(receipt_path, receipt)
    try:
        env = dict(os.environ, PYTHONPATH=str(ROOT / "src") + os.pathsep + str(ROOT),
                   LITELLM_LOCAL_MODEL_COST_MAP="True")
        if args.backend == "microvm":
            env["AENV_SERVER_URL"] = args.server_url
            if args.api_key_file:
                env["AENV_API_KEY"] = args.api_key_file.read_text().strip()
        with (output.parent / (output.name + ".log")).open("wb") as log:
            code = subprocess.call(command(args, task, output, method, parent, checkpoint),
                                   cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        receipt["exit_code"] = code
        if code != 0:
            raise RuntimeError(f"Stage exited {code}; see {output.name}.log")
        rows = attempts(output)
        if method == "baseline" and len(rows) != 1:
            raise ValueError("Baseline stage must grade exactly one independent rollout")
        if method != "baseline" and not rows[0]["resolved"] and len(rows) < 2:
            raise ValueError(f"{method} produced no graded continuation for a failed parent")
        if len(rows) > args.max_rollouts:
            raise ValueError("Method exceeded total rollout allowance")
        receipt["complete"] = True
        return rows
    except Exception as exc:
        receipt["error_type"] = type(exc).__name__
        receipt["error"] = str(exc)
        raise
    finally:
        receipt["finished_at"] = time.time()
        receipt["seconds"] = receipt["finished_at"] - receipt["started_at"]
        write(receipt_path, receipt)


def run_task(args, task: str) -> dict:
    root = args.output / "tasks" / task
    base = (args.baseline_root or args.output) / "tasks" / task / "baseline"
    state = {"task": task, "dataset": args.dataset, "methods": {}, "complete": False}
    try:
        baseline = []
        for index in range(1, args.baseline_rollouts + 1):
            out = base / f"attempt-{index}"
            if args.baseline_root:
                rows = attempts(out)
                if len(rows) != 1:
                    raise ValueError("Shared baseline must contain %d independent single-rollout stages"
                                     % args.baseline_rollouts)
            else:
                rows = stage(args, task, out, "baseline", checkpoint=index == 1)
            baseline.extend(rows)
            if not args.baseline_root:
                if index == 1 and not rows[0]["resolved"]:
                    try:
                        retain_parent(out)
                    except ValueError as exc:
                        # Lack of a branch point must not truncate the fixed-count baseline.
                        state["retention_error"] = str(exc)
        state["baseline"] = {"output": str(base), "attempts": baseline}
        parent = base / "attempt-1"
        initial = baseline[0]["resolved"]
        state["initial_resolved"] = initial
        requested = METHODS[1:] if args.method == "all" else (() if args.method == "baseline" else (args.method,))
        if requested and not initial and not (parent / "retained-checkpoints.json").is_file():
            raise ValueError("Shared failure lacks retained-checkpoints.json")
        for method in requested:
            out = root / method
            if initial:
                state["methods"][method] = {"status": "skipped_initial_success", "attempts": []}
            else:
                rows = stage(args, task, out, method, parent)
                # Row 0 is the shared parent, already counted under baseline.
                state["methods"][method] = {"status": "complete", "output": str(out), "attempts": rows[1:]}
            write(root / f"{args.method}.task.json", state)
        state["complete"] = True
    except Exception as exc:
        state.update(error_type=type(exc).__name__, error=str(exc), status="infrastructure_error")
    write(root / f"{args.method}.task.json", state)
    print(json.dumps({"task": task, "complete": state["complete"],
                      "initial_resolved": state.get("initial_resolved"),
                      "error": state.get("error")}), flush=True)
    return state


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", choices=("deepswe", "terminalbench21", "swebenchpro-v2-hard"), required=True)
    p.add_argument("--method", choices=(*METHODS, "all"), required=True)
    p.add_argument("--tasks-dir", type=Path, required=True)
    p.add_argument("--task", action="append", default=[])
    p.add_argument("--selection", type=Path)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--baseline-root", type=Path)
    p.add_argument("--workers", type=int, default=16)
    p.add_argument("--model", default="qwen3.8-27b")
    p.add_argument("--model-endpoint", default=os.environ.get("QWEN_BASE_URL"))
    p.add_argument("--model-key-env", default="QWEN_API_KEY",
                   help="environment variable holding the bearer key for --model-endpoint (default QWEN_API_KEY)")
    p.add_argument("--runtime-bin", type=Path, required=True)
    p.add_argument("--backend", choices=("microvm", "podman"), default="podman",
                   help="where every trial's sandboxes come from: podman (rootless containers on btrfs "
                        "subvolumes of --store, the default) or microvm (AgentENV)")
    p.add_argument("--store", type=Path,
                   help="podman: the btrfs store (default $SPROUT_PODMAN_STORE, else ~/sprout-btrfs)")
    p.add_argument("--server-url", default=os.environ.get("AENV_SERVER_URL", "http://127.0.0.1:18000"),
                   help="microvm: the AgentENV server")
    p.add_argument("--api-key-file", type=Path, help="microvm: the AgentENV key")
    p.add_argument("--max-turns", type=int, default=300)
    p.add_argument("--review-timeout", type=int, default=3600,
                   help="seconds one analyst or reviewer request may take (a timed-out request is retried "
                        "once with half the output budget)")
    p.add_argument("--review-model",
                   help="sprout: the analyst and reviewer model (default --model); the actor that continues "
                        "each branch is --model either way")
    p.add_argument("--review-endpoint",
                   help="sprout: the Chat Completions endpoint the analyst and reviewer use (default "
                        "--model-endpoint), e.g. a LiteLLM proxy alias for a model the actor's endpoint "
                        "does not serve")
    p.add_argument("--review-key-env",
                   help="sprout: environment variable holding the bearer key for --review-endpoint "
                        "(default --model-key-env)")
    p.add_argument("--review-effort",
                   help=f"sprout: reasoning_effort of the analyst and reviewer (default: the actor's, {REASONING_EFFORT})")
    p.add_argument("--review-output-tokens", type=int,
                   help="output budget of each analyst and reviewer request, thinking included (default "
                        "--max-output-tokens); a reviewer that thinks at high effort can spend 32k before "
                        "writing its plan")
    p.add_argument("--leak-judge", action="store_true",
                   help="sprout: check every turn the reviewer writes for hidden-test content (leak_terms and Jev, "
                        f"${LEAK_KEY_ENV}) before it is taken, sending a flagged one back (docs/LEAK_JUDGE.md)")
    p.add_argument("--leak-judge-resamples", type=int, default=2,
                   help="sprout with --leak-judge: how often a flagged turn goes back before it is taken, marked")
    p.add_argument("--leak-judge-threshold", type=float, default=LEAK_THRESHOLD,
                   help="sprout with --leak-judge: the Jev score at which a turn is flagged")
    p.add_argument("--command-timeout", type=int, default=DEFAULT_COMMAND_TIMEOUT_S,
                   help="seconds each agent command may run before the runtime kills it; recorded in the "
                        "manifest, and a shared baseline must have used the same value")
    p.add_argument("--max-output-tokens", type=int, default=64000)
    p.add_argument("--max-rollouts", type=int, default=8,
                   help="total trajectories per method, including its shared parent: 2..8 for the protocol's "
                        f"own schedule, up to {MAX_ROLLOUTS} with an explicit --branches schedule")
    p.add_argument("--branches", type=int, nargs="+",
                   help="SPROUT branches per round, one number per round; default 4 then 3 under "
                        "the default budget; counts plus the parent must fit --max-rollouts")
    p.add_argument("--sprout-rounds", type=int,
                   help="sprout: run exactly this many rounds of --sprout-branches branches each instead of the "
                        "schedule --max-rollouts implies; the rollout budget becomes 1 + rounds * branches "
                        "(the shared parent included). Prompts do not depend on the round count, so the first "
                        "k rounds of an R-round run are a k-round run")
    p.add_argument("--sprout-branches", type=int, default=1,
                   help="sprout with --sprout-rounds: branches per round (default 1)")
    p.add_argument("--branch-count-mode", choices=("adaptive", "fixed"), default="adaptive",
                   help="adaptive treats each width as a cap; fixed requires exactly that many branches")
    p.add_argument("--prompts", type=Path,
                   help="YAML with every analyst/reviewer/shepherd prompt, passed to every methods.evaluate "
                        "(default: the packaged src/sprout/methods/prompts.yaml); its sha256 is recorded in "
                        "the manifest as prompts_sha256")
    p.add_argument("--structured-review", action="store_true",
                   help="passed to every methods.evaluate: the analyst and reviewer answer in schema-constrained "
                        "JSON (response_format json_schema) instead of fenced plain-text plans; needs an "
                        "endpoint that supports json_schema, e.g. a LiteLLM proxy; recorded in the manifest")
    p.add_argument("--prompt-budget-chars", type=int, default=DEFAULT_PROMPT_CHARS,
                   help="size of every analyst and reviewer request, in characters (about chars/4 tokens); "
                        "passed to methods.evaluate, which reduces a reviewer request over it")
    p.add_argument("--final-memory-snapshot", choices=("auto", "never", "always", "on-failure"), default="auto",
                   help="memory snapshot at the end of each rollout; auto = on-failure for "
                        "terminalbench21 (shared verifier), never for patch-graded datasets")
    p.add_argument("--baseline-rollouts", type=int, default=4,
                   help="independent baseline rollouts per task; the first is the parent every branch "
                        "method starts from (1 = pass@1 only); recorded in the manifest, and a shared "
                        "baseline must have used the same count")
    p.add_argument("--plan", action="store_true", help="Validate data and print commands without API or VM calls")
    args = p.parse_args(argv)
    if args.sprout_rounds is not None:
        if args.method != "sprout":
            p.error("--sprout-rounds sets the sprout schedule; use it with --method sprout")
        if args.branches is not None:
            p.error("--sprout-rounds and --branches each give the whole schedule; pass one of them")
        if not 1 <= args.sprout_rounds <= 16 or not 1 <= args.sprout_branches <= 4:
            p.error("--sprout-rounds must be 1..16 and --sprout-branches 1..4")
        args.max_rollouts = 1 + args.sprout_rounds * args.sprout_branches
        if args.max_rollouts > MAX_SCHEDULED_ROLLOUTS:
            p.error(f"a sprout schedule may use at most {MAX_SCHEDULED_ROLLOUTS} rollouts, the parent included; "
                    "a wider one is spelled out with --branches")
    elif args.branches is None and not 2 <= args.max_rollouts <= 8:
        p.error("Select 2..8 total method rollouts: the protocol's schedule derives from the budget, "
                "and a larger budget takes an explicit --branches or --sprout-rounds schedule")
    if not args.dataset or args.workers < 1 or not 2 <= args.max_rollouts <= MAX_ROLLOUTS or args.max_turns < 1:
        p.error(f"Select a dataset, positive workers/turns, and 2..{MAX_ROLLOUTS} total method rollouts")
    if args.branches is not None and (any(n < 1 for n in args.branches)
                                     or 1 + sum(args.branches) > args.max_rollouts):
        p.error("--branches must be positive counts whose sum plus the parent fits --max-rollouts")
    if args.baseline_rollouts < 1:
        p.error("--baseline-rollouts must be at least 1: the first rollout is every branch method's parent")
    if args.prompt_budget_chars < 50_000:
        p.error("--prompt-budget-chars below 50000 leaves no room for the task and a transcript")
    if args.command_timeout <= 0:
        p.error("--command-timeout must be a positive number of seconds")
    if args.review_timeout <= 0:
        p.error("--review-timeout must be a positive number of seconds")
    if not args.model_endpoint:
        p.error("Set QWEN_BASE_URL or --model-endpoint")
    if args.review_key_env and not args.review_key_env.isidentifier():
        p.error("--review-key-env must be an environment variable name")
    if args.review_output_tokens is not None and args.review_output_tokens < 1:
        p.error("--review-output-tokens must be positive")
    if args.leak_judge and args.method != "sprout":
        p.error("--leak-judge checks reviewer-written turns; use it with --method sprout")
    if args.leak_judge_resamples < 0 or not 0 < args.leak_judge_threshold <= 1:
        p.error("--leak-judge-resamples must be >= 0 and --leak-judge-threshold in (0, 1]")
    if args.method in ("bpo", "shepherd") and not args.baseline_root:
        p.error("bpo and shepherd branch from a shared baseline: pass --baseline-root")
    if not args.model_key_env.isidentifier():
        p.error("--model-key-env must be an environment variable name")
    if args.backend == "podman" and not args.store:
        args.store = Path(default_podman_store())
    if args.backend == "podman" and args.final_memory_snapshot not in ("auto", "never"):
        p.error("--final-memory-snapshot needs a backend with memory snapshots; podman snapshots are disk-only")
    for key in ("tasks_dir", "output", "runtime_bin", "baseline_root", "api_key_file", "store", "prompts"):
        value = getattr(args, key)
        if value is not None:
            setattr(args, key, value.expanduser().resolve())
    if args.prompts is not None:
        try:
            load_prompts(args.prompts)
        except (OSError, ValueError) as error:
            p.error("--prompts: %s" % error)
    names = args.task
    if args.selection:
        selection = json.loads(args.selection.read_text())
        names += selection if isinstance(selection, list) else next(iter(selection.values()))
    if not names:
        p.error("Explicit --task or --selection is required; subsets are never silently expanded")
    if len(set(names)) != len(names) or any(Path(n).name != n or n in (".", "..") for n in names):
        p.error("Task IDs must be unique directory names")
    args.tasks = names
    return args


#: Settings a branch method must share with the baseline it continues from.
SHARED_BASELINE_KEYS = ("dataset", "dataset_digest", "tasks", "model", "reasoning_effort", "max_turns",
                        "max_output_tokens", "command_timeout", "backend", "store", "baseline_rollouts")
#: What a manifest written before a key existed implicitly used: every run
#: before `--command-timeout` gave each command the constant 60 seconds, and
#: every one before `--baseline-rollouts` ran four baseline rollouts.
SHARED_BASELINE_HISTORY = {"command_timeout": 60, "baseline_rollouts": 4}


def shared_baseline_mismatch(prior: dict, manifest: dict) -> str | None:
    """The first shared setting on which ``manifest`` differs from the baseline's ``prior``, or None."""
    for key in SHARED_BASELINE_KEYS:
        if prior.get(key, SHARED_BASELINE_HISTORY.get(key)) != manifest[key]:
            return key
    return None


def manifest_for(args) -> dict:
    """The experiment's manifest: every setting its results depend on, with the
    task digest and source commit ``main`` puts on ``args`` first. A resumed run
    must match it whole, ``source_commit`` included, so a root resumes under the
    commit that wrote it and a key added here never changes which roots resume;
    a baseline shared across commits is matched on SHARED_BASELINE_KEYS alone."""
    manifest = {"schema_version": 1, "dataset": args.dataset, "dataset_digest": args.digest,
                "tasks": args.tasks, "model": args.model, "reasoning_effort": REASONING_EFFORT,
                "agent": "mini-swe-agent==2.4.6", "branching": "assistant-turn",
                "baseline_rollouts": args.baseline_rollouts, "max_total_method_rollouts": args.max_rollouts,
                "branch_caps": branch_widths(args), "branch_count_mode": args.branch_count_mode,
                "prompts_sha256": load_prompts(args.prompts).sha256,
                "structured_review": bool(getattr(args, "structured_review", False)),
                "max_turns": args.max_turns, "max_output_tokens": args.max_output_tokens,
                "prompt_budget_chars": args.prompt_budget_chars, "command_timeout": args.command_timeout,
                "review_timeout": args.review_timeout,
                # The analyst and reviewer's settings: whether they had an endpoint of their
                # own is recorded, never the endpoint itself or its key.
                "review": {"model": args.review_model or args.model,
                           "own_endpoint": bool(args.review_endpoint),
                           "effort": args.review_effort or REASONING_EFFORT,
                           "output_tokens": args.review_output_tokens or args.max_output_tokens},
                "final_memory_snapshot": memory_snapshot_policy(args),
                "backend": args.backend, "store": str(args.store) if args.store else None,
                "workers": args.workers, "source_commit": args.source_commit,
                "method": args.method, "baseline_root": str(args.baseline_root) if args.baseline_root else None}
    if args.leak_judge:
        # Only a judged run records its judge.
        manifest["review"]["leak_judge"] = {"resamples": args.leak_judge_resamples,
                                            "threshold": args.leak_judge_threshold}
    return manifest


def main(argv=None) -> int:
    args = parse_args(argv)
    args.digest = task_digest(args.tasks_dir, args.tasks)
    args.source_commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    args.preparation = args.output / "preparation.json"
    manifest = manifest_for(args)
    if args.plan:
        print(json.dumps({"manifest": manifest, "example_commands": {
            method: command(args, args.tasks[0], args.output / "plan" / method, method,
                            args.output / "plan/baseline" if method != "baseline" else None)
            for method in METHODS}}, indent=2))
        return 0
    bearer_key(args.model_key_env)   # an unset key fails here, before any stage is launched
    if args.review_key_env:
        bearer_key(args.review_key_env)   # the analyst and reviewer's too
    if args.leak_judge and not os.environ.get(LEAK_KEY_ENV, "").strip():
        raise ValueError(f"--leak-judge needs the TypeSafe key in ${LEAK_KEY_ENV}")
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip():
        raise ValueError("Commit source changes before running")
    if args.baseline_root:
        mismatch = shared_baseline_mismatch(read(args.baseline_root / "manifest-baseline.json"), manifest)
        if mismatch:
            raise ValueError(f"Shared baseline configuration mismatch: {mismatch}")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / f"manifest-{args.method}.json"
    if manifest_path.exists() and read(manifest_path) != manifest:
        raise ValueError("Experiment configuration changed; use a fresh output")
    lock = args.output / f".{args.method}.lock"
    fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.write(fd, str(os.getpid()).encode())
    os.close(fd)
    try:
        write(manifest_path, manifest)
        if args.method == "all":
            write(args.output / "manifest-baseline.json", dict(manifest, method="baseline"))
        write(args.preparation, {"dataset": args.dataset, "dataset_digest": args.digest,
              "tasks": [{"name": tomllib.loads((args.tasks_dir / n / "task.toml").read_text())["task"]["name"], "path": str(args.tasks_dir / n),
                         "task_toml_sha256": hashlib.sha256((args.tasks_dir / n / "task.toml").read_bytes()).hexdigest()}
                        for n in args.tasks]})
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            states = list(pool.map(lambda task: run_task(args, task), args.tasks))
        from examples.qwen.report import build_report
        write(args.output / f"metrics-{args.method}.json", build_report(args.output, args.method))
        return 0 if all(s["complete"] for s in states) else 2
    finally:
        lock.unlink()


if __name__ == "__main__":
    raise SystemExit(main())

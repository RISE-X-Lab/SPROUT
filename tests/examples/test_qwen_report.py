"""``examples.qwen.report`` and ``examples.qwen.compare``: the metrics of experiment roots that ``run`` wrote.

Each test writes the files of a root as ``examples.qwen.run`` and ``sprout.eval.evaluate`` leave them: the
manifest, the stage summaries and receipts, the journals, the selector and critic receipts.
"""

import csv
import json
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from examples.qwen import compare, report
from examples.qwen.run import write
from sprout.eval import policies
from sprout.eval.policies import ModelClient
from tests.ops.trees import (
    artifact_journal,
    attempt_row,
    numbered,
    write_attempt,
    write_manifest,
    write_stage,
    write_summary,
    write_task_state,
)
from tests.support.http import chat_server

USAGE = {"prompt_tokens": 100, "completion_tokens": 20}


def journal(path: Path, *, identity: str, steps: int = 1, usage: dict[str, Any] | None = USAGE) -> Path:
    """A run's journal at ``path``: ``steps`` tool calls and one model response of ``usage``.

    ``identity`` makes the content of each run's journal its own."""
    rows = [{"type": "run.started", "identity": identity}, *[{"type": "tool.started"}] * steps]
    rows.append({"type": "raw.mini-swe-agent", "response": {"usage": usage}})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    return path


# --- usage ------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [
        (
            {"prompt_tokens": 11, "completion_tokens": 7, "prompt_tokens_details": {"cached_tokens": 5}},
            {"input_tokens": 11, "cached_input_tokens": 5, "output_tokens": 7, "cost_usd": None},
        ),
        (
            {
                "input_tokens": 11,
                "output_tokens": 7,
                "input_tokens_details": {"cached_tokens": 5},
                "cost": 0.5,
            },
            {"input_tokens": 11, "cached_input_tokens": 5, "output_tokens": 7, "cost_usd": 0.5},
        ),
        (
            {"input_tokens": 11, "cached_input_tokens": 5, "output_tokens": 7, "cost_usd": 0.0},
            {"input_tokens": 11, "cached_input_tokens": 5, "output_tokens": 7, "cost_usd": 0.0},
        ),
        (None, {"input_tokens": None, "cached_input_tokens": None, "output_tokens": None, "cost_usd": None}),
    ],
    ids=["chat-completions", "responses", "normalized", "unknown"],
)
def test_usage_reads_each_providers_record_and_leaves_an_unknown_value_none(
    raw: dict[str, Any] | None, expected: dict[str, Any]
) -> None:
    """An unknown cost is not a cost of zero."""
    assert report.usage(raw) == expected


# --- the metrics of a phase ---------------------------------------------------------------------


def test_a_phase_counts_each_model_call_and_tool_step_and_an_unknown_usage_makes_its_total_unknown(
    tmp_path: Path,
) -> None:
    """A failed request may have been billed, so its usage stays unknown.

    The ``known_`` totals are lower bounds."""
    rows = [
        {"type": "run.started"},
        {"type": "tool.started"},
        {
            "type": "raw.mini-swe-agent",
            "response": {"usage": {"prompt_tokens": 100, "completion_tokens": 5, "cost": 0.1}},
        },
        {"type": "tool.started"},
        {"type": "model.request.retry"},
    ]
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "trajectory.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n")

    metrics = report.phase_metrics([tmp_path / "out"])

    assert (metrics["model_calls"], metrics["model_overhead_calls"], metrics["actual_new_steps"]) == (2, 1, 2)
    assert metrics["usage"] == {
        "input_tokens": None,
        "known_input_tokens": 100,
        "cached_input_tokens": None,
        "known_cached_input_tokens": 0,
        "output_tokens": None,
        "known_output_tokens": 5,
        "cost_usd": None,
        "known_cost_usd": 0.1,
    }


def test_a_journal_counts_once_and_not_at_all_when_an_earlier_phase_counted_it(tmp_path: Path) -> None:
    """A branch method's output holds a copy of the shared parent's journal. The initial phase counts it."""
    first = journal(tmp_path / "initial" / "parent.jsonl", identity="parent")
    (tmp_path / "extra").mkdir()
    (tmp_path / "extra" / "parent.jsonl").write_bytes(first.read_bytes())
    (tmp_path / "extra" / "copy.jsonl").write_bytes(first.read_bytes())
    journal(tmp_path / "extra" / "child.jsonl", identity="child", steps=3)
    initial = report.phase_metrics([tmp_path / "initial"])

    additional = report.phase_metrics([tmp_path / "extra"], set(initial["journal_hashes"]))

    assert (additional["actual_new_steps"], additional["model_calls"]) == (3, 1)


def actor_call(input_tokens: int) -> dict[str, Any]:
    """The raw record of one actor model call of ``input_tokens`` input tokens and 1 output token."""
    return {
        "type": "raw.mini-swe-agent",
        "response": {"usage": {"prompt_tokens": input_tokens, "completion_tokens": 1}},
    }


def totals(metrics: dict[str, Any]) -> tuple[Any, ...]:
    """The input token total, its known part, the model calls and the steps of a phase."""
    return (
        metrics["usage"]["input_tokens"],
        metrics["usage"]["known_input_tokens"],
        metrics["model_calls"],
        metrics["actual_new_steps"],
    )


def test_a_jsonl_file_without_run_started_is_no_journal(tmp_path: Path) -> None:
    """A native transcript or a resource ledger is no new execution."""
    (tmp_path / "native.jsonl").write_text(
        json.dumps(actor_call(100)) + "\n" + json.dumps({"type": "tool.started"}) + "\n"
    )

    metrics = report.phase_metrics([tmp_path])

    assert totals(metrics) == (0, 0, 0, 0)


def test_a_journal_record_with_unicode_line_separators_stays_one_record(tmp_path: Path) -> None:
    """A model's text keeps U+2028 and U+0085 verbatim. Splitting at them would lose the response's usage."""
    text = "before 1 after\u00852\u0085tail"
    rows = [
        {"type": "run.started"},
        {
            "type": "raw.mini-swe-agent",
            "response": {
                "choices": [{"message": {"content": text}}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 7},
            },
        },
    ]
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n"
    )

    metrics = report.phase_metrics([tmp_path / "out"])

    assert (metrics["model_calls"], metrics["usage"]["input_tokens"], metrics["usage"]["output_tokens"]) == (
        1,
        11,
        7,
    )


def receipt(path: Path, usage: dict[str, Any], *failed: dict[str, Any]) -> None:
    """A selector or critic receipt at ``path``: the tries that failed, then the last, of ``usage``."""
    attempts = [{"usage": tried, "seconds": 1.0} for tried in failed] + [{"usage": usage, "seconds": 2.0}]
    write(path, {"request": {}, "response": {"usage": usage}, "attempts": attempts, "seconds": 3.0})


def test_the_selector_and_critic_receipts_of_a_stage_count_with_each_failed_try(tmp_path: Path) -> None:
    out = tmp_path / "out"
    receipt(out / "meta-request.json", {"prompt_tokens": 10, "completion_tokens": 2})
    receipt(
        out / "entropy" / "step-1.request.json",
        {"prompt_tokens": 30, "completion_tokens": 1},
        {"prompt_tokens": 30},
    )
    receipt(out / "reviews" / "review-001.json", {"prompt_tokens": 50, "completion_tokens": 9})

    metrics = report.phase_metrics([out])

    assert [(call["role"], call["usage"]["input_tokens"]) for call in metrics["calls"]] == [
        ("selector_or_critic", 50),
        ("selector_or_critic", 10),
        ("selector_or_critic", 30),
        ("failed_selector_request", 30),
    ]


def test_a_stage_through_the_run_store_counts_the_journals_and_review_receipts_that_its_output_names(
    tmp_path: Path,
) -> None:
    """The worker keeps them in its artifact directory.

    The summary names the journals, and the round's plan names the review's attempt directory."""
    artifacts = tmp_path / "artifacts"
    parent = journal(artifacts / "job-0" / "attempt-0" / "trajectory.jsonl", identity="parent")
    out = tmp_path / "out"
    write(out / "summary.json", {"attempts": [{"name": "parent", "journal": str(parent), "resolved": False}]})
    review = artifacts / "job-2" / "attempt-2"
    write(out / "plan-round1.json", {"review_job": {"directory": str(review)}})
    receipt(review / "reviews" / "points" / "review-001.json", {"prompt_tokens": 10, "completion_tokens": 2})

    metrics = report.phase_metrics([out])

    assert (metrics["actual_new_steps"], metrics["model_calls"], metrics["model_overhead_calls"]) == (1, 2, 1)
    assert metrics["usage"]["input_tokens"] == 110


def test_a_phase_sums_the_wall_time_of_its_stages_and_names_each_one_that_did_not_complete(
    tmp_path: Path,
) -> None:
    write(
        tmp_path / "attempt-1.stage.json",
        {"complete": True, "started_at": 100.0, "finished_at": 160.0, "seconds": 60.0},
    )
    write(
        tmp_path / "sprout.stage.json",
        {
            "complete": False,
            "started_at": 170.0,
            "finished_at": 200.0,
            "seconds": 30.0,
            "error_type": "RuntimeError",
        },
    )

    metrics = report.phase_metrics([tmp_path / "attempt-1", tmp_path / "sprout"])

    assert (metrics["stage_wall_seconds"], metrics["started_at"], metrics["finished_at"]) == (
        90.0,
        100.0,
        200.0,
    )
    assert metrics["infrastructure_errors"] == [
        {"output": str(tmp_path / "sprout"), "error_type": "RuntimeError"}
    ]


def test_the_usage_of_a_selector_try_that_the_model_client_retried_is_counted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model client's audit file is the receipt. A try without exactly one choice was still billed."""
    monkeypatch.setattr(policies, "time", SimpleNamespace(time=time.time, sleep=lambda seconds: None))
    replies = [
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        {"choices": [{"message": {"content": "ok"}}], "usage": {"prompt_tokens": 11, "completion_tokens": 5}},
    ]
    with chat_server(replies) as server:
        ModelClient(server.url + "/v1", None).complete(
            {"model": "policy"}, tmp_path / "entropy" / "step-1.request.json"
        )

    metrics = report.phase_metrics([tmp_path])

    assert (
        metrics["model_overhead_calls"],
        metrics["usage"]["input_tokens"],
        metrics["usage"]["output_tokens"],
    ) == (2, 18, 8)


# --- the report of an experiment ----------------------------------------------------------------


def baseline_stages(root: Path, task: str, resolved: list[bool]) -> None:
    """The baseline stages of ``task``: one rollout each, with a journal of one step and 120 tokens."""
    for index, verdict in enumerate(resolved, 1):
        output = root / "tasks" / task / "baseline" / f"attempt-{index}"
        write(output / "summary.json", {"attempts": [{"name": "parent", "resolved": verdict}]})
        journal(output / "journal.jsonl", identity=f"{task}-{index}")


def method_stage(root: Path, task: str, method: str, branches: list[bool]) -> None:
    """A branch stage of ``task``: the shared parent, as the evaluator copies it, then ``branches``."""
    rows = [{"name": "parent", "resolved": False}] + [
        {"name": f"b{n:02d}", "resolved": v} for n, v in enumerate(branches, 1)
    ]
    write(root / "tasks" / task / method / "summary.json", {"attempts": rows})


def comparison(rows: list[dict[str, Any]]) -> list[tuple[Any, ...]]:
    return [
        (
            row["method"],
            row["Resolve Rate"],
            row["Recovery"],
            row["Steps"],
            row["Tokens (M)"],
            row["complete"],
        )
        for row in rows
    ]


def test_the_report_of_all_methods_has_the_five_rows_of_the_protocol(tmp_path: Path) -> None:
    """Task a resolves at once, so its methods are skipped. Task b fails, and its third rollout resolves.

    Each row's steps and tokens count the shared initial phase once, then the row's own additional phase."""
    write(
        tmp_path / "manifest-all.json", {"method": "all", "tasks": ["a", "b"], "max_total_method_rollouts": 8}
    )
    baseline_stages(tmp_path, "a", [True, True, True, True])
    baseline_stages(tmp_path, "b", [False, False, True, False])
    method_stage(tmp_path, "b", "bpo", [False])
    method_stage(tmp_path, "b", "shepherd", [True])
    method_stage(tmp_path, "b", "sprout", [False, True])
    skipped = {method: {"status": "skipped_initial_success"} for method in ("bpo", "shepherd", "sprout")}
    write(tmp_path / "tasks" / "a" / "all.task.json", {"methods": skipped})
    write(
        tmp_path / "tasks" / "b" / "all.task.json",
        {"methods": {m: {"status": "complete"} for m in ("bpo", "shepherd", "sprout")}},
    )

    rows = report.build_report(tmp_path, "all")["comparison"]

    assert comparison(rows) == [
        ("pass@1", 0.5, 0.0, 2, 240 / 1e6, True),
        ("pass@4 (baseline)", 1.0, 1.0, 8, 960 / 1e6, True),
        ("entropy-based (BPO)", 0.5, 0.0, 2, 240 / 1e6, True),
        ("Shepherd", 1.0, 1.0, 2, 240 / 1e6, True),
        ("SPROUT", 1.0, 1.0, 2, 240 / 1e6, True),
    ]


def test_a_branch_method_counts_the_shared_parent_once(tmp_path: Path) -> None:
    """Row 0 of each branch method's summary is the shared parent.

    Counted again, it moves the first success."""
    write(
        tmp_path / "manifest-sprout.json",
        {"method": "sprout", "tasks": ["t"], "max_total_method_rollouts": 8},
    )
    baseline_stages(tmp_path, "t", [False, False, False, False])
    method_stage(tmp_path, "t", "sprout", [False, False, False, True])
    write(tmp_path / "tasks" / "t" / "sprout.task.json", {"methods": {"sprout": {"status": "complete"}}})

    (row,) = report.build_report(tmp_path, "sprout")["methods"]["sprout"]["tasks"]

    assert (
        row["phases"]["additional"]["graded_trajectories"],
        row["graded_trajectories"],
        row["first_success"]["ordinal"],
    ) == (
        4,
        5,
        5,
    )
    assert row["success_at"] == {"1": False, "5": True, "8": True}


def test_a_baseline_of_one_rollout_is_reported_as_pass_at_1(tmp_path: Path) -> None:
    write(
        tmp_path / "manifest-baseline.json",
        {"method": "baseline", "tasks": ["a"], "baseline_rollouts": 1, "max_total_method_rollouts": 8},
    )
    baseline_stages(tmp_path, "a", [False])

    result = report.build_report(tmp_path, "baseline")

    assert [row["method"] for row in result["comparison"]] == ["pass@1", "pass@1 (baseline)"]
    assert result["methods"]["baseline"]["tasks"][0]["attempt_budget"] == 1


def test_an_incomplete_cohort_has_no_resolve_rate_but_says_how_much_completed(tmp_path: Path) -> None:
    """Task b has no state: its stage never finished."""
    write(
        tmp_path / "manifest-sprout.json",
        {"method": "sprout", "tasks": ["a", "b"], "max_total_method_rollouts": 8},
    )
    baseline_stages(tmp_path, "a", [False, False, False, False])
    baseline_stages(tmp_path, "b", [False, False, False, False])
    method_stage(tmp_path, "a", "sprout", [True])
    write(tmp_path / "tasks" / "a" / "sprout.task.json", {"methods": {"sprout": {"status": "complete"}}})

    (row,) = report.build_report(tmp_path, "sprout")["comparison"]

    assert (
        row["Resolve Rate"],
        row["Recovery"],
        row["complete"],
        row["completed_tasks"],
        row["total_tasks"],
    ) == (None, None, False, 1, 2)


def baseline_attempts(root: Path, task: str, verdicts: list[bool]) -> None:
    """The complete baseline attempts of ``task``, each with a receipt and a journal of two steps."""
    for number, resolved in enumerate(verdicts, 1):
        write_attempt(
            root / "tasks" / task, number, artifacts=root.parent / "artifacts", resolved=resolved, turns=2
        )


def branch_stage(root: Path, task: str, method: str, verdicts: list[bool]) -> None:
    """A complete branch stage and its receipt: the parent row, then a one-step branch for each verdict."""
    parent = root / "tasks" / task / "baseline" / "attempt-1" / "summary.json"
    rows = [json.loads(parent.read_text())["attempts"][0]]
    for number, resolved in enumerate(verdicts, 1):
        journal = artifact_journal(
            root.parent / "artifacts",
            "job-%s-%s-%s-%d" % (root.name, task, method, number),
            turns=numbered(1),
        )
        rows.append(attempt_row("r1b%d" % number, journal=journal, resolved=resolved, round_no=1))
    output = root / "tasks" / task / method
    write_summary(output, attempts=rows, task="datacurve/" + task, method=method)
    write_stage(output, state="complete", task=task, method=method)


def all_methods_root(tmp_path: Path) -> Path:
    """Alpha fails attempt-1, attempt-2 resolves it, and sprout alone recovers it. Beta resolves attempt-1."""
    root = tmp_path / "all"
    write_manifest(root, method="all", tasks=["alpha", "beta"])
    baseline_attempts(root, "alpha", [False, True, False, False])
    baseline_attempts(root, "beta", [True, False, False, False])
    branch_stage(root, "alpha", "sprout", [False, True])
    branch_stage(root, "alpha", "bpo", [False])
    branch_stage(root, "alpha", "shepherd", [False, False])
    complete = {"status": "complete"}
    write_task_state(
        root, "alpha", complete=True, methods={"sprout": complete, "bpo": complete, "shepherd": complete}
    )
    skipped = {"status": "skipped_initial_success"}
    write_task_state(
        root, "beta", complete=True, methods={"sprout": skipped, "bpo": skipped, "shepherd": skipped}
    )
    return root


def success(result: dict[str, Any]) -> dict[str, tuple[Any, ...]]:
    """Success@1, @5 and @8 of each method of the report ``result``."""
    return {
        method: (m["Success@1"], m["Success@5"], m["Success@8"]) for method, m in result["methods"].items()
    }


def test_success_at_k_counts_the_tasks_resolved_within_the_first_k_trajectories(tmp_path: Path) -> None:
    result = report.build_report(all_methods_root(tmp_path), "all")

    assert success(result) == {
        "pass@1": (0.5, 0.5, 0.5),
        "baseline": (0.5, 1.0, 1.0),
        "bpo": (0.5, 0.5, 0.5),
        "shepherd": (0.5, 0.5, 0.5),
        "sprout": (0.5, 1.0, 1.0),
    }


def test_a_branch_methods_time_to_first_success_leaves_out_the_baseline_attempts_after_the_first(
    tmp_path: Path,
) -> None:
    """Its logical time is attempt-1's stage time plus the time from its own stage's start to the success."""
    root = all_methods_root(tmp_path)

    [alpha, _] = report.build_report(root, "all")["methods"]["sprout"]["tasks"]

    assert alpha["first_success"] == {
        "ordinal": 3,
        "validated_at": 1790800120.0,
        "logical_method_seconds": 600.0 + 1790800120.0 - 1790799500.0,
        "calendar_seconds": 1790800120.0 - 1790799500.0,
    }


# --- comparing roots ------------------------------------------------------------------------------


SHARED = {
    "dataset": "deepswe",
    "dataset_digest": "sha256:x",
    "tasks": ["a"],
    "model": "m",
    "reasoning_effort": "high",
    "max_turns": 300,
    "max_output_tokens": 64000,
    "command_timeout": 60,
    "runstore_url": "http://runstore.test:18110",
    "profile": "mini",
    "grade_profile": "grade",
    "baseline_rollouts": 4,
}
METHODS = ("bpo", "shepherd", "sprout")


def roots(tmp_path: Path, baseline: dict[str, Any], **overrides: dict[str, Any]) -> dict[str, Path]:
    """Four run roots of task a: a baseline whose third rollout resolves, and a root on it for each method.

    ``overrides`` changes the manifest of one method."""
    paths = {method: tmp_path / method for method in ("baseline", *METHODS)}
    write(paths["baseline"] / "manifest-baseline.json", {**baseline, "method": "baseline"})
    baseline_stages(paths["baseline"], "a", [False, False, True, False])
    for method in METHODS:
        manifest = {
            **baseline,
            "method": method,
            "max_total_method_rollouts": 8,
            "baseline_root": str(paths["baseline"]),
        }
        write(paths[method] / f"manifest-{method}.json", {**manifest, **overrides.get(method, {})})
        method_stage(paths[method], "a", method, [True])
        write(
            paths[method] / "tasks" / "a" / f"{method}.task.json",
            {"methods": {method: {"status": "complete"}}},
        )
    return paths


@pytest.mark.parametrize(
    "key, value",
    [
        ("dataset", "terminalbench21"),
        ("dataset_digest", "sha256:other"),
        ("tasks", ["a", "b"]),
        ("model", "other"),
        ("reasoning_effort", "medium"),
        ("max_turns", 100),
        ("max_output_tokens", 32000),
        ("command_timeout", 600),
        ("runstore_url", "http://other.test:18110"),
        ("profile", "mini-microvm"),
        ("grade_profile", "grade-microvm"),
        ("baseline_rollouts", 1),
        ("image_user", False),
        ("image_env", False),
    ],
)
def test_roots_that_differ_on_a_shared_baseline_setting_are_not_compared(
    tmp_path: Path, key: str, value: object
) -> None:
    with pytest.raises(ValueError, match=f"^Incomparable bpo input: {key}$"):
        compare.compare(roots(tmp_path, SHARED, bpo={key: value}))


LEGACY = {key: value for key, value in SHARED.items() if key != "command_timeout"}


@pytest.mark.parametrize("sprout", [{"command_timeout": 60}, {}], ids=["names-60-s", "names-none"])
def test_a_baseline_that_predates_the_command_timeout_ran_with_60_second_commands(
    tmp_path: Path, sprout: dict[str, Any]
) -> None:
    """Both manifests come from disk, so a key that either side lacks takes its historical value."""
    assert compare.compare(roots(tmp_path, LEGACY, sprout=sprout))["complete"] is True


def test_a_baseline_that_predates_the_command_timeout_is_not_compared_with_other_commands(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="^Incomparable sprout input: command_timeout$"):
        compare.compare(roots(tmp_path, LEGACY, sprout={"command_timeout": 600}))


def test_a_method_root_on_another_baseline_is_not_compared(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="^shepherd did not use the selected shared baseline$"):
        compare.compare(roots(tmp_path, SHARED, shepherd={"baseline_root": str(tmp_path / "elsewhere")}))


def test_the_comparison_is_written_as_json_and_as_one_csv_row_per_method(tmp_path: Path) -> None:
    paths = roots(tmp_path, SHARED)
    options = [argument for method, path in paths.items() for argument in (f"--{method}", str(path))]

    compare.main([*options, "--output", str(tmp_path / "comparison")])

    rows = json.loads((tmp_path / "comparison" / "comparison.json").read_text())["comparison"]
    with (tmp_path / "comparison" / "comparison.csv").open() as stream:
        table = list(csv.reader(stream))
    assert [row["method"] for row in rows] == [
        "pass@1",
        "pass@4 (baseline)",
        "entropy-based (BPO)",
        "Shepherd",
        "SPROUT",
    ]
    assert table[0] == [
        "method",
        "Resolve Rate",
        "Recovery",
        "Steps",
        "Tokens (M)",
        "complete",
        "completed_tasks",
        "total_tasks",
    ]
    assert [line[0] for line in table[1:]] == [row["method"] for row in rows]

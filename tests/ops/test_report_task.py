"""``python -m ops.report task``: a task's fork tree, from its baseline attempts to its rounds and branches.

The evaluations are the Run Store's, except where a test names a Harbor trial.
"""

from __future__ import annotations

import json
import os

from ops.report import cli
from tests.ops.trees import (
    actor,
    analyst_report,
    artifact_journal,
    attempt_row,
    numbered,
    review_attempt,
    reviewer,
    selected_branch,
    write_attempt,
    write_journal,
    write_lock,
    write_manifest,
    write_plan,
    write_retained,
    write_review_receipt,
    write_stage,
    write_summary,
    write_task_state,
)

BRANCH_COMMANDS = ("cd /app && go test ./...", "set -e; cd /app; git commit -am fix")
FIX = selected_branch("r1b1-fix", step=3, snapshot_id="par-0003", commands=BRANCH_COMMANDS)
TWO_ATTEMPTS = [
    review_attempt(
        1, status="validation_error", response="BAD PLAN", error="Invalid JSON inside branch-plan block"
    ),
    review_attempt(2, status="validated", response="GOOD PLAN"),
]


def baseline_of_alpha(tmp_path):
    """The baseline root of task alpha: attempt-1 failed with 3 retained points, attempt-2 resolved."""
    root = tmp_path / "exp" / "baseline"
    write_manifest(root, method="baseline", tasks=["alpha"])
    write_lock(root, method="baseline", pid=os.getpid())
    write_retained(
        write_attempt(root / "tasks" / "alpha", 1, artifacts=tmp_path / "artifacts", resolved=False)
    )
    write_attempt(
        root / "tasks" / "alpha", 2, artifacts=tmp_path / "artifacts", resolved=True, seconds=1200.0
    )
    return root


def review_job(tmp_path, reviews):
    """The review job's attempt directory: an analyst receipt, then one receipt for each (text, seconds)."""
    directory = tmp_path / "artifacts" / "job-review" / "attempt-job-review"
    write_review_receipt(
        directory / "reviews" / "analyst-1" / "review-001.json",
        kind="analyst",
        text="ANALYSIS",
        seconds=264.0,
    )
    for number, (text, seconds) in enumerate(reviews, 1):
        path = directory / "reviews" / "points" / ("review-%03d.json" % number)
        write_review_receipt(path, kind="reviewer", text=text, seconds=seconds)
    return directory


def sprout_of_alpha(
    tmp_path,
    *,
    attempts=TWO_ATTEMPTS,
    reviews=(("BAD PLAN", 702.0), ("GOOD PLAN", 450.0)),
    selected=(FIX,),
    name="sprout",
    **plan,
):
    """A sprout evaluation of alpha in the experiment's sprout root: one round from attempt-1 of the baseline.

    A selected r1b1-fix gets a resolved branch journal of a reviewer turn and an actor turn."""
    baseline = tmp_path / "exp" / "baseline"
    root = tmp_path / "exp" / name
    write_manifest(root, method="sprout", tasks=["alpha"], baseline_root=str(baseline))
    output = root / "tasks" / "alpha" / "sprout"
    parent = baseline / "tasks" / "alpha" / "baseline" / "attempt-1"
    rows = [json.loads((parent / "summary.json").read_text())["attempts"][0]]
    if selected:
        branch = artifact_journal(
            tmp_path / "artifacts",
            "job-" + name,
            turns=[reviewer(*BRANCH_COMMANDS), actor("echo done")],
            sandbox="brn",
        )
        rows.append(attempt_row("r1b1-fix", journal=branch, resolved=True, round_no=1))
    write_summary(output, attempts=rows, parent_output=str(parent), stop_reason=plan.get("validation_error"))
    directory = review_job(tmp_path, reviews)
    write_plan(
        output,
        1,
        reports={"parent": analyst_report(steps=4, candidates=[3, 2])},
        available_steps={"parent": [1, 2, 3, 4]},
        review_attempts=list(attempts),
        selected_branches=list(selected),
        review_job={"job_id": "job-review", "attempt_id": "attempt-job-review", "directory": str(directory)},
        **plan,
    )
    write_stage(output, state="complete", method="sprout")
    write_task_state(root, "alpha", complete=True, methods={"sprout": {"status": "complete"}})
    return output


def task(capsys, *argv):
    assert cli.main(["task", *map(str, argv)]) == 0
    return capsys.readouterr().out


def lines_from(text, prefix):
    """The lines of ``text`` from the first one that starts with ``prefix`` to the end."""
    lines = text.splitlines()
    return lines[next((index for index, line in enumerate(lines) if line.startswith(prefix)), len(lines)) :]


def lines_with(text, needle):
    """The lines of ``text`` that contain ``needle``."""
    return [line for line in text.splitlines() if needle in line]


def test_the_tree_lists_each_baseline_attempt_with_its_verdict_turns_steps_and_snapshots(tmp_path, capsys):
    baseline = baseline_of_alpha(tmp_path)

    text = task(capsys, baseline.parent, "alpha")

    assert text.splitlines()[:4] == [
        "task datacurve/alpha · deepswe · m · podman /path/to/store",
        "baseline %s/tasks/alpha/baseline   task: no task.json" % baseline,
        "├── attempt-1  complete  completed  FAIL  reward 0.0     4 turns    4 steps  10.0 min"
        "    4 snapshots  final par-0004  PARENT · branchable: 3 points retained",
        "└── attempt-2  complete  completed  PASS  reward 1.0     4 turns    4 steps  20.0 min"
        "    4 snapshots  final sb2-0004",
    ]


def test_the_tree_shows_a_round_with_its_analyst_its_reviewer_attempts_and_its_branch(tmp_path, capsys):
    baseline_of_alpha(tmp_path)
    output = sprout_of_alpha(tmp_path)

    text = task(capsys, tmp_path / "exp", "alpha")

    assert lines_from(text, "sprout ") == [
        "sprout %s   stage complete · resolved · guidance assistant-turn · caps [4, 3]"
        " · rounds 2 (1 run: resolved in round 1)" % output,
        "└── round 1  limit 4 · adaptive · 1 branches selected",
        "    ├── analyst parent  [review-001.json 4.4 min]  4 steps, 4 available (1..4)",
        "    │   ├── failure: FAILURE REASON",
        "    │   ├── lesson:  LESSON",
        "    │   ├── salvage: SALVAGE",
        "    │   └── candidates: steps 3, 2",
        "    ├── reviewer attempt 1  validation_error  [review-001.json 11.7 min]"
        "  Invalid JSON inside branch-plan block",
        "    ├── reviewer attempt 2  validated         [review-002.json 7.5 min]  1 branches",
        "    └── r1b1-fix  parent@3 (turn 3)  snapshot par-0003  cut t3",
        "        ├── assistant-turn (reviewer) 2 bash calls: go test ./... | git commit -am fix",
        "        └── trial  graded  completed  PASS  reward 1.0     2 turns (1 reviewer + 1 actor)"
        "    3 steps   2.0 min    2 snapshots  final brn-0002  sandbox brn",
    ]


def test_a_reviewer_attempt_gets_the_receipt_that_holds_its_response_past_a_retried_request(tmp_path, capsys):
    """A truncated reply is retried in a receipt of its own, so the receipts outnumber the attempts."""
    baseline_of_alpha(tmp_path)
    reviews = (("BAD PLAN", 702.0), ("TRUNCATED", 30.0), ("GOOD PLAN", 450.0))
    output = sprout_of_alpha(tmp_path, reviews=reviews)

    text = task(capsys, output)

    assert lines_with(text, "reviewer attempt") == [
        "    ├── reviewer attempt 1  validation_error  [review-001.json 11.7 min]"
        "  Invalid JSON inside branch-plan block",
        "    ├── reviewer attempt 2  validated         [review-003.json 7.5 min]  1 branches",
    ]


def test_a_reviewer_request_in_flight_is_shown_in_flight(tmp_path, capsys):
    baseline_of_alpha(tmp_path)
    attempts = [TWO_ATTEMPTS[0], review_attempt(2, status="requesting")]
    output = sprout_of_alpha(
        tmp_path, attempts=attempts, reviews=(("BAD PLAN", 702.0), (None, None)), selected=()
    )

    text = task(capsys, output)

    assert lines_from(text, "    └── reviewer attempt 2") == [
        "    └── reviewer attempt 2  requesting        [review-002.json in flight]  "
    ]


def test_a_failed_review_says_that_it_selected_no_branch_and_why(tmp_path, capsys):
    baseline_of_alpha(tmp_path)
    output = sprout_of_alpha(
        tmp_path, attempts=TWO_ATTEMPTS[:1], selected=(), validation_error="reviewer failed: Expecting value"
    )

    text = task(capsys, output)

    assert lines_with(text, "reviewer failed") == [
        "sprout %s   stage complete · unresolved · guidance assistant-turn · caps [4, 3] · rounds 2 (1 run) "
        "· stopped: reviewer failed: Expecting value" % output,
        "    └── reviewer selected no branches: reviewer failed: Expecting value",
    ]


def test_an_errored_attempt_shows_its_cause_on_one_clipped_line_and_whole_in_json(tmp_path, capsys):
    baseline = baseline_of_alpha(tmp_path)
    cause = "RuntimeError: Official Harbor trial failed: " + "x" * 2000
    write_stage(
        baseline / "tasks" / "alpha" / "baseline" / "attempt-3",
        state="errored",
        log="Traceback\n" + cause + "\n",
    )
    error = "Stage exited 1; see attempt-3.log · cause: " + cause

    text = task(capsys, baseline.parent, "alpha")
    document = json.loads(task(capsys, baseline.parent, "alpha", "--format", "json"))

    assert (
        text.splitlines()[4] == "└── attempt-3  errored   %s… · 10.0 min · no trial directory" % error[:200]
    )
    assert document["baseline"]["attempts"][2]["error"] == error


def test_the_task_directory_and_the_evaluate_output_give_the_tree_of_the_experiment_root(tmp_path, capsys):
    baseline_of_alpha(tmp_path)
    output = sprout_of_alpha(tmp_path)
    expected = task(capsys, tmp_path / "exp", "alpha")

    from_task_directory = task(capsys, output.parent)
    from_output = task(capsys, output)

    assert (from_task_directory, from_output) == (expected, expected)


def test_with_attaches_an_evaluation_that_branched_from_attempt_1_and_skips_the_others(tmp_path, capsys):
    baseline_of_alpha(tmp_path)
    other = sprout_of_alpha(tmp_path, name="other")
    stranger = tmp_path / "stranger"
    elsewhere = write_journal(tmp_path / "elsewhere.jsonl", turns=numbered(1))
    write_summary(stranger, attempts=[attempt_row("parent", journal=elsewhere, resolved=False)])
    plain = tmp_path / "plain"
    plain.mkdir()

    text = task(
        capsys, tmp_path / "exp" / "baseline", "alpha", "--with", other, "--with", stranger, "--with", plain
    )

    assert (lines_from(text, "sprout ")[0], lines_from(text, "--with")) == (
        "sprout %s   complete · resolved · guidance assistant-turn · caps [4, 3]"
        " · rounds 2 (1 run: resolved in round 1)" % other,
        [
            "--with %s does not branch from this task's attempt-1: skipped" % stranger,
            "--with %s is not an evaluate output: skipped" % plain,
        ],
    )


def test_mermaid_hangs_the_branch_off_its_step_of_the_parent(tmp_path, capsys):
    baseline_of_alpha(tmp_path)
    sprout_of_alpha(tmp_path)

    text = task(capsys, tmp_path / "exp", "alpha", "--format", "mermaid")

    assert text.splitlines() == [
        "graph LR",
        '  parent(["parent · attempt-1"]) --> parent_s3["step 3 (turn 3)<br/>par-0003"]',
        '  parent_s3 --> parent_end["step 4 · final<br/>FAIL reward 0.0<br/>par-0004"]',
        '  parent_s3 --> r1b1_fix["r1b1-fix<br/>round 1 · assistant-turn<br/>2 turns · 3 steps · 2.0 min'
        '<br/>PASS reward 1.0"]',
        "  subgraph independent baseline attempts",
        '    attempt_2["attempt-2<br/>4 turns · 4 steps · 20.0 min<br/>PASS reward 1.0"]',
        "  end",
        "  classDef pass fill:#dff5df,stroke:#2e7d32; classDef fail fill:#fbe0e0,stroke:#c62828; "
        "classDef err fill:#eeeeee,stroke:#616161,stroke-dasharray:4 2;",
        "  class parent_end fail; class r1b1_fix,attempt_2 pass;",
    ]


def test_an_experiment_without_a_sprout_root_says_how_to_attach_an_evaluation(tmp_path, capsys):
    baseline = baseline_of_alpha(tmp_path)

    text = task(capsys, baseline.parent, "alpha")

    assert text.splitlines()[-1] == (
        "sprout   no sprout stage under this root"
        " (no sprout/ method root; pass --with <evaluate output> to attach one)"
    )


def test_a_branch_method_that_skipped_the_task_says_that_attempt_1_resolved(tmp_path, capsys):
    baseline_of_alpha(tmp_path)
    root = tmp_path / "exp" / "sprout"
    write_manifest(root, method="sprout", tasks=["alpha"], baseline_root=str(tmp_path / "exp" / "baseline"))
    write_task_state(
        root,
        "alpha",
        complete=True,
        initial_resolved=True,
        methods={"sprout": {"status": "skipped_initial_success"}},
    )

    text = task(capsys, tmp_path / "exp", "alpha")

    assert text.splitlines()[-1] == "sprout   skipped (attempt-1 resolved)"


def test_an_unlaunched_attempt_says_that_this_runner_never_launched_it(tmp_path, capsys):
    baseline = baseline_of_alpha(tmp_path)
    write_stage(
        baseline / "tasks" / "alpha" / "baseline" / "attempt-3", state="unlaunched", sentinel="handed over"
    )

    text = task(capsys, baseline.parent, "alpha")

    assert text.splitlines()[4] == (
        "└── attempt-3  unlaunched   receipt only, no log or output: this runner never launched it"
        " · sentinel: handed over"
    )


def running_attempt_3(baseline):
    """Attempt 3 of alpha, live: a Harbor trial whose fifth turn waits for its model call."""
    output = baseline / "tasks" / "alpha" / "baseline" / "attempt-3"
    write_stage(output, state="running")
    write_summary(output, attempts=[], method="baseline", complete=False)
    journal = output / "trials" / "alpha-parent" / "agent" / "trajectory.jsonl"
    write_journal(journal, turns=numbered(4), status=None, open_turn=True)
    return output


def test_a_running_attempt_with_a_harbor_trial_shows_its_turn_and_last_event(tmp_path, capsys):
    baseline = baseline_of_alpha(tmp_path)
    running_attempt_3(baseline)

    text = task(capsys, baseline.parent, "alpha")

    assert (
        text.splitlines()[4] == "└── attempt-3  running   turn 5 (4 closed, 4 steps)   actor running"
        "   last event 2026-09-30T20:00:43Z"
    )


def test_the_task_directory_and_a_running_attempt_give_the_attempts_of_the_experiment_root(tmp_path, capsys):
    """Before its first method stage, a task directory holds only baseline/attempt-N files."""
    baseline = baseline_of_alpha(tmp_path)
    attempt = running_attempt_3(baseline)
    expected = task(capsys, baseline.parent, "alpha").splitlines()[:5]

    from_task_directory = task(capsys, attempt.parents[1]).splitlines()[:5]
    from_attempt = task(capsys, attempt).splitlines()[:5]

    assert (from_task_directory, from_attempt) == (expected, expected)

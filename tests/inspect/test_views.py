"""runs / task / snapshots / list over synthetic run trees, plus the real trimmed fixture."""

from __future__ import annotations

import json
import os
import shutil

from sprout.inspect import cli
from sprout.inspect.layout import (
    load_evaluation, load_grade, load_stage, load_trial, read_dict, detect,
)

from tests.inspect.conftest import (
    FIXTURES, SENTINEL, WORKTREE_ERROR, make_evaluation, make_experiment, make_journal, make_sprout_output,
    metrics_document, simple_turns, write_json,
)


def run(capsys, *argv) -> str:
    assert cli.main(list(argv)) == 0
    return capsys.readouterr().out


def test_runs_counts_tasks_attempts_first_attempt_rate_and_live_turns(experiment, capsys):
    text = run(capsys, "runs", str(experiment))
    assert "experiment %s   driver.log: baseline start 2026-09-30T20:18:20+00:00" % experiment in text
    assert "== baseline   4 tasks · deepswe · m · podman %s/store" % experiment in text
    assert "runner pid %d alive" % os.getpid() in text and "metrics-baseline.json: not written yet" in text
    assert "tasks       1 complete · 1 errored · 1 in progress · 1 not started" in text
    assert "attempts    7 finished (4 resolved, 57.1%) · 1 running · 1 errored · mean" in text
    assert "attempt-1   3 finished, 1 resolved (33.3%) · 2 failed parents retained (retained-checkpoints.json)" in text
    assert "resolved    alpha beta gamma" in text
    assert "in progress gamma   attempt-3   turn 5 (4 closed, 4 steps)   actor running   last event 2026-09-30T20:" in text
    assert "1× %s   (attempt-2: 1)" % WORKTREE_ERROR in text
    assert "== sprout " in text and "branch      1 skipped (attempt-1 resolved) · recovered 1/2 failed parents" in text
    document = json.loads(run(capsys, "runs", str(experiment), "--format", "json"))
    baseline = document["methods"][0]
    assert baseline["task_counts"] == {"complete": 1, "errored": 1, "in-progress": 1, "stale": 0, "unlaunched": 0,
                                       "not-started": 1}
    assert baseline["attempts"]["finished"] == 7 and baseline["live"][0]["turns_started"] == 5
    assert baseline["first_attempt"] == {"finished": 3, "resolved": 1, "failed": 2, "retained": 2}
    assert baseline["backend"] == "podman" and baseline["backend_source"] == "manifest" and baseline["unlaunched"] == []


def test_a_receipt_without_a_log_is_unlaunched_not_running_and_a_task_error_is_grouped(tmp_path, capsys):
    """run.py's stage() opens <stage>.log before it launches anything, so a receipt with no log beside
    it (written by hand to hand a task to another root) was never launched; when this runner reaches
    it, it refuses the retained receipt and the task errors without any stage erroring."""
    root = tmp_path / "exp"
    retained = ("Incomplete stage retained at %s; choose a fresh experiment root to retry"
                % (root / "baseline/tasks/beta/baseline/attempt-1"))
    make_experiment(root, lock_pid=os.getpid(), with_sprout=False, tasks={
        "alpha": {"attempts": [("unlaunched", None)]},                              # not reached yet: no task.json
        "beta": {"attempts": [("unlaunched", None)], "task_error": retained},        # reached and refused
        "gamma": {"attempts": [("complete", True), ("live", None)]}})
    text = run(capsys, "runs", str(root))
    assert "tasks       0 complete · 1 errored · 1 in progress · 1 unlaunched · 0 not started" in text
    assert "attempts    1 finished (1 resolved, 100.0%) · 1 running · 0 errored · 2 unlaunched · mean 10.0 min" in text
    assert "in progress gamma   attempt-2   turn 5 (4 closed, 4 steps)   actor running" in text
    assert "in progress alpha" not in text and "in progress beta" not in text and "no journal yet" not in text
    assert "unlaunched  2 receipts without a log or output directory (this runner never launched them)" in text
    assert "  2× sentinel: %s" % SENTINEL in text and "               alpha/attempt-1 beta/attempt-1" in text
    assert ("errors        1× Incomplete stage retained at <path>; choose a fresh experiment root to retry"
            "   (baseline.task.json: 1)") in text
    document = json.loads(run(capsys, "runs", str(root), "--format", "json"))
    report = document["methods"][0]
    assert report["unlaunched"] == [{"task": "alpha", "stage": "attempt-1", "extras": {"sentinel": SENTINEL}},
                                    {"task": "beta", "stage": "attempt-1", "extras": {"sentinel": SENTINEL}}]
    assert report["task_counts"]["unlaunched"] == 1 and report["attempts"]["running"] == 1 and len(report["live"]) == 1
    assert report["errors"][0]["cause"].startswith("Incomplete stage retained at <path>;")
    tree = run(capsys, "task", str(root), "beta")
    assert "task: errored (Incomplete stage retained at %s" % (root / "baseline/tasks/beta/baseline/attempt-1") in tree
    assert "└── attempt-1  unlaunched   receipt only, no log or output: this runner never launched it · sentinel: %s" % SENTINEL in tree
    assert "running" not in tree and "unlaunched: receipt only" in run(capsys, "task", str(root), "beta", "--format", "mermaid")
    rows = [line.split() for line in run(capsys, "list", str(tmp_path)).splitlines()]
    assert ["experiment", "baseline", "3", "0", "1", "1", "0", "alive", "exp"] in rows
    stage = load_stage(root / "baseline/tasks/alpha/baseline/attempt-1.stage.json", lock_alive=True)
    assert (stage.state, stage.error, stage.cause, stage.extras) == ("unlaunched", None, None, {"sentinel": SENTINEL})


def test_an_exit_2_stage_log_names_the_stop_reason_not_the_closing_brace(tmp_path):
    """methods.evaluate prints its state as indented JSON last and exits 2 when incomplete."""
    receipt = tmp_path / "sprout.stage.json"
    write_json(receipt, {"task": "alpha", "method": "sprout", "started_at": 1.0, "complete": False, "exit_code": 2,
                         "error": "Stage exited 2; see sprout.log", "finished_at": 9.0, "seconds": 8.0})
    state = {"task": "datacurve/alpha", "method": "sprout", "complete": False, "resolved": False,
             "stop_reason": "reviewer failed: Expecting value: line 1 column 1 (char 0)", "branch_schedule_complete": False}
    log = tmp_path / "sprout.log"
    log.write_text("harbor noise\n}\n" + json.dumps(state, indent=2) + "\n")
    assert load_stage(receipt, lock_alive=False).cause == "stop_reason: reviewer failed: Expecting value: line 1 column 1 (char 0)"
    del state["stop_reason"]
    log.write_text("Traceback (most recent call last):\n  File x\nValueError: no graded continuation\ncleanup done\n"
                   + json.dumps(state, indent=2) + "\n")
    assert load_stage(receipt, lock_alive=False).cause == "ValueError: no graded continuation"   # the dump names no cause
    log.write_text("only noise\n{ not json\n}\n")
    assert load_stage(receipt, lock_alive=False).cause == "}"                                     # nothing better to say
def test_runs_marks_a_dead_runner_as_stale_and_prints_metrics_when_present(tmp_path, monkeypatch, capsys):
    root = make_experiment(tmp_path / "exp", lock_pid=4242, with_sprout=False,
                           tasks={"alpha": {"attempts": [("complete", False), ("live", None)]}})

    def dead(pid, signal):
        raise ProcessLookupError(pid)
    monkeypatch.setattr(os, "kill", dead)
    text = run(capsys, "runs", str(root))
    assert "runner pid 4242 dead" in text and "0 in progress · 1 stale · 0 not started" in text
    assert "1 stale (runner dead; these stages will never finish)" in text
    rows = [line.split() for line in run(capsys, "list", str(tmp_path)).splitlines()]
    assert rows[0] == ["kind", "methods", "tasks", "complete", "errored", "running", "stale", "runner", "path"]
    assert ["experiment", "baseline", "1", "0", "0", "0", "1", "dead", "exp"] in rows
    base = root / "baseline"
    write_json(base / "metrics-baseline.json", metrics_document(read_dict(base / "manifest-baseline.json"), "baseline"))
    text = run(capsys, "runs", str(root))
    assert "metrics-baseline.json: written" in text
    assert "metrics-baseline.json  pass@1: completed 1/1 · Acc 0.0% · Success@1 0.0% @5 0.0% @8 0.0% · recovery 0/1 (0.0%) · graded 2 · positive 0" in text
    assert "metrics-baseline.json  baseline: completed 1/1 · Acc 0.0% · Success@1 0.0% @5 100.0% @8 100.0% · recovery 1/1 (100.0%)" in text
    assert "comparison   baseline: Resolve Rate 1.00 · Recovery 1.00 · Steps 176 · Tokens (M) 13.48 · complete" in text


def test_runs_on_the_sprout_method_root_counts_branch_attempts_and_recovery(tmp_path, capsys):
    root = make_experiment(tmp_path / "exp", tasks={
        "alpha": {"attempts": [("complete", False), ("complete", False), ("complete", False), ("complete", False)],
                  "sprout": "complete"},
        "beta": {"attempts": [("complete", True), ("complete", True), ("complete", True), ("complete", True)],
                 "sprout": "skipped"}})
    manifest = root / "sprout" / "manifest-sprout.json"
    body = read_dict(manifest)
    body["baseline_root"] = "/old/ash-runs/exp/baseline"            # the root moved since the manifest was written
    write_json(manifest, body)
    text = run(capsys, "runs", str(root / "sprout"))
    assert text.startswith("== sprout     2 tasks") and "no lock (runner finished)" in text
    assert "attempts    1 finished (1 resolved, 100.0%) · 0 running · 0 errored · mean 2.0 min · max 2.0 min" in text
    assert "attempt-1   2 finished, 1 resolved (50.0%) · 1 failed parents retained" in text
    assert "branch      1 skipped (attempt-1 resolved) · recovered 1/1 failed parents" in text
    assert "resolved    alpha" in text


def test_runs_on_a_gate_root_and_a_method_root_directly(tmp_path, capsys):
    gate = tmp_path / "gate"
    gate.mkdir()
    (gate / "gate.jsonl").write_text(json.dumps({"task": "t", "mode": "oracle", "ok": True}) + "\n"
                                     + json.dumps({"task": "t", "mode": "nop", "ok": False, "error": "x"}) + "\n")
    write_json(gate / "summary.json", {"tasks": 1, "modes": ["oracle", "nop"], "ok": ["t/oracle"], "failures": ["t/nop"]})
    assert run(capsys, "runs", str(gate)).strip() == "gate root %s: 2 checks in gate.jsonl, ok 1, failures 1 (see summary.json)" % gate
    root = make_experiment(tmp_path / "exp", with_sprout=False, tasks={"alpha": {"attempts": [("complete", True)]}})
    text = run(capsys, "runs", str(root / "baseline"))
    assert text.startswith("== baseline   1 tasks") and "experiment" not in text and "resolved    alpha" in text


def test_task_tree_shows_attempts_rounds_reviewer_attempts_and_branches(experiment, tmp_path, capsys):
    text = run(capsys, "task", str(experiment), "alpha")
    assert text.splitlines()[0] == "task datacurve/alpha · deepswe · m · podman %s/store" % experiment
    assert "baseline %s/baseline/tasks/alpha/baseline   task: complete" % experiment in text
    assert ("├── attempt-1  complete  completed  FAIL  reward 0.0  f2p 9/20 p2p 3/3     5 turns    5 steps  10.0 min"
            "    5 snapshots  final par-0005  PARENT · branchable: 3 points retained") in text
    assert "├── attempt-2  complete  completed  PASS  reward 1.0  f2p 20/20 p2p 3/3" in text
    assert "└── attempt-4  complete  completed  FAIL" in text
    assert ("sprout %s/sprout/tasks/alpha/sprout   stage complete · resolved · guidance assistant-turn · caps [4, 3]"
            " · rounds 2 (1 run: resolved in round 1)" % experiment) in text
    assert "└── round 1  limit 4 · adaptive · 1 branches selected" in text
    assert "    ├── analyst parent  [review-001.json 4.4 min]  4 steps, 4 available (1..4)" in text
    assert "    │   ├── failure: FAILURE REASON" in text and "    │   └── candidates: steps 3, 2" in text
    assert "    ├── reviewer attempt 1  validation_error  [review-002.json 11.7 min]  Invalid JSON inside branch-plan block" in text
    assert "    ├── reviewer attempt 2  validated         [review-003.json 7.5 min]  1 branches" in text
    assert "    └── r1b1-fix  parent@3 (turn 3)  snapshot par-0003  cut t3" in text
    assert "        ├── assistant-turn (reviewer) 2 bash calls: go test ./... | git commit -am fix" in text
    assert "        └── trial  graded  completed  PASS  reward 1.0  f2p 20/20 p2p 3/3     2 turns (1 reviewer + 1 actor)    3 steps   2.0 min    2 snapshots  final brn-0002  sandbox brn" in text
    # the errored attempt of another task, with its cause from the log
    beta = run(capsys, "task", str(experiment), "beta")
    assert "task: errored (Stage exited 1; see attempt-2.log)" in beta
    assert "└── attempt-2  errored   Stage exited 1; see attempt-2.log · cause: %s · 0.0 min · no trial directory" % WORKTREE_ERROR in beta
    assert "sprout   skipped (attempt-1 resolved)" in beta
    # a cause as long as a Harbor traceback is one clipped line in text and whole in json
    log = experiment / "baseline/tasks/beta/baseline/attempt-2.log"
    log.write_text("Traceback\nRuntimeError: Official Harbor trial failed: " + "x" * 2000 + "\n")
    beta = run(capsys, "task", str(experiment), "beta")
    line = next(l for l in beta.splitlines() if "attempt-2  errored" in l)
    assert len(line) < 300 and line.count("…") == 1 and "no trial directory" in line
    document = json.loads(run(capsys, "task", str(experiment), "beta", "--format", "json"))
    assert len(document["baseline"]["attempts"][1]["error"]) > 2000
    # alpha's json is whole too: the analyst report the text clips at 150, and each call with its id and un-summarized command
    document = json.loads(run(capsys, "task", str(experiment), "alpha", "--format", "json"))
    report = document["methods"][0]["rounds"][0]["reports"]["parent"]
    assert report["failure_reason"].startswith("FAILURE REASON") and len(report["failure_reason"]) > 150
    assert report["available_steps"] == [1, 2, 3, 4] and report["candidates"] == [3, 2]
    branch = document["methods"][0]["rounds"][0]["branches"][0]
    assert branch["trial"]["turns"] == 2 and branch["calls"][0] == ["rev3-1", "cd /app && go test ./..."]
    assert document["baseline"]["attempts"][0]["retained_points"] == 3
    # the same tree from the bare evaluate output, and from the task directory
    same = run(capsys, "task", str(experiment / "sprout/tasks/alpha/sprout"))
    assert same.splitlines()[1:] == text.splitlines()[1:]
    assert run(capsys, "task", str(experiment / "baseline/tasks/alpha")).splitlines()[1:4] == text.splitlines()[1:4]
    # --with attaches a second evaluation branching from attempt-1, and refuses one that does not
    parent = experiment / "baseline/tasks/alpha/baseline/attempt-1/trials/alpha-parent/agent/trajectory.jsonl"
    other = make_sprout_output(tmp_path / "other", parent_journal=parent, parent_sha="0" * 64, branch_resolved=False)
    stranger = make_sprout_output(tmp_path / "stranger", parent_sha="0" * 64,
                                  parent_journal=experiment / "baseline/tasks/beta/baseline/attempt-1/trials/beta-parent/agent/trajectory.jsonl")
    text = run(capsys, "task", str(experiment), "alpha", "--with", str(other), "--with", str(stranger))
    assert text.count("└── round 1") == 2 and "sprout %s   complete · unresolved" % other in text
    assert "--with %s does not branch from this task's attempt-1: skipped" % stranger in text


def test_a_live_task_directory_and_a_running_attempt_directory_show_the_same_attempts(tmp_path, capsys):
    """During the baseline attempts tasks/<task>/ holds only baseline/attempt-N{,.stage.json,.log}: no
    task.json yet, and a running attempt's summary.json has no journal row."""
    root = make_experiment(tmp_path / "exp", lock_pid=os.getpid(), with_sprout=False,
                           tasks={"alpha": {"attempts": [("complete", False), ("live", None)]}})
    task_dir = root / "baseline/tasks/alpha"
    assert not list(task_dir.glob("*.json")) and detect(task_dir) == "task"
    expected = run(capsys, "task", str(root), "alpha")
    assert "├── attempt-1  complete  completed  FAIL" in expected
    assert "└── attempt-2  running   turn 5 (4 closed, 4 steps)   actor running   last event 2026-09-30T20:" in expected
    attempts = expected.splitlines()[:4]                           # header, baseline line, the two attempts
    assert run(capsys, "task", str(task_dir)).splitlines()[:4] == attempts
    assert detect(task_dir / "baseline/attempt-2") == "evaluate"
    assert run(capsys, "task", str(task_dir / "baseline/attempt-2")).splitlines()[:4] == attempts
    assert run(capsys, "task", str(task_dir / "baseline/attempt-1")).splitlines()[:4] == attempts
    assert "totals  2 journals" in run(capsys, "snapshots", str(task_dir))


def test_runs_takes_the_backend_from_the_receipts_when_the_manifest_predates_it(tmp_path, capsys):
    root = make_experiment(tmp_path / "exp", with_sprout=False, tasks={"alpha": {"attempts": [("complete", True)]}})
    manifest = root / "baseline/manifest-baseline.json"
    body = read_dict(manifest)
    del body["backend"], body["store"]
    write_json(manifest, body)
    text = run(capsys, "runs", str(root))
    assert "· podman %s/store (from receipts) · commit" % root in text and "backend unknown" not in text
    document = json.loads(run(capsys, "runs", str(root), "--format", "json"))
    assert (document["methods"][0]["backend"], document["methods"][0]["backend_source"]) == ("podman", "receipts")
    receipts = root.glob("baseline/tasks/alpha/baseline/attempt-1/trials/*/*.podman.json")
    for receipt in receipts:
        receipt.unlink()
    document = json.loads(run(capsys, "runs", str(root), "--format", "json"))
    assert (document["methods"][0]["backend"], document["methods"][0]["backend_source"]) == (None, None)
    assert "backend unknown" in run(capsys, "runs", str(root))


def test_task_mermaid_hangs_branches_off_their_steps(experiment, capsys):
    text = run(capsys, "task", str(experiment), "alpha", "--format", "mermaid")
    assert text.startswith("graph LR\n")
    assert '  parent(["parent · attempt-1"]) --> parent_s3["step 3 (turn 3)<br/>par-0003"]' in text
    assert '  parent_s3 --> parent_end["step 5 · final<br/>FAIL reward 0.0<br/>par-0005"]' in text
    assert '  parent_s3 --> r1b1_fix["r1b1-fix<br/>round 1 · assistant-turn<br/>2 turns · 3 steps · 2.0 min<br/>PASS reward 1.0"]' in text
    assert "  subgraph independent baseline attempts" in text
    assert '    attempt_2["attempt-2<br/>5 turns · 5 steps · 20.0 min<br/>PASS reward 1.0"]' in text
    assert "class r1b1_fix,attempt_2,attempt_3 pass;" in text and "class parent_end,attempt_4 fail;" in text


def test_task_degrades_gracefully(tmp_path, capsys):
    root = make_experiment(tmp_path / "exp", with_sprout=False,
                           tasks={"alpha": {"attempts": [("complete", False), ("complete", True)]}})
    text = run(capsys, "task", str(root), "alpha")
    assert "sprout   no sprout stage under this root (no sprout/ method root; pass --with <evaluate output> to attach one)" in text
    parent = root / "baseline/tasks/alpha/baseline/attempt-1/trials/alpha-parent/agent/trajectory.jsonl"
    pending = make_sprout_output(tmp_path / "pending", parent_journal=parent, parent_sha="0" * 64, plan_state="requesting")
    text = run(capsys, "task", str(pending))
    assert "reviewer attempt 2  requesting        [review-003.json in flight]" in text and "in progress" in text
    live = make_sprout_output(tmp_path / "live", parent_journal=parent, parent_sha="0" * 64, live_branch=True)
    text = run(capsys, "task", str(live))
    assert "trial  running   turn 3 (2 closed, 3 steps)   actor running   last event 2026-09-30T20:" in text
    failed = make_sprout_output(tmp_path / "failed", parent_journal=parent, parent_sha="0" * 64, plan_state="failed")
    text = run(capsys, "task", str(failed))
    assert "reviewer selected no branches: reviewer failed: Expecting value" in text
    assert "stopped: reviewer failed: Expecting value" in text
    # a journal before sandbox.created: the receipt supplies backend and sandbox
    old = make_evaluation(tmp_path / "old", method="baseline", attempts=[
        {"name": "parent", "resolved": False, "backend": None, "sandbox": "01a0-vm"}], backend="microvm")
    document = json.loads(run(capsys, "task", str(old), "--format", "json"))
    assert document["baseline"]["attempts"] == [] and document["backend"] == "microvm"
    trial = load_trial(next((old / "trials").iterdir()))
    assert trial.receipt["sandbox_id"] == "01a0-vm" and trial.receipt["backend"] == "microvm"


def test_snapshots_checks_the_store_marks_selections_and_reports_unknown_stores(experiment, tmp_path, capsys):
    store = tmp_path / "store"
    for snapshot in ("par-0001", "par-0002", "par-0003", "par-0004", "brn-0001", "brn-0002"):
        (store / "snapshots" / snapshot).mkdir(parents=True)
    output = experiment / "sprout/tasks/alpha/sprout"
    text = run(capsys, "snapshots", str(output), "--store", str(store))
    assert text.startswith("store %s (from --store) · podman: present = <store>/snapshots/<id> is a directory · nothing is deleted or executed" % store)
    assert "selected by plan-round1.json: par-0003 (r1b1-fix, parent@3)" in text
    assert "== alpha-parent   podman   sandbox par   recorded parent, outside this output" in text
    assert "== alpha-r1b1-fix   podman   sandbox brn   branch of alpha-parent@3 from par-0003" in text
    rows = {line.split()[0]: line for line in text.splitlines() if line.strip().startswith(("par-", "brn-"))}
    assert rows["par-0003"].split()[-1] == "subvolume)" and "r1b1-fix, parent@3" in rows["par-0003"]
    assert rows["par-0005"].endswith("missing") and "final" not in rows["par-0004"]
    assert "5 records · 5 captured (5 ids) · 0 clean · 0 failed · 5 branch points · 1 selected · store: 4 present, 1 missing" in text
    assert "totals  2 journals · 7 records · 7 snapshot ids · 6 present · 1 missing · 0 not checked · 1 selected by plans · 0 swept · 0 released" in text
    # the sprout METHOD root lists the same recorded parent (its snapshots are the branch points)
    method = run(capsys, "snapshots", str(experiment / "sprout"), "--store", str(store))
    assert "== alpha-parent   podman   sandbox par   recorded parent, outside this output" in method
    assert "== alpha-r1b1-fix   podman   sandbox brn   branch of alpha-parent@3" in method
    assert method.count("== alpha-parent") == 1 and "selected by plan-round1.json: par-0003 (r1b1-fix, parent@3)" in method
    whole_experiment = run(capsys, "snapshots", str(experiment), "--store", str(store), "--task", "alpha")
    assert "recorded parent, outside this output" not in whole_experiment      # the baseline root is walked already
    assert whole_experiment.count("== alpha-parent") == 4                       # the four attempts' trials
    # two plan selections, a branch no plan names, a microvm journal and a swept one
    parent, _ = make_journal(tmp_path / "mix/trials/t-parent/agent/trajectory.jsonl", run_id="t-parent", sandbox="par",
                             turns=simple_turns(3), store="/gone/store")
    origin = {"parent_journal": str(parent), "branch_step": 2, "snapshot_id": "par-0002", "conversation_cut": "t2",
              "branch_guidance": "none"}
    swept = json.dumps({"v": 2, "type": "snapshots.swept", "seq": 99, "run_id": "sweep", "backend": "podman",
                        "deleted": 1, "failed": 0})
    make_journal(tmp_path / "mix/trials/t-b1/agent/trajectory.jsonl", run_id="t-b1", sandbox="b1", turns=simple_turns(1),
                 origin=origin, extra_tail=(swept,))
    make_journal(tmp_path / "mix/trials/t-vm/agent/trajectory.jsonl", run_id="t-vm", sandbox="vm", turns=simple_turns(1),
                 backend="microvm")
    write_json(tmp_path / "mix/summary.json", {"attempts": [], "method": "sprout", "complete": False})
    write_json(tmp_path / "mix/plan-round1.json", {"selected_branches": [
        {"run_name": "r1b1-a", "base": "parent", "branch_step": 1, "snapshot_id": "par-0001"},
        {"run_name": "r1b2-b", "base": "parent", "branch_step": 3, "snapshot_id": "par-0003"}]})
    text = run(capsys, "snapshots", str(tmp_path / "mix"), "--store", str(store))
    assert "forked: t-b1" in text and "2 selected by plans" in text
    assert "== t-vm   microvm   sandbox vm" in text and "not checked (AgentENV is not queried in this version)" in text
    assert [line.split()[-1] for line in text.splitlines() if line.strip().startswith("b1-")] == ["swept"]
    swept_line = next(l for l in text.splitlines() if l.strip().startswith("1 records"))
    assert swept_line.strip() == "1 records · 1 captured (1 ids) · 0 clean · 0 failed · 1 branch points · 0 selected · 1 swept"
    vm_line = [l for l in text.splitlines() if l.strip().startswith("1 records")][1]
    assert vm_line.endswith("· 0 selected · store: not checked")
    # a journal under <output>/trials/<trial>/agent/ reads the output's plans
    text = run(capsys, "snapshots", str(parent))
    assert "not checked: store /gone/store has no snapshots/ directory" in text
    assert "selected by plan-round1.json: par-0001 (r1b1-a, parent@1) · par-0003 (r1b2-b, parent@3)" in text
    assert "2 selected by plans" in text and "3 records · 3 captured (3 ids) · 0 clean · 0 failed · 3 branch points · 2 selected · store: not checked" in text
    # an experiment root is walked whole; --task narrows it
    whole = json.loads(run(capsys, "snapshots", str(experiment), "--format", "json"))
    assert len(whole["journals"]) >= 3
    narrowed = json.loads(run(capsys, "snapshots", str(experiment), "--task", "beta", "--format", "json"))
    assert [j["run_id"] for j in narrowed["journals"]] == ["beta-parent"]


def test_snapshots_and_the_other_commands_start_no_process_and_write_nothing(experiment, monkeypatch, capsys):
    import builtins
    import io
    import subprocess

    def forbidden(*args, **kwargs):
        raise AssertionError("a process was started: %r" % (args,))
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(os, "system", forbidden)
    real_open = builtins.open

    def guarded(file, mode="r", *args, **kwargs):
        if any(flag in str(mode) for flag in "wa+") and str(file).startswith(str(experiment)):
            raise AssertionError("opened %s for writing (%s)" % (file, mode))
        return real_open(file, mode, *args, **kwargs)
    monkeypatch.setattr(builtins, "open", guarded)
    monkeypatch.setattr(io, "open", guarded)

    def snapshot():
        return sorted((str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in experiment.rglob("*"))
    before = snapshot()
    for argv in (["runs", str(experiment)], ["task", str(experiment), "alpha"], ["task", str(experiment), "alpha", "--format", "mermaid"],
                 ["export", str(experiment / "sprout/tasks/alpha/sprout/trials/alpha-r1b1-fix/agent/trajectory.jsonl")],
                 ["snapshots", str(experiment)], ["list", str(experiment.parent)]):
        assert cli.main(argv) == 0, argv
        capsys.readouterr()
    assert snapshot() == before


def test_list_recognizes_every_kind(experiment, tmp_path, capsys):
    base = experiment.parent
    parent = experiment / "baseline/tasks/alpha/baseline/attempt-1/trials/alpha-parent/agent/trajectory.jsonl"
    make_sprout_output(base / "evals" / "one", parent_journal=parent, parent_sha="0" * 64)
    gate = base / "gate"
    gate.mkdir()
    (gate / "gate.jsonl").write_text(json.dumps({"task": "t", "mode": "oracle", "ok": True}) + "\n")
    write_json(gate / "summary.json", {"tasks": 1, "modes": ["oracle", "nop"], "ok": [], "failures": []})
    make_journal(base / "lone.jsonl", run_id="lone", sandbox="s", turns=simple_turns(1))
    (base / "unrelated").mkdir()
    (base / "unrelated" / "notes.txt").write_text("x")
    text = run(capsys, "list", str(base))
    rows = [line.split() for line in text.splitlines()[1:]]
    assert len(rows) >= 4 and "unrelated" not in text
    assert ["experiment", "baseline,sprout", "4", "1", "1", "1", "0", "alive", "exp"] in rows
    assert ["evaluate", "sprout", "1", "1", "0", "0", "-", "-", "evals/one"] in rows
    assert ["gate", "oracle,nop", "1", "-", "-", "-", "-", "-", "gate"] in rows
    assert ["journal", "-", "-", "-", "-", "-", "-", "-", "lone.jsonl"] in rows
    assert detect(base) is None


def test_real_fixture_json_shapes_load(tmp_path):
    files = sorted(p for p in FIXTURES.iterdir() if p.is_file())
    assert len(files) >= 10
    output = tmp_path / "abs-module-cache-flags"
    trial = output / "trials" / "abs-module-cache-flags-r1b1-abs-absolute-fix-precomm"
    (trial / "agent").mkdir(parents=True)
    shutil.copy(FIXTURES / "summary.json", output / "summary.json")
    shutil.copy(FIXTURES / "plan-round1.json", output / "plan-round1.json")
    shutil.copy(FIXTURES / "r1b1.trajectory.jsonl", trial / "agent" / "trajectory.jsonl")
    for name in ("execution.json", "snapshot.json"):
        shutil.copy(FIXTURES / name, trial / "agent" / name)
    for name in ("restored-state.json", "abs-module-cache-flags-r1b1-abs-absolute-fix-precomm__env.podman.json",
                 "abs-module-cache-flags-r1b1-abs-absolute-fix-precomm__4b20425b.podman.json"):
        shutil.copy(FIXTURES / name, trial / name)
    evaluation = load_evaluation(output)
    assert evaluation.summary["task"] == "datacurve/abs-module-cache-flags" and len(evaluation.attempts) == 3
    assert evaluation.attempts[0].journal is None                       # the recorded parent lives in a moved root
    assert evaluation.attempts[1].journal == trial / "agent" / "trajectory.jsonl"
    plan = evaluation.plans[0]
    assert plan.round == 1 and plan.limit == 4 and plan.count_mode == "adaptive"
    assert [a["status"] for a in plan.review_attempts] == ["validation_error", "validated"]
    assert plan.review_attempts[0]["validation_error"].startswith("Invalid JSON inside branch-plan block")
    assert [b["run_name"] for b in plan.selected] == ["r1b1-abs-absolute-fix-precomm", "r1b2-abs-absolute-fix-early"]
    assert plan.reports["parent"]["branch_candidates"][0]["step"] == 136 and len(plan.available_steps["parent"]) == 110
    [loaded] = evaluation.trials
    assert loaded.phase == "grading" and loaded.restored["snapshot_id"] == "sprout-b2867d643e67-0105"
    assert loaded.receipt["sandbox_id"] == "sprout-69b1adc970d8" and loaded.receipt["overlay"] is None
    assert loaded.receipt["image"] == "sprout-b2867d643e67-0105" and loaded.receipt["store"] == "/path/to/sprout-btrfs"
    # the log is stored as .log.txt because the repository ignores *.log; the stage reads <name>.log
    (tmp_path / "attempt-4.log").write_text((FIXTURES / "attempt-4.log.txt").read_text())
    shutil.copy(FIXTURES / "attempt-4.stage.json", tmp_path / "attempt-4.stage.json")
    stage = load_stage(tmp_path / "attempt-4.stage.json", lock_alive=False)
    assert (stage.state, stage.error, stage.cause) == ("errored", "Stage exited 1; see attempt-4.log", WORKTREE_ERROR)
    assert load_stage(FIXTURES / "attempt-1.stage.json", lock_alive=False).state == "complete"
    failure = read_dict(FIXTURES / "baseline.task.json")
    assert failure["status"] == "infrastructure_error" and "baseline" not in failure
    success = read_dict(FIXTURES / "baseline.task.complete.json")
    assert success["complete"] and len(success["baseline"]["attempts"]) == 4
    assert isinstance(success["initial_resolved"], bool) and success["baseline"]["attempts"][0]["name"] == "parent"
    retained = read_dict(FIXTURES / "retained-checkpoints.json")
    assert len(retained["points"]) == 2 and retained["points"][0]["checkpoint"]["snapshot_id"] == "sprout-b2867d643e67-0001"
    assert retained["backend"] == "podman" and retained["store"] == "/path/to/sprout-btrfs"
    grade = load_grade({"verifier_result": {"rewards": {"reward": 0, "f2p_total": 20, "f2p_passed": 9,
                                                        "p2p_total": 3, "p2p_passed": 3}}, "exception_info": None})
    assert (grade.resolved, grade.reward, grade.detail) == (False, 0.0, "f2p 9/20 p2p 3/3")

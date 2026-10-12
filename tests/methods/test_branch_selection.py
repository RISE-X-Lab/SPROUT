"""The sprout loop: analyst -> reviewer -> exact branches, driven with fake trials.

Every trial is faked at ``methods.evaluate.run_trial`` (a journal with N steps,
each with a snapshot) and the reviewer at ``ReviewTransport``; everything
between them -- eligible steps, plan validation, branch contexts, rounds and the
summary -- is the real code.
"""

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

pytest.importorskip("harbor")

from sprout.methods import evaluate
from sprout.methods.branching import Attempt, Grade, branch_run_name
from sprout.methods.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH, load_prompts


def write_journal(path: Path, steps: int = 3, status: str = "completed", turns=None) -> Path:
    """A mini journal for the trial at ``<trial>/agent/trajectory.jsonl``: one closed
    turn per entry of ``turns`` (default: one call per turn), each captured at its
    last call. Snapshot ids carry the trial name."""
    path.parent.mkdir(parents=True, exist_ok=True)
    run = path.parent.parent.name
    turns = turns or [[step] for step in range(1, steps + 1)]
    native_path = path.parent / "mini-native.jsonl"
    native = [{"type": "mini.session", "format": "sprout-mini-v1",
               "mini_version": "2.4.6", "session_id": "ses-1"}]
    records = [{"type": "run.started", "run_id": run, "slot": "mini-swe-agent"},
               {"type": "session.ref", "native_session_id": "ses-1",
                "transcript_path": str(native_path)}]
    for calls in turns:
        turn_id = "t%d" % calls[-1]
        native.append({"type": "mini.message", "message": {
            "role": "assistant", "content": "Inspect the files.", "tool_calls": [
                {"id": "c%d" % step, "type": "function",
                 "function": {"name": "bash", "arguments": json.dumps({"command": "ls"})}}
                for step in calls]}})
        for step in calls:
            records += [{"type": "tool.started", "step": step, "call_id": "c%d" % step,
                         "turn_id": turn_id, "name": "shell", "args": {"command": "ls"}},
                        {"type": "tool.finished", "call_id": "c%d" % step, "turn_id": turn_id,
                         "status": "ok", "output": "listing"}]
            native.append({"type": "mini.message", "message": {
                "role": "tool", "tool_call_id": "c%d" % step, "content": "listing"}})
        native.append({"type": "mini.turn", "turn_id": turn_id,
                       "call_ids": ["c%d" % step for step in calls]})
        records += [{"type": "model.turn.completed", "turn_id": turn_id,
                     "call_ids": ["c%d" % s for s in calls], "step": calls[-1]},
                    {"type": "checkpoint.captured", "step": calls[-1], "turn_id": turn_id,
                     "snapshot_id": "%s-snap-%d" % (run, calls[-1]), "session_ckpt": "ses-1",
                     "reason": "captured"}]
    records.append({"type": "run.finished", "run_id": run, "status": status, "usage": {}})
    for seq, record in enumerate(records):
        record.update(v=2, seq=seq, ts="2026-09-04T00:00:%02dZ" % min(seq, 59))
    path.write_text("\n".join(json.dumps(r) for r in records) + "\n")
    native_path.write_text("\n".join(json.dumps(r) for r in native) + "\n")
    return path


def direction(step, name="lead", base="parent", hint="  Check the boundary.  "):
    return {"name": name, "base": base, "branch_step": step,
            "why": "Relevant work is present.", "hint": hint}


def args_for(tmp_path, **changes):
    values = dict(task_dir=tmp_path / "task", dataset_digest="sha256:pinned",
                  preparation_manifest=tmp_path / "prep.json", output=tmp_path / "out",
                  method="sprout", parent_output=None, model="model", model_endpoint="http://127.0.0.1:1",
                  model_key_env="KEY", runtime_bin=tmp_path / "runtime", backend="microvm", store=None,
                  server_url=None, api_key_file=None, sandbox_ttl=36000, checkpoint_mode="disk_only",
                  max_output_tokens=1000, max_turns=10, command_timeout=60, review_timeout=3600, reasoning_effort=None,
                  branch_guidance="user-hint", branch_count_mode="adaptive", prompts=DEFAULT_PROMPTS, rounds=2,
                  branches=[4], review_output_tokens=1000, reviewer_max_attempts=1, analyst_tokens=1000,
                  prompt_budget_chars=400_000, max_rollouts=8)
    values.update(changes)
    return SimpleNamespace(**values)


def exercise(tmp_path, monkeypatch, plans, *, limits=None, alter_parent=None, count_mode="adaptive",
             reviewer_max_attempts=1, guidance="user-hint", rounds=None, parent_steps=5,
             parent_turns=None, prompts=DEFAULT_PROMPTS):
    """Run the loop with canned reviewer plans; return (attempts, branch trials, analyst, reviewer prompts)."""
    rounds = len(plans) if rounds is None else rounds
    trials, analyst_inputs, reviewer_inputs = [], [], []

    class Transport:
        def __init__(self, *_, **__):
            pass

        def __call__(self, model, prompt):
            if "## Every attempt so far" in prompt:
                reviewer_inputs.append(prompt)
                plan = json.dumps(plans[len(reviewer_inputs) - 1])
                # Hint reviews answer in JSON; the other modes in a labelled fence.
                return plan if "labelled branch-plan" not in prompt else f"```branch-plan\n{plan}\n```"
            analyst_inputs.append(prompt)
            return json.dumps({"failure_reason": "A boundary remains.", "lesson": "Inspect it.",
                               "salvage": "Existing implementation.",
                               "branch_candidates": [{"step": 1, "why": "Useful state."}]})

    async def run_trial(args, task, *, name, branch_context=None):
        journal = write_journal(args.output / "trials" / name / "agent" / "trajectory.jsonl",
                                steps=parent_steps if name == "parent" else 3,
                                turns=parent_turns if name == "parent" else None)
        if name == "parent" and alter_parent:
            records = [json.loads(line) for line in journal.read_text().splitlines()]
            journal.write_text("\n".join(json.dumps(e) for e in alter_parent(records)) + "\n")
        context = json.loads(branch_context.read_text()) if branch_context else None
        trials.append({"name": name, "context": context})
        outcome = evaluate.outcome_from_journal(journal, run_id=name)
        return evaluate.TrialRecord(Attempt(name, outcome, Grade(patch="change")),
                                    {"image": "image@sha256:digest", "checkpoint_mode": "disk_only"})

    monkeypatch.setattr(evaluate, "run_trial", run_trial)
    monkeypatch.setattr(evaluate, "ReviewTransport", Transport)
    monkeypatch.setattr(evaluate, "_task_image_config", lambda image: {"WorkingDir": "/app"})
    monkeypatch.setattr(evaluate, "_verify_prepared_task", lambda *a: None)
    monkeypatch.setattr(evaluate, "_source_commit", lambda: "deadbeef")
    def restore(path, step, session=None):
        return "%s-cut-%d" % (Path(path).parent.parent.name, step)

    # Fake trials consume deterministic context labels; candidate eligibility
    # still uses the real native transcript written beside the journal.
    monkeypatch.setattr("sprout.methods.branching.conversation_restore", restore)
    (tmp_path / "task").mkdir(exist_ok=True)
    (tmp_path / "task" / "task.toml").write_text("[task]\nname='org/task'\n")
    task = SimpleNamespace(name="org/task", instruction="Fix the public task.",
                           config=SimpleNamespace(verifier=SimpleNamespace(collect=[])))
    args = args_for(tmp_path, rounds=rounds, branches=limits or [4], branch_count_mode=count_mode,
                    reviewer_max_attempts=reviewer_max_attempts, branch_guidance=guidance, prompts=prompts)
    import asyncio
    evaluation = evaluate.Evaluation(args, task)
    asyncio.run(evaluation.run())
    return evaluation.attempts, [t for t in trials if t["name"] != "parent"], analyst_inputs, reviewer_inputs


@pytest.mark.parametrize("steps", [[2], [2, 2], [1, 2, 3, 4]])
def test_reviewer_may_choose_fewer_repeated_or_distinct_points(tmp_path, monkeypatch, steps):
    branches = [direction(step, name="lead%d" % i, hint="  Hint %d.  " % i)
                for i, step in enumerate(steps)]
    attempts, trials, analyst, reviewer = exercise(
        tmp_path, monkeypatch, [{"synthesis": "Useful choices.", "branches": branches}])
    assert len(trials) == len(steps)
    assert [t["context"]["snapshot_id"] for t in trials] == ["parent-snap-%d" % step for step in steps]
    assert [t["context"]["conversation_cut"] for t in trials] == ["parent-cut-%d" % step for step in steps]
    assert [t["context"]["hint"] for t in trials] == [b["hint"] for b in branches]
    assert all(t["context"]["branch_guidance"] == "user-hint" for t in trials)
    assert [a.hint for a in attempts[1:]] == [b["hint"] for b in branches]
    assert len(analyst) == len(reviewer) == 1
    assert '"available_steps": [' in reviewer[0]
    # The request carries the adaptive count rule and none of the fixed one.
    assert "Return at most 4 branches" in reviewer[0] and "upper bound, not a quota" in reviewer[0]
    assert "Return exactly" not in reviewer[0]
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert saved["branch_limit"] == 4 and saved["review"]["branches"] == branches
    assert saved["branch_count_mode"] == "adaptive"
    assert saved["structured_review"] is False, "a plain transport: the plan says the review was not schema-constrained"
    assert saved["prompts"] == {"path": str(DEFAULT_PROMPTS_PATH), "sha256": DEFAULT_PROMPTS.sha256}
    assert len(saved["selected_branches"]) == len(steps)
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["complete"] and [a["name"] for a in summary["attempts"]][0] == "parent"
    assert summary["prompts"] == saved["prompts"], "the summary and every plan name the prompts file"


def test_branches_can_choose_different_base_attempts(tmp_path, monkeypatch):
    child = branch_run_name(1, 1, "first")
    plans = [{"synthesis": "Start.", "branches": [direction(1, "first")]},
             {"synthesis": "Two bases.", "branches": [direction(2), direction(3, base=child)]}]
    _, trials, _, _ = exercise(tmp_path, monkeypatch, plans, limits=[4, 3])
    assert len(trials) == 3
    assert trials[1]["context"]["snapshot_id"] == "parent-snap-2"
    assert trials[2]["context"]["snapshot_id"] == child + "-snap-3"
    assert trials[2]["context"]["parent_journal"].endswith(child + "/agent/trajectory.jsonl")
    assert trials[2]["context"]["conversation_cut"] == child + "-cut-3"


def test_analyst_and_reviewer_only_receive_complete_turn_steps(tmp_path, monkeypatch):
    _, trials, analysts, reviewers = exercise(tmp_path, monkeypatch, [{"branches": [direction(2)]}],
                                              parent_turns=[[1, 2], [3, 4, 5]])
    plan = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert plan["available_steps"]["parent"] == [2, 5]
    assert "[2, 5]" in analysts[0]
    assert len(reviewers) == len(trials) == 1
    assert trials[0]["context"]["snapshot_id"] == "parent-snap-2"


@pytest.mark.parametrize("branches,reason", [
    ([direction(1)] * 5, "above limit"),
    ([direction(1), direction(9)], "no eligible checkpoint"),
    ([direction(1), direction(2, base="unknown")], "unknown base"),
    ([direction(True)], "no eligible checkpoint"),
    ([direction(1, hint=" ")], "non-empty hint"),
])
def test_invalid_plan_is_rejected_before_any_branch_launch(tmp_path, monkeypatch, branches, reason):
    _, trials, _, _ = exercise(tmp_path, monkeypatch, [{"branches": branches}])
    assert not trials
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert reason in saved["validation_error"]
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert summary["complete"] and summary["branch_schedule_complete"] is False


def test_empty_selection_stops_without_padding_or_reasking(tmp_path, monkeypatch):
    _, trials, _, reviews = exercise(tmp_path, monkeypatch,
                                     [{"synthesis": "No useful lead.", "branches": []},
                                      {"branches": [direction(1)]}])
    assert not trials and len(reviews) == 1


@pytest.mark.parametrize("missing", [False, True])
def test_failed_or_missing_snapshot_is_not_offered_or_silently_replaced(tmp_path, monkeypatch, missing):
    def alter(records):
        out = []
        for e in records:
            if e.get("type") == "checkpoint.captured" and e["step"] == 2:
                if missing:
                    continue
                e.update(reason="failed", snapshot_id="parent-snap-1", captured=False)
            out.append(e)
        return out

    _, trials, analyst, reviews = exercise(tmp_path, monkeypatch,
                                           [{"branches": [direction(2)]}], alter_parent=alter)
    assert not trials
    assert "[1, 3, 4, 5]" in analyst[0]
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert saved["available_steps"]["parent"] == [1, 3, 4, 5]
    assert "no eligible checkpoint" in saved["validation_error"]


@pytest.mark.parametrize("steps,reason", [
    ([], "requires exactly 4 branches"),
    ([2], "requires exactly 4 branches"),
    ([2, 2], "requires exactly 4 branches"),
    ([2, 2, 2], "requires exactly 4 branches"),
    ([2] * 5, "requires exactly 4 branches"),
    # The right count with an ineligible step: per-branch validation is additional
    # to the count check, not replaced by it.
    ([1, 2, 3, 9], "no eligible checkpoint"),
])
def test_fixed_mode_rejects_wrong_count_without_padding_or_extra_review(tmp_path, monkeypatch, steps, reason):
    plan = {"branches": [direction(s, name="lead%d" % i) for i, s in enumerate(steps)]}
    _, trials, _, reviews = exercise(tmp_path, monkeypatch, [plan], count_mode="fixed")
    assert not trials and len(reviews) == 1
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert reason in saved["validation_error"]
    assert len(saved["review"]["branches"]) == len(steps)


def test_fixed_count_follows_each_round_schedule(tmp_path, monkeypatch):
    # Round 1 picks distinct points, round 2 repeats one: each runs exactly its count.
    plans = [{"branches": [direction(s, name="lead%d" % i) for i, s in enumerate(steps)]}
             for steps in [[1, 2, 3, 4], [2, 2, 2]]]
    _, trials, _, reviews = exercise(tmp_path, monkeypatch, plans, limits=[4, 3], count_mode="fixed")
    assert len(trials) == 7
    assert [t["context"]["step"] for t in trials] == [1, 2, 3, 4, 2, 2, 2]
    assert "Return exactly 4 branches" in reviews[0]
    assert "Return exactly 3 branches" in reviews[1]
    for prompt in reviews:  # the fixed rule, and none of the adaptive wording
        assert "Return at most" not in prompt and "upper bound" not in prompt
        assert "return an empty branches list" not in prompt
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    assert saved["branch_count_mode"] == "fixed"
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert [a["round"] for a in summary["attempts"]] == [0] + [1] * 4 + [2] * 3


def test_a_resolved_branch_ends_the_rounds(tmp_path, monkeypatch):
    original = evaluate.Evaluation.branch

    async def resolving(self, choice, round_no, **kwargs):
        attempt = await original(self, choice, round_no, **kwargs)
        attempt.grade.resolved = True
        return attempt

    monkeypatch.setattr(evaluate.Evaluation, "branch", resolving)
    plans = [{"branches": [direction(1)]}, {"branches": [direction(2)]}]
    attempts, trials, _, reviews = exercise(tmp_path, monkeypatch, plans, limits=[4, 3])
    assert len(trials) == len(reviews) == 1
    assert json.loads((tmp_path / "out/summary.json").read_text())["resolved"] is True


def test_point_only_and_assistant_turn_branches_carry_no_hint(tmp_path, monkeypatch):
    """What the actor receives is decided by the guidance mode alone."""
    plan = {"branches": [{"name": "restart", "base": "parent", "branch_step": 2, "why": "useful"}]}
    monkeypatch.setattr("sprout.methods.branching.selected_prefix", lambda *a: [])
    monkeypatch.setattr("sprout.methods.branching.reviewer_context", lambda *a: {"messages": [], "prefix_message_counts": {}})
    _, trials, _, _ = exercise(tmp_path, monkeypatch, [plan], guidance="none", rounds=1)
    context = trials[0]["context"]
    assert context["branch_guidance"] == "none"
    assert "hint" not in context and "assistant_turn" not in context
    assert context["capture_checkpoints"] is True and context["commit_required"] is False


# --- the analyst's report is validated, never repaired ------------------------------
def _analyst_case(tmp_path, replies):
    from sprout.methods import branching
    from sprout.methods.evaluate import outcome_from_journal

    journal = write_journal(tmp_path / "trials" / "parent" / "agent" / "trajectory.jsonl", steps=3)
    attempt = Attempt("parent", outcome_from_journal(journal, run_id="parent"), Grade(patch="x"))
    prompts = []

    def transport(model, prompt):
        prompts.append(prompt)
        return replies[len(prompts) - 1]

    return branching, attempt, transport, prompts


def test_an_invalid_analyst_report_is_returned_with_the_validators_error(tmp_path):
    good = {"failure_reason": "r", "lesson": "l", "salvage": "s", "branch_candidates": [{"step": 2, "why": "w"}]}
    replies = [json.dumps({"failure_reason": "r", "branch_candidates": []}),   # lesson, salvage missing
               "no json here",
               json.dumps({**good, "branch_candidates": [{"step": "two", "why": "w"}]}),  # nested type
               json.dumps({**good, "extra": 1}),                                          # extra key
               json.dumps(good)]
    branching, attempt, transport, prompts = _analyst_case(tmp_path, replies)
    report = branching.analyse(transport, "m", "task", attempt, {1: None, 2: None, 3: None}, 2, 10000,
                               max_attempts=5)
    sizing = report.pop("prompt_budget")
    assert report == {**good, "steps": 3}
    assert sizing["budget"] == branching.DEFAULT_PROMPT_CHARS and sizing["within_budget"]
    assert len(prompts) == 5 and "Validation feedback" not in prompts[0]
    assert "Validation feedback" in prompts[1] and "'lesson' is a required property" in prompts[1]
    assert "Validation feedback" in prompts[2] and "no JSON" in prompts[2]
    assert "Validation feedback" in prompts[3] and "'two' is not of type 'integer'" in prompts[3]
    assert "Validation feedback" in prompts[4] and "Additional properties are not allowed" in prompts[4]


# --- the prompts passed in are the ones sent ----------------------------------------
# evaluate loads one Prompts (--prompts) and hands it to the analyst, the reviewer, the
# transport and the Shepherd selector. Each has the packaged file as its default, so a
# call site that dropped ``prompts=`` would still run: these tests edit a copy and
# check that the edit reaches the request.

def edited_prompts(tmp_path, edit):
    """The packaged prompts file with ``edit`` applied to its parsed mapping, loaded."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    edit(data)
    path = tmp_path / "edited-prompts.yaml"
    path.write_text(yaml.safe_dump(data))
    return load_prompts(path)


def test_the_analyst_renders_the_given_template_and_feedback(tmp_path):
    """analyse renders prompts.analyst_case and, after an invalid reply, appends
    prompts.analyst_feedback -- both from the Prompts passed in."""
    def edit(data):
        data["analyst"]["case"] = data["analyst"]["case"].replace(
            "## Checkpoint rules", "## Checkpoint rules (EDITED CASE)")
        data["analyst"]["feedback"] = data["analyst"]["feedback"].replace(
            "was rejected:", "was rejected (EDITED FEEDBACK):")
    edited = edited_prompts(tmp_path, edit)
    good = json.dumps({"failure_reason": "r", "lesson": "l", "salvage": "s",
                       "branch_candidates": [{"step": 2, "why": "w"}]})
    branching, attempt, transport, sent = _analyst_case(tmp_path, ["no json here", good] * 2)
    points = {1: None, 2: None, 3: None}
    branching.analyse(transport, "m", "task", attempt, points, 2, 10000)
    branching.analyse(transport, "m", "task", attempt, points, 2, 10000, prompts=edited)
    default_first, default_retry, edited_first, edited_retry = sent
    assert "## Checkpoint rules\n" in default_first and "EDITED" not in default_first
    assert edited_first == default_first.replace("## Checkpoint rules", "## Checkpoint rules (EDITED CASE)")
    assert default_retry.startswith(default_first) and edited_retry.startswith(edited_first), "feedback is appended"
    assert "was rejected: " in default_retry and "EDITED FEEDBACK" not in default_retry
    assert "was rejected (EDITED FEEDBACK): " in edited_retry and "no JSON" in edited_retry


def test_the_shepherd_selector_sends_the_given_system_prompt(tmp_path):
    """select_points sends prompts.shepherd_system as the system message, verbatim (its
    braces are literal), from the Prompts passed in."""
    from sprout.methods import policies

    def edit(data):
        data["shepherd"]["system"] += "\nEDITED FOR THIS TEST."
    edited = edited_prompts(tmp_path, edit)
    journal = write_journal(tmp_path / "trials" / "parent" / "agent" / "trajectory.jsonl", steps=3)
    payloads = []

    class Client:
        def complete(self, payload, audit):
            payloads.append(payload)
            return {"choices": [{"finish_reason": "stop", "message": {
                "content": json.dumps({"checkpoint_step": 2, "reason": "The decisive turn."})}}]}

    args = SimpleNamespace(method="shepherd", model="m", reasoning_effort=None, meta_transcript_tokens=1000,
                           meta_max_tokens=500, output=tmp_path / "out", max_rollouts=3)
    points = {1: None, 2: None, 3: None}
    selected = policies.select_points(args, "task", journal, points, Client())
    policies.select_points(args, "task", journal, points, Client(), prompts=edited)
    assert selected == [{"step": 2, "sibling_index": 2, "reason": "The decisive turn."},
                        {"step": 2, "sibling_index": 3, "reason": "The decisive turn."}]
    default, custom = (payload["messages"][0] for payload in payloads)
    assert default == {"role": "system", "content": DEFAULT_PROMPTS.shepherd_system}
    assert custom == {"role": "system", "content": edited.shepherd_system}
    assert custom["content"].endswith("EDITED FOR THIS TEST.") and "EDITED" not in default["content"]
    assert '{"checkpoint_step": integer, "reason": string}' in default["content"], "verbatim: braces single"
    assert json.loads(payloads[0]["messages"][1]["content"])["eligible_checkpoint_steps"] == [1, 2, 3]


def test_the_loop_sends_the_prompts_it_was_given(tmp_path, monkeypatch):
    """The analyst's and the reviewer's requests come from ``args.prompts`` -- the file
    parse_args loaded -- not the packaged default: an edited copy changes both requests,
    and the plan and the summary name the copy."""
    def edit(data):
        data["analyst"]["case"] = data["analyst"]["case"].replace(
            "## Checkpoint rules", "## Checkpoint rules (EDITED CASE)")
        data["reviewer"]["user_hint"] = data["reviewer"]["user_hint"].replace(
            "## Selection rules", "## Selection rules (EDITED HINT)")
    edited = edited_prompts(tmp_path, edit)
    _, trials, analyst, reviewer = exercise(tmp_path, monkeypatch, [{"branches": [direction(2)]}], prompts=edited)
    assert len(trials) == 1 and len(analyst) == len(reviewer) == 1
    assert "## Checkpoint rules (EDITED CASE)\n" in analyst[0]
    assert "## Selection rules (EDITED HINT)\n" in reviewer[0]
    assert "EDITED" not in DEFAULT_PROMPTS.analyst_case + DEFAULT_PROMPTS.reviewer_user_hint
    saved = json.loads((tmp_path / "out/plan-round1.json").read_text())
    summary = json.loads((tmp_path / "out/summary.json").read_text())
    assert saved["prompts"] == summary["prompts"] == {"path": str(edited.path), "sha256": edited.sha256}
    assert edited.sha256 != DEFAULT_PROMPTS.sha256


def test_the_loop_hands_its_prompts_to_the_transport_and_the_selector(tmp_path, monkeypatch):
    """The transport's JSON directive and the Shepherd selector's prompts are
    ``args.prompts`` too: ReviewTransport is built with its transport_json_directive,
    and select_points receives the same Prompts."""
    import asyncio

    def edit(data):
        data["transport"]["json_directive"] += " EDITED DIRECTIVE."
    edited = edited_prompts(tmp_path, edit)
    received = {}

    class Transport:
        def __init__(self, *_, **kwargs):
            received["json_directive"] = kwargs.get("json_directive")

    def select_points(args, problem, journal, points, client, **kwargs):
        received["selector"] = kwargs.get("prompts")
        return []

    monkeypatch.setattr(evaluate, "ReviewTransport", Transport)
    monkeypatch.setattr(evaluate.policies, "select_points", select_points)
    monkeypatch.setenv("KEY", "k")
    (tmp_path / "task").mkdir()
    (tmp_path / "task" / "task.toml").write_text("[task]\nname='org/task'\n")
    task = SimpleNamespace(name="org/task", instruction="Fix the public task.")
    # No round: the sprout loop only builds its transport.
    asyncio.run(evaluate.Evaluation(args_for(tmp_path, prompts=edited, rounds=0), task).sprout())
    assert received["json_directive"] == edited.transport_json_directive != DEFAULT_PROMPTS.transport_json_directive
    journal = write_journal(tmp_path / "trials" / "parent" / "agent" / "trajectory.jsonl", steps=3)
    evaluation = evaluate.Evaluation(args_for(tmp_path, prompts=edited, method="shepherd"), task)
    evaluation.attempts = [Attempt("parent", evaluate.outcome_from_journal(journal, run_id="parent"), Grade(patch="x"))]
    asyncio.run(evaluation.sample())
    assert received["selector"] is edited
    assert json.loads((tmp_path / "out/plan.json").read_text())["eligible_steps"] == [1, 2, 3]

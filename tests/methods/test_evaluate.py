from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

pytest.importorskip("harbor")

from harbor.models.task.task import Task
from sprout.worker.jobs.review import structured_review
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS
from sprout.worker.jobs.review.transport import ReviewTransport
from sprout.eval import evaluate


def _result(directory: Path, reward, *, error=None):
    directory.mkdir(exist_ok=True)
    (directory / "result.json").write_text(json.dumps({
        "finished_at": "2026-09-23T00:00:00Z", "exception_info": error,
        "verifier_result": {"rewards": {"reward": reward}}}))


def test_only_a_valid_official_harbor_grade_can_resolve_a_branch(tmp_path):
    trial = tmp_path / "trial"
    _result(trial, 1)
    grade = evaluate.read_grade(trial)
    assert grade.resolved and grade.reward == 1
    assert "official Harbor reward=1" in grade.summary()
    _result(trial, 0)
    assert not evaluate.read_grade(trial).resolved
    _result(trial, None)
    with pytest.raises(RuntimeError, match="missing or invalid"):
        evaluate.read_grade(trial)
    _result(trial, 1, error="verifier failed")
    with pytest.raises(RuntimeError, match="verifier failed"):
        evaluate.read_grade(trial)
    _result(trial, 0, error={"exception_type": "AgentTimeoutError"})
    assert not evaluate.read_grade(trial).resolved
    _result(trial, None, error={"exception_type": "AgentTimeoutError"})
    with pytest.raises(RuntimeError, match="missing or invalid"):
        evaluate.read_grade(trial)
    _result(trial, 0, error={"exception_type": "VerifierTimeoutError"})
    with pytest.raises(RuntimeError, match="VerifierTimeoutError"):
        evaluate.read_grade(trial)


def test_the_grade_carries_the_collected_patch_and_verifier_logs_for_the_analyst(tmp_path):
    trial = tmp_path / "trial"
    _result(trial, 0)
    (trial / "artifacts" / "logs" / "artifacts").mkdir(parents=True)
    (trial / "artifacts" / "logs" / "artifacts" / "model.patch").write_text("diff --git a b\n")
    (trial / "verifier").mkdir()
    (trial / "verifier" / "test-stdout.txt").write_text("FAILED test_x\n")
    grade = evaluate.read_grade(trial)
    assert grade.patch == "diff --git a b\n"
    assert "FAILED test_x" in grade.detail and '"reward": 0' in grade.detail


def test_root_and_branch_trials_use_the_same_official_task(tmp_path):
    task_dir = Path(__file__).resolve().parents[1] / "harbor_plugin" / "agentenv_fixtures" / "shell-task"
    task = Task(task_dir)
    runtime = tmp_path / "runtime"
    runtime.touch()
    args = SimpleNamespace(
        task_dir=task_dir, dataset_digest="sha256:pinned", output=tmp_path,
        model="fixture", model_endpoint="http://127.0.0.1:18252",
        model_key_env="TEST_KEY", max_output_tokens=64000, max_turns=300, command_timeout=90,
        runtime_bin=runtime, backend="microvm", store=None, sandbox_ttl=36000,
        server_url="http://127.0.0.1:18000", api_key_file=None, reasoning_effort="high",
    )
    root = evaluate.trial_config(args, task, name="parent")
    context = tmp_path / "selected.json"
    branch = evaluate.trial_config(args, task, name="r1b1", agent="branch", branch_context=context)
    assert root.task.path == branch.task.path == task_dir.resolve()
    assert root.agent.kwargs["actor_timeout_s"] == branch.agent.kwargs["actor_timeout_s"] == task.config.agent.timeout_sec
    assert root.agent.kwargs["reasoning_effort"] == branch.agent.kwargs["reasoning_effort"] == "high"
    assert root.agent.kwargs["command_timeout_s"] == branch.agent.kwargs["command_timeout_s"] == 90, \
        "a branch's commands run under the same per-command limit as its parent's"
    assert root.agent_timeout_multiplier == branch.agent_timeout_multiplier == 1
    # Harbor records the dataset the task came from by its name prefix.
    pro = evaluate.trial_config(args, SimpleNamespace(name="swebench-pro/example", config=task.config), name="parent")
    assert pro.task.source == "scaleapi/SWE-bench_Pro-os/v2@sha256:pinned"
    assert root.agent.import_path == "sprout.eval.harbor.agentenv_mini:AgentENVMini"
    assert branch.agent.import_path == "sprout.eval.harbor.branching:BranchMini"
    assert root.environment.import_path == "sprout.eval.harbor.agentenv:AgentENVEnvironment"
    assert branch.environment.import_path == "sprout.eval.harbor.branching:SnapshotEnvironment"
    assert branch.agent.kwargs["branch_context"] == branch.environment.kwargs["branch_context"] == str(context)
    assert root.environment.kwargs["server_url"] == "http://127.0.0.1:18000" and "store" not in root.environment.kwargs


def test_podman_trials_run_the_same_agents_in_the_podman_environment_on_the_store(tmp_path):
    """`--backend podman --store S` swaps only the environment class (and its
    branch counterpart); the agents, the task and the timeouts are the same."""
    task_dir = Path(__file__).resolve().parents[1] / "harbor_plugin" / "agentenv_fixtures" / "shell-task"
    task = Task(task_dir)
    runtime = tmp_path / "runtime"
    runtime.touch()
    args = SimpleNamespace(
        task_dir=task_dir, dataset_digest="sha256:pinned", output=tmp_path, model="fixture",
        model_endpoint="http://127.0.0.1:18252", model_key_env="TEST_KEY", max_output_tokens=64000,
        max_turns=300, command_timeout=60, runtime_bin=runtime, backend="podman", store=tmp_path / "store",
        sandbox_ttl=36000, server_url=None, api_key_file=None, reasoning_effort="high", checkpoint_mode="disk_only",
    )
    root = evaluate.trial_config(args, task, name="parent")
    branch = evaluate.trial_config(args, task, name="r1b1", agent="branch", branch_context=tmp_path / "ctx.json")
    assert root.environment.import_path == "sprout.eval.harbor.podman:PodmanEnvironment"
    assert branch.environment.import_path == "sprout.eval.harbor.branching:PodmanSnapshotEnvironment"
    assert root.environment.kwargs == {"runtime_bin": str(runtime.resolve()), "checkpoint_mode": "disk_only",
                                       "store": str((tmp_path / "store").resolve())}
    assert branch.environment.kwargs["store"] == root.environment.kwargs["store"]
    assert root.agent.import_path == "sprout.eval.harbor.agentenv_mini:AgentENVMini"
    assert branch.agent.import_path == "sprout.eval.harbor.branching:BranchMini"
    for agent in ("oracle", "nop"):
        gate = evaluate.trial_config(args, task, name=agent, agent=agent)
        assert gate.environment.import_path == "sprout.eval.harbor.podman:PodmanEnvironment"
        assert gate.agent.name == agent


def test_the_agent_environments_receipt_is_found_by_backend_type(tmp_path):
    """Each environment writes `<session_id>.<type>.json`, the verifier's included;
    the one for the actor's sandbox on this backend is the trial's. A microVM
    receipt is never mistaken for a podman one, and the reverse."""
    (tmp_path / "t__env.podman.json").write_text(json.dumps(
        {"sandbox_id": "sprout-abc", "checkpoint_mode": "disk_only", "store": "/store", "image": "img@sha256:d"}))
    (tmp_path / "t__verifier__grade.podman.json").write_text(json.dumps(
        {"sandbox_id": "sprout-def", "checkpoint_mode": "disk_only"}))
    (tmp_path / "t__env.agentenv.json").write_text(json.dumps(
        {"sandbox_id": "sprout-abc", "checkpoint_mode": "disk_only"}))
    receipt = evaluate.sandbox_receipt(tmp_path, "podman", "sprout-abc", "disk_only")
    assert receipt["store"] == "/store"
    assert evaluate.sandbox_receipt(tmp_path, "microvm", "sprout-abc", "disk_only") == {
        "sandbox_id": "sprout-abc", "checkpoint_mode": "disk_only"}
    with pytest.raises(RuntimeError, match="podman agent environment receipt"):
        evaluate.sandbox_receipt(tmp_path, "podman", "sprout-abc", "none")
    with pytest.raises(RuntimeError, match="microvm"):
        evaluate.sandbox_receipt(tmp_path, "microvm", "sprout-def", "disk_only")


def test_podman_is_the_default_takes_the_default_store_and_no_memory_snapshot(tmp_path, monkeypatch):
    runtime = tmp_path / "runtime"
    runtime.touch()
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "m", "--model-endpoint", "http://127.0.0.1:1", "--model-key-env", "KEY",
            "--runtime-bin", str(runtime), "--output", str(tmp_path / "out"), "--volatile-ok",
            "--backend", "podman"]
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(tmp_path / "default-store"))
    assert evaluate.parse_args(argv).store == tmp_path / "default-store", "no --store: the default one"
    args = evaluate.parse_args(argv + ["--store", str(tmp_path / "store")])
    assert args.backend == "podman" and args.store == tmp_path / "store"
    with pytest.raises(SystemExit):
        evaluate.parse_args(argv + ["--store", str(tmp_path / "store"), "--final-memory-snapshot", "always"])
    assert evaluate.parse_args(argv[:-2]).backend == "podman", "podman is the default backend"


def test_the_analyst_and_reviewer_may_run_on_another_model_and_endpoint_than_the_actor(tmp_path, monkeypatch):
    """--review-model, --review-endpoint, --review-key-env and --review-effort reach the
    round: the analyst and the reviewer are called through one ReviewTransport built for
    that model, endpoint, key and reasoning effort, while --model stays the actor's. Each
    defaults to the actor's own, so without them both run on the actor's endpoint and
    model. summary.json records which model, endpoint and effort reviewed -- no provider."""
    runtime = tmp_path / "runtime"
    runtime.touch()
    monkeypatch.setenv("KEY", "actor-key")
    monkeypatch.setenv("REVIEW_KEY", "review-key")
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "Qwen3.6-35B-A3B", "--model-endpoint", "http://127.0.0.1:1", "--model-key-env", "KEY",
            "--reasoning-effort", "medium", "--runtime-bin", str(runtime), "--output", str(tmp_path / "out"),
            "--volatile-ok", "--method", "sprout"]
    reviewer = ["--review-model", "claude-opus-5-5", "--review-endpoint", "http://127.0.0.1:4000/v1",
                "--review-key-env", "REVIEW_KEY", "--review-effort", "high"]
    built, calls = [], []

    class Transport:
        def __init__(self, directory, **kwargs):
            built.append(kwargs)

    def analyse(transport, model, *rest, **kwargs):
        calls.append(("analyse", transport, model))
        return {}

    def review(transport, model, *rest, plan_path, **kwargs):
        calls.append(("review", transport, model))
        plan_path.write_text(json.dumps({"validation_error": "fixture"}))
        return []

    monkeypatch.setattr(evaluate, "ReviewTransport", Transport)
    monkeypatch.setattr(evaluate, "analyse", analyse)
    monkeypatch.setattr(evaluate, "review", review)
    monkeypatch.setattr(evaluate, "available_branch_points", lambda journal: {})

    def round_with(extra):
        args = evaluate.parse_args(argv + extra)
        (tmp_path / "out").mkdir(exist_ok=True)
        evaluation = SimpleNamespace(args=args, output=tmp_path / "out", resolved=False, cases={}, state={},
                                     task=SimpleNamespace(instruction="task"),
                                     attempts=[SimpleNamespace(name="parent", outcome=SimpleNamespace(journal_path=None))])
        import asyncio
        asyncio.run(evaluate.Evaluation.sprout(evaluation))
        return args, evaluation.state

    args, state = round_with(reviewer)
    assert args.model == "Qwen3.6-35B-A3B" and args.model_endpoint == "http://127.0.0.1:1", "the actor keeps its own"
    assert [(stage, model) for stage, _, model in calls] == [("analyse", "claude-opus-5-5"), ("review", "claude-opus-5-5")]
    assert len(built) == 1 and calls[0][1] is calls[1][1] and isinstance(calls[0][1], Transport), \
        "one transport, the analyst's and the reviewer's"
    assert built[0]["model"] == "claude-opus-5-5" and built[0]["model_endpoint"] == "http://127.0.0.1:4000/v1"
    assert built[0]["api_key_env"] == "REVIEW_KEY" and built[0]["reasoning_effort"] == "high"
    assert built[0]["json_directive"] == args.prompts.transport_json_directive
    assert state["stop_reason"] == "fixture"
    calls.clear()
    round_with([])
    assert [(stage, model) for stage, _, model in calls] == [("analyse", "Qwen3.6-35B-A3B"), ("review", "Qwen3.6-35B-A3B")]
    assert built[-1]["model"] == "Qwen3.6-35B-A3B" and built[-1]["model_endpoint"] == "http://127.0.0.1:1"
    assert built[-1]["api_key_env"] == "KEY" and built[-1]["reasoning_effort"] == "medium", "the actor's, by default"
    # The summary says which model, endpoint and effort reviewed: no provider, never a key.
    (tmp_path / "task.toml").write_text("")
    task = SimpleNamespace(name="t")
    recorded = evaluate.Evaluation(evaluate.parse_args(argv + reviewer), task).state["review"]
    assert recorded["model"] == "claude-opus-5-5" and recorded["endpoint"] == "http://127.0.0.1:4000/v1"
    assert recorded["effort"] == "high" and "provider" not in recorded
    assert "review-key" not in json.dumps(recorded)
    recorded = evaluate.Evaluation(evaluate.parse_args(argv), task).state["review"]
    assert recorded["model"] == "Qwen3.6-35B-A3B" and recorded["endpoint"] == "http://127.0.0.1:1"
    assert recorded["effort"] == "medium"


def test_leak_judge_checks_each_reviewer_turn_against_the_tasks_hidden_tests(tmp_path, monkeypatch):
    """--leak-judge reaches the round: review() gets a turn_check, which judges a
    turn against the task's test patch, the verifier output the reviewer read
    and h_s at the turn's point, and is closed after the round."""
    from sprout.worker.jobs.review.leak.judge import LeakJudgement
    runtime = tmp_path / "runtime"
    runtime.touch()
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test.patch").write_text(
        "diff --git a/tests/test_x.py b/tests/test_x.py\n+++ b/tests/test_x.py\n+    assert f('FIXTURE-9') == 1\n")
    (tmp_path / "tests" / "config.json").write_text(json.dumps({"f2p_node_ids": ["tests/test_x.py::test_secret_case"]}))
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "m", "--model-endpoint", "http://127.0.0.1:1", "--model-key-env", "KEY",
            "--runtime-bin", str(runtime), "--output", str(tmp_path / "out"), "--volatile-ok", "--method", "sprout"]
    monkeypatch.delenv("TS", raising=False)
    with pytest.raises(SystemExit):   # no key: refused before anything runs
        evaluate.parse_args(argv + ["--leak-judge", "--leak-judge-key-env", "TS"])
    monkeypatch.setenv("TS", "k")
    with pytest.raises(SystemExit):   # only reviewer-written turns are checked
        evaluate.parse_args(argv[:-2] + ["--method", "baseline", "--leak-judge", "--leak-judge-key-env", "TS"])
    with pytest.raises(SystemExit):
        evaluate.parse_args(argv + ["--leak-judge", "--leak-judge-key-env", "TS", "--branch-guidance", "user-hint"])

    judges = []

    class FakeJudge:
        def __init__(self, key, **settings):
            self.key, self.settings, self.seen, self.closed = key, settings, [], False
            judges.append(self)

        def judge(self, turn, *, task_statement, hidden, visible):
            self.seen.append((task_statement, hidden, visible))
            score = 0.9 if "FIXTURE-9" in json.dumps(turn) else 0.0
            return LeakJudgement(score, (("FIXTURE-9", score),), "jev-fake")

        def close(self):
            self.closed = True

    reviewed = []

    def review(transport, model, *rest, plan_path, turn_check=None, turn_resamples=0, **kwargs):
        choice = SimpleNamespace(base=SimpleNamespace(name="parent", outcome=SimpleNamespace(journal_path=None)),
                                 checkpoint=SimpleNamespace(step=2), assistant_turn={
                                     "role": "assistant", "content": "check", "tool_calls": [{
                                         "id": "reviewer-r1-b1-c1", "type": "function", "function": {
                                             "name": "bash", "arguments": '{"command": "python -c \\"f(\'FIXTURE-9\')\\""}'}}]})
        reviewed.append({"turn_check": turn_check, "turn_resamples": turn_resamples,
                         "check": turn_check(choice) if turn_check else None})
        plan_path.write_text(json.dumps({"validation_error": "fixture"}))
        return []

    monkeypatch.setattr(evaluate.leak_judge, "JevLeakJudge", FakeJudge)
    monkeypatch.setattr(evaluate, "ReviewTransport", lambda directory, **kwargs: None)
    monkeypatch.setattr(evaluate, "analyse", lambda *a, **k: {})
    monkeypatch.setattr(evaluate, "review", review)
    monkeypatch.setattr(evaluate, "available_branch_points", lambda journal: {})
    monkeypatch.setattr(evaluate, "selected_prefix", lambda journal, checkpoint: [
        {"type": "mini.message", "message": {"role": "user", "content": "the task, as the actor saw it"}}])

    def round_with(extra):
        args = evaluate.parse_args(argv + extra)
        (tmp_path / "out").mkdir(exist_ok=True)
        evaluation = SimpleNamespace(
            args=args, output=tmp_path / "out", resolved=False, cases={}, state={},
            task=SimpleNamespace(instruction="Fix f."),
            attempts=[SimpleNamespace(name="parent", outcome=SimpleNamespace(journal_path=None),
                                      grade=SimpleNamespace(detail="FAILED tests/test_x.py::test_secret_case"))])
        evaluation.turn_check = lambda: evaluate.Evaluation.turn_check(evaluation)
        import asyncio
        asyncio.run(evaluate.Evaluation.sprout(evaluation))
        return evaluation

    round_with(["--leak-judge", "--leak-judge-key-env", "TS", "--leak-judge-resamples", "3"])
    (seen,) = reviewed
    assert seen["turn_resamples"] == 3 and seen["check"]["leak"] is True and seen["check"]["jev"]["score"] == 0.9
    (judge,) = judges
    assert judge.key == "k" and judge.closed, "a judge per round, closed after it"
    assert judge.settings["timeout_s"] * judge.settings["max_attempts"] <= 120
    statement, hidden, visible = judge.seen[0]
    assert statement == "Fix f." and "FIXTURE-9" in hidden.test_patch
    assert hidden.fail_to_pass == ("tests/test_x.py::test_secret_case",)
    assert "FAILED tests/test_x.py::test_secret_case" in hidden.verifier_output
    assert "the task, as the actor saw it" in visible
    # Without --leak-judge nothing is checked and nothing is recorded.
    reviewed.clear()
    round_with([])
    assert reviewed[0]["turn_check"] is None and reviewed[0]["turn_resamples"] == 0
    assert len(judges) == 1
    # The summary records the judge only for a judged run.
    (tmp_path / "task.toml").write_text("")
    task = SimpleNamespace(name="t")
    judged = evaluate.Evaluation(evaluate.parse_args(argv + ["--leak-judge", "--leak-judge-key-env", "TS"]), task)
    assert judged.state["review"]["leak_judge"] == {"threshold": 0.75, "model": "jev-1.13.0", "resamples": 2}
    assert "leak_judge" not in evaluate.Evaluation(evaluate.parse_args(argv), task).state["review"]
    # A task with nothing hidden to check against leaves every turn unchecked, never clean.
    (tmp_path / "tests" / "test.patch").unlink()
    (tmp_path / "tests" / "config.json").unlink()
    reviewed.clear()
    round_with(["--leak-judge", "--leak-judge-key-env", "TS"])
    assert reviewed[0]["check"]["leak"] is None


def test_prompts_option_loads_the_file_once_and_the_summary_records_its_hash(tmp_path):
    """--prompts names the YAML every analyst, reviewer and selector prompt comes from.
    parse_args loads it (a bad file fails before any sandbox starts) and the summary
    state records its path and sha256 -- the packaged file's by default."""
    import yaml
    from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH

    runtime = tmp_path / "runtime"
    runtime.touch()
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "m", "--model-endpoint", "http://127.0.0.1:1", "--model-key-env", "KEY",
            "--runtime-bin", str(runtime), "--output", str(tmp_path / "out"), "--volatile-ok"]
    args = evaluate.parse_args(argv)
    assert args.prompts == DEFAULT_PROMPTS and args.prompts.path == DEFAULT_PROMPTS_PATH
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["shepherd"]["system"] += "\nEdited for this test."
    copy = tmp_path / "prompts.yaml"
    copy.write_text(yaml.safe_dump(data))
    args = evaluate.parse_args(argv + ["--prompts", str(copy)])
    assert args.prompts.path == copy and args.prompts.shepherd_system.endswith("Edited for this test.")
    assert args.prompts.sha256 == hashlib.sha256(copy.read_bytes()).hexdigest() != DEFAULT_PROMPTS.sha256
    (tmp_path / "task.toml").write_text("[task]\nname='org/task'\n")
    state = evaluate.Evaluation(args, SimpleNamespace(name="org/task")).state
    assert state["prompts"] == {"path": str(copy), "sha256": args.prompts.sha256}
    del data["analyst"]["case"]
    copy.write_text(yaml.safe_dump(data))
    for bad in (copy, tmp_path / "absent.yaml"):
        with pytest.raises(SystemExit):
            evaluate.parse_args(argv + ["--prompts", str(bad)])


def test_preparation_membership_requires_the_exact_task_and_dataset(tmp_path):
    task_dir = Path(__file__).resolve().parents[1] / "harbor_plugin" / "agentenv_fixtures" / "shell-task"
    task = Task(task_dir)
    digest = hashlib.sha256((task_dir / "task.toml").read_bytes()).hexdigest()
    path = tmp_path / "manifest.json"
    row = {"name": task.name, "path": str(task_dir.resolve()), "task_toml_sha256": digest}
    path.write_text(json.dumps({"dataset_digest": "sha256:pinned", "tasks": [row]}))
    args = SimpleNamespace(preparation_manifest=path, dataset_digest="sha256:pinned", task_dir=task_dir)
    evaluate._verify_prepared_task(args, task, digest)
    args.dataset_digest = "sha256:different"
    with pytest.raises(ValueError, match="digest"):
        evaluate._verify_prepared_task(args, task, digest)
    args.dataset_digest = "sha256:pinned"
    row["task_toml_sha256"] = "changed"
    path.write_text(json.dumps({"dataset_digest": args.dataset_digest, "tasks": [row]}))
    with pytest.raises(ValueError, match="definition changed"):
        evaluate._verify_prepared_task(args, task, digest)


def test_review_transport_is_plain_text_unless_told_the_endpoint_takes_json_schema(tmp_path, monkeypatch):
    """Structured review is an explicit switch, not something read off the URL: the
    default transport leaves the analyst and reviewer on fenced plain-text plans."""
    common = dict(model="fixture", model_endpoint="http://127.0.0.1:4000/v1", api_key_env="TEST_KEY",
                  max_output_tokens=1000)
    assert ReviewTransport(tmp_path / "plain", **common).supports_json_schema is False
    assert ReviewTransport(tmp_path / "structured", supports_json_schema=True, **common).supports_json_schema is True
    monkeypatch.delenv("TEST_KEY", raising=False)
    with pytest.raises(ValueError, match="TEST_KEY is unset"):
        ReviewTransport(tmp_path / "unset", **common)("fixture", "prompt")
    assert not list((tmp_path / "unset").glob("review-*.json")), "no receipt for a request never sent"


def test_structured_review_flag_reaches_the_transport_and_the_summary(tmp_path, monkeypatch):
    """``--structured-review`` builds the ReviewTransport with ``supports_json_schema``
    and is recorded in summary.json; off by default, and meaningless (recorded False)
    for a method with no reviewer."""
    import asyncio

    runtime = tmp_path / "runtime"
    runtime.touch()
    (tmp_path / "task.toml").write_text("[task]\nname='org/task'\n")
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "m", "--model-endpoint", "http://127.0.0.1:4000/v1", "--model-key-env", "KEY",
            "--runtime-bin", str(runtime), "--output", str(tmp_path / "out"), "--volatile-ok", "--rounds", "0"]
    received = []

    class Transport:
        def __init__(self, *_, **kwargs):
            received.append(kwargs.get("supports_json_schema"))

    monkeypatch.setattr(evaluate, "ReviewTransport", Transport)
    task = SimpleNamespace(name="org/task", instruction="Fix it.")
    for extra, expected in (([], False), (["--structured-review"], True)):
        args = evaluate.parse_args(argv + extra)
        assert args.structured_review is expected
        evaluation = evaluate.Evaluation(args, task)
        assert evaluation.state["structured_review"] is expected
        # No round: the sprout loop only builds its transport.
        asyncio.run(evaluation.sprout())
    assert received == [False, True]
    bpo = evaluate.parse_args(argv + ["--structured-review", "--method", "bpo", "--parent-output", str(tmp_path)])
    assert evaluate.Evaluation(bpo, task).state["structured_review"] is False, "bpo has no reviewer"


def test_review_transport_keeps_keys_out_of_receipts_and_rejects_model_switch(tmp_path, monkeypatch):
    transport = ReviewTransport(tmp_path, model="fixture",
                                model_endpoint="http://127.0.0.1:18252",
                                api_key_env="TEST_KEY", max_output_tokens=1000)
    monkeypatch.setenv("TEST_KEY", "super-secret")
    with pytest.raises(ValueError, match="differs"):
        transport("other-model", "prompt")
    requests = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "valid"}}]}

    def post(url, **kwargs):
        requests.append((url, kwargs))
        return Response()

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", post)
    assert transport("fixture", "prompt") == "valid"
    assert requests[0][0].endswith("/v1/chat/completions")
    assert requests[0][1]["headers"]["Authorization"] == "Bearer super-secret"
    assert "super-secret" not in (tmp_path / "review-001.json").read_text()


def test_review_transport_retries_a_timeout_once_with_half_the_output_budget(tmp_path, monkeypatch):
    """A request that runs to the timeout spent its whole output budget reasoning; the
    retry asks for half, in its own receipt that names the one it repeats."""
    import httpx
    transport = ReviewTransport(tmp_path, model="fixture", model_endpoint="http://127.0.0.1:18252",
                                api_key_env="TEST_KEY", max_output_tokens=64000, timeout_s=7)
    monkeypatch.setenv("TEST_KEY", "k")
    budgets = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "plan"}}]}

    def post(url, **kwargs):
        budgets.append(kwargs["json"]["max_tokens"])
        if len(budgets) == 1:
            raise httpx.ReadTimeout("timed out")
        return Response()

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", post)
    assert transport("fixture", "prompt") == "plan"
    assert budgets == [64000, 32000]
    first = json.loads((tmp_path / "review-001.json").read_text())
    second = json.loads((tmp_path / "review-002.json").read_text())
    assert first["error"].startswith("ReadTimeout") and first["timeout_s"] == 7 and first["request"]["max_tokens"] == 64000
    assert second["retry_of"] == "review-001.json" and second["request"]["max_tokens"] == 32000 and second["text"] == "plan"

    def always_slow(url, **kwargs):
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", always_slow)
    with pytest.raises(httpx.ReadTimeout):
        transport("fixture", "prompt")
    assert sorted(p.name for p in tmp_path.glob("review-*.json")) == [f"review-00{i}.json" for i in range(1, 5)], \
        "two tries, then the error propagates"


def test_review_transport_appends_and_counts_the_directive_it_was_given(tmp_path, monkeypatch):
    """A structured request carries the transport's JSON directive after the prompt, and
    json_overhead counts that same text -- the packaged directive by default, the loaded
    prompts' when evaluate passes one -- so the characters the callers reserve are the
    ones sent."""
    sent = []

    class Response:
        def raise_for_status(self):
            pass

        def json(self):
            return {"choices": [{"message": {"content": "{}"}}]}

    def post(url, **kwargs):
        sent.append(kwargs["json"])
        return Response()

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", post)
    monkeypatch.setenv("TEST_KEY", "k")
    common = dict(model="fixture", model_endpoint="http://127.0.0.1:18252", api_key_env="TEST_KEY",
                  max_output_tokens=1000)
    default = ReviewTransport(tmp_path / "default", **common)
    edited = ReviewTransport(tmp_path / "edited", json_directive="\n\nEDITED DIRECTIVE.", **common)
    assert default.json_directive == DEFAULT_PROMPTS.transport_json_directive
    schema, name = structured_review.ANALYST_SCHEMA, "sprout_analysis"
    for transport in (default, edited):
        assert transport.json("fixture", "prompt", schema=schema, name=name) == "{}"
        assert transport.json_overhead(schema, name) == (
            len(transport.json_directive) + len(json.dumps(transport.json_format(schema, name))))
    first, second = (request["messages"][0]["content"] for request in sent)
    assert first == "prompt" + DEFAULT_PROMPTS.transport_json_directive
    assert second == "prompt\n\nEDITED DIRECTIVE."
    assert [request["response_format"] for request in sent] == [default.json_format(schema, name)] * 2
    assert edited.json_overhead(schema, name) - default.json_overhead(schema, name) == (
        len("\n\nEDITED DIRECTIVE.") - len(DEFAULT_PROMPTS.transport_json_directive))
    assert json.loads((tmp_path / "edited" / "review-001.json").read_text())["request"]["prompt"] == second


def test_review_transport_retries_transient_gateway_replies_with_the_same_budget(tmp_path, monkeypatch):
    """A 502 or a dropped connection says nothing about the request: it is sent
    again after a pause, each try its own receipt naming the one it repeats and
    why; a 400 is the request's fault and is not retried here."""
    import httpx
    from sprout.worker.jobs.review import transport as review_transport
    from sprout.worker.jobs.review.branch_plan import PromptRejected
    transport = ReviewTransport(tmp_path, model="fixture", model_endpoint="http://127.0.0.1:18252",
                                api_key_env="TEST_KEY", max_output_tokens=4000)
    monkeypatch.setenv("TEST_KEY", "k")
    naps, budgets = [], []
    monkeypatch.setattr(review_transport.time, "sleep", lambda s: naps.append(s))

    request = httpx.Request("POST", "http://127.0.0.1:18252/v1/chat/completions")

    def status(code):
        """What httpx.post returns for a gateway reply; raise_for_status() raises on 4xx/5xx."""
        return httpx.Response(code, request=request, text="Bad Gateway")

    answer = httpx.Response(200, request=request, json={"choices": [{"message": {"content": "plan"}}]})
    replies = iter([status(502), httpx.RemoteProtocolError("peer closed"), status(503), answer])

    def post(url, **kwargs):
        budgets.append(kwargs["json"]["max_tokens"])
        outcome = next(replies)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", post)
    assert transport("fixture", "prompt") == "plan"
    assert budgets == [4000] * 4, "the output budget is only halved for a timeout, never for an outage"
    assert naps == list(review_transport.TRANSIENT_DELAYS_S)
    receipts = [json.loads((tmp_path / f"review-00{i}.json").read_text()) for i in range(1, 5)]
    assert [r.get("retry_of") for r in receipts] == [None, "review-001.json", "review-002.json", "review-003.json"]
    assert [r.get("retry_reason") for r in receipts] == [None, "http 502", "RemoteProtocolError", "http 503"]
    assert receipts[0]["error"].startswith("HTTPStatusError") and receipts[3]["text"] == "plan"

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", lambda url, **kw: status(502))
    with pytest.raises(httpx.HTTPStatusError):
        transport("fixture", "prompt")
    assert len(list(tmp_path.glob("review-*.json"))) == 8, "three retries, then the error propagates"

    monkeypatch.setattr("sprout.worker.jobs.review.transport.httpx.post", lambda url, **kw: status(400))
    naps.clear()
    with pytest.raises(PromptRejected):
        transport("fixture", "prompt")
    assert naps == [] and len(list(tmp_path.glob("review-*.json"))) == 9, "a refused request is not an outage"


def test_gate_agents_are_harbors_own_on_our_environment(tmp_path):
    """The oracle and nop trials exercise exactly the environment the evaluations use."""
    task_dir = Path(__file__).resolve().parents[1] / "harbor_plugin" / "agentenv_fixtures" / "shell-task"
    task = Task(task_dir)
    runtime = tmp_path / "runtime"
    runtime.touch()
    args = SimpleNamespace(task_dir=task_dir, dataset_digest="gate", output=tmp_path, checkpoint_mode="none",
                           runtime_bin=runtime, backend="microvm", store=None, sandbox_ttl=36000,
                           server_url=None, api_key_file=None)
    for agent in ("oracle", "nop"):
        config = evaluate.trial_config(args, task, name=agent, agent=agent)
        assert config.agent.name == agent and config.agent.import_path is None
        assert config.environment.import_path == "sprout.eval.harbor.agentenv:AgentENVEnvironment"
        assert config.environment.kwargs["checkpoint_mode"] == "none"
    with pytest.raises(ValueError, match="branch"):
        evaluate.trial_config(args, task, name="x", agent="branch")


def test_a_volatile_output_directory_is_refused(tmp_path):
    """The trials under --output are the evaluation's only record."""
    runtime = tmp_path / "runtime"
    runtime.touch()
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "m", "--model-endpoint", "http://127.0.0.1:1", "--model-key-env", "KEY",
            "--runtime-bin", str(runtime), "--output", "/tmp/doomed"]
    with pytest.raises(SystemExit):
        evaluate.parse_args(argv)
    assert evaluate.parse_args(argv + ["--volatile-ok"]).output == Path("/tmp/doomed")


# --- the memory snapshot after grading -----------------------------------------------
def _graded_trial(directory: Path, *, memory_id="mem-1"):
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "agent").mkdir(exist_ok=True)
    (directory / "agent" / "snapshot.json").write_text(json.dumps(
        {"final_snapshot_id": "disk-1", "final_memory_snapshot_id": memory_id, "sandbox_id": "sb"}))
    (directory / "agent" / "trajectory.jsonl").write_text(
        json.dumps({"v": 2, "type": "run.started", "seq": 0}) + "\n"
        + json.dumps({"v": 2, "type": "checkpoint.captured", "seq": 1, "step": 1, "snapshot_id": "disk-1",
                      "reason": "captured", "final": True}) + "\n")
    return directory


class _Client:
    def __init__(self, *args, **kwargs):
        self.deleted = []

    def delete_snapshot(self, snapshot_id):
        _Client.deleted.append(snapshot_id)
        return _Client.ok


@pytest.mark.parametrize("policy,resolved,memory_id,ok,expected,deleted", [
    ("on-failure", True, "mem-1", True, "released", ["mem-1"]),
    ("on-failure", False, "mem-1", True, "kept", []),
    ("always", True, "mem-1", True, "kept", []),
    # The DELETE fails: the grade stands either way; the marker must only ever say what happened.
    ("on-failure", True, "mem-1", False, "release_failed", ["mem-1"]),
    # No memory snapshot was taken (every patch-graded dataset): nothing to release, no client call.
    ("on-failure", True, None, True, None, []),
])
def test_on_failure_releases_only_a_resolved_attempts_memory_snapshot(
        tmp_path, monkeypatch, policy, resolved, memory_id, ok, expected, deleted):
    _Client.deleted, _Client.ok = [], ok
    monkeypatch.setattr(evaluate, "AgentEnvClient", _Client)
    args = SimpleNamespace(final_memory_snapshot=policy, server_url="http://aenv", api_key_file=None)
    trial = _graded_trial(tmp_path / "trial", memory_id=memory_id)
    grade = evaluate.Grade(resolved=resolved, reward=1.0 if resolved else 0.0)
    assert evaluate.apply_memory_snapshot_policy(args, trial, grade) == expected
    assert _Client.deleted == deleted
    records = [json.loads(l) for l in (trial / "agent" / "trajectory.jsonl").read_text().splitlines()]
    released = [r for r in records if r["type"] == "snapshot.released"]
    if expected == "released":
        assert released[0]["snapshot_id"] == "mem-1" and released[0]["seq"] == 2
        # Idempotent: a re-read after the fact neither deletes again nor forgets.
        assert evaluate.apply_memory_snapshot_policy(args, trial, grade) == "released"
        assert _Client.deleted == deleted
        assert evaluate.memory_snapshot_state(trial) == ("mem-1", "released")
    else:
        assert not released and evaluate.memory_snapshot_state(trial) == (memory_id, "kept" if memory_id else None)

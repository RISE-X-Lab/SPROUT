import json
from types import SimpleNamespace

import pytest

from examples.qwen import run
from examples.qwen.report import phase_metrics, usage


def args(tmp_path, method="all", baseline_rollouts=4):
    return SimpleNamespace(output=tmp_path, baseline_root=None, method=method,
                           dataset="deepswe", max_rollouts=8, baseline_rollouts=baseline_rollouts)


def test_an_unset_model_key_fails_before_any_stage_is_launched(tmp_path, monkeypatch):
    """The key is the only model credential (no provider is recognised from the URL, no
    credential chain stands in): a variable that is not set fails ``main`` before the
    output directory, the manifest or any trial exists."""
    (tmp_path / "fixture").mkdir()
    (tmp_path / "fixture" / "task.toml").write_text("[task]\nname='org/fixture'\n")
    base = ["--dataset", "deepswe", "--method", "baseline", "--tasks-dir", str(tmp_path),
            "--task", "fixture", "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "runtime"),
            "--model-endpoint", "http://127.0.0.1:4000/v1", "--model", "luna"]
    assert run.parse_args(base).model_key_env == "QWEN_API_KEY", "the Qwen experiments' default"
    with pytest.raises(SystemExit):
        run.parse_args(base + ["--model-key-env", "not-a-name"])
    monkeypatch.delenv("UNSET_FIXTURE_KEY", raising=False)
    monkeypatch.setattr(run.subprocess, "check_output", lambda *a, **k: "0" * 40 + "\n")
    monkeypatch.setattr(run, "run_task", lambda *a: pytest.fail("a stage was launched without a model key"))
    with pytest.raises(ValueError, match="UNSET_FIXTURE_KEY is unset"):
        run.main(base + ["--model-key-env", "UNSET_FIXTURE_KEY"])
    assert not (tmp_path / "out").exists(), "nothing was written before the key check"
    assert run.main(base + ["--model-key-env", "UNSET_FIXTURE_KEY", "--plan"]) == 0, "--plan needs no key"


def test_structured_review_reaches_the_evaluator_and_the_manifest(tmp_path):
    """``--structured-review`` is an explicit switch, off by default, passed to every
    ``methods.evaluate`` and recorded in the manifest; the baseline has no reviewer,
    so it is not a setting a shared baseline has to match."""
    from sprout.methods import evaluate

    base = ["--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path),
            "--task", "fixture", "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "runtime"),
            "--model-endpoint", "http://127.0.0.1:4000/v1", "--model", "luna"]
    for extra, expected in (([], False), (["--structured-review"], True)):
        parsed = run.parse_args(base + extra)
        parsed.digest, parsed.source_commit = "sha256:fixture", "0" * 40
        parsed.preparation = tmp_path / "preparation.json"
        assert parsed.structured_review is expected
        for method in run.METHODS:
            command = run.command(parsed, "fixture", tmp_path / method, method,
                                  tmp_path / "parent" if method != "baseline" else None)
            assert ("--structured-review" in command) is expected
            assert evaluate.parse_args(command[4:] + ["--volatile-ok"]).structured_review is expected
        assert run.manifest_for(parsed)["structured_review"] is expected
    assert "structured_review" not in run.SHARED_BASELINE_KEYS


@pytest.mark.parametrize("width", [1, 2, 4, 8, 16, 32])
def test_fixed_single_round_scaling_reaches_the_evaluator(tmp_path, width):
    from sprout.methods import evaluate

    parsed = run.parse_args([
        "--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path),
        "--task", "fixture", "--output", str(tmp_path / "out"),
        "--runtime-bin", str(tmp_path / "runtime"),
        "--model-endpoint", "http://127.0.0.1:4000/v1", "--model", "luna",
        "--max-rollouts", str(width + 1), "--branches", str(width), "--branch-count-mode", "fixed",
    ])
    parsed.digest = "sha256:fixture"
    parsed.preparation = tmp_path / "preparation.json"
    command = run.command(parsed, "fixture", tmp_path / "evaluation", "sprout", tmp_path / "parent")
    args = evaluate.parse_args(command[4:] + ["--volatile-ok"])
    assert args.rounds == 1 and args.branches == [width]
    assert args.max_rollouts == width + 1 and args.branch_count_mode == "fixed"


def test_the_prompts_file_reaches_the_evaluator_and_its_hash_is_in_the_manifest(tmp_path):
    import hashlib
    import yaml
    from sprout.methods import evaluate
    from sprout.methods.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH

    base = ["--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path),
            "--task", "fixture", "--output", str(tmp_path / "out"),
            "--runtime-bin", str(tmp_path / "runtime"),
            "--model-endpoint", "http://127.0.0.1:4000/v1", "--model", "luna"]
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["reviewer"]["user_hint"] += "Edited for this test.\n"
    edited = tmp_path / "edited.yaml"
    edited.write_text(yaml.safe_dump(data))
    edited_sha = hashlib.sha256(edited.read_bytes()).hexdigest()
    assert edited_sha != DEFAULT_PROMPTS.sha256
    copy = tmp_path / "prompts.yaml"
    copy.write_bytes(DEFAULT_PROMPTS_PATH.read_bytes())
    manifests = {}
    for extra, sha in (([], DEFAULT_PROMPTS.sha256), (["--prompts", str(edited)], edited_sha),
                       (["--prompts", str(copy)], DEFAULT_PROMPTS.sha256)):  # same bytes, same hash
        parsed = run.parse_args(base + extra)
        parsed.digest, parsed.source_commit = "sha256:fixture", "0" * 40
        parsed.preparation = tmp_path / "preparation.json"
        # Every method's command, not only sprout's: Shepherd's system prompt and the
        # transport directive come from the file too, so a BPO/Shepherd run on the
        # packaged prompts under a manifest claiming the edited hash would lie quietly.
        for method in run.METHODS:
            command = run.command(parsed, "fixture", tmp_path / method, method,
                                  tmp_path / "parent" if method != "baseline" else None)
            assert ("--prompts" in command) == bool(extra), "absent, the evaluator reads its packaged file"
            if extra:
                assert command[command.index("--prompts") + 1] == extra[-1]
            assert evaluate.parse_args(command[4:] + ["--volatile-ok"]).prompts.sha256 == sha
        manifests[sha] = run.manifest_for(parsed)
        assert manifests[sha]["prompts_sha256"] == sha
    # The baseline has no reviewer: a SPROUT run under edited prompts may share a
    # baseline run under the packaged ones, and one whose manifest predates the key.
    assert "prompts_sha256" not in run.SHARED_BASELINE_KEYS
    prior = {key: value for key, value in manifests[DEFAULT_PROMPTS.sha256].items() if key != "prompts_sha256"}
    assert run.shared_baseline_mismatch(prior, manifests[edited_sha]) is None
    with pytest.raises(SystemExit):
        run.parse_args(base + ["--prompts", str(tmp_path / "absent.yaml")])
    del data["reviewer"]
    edited.write_text(yaml.safe_dump(data))
    with pytest.raises(SystemExit):
        run.parse_args(base + ["--prompts", str(edited)])


def test_wide_schedule_still_has_a_finite_budget(tmp_path):
    command = [
        "--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path),
        "--task", "fixture", "--output", str(tmp_path / "out"),
        "--runtime-bin", str(tmp_path / "runtime"), "--model-endpoint", "http://localhost:1",
    ]
    with pytest.raises(SystemExit):
        run.parse_args(command + ["--max-rollouts", "33", "--branches", "32", "1"])
    with pytest.raises(SystemExit):
        run.parse_args(command + ["--max-rollouts", "129"])


def test_unicode_line_separators_do_not_split_journal_records(tmp_path):
    text = "before\u20281\u2028after\u00852\u0085tail"
    rows = [
        {"type": "run.started"},
        {"type": "raw.mini-swe-agent", "response": {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7}}},
        {"type": "checkpoint.captured", "snapshot_id": "snapshot-kept", "note": text},
    ]
    (tmp_path / "trajectory.jsonl").write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n")
    # Unrelated artifacts under the same trial need not be journal objects.
    (tmp_path / "numbers.jsonl").write_text("1\n2\n")
    assert run.snapshot_ids(tmp_path) == {"snapshot-kept"}
    metrics = phase_metrics([tmp_path])
    assert metrics["usage"]["known_input_tokens"] == 11
    assert metrics["usage"]["known_output_tokens"] == 7
    assert metrics["model_calls"] == 1


def test_one_baseline_rollout_is_the_parent_a_branch_method_starts_from(tmp_path, monkeypatch):
    """--baseline-rollouts 1 is pass@1 alone: the one rollout is retained as the
    parent and the branch method follows it at once."""
    calls = []

    def stage(a, t, output, method, *rest, **kw):
        calls.append((method, output.name))
        return [{"resolved": False}] + ([{"resolved": True}] if method == "sprout" else [])

    monkeypatch.setattr(run, "stage", stage)
    monkeypatch.setattr(run, "retain_parent", lambda out: run.write(out / "retained-checkpoints.json", {}))
    result = run.run_task(args(tmp_path, method="sprout", baseline_rollouts=1), "task")
    assert result["complete"] and calls == [("baseline", "attempt-1"), ("sprout", "sprout")]
    assert len(result["baseline"]["attempts"]) == 1 and result["methods"]["sprout"]["attempts"] == [{"resolved": True}]
    assert run.parse_args(["--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path), "--task", "t",
                           "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "rt"),
                           "--model-endpoint", "http://localhost:9000", "--store", str(tmp_path / "store"),
                           "--baseline-rollouts", "1"]).baseline_rollouts == 1


def test_a_baseline_of_one_rollout_is_reported_as_pass_at_1(tmp_path):
    from examples.qwen.report import build_report
    run.write(tmp_path / "manifest-baseline.json", {"method": "baseline", "tasks": ["a"],
                                                    "baseline_rollouts": 1, "max_total_method_rollouts": 8})
    run.write(tmp_path / "tasks/a/baseline/attempt-1/summary.json", {"attempts": [{"resolved": False}]})
    (tmp_path / "tasks/a/baseline/attempt-1/journal.jsonl").write_text("")
    report = build_report(tmp_path, "baseline")
    row = next(r for r in report["comparison"] if r["method"].endswith("(baseline)"))
    assert row["method"] == "pass@1 (baseline)" and row["Resolve Rate"] == 0
    assert report["methods"]["baseline"]["tasks"][0]["attempt_budget"] == 1


def test_success_still_runs_four_and_skips_all_branches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(run, "stage", lambda a, t, o, m, *rest, **kw:
                        calls.append(m) or [{"resolved": True}])
    result = run.run_task(args(tmp_path), "task")
    assert result["complete"] and calls == ["baseline"] * 4
    assert all(s["status"] == "skipped_initial_success" for s in result["methods"].values())


def test_initial_failure_stays_shared_even_if_later_baseline_passes(tmp_path, monkeypatch):
    calls = []
    def stage(a, t, output, method, *rest, **kw):
        calls.append(method)
        if method == "baseline":
            return [{"resolved": output.name != "attempt-1"}]
        return ([{"resolved": False}] if method == "sprout" else []) + [{"resolved": True}]
    def retain(out):
        run.write(out / "retained-checkpoints.json", {"keep": True})
    monkeypatch.setattr(run, "stage", stage)
    monkeypatch.setattr(run, "retain_parent", retain)
    monkeypatch.setattr(run, "snapshot_ids", lambda *a: {"shared-first"})
    result = run.run_task(args(tmp_path), "task")
    assert result["complete"] and calls == ["baseline"] * 4 + ["sprout", "bpo", "shepherd"]
    assert (tmp_path / "tasks/task/baseline/attempt-1/retained-checkpoints.json").exists()


def test_failure_never_deletes_shared_parent(tmp_path, monkeypatch):
    def stage(a, t, out, method, *rest, **kw):
        if method != "baseline":
            raise RuntimeError("network error")
        return [{"resolved": False}]
    monkeypatch.setattr(run, "stage", stage)
    monkeypatch.setattr(run, "retain_parent", lambda out: run.write(out / "retained-checkpoints.json", {}))
    monkeypatch.setattr(run, "snapshot_ids", lambda *a: {"parent"})
    result = run.run_task(args(tmp_path), "task")
    assert not result["complete"]
    assert (tmp_path / "tasks/task/baseline/attempt-1/retained-checkpoints.json").exists()


def test_unknown_cost_is_not_zero_and_cached_subset_preserved():
    u = usage({"prompt_tokens": 100, "completion_tokens": 20,
               "prompt_tokens_details": {"cached_tokens": 75}})
    assert u["cost_usd"] is None and u["cached_input_tokens"] == 75


def test_a_timed_out_rollout_is_graded_but_an_error_or_failed_stage_is_not(tmp_path, monkeypatch):
    import pytest
    a = SimpleNamespace(dataset="deepswe", source_commit="pinned", server_url="http://localhost",
                        api_key_file=None, max_rollouts=8, backend="microvm")
    monkeypatch.setattr(run, "command", lambda *args: ["fixture"])
    output = tmp_path / "attempt"
    def completed_negative(*args, **kwargs):
        run.write(output / "summary.json", {"attempts": [{"status": "timeout", "resolved": False}]})
        return 0
    monkeypatch.setattr(run.subprocess, "call", completed_negative)
    assert run.stage(a, "task", output, "baseline")[0]["resolved"] is False
    output = tmp_path / "broken"
    def broken(*args, **kwargs):
        run.write(output / "summary.json", {"attempts": [{"status": "error", "resolved": False}]})
        return 0
    monkeypatch.setattr(run.subprocess, "call", broken)
    with pytest.raises(ValueError, match="infrastructure"):
        run.stage(a, "task", output, "baseline")
    output = tmp_path / "exited"
    monkeypatch.setattr(run.subprocess, "call", lambda *args, **kwargs: 2)
    with pytest.raises(RuntimeError, match="exited 2"):
        run.stage(a, "task", output, "baseline")


def test_metrics_include_failed_calls_critic_and_no_inherited_steps(tmp_path):
    initial = tmp_path / "initial"
    extra = tmp_path / "extra"
    initial.mkdir(); extra.mkdir()
    rows = [{"type": "run.started"}, {"type": "tool.started"},
            {"type": "raw.mini-swe-agent", "response": {"usage": {
                "prompt_tokens": 100, "completion_tokens": 5,
                "prompt_tokens_details": {"cached_tokens": 0}, "cost": 0.1}}}]
    content = "\n".join(json.dumps(r) for r in rows)
    (initial / "parent.jsonl").write_text(content)
    (extra / "parent.jsonl").write_text(content)
    (extra / "child.jsonl").write_text(content + '\n{"type":"model.request.retry"}')
    run.write(extra / "meta-request.json", {"request": {}, "response": {"usage": {
        "prompt_tokens": 10, "completion_tokens": 2, "cost": 0.02}}})
    first = phase_metrics([initial])
    additional = phase_metrics([extra], set(first["journal_hashes"]))
    assert additional["actual_new_steps"] == 1
    assert additional["usage"]["known_input_tokens"] == 110
    assert additional["usage"]["cost_usd"] is None
    assert additional["model_overhead_calls"] == 2


def test_failed_selector_response_usage_survives_retry(tmp_path, monkeypatch):
    from sprout.methods import policies
    responses = iter([
        {"choices": [], "usage": {"prompt_tokens": 7, "completion_tokens": 3}},
        {"choices": [{"message": {"content": "ok"}}],
         "usage": {"prompt_tokens": 11, "completion_tokens": 5}},
    ])
    monkeypatch.setattr(policies.httpx, "post", lambda *args, **kwargs:
                        SimpleNamespace(status_code=200, json=lambda: next(responses)))
    monkeypatch.setattr(policies.time, "sleep", lambda _: None)
    policies.ModelClient("http://localhost:9000", "test-only").complete(
        {"model": "fixture"}, tmp_path / "entropy/step-1.request.json")
    metrics = phase_metrics([tmp_path])
    assert metrics["model_overhead_calls"] == 2
    assert metrics["usage"]["input_tokens"] == 18
    assert metrics["usage"]["output_tokens"] == 8


def test_commands_have_bounded_shared_parent_and_high(tmp_path):
    a = SimpleNamespace(model="qwen3.8-27b", model_endpoint="http://localhost:9000",
        model_key_env="QWEN_API_KEY", runtime_bin=tmp_path / "runtime", max_turns=300,
        max_output_tokens=64000, dataset="terminalbench21", max_rollouts=8,
        preparation=tmp_path / "prep.json", tasks_dir=tmp_path, digest="sha256:test",
        server_url="http://localhost:8000", api_key_file=None,
        final_memory_snapshot="auto", backend="microvm", store=None, prompt_budget_chars=400_000, command_timeout=60, review_timeout=3600)
    for method in run.METHODS:
        cmd = run.command(a, "task", tmp_path / method, method, tmp_path / "parent")
        assert cmd[cmd.index("--reasoning-effort") + 1] == "high"
        assert cmd[cmd.index("--prompt-budget-chars") + 1] == "400000"
        assert cmd[cmd.index("--command-timeout") + 1] == "60", "every evaluate gets the per-command limit"
        assert cmd[cmd.index("--review-timeout") + 1] == "3600"
        # TerminalBench grades inside the agent's VM: keep memory for failed attempts.
        assert cmd[cmd.index("--final-memory-snapshot") + 1] == "on-failure"
        assert cmd[cmd.index("--backend") + 1] == "microvm" and "--store" not in cmd
        assert cmd[cmd.index("--server-url") + 1] == "http://localhost:8000"
        if method in ("bpo", "shepherd"):
            assert cmd[cmd.index("--max-rollouts") + 1] == "8"
        if method == "sprout":
            assert cmd[cmd.index("--branches") + 1:cmd.index("--branches") + 3] == ["4", "3"]


def test_podman_commands_name_the_store_and_never_a_memory_snapshot(tmp_path, monkeypatch):
    """The same evaluate invocation on podman: the store replaces the AgentENV
    endpoint, and a shared verifier's memory snapshot resolves to `never`, since
    btrfs snapshots hold no processes."""
    a = SimpleNamespace(model="qwen3.8-27b", model_endpoint="http://localhost:9000",
        model_key_env="QWEN_API_KEY", runtime_bin=tmp_path / "runtime", max_turns=300,
        max_output_tokens=64000, dataset="terminalbench21", max_rollouts=8,
        preparation=tmp_path / "prep.json", tasks_dir=tmp_path, digest="sha256:test",
        server_url="http://localhost:8000", api_key_file=tmp_path / "key",
        final_memory_snapshot="auto", backend="podman", store=tmp_path / "store", prompt_budget_chars=400_000, command_timeout=60, review_timeout=3600)
    cmd = run.command(a, "task", tmp_path / "baseline", "baseline")
    assert cmd[cmd.index("--backend") + 1] == "podman"
    assert cmd[cmd.index("--store") + 1] == str(tmp_path / "store")
    assert "--server-url" not in cmd and "--api-key-file" not in cmd
    assert cmd[cmd.index("--final-memory-snapshot") + 1] == "never"
    argv = ["--dataset", "deepswe", "--method", "baseline", "--tasks-dir", str(tmp_path), "--task", "t",
            "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "runtime"),
            "--model-endpoint", "http://localhost:9000", "--backend", "podman"]
    import pytest
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(tmp_path / "default-store"))
    assert run.parse_args(argv).store == (tmp_path / "default-store").resolve(), "no --store: the default one"
    with pytest.raises(SystemExit):
        run.parse_args(argv + ["--store", str(tmp_path / "store"), "--final-memory-snapshot", "on-failure"])
    args = run.parse_args(argv + ["--store", str(tmp_path / "store")])
    assert args.backend == "podman" and args.store == (tmp_path / "store").resolve()
    assert run.parse_args(argv[:-2]).backend == "podman", "podman is the default backend"


def test_memory_snapshots_are_never_taken_for_patch_graded_datasets():
    for dataset in ("deepswe", "swebenchpro-v2-hard"):
        assert run.memory_snapshot_policy(SimpleNamespace(dataset=dataset, final_memory_snapshot="auto")) == "never"
    assert run.memory_snapshot_policy(SimpleNamespace(dataset="deepswe", final_memory_snapshot="always")) == "always"
    assert run.memory_snapshot_policy(SimpleNamespace(
        dataset="terminalbench21", final_memory_snapshot="auto", backend="podman")) == "never"


def test_five_row_metrics_resolve_recovery_and_actual_tokens(tmp_path):
    from examples.qwen.report import build_report
    run.write(tmp_path / "manifest-all.json", {"method": "all", "tasks": ["a", "b"],
        "max_total_method_rollouts": 8})
    for name in ("a", "b"):
        root = tmp_path / "tasks" / name
        for i in range(1, 5):
            output = root / "baseline" / f"attempt-{i}"
            run.write(output / "summary.json", {"attempts": [{"resolved": name == "a" or i == 3}]})
            events = [{"type": "run.started", "identity": f"{name}-{i}"}, {"type": "tool.started"},
                      {"type": "raw.mini-swe-agent", "response": {"usage": {
                          "prompt_tokens": 100, "completion_tokens": 20,
                          "prompt_tokens_details": {"cached_tokens": 50}}}}]
            (output / "journal.jsonl").write_text("\n".join(json.dumps(e) for e in events))
        methods = {}
        for method in ("bpo", "shepherd", "sprout"):
            methods[method] = {"status": "skipped_initial_success" if name == "a" else "complete"}
            if name == "b":
                # Row 0 is the shared parent, as methods.evaluate writes it for every branch method.
                rows = [{"resolved": False}, {"resolved": method != "bpo"}]
                run.write(root / method / "summary.json", {"attempts": rows})
        run.write(root / "all.task.json", {"methods": methods})
    rows = build_report(tmp_path, "all")["comparison"]
    assert len(rows) == 5
    assert rows[0]["Resolve Rate"] == 0.5 and rows[0]["Recovery"] == 0
    assert rows[0]["Steps"] == 2 and rows[0]["Tokens (M)"] == 240 / 1e6
    assert rows[1]["Resolve Rate"] == 1 and rows[1]["Recovery"] == 1
    assert rows[1]["Steps"] == 8 and rows[1]["Tokens (M)"] == 960 / 1e6
    assert rows[2]["Resolve Rate"] == 0.5 and rows[2]["Recovery"] == 0
    assert rows[3]["Resolve Rate"] == rows[4]["Resolve Rate"] == 1


def test_every_branch_method_counts_the_shared_parent_once(tmp_path):
    """methods.evaluate writes the shared parent as row 0 of every branch method's
    summary, BPO's and Shepherd's as much as SPROUT's. It is the initial rollout:
    counted again, it pads the additional phase and moves the first success."""
    from examples.qwen.report import build_report
    run.write(tmp_path / "manifest-all.json", {"method": "all", "tasks": ["t"], "max_total_method_rollouts": 8})
    root = tmp_path / "tasks" / "t"
    parent = {"name": "parent", "resolved": False}
    for i in range(1, 5):
        run.write(root / "baseline" / f"attempt-{i}" / "summary.json", {"attempts": [parent]})
    for method in ("bpo", "shepherd", "sprout"):
        branches = [{"name": f"b{n:02d}", "resolved": n == 4} for n in range(1, 5)]
        run.write(root / method / "summary.json", {"attempts": [parent] + branches})
    run.write(root / "all.task.json", {"methods": {m: {"status": "complete"} for m in ("bpo", "shepherd", "sprout")}})
    report = build_report(tmp_path, "all")
    for method in ("bpo", "shepherd", "sprout"):
        row = report["methods"][method]["tasks"][0]
        assert row["phases"]["additional"]["graded_trajectories"] == 4, method
        assert row["graded_trajectories"] == 5 and row["first_success"]["ordinal"] == 5, method
        assert row["success_at"] == {"1": False, "5": True, "8": True}, method
        assert report["methods"][method]["positive_rate"] == 1 / 5, method


def test_a_shared_baseline_must_match_and_an_older_manifest_implies_the_constant_timeout():
    """A branch method continues the baseline's exact conditions. Manifests written before
    --command-timeout existed carry no value: every such run used the constant 60 seconds,
    so they match a 60 and refuse anything else."""
    manifest = {"dataset": "deepswe", "dataset_digest": "sha256:x", "tasks": ["a", "b"], "model": "m",
                "reasoning_effort": "high", "max_turns": 300, "max_output_tokens": 64000,
                "command_timeout": 60, "backend": "podman", "store": "/store", "baseline_rollouts": 4}
    prior = dict(manifest)
    assert run.shared_baseline_mismatch(prior, manifest) is None
    del prior["command_timeout"]
    assert run.shared_baseline_mismatch(prior, manifest) is None, "an older manifest means 60"
    assert run.shared_baseline_mismatch(prior, dict(manifest, command_timeout=300)) == "command_timeout"
    assert run.shared_baseline_mismatch(dict(manifest, command_timeout=120), manifest) == "command_timeout"
    assert run.shared_baseline_mismatch(dict(manifest, tasks=["a"]), manifest) == "tasks"
    assert run.shared_baseline_mismatch(dict(manifest, store="/other"), manifest) == "store"
    assert run.shared_baseline_mismatch(prior, dict(manifest, baseline_rollouts=1)) == "baseline_rollouts"
    del prior["baseline_rollouts"]
    assert run.shared_baseline_mismatch(prior, manifest) is None, "an older manifest means four rollouts"


def test_a_sprout_round_schedule_reaches_evaluate_and_sets_the_rollout_budget(tmp_path):
    """--sprout-rounds R --sprout-branches W runs R rounds of W branches: evaluate
    gets --rounds R and W per round, and the budget is 1 + R*W with the parent.
    Without it the protocol's 4-then-3 schedule is unchanged."""
    import pytest
    argv = ["--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path), "--task", "t",
            "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "rt"),
            "--model-endpoint", "http://localhost:9000", "--store", str(tmp_path / "store")]
    a = run.parse_args(argv + ["--sprout-rounds", "8"])
    assert a.max_rollouts == 9 and run.branch_widths(a) == [1] * 8
    a.tasks_dir, a.digest, a.preparation = tmp_path, "sha256:test", tmp_path / "prep.json"
    cmd = run.command(a, "t", tmp_path / "sprout", "sprout", tmp_path / "parent")
    assert cmd[cmd.index("--rounds") + 1] == "8" and cmd[cmd.index("--max-rollouts") + 1] == "9"
    assert cmd[cmd.index("--branches") + 1:cmd.index("--checkpoint-mode")] == ["1"] * 8
    assert run.branch_widths(run.parse_args(argv + ["--sprout-rounds", "2", "--sprout-branches", "3"])) == [3, 3]
    default = run.parse_args(argv)
    assert default.max_rollouts == 8 and run.branch_widths(default) == [4, 3]
    assert default.sprout_rounds is None, "the protocol's own schedule needs no option"
    for bad in (["--sprout-rounds", "0"], ["--sprout-rounds", "17"], ["--sprout-rounds", "8", "--sprout-branches", "3"],
                ["--max-rollouts", "9"]):
        with pytest.raises(SystemExit):
            run.parse_args(argv + bad)
    with pytest.raises(SystemExit):
        run.parse_args([x if x != "sprout" else "bpo" for x in argv] + ["--sprout-rounds", "4"])


def test_evaluate_takes_a_scheduled_sprout_budget_but_keeps_bpo_and_shepherd_at_eight(tmp_path):
    import pytest
    from sprout.methods import evaluate
    runtime = tmp_path / "runtime"
    runtime.touch()
    argv = ["--task-dir", str(tmp_path), "--dataset-digest", "d", "--preparation-manifest", str(tmp_path / "p.json"),
            "--model", "m", "--model-endpoint", "http://127.0.0.1:1", "--model-key-env", "KEY",
            "--runtime-bin", str(runtime), "--output", str(tmp_path / "out"), "--volatile-ok", "--store", str(tmp_path)]
    assert evaluate.parse_args(argv + ["--method", "sprout", "--rounds", "8", "--branches", "1",
                                       "--max-rollouts", "9"]).max_rollouts == 9
    for method in ("bpo", "shepherd"):
        with pytest.raises(SystemExit):
            evaluate.parse_args(argv + ["--method", method, "--parent-output", str(tmp_path), "--max-rollouts", "9"])


def test_leak_judge_reaches_evaluate_for_sprout_only(tmp_path, monkeypatch):
    """--leak-judge on the runner becomes evaluate's --leak-judge with its resamples
    and threshold, for sprout's stages only, and evaluate accepts what it is given."""
    import pytest
    from sprout.methods import evaluate
    argv = ["--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path), "--task", "t",
            "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "rt"),
            "--model-endpoint", "http://localhost:9000", "--store", str(tmp_path / "store")]
    a = run.parse_args(argv + ["--sprout-rounds", "8", "--leak-judge", "--leak-judge-resamples", "1"])
    a.tasks_dir, a.digest, a.preparation = tmp_path, "sha256:test", tmp_path / "prep.json"
    cmd = run.command(a, "t", tmp_path / "sprout", "sprout", tmp_path / "parent")
    assert "--leak-judge" in cmd and cmd[cmd.index("--leak-judge-resamples") + 1] == "1"
    assert float(cmd[cmd.index("--leak-judge-threshold") + 1]) == run.LEAK_THRESHOLD
    assert "--leak-judge" not in run.command(a, "t", tmp_path / "base", "baseline"), "a baseline has no reviewer"
    (tmp_path / "rt").touch()
    monkeypatch.setenv("TYPESAFE_API_KEY", "k")
    parsed = evaluate.parse_args(cmd[cmd.index("--task-dir"):] + ["--volatile-ok"])
    assert parsed.leak_judge and parsed.leak_judge_resamples == 1
    plain = run.parse_args(argv)
    plain.tasks_dir, plain.digest, plain.preparation = tmp_path, "sha256:test", tmp_path / "prep.json"
    assert "--leak-judge" not in run.command(plain, "t", tmp_path / "s", "sprout", tmp_path / "p")
    for bad in (["--leak-judge-resamples", "-1"], ["--leak-judge-threshold", "0"]):
        with pytest.raises(SystemExit):
            run.parse_args(argv + ["--leak-judge"] + bad)
    with pytest.raises(SystemExit):
        run.parse_args([x if x != "sprout" else "bpo" for x in argv] + ["--leak-judge"])


def test_a_judged_run_alone_records_its_judge_and_a_root_resumes_under_its_own_commit(tmp_path, monkeypatch):
    """Only a judged run's manifest names the judge, under ``review``; the rest of the
    manifest is the same. Which roots resume is decided elsewhere: the manifest
    records ``source_commit`` and ``main`` matches it whole, so a root resumes under
    the commit that wrote it and is refused under any other. A key that is always
    written (``branch_caps``, ``review``) or only when set (``leak_judge``) changes
    what a manifest says, never which roots resume."""
    import examples.qwen.report as report
    (tmp_path / "t").mkdir()
    (tmp_path / "t" / "task.toml").write_text("[task]\nname = 'org/t'\n")
    argv = ["--dataset", "deepswe", "--method", "sprout", "--tasks-dir", str(tmp_path), "--task", "t",
            "--output", str(tmp_path / "out"), "--runtime-bin", str(tmp_path / "rt"),
            "--model-endpoint", "http://localhost:9000", "--store", str(tmp_path / "store")]

    def manifest(extra):
        a = run.parse_args(argv + extra)
        a.digest, a.source_commit = "sha256:test", "abc"
        return run.manifest_for(a)

    plain = manifest([])
    assert "leak_judge" not in plain["review"] and plain["branch_caps"] == [4, 3]
    judged = manifest(["--leak-judge", "--leak-judge-resamples", "1"])
    assert judged["review"]["leak_judge"] == {"resamples": 1, "threshold": run.LEAK_THRESHOLD}
    assert {k: v for k, v in judged.items() if k != "review"} == {k: v for k, v in plain.items() if k != "review"}

    commit = {"value": "a" * 40}
    monkeypatch.setattr(run.subprocess, "check_output",
                        lambda cmd, **k: commit["value"] + "\n" if cmd[1] == "rev-parse" else "")
    monkeypatch.setenv("QWEN_API_KEY", "k")
    monkeypatch.setattr(run, "run_task", lambda a, task: {"task": task, "complete": True})
    monkeypatch.setattr(report, "build_report", lambda root, method: {})
    assert run.main(argv) == 0
    written = run.read(tmp_path / "out" / "manifest-sprout.json")
    assert written["source_commit"] == "a" * 40 and written["branch_caps"] == [4, 3] and "review" in written
    assert run.main(argv) == 0, "the same configuration under the same commit resumes"
    commit["value"] = "b" * 40
    with pytest.raises(ValueError, match="Experiment configuration changed"):
        run.main(argv)
    assert run.read(tmp_path / "out" / "manifest-sprout.json") == written, "a refused resume rewrites nothing"


def test_compare_refuses_roots_that_differ_on_any_shared_baseline_setting(tmp_path):
    """compare.py joins four run directories. The settings they must agree on are the
    ones run.py enforces when a branch method is launched on a shared baseline -- the
    same tuple, so a key added there cannot be forgotten here (three were). Both
    manifests come from disk, so a key written before it existed takes its historical
    value on either side."""
    from examples.qwen import compare

    assert compare.SHARED_BASELINE_KEYS is run.SHARED_BASELINE_KEYS
    shared = {"dataset": "deepswe", "dataset_digest": "sha256:x", "tasks": ["a"], "model": "m",
              "reasoning_effort": "high", "max_turns": 300, "max_output_tokens": 64000,
              "command_timeout": 60, "backend": "podman", "store": "/store", "baseline_rollouts": 4}
    roots = {method: tmp_path / method for method in ("baseline", "bpo", "shepherd", "sprout")}

    def write_roots(baseline, **overrides):
        run.write(roots["baseline"] / "manifest-baseline.json", {**baseline, "method": "baseline"})
        for i in range(1, 5):
            run.write(roots["baseline"] / "tasks" / "a" / "baseline" / f"attempt-{i}" / "summary.json",
                      {"attempts": [{"resolved": i == 3}]})
        for method in ("bpo", "shepherd", "sprout"):
            run.write(roots[method] / f"manifest-{method}.json",
                      {**baseline, "method": method, "max_total_method_rollouts": 8,
                       "baseline_root": str(roots["baseline"]), **overrides.get(method, {})})
            run.write(roots[method] / "tasks" / "a" / method / "summary.json", {"attempts": [{"resolved": True}]})
            run.write(roots[method] / "tasks" / "a" / f"{method}.task.json",
                      {"methods": {method: {"status": "complete"}}})

    for key, value in (("command_timeout", 600), ("backend", "microvm"), ("store", "/other"),
                       ("baseline_rollouts", 1)):
        write_roots(shared, bpo={key: value})
        with pytest.raises(ValueError, match=key):
            compare.compare(roots)
    # A baseline recorded before --command-timeout, --backend and --store existed, beside a
    # method manifest naming the constant 60 s (or lacking the key too), is comparable.
    legacy = {k: v for k, v in shared.items() if k not in ("command_timeout", "backend", "store")}
    write_roots(legacy, sprout={"command_timeout": 60})
    assert compare.compare(roots)["complete"]
    write_roots(legacy)
    assert compare.compare(roots)["complete"]
    write_roots(legacy, sprout={"command_timeout": 600})
    with pytest.raises(ValueError, match="command_timeout"):
        compare.compare(roots)

    write_roots(shared)
    out = tmp_path / "comparison"
    compare.main([*(arg for method in roots for arg in (f"--{method}", str(roots[method]))), "--output", str(out)])
    rows = json.loads((out / "comparison.json").read_text())["comparison"]
    assert [r["method"] for r in rows] == ["pass@1", "pass@4 (baseline)", "entropy-based (BPO)", "Shepherd", "SPROUT"]
    assert len((out / "comparison.csv").read_text().splitlines()) == 1 + 5

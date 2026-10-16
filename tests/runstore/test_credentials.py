import os
import time

from sprout.worker.run.control import RunControl
from sprout.worker.agents.slot import TaskSpec
from sprout.worker.run.rollout import RolloutControls
from sprout.worker.run.runner import Orchestrator, RunOutcome
from sprout.worker.jobs.rollout import execute


def test_worker_credentials_are_hidden_and_gateway_wins_over_agent_env(tmp_path, monkeypatch):
    environment = {**os.environ, "RS_BACKEND": "backend-fixture-secret",
                   "RS_PROVIDER": "provider-fixture-secret"}
    monkeypatch.setattr(os, "environ", environment)
    captured = {}

    def run(orchestrator, spec):
        task = TaskSpec(prompt=spec.prompt, cwd=spec.cwd, extra={})
        # A rollout contract is what makes the orchestrator start the gateway.
        task._rollout_controls = RolloutControls(spec.extra["rollout_contract"], None, RunControl())
        gateway = orchestrator._wire_gateway(spec, None, task, spec.run_id)
        try:
            # mini runs in-process; the environment it is given is task.env.
            captured.update(task.env)
            assert spec.backend["microvm"]["api_key"] == "backend-fixture-secret"
            assert task.env["OPENAI_BASE_URL"] == gateway.base_url + "/v1"
        finally:
            gateway.stop()
        return RunOutcome(spec.run_id, spec.journal_path, "completed")

    monkeypatch.setattr(Orchestrator, "run", run)
    execute({"kind": "rollout", "attempt_id": "fixture", "effective_spec": {
        "prompt": "fixture", "slot": "mini-swe-agent",
        "backend": {"microvm": {"api_key": {"$env": "RS_BACKEND"}}},
        "extra": {"rollout_contract": {"model_endpoint": "http://127.0.0.1:1", "model": "fixture",
                                       "deadline_at": time.time() + 60, "max_turns": 1,
                                       "api_key_env": "UPSTREAM_API_KEY"}}},
        "profile_config": {"worker_env": {"UPSTREAM_API_KEY": {"$env": "RS_PROVIDER"}},
                           "env": {"OPENAI_BASE_URL": "http://must-not-bypass-gateway"}}}, tmp_path)
    assert captured["RS_BACKEND"] == ""
    assert captured["RS_PROVIDER"] == ""
    assert captured["UPSTREAM_API_KEY"] == ""
    assert captured["OPENAI_API_KEY"].startswith("sprout-slot-")
    assert "backend-fixture-secret" not in captured.values()
    assert "provider-fixture-secret" not in captured.values()

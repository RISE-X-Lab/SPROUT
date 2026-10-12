"""The gate's bookkeeping: where a check's trial lands and what it records."""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from sprout.methods import gate

FIXTURE = Path(__file__).resolve().parents[1] / "harbor_plugin" / "agentenv_fixtures" / "shell-task"


def test_tasks_whose_names_share_36_characters_get_trial_directories_of_their_own(tmp_path, monkeypatch):
    """A trial's name keeps the first 36 characters of the task's. SWE-bench Pro's
    eight openlibrary instances share theirs and once wrote one trial directory,
    each check reading whatever result landed there last."""
    tasks = []
    for suffix in ("one", "two"):
        task_dir = tmp_path / "tasks" / f"instance_internetarchive__openlibrary-{suffix}"
        shutil.copytree(FIXTURE, task_dir)
        tasks.append(task_dir)

    class Trial:
        """Stands in for Harbor's: writes the result a real trial would, with
        a reward that says which task it ran."""

        def __init__(self, config):
            self.config = config

        @classmethod
        async def create(cls, config):
            return cls(config)

        async def run(self):
            directory = Path(self.config.trials_dir) / self.config.trial_name
            directory.mkdir(parents=True)
            reward = 1.0 if str(self.config.task.path).endswith("one") else 0.0
            (directory / "result.json").write_text(json.dumps({"verifier_result": {"rewards": {"reward": reward}}}))

    monkeypatch.setattr(gate, "Trial", Trial)
    runtime = tmp_path / "sprout-runtime"
    runtime.touch()
    args = SimpleNamespace(output=tmp_path / "gate", runtime_bin=runtime, backend="podman", store=tmp_path / "store",
                           server_url=None, api_key_file=None, sandbox_ttl=36000, dataset_digest="gate",
                           checkpoint_mode="none")
    one, two = (gate.check(args, task_dir, "oracle") for task_dir in tasks)
    assert one["trial"] != two["trial"], "one trial directory for two tasks"
    assert (one["reward"], one["ok"]) == (1.0, True) and (two["reward"], two["ok"]) == (0.0, False)

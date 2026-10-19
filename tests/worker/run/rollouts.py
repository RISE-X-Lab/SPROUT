"""Creation functions for runs of mini on a scripted model endpoint, and readers of what a run left.

A test that runs mini skips without the mini extra. The check does not import mini, because its first
import loads the developer's global mini config into the process.
"""

from __future__ import annotations

import importlib.util
import time
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.run_spec import RunSpec
from sprout.worker.run.journal import read_journal
from sprout.worker.run.runner import RunOutcome

needs_mini = pytest.mark.skipif(
    importlib.util.find_spec("minisweagent") is None,
    reason='install the mini extra: pip install -e ".[mini]"',
)

#: The bash command with which mini submits its task.
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def rollout_contract(url: str, **changes: Any) -> dict[str, Any]:
    """A rollout contract for the endpoint ``url``: 5 turns, 30 s, and fixed sampling parameters."""
    return {
        "model_endpoint": url,
        "model": "contract-model",
        "deadline_at": time.time() + 30,
        "max_turns": 5,
        "sampling_params": {"temperature": 0.4, "max_new_tokens": 7, "top_k": 11, "stop": ["stop-here"]},
        **changes,
    }


def rollout_spec(
    directory: Path, url: str, *, contract: dict[str, Any] | None = None, **changes: Any
) -> RunSpec:
    """The spec of a mini rollout on the endpoint ``url``, with its journal and history in ``directory``."""
    fields: dict[str, Any] = {
        "prompt": "Write the answer.",
        "model": "fixture",
        "timeout_s": 30,
        "sandbox_image": "image",
        "journal_path": directory / "trajectory.jsonl",
        "extra": {
            "native_home": str(directory / "native-home"),
            "rollout_contract": contract or rollout_contract(url),
        },
        **changes,
    }
    return RunSpec(**fields)


def events(outcome: RunOutcome, *types: str) -> list[dict[str, Any]]:
    """The journal records of ``outcome`` whose type is one of ``types``, in journal order."""
    return [record for record in read_journal(outcome.journal_path) if record["type"] in types]


def transcript(directory: Path, outcome: RunOutcome) -> Path:
    """The native history that mini wrote for the run of ``outcome`` in ``directory``."""
    return directory / "native-home" / ("%s.jsonl" % outcome.native_session_id)

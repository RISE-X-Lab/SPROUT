"""Creation functions for a mini rollout through the worker: its ``RunSpec``, and what it leaves behind."""

from __future__ import annotations

import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import read_entries
from sprout.contract.run_spec import RunSpec
from sprout.worker.run.journal import read_journal

#: The command whose output tells mini that the agent submits.
SUBMIT = "echo COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"
#: The mark of a module whose tests run mini-swe-agent.
NEEDS_MINI = pytest.mark.skipif(
    find_spec("minisweagent") is None, reason='needs mini-swe-agent 2.4.6: pip install -e ".[mini]"'
)


def rollout_contract(url: str, **changes: Any) -> dict[str, Any]:
    """A message-export rollout contract for the chat endpoint at ``url``, with ``changes``."""
    return {
        "message_export": True,
        "model_endpoint": url,
        "model": "fixture",
        "deadline_at": time.time() + 60,
        "max_turns": 5,
        "sampling_params": {"temperature": 0.8, "top_k": 20, "max_new_tokens": 300},
        **changes,
    }


def mini_spec(directory: Path, url: str, *, contract: dict[str, Any] | None = None, **fields: Any) -> RunSpec:
    """The ``RunSpec`` of a mini rollout that journals under ``directory``, with ``fields`` set."""
    return RunSpec(
        prompt="Write the answer.",
        model="fixture",
        timeout_s=60,
        sandbox_image="fixture-image",
        journal_path=directory / "trajectory.jsonl",
        extra={
            "native_home": str(directory / "native-home"),
            "rollout_contract": contract or rollout_contract(url),
        },
        **fields,
    )


def native_entries(native_home: Path, session_id: str | None) -> list[dict[str, Any]]:
    """The entries of the native history of mini session ``session_id`` under ``native_home``."""
    return read_entries((native_home / ("%s.jsonl" % session_id)).read_bytes())


def events(journal: Path, kind: str) -> list[dict[str, Any]]:
    """The events of type ``kind`` in ``journal``, in order."""
    return [event for event in read_journal(journal) if event["type"] == kind]

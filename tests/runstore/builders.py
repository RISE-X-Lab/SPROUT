"""Creation functions for the Run Store tests: a valid request of each job kind, and an attempt payload.

Each function fills the fields that a test does not care about with fixed values. A test passes the
fields that matter to it as keyword arguments, so that they stay visible in the test. A store accepts
a review only when the store holds its parents, so ``stored_job`` adds one.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

from sprout.runstore.specs import JobSpec

if TYPE_CHECKING:
    from sprout.runstore.store import Store

#: A Harbor task directory. No test reads it.
TASK_DIR = "/path/to/tasks/fix-git"


def rollout_spec(**changes: Any) -> dict[str, Any]:
    """The spec of a valid rollout, with ``changes`` over it."""
    return {"prompt": "Write the answer.", **changes}


def harbor_grade_spec(**changes: Any) -> dict[str, Any]:
    """The spec of a valid ``harbor`` grade from a snapshot, with ``changes`` over it."""
    return {"benchmark": "harbor", "task_dir": TASK_DIR, "snapshot_id": "snapshot-1", **changes}


def swerebench_grade_spec(**changes: Any) -> dict[str, Any]:
    """The spec of a valid ``swe-rebench-v2`` grade, with ``changes`` over it."""
    return {
        "benchmark": "swe-rebench-v2",
        "instance_id": "owner__repo-1",
        "snapshot_id": "snapshot-1",
        "dataset_path": "/path/to/tasks.jsonl",
        "dataset_sha256": "0" * 64,
        "grader_revision": "sha256:" + "1" * 64,
        "parser_path": "/path/to/log_parsers.py",
        "image": "registry.example/task:1",
        **changes,
    }


def review_parent(**changes: Any) -> dict[str, Any]:
    """A valid parent of a review: a finished attempt with its grade, with ``changes`` over it."""
    return {
        "name": "root",
        "job_id": "parent-job",
        "attempt_id": "parent-attempt",
        "grade": {"resolved": False},
        **changes,
    }


def review_spec(**changes: Any) -> dict[str, Any]:
    """The spec of a valid review of one parent, with ``changes`` over it."""
    return {
        "problem": "Fix the failing test.",
        "model": "policy",
        "model_endpoint": "http://127.0.0.1:1/v1",
        "parents": [review_parent()],
        **changes,
    }


#: The creation function of a valid spec for each job kind.
SPECS = {"rollout": rollout_spec, "grade": harbor_grade_spec, "review": review_spec}


#: The affinity key of a grade from a snapshot: the key of the rollout that took the snapshot.
GRADE_AFFINITY = "group-1"


def job(kind: str = "rollout", *, max_infra_retries: int = 2, **changes: Any) -> JobSpec:
    """A valid request for a job of ``kind``, with ``changes`` over its spec, and a grade's affinity key."""
    affinity = GRADE_AFFINITY if kind == "grade" else None
    return JobSpec(kind, SPECS[kind](**changes), max_infra_retries=max_infra_retries, affinity=affinity)


def stored_job(store: Store, kind: str = "rollout", *, max_infra_retries: int = 2) -> JobSpec:
    """A valid request for a job of ``kind`` that ``store`` accepts: a review reads a parent in ``store``.

    The parent is a cancelled rollout, so that no claim takes it.
    """
    if kind != "review":
        return job(kind, max_infra_retries=max_infra_retries)
    parent = store.submit(job("rollout"), "review-parent")["id"]
    store.request_stop(parent)
    return job("review", max_infra_retries=max_infra_retries, parents=[review_parent(job_id=parent)])


def payload_for(claimed: dict[str, Any], **changes: Any) -> dict[str, Any]:
    """The version-1 attempt payload of the claimed job row ``claimed``, with ``changes`` over it."""
    return {
        "version": 1,
        "job_id": claimed["id"],
        "attempt_id": claimed["active_attempt"],
        "kind": claimed["kind"],
        "effective_spec": claimed["request"]["spec"],
        "profile_config": {},
        "recovery": None,
        **changes,
    }


def bash_call(call_id: str, command: str) -> dict[str, Any]:
    """An OpenAI tool call of mini's ``bash`` tool."""
    arguments = json.dumps({"command": command})
    return {"id": call_id, "type": "function", "function": {"name": "bash", "arguments": arguments}}


def native_history(directory: Path, *, call_id: str = "call-1") -> dict[str, Any]:
    """The prefix reference of a mini history in ``directory``: one closed turn that calls ``bash`` once.

    The reference names the whole file, the way a recovery point names mini's history up to its turn.
    """
    entries = [
        {"type": "mini.session", "format": "sprout-mini-v1", "mini_version": "2.4.6"},
        {"type": "mini.message", "message": {"role": "user", "content": "Write the answer."}},
        {
            "type": "mini.message",
            "message": {"role": "assistant", "content": "Look.", "tool_calls": [bash_call(call_id, "ls")]},
        },
        {"type": "mini.message", "message": {"role": "tool", "tool_call_id": call_id, "content": "answer"}},
        {"type": "mini.turn", "step": 1},
    ]
    data = "".join(json.dumps(entry) + "\n" for entry in entries).encode()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "native.jsonl"
    path.write_bytes(data)
    digest = hashlib.sha256(data).hexdigest()
    return {"slot": "mini-swe-agent", "path": str(path), "byte_length": len(data), "sha256": digest}

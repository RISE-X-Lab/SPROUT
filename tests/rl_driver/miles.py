"""Creation functions for the Miles adapter tests: requests, adapter configs and a worker's results.

Each function gives a valid value whose fields do not matter to the test, and takes the fields that do.
The ``run_*`` functions play the worker on a ``FakeRunStore`` between the driver's ticks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.messages import MessageAdapter
from tests.rl_driver.fake_runstore import FakeRunStore

#: The messages of a finished rollout in their cleaned training form.
MESSAGES = [
    {"role": "user", "content": "Fix the parser"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call-1",
                "type": "function",
                "function": {"name": "bash", "arguments": '{"command":"ls"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call-1", "content": "parser.py"},
    {"role": "assistant", "content": "Fixed."},
]

#: The grade job template of task-1 in ``adapter_config``.
GRADE_TEMPLATE = {
    "profile": "grade",
    "spec": {
        "benchmark": "swe-rebench-v2",
        "instance_id": "task-1",
        "dataset_path": "/path/to/tasks.jsonl",
        "dataset_sha256": "dataset-digest",
        "grader_revision": "sha256:parser-digest",
        "parser_path": "/path/to/log_parsers.py",
    },
}

#: The ``review`` section of a search adapter's config.
REVIEW_CONFIG = {"profile": "review", "temperature": 0.7, "max_output_tokens": 4096}


def miles_request(**fields: Any) -> dict[str, Any]:
    """A valid Miles request for one sample of task-1, with ``fields`` in place of its own."""
    return {
        "rollout_job_id": "rollout-1",
        "rollout_id": 1,
        "prompt_group_id": "prompt-1",
        "task_id": "task-1",
        "image": "docker.io/swerebenchv2/task-1:base",
        "prompt": "Fix the parser",
        "model_endpoint": "http://model.test:30000",
        "model": "policy",
        "sample_slots": slots(1),
        "max_samples": 1,
        "minimum_returned_samples": 1,
        "max_turns": 10,
        "sampling_params": {"temperature": 0.6},
        "budgets": {"max_wall_time_seconds": 900.0},
        **fields,
    }


def slots(count: int) -> list[dict[str, Any]]:
    """``count`` sample slots, ``slot-0`` at index 0 and so on."""
    return [{"sample_slot_id": f"slot-{index}", "sample_index": index} for index in range(count)]


def search(**fields: Any) -> dict[str, Any]:
    """A valid ``search`` object: one root, one point, one student and one repair, with ``fields``."""
    return {
        "roots": 1,
        "points": 1,
        "candidates": 1,
        "student_continuations": 1,
        "candidate_continuations": 1,
        "tf_student_continuations": 2,
        "repair_rounds": 1,
        "review_seconds": 600.0,
        "wall_time_seconds": 900.0,
        **fields,
    }


def search_request(*, slot_count: int, **search_fields: Any) -> dict[str, Any]:
    """A Miles search request with ``slot_count`` slots, all of which it may fill, and ``search_fields``."""
    return miles_request(
        sample_slots=slots(slot_count), max_samples=slot_count, search=search(**search_fields)
    )


def adapter_config(**fields: Any) -> dict[str, Any]:
    """A valid ``miles`` section that grades task-1, with ``fields`` in place of its own."""
    return {
        "profile": "mini",
        "run_defaults": {"slot": "mini-swe-agent"},
        "image_resources": {"cpu": 2, "memory_mb": 12288},
        "tasks": {"task-1": {"grade": GRADE_TEMPLATE}},
        **fields,
    }


def miles_adapter(fake: FakeRunStore, ledger: Path, config: dict[str, Any] | None = None) -> MessageAdapter:
    """An adapter over a new driver on the ledger file ``ledger`` and the fake Run Store."""
    return MessageAdapter(Driver(fake.client(), Ledger(ledger)), config or adapter_config())


def exported(**fields: Any) -> dict[str, Any]:
    """An actor's result with its training export and final snapshot, with ``fields`` in place of its own."""
    return {
        "status": "completed",
        "final_snapshot_id": "snap-final",
        "training_messages": MESSAGES,
        "training_provenance": {"prefix_messages": 0, "inserted_messages": []},
        **fields,
    }


def verdict(resolved: bool, **facts: Any) -> dict[str, Any]:
    """A grade's result that says whether the task resolved, with the hidden-test ``facts``."""
    return {"status": "completed", "resolved": resolved, "progress": 1.0 if resolved else 0.0, **facts}


def succeeded(result: dict[str, Any]) -> dict[str, Any]:
    """The outcome of a job that succeeded with ``result`` (``FakeRunStore.finish`` arguments)."""
    return {"result": result}


def failed(error: str) -> dict[str, Any]:
    """The outcome of a job that failed with ``error`` (``FakeRunStore.finish`` arguments)."""
    return {"state": "failed", "error": error, "result": {"status": "error", "error": error}}


def data_lost() -> dict[str, Any]:
    """The outcome of a job that the Run Store failed because the host that held its data was lost."""
    error = "Host host-a is lost: none of its workers reported a heartbeat in 600 s"
    return {
        "state": "failed",
        "error": error,
        "result": {
            "status": "error",
            "failure_kind": "infrastructure",
            "reason": "data_lost",
            "error": error,
        },
    }


def finish_open(fake: FakeRunStore, kind: str, outcomes: list[dict[str, Any]]) -> None:
    """Finish the queued or running jobs of ``kind`` with ``outcomes``, in submission order."""
    open_jobs = [job_id for job_id in fake.ids(kind) if fake.jobs[job_id]["state"] in {"queued", "running"}]
    for job_id, outcome in zip(open_jobs, outcomes, strict=True):
        fake.finish(job_id, **outcome)


def run_samples(
    adapter: MessageAdapter,
    fake: FakeRunStore,
    *,
    actors: list[dict[str, Any]],
    grades: list[dict[str, Any]],
) -> None:
    """Run the open samples of a group: their actors end as ``actors``, then their grades as ``grades``.

    A search then submits its review in the last tick.
    """
    adapter.driver.tick()
    finish_open(fake, "rollout", actors)
    adapter.driver.tick()
    finish_open(fake, "grade", grades)
    adapter.driver.tick()


def run_review(adapter: MessageAdapter, fake: FakeRunStore, outcome: dict[str, Any]) -> None:
    """End the open review as ``outcome``. The driver then submits the branches that it plans."""
    finish_open(fake, "review", [outcome])
    adapter.driver.tick()

"""The Miles wire of a search: what the RL driver returns for the trajectories of a search group.

Miles credits a search group by what the driver writes on each trajectory. ``search_result.example.json``
in examples/rl_driver/ is the result of one scenario: two roots that both fail, a point in each, and at
each point two students and two repairs after each of two repair turns. Miles's GSML tests read a byte
copy of the file, so the driver must return exactly it (docs/RL_DRIVER.md). The driver and its Miles
adapter run the real coordinator on a ``FakeRunStore``, and the test plays each worker.
"""

from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path
from typing import Any

from sprout.contract.continuation import Continuation, prepare
from sprout.contract.messages import clean_messages, mark_hint
from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.messages import MessageAdapter
from sprout.search.spec import SearchSpec
from tests.search.documents import reviewed, verdict
from tests.search.fake_runstore import FakeRunStore
from tests.support.paths import REPO

EXAMPLES = REPO / "examples" / "rl_driver"
#: The search of the shared example, as Miles asks for it.
SEARCH = {
    "roots": 2,
    "points": 1,
    "candidates": 2,
    "student_continuations": 2,
    "candidate_continuations": 2,
    "tf_student_continuations": 2,
    "repair_rounds": 1,
    "review_seconds": 600.0,
    "wall_time_seconds": 900.0,
}
#: The history that each root exports. The training copy drops the hint.
HISTORY = [
    {"role": "user", "content": "fix task"},
    {
        "role": "assistant",
        "content": "",
        "tool_calls": [
            {
                "id": "call",
                "type": "function",
                "function": {"name": "shell", "arguments": '{"command":"pwd"}'},
            }
        ],
    },
    {"role": "tool", "tool_call_id": "call", "content": "/workspace"},
    {"role": "user", "content": mark_hint("check the parser")},
    {"role": "assistant", "content": "fixed"},
]
#: What a branch restores: its root's history up to the point, after the root's first tool result.
PREFIX = clean_messages(HISTORY)[:3]
#: The repair turns that the review writes at the point of each failed root: the text, then the command.
REPAIRS = {
    1: [("parser broken", "sed -n 1,40p src/parser.py"), ("inspect tests", "pytest -q tests")],
    2: [
        ("parser broken", "grep -n parse src/parser.py"),
        ("inspect tests", "pytest -q tests/test_hidden.py"),
    ],
}
#: The hidden test names that the review's leak check found in a turn, by point and candidate.
LEAKS = {(2, 2): ["tests/test_hidden.py"]}
#: What each grade reports about the task's hidden tests beside its verdict.
HIDDEN = {
    "fail_to_pass": ["tests/test_hidden.py::test_secret_behaviour"],
    "test_patch_paths": ["tests/test_hidden.py"],
}
#: The verdicts of the branches, in slot order: at each point the two students, then the two repairs of
#: candidate 1 and the two of candidate 2.
FF_BRANCHES = (True, False, True, False, False, False, False, False, True, True, True, False)


def miles_request(*, roots: int = 2, search: dict[str, Any] | None = None) -> dict[str, Any]:
    """The shipped Miles request with ``search``, a slot for each root and one for each possible branch."""
    body: dict[str, Any] = json.loads((EXAMPLES / "request.example.json").read_text())
    search = {**SEARCH, "roots": roots} if search is None else search
    job = body["rollout_job_id"]
    slots = roots + SearchSpec.from_dict(search).branches()
    body["sample_slots"] = [
        {"sample_slot_id": "%s:slot:%d" % (job, 11 + k), "sample_index": 11 + k} for k in range(slots)
    ]
    body.update(max_samples=slots, minimum_returned_samples=1, search=search)
    return body


def adapter_config(body: dict[str, Any]) -> dict[str, Any]:
    grade = {
        "benchmark": "swe-rebench-v2",
        "instance_id": body["task_id"],
        "dataset_path": "/path/to/tasks.jsonl",
        "dataset_sha256": "fixture",
        "grader_revision": "sha256:fixture",
        "parser_path": "/path/to/log_parsers.py",
    }
    return {
        "profile": "mini",
        "run_defaults": {"slot": "mini-swe-agent"},
        "image_resources": {"cpu": 2, "memory_mb": 12288},
        "tasks": {body["task_id"]: {"grade": {"profile": "grade", "spec": grade}}},
        "review": {"profile": "review", "temperature": 0.7, "max_output_tokens": 4096},
    }


def exported_root() -> dict[str, Any]:
    return {
        "status": "completed",
        "training_messages": clean_messages(HISTORY),
        "training_provenance": {"prefix_messages": 0, "inserted_messages": []},
    }


def repair_turn(point: int, candidate: int) -> dict[str, Any]:
    text, command = REPAIRS[point][candidate - 1]
    call = {
        "id": "repair-%d-%d" % (point, candidate),
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": text, "tool_calls": [call]}


def review_plan(parents: list[dict[str, Any]]) -> dict[str, Any]:
    """A point at step 2 of each parent and, at each point, two repair turns with their leak terms."""
    return reviewed(
        *(
            {
                "job_id": parent["job_id"],
                "attempt_id": parent["attempt_id"],
                "step": 2,
                "why": "before the edit",
                "samples": 2,
                "candidates": [
                    {
                        "assistant_turn": repair_turn(point, candidate),
                        "multiplicity": 1,
                        "leak_terms": LEAKS.get((point, candidate), []),
                    }
                    for candidate in (1, 2)
                ],
            }
            for point, parent in enumerate(parents, 1)
        )
    )


def finish_branch(fake: FakeRunStore, job_id: str, body: dict[str, Any]) -> None:
    """Finish a branch as its worker exports it: the restored prefix, the inserted turn as
    ``continuation.prepare`` gives it to mini with that turn's result, then the branch's own last turn."""
    entries = [{"type": "mini.message", "message": message} for message in PREFIX]
    _, options = prepare(Continuation.from_dict(body["continuation"]), entries)
    turn = options.get("assistant_turn")
    inserted = (
        [turn, {"role": "tool", "tool_call_id": turn["tool_calls"][0]["id"], "content": "tests"}]
        if turn
        else []
    )
    messages = [*deepcopy(PREFIX), *inserted, {"role": "assistant", "content": "fixed"}]
    provenance = {"prefix_messages": len(PREFIX), "inserted_messages": [len(PREFIX)] if turn else []}
    fake.finish_rollout(
        job_id, {"status": "completed", "training_messages": messages, "training_provenance": provenance}
    )


def run_search(
    tmp_path: Path,
    fake: FakeRunStore,
    body: dict[str, Any],
    roots: tuple[dict[str, Any], ...],
    branches: tuple[bool, ...],
) -> dict[str, Any]:
    """The Miles result of a search of ``body``: the roots are graded ``roots``, the review picks
    ``review_plan`` for the parents that it gets, and the branches are graded ``branches``, in slot order."""
    driver = Driver(fake.client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, adapter_config(body))
    adapter.submit(body)
    driver.tick()
    for job_id in fake.job_ids("rollout"):
        fake.finish_rollout(job_id, exported_root())
    driver.tick()
    for job_id, grade in zip(fake.job_ids("grade"), roots, strict=True):
        fake.finish(job_id, grade)
    driver.tick()
    (review,) = fake.submitted("review")
    (review_id,) = fake.job_ids("review")
    fake.finish(review_id, review_plan(review["spec"]["parents"]))
    driver.tick()
    branch_ids = fake.job_ids("rollout")[len(roots) :]
    for job_id, (_, branch) in zip(branch_ids, fake.branches(), strict=True):
        finish_branch(fake, job_id, branch)
    driver.tick()
    samples = adapter.execution(body["rollout_job_id"])["samples"][len(roots) :]
    for sample, resolved in zip(samples, branches, strict=True):
        fake.finish(sample["grade"]["job_id"], verdict(resolved, touched_test_paths=[], **HIDDEN))
    driver.tick()
    return adapter.get(body["rollout_job_id"])


def rewards_and_hidden_tests(trajectories: list[dict[str, Any]]) -> list[tuple[float, str | None, Any]]:
    """The reward, the ``zero_reason`` and the ``touched_test_paths`` of each trajectory."""
    return [
        (
            trajectory["reward"],
            trajectory["metadata"]["zero_reason"],
            trajectory["metadata"]["touched_test_paths"],
        )
        for trajectory in trajectories
    ]


def test_the_driver_returns_the_shared_ff_search_example_for_its_scenario(tmp_path, fake_run_store):
    root = verdict(False, touched_test_paths=[], **HIDDEN)

    result = run_search(tmp_path, fake_run_store, miles_request(), (root, root), FF_BRANCHES)

    assert result == json.loads((EXAMPLES / "search_result.example.json").read_text())


def test_a_root_returns_the_hidden_test_files_that_its_graded_diff_touched(tmp_path, fake_run_store):
    """Miles counts a resolved repair that changed a hidden test file as unresolved. A root reports it too."""
    body = miles_request(roots=1, search={**SEARCH, "roots": 1, "tf_student_continuations": 0})
    driver = Driver(fake_run_store.client, Ledger(tmp_path / "ledger"))
    adapter = MessageAdapter(driver, adapter_config(body))
    adapter.submit(body)
    driver.tick()
    fake_run_store.finish_rollout("job-0", exported_root())
    driver.tick()
    fake_run_store.finish("job-1", verdict(False, touched_test_paths=["tests/test_hidden.py"], **HIDDEN))
    driver.tick()
    fake_run_store.finish("job-2", reviewed())

    driver.tick()

    trajectories = adapter.get(body["rollout_job_id"])["trajectories"]
    assert rewards_and_hidden_tests(trajectories) == [(0.0, None, ["tests/test_hidden.py"])]

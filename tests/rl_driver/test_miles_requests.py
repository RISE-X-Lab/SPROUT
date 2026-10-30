"""What a Miles request and the adapter's ``miles`` config may be, before anything is submitted.

docs/RL_DRIVER.md ("The Miles protocol") gives the fields. ``MessageRequest.from_dict`` refuses an
invalid request with ValueError, and the ``MessageAdapter`` constructor an invalid config.
"""

from __future__ import annotations

import json
import re
from typing import Any

import pytest

from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.messages import MessageAdapter, MessageRequest
from tests.rl_driver.miles import adapter_config, miles_request, search, search_request, slots
from tests.support.paths import REPO

REQUEST_EXAMPLE = REPO / "examples" / "rl_driver" / "request.example.json"


def without(field: str) -> dict[str, Any]:
    """``miles_request()`` without ``field``."""
    body = miles_request()
    del body[field]
    return body


def test_the_shipped_request_example_is_a_valid_request_and_round_trips():
    """Miles's tests send this file, and docs/RL_DRIVER.md points at it."""
    body = json.loads(REQUEST_EXAMPLE.read_text())

    parsed = MessageRequest.from_dict(body).to_dict()

    assert parsed == {**body, "finalization_timeout_seconds": 1800.0}


@pytest.mark.parametrize(
    "body, expected",
    [
        pytest.param(
            miles_request(rollout_id=0, max_turns=1),
            miles_request(rollout_id=0, max_turns=1, finalization_timeout_seconds=1800.0),
            id="smallest-ids-and-turns",
        ),
        pytest.param(
            miles_request(sample_slots=slots(2), max_samples=2, minimum_returned_samples=2),
            miles_request(
                sample_slots=slots(2),
                max_samples=2,
                minimum_returned_samples=2,
                finalization_timeout_seconds=1800.0,
            ),
            id="every-slot-required",
        ),
        pytest.param(
            without("model") | {"finalization_timeout_seconds": 60},
            miles_request(model=None, finalization_timeout_seconds=60.0),
            id="model-left-to-run-defaults",
        ),
        pytest.param(
            without("sampling_params") | {"prompt": [{"role": "user", "content": "Fix the parser"}]},
            miles_request(
                sampling_params={},
                prompt=[{"role": "user", "content": "Fix the parser"}],
                finalization_timeout_seconds=1800.0,
            ),
            id="message-prompt-and-default-sampling",
        ),
        pytest.param(
            search_request(slot_count=3),
            search_request(slot_count=3) | {"finalization_timeout_seconds": 1800.0},
            id="search",
        ),
    ],
)
def test_a_valid_request_round_trips_with_its_defaults_filled_in(body, expected):
    parsed = MessageRequest.from_dict(body).to_dict()

    assert parsed == expected


@pytest.mark.parametrize(
    "body, message",
    [
        pytest.param(None, "Unknown message request fields", id="not-an-object"),
        pytest.param(
            miles_request(prompt_token_ids=[1, 2]), "Unknown message request fields", id="unknown-field"
        ),
        pytest.param(
            miles_request(rollout_job_id=""), "rollout_job_id must be a non-empty string", id="empty-id"
        ),
        pytest.param(
            miles_request(rollout_job_id="  "), "rollout_job_id must be a non-empty string", id="blank-id"
        ),
        pytest.param(
            miles_request(rollout_job_id=7), "rollout_job_id must be a non-empty string", id="number-id"
        ),
        pytest.param(without("rollout_job_id"), "rollout_job_id must be a non-empty string", id="missing-id"),
        pytest.param(
            miles_request(prompt_group_id=""), "prompt_group_id must be a non-empty string", id="group"
        ),
        pytest.param(miles_request(task_id=""), "task_id must be a non-empty string", id="task"),
        pytest.param(miles_request(image=""), "image must be a non-empty string", id="image"),
        pytest.param(
            miles_request(model_endpoint=""), "model_endpoint must be a non-empty string", id="endpoint"
        ),
        pytest.param(miles_request(model=""), "model must be a non-empty string", id="empty-model"),
        pytest.param(
            miles_request(rollout_id=-1), "rollout_id must be a non-negative integer", id="negative-rollout"
        ),
        pytest.param(
            miles_request(rollout_id=True), "rollout_id must be a non-negative integer", id="bool-rollout"
        ),
        pytest.param(miles_request(prompt=""), "prompt must be nonempty text or messages", id="empty-prompt"),
        pytest.param(miles_request(prompt=[]), "prompt must be nonempty text or messages", id="no-messages"),
        pytest.param(miles_request(prompt=5), "prompt must be nonempty text or messages", id="number-prompt"),
        pytest.param(miles_request(sample_slots=[]), "sample_slots must be nonempty", id="no-slots"),
        pytest.param(
            miles_request(sample_slots=["slot-0"]), "sample_slots entries must be objects", id="text-slot"
        ),
        pytest.param(
            miles_request(sample_slots=[{"sample_slot_id": "slot-0", "sample_index": 0, "weight": 1}]),
            "sample slot contains unknown fields: ['weight']",
            id="slot-field",
        ),
        pytest.param(
            miles_request(sample_slots=[{"sample_slot_id": "", "sample_index": 0}]),
            "sample_slot_id must be a non-empty string",
            id="slot-id",
        ),
        pytest.param(
            miles_request(sample_slots=[{"sample_slot_id": "slot-0", "sample_index": -1}]),
            "sample_index must be a non-negative integer",
            id="slot-index",
        ),
        pytest.param(
            miles_request(sample_slots=[{"sample_slot_id": "slot-0", "sample_index": i} for i in (0, 1)]),
            "Sample slots must have unique ids and indices",
            id="repeated-slot-id",
        ),
        pytest.param(
            miles_request(sample_slots=[{"sample_slot_id": f"slot-{i}", "sample_index": 0} for i in (0, 1)]),
            "Sample slots must have unique ids and indices",
            id="repeated-slot-index",
        ),
        pytest.param(miles_request(max_samples=0), "max_samples must be positive", id="no-samples"),
        pytest.param(
            miles_request(minimum_returned_samples=0),
            "minimum_returned_samples must be positive",
            id="no-minimum",
        ),
        pytest.param(miles_request(max_turns=0), "max_turns must be positive", id="no-turns"),
        pytest.param(
            miles_request(max_turns=-1), "max_turns must be a non-negative integer", id="negative-turns"
        ),
        pytest.param(
            miles_request(max_turns=True), "max_turns must be a non-negative integer", id="bool-turns"
        ),
        pytest.param(
            miles_request(max_turns=1.5), "max_turns must be a non-negative integer", id="float-turns"
        ),
        pytest.param(without("max_turns"), "max_turns must be a non-negative integer", id="missing-turns"),
        pytest.param(
            miles_request(sample_slots=slots(2), max_samples=1, minimum_returned_samples=2),
            "Invalid sample count bounds",
            id="minimum-above-maximum",
        ),
        pytest.param(miles_request(max_samples=2), "Invalid sample count bounds", id="maximum-above-slots"),
        pytest.param(
            miles_request(sampling_params=[]), "sampling_params must be an object", id="sampling-list"
        ),
        pytest.param(
            without("budgets"),
            "budgets contains only max_wall_time_seconds; use per-trajectory max_turns",
            id="missing-budgets",
        ),
        pytest.param(
            miles_request(budgets={"max_wall_time_seconds": 900.0, "max_model_calls": 100}),
            "budgets contains only max_wall_time_seconds; use per-trajectory max_turns",
            id="model-call-budget",
        ),
        pytest.param(
            miles_request(budgets={"max_wall_time_seconds": 900.0, "max_tool_calls": 100}),
            "budgets contains only max_wall_time_seconds; use per-trajectory max_turns",
            id="tool-call-budget",
        ),
        pytest.param(
            miles_request(budgets={"max_wall_time_seconds": 0}),
            "budgets.max_wall_time_seconds must be finite and positive",
            id="no-wall-time",
        ),
        pytest.param(
            miles_request(budgets={"max_wall_time_seconds": float("inf")}),
            "budgets.max_wall_time_seconds must be finite and positive",
            id="endless-wall-time",
        ),
        pytest.param(
            miles_request(budgets={"max_wall_time_seconds": True}),
            "budgets.max_wall_time_seconds must be finite and positive",
            id="bool-wall-time",
        ),
        pytest.param(
            miles_request(finalization_timeout_seconds=0),
            "finalization_timeout_seconds must be finite and positive",
            id="no-finalization",
        ),
        pytest.param(
            miles_request(finalization_timeout_seconds=float("nan")),
            "finalization_timeout_seconds must be finite and positive",
            id="nan-finalization",
        ),
        pytest.param(
            miles_request(sample_slots=slots(3), max_samples=3, search=search(roots=4)),
            "search.roots exceeds max_samples",
            id="roots-above-maximum",
        ),
        pytest.param(
            search_request(slot_count=3) | {"minimum_returned_samples": 2},
            "minimum_returned_samples may not exceed search.roots: only the roots are guaranteed",
            id="minimum-above-roots",
        ),
        pytest.param(
            miles_request(search={key: value for key, value in search().items() if key != "repair_rounds"}),
            "search names roots, points, candidates",
            id="search-key-missing",
        ),
        pytest.param(
            miles_request(search=search(rounds=2)),
            "search names roots, points, candidates",
            id="search-key-unknown",
        ),
        pytest.param(search_request(slot_count=3, roots=0), "search.roots must be positive", id="no-roots"),
        pytest.param(
            search_request(slot_count=3, candidates=0),
            "search.candidates and search.candidate_continuations are both zero or both positive",
            id="candidates-without-continuations",
        ),
        pytest.param(
            search_request(slot_count=3, tf_student_continuations=-1),
            "search.tf_student_continuations must be a non-negative integer",
            id="negative-tf-students",
        ),
        pytest.param(
            search_request(slot_count=3, tf_student_continuations=True),
            "search.tf_student_continuations must be a non-negative integer",
            id="bool-tf-students",
        ),
        pytest.param(
            search_request(slot_count=3, review_seconds=0),
            "search.review_seconds must be finite and positive",
            id="no-review-time",
        ),
        pytest.param(
            search_request(slot_count=9, repair_rounds=2, candidates=2),
            "more than one search.repair_rounds takes one candidate and one candidate continuation",
            id="rounds-with-two-candidates",
        ),
    ],
)
def test_an_invalid_request_is_refused_with_what_is_wrong(body, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        MessageRequest.from_dict(body)


# The slots that roots + SearchSpec.branches() need, by the formula in docs/RL_DRIVER.md.
BRANCH_BOUNDS = [
    pytest.param(
        {"roots": 2, "candidates": 2, "student_continuations": 2, "candidate_continuations": 2},
        14,
        id="doc-example",
    ),
    pytest.param(
        {
            "roots": 2,
            "points": 2,
            "candidates": 0,
            "candidate_continuations": 0,
            "tf_student_continuations": 5,
        },
        12,
        id="split-roots-widest",
    ),
    pytest.param({"roots": 2, "repair_rounds": 3}, 10, id="three-rounds"),
    pytest.param({"roots": 1, "tf_student_continuations": 9}, 3, id="one-root-never-splits"),
]


@pytest.mark.parametrize("fields, slot_count", BRANCH_BOUNDS)
def test_a_search_request_with_a_slot_for_every_branch_that_it_may_make_is_accepted(fields, slot_count):
    parsed = MessageRequest.from_dict(search_request(slot_count=slot_count, **fields))

    assert parsed.max_samples == slot_count


@pytest.mark.parametrize("fields, slot_count", BRANCH_BOUNDS)
def test_a_search_request_one_slot_short_of_its_branches_is_refused(fields, slot_count):
    with pytest.raises(
        ValueError, match="max_samples must cover the roots and every branch the search may make"
    ):
        MessageRequest.from_dict(search_request(slot_count=slot_count - 1, **fields))


@pytest.mark.parametrize(
    "config, message",
    [
        pytest.param(None, "Invalid Miles adapter config", id="not-an-object"),
        pytest.param(adapter_config(workers=4), "Invalid Miles adapter config", id="unknown-field"),
        pytest.param(
            adapter_config(run_defaults={"slot": "codex"}),
            "The Miles adapter runs the mini agent in mini-swe-agent or the harness in sprout-harness",
            id="other-slot",
        ),
        pytest.param(
            adapter_config(run_defaults={"extra": {"rollout_contract": {"max_turns": 5}}}),
            "rollout_contract is derived from each Miles request",
            id="fixed-contract",
        ),
        pytest.param(
            adapter_config(review=7),
            "The review config names a worker profile and the reviewer's model-call settings",
            id="review-not-an-object",
        ),
        pytest.param(
            adapter_config(review={"profile": "review", "top_p": 0.9}),
            "The review config names a worker profile and the reviewer's model-call settings",
            id="review-field",
        ),
        pytest.param(
            adapter_config(review={"temperature": 0.7}),
            "The review config names a worker profile and the reviewer's model-call settings",
            id="review-without-profile",
        ),
        pytest.param(
            adapter_config(review={"profile": ""}),
            "The review config names a worker profile and the reviewer's model-call settings",
            id="review-empty-profile",
        ),
        pytest.param(
            adapter_config(grade_wall_seconds=239),
            "grade_wall_seconds must be at least 240 s",
            id="short-grade",
        ),
        pytest.param(
            adapter_config(grade_wall_seconds=float("inf")),
            "grade_wall_seconds must be at least 240 s",
            id="endless-grade",
        ),
        pytest.param(
            adapter_config(grade_wall_seconds="600"),
            "grade_wall_seconds must be at least 240 s",
            id="text-grade",
        ),
    ],
)
def test_an_invalid_adapter_config_is_refused_when_the_adapter_is_made(
    fake_run_store, tmp_path, config, message
):
    driver = Driver(fake_run_store.client(), Ledger(tmp_path / "ledger.sqlite3"))

    with pytest.raises(ValueError, match=re.escape(message)):
        MessageAdapter(driver, config)

"""A review job's spec is checked when it is submitted, so that a worker never claims one it cannot run."""

from __future__ import annotations

from typing import Any

import pytest

from sprout.runstore.payload import encode_payload
from sprout.runstore.specs import JobSpec, ReviewSpec


def submitted(**changes: Any) -> dict[str, Any]:
    """A review as a submitter sends it: the worker fills in ``artifact_root``."""
    spec: dict[str, Any] = {
        "problem": "Fix the failing test.",
        "model": "fixture",
        "model_endpoint": "http://127.0.0.1:1",
        "api_key_env": "REVIEW_KEY",
        "parents": [
            {"name": "root-1", "job_id": "job-1", "attempt_id": "attempt-1", "grade": {"resolved": False}}
        ],
    }
    spec.update(changes)
    return spec


def one_parent(**changes: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": "root-1",
            "job_id": "job-1",
            "attempt_id": "attempt-1",
            "grade": {"resolved": False},
            **changes,
        }
    ]


def test_a_review_without_its_artifact_root_is_accepted_with_the_documented_defaults():
    spec = ReviewSpec(**submitted()).validate()

    assert {
        key: spec[key]
        for key in (
            "guidance",
            "count_mode",
            "points_per_attempt",
            "structured",
            "temperature",
            "max_concurrent_requests",
            "artifact_root",
        )
    } == {
        "guidance": "assistant-turn",
        "count_mode": "adaptive",
        "points_per_attempt": False,
        "structured": False,
        "temperature": 0.8,
        "max_concurrent_requests": 8,
        "artifact_root": None,
    }


@pytest.mark.parametrize(
    "changes",
    [
        {"points": 1, "candidates": 1, "max_concurrent_requests": 1},
        {"temperature": 0},
        {"temperature": 2},
        {"temperature": None},
        {"guidance": "none", "candidates": 1},
        {"leak_judge": {"threshold": 1, "resamples": 0, "model": "jev-1.13.0", "api_key_env": "TS_KEY"}},
    ],
    ids=[
        "the-least-counts",
        "temperature-0",
        "temperature-2",
        "the-endpoints-temperature",
        "none-guidance",
        "a-full-leak-judge",
    ],
)
def test_a_review_at_the_edge_of_each_range_is_accepted(changes):
    JobSpec("review", submitted(**changes)).validate()


@pytest.mark.parametrize(
    "changes, error",
    [
        ({"problem": " "}, "A review needs the task's problem statement"),
        ({"model": ""}, "A review needs model"),
        ({"model_endpoint": None}, "A review needs model_endpoint"),
        ({"guidance": "prose"}, "Review guidance must be assistant-turn, none or user-hint"),
        ({"count_mode": "exact"}, "Review count_mode must be adaptive or fixed"),
        ({"points": 0}, "Review points must be a positive integer"),
        ({"candidates": -1}, "Review candidates must be a positive integer"),
        ({"max_output_tokens": True}, "Review max_output_tokens must be a positive integer"),
        ({"prompt_chars": 2.0}, "Review prompt_chars must be a positive integer"),
        ({"max_concurrent_requests": "8"}, "Review max_concurrent_requests must be a positive integer"),
        ({"points_per_attempt": "yes"}, "Review points_per_attempt must be a boolean"),
        ({"structured": 1}, "Review structured must be a boolean"),
        (
            {"guidance": "user-hint", "candidates": 2},
            "A user-hint review writes no repair turns; candidates must be 1",
        ),
        ({"temperature": 2.1}, "Review temperature must be between 0 and 2"),
        ({"temperature": -0.1}, "Review temperature must be between 0 and 2"),
        ({"request_timeout_s": 0}, "Review request_timeout_s must be finite and positive"),
        ({"request_timeout_s": "90"}, "Review request_timeout_s must be finite and positive"),
        ({"prompts": {"path": "/path/to/prompts.yaml"}}, "Review prompts names the path and the sha256"),
        ({"leak_judge": ["threshold"]}, "leak_judge names only threshold, model, resamples and api_key_env"),
        ({"leak_judge": {"other": 1}}, "leak_judge names only threshold, model, resamples and api_key_env"),
        ({"leak_judge": {"threshold": 0}}, r"leak_judge threshold must be in \(0, 1\]"),
        ({"leak_judge": {"threshold": 1.5}}, r"leak_judge threshold must be in \(0, 1\]"),
        ({"leak_judge": {"model": ""}}, "leak_judge model must be a model name"),
        ({"leak_judge": {"resamples": -1}}, "leak_judge resamples must be a non-negative integer"),
        ({"leak_judge": {"resamples": 1.0}}, "leak_judge resamples must be a non-negative integer"),
        (
            {"leak_judge": {"api_key_env": "NOT A NAME"}},
            "leak_judge api_key_env must name one environment variable",
        ),
        ({"parents": []}, "A review needs at least one finished attempt to read"),
        (
            {"parents": one_parent(verdict=True)},
            "A review parent names name, job_id, attempt_id, grade and optionally hint",
        ),
        ({"parents": one_parent(job_id=" ")}, "A review parent needs job_id"),
        ({"parents": one_parent() * 2}, "Review parents need distinct names"),
        (
            {"parents": one_parent(grade={"progress": 0.5})},
            "A review parent needs its grade with a resolved verdict",
        ),
        (
            {"parents": one_parent(grade={"resolved": 0})},
            "A review parent needs its grade with a resolved verdict",
        ),
        ({"parents": one_parent(hint=["a", "list"])}, "A review parent's hint must be text"),
        ({"prompt": "x"}, "Unknown ReviewSpec fields"),
    ],
)
def test_a_review_that_a_worker_cannot_run_is_refused_at_submission(changes, error):
    with pytest.raises(ValueError, match=error):
        JobSpec("review", submitted(**changes)).validate()


def test_a_review_of_tests_named_after_secrets_is_accepted_and_sent_to_the_worker():
    """A grade's ``test_statuses`` is keyed by test ids, which the credential check must not read as keys.

    tfsec's suite has ``Test_AzureVMWithPasswordAuth``: its review was refused at submission."""
    statuses = {"Test_AzureVMWithPasswordAuth": "FAILED", "TestIsSensitiveAttribute/secret": "PASSED"}
    job = JobSpec(
        "review", submitted(parents=one_parent(grade={"resolved": False, "test_statuses": statuses}))
    ).validate()
    envelope = {"version": 1, "job_id": "review-job", "attempt_id": "attempt-1", "kind": "review"}

    data = encode_payload(
        {**envelope, "effective_spec": job["spec"], "profile_config": {}, "recovery": None},
        "review-job",
        "attempt-1",
    )

    assert b"Test_AzureVMWithPasswordAuth" in data


def test_a_review_whose_grade_holds_an_inline_credential_is_refused():
    with pytest.raises(ValueError, match="inline credentials"):
        JobSpec(
            "review", submitted(parents=one_parent(grade={"resolved": False, "api_key": "sk-literal"}))
        ).validate()

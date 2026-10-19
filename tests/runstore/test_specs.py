"""Which requests the Run Store accepts (``JobSpec.validate``), and the JSON form by which it compares them.

Each refusal is a request that a worker cannot run, refused before the Run Store stores it.
"""

from __future__ import annotations

import re
from typing import Any

import pytest

from sprout.runstore.specs import (
    JobSpec,
    canonical,
    defaults_key,
    digest,
    no_credentials,
    validate_branch_budget,
    validate_continuation,
)
from tests.runstore.builders import (
    TASK_DIR,
    harbor_grade_spec,
    review_parent,
    review_spec,
    rollout_spec,
    swerebench_grade_spec,
)

NAN, INF = float("nan"), float("inf")


def request(**fields: Any) -> JobSpec:
    """A rollout request with ``fields`` over its own fields."""
    return JobSpec(**{"kind": "rollout", "spec": rollout_spec(), **fields})


def refused(spec: JobSpec, error: str) -> None:
    with pytest.raises(ValueError, match=re.escape(error)):
        spec.validate()


# The canonical form and its digest


def test_the_canonical_form_sorts_keys_drops_spaces_and_keeps_non_ascii_text():
    """Two submissions are the same request when their canonical forms are equal."""
    assert canonical({"b": [1, 2.5, None, True], "a": "caf\u00e9 \u6d4b"}) == (
        '{"a":"caf\u00e9 \u6d4b","b":[1,2.5,null,true]}'
    )


def test_the_digest_of_a_value_stays_the_hash_that_stored_requests_carry():
    """Each stored job carries the digest of its request. Another digest makes its repeat a conflict."""
    assert digest({"a": "caf\u00e9 \u6d4b", "b": [1, 2.5, None, True]}) == (
        "7509daa0b8816d131a2437801ef6791c9c6b90a5853bea08a3801bc2c3d976c6"
    )


@pytest.mark.parametrize("value", [NAN, INF, -INF], ids=["nan", "infinity", "minus-infinity"])
def test_the_canonical_form_refuses_a_number_that_json_cannot_hold(value):
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        canonical({"timeout_s": value})


# Inline credentials


@pytest.mark.parametrize(
    "value",
    [
        {"api_key": "sk-live"},
        {"Api-Key": "sk-live"},
        {"OPENAI_API_KEY": "sk-live"},
        {"auth_token": "token"},
        {"Authorization": "Bearer token"},
        {"password": "hunter2"},
        {"bearer_token": "token"},
        {"client_secret": "secret"},
        {"backend": {"microvm": {"api_key": "sk-live"}}},
        {"servers": [{"api_key": "sk-live"}]},
        {"api_key": {"$env": "MODEL_KEY", "default": "sk-live"}},
    ],
    ids=[
        "api-key",
        "dashed-mixed-case",
        "prefixed",
        "auth-token",
        "authorization",
        "password",
        "bearer-token",
        "secret",
        "nested-in-a-mapping",
        "nested-in-a-list",
        "env-reference-with-a-fallback",
    ],
)
def test_an_inline_credential_is_refused(value):
    with pytest.raises(ValueError, match=r"Use a worker-side \$env reference instead of inline credentials"):
        no_credentials(value)


@pytest.mark.parametrize(
    "value",
    [
        {"api_key": {"$env": "MODEL_KEY"}},
        {"api_key_env": "MODEL_KEY"},
        {"api_key": ""},
        {"api_key": None},
        {"test_statuses": {"Test_AzureVMWithPasswordAuth": "PASSED"}},
        {"model": "secret-model"},
    ],
    ids=["env-reference", "env-name", "empty", "null", "test-statuses", "secret-word-in-a-value"],
)
def test_a_reference_an_empty_value_and_data_are_not_credentials(value):
    assert no_credentials(value) is None


# The JobSpec body


def test_the_body_of_a_request_leaves_out_the_branch_fields_and_the_affinity_that_it_does_not_set():
    """A job stored before these fields existed must get the same hash when submitted again."""
    assert request().validate() == {
        "kind": "rollout",
        "spec": {"prompt": "Write the answer."},
        "profile": "default",
        "context": {},
        "max_infra_retries": 2,
        "parent_point": None,
    }


@pytest.mark.parametrize("affinity", ["g", "g" * 256], ids=["one-character", "256-characters"])
def test_the_body_of_a_request_keeps_the_affinity_that_it_sets(affinity):
    assert request(affinity=affinity).validate()["affinity"] == affinity


@pytest.mark.parametrize("retries", [0, 2])
def test_zero_to_two_infrastructure_retries_are_accepted(retries):
    assert request(max_infra_retries=retries).validate()["max_infra_retries"] == retries


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"spec": ["Write the answer."]}, "spec and context must be JSON objects"),
        ({"context": "task-1"}, "spec and context must be JSON objects"),
        ({"context": {"score": NAN}}, "Out of range float values are not JSON compliant"),
        ({"max_infra_retries": -1}, "max_infra_retries must be 0..2"),
        ({"max_infra_retries": 3}, "max_infra_retries must be 0..2"),
        ({"max_infra_retries": True}, "max_infra_retries must be 0..2"),
        ({"profile": ""}, "profile must be nonempty"),
        ({"profile": None}, "profile must be nonempty"),
        ({"kind": "train"}, "kind must be rollout, grade or review"),
        ({"spec": rollout_spec(backend={"microvm": {"api_key": "sk-live"}})}, "Use a worker-side $env"),
        ({"affinity": ""}, "affinity must be nonempty text of at most 256 characters"),
        ({"affinity": " "}, "affinity must be nonempty text of at most 256 characters"),
        ({"affinity": "g" * 257}, "affinity must be nonempty text of at most 256 characters"),
        ({"affinity": 7}, "affinity must be nonempty text of at most 256 characters"),
    ],
    ids=[
        "spec-not-an-object",
        "context-not-an-object",
        "context-not-json",
        "retries-below-0",
        "retries-above-2",
        "retries-a-boolean",
        "blank-profile",
        "no-profile",
        "unknown-kind",
        "inline-credential",
        "empty-affinity",
        "blank-affinity",
        "affinity-of-257-characters",
        "affinity-not-text",
    ],
)
def test_a_request_with_an_invalid_field_is_refused(fields, error):
    refused(request(**fields), error)


@pytest.mark.parametrize(
    "kind,spec",
    [
        ("rollout", rollout_spec(timeout_s=0)),
        ("rollout", rollout_spec(timeout_s=-1)),
        ("rollout", rollout_spec(timeout_s="3600")),
        ("rollout", rollout_spec(timeout_s=None)),
        ("grade", harbor_grade_spec(timeout_s=0)),
        ("review", review_spec(timeout_s=0)),
    ],
    ids=["zero", "negative", "text", "null", "grade", "review"],
)
def test_a_job_of_any_kind_needs_a_finite_positive_timeout(kind, spec):
    refused(JobSpec(kind, spec), "timeout_s must be finite and positive")


def test_a_request_from_a_body_with_an_unknown_field_raises_type_error():
    with pytest.raises(TypeError):
        JobSpec.from_dict({"kind": "rollout", "spec": rollout_spec(), "priority": 1})


# Branch fields on a request


def test_a_rollout_branch_names_its_parent_point_and_its_continuation():
    body = request(parent_point="point-1", continuation={"guidance": "none"}).validate()

    assert (body["parent_point"], body["continuation"]) == ("point-1", {"guidance": "none"})


@pytest.mark.parametrize(
    "fields,error",
    [
        ({"continuation": {"guidance": "none"}}, "A continuation belongs to a rollout branch"),
        (
            {"kind": "grade", "spec": harbor_grade_spec(), "parent_point": "point-1"},
            "A continuation belongs to a rollout branch",
        ),
        ({"parent_point": "point-1"}, "A continuation is an object with guidance"),
        (
            {"parent_point": "point-1", "continuation": {"guidance": "user-hint"}},
            "A user-hint branch needs a non-empty hint",
        ),
        ({"branch_budget": {"max_turns": 3}}, "A branch budget belongs to a branch"),
        (
            {"parent_point": "point-1", "continuation": {"guidance": "none"}, "branch_budget": {}},
            "A branch budget sets max_turns and/or deadline_at",
        ),
    ],
    ids=[
        "continuation-without-a-point",
        "branch-of-a-grade",
        "point-without-a-continuation",
        "invalid-continuation",
        "budget-without-a-point",
        "invalid-budget",
    ],
)
def test_branch_fields_outside_a_valid_rollout_branch_are_refused(fields, error):
    refused(request(**fields), error)


@pytest.mark.parametrize(
    "budget",
    [
        {"max_turns": 1},
        {"deadline_at": 1.5},
        {"deadline_at": None},
        {"max_turns": 5, "deadline_at": 1700000000},
    ],
    ids=["turns", "deadline", "no-deadline", "both"],
)
def test_a_branch_budget_sets_max_turns_or_deadline_at_or_both(budget):
    assert validate_branch_budget(budget) == budget


@pytest.mark.parametrize(
    "budget,error",
    [
        ({}, "A branch budget sets max_turns and/or deadline_at"),
        (["max_turns"], "A branch budget sets max_turns and/or deadline_at"),
        ({"max_turns": 3, "budget_usd": 1.0}, "A branch budget sets max_turns and/or deadline_at"),
        ({"max_turns": 0}, "branch budget max_turns must be a positive integer"),
        ({"max_turns": True}, "branch budget max_turns must be a positive integer"),
        ({"max_turns": 2.5}, "branch budget max_turns must be a positive integer"),
        ({"deadline_at": 0}, "branch budget deadline_at must be a finite positive timestamp"),
        ({"deadline_at": INF}, "branch budget deadline_at must be a finite positive timestamp"),
        ({"deadline_at": NAN}, "branch budget deadline_at must be a finite positive timestamp"),
        ({"deadline_at": "2030-01-01"}, "branch budget deadline_at must be a finite positive timestamp"),
    ],
    ids=[
        "empty",
        "not-an-object",
        "unknown-key",
        "zero-turns",
        "boolean-turns",
        "fractional-turns",
        "zero-deadline",
        "infinite-deadline",
        "nan-deadline",
        "text-deadline",
    ],
)
def test_an_invalid_branch_budget_is_refused(budget, error):
    with pytest.raises(ValueError, match=re.escape(error)):
        validate_branch_budget(budget)


@pytest.mark.parametrize(
    "change",
    [{"model": "another-model"}, {"timeout_s": 60}, {"budget_usd": 0.5}],
    ids=["model", "timeout", "budget"],
)
def test_a_continuation_may_change_the_model_the_timeout_and_the_budget(change):
    parent = request(spec=rollout_spec(model="policy", timeout_s=600)).validate()
    child = request(spec={**rollout_spec(model="policy", timeout_s=600), **change}).validate()

    assert validate_continuation(parent, child) is None


@pytest.mark.parametrize(
    "child",
    [
        {"kind": "rollout", "profile": "default", "spec": rollout_spec(prompt="Another task.")},
        {"kind": "rollout", "profile": "default", "spec": rollout_spec(sandbox_image="another:1")},
        {"kind": "rollout", "profile": "other", "spec": rollout_spec()},
        {"kind": "grade", "profile": "default", "spec": rollout_spec()},
    ],
    ids=["prompt", "environment", "profile", "kind"],
)
def test_a_continuation_that_changes_anything_else_is_refused(child):
    parent = {"kind": "rollout", "profile": "default", "spec": rollout_spec()}

    with pytest.raises(ValueError, match="Continuation must retain its source environment"):
        validate_continuation(parent, child)


# Rollouts


@pytest.mark.parametrize(
    "spec",
    [
        rollout_spec(keep_sandbox=True),
        rollout_spec(agent="oracle", harbor_task=TASK_DIR),
        rollout_spec(agent="nop"),
        rollout_spec(agent="mini", harbor_task=TASK_DIR, keep_sandbox=True),
        rollout_spec(slot="mini-swe-agent", cwd=".", run_id=None),
    ],
    ids=["kept-sandbox", "oracle", "nop", "mini-on-a-harbor-task", "the-defaults-spelled-out"],
)
def test_a_queued_rollout_may_keep_its_sandbox_and_run_each_agent(spec):
    assert JobSpec("rollout", spec).validate()["spec"] == spec


@pytest.mark.parametrize(
    "spec,error",
    [
        ({"prompt": "  "}, "RunSpec requires a nonempty prompt"),
        ({"model": "policy"}, "RunSpec requires a nonempty prompt"),
        (rollout_spec(sandbox="podman"), "Unknown RunSpec fields"),
        (rollout_spec(slot="codex-cli"), "The only queued-run slot is mini-swe-agent"),
        (rollout_spec(agent="codex"), "RunSpec agent must be mini, oracle or nop"),
        (rollout_spec(agent="nop", harbor_task=" "), "RunSpec harbor_task must name a Harbor task directory"),
        (rollout_spec(agent="oracle"), "An oracle run needs the harbor_task whose solution it runs"),
        (rollout_spec(journal_path="/path/to/journal.jsonl"), "Queued runs cannot supply journal_path"),
        (rollout_spec(run_id="run-1"), "Queued runs cannot supply run_id"),
        (rollout_spec(cwd="/path/to/repo"), "Queued runs use an isolated worker-owned cwd"),
        (rollout_spec(keep_sandbox="yes"), "RunSpec keep_sandbox must be a boolean"),
        (rollout_spec(extra=["rollout_contract"]), "RunSpec extra must be a JSON object"),
        (rollout_spec(extra={"native_prefix": {}}), "Native restoration fields are worker-owned"),
        (rollout_spec(extra={"native_home": "/path/to/home"}), "Native restoration fields are worker-owned"),
        (rollout_spec(extra={"checkpoint_identity": "c1"}), "Native restoration fields are worker-owned"),
    ],
    ids=[
        "blank-prompt",
        "no-prompt",
        "unknown-field",
        "other-slot",
        "unknown-agent",
        "blank-harbor-task",
        "oracle-without-a-task",
        "journal-path",
        "run-id",
        "host-cwd",
        "keep-sandbox-not-boolean",
        "extra-not-an-object",
        "native-prefix",
        "native-home",
        "checkpoint-identity",
    ],
)
def test_a_rollout_that_a_worker_cannot_run_is_refused(spec, error):
    refused(JobSpec("rollout", spec), error)


# Grades


@pytest.mark.parametrize(
    "spec",
    [
        {"benchmark": "harbor", "task_dir": TASK_DIR, "snapshot_id": "snapshot-1"},
        {"benchmark": "harbor", "task_dir": TASK_DIR, "in_place_job": "rollout-job"},
        swerebench_grade_spec(),
        swerebench_grade_spec(verifier_network="allow"),
    ],
    ids=["harbor-from-a-snapshot", "harbor-in-a-kept-sandbox", "swe-rebench", "verifier-network-allowed"],
)
def test_a_grade_with_its_task_and_one_source_is_accepted(spec):
    assert JobSpec("grade", spec).validate()["spec"] == spec


@pytest.mark.parametrize(
    "spec,error",
    [
        (harbor_grade_spec(benchmark="swe-bench"), "Unsupported official grader"),
        (harbor_grade_spec(verifier_network="open"), "verifier_network must be allow or deny"),
        (harbor_grade_spec(task_dir=None), "A harbor grade needs the task_dir of its Harbor task"),
        (harbor_grade_spec(task_dir=" "), "A harbor grade needs the task_dir of its Harbor task"),
        (
            harbor_grade_spec(snapshot_id=None),
            "A harbor grade names exactly one of snapshot_id and in_place_job",
        ),
        (
            harbor_grade_spec(in_place_job="rollout-job"),
            "A harbor grade names exactly one of snapshot_id and in_place_job",
        ),
        (
            harbor_grade_spec(snapshot_id=""),
            "A harbor grade's snapshot_id or in_place_job must be nonempty text",
        ),
        (
            harbor_grade_spec(snapshot_id=None, in_place_job=" "),
            "A harbor grade's snapshot_id or in_place_job must be nonempty text",
        ),
        (swerebench_grade_spec(task_dir=TASK_DIR), "task_dir and in_place_job belong to harbor grades"),
        (
            swerebench_grade_spec(in_place_job="rollout-job"),
            "task_dir and in_place_job belong to harbor grades",
        ),
        (swerebench_grade_spec(parser_path=None), "SWE-rebench grading requires a deployment parser_path"),
        (
            swerebench_grade_spec(image=None),
            "SWE-rebench grading requires the task image the diff is applied to",
        ),
    ],
    ids=[
        "unknown-benchmark",
        "unknown-verifier-network",
        "harbor-without-a-task",
        "harbor-with-a-blank-task",
        "harbor-without-a-source",
        "harbor-with-two-sources",
        "harbor-with-a-blank-snapshot",
        "harbor-with-a-blank-job",
        "swe-rebench-with-a-task-dir",
        "swe-rebench-in-place",
        "swe-rebench-without-a-parser",
        "swe-rebench-without-an-image",
    ],
)
def test_a_grade_that_its_grader_cannot_run_is_refused(spec, error):
    refused(JobSpec("grade", spec), error)


@pytest.mark.parametrize(
    "missing", ["instance_id", "snapshot_id", "dataset_path", "dataset_sha256", "grader_revision"]
)
def test_a_swerebench_grade_needs_each_frozen_reference(missing):
    refused(
        JobSpec("grade", swerebench_grade_spec(**{missing: None})),
        "Grading requires frozen task, snapshot and grader references",
    )


def test_a_grade_with_a_field_that_grade_spec_lacks_raises_type_error():
    with pytest.raises(TypeError):
        JobSpec("grade", harbor_grade_spec(priority=1)).validate()


# Reviews


@pytest.mark.parametrize(
    "spec",
    [
        review_spec(),
        review_spec(guidance="user-hint", candidates=1),
        review_spec(guidance="none", candidates=1, count_mode="fixed"),
        review_spec(temperature=None),
        review_spec(temperature=0),
        review_spec(temperature=2),
        review_spec(request_timeout_s=None),
        review_spec(prompts={"path": "/path/to/prompts.yaml", "sha256": "2" * 64}),
        review_spec(leak_judge={}),
        review_spec(
            leak_judge={"threshold": 1, "model": "judge", "resamples": 0, "api_key_env": "JUDGE_KEY"}
        ),
        review_spec(parents=[review_parent(hint="Look at the parser.")]),
    ],
    ids=[
        "assistant-turn",
        "user-hint",
        "none",
        "no-temperature",
        "lowest-temperature",
        "highest-temperature",
        "no-request-timeout",
        "prompts-file",
        "default-leak-judge",
        "leak-judge-at-its-limits",
        "parent-with-a-hint",
    ],
)
def test_a_review_that_a_worker_can_run_is_accepted(spec):
    assert JobSpec("review", spec).validate()["spec"] == spec


@pytest.mark.parametrize(
    "name",
    [
        "points",
        "candidates",
        "max_output_tokens",
        "prompt_chars",
        "transcript_tokens",
        "max_attempts",
        "round_no",
        "max_concurrent_requests",
    ],
)
def test_each_count_of_a_review_must_be_a_positive_integer(name):
    refused(JobSpec("review", review_spec(**{name: 0})), "Review %s must be a positive integer" % name)


@pytest.mark.parametrize(
    "changes,error",
    [
        ({"problem": " "}, "A review needs the task's problem statement"),
        ({"model": ""}, "A review needs model"),
        ({"model_endpoint": None}, "A review needs model_endpoint"),
        ({"guidance": "rewrite"}, "Review guidance must be assistant-turn, none or user-hint"),
        ({"count_mode": "exact"}, "Review count_mode must be adaptive or fixed"),
        ({"points": True}, "Review points must be a positive integer"),
        ({"points": 1.5}, "Review points must be a positive integer"),
        ({"points_per_attempt": 1}, "Review points_per_attempt must be a boolean"),
        ({"structured": "yes"}, "Review structured must be a boolean"),
        ({"guidance": "user-hint", "candidates": 4}, "A user-hint review writes no repair turns"),
        ({"guidance": "none", "candidates": 2}, "A none review writes no repair turns"),
        ({"temperature": -0.1}, "Review temperature must be between 0 and 2"),
        ({"temperature": 2.1}, "Review temperature must be between 0 and 2"),
        ({"temperature": "0.8"}, "Review temperature must be between 0 and 2"),
        ({"request_timeout_s": 0}, "Review request_timeout_s must be finite and positive"),
        ({"request_timeout_s": "60"}, "Review request_timeout_s must be finite and positive"),
        ({"prompts": {"path": "/path/to/prompts.yaml"}}, "Review prompts names the path and the sha256"),
        ({"prompts": {"path": " ", "sha256": "2" * 64}}, "Review prompts names the path and the sha256"),
        (
            {"leak_judge": {"budget": 1}},
            "Review leak_judge names only threshold, model, resamples and api_key_env",
        ),
        ({"leak_judge": {"threshold": 0}}, "Review leak_judge threshold must be in (0, 1]"),
        ({"leak_judge": {"threshold": 1.5}}, "Review leak_judge threshold must be in (0, 1]"),
        ({"leak_judge": {"model": " "}}, "Review leak_judge model must be a model name"),
        ({"leak_judge": {"resamples": -1}}, "Review leak_judge resamples must be a non-negative integer"),
        (
            {"leak_judge": {"api_key_env": "JUDGE KEY"}},
            "Review leak_judge api_key_env must name one environment",
        ),
        ({"priority": 1}, "Unknown ReviewSpec fields"),
    ],
    ids=[
        "blank-problem",
        "blank-model",
        "no-endpoint",
        "unknown-guidance",
        "unknown-count-mode",
        "boolean-count",
        "fractional-count",
        "per-attempt-not-boolean",
        "structured-not-boolean",
        "user-hint-with-candidates",
        "none-with-candidates",
        "temperature-below-0",
        "temperature-above-2",
        "temperature-text",
        "zero-request-timeout",
        "text-request-timeout",
        "prompts-without-a-checksum",
        "prompts-with-a-blank-path",
        "unknown-judge-setting",
        "judge-threshold-0",
        "judge-threshold-above-1",
        "blank-judge-model",
        "negative-resamples",
        "judge-key-not-a-variable-name",
        "unknown-field",
    ],
)
def test_a_review_that_a_worker_cannot_run_is_refused(changes, error):
    refused(JobSpec("review", review_spec(**changes)), error)


@pytest.mark.parametrize(
    "parents,error",
    [
        ([], "A review needs at least one finished attempt to read"),
        (review_parent(), "A review needs at least one finished attempt to read"),
        (
            [review_parent(round=1)],
            "A review parent names name, job_id, attempt_id, grade and optionally hint",
        ),
        ([review_parent(name=" ")], "A review parent needs name"),
        ([review_parent(job_id=None)], "A review parent needs job_id"),
        ([review_parent(attempt_id="")], "A review parent needs attempt_id"),
        ([review_parent(), review_parent(job_id="other")], "Review parents need distinct names"),
        ([review_parent(grade=None)], "A review parent needs its grade with a resolved verdict"),
        ([review_parent(grade={"resolved": 0})], "A review parent needs its grade with a resolved verdict"),
        ([review_parent(hint=["Look."])], "A review parent's hint must be text"),
    ],
    ids=[
        "no-parent",
        "not-a-list",
        "unknown-key",
        "blank-name",
        "no-job",
        "blank-attempt",
        "duplicate-names",
        "no-grade",
        "verdict-not-boolean",
        "hint-not-text",
    ],
)
def test_a_review_needs_distinct_graded_parents(parents, error):
    refused(JobSpec("review", review_spec(parents=parents)), error)


# Profile sections


@pytest.mark.parametrize(
    "kind,section",
    [("rollout", "run_defaults"), ("grade", "grade_defaults"), ("review", "review_defaults")],
)
def test_each_job_kind_takes_its_defaults_from_its_own_profile_section(kind, section):
    assert defaults_key(kind) == section


def test_a_kind_without_a_profile_section_is_refused():
    with pytest.raises(ValueError, match="kind must be rollout, grade or review"):
        defaults_key("train")

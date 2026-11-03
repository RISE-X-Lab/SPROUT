"""The request bodies that the Run Store accepts: ``JobSpec`` and the specs of its three job kinds.

``JobSpec.validate`` refuses a request that a worker cannot run, before the Run Store stores it.
A request says what to run. The Run Store keeps the lease and the result of a job apart from it.
``canonical`` and ``digest`` give the JSON form and the hash by which two submissions are compared.
docs/RUNSTORE.md describes the fields.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from sprout.contract.continuation import Continuation
from sprout.contract.run_spec import RunSpec

#: What a branch may change about the run it continues. What the actor is given
#: next is the branch's continuation, never a different prompt.
BRANCH_OVERRIDES = frozenset({"model", "timeout_s", "budget_usd"})
#: The rollout-contract limits that a branch can set over the contract that it inherits.
#: docs/RUNSTORE.md ("Branch") says when a branch needs them.
BRANCH_BUDGET_FIELDS = frozenset({"max_turns", "deadline_at"})


def validate_branch_budget(value: object) -> dict[str, Any]:
    """A copy of the branch budget ``value``.

    Raises ValueError unless it sets ``max_turns``, ``deadline_at`` or both. ``max_turns`` must be a
    positive int, and ``deadline_at`` a finite positive Unix time or None.
    """
    if not isinstance(value, dict) or not value or set(value) - BRANCH_BUDGET_FIELDS:
        raise ValueError("A branch budget sets max_turns and/or deadline_at")
    if "max_turns" in value and (type(value["max_turns"]) is not int or value["max_turns"] <= 0):
        raise ValueError("branch budget max_turns must be a positive integer")
    deadline = value.get("deadline_at")
    if deadline is not None and (
        type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0
    ):
        raise ValueError("branch budget deadline_at must be a finite positive timestamp")
    return dict(value)


def canonical(value: Any) -> str:
    """The canonical JSON text of ``value``: sorted keys and no spaces.

    Raises ValueError for a NaN or an infinity, and TypeError for a value that JSON cannot hold.
    """
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    """The SHA-256 hex digest of ``canonical(value)``."""
    return hashlib.sha256(canonical(value).encode()).hexdigest()


#: Keys whose mappings hold data, not configuration, so ``no_credentials`` skips them. A test id in a
#: grade's ``test_statuses`` can contain a secret's name, as ``Test_AzureVMWithPasswordAuth`` does.
DATA_MAPPINGS = frozenset({"test_statuses"})


def no_credentials(value: Any) -> None:
    """Raise ValueError if the JSON ``value`` holds an inline credential.

    A credential is a nonempty value under a key that names a secret, such as ``api_key``. An
    ``{"$env": NAME}`` reference, a key that ends in ``_env`` and a key in ``DATA_MAPPINGS`` are allowed.
    """
    if isinstance(value, dict):
        for key, item in value.items():
            if key in DATA_MAPPINGS:
                continue
            normalized = key.lower().replace("-", "_")
            secret = any(
                part in normalized
                for part in ("api_key", "auth_token", "authorization", "password", "bearer_token", "secret")
            )
            reference = isinstance(item, dict) and set(item) == {"$env"}
            if secret and item and not reference and not normalized.endswith("_env"):
                raise ValueError("Use a worker-side $env reference instead of inline credentials")
            no_credentials(item)
    elif isinstance(value, list):
        for item in value:
            no_credentials(item)


def validate_continuation(parent: dict[str, Any], child: dict[str, Any]) -> None:
    """Raise ValueError unless the request body ``child`` can continue the request body ``parent``.

    Both are rollouts of one profile, and their specs differ only in ``BRANCH_OVERRIDES``.
    """
    parent_spec = {key: value for key, value in parent["spec"].items() if key not in BRANCH_OVERRIDES}
    child_spec = {key: value for key, value in child["spec"].items() if key not in BRANCH_OVERRIDES}
    if (
        parent["kind"] != "rollout"
        or child["kind"] != "rollout"
        or parent["profile"] != child["profile"]
        or parent_spec != child_spec
    ):
        raise ValueError("Continuation must retain its source environment, slot and tool configuration")


def _is_text(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


@dataclass(frozen=True)
class GradeSpec:
    """A grade job: the verdict of one benchmark on the work of one rollout.

    ``swe-rebench-v2`` grades the diff in ``snapshot_id`` against a pinned dataset row. ``harbor``
    runs the verifier of ``task_dir`` from ``snapshot_id``, or in the sandbox that ``in_place_job`` kept.
    """

    benchmark: str
    instance_id: str | None = None
    snapshot_id: str | None = None
    dataset_path: str | None = None
    dataset_sha256: str | None = None
    grader_revision: str | None = None
    backend: dict[str, Any] = field(default_factory=dict)
    resources: dict[str, Any] = field(default_factory=lambda: {"cpu": 2, "memory_mb": 12288})
    timeout_s: float = 3600
    verifier_network: str = "deny"
    #: The task's pristine image. The ``swe-rebench-v2`` grader applies the model's diff to a fresh
    #: sandbox of it, never to the agent's own filesystem.
    image: str | None = None
    parser_path: str | None = None
    #: The Harbor task directory whose verifier a ``harbor`` grade runs.
    task_dir: str | None = None
    #: The rollout job whose kept sandbox a ``harbor`` grade runs in. The Run Store
    #: binds the grade to the worker that keeps the sandbox (``Store.claim``).
    in_place_job: str | None = None

    def validate(self) -> None:
        """Raise ValueError unless the grader of ``benchmark`` can run this grade."""
        if self.benchmark not in ("swe-rebench-v2", "harbor"):
            raise ValueError("Unsupported official grader")
        if self.verifier_network not in ("allow", "deny"):
            raise ValueError("verifier_network must be allow or deny")
        if self.benchmark == "harbor":
            self._validate_harbor()
        else:
            self._validate_swerebench()

    def _validate_swerebench(self) -> None:
        if self.task_dir is not None or self.in_place_job is not None:
            raise ValueError("task_dir and in_place_job belong to harbor grades")
        if not all(
            (
                self.instance_id,
                self.snapshot_id,
                self.dataset_path,
                self.dataset_sha256,
                self.grader_revision,
            )
        ):
            raise ValueError("Grading requires frozen task, snapshot and grader references")
        if not self.parser_path:
            raise ValueError("SWE-rebench grading requires a deployment parser_path")
        if not self.image:
            raise ValueError("SWE-rebench grading requires the task image the diff is applied to")

    def _validate_harbor(self) -> None:
        if not _is_text(self.task_dir):
            raise ValueError("A harbor grade needs the task_dir of its Harbor task")
        if (self.snapshot_id is None) == (self.in_place_job is None):
            raise ValueError("A harbor grade names exactly one of snapshot_id and in_place_job")
        if not _is_text(self.in_place_job if self.snapshot_id is None else self.snapshot_id):
            raise ValueError("A harbor grade's snapshot_id or in_place_job must be nonempty text")


#: What ``RunSpec.agent`` of a queued rollout may name.
_RUN_AGENTS = ("mini", "oracle", "nop")


def _validate_run(spec: dict[str, Any]) -> None:
    """Raise ValueError unless a worker can run ``spec`` as a queued rollout."""
    allowed = {item.name for item in fields(RunSpec)}
    if set(spec) - allowed:
        raise ValueError("Unknown RunSpec fields")
    if not isinstance(spec.get("prompt"), str) or not spec["prompt"].strip():
        raise ValueError("RunSpec requires a nonempty prompt")
    if spec.get("slot", "mini-swe-agent") not in {"mini-swe-agent", "opencollab-base"}:
        raise ValueError("Queued-run slot must be mini-swe-agent or opencollab-base")
    agent = spec.get("agent", "mini")
    if agent not in _RUN_AGENTS:
        raise ValueError("RunSpec agent must be mini, oracle or nop")
    if spec.get("harbor_task") is not None and not _is_text(spec["harbor_task"]):
        raise ValueError("RunSpec harbor_task must name a Harbor task directory")
    if agent == "oracle" and spec.get("harbor_task") is None:
        raise ValueError("An oracle run needs the harbor_task whose solution it runs")
    for name in ("session", "journal_path", "run_id"):
        if spec.get(name) is not None:
            raise ValueError(f"Queued runs cannot supply {name}")
    if spec.get("cwd", ".") != ".":
        raise ValueError("Queued runs use an isolated worker-owned cwd")
    if type(spec.get("keep_sandbox", False)) is not bool:
        raise ValueError("RunSpec keep_sandbox must be a boolean")
    if not isinstance(spec.get("extra", {}), dict):
        raise ValueError("RunSpec extra must be a JSON object")
    if any(key in spec.get("extra", {}) for key in ("native_prefix", "native_home", "checkpoint_identity")):
        raise ValueError("Native restoration fields are worker-owned")


#: Which profile section a job kind takes its defaults from.
DEFAULTS_KEY = {"rollout": "run_defaults", "grade": "grade_defaults", "review": "review_defaults"}


def defaults_key(kind: str) -> str:
    """The profile section that jobs of ``kind`` take their defaults from. Raises ValueError if none."""
    try:
        return DEFAULTS_KEY[kind]
    except KeyError:
        raise ValueError("kind must be rollout, grade or review") from None


REVIEW_GUIDANCE = ("assistant-turn", "none", "user-hint")
#: How a review's ``points`` counts: at most that many, or exactly that many.
REVIEW_COUNT_MODES = ("adaptive", "fixed")
REVIEW_PARENT_FIELDS = frozenset({"name", "job_id", "attempt_id", "grade", "hint"})
#: The keys of a review's ``leak_judge``, each optional: Jev's ``threshold`` and ``model`` (defaults in
#: ``worker.jobs.review.leak.judge``), ``resamples``, and ``api_key_env`` for the TypeSafe key.
LEAK_JUDGE_FIELDS = frozenset({"threshold", "model", "resamples", "api_key_env"})


@dataclass(frozen=True)
class ReviewSpec:
    """A review job: hindsight over finished attempts of one task (``worker.jobs.review.job``).

    It reads ``parents``, each with its grade job's result, and picks branch points in them. Its
    ``guidance`` says what it writes at each point: ``candidates`` repair turns (``assistant-turn``),
    a hint (``user-hint``) or nothing (``none``). docs/RUNSTORE.md gives the role of each field."""

    problem: str
    model: str
    model_endpoint: str
    parents: list[dict[str, Any]]
    api_key_env: str | None = None
    guidance: str = "assistant-turn"
    points: int = 2
    points_per_attempt: bool = False
    #: ``adaptive``: at most ``points`` points, ``fixed``: exactly that many.
    count_mode: str = "adaptive"
    candidates: int = 4
    max_output_tokens: int = 16384
    prompt_chars: int = 400_000
    transcript_tokens: int = 100_000
    reasoning_effort: str | None = None
    temperature: float | None = 0.8
    max_attempts: int = 3
    round_no: int = 1
    timeout_s: float = 3600
    #: The longest one model request may take, in seconds. None: ``timeout_s``.
    request_timeout_s: float | None = None
    #: The most model requests that the review has in flight at once. The endpoint also
    #: serves the rollouts.
    max_concurrent_requests: int = 8
    #: Whether the endpoint takes schema-constrained JSON (``response_format: json_schema``).
    structured: bool = False
    #: The prompts file on the worker host, ``{"path": ..., "sha256": ...}``. None: the packaged file.
    prompts: dict[str, Any] | None = None
    #: The directory of the attempts on the worker host. The worker sets it.
    artifact_root: str | None = None
    #: With ``assistant-turn`` guidance, the check of each turn that the reviewer writes for hidden-test
    #: content, before the review takes the turn (``LEAK_JUDGE_FIELDS``). None: no check.
    leak_judge: dict[str, Any] | None = None

    def validate(self) -> dict[str, Any]:
        """The spec as a dict. Raises ValueError unless a worker can run it as a review."""
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("A review needs the task's problem statement")
        for name in ("model", "model_endpoint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"A review needs {name}")
        if self.guidance not in REVIEW_GUIDANCE:
            raise ValueError("Review guidance must be assistant-turn, none or user-hint")
        if self.count_mode not in REVIEW_COUNT_MODES:
            raise ValueError("Review count_mode must be adaptive or fixed")
        self._validate_settings()
        if self.leak_judge is not None:
            _validate_leak_judge(self.leak_judge)
        _validate_review_parents(self.parents)
        return asdict(self)

    def _validate_settings(self) -> None:
        for name in (
            "points",
            "candidates",
            "max_output_tokens",
            "prompt_chars",
            "transcript_tokens",
            "max_attempts",
            "round_no",
            "max_concurrent_requests",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Review {name} must be a positive integer")
        for name in ("points_per_attempt", "structured"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"Review {name} must be a boolean")
        if self.guidance != "assistant-turn" and self.candidates != 1:
            raise ValueError(f"A {self.guidance} review writes no repair turns; candidates must be 1")
        if self.temperature is not None and (
            type(self.temperature) not in (int, float) or not 0 <= self.temperature <= 2
        ):
            raise ValueError("Review temperature must be between 0 and 2")
        timeout = self.request_timeout_s
        if timeout is not None and (
            type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0
        ):
            raise ValueError("Review request_timeout_s must be finite and positive")
        if self.prompts is not None and (
            not isinstance(self.prompts, dict)
            or set(self.prompts) != {"path", "sha256"}
            or not all(_is_text(value) for value in self.prompts.values())
        ):
            raise ValueError("Review prompts names the path and the sha256 of a prompts file")


def _validate_leak_judge(judge: object) -> None:
    if not isinstance(judge, dict) or set(judge) - LEAK_JUDGE_FIELDS:
        raise ValueError("Review leak_judge names only threshold, model, resamples and api_key_env")
    if "threshold" in judge and (
        type(judge["threshold"]) not in (int, float) or not 0 < judge["threshold"] <= 1
    ):
        raise ValueError("Review leak_judge threshold must be in (0, 1]")
    if "model" in judge and (not isinstance(judge["model"], str) or not judge["model"].strip()):
        raise ValueError("Review leak_judge model must be a model name")
    if "resamples" in judge and (type(judge["resamples"]) is not int or judge["resamples"] < 0):
        raise ValueError("Review leak_judge resamples must be a non-negative integer")
    if "api_key_env" in judge and (
        not isinstance(judge["api_key_env"], str) or not judge["api_key_env"].isidentifier()
    ):
        raise ValueError("Review leak_judge api_key_env must name one environment variable")


def _validate_review_parents(parents: object) -> None:
    if not isinstance(parents, list) or not parents:
        raise ValueError("A review needs at least one finished attempt to read")
    names = set()
    for parent in parents:
        if not isinstance(parent, dict) or set(parent) - REVIEW_PARENT_FIELDS:
            raise ValueError("A review parent names name, job_id, attempt_id, grade and optionally hint")
        for key in ("name", "job_id", "attempt_id"):
            if not isinstance(parent.get(key), str) or not parent[key].strip():
                raise ValueError(f"A review parent needs {key}")
        if parent["name"] in names:
            raise ValueError("Review parents need distinct names")
        names.add(parent["name"])
        grade = parent.get("grade")
        if not isinstance(grade, dict) or type(grade.get("resolved")) is not bool:
            raise ValueError("A review parent needs its grade with a resolved verdict")
        if parent.get("hint") is not None and not isinstance(parent["hint"], str):
            raise ValueError("A review parent's hint must be text")


@dataclass(frozen=True)
class JobSpec:
    """A request to the Run Store: one ``rollout``, ``grade`` or ``review`` job and the profile that runs it.

    ``spec`` is the spec of the job's kind as a dict: a ``RunSpec``, a ``GradeSpec`` or a ``ReviewSpec``.
    """

    kind: str
    spec: dict[str, Any]
    profile: str = "default"
    context: dict[str, Any] = field(default_factory=dict)
    max_infra_retries: int = 2
    parent_point: str | None = None
    #: What a branch adds after the history that it restores (``Continuation``). A job with a
    #: ``parent_point`` needs one. Any other job has none.
    continuation: dict[str, Any] | None = None
    #: A branch's own rollout-contract limits (``BRANCH_BUDGET_FIELDS``). The worker applies them over
    #: the contract that the branch inherits. Optional, and only for a branch.
    branch_budget: dict[str, Any] | None = None
    #: The affinity key: the jobs of one key run on one host (``Store.claim``). None: the key of the job
    #: that this one names, else the job's own id (``Store.submit``). A grade from a snapshot needs one.
    affinity: str | None = None

    def validate(self) -> dict[str, Any]:
        """The request body to store.

        Raises ValueError unless a worker can run the job, and TypeError for a field that ``GradeSpec``
        or JSON does not take.
        """
        if not isinstance(self.spec, dict) or not isinstance(self.context, dict):
            raise ValueError("spec and context must be JSON objects")
        body = asdict(self)
        for name in ("continuation", "branch_budget", "affinity"):
            if body[name] is None:
                # Leave out an absent optional field. A job stored without these keys must get
                # the same hash when its request is submitted again.
                del body[name]
        canonical(body)
        if self.parent_point is not None or self.continuation is not None:
            if self.kind != "rollout" or self.parent_point is None:
                raise ValueError(
                    "A continuation belongs to a rollout branch, with the parent_point it continues"
                )
            Continuation.from_dict(self.continuation)
        if self.branch_budget is not None:
            if self.parent_point is None:
                raise ValueError("A branch budget belongs to a branch")
            validate_branch_budget(self.branch_budget)
        no_credentials(self.spec)
        if type(self.max_infra_retries) is not int or not 0 <= self.max_infra_retries <= 2:
            raise ValueError("max_infra_retries must be 0..2")
        if not isinstance(self.profile, str) or not self.profile:
            raise ValueError("profile must be nonempty")
        if self.affinity is not None and (not _is_text(self.affinity) or len(self.affinity) > 256):
            raise ValueError("affinity must be nonempty text of at most 256 characters")
        if self.kind == "rollout":
            _validate_run(self.spec)
        elif self.kind == "grade":
            GradeSpec(**self.spec).validate()
        elif self.kind == "review":
            try:
                ReviewSpec(**self.spec).validate()
            except TypeError as error:
                raise ValueError("Unknown ReviewSpec fields") from error
        else:
            raise ValueError("kind must be rollout, grade or review")
        timeout = self.spec.get("timeout_s", 3600)
        if type(timeout) not in (int, float) or not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("timeout_s must be finite and positive")
        return body

    @classmethod
    def from_dict(cls, body: dict[str, Any]) -> JobSpec:
        """The valid ``JobSpec`` in ``body``. Raises TypeError for an unknown field, ValueError otherwise."""
        result = cls(**body)
        result.validate()
        return result

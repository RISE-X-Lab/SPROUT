"""JSON contracts; leases and results deliberately do not belong in RunSpec."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import hashlib
import json
import math
from typing import Any

from sprout.harness_agents.mini_swe.continuation import Continuation
from sprout.harness_orchestrator.run import RunSpec

#: What a branch may change about the run it continues. What the actor is given
#: next is the branch's continuation, never a different prompt.
BRANCH_OVERRIDES = frozenset({"model", "timeout_s", "budget_usd"})
#: The rollout-contract limits a branch may set for itself. A branch inherits
#: its parent's contract, whose deadline is the parent group's and whose turn
#: limit restarts from zero; a branch made after that deadline, or one meant to
#: run only the turns its parent had left at the point, needs its own.
BRANCH_BUDGET_FIELDS = frozenset({"max_turns", "deadline_at"})


def validate_branch_budget(value: Any) -> dict:
    if not isinstance(value, dict) or not value or set(value) - BRANCH_BUDGET_FIELDS:
        raise ValueError("A branch budget sets max_turns and/or deadline_at")
    if "max_turns" in value and (type(value["max_turns"]) is not int or value["max_turns"] <= 0):
        raise ValueError("branch budget max_turns must be a positive integer")
    deadline = value.get("deadline_at")
    if deadline is not None and (type(deadline) not in (int, float) or not math.isfinite(deadline) or deadline <= 0):
        raise ValueError("branch budget deadline_at must be a finite positive timestamp")
    return dict(value)


def canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=False, allow_nan=False)


def digest(value: Any) -> str:
    return hashlib.sha256(canonical(value).encode()).hexdigest()


#: Mappings whose keys are data, not configuration: a grade's ``test_statuses``
#: is keyed by test ids, and a test may be named after what it checks
#: (``Test_AzureVMWithPasswordAuth``), which once kept a review from submitting.
DATA_MAPPINGS = frozenset({"test_statuses"})


def no_credentials(value: Any) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            if key in DATA_MAPPINGS:
                continue
            normalized = key.lower().replace("-", "_")
            secret = any(part in normalized for part in (
                "api_key", "auth_token", "authorization", "password", "bearer_token", "secret"))
            reference = isinstance(item, dict) and set(item) == {"$env"}
            if secret and item and not reference and not normalized.endswith("_env"):
                raise ValueError("Use a worker-side $env reference instead of inline credentials")
            no_credentials(item)
    elif isinstance(value, list):
        for item in value:
            no_credentials(item)


def validate_continuation(parent: dict, child: dict) -> None:
    parent_spec = {key: value for key, value in parent["spec"].items() if key not in BRANCH_OVERRIDES}
    child_spec = {key: value for key, value in child["spec"].items() if key not in BRANCH_OVERRIDES}
    if (parent["kind"] != "rollout" or child["kind"] != "rollout"
            or parent["profile"] != child["profile"] or parent_spec != child_spec):
        raise ValueError("Continuation must retain its source environment, slot and tool configuration")


@dataclass(frozen=True)
class GradeSpec:
    benchmark: str
    instance_id: str
    snapshot_id: str
    dataset_path: str
    dataset_sha256: str
    grader_revision: str
    backend: dict = field(default_factory=dict)
    resources: dict = field(default_factory=lambda: {"cpu": 2, "memory_mb": 12288})
    timeout_s: float = 3600
    verifier_network: str = "deny"
    #: The task's pristine image: the official graders apply the model's diff to
    #: a fresh copy of it, never to the agent's own filesystem.
    image: str | None = None
    parser_path: str | None = None


#: Which profile section a job kind takes its defaults from.
DEFAULTS_KEY = {"rollout": "run_defaults", "grade": "grade_defaults", "review": "review_defaults"}


def defaults_key(kind: str) -> str:
    try:
        return DEFAULTS_KEY[kind]
    except KeyError:
        raise ValueError("kind must be rollout, grade or review") from None


REVIEW_GUIDANCE = ("assistant-turn", "none")
REVIEW_PARENT_FIELDS = frozenset({"name", "job_id", "attempt_id", "grade", "hint"})
#: A review's ``leak_judge``: the Jev threshold and model (``methods.leak_judge``
#: defaults when absent), how often a flagged turn goes back to the reviewer,
#: and the worker-side variable holding the TypeSafe key.
LEAK_JUDGE_FIELDS = frozenset({"threshold", "model", "resamples", "api_key_env"})


@dataclass(frozen=True)
class ReviewSpec:
    """A review job: hindsight over finished attempts of one task (``runstore.review``).

    ``parents`` are the attempts to read, each with its grade job's result;
    ``points`` is how many branch points the reviewer picks across them --
    or, with ``points_per_attempt``, in each failed attempt on its own, so
    every failure is branched from -- and ``candidates`` how many independent
    repair turns it writes per point (``assistant-turn`` guidance; a ``none``
    review picks points only). ``max_concurrent_requests`` bounds how many of
    the review's model requests are in flight at once: its calls run side by
    side wherever no call needs another's answer, and with 1 they run one
    after another, in the order the stages list them.
    """

    problem: str
    model: str
    model_endpoint: str
    parents: list
    api_key_env: str | None = None
    guidance: str = "assistant-turn"
    points: int = 2
    points_per_attempt: bool = False
    candidates: int = 4
    max_output_tokens: int = 16384
    prompt_chars: int = 400_000
    transcript_tokens: int = 100_000
    reasoning_effort: str | None = None
    temperature: float | None = 0.8
    max_attempts: int = 3
    round_no: int = 1
    timeout_s: float = 3600
    #: How many of its model requests may be in flight at once; the endpoint
    #: serves the rollouts too.
    max_concurrent_requests: int = 8
    #: Where the attempts live on the worker host; the worker fills it in.
    artifact_root: str | None = None
    #: With assistant-turn guidance, check every turn the reviewer writes for
    #: hidden-test content before it is taken (``LEAK_JUDGE_FIELDS``); None: no check.
    leak_judge: dict | None = None

    def validate(self) -> dict:
        if not isinstance(self.problem, str) or not self.problem.strip():
            raise ValueError("A review needs the task's problem statement")
        for name in ("model", "model_endpoint"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"A review needs {name}")
        if self.guidance not in REVIEW_GUIDANCE:
            raise ValueError("Review guidance must be assistant-turn or none")
        for name in ("points", "candidates", "max_output_tokens", "prompt_chars", "transcript_tokens",
                     "max_attempts", "round_no", "max_concurrent_requests"):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"Review {name} must be a positive integer")
        if type(self.points_per_attempt) is not bool:
            raise ValueError("Review points_per_attempt must be a boolean")
        if self.guidance == "none" and self.candidates != 1:
            raise ValueError("A none review proposes points only; candidates must be 1")
        if self.temperature is not None and (type(self.temperature) not in (int, float)
                                             or not 0 <= self.temperature <= 2):
            raise ValueError("Review temperature must be between 0 and 2")
        if self.leak_judge is not None:
            judge = self.leak_judge
            if not isinstance(judge, dict) or set(judge) - LEAK_JUDGE_FIELDS:
                raise ValueError("Review leak_judge names only threshold, model, resamples and api_key_env")
            if "threshold" in judge and (type(judge["threshold"]) not in (int, float)
                                         or not 0 < judge["threshold"] <= 1):
                raise ValueError("Review leak_judge threshold must be in (0, 1]")
            if "model" in judge and (not isinstance(judge["model"], str) or not judge["model"].strip()):
                raise ValueError("Review leak_judge model must be a model name")
            if "resamples" in judge and (type(judge["resamples"]) is not int or judge["resamples"] < 0):
                raise ValueError("Review leak_judge resamples must be a non-negative integer")
            if "api_key_env" in judge and (not isinstance(judge["api_key_env"], str)
                                           or not judge["api_key_env"].isidentifier()):
                raise ValueError("Review leak_judge api_key_env must name one environment variable")
        if not isinstance(self.parents, list) or not self.parents:
            raise ValueError("A review needs at least one finished attempt to read")
        names = set()
        for parent in self.parents:
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
        return asdict(self)


@dataclass(frozen=True)
class JobSpec:
    kind: str
    spec: dict
    profile: str = "default"
    context: dict = field(default_factory=dict)
    max_infra_retries: int = 2
    parent_point: str | None = None
    profile_hash: str | None = None
    #: What a branch adds after the history it restores (``Continuation``); a
    #: job with a ``parent_point`` needs one, any other job has none.
    continuation: dict | None = None
    #: A branch's own rollout-contract limits (``BRANCH_BUDGET_FIELDS``), applied
    #: by the worker over the contract it inherits; optional, branches only.
    branch_budget: dict | None = None

    def validate(self) -> dict:
        if not isinstance(self.spec, dict) or not isinstance(self.context, dict):
            raise ValueError("spec and context must be JSON objects")
        body = asdict(self)
        for name in ("continuation", "branch_budget"):
            if body[name] is None:
                # Only a branch carries these, so other requests keep the body
                # (and the hash an idempotent resubmission is compared by) they had.
                del body[name]
        canonical(body)
        if self.parent_point is not None or self.continuation is not None:
            if self.kind != "rollout" or self.parent_point is None:
                raise ValueError("A continuation belongs to a rollout branch, with the parent_point it continues")
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
        if self.kind == "rollout":
            allowed = {item.name for item in fields(RunSpec)}
            if set(self.spec) - allowed:
                raise ValueError("Unknown RunSpec fields")
            if not isinstance(self.spec.get("prompt"), str) or not self.spec["prompt"].strip():
                raise ValueError("RunSpec requires a nonempty prompt")
            if self.spec.get("slot", "mini-swe-agent") != "mini-swe-agent":
                raise ValueError("The only queued-run slot is mini-swe-agent")
            for name in ("session", "journal_path", "run_id"):
                if self.spec.get(name) is not None:
                    raise ValueError(f"Queued runs cannot supply {name}")
            if self.spec.get("cwd", ".") != ".":
                raise ValueError("Queued runs use an isolated worker-owned cwd")
            if self.spec.get("keep_sandbox"):
                raise ValueError("Queued runs retain snapshots, not live sandboxes")
            if not isinstance(self.spec.get("extra", {}), dict):
                raise ValueError("RunSpec extra must be a JSON object")
            if any(key in self.spec.get("extra", {}) for key in (
                    "native_prefix", "native_home", "checkpoint_identity")):
                raise ValueError("Native restoration fields are worker-owned")
        elif self.kind == "grade":
            grade = GradeSpec(**self.spec)
            if grade.benchmark != "swe-rebench-v2":
                raise ValueError("Unsupported official grader")
            if grade.verifier_network not in {"allow", "deny"}:
                raise ValueError("verifier_network must be allow or deny")
            if not all((grade.instance_id, grade.snapshot_id, grade.dataset_path,
                        grade.dataset_sha256, grade.grader_revision)):
                raise ValueError("Grading requires frozen task, snapshot and grader references")
            if not grade.parser_path:
                raise ValueError("SWE-rebench grading requires a deployment parser_path")
            if not grade.image:
                raise ValueError("SWE-rebench grading requires the task image the diff is applied to")
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
    def from_dict(cls, body: dict) -> JobSpec:
        result = cls(**body)
        result.validate()
        return result

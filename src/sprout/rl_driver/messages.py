"""The Miles adapter; execution and grading stay in Run Store."""

import hashlib
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Any
from urllib.parse import urlsplit

from sprout.contract.sampling import validate as validate_sampling
from sprout.rl_driver.ledger import Conflict, canonical
from sprout.rl_driver.outcome import export_problem, grade_verdict, zero_reason
from sprout.runstore.messages import is_truncated_result


def internal_id(value: str) -> str:
    return "miles-" + hashlib.sha256(value.encode()).hexdigest()


def validate_endpoint(value: str) -> None:
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Model/session endpoint must be HTTP(S) without credentials/query/fragment")


#: How the adapter runs a search request's review job: the worker profile and
#: the model-call settings SPROUT's reviewer takes. What to search for -- how
#: many points, candidates and continuations -- is the request's.
REVIEW_CONFIG_FIELDS = frozenset(
    {
        "profile",
        "max_output_tokens",
        "prompt_chars",
        "transcript_tokens",
        "temperature",
        "reasoning_effort",
        "max_attempts",
        "max_concurrent_requests",
        "leak_judge",
    }
)


#: How long one grade may run before its sample is given up as missing
#: (``grade_expired``), and the part of that kept for starting its sandboxes
#: and reading the diff: the tests get the rest.
GRADE_WALL_SECONDS = 600.0
GRADE_PREPARATION_SECONDS = 120


class MessageAdapter:
    def __init__(self, driver, config):
        if not isinstance(config, dict) or set(config) - {
            "profile",
            "run_defaults",
            "tasks",
            "api_key_env",
            "max_samples",
            "image_resources",
            "review",
            "grade_wall_seconds",
        }:
            raise ValueError("Invalid Miles adapter config")
        defaults = config.get("run_defaults", {})
        if defaults.get("slot", "mini-swe-agent") != "mini-swe-agent":
            raise ValueError("The Miles adapter supports mini-swe-agent only")
        if "rollout_contract" in defaults.get("extra", {}):
            raise ValueError("rollout_contract is derived from each Miles request")
        review = config.get("review")
        if review is not None and (
            not isinstance(review, dict)
            or set(review) - REVIEW_CONFIG_FIELDS
            or not isinstance(review.get("profile"), str)
            or not review["profile"]
        ):
            raise ValueError(
                "The review config names a worker profile and the reviewer's model-call settings"
            )
        wall = config.get("grade_wall_seconds", GRADE_WALL_SECONDS)
        if type(wall) not in (int, float) or not math.isfinite(wall) or wall < 2 * GRADE_PREPARATION_SECONDS:
            raise ValueError(f"grade_wall_seconds must be at least {2 * GRADE_PREPARATION_SECONDS} s")
        self.driver, self.config = driver, deepcopy(config)
        self.grade_wall_seconds = float(wall)

    def submit(self, body):
        request = MessageRequest.from_dict(body)
        normalized = request.to_dict()
        identifier = internal_id(request.rollout_job_id)
        try:
            previous = self.driver.ledger.get(identifier)
        except KeyError:
            previous = None
        if previous is not None:
            if canonical(previous["request"]) != canonical(normalized):
                raise Conflict("rollout_job_id already names a different request")
        else:
            plan = self._plan(request)
            extra = {
                "message_request": normalized,
                "execution_deadline_at": plan["context"]["deadline_at"],
                "deadline_at": plan["context"]["deadline_at"] + request.finalization_timeout_seconds,
                "grade_wall_seconds": self.grade_wall_seconds,
            }
            if request.search is not None:
                # The roots' deadline, then the review and the branches, then finalization.
                extra["search"] = self._search_document(request, plan)
                extra["deadline_at"] += request.search.search_seconds(self.grade_wall_seconds)
            self.driver.submit(plan, source_request=normalized, extra_document=extra)
        result = self.get(request.rollout_job_id)
        return {key: result[key] for key in ("rollout_job_id", "status")}

    def _plan(self, request):
        validate_endpoint(request.model_endpoint)
        validate_sampling(request.sampling_params)
        if request.max_samples > self.config.get("max_samples", 1000):
            raise ValueError("Group exceeds deployment max_samples")
        defaults = self.config.get("run_defaults", {})
        model = request.model or defaults.get("model")
        if not isinstance(model, str) or not model:
            raise ValueError("Provide model or run_defaults.model")
        resources = self.config.get("image_resources")
        if (
            not isinstance(resources, dict)
            or set(resources) != {"cpu", "memory_mb"}
            or any(type(n) is not int or n <= 0 for n in resources.values())
        ):
            raise ValueError("Sprout must configure positive image_resources cpu/memory_mb")
        task = self.config.get("tasks", {}).get(request.task_id, {})
        if not task.get("grade"):
            raise ValueError(f"Configure Sprout tasks[{request.task_id!r}].grade for message rollout rewards")
        image = request.image
        if image.startswith("prime/primeintellect/"):
            # Prime's dataset documents this public-registry namespace rewrite.
            image = "docker.io/swerebenchv2/" + image.removeprefix("prime/primeintellect/")
        deadline = time.time() + request.budgets.max_wall_time_seconds
        prompt = request.prompt
        if isinstance(prompt, list):
            if (
                len(prompt) != 1
                or prompt[0].get("role") != "user"
                or set(prompt[0]) != {"role", "content"}
                or not isinstance(prompt[0]["content"], str)
            ):
                raise ValueError("Native task input must be text or one user message")
            prompt = prompt[0]["content"]
        samples = []
        # With a search, the roots take the first slots; the branches the review
        # proposes later take the rest, in the order they are made.
        roots = request.search.roots if request.search is not None else request.max_samples
        for slot in request.sample_slots[:roots]:
            spec = deepcopy(defaults)
            spec.update(
                prompt=prompt,
                model=model,
                slot=spec.get("slot", "mini-swe-agent"),
                sandbox_image=image,
                sandbox_resources=resources,
                timeout_s=min(spec.get("timeout_s", float("inf")), request.budgets.max_wall_time_seconds),
            )
            spec["extra"] = {
                **spec.get("extra", {}),
                "rollout_contract": {
                    "message_export": True,
                    "model_endpoint": request.model_endpoint,
                    "model": model,
                    "sampling_params": request.sampling_params,
                    "deadline_at": deadline,
                    "max_turns": request.max_turns,
                    "api_key_env": self.config.get("api_key_env"),
                },
            }
            samples.append(
                {
                    "sample_slot_id": internal_id(slot.sample_slot_id),
                    "run": {
                        "kind": "rollout",
                        "profile": self.config.get("profile", "mini"),
                        "spec": spec,
                        "max_infra_retries": 0,
                        "context": {
                            "task_id": request.task_id,
                            "image": request.image,
                            "sample_slot_id": slot.sample_slot_id,
                        },
                    },
                    # The grader applies the diff to a fresh copy of the same image. A
                    # grade that fails leaves its sample missing rather than running
                    # again, so it is never retried, and its tests fit inside its wall time.
                    "grade": {
                        **deepcopy(task["grade"]),
                        "max_infra_retries": 0,
                        "spec": {
                            **task["grade"]["spec"],
                            "image": image,
                            "timeout_s": min(
                                task["grade"]["spec"].get("timeout_s", math.inf),
                                self.grade_wall_seconds - GRADE_PREPARATION_SECONDS,
                            ),
                        },
                    },
                }
            )
        return {
            "rollout_job_id": internal_id(request.rollout_job_id),
            "prompt_group_id": internal_id(request.prompt_group_id),
            "context": {"deadline_at": deadline},
            "samples": samples,
        }

    def _search_document(self, request, plan):
        """What the driver needs to branch the failed roots once they are graded.

        One review job template per shape that reviews (``outcome.SHAPES``):
        ``full`` writes the request's repair turns at each point, and
        ``points_only`` picks the points alone (guidance ``none``); with more
        than one repair round, ``chain`` reviews a round's failed repairs for
        the next. Which one
        runs depends on how the roots did. A template lacks only ``parents``,
        filled in when the roots are known; both are validated here with a
        placeholder so a misconfigured review fails the request, not the step
        an hour later.
        """
        from sprout.runstore.specs import ReviewSpec

        review = self.config.get("review")
        if review is None:
            raise ValueError("A search request needs the adapter's review config (the review worker profile)")
        search = request.search
        root = plan["samples"][0]["run"]
        spec = {
            "problem": root["spec"]["prompt"],
            "model": root["spec"]["model"],
            "model_endpoint": request.model_endpoint,
            "api_key_env": self.config.get("api_key_env"),
            "guidance": "assistant-turn" if search.candidates else "none",
            "points": search.points,
            "points_per_attempt": True,
            "candidates": search.candidates or 1,
            "timeout_s": search.review_seconds,
            **{key: value for key, value in review.items() if key != "profile"},
        }
        specs = {"full": spec, "points_only": {**spec, "guidance": "none", "candidates": 1}}
        if search.candidates and search.repair_rounds > 1:
            # A later round reviews each failed repair on its own: one point in it, one turn.
            specs["chain"] = {**spec, "points": 1, "candidates": 1}
        for shape_spec in specs.values():
            ReviewSpec(
                **{
                    **shape_spec,
                    "parents": [
                        {
                            "name": "placeholder",
                            "job_id": "x",
                            "attempt_id": "x",
                            "grade": {"resolved": False},
                        }
                    ],
                }
            ).validate()
        return {
            "config": search.to_dict(),
            "state": "roots",
            "review": None,
            "points": [],
            "review_templates": {
                shape: {
                    "kind": "review",
                    "profile": review["profile"],
                    "max_infra_retries": 0,
                    "spec": shape_spec,
                }
                for shape, shape_spec in specs.items()
            },
            "parent_max_turns": request.max_turns,
            # Internal ids: the roots' slots, and the slots the branches may fill, in order.
            "root_slots": [internal_id(slot.sample_slot_id) for slot in request.sample_slots[: search.roots]],
            "free_slots": [internal_id(slot.sample_slot_id) for slot in request.sample_slots[search.roots :]],
        }

    def get(self, group_id):
        row = self.driver.ledger.get(internal_id(group_id))
        document = row["document"]
        request = MessageRequest.from_dict(document["message_request"])
        if request.rollout_job_id != group_id:
            raise KeyError(group_id)
        if "message_result" in document:
            return document["message_result"]
        result = {
            "rollout_job_id": group_id,
            "prompt_group_id": request.prompt_group_id,
            "max_samples": request.max_samples,
            "actual_samples": 0,
            "trajectories": [],
            "failed_samples": [],
            "search_branches": 0,
            "consumed_budget": {},
            "status": "queued" if document["status"] == "queued" else "running",
            "stop_reason": None,
        }
        if row["cancel_requested"]:
            result["status"] = "cancelled"
            return result
        if not document["ready"]:
            return result
        self._export(request, document, result)
        document["message_result"] = result
        self.driver.ledger.save(internal_id(group_id), document, terminal=True)
        return result

    def _export(self, request, document, result):
        errors = []
        for slot, sample in zip(
            request.sample_slots[: request.max_samples], document["samples"], strict=False
        ):
            try:
                problem = export_problem(sample)
                if problem is not None:
                    raise ValueError(problem)
                actor, grade = sample["actor"], sample["grade"]
                output = actor.get("result") or {}
                truncated = is_truncated_result(output)
                # None when the grade failed, ran out of time or gave no verdict:
                # the trajectory is returned, marked missing (zero_reason), with
                # the 0.0 the wire needs as its reward; the trainer leaves it out.
                verdict = grade_verdict(sample)
                origin = actor.get("origin") or output.get("training_origin") or {}
                parent = origin.get("job_id")
                search = sample.get("search")
                if search is None and document.get("search"):
                    # A root, labelled with the roots' outcome and counts, which the
                    # trainer checks against the roots it is sent.
                    search = {
                        "kind": "root",
                        "group": "root",
                        "outcome": document["search"].get("outcome"),
                        "root_counts": document["search"].get("root_counts"),
                    }
                result["trajectories"].append(
                    {
                        "sample_slot_id": slot.sample_slot_id,
                        "branch_id": actor["job_id"],
                        "parent_branch_id": parent,
                        "messages": output["training_messages"],
                        # Which trajectories were sampled under one condition: the
                        # trainer normalizes rewards within a group. Roots share the
                        # request's group; branches group by point and inserted turn.
                        "group": search["group"] if search else None,
                        "tools": output.get("training_tools", []),
                        "reward": 0.0 if verdict is None else float(verdict["resolved"]),
                        "status": "truncated" if truncated else "completed",
                        "stop_reason": output.get("stop_reason") if truncated else None,
                        "hints_removed": True,
                        "metadata": {
                            "job_id": actor["job_id"],
                            "attempt_id": actor["attempt_id"],
                            "source_image": request.image,
                            "graded_snapshot_id": grade.get("snapshot_id"),
                            # The share of the task's FAIL_TO_PASS tests that
                            # passed; the reward is whether every expected test did.
                            "progress": None if verdict is None else verdict.get("progress"),
                            "origin": origin,
                            "provenance": output["training_provenance"],
                            "search": {key: value for key, value in search.items() if key != "group"}
                            if search
                            else None,
                            "logprob_context": "hint_free_messages",
                            "zero_reason": zero_reason(sample),
                            # The files of the task's hidden test patch the graded
                            # diff changed as well; None when the grade cannot say.
                            "touched_test_paths": None
                            if verdict is None
                            else verdict.get("touched_test_paths"),
                        },
                    }
                )
                result["search_branches"] += int(parent is not None)
                for name, count in output.get("rollout_usage", {}).items():
                    result["consumed_budget"][name] = result["consumed_budget"].get(name, 0) + count
            except (ValueError, KeyError, TypeError) as error:
                errors.append(f"{slot.sample_slot_id}: {error}")
                # No trajectory: the trainer counts the slot as missing, never as a 0.
                result["failed_samples"].append({"sample_slot_id": slot.sample_slot_id, "reason": str(error)})
        result["actual_samples"] = len(result["trajectories"])
        result["status"] = (
            "failed"
            if result["actual_samples"] < request.minimum_returned_samples
            else "early_stopped"
            if errors
            else "completed"
        )
        result["stop_reason"] = "; ".join(errors) or None

    def execution(self, group_id):
        return self.driver.get(internal_id(group_id))

    def release(self, group_id):
        result = self.get(group_id)
        self.driver.release(internal_id(group_id))
        return {
            "rollout_job_id": group_id,
            "status": result["status"] if result["status"] not in {"queued", "running"} else "cancelled",
        }


def _required_string(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _nonnegative_int(value: Any, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _reject_unknown_keys(value: dict[str, Any], allowed: set[str], name: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"{name} contains unknown fields: {sorted(unknown)}")


@dataclass(frozen=True)
class SampleSlot:
    sample_slot_id: str
    sample_index: int

    @classmethod
    def from_dict(cls, value: Any) -> "SampleSlot":
        if not isinstance(value, dict):
            raise ValueError("sample_slots entries must be objects")
        _reject_unknown_keys(value, {"sample_slot_id", "sample_index"}, "sample slot")
        return cls(
            _required_string(value.get("sample_slot_id"), "sample_slot_id"),
            _nonnegative_int(value.get("sample_index"), "sample_index"),
        )


@dataclass(frozen=True)
class MessageBudget:
    max_wall_time_seconds: float

    @classmethod
    def from_dict(cls, value: object) -> "MessageBudget":
        if not isinstance(value, dict) or set(value) != {"max_wall_time_seconds"}:
            raise ValueError("budgets contains only max_wall_time_seconds; use per-trajectory max_turns")
        wall_time = value["max_wall_time_seconds"]
        if type(wall_time) not in (int, float) or not math.isfinite(wall_time) or wall_time <= 0:
            raise ValueError("budgets.max_wall_time_seconds must be finite and positive")
        return cls(float(wall_time))


@dataclass(frozen=True)
class SearchSpec:
    """Hindsight branches after the roots: what a search request asks for.

    ``roots`` scratch rollouts take the first slots. How they came out once
    graded (``outcome.search_outcome``) decides what follows
    (``outcome.SHAPES``). When every root was graded and none resolved (FF), a
    review of each failed one on its own picks up to ``points`` branch points
    in it and, per point, writes ``candidates`` repair turns; from each point
    run ``student_continuations`` branches with nothing added and
    ``candidate_continuations`` branches after each repair. When the roots
    split (TF), or none resolved and one is missing (FF_partial), the review
    picks the points alone, and ``tf_student_continuations`` branches with
    nothing added run from each; with 0, neither the review nor a branch
    runs. When every graded root resolved, or none was graded, nothing more
    runs. Under FF the repairs come in up to ``repair_rounds`` rounds: after
    round r, a point where no student and no repair has resolved is reviewed
    again, over its failed round-r repairs, and one more repair turn is
    written at a point in that repair's own history (a chain, each round's
    turn inserted after the last); more than one round takes one candidate
    and one continuation per round. The branches fill the remaining slots, so
    the request needs ``roots + branches()`` of them. ``review_seconds`` and
    ``wall_time_seconds`` bound each round's review and branch phase; the
    roots keep ``budgets.max_wall_time_seconds``.
    """

    roots: int
    points: int
    candidates: int
    student_continuations: int
    candidate_continuations: int
    tf_student_continuations: int
    repair_rounds: int
    review_seconds: float
    wall_time_seconds: float

    @classmethod
    def from_dict(cls, value: object) -> "SearchSpec":
        if not isinstance(value, dict) or set(value) != {f.name for f in fields(cls)}:
            raise ValueError(
                "search names roots, points, candidates, student_continuations, "
                "candidate_continuations, tf_student_continuations, repair_rounds, "
                "review_seconds and wall_time_seconds"
            )
        for key in ("roots", "points", "student_continuations", "repair_rounds"):
            if _nonnegative_int(value[key], f"search.{key}") < 1:
                raise ValueError(f"search.{key} must be positive")
        for key in ("candidates", "candidate_continuations", "tf_student_continuations"):
            _nonnegative_int(value[key], f"search.{key}")
        if (value["candidates"] == 0) != (value["candidate_continuations"] == 0):
            raise ValueError(
                "search.candidates and search.candidate_continuations are both zero or both positive"
            )
        if value["repair_rounds"] > 1 and (value["candidates"], value["candidate_continuations"]) != (1, 1):
            raise ValueError(
                "more than one search.repair_rounds takes one candidate and one candidate continuation"
            )
        for key in ("review_seconds", "wall_time_seconds"):
            if type(value[key]) not in (int, float) or not math.isfinite(value[key]) or value[key] <= 0:
                raise ValueError(f"search.{key} must be finite and positive")
        return cls(
            **{
                **value,
                "review_seconds": float(value["review_seconds"]),
                "wall_time_seconds": float(value["wall_time_seconds"]),
            }
        )

    def branches(self) -> int:
        """The most the search may make, ``points`` in each failed root, whatever the roots came to.

        FF fails every root and makes ``student_continuations + candidates *
        candidate_continuations * repair_rounds`` branches at a point; TF and
        FF_partial fail at most ``roots - 1`` and make
        ``tf_student_continuations``. Miles allocates this many slots by the
        same formula.
        """
        return self.points * max(
            self.roots
            * (
                self.student_continuations
                + self.candidates * self.candidate_continuations * self.repair_rounds
            ),
            (self.roots - 1) * self.tf_student_continuations,
        )

    def search_seconds(self, grade_seconds: float) -> float:
        """How long the search may run after its roots: each round's review and branches,
        and the grades of every round but the last (the last is finalization's)."""
        rounds = self.repair_rounds if self.candidates else 1
        return rounds * (self.review_seconds + self.wall_time_seconds) + (rounds - 1) * grade_seconds

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class MessageRequest:
    rollout_job_id: str
    rollout_id: int
    prompt_group_id: str
    task_id: str
    image: str
    prompt: str | list
    model_endpoint: str
    sample_slots: tuple
    max_samples: int
    minimum_returned_samples: int
    max_turns: int
    sampling_params: dict
    budgets: MessageBudget
    model: str | None = None
    finalization_timeout_seconds: float = 1800.0
    search: SearchSpec | None = None

    @classmethod
    def from_dict(cls, body):
        if not isinstance(body, dict) or set(body) - {f.name for f in fields(cls)}:
            raise ValueError("Unknown message request fields")
        value = deepcopy(body)
        for key in ("rollout_job_id", "prompt_group_id", "task_id", "image", "model_endpoint"):
            _required_string(value.get(key), key)
        _nonnegative_int(value.get("rollout_id"), "rollout_id")
        if value.get("model") is not None:
            _required_string(value["model"], "model")
        prompt = value.get("prompt")
        if not isinstance(prompt, (str, list)) or not prompt:
            raise ValueError("prompt must be nonempty text or messages")
        slots = value.get("sample_slots")
        if not isinstance(slots, list) or not slots:
            raise ValueError("sample_slots must be nonempty")
        value["sample_slots"] = tuple(SampleSlot.from_dict(slot) for slot in slots)
        if any(
            len({getattr(s, key) for s in value["sample_slots"]}) != len(slots)
            for key in ("sample_slot_id", "sample_index")
        ):
            raise ValueError("Sample slots must have unique ids and indices")
        for key in ("max_samples", "minimum_returned_samples", "max_turns"):
            if _nonnegative_int(value.get(key), key) < 1:
                raise ValueError(f"{key} must be positive")
        if not value["minimum_returned_samples"] <= value["max_samples"] <= len(slots):
            raise ValueError("Invalid sample count bounds")
        value.setdefault("sampling_params", {})
        if not isinstance(value["sampling_params"], dict):
            raise ValueError("sampling_params must be an object")
        value["budgets"] = MessageBudget.from_dict(value.get("budgets"))
        finalization = value.get("finalization_timeout_seconds", 1800.0)
        if type(finalization) not in (int, float) or not math.isfinite(finalization) or finalization <= 0:
            raise ValueError("finalization_timeout_seconds must be finite and positive")
        value["finalization_timeout_seconds"] = float(finalization)
        if value.get("search") is not None:
            search = SearchSpec.from_dict(value["search"])
            if search.roots > value["max_samples"]:
                raise ValueError("search.roots exceeds max_samples")
            if value["minimum_returned_samples"] > search.roots:
                raise ValueError(
                    "minimum_returned_samples may not exceed search.roots: only the roots are guaranteed"
                )
            if search.roots + search.branches() > value["max_samples"]:
                raise ValueError("max_samples must cover the roots and every branch the search may make")
            value["search"] = search
        else:
            value.pop("search", None)
        return cls(**value)

    def to_dict(self):
        value = asdict(self)
        value["sample_slots"] = list(value["sample_slots"])
        if value["search"] is None:
            del value["search"]
        return value

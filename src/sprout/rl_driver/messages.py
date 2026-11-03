"""The Miles adapter: a Miles rollout request as an execution group, and the group's result as Miles's.

``MessageAdapter`` checks each request (``MessageRequest``), submits its group through the
driver, and exports the trajectories and the rewards once the group is ready. A group that
failed after its runs again exports only its error. The Run Store's workers run and grade the
samples. ``docs/RL_DRIVER.md`` describes the protocol.
"""

import hashlib
import math
import time
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Any
from urllib.parse import urlsplit

from sprout.contract.messages import is_truncated_result
from sprout.contract.sampling import validate as validate_sampling
from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Conflict
from sprout.runstore.specs import canonical
from sprout.search.outcome import export_problem, grade_verdict, zero_reason
from sprout.search.spec import SearchSpec, search_document


def internal_id(value: str) -> str:
    """The driver's id for the Miles id ``value``: the same on every call, and a valid group or sample id."""
    return "miles-" + hashlib.sha256(value.encode()).hexdigest()


def validate_endpoint(value: str) -> None:
    """Raise ValueError unless ``value`` is an HTTP(S) URL with a host and no userinfo, query or fragment."""
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


#: The fields of the adapter's ``review`` config: the review job's worker profile and the reviewer's
#: model-call settings. The request sets how many points, candidates and continuations a search makes.
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


#: The default time that one grade may run before its sample is missing (``grade_expired``).
GRADE_WALL_SECONDS = 600.0
#: The part of a grade's wall time kept for starting its sandboxes and reading the diff. The tests get
#: the rest.
GRADE_PREPARATION_SECONDS = 120


def _check_run_defaults(defaults: object) -> None:
    """Raise ValueError unless ``defaults``, the ``run_defaults`` of a Miles section, can go into a rollout.

    Its ``slot`` is ``mini-swe-agent`` or ``opencollab-base``, and its ``extra`` is an object without
    ``rollout_contract``. Either key can be absent."""
    if not isinstance(defaults, dict):
        raise ValueError("run_defaults must be a JSON object")
    if defaults.get("slot", "mini-swe-agent") not in {"mini-swe-agent", "opencollab-base"}:
        raise ValueError("The Miles adapter requires mini-swe-agent or opencollab-base")
    extra = defaults.get("extra", {})
    if not isinstance(extra, dict):
        raise ValueError("run_defaults.extra must be a JSON object")
    if "rollout_contract" in extra:
        raise ValueError("rollout_contract is derived from each Miles request")


class MessageAdapter:
    """The Miles routes of one driver, set up by the driver config's ``miles`` section.

    The constructor raises ValueError for an invalid section. Each method takes the Miles
    ``rollout_job_id`` of a group. ``docs/RL_DRIVER.md`` describes the section and the routes.
    """

    def __init__(self, driver: Driver, config: object) -> None:
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
        _check_run_defaults(config.get("run_defaults", {}))
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

    def submit(self, body: object) -> dict[str, Any]:
        """Submit the Miles request ``body`` once, and return its ``rollout_job_id`` and ``status``.

        A repeated identical request returns the same group. Raises ValueError for an invalid
        request, and ``Conflict`` if its ``rollout_job_id`` names a different request.
        """
        request = MessageRequest.from_dict(body)
        normalized = request.to_dict()
        identifier = internal_id(request.rollout_job_id)
        try:
            previous: dict[str, Any] | None = self.driver.ledger.get(identifier)
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
                # The group deadline covers the roots, then the review and the branches, then finalization.
                extra["search"] = self._search_document(request=request, search=request.search, plan=plan)
                extra["deadline_at"] += request.search.search_seconds(self.grade_wall_seconds)
            self.driver.submit(plan, source_request=normalized, extra_document=extra)
        result = self.get(request.rollout_job_id)
        return {key: result[key] for key in ("rollout_job_id", "status")}

    def _plan(self, request: "MessageRequest") -> dict[str, Any]:
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
        samples: list[dict[str, Any]] = []
        # With a search, the roots take the first slots. The branches that the review proposes
        # later take the rest, in the order in which they are made.
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
                    # The grade applies the diff in a fresh sandbox of the same image. A failed grade leaves
                    # its sample missing and is not retried. Its tests end within the grade's wall time.
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

    def _search_document(
        self, *, request: "MessageRequest", search: SearchSpec, plan: dict[str, Any]
    ) -> dict[str, Any]:
        """The group's ``search`` (``search_document``): what the driver needs to branch the failed roots.

        The review reads the roots' prompt and model, the request's endpoint and the adapter's reviewer
        settings. A misconfigured review fails the request here, not the step an hour later.
        """
        review = self.config.get("review")
        if review is None:
            raise ValueError("A search request needs the adapter's review config (the review worker profile)")
        root = plan["samples"][0]["run"]
        return search_document(
            config=search,
            review={
                "problem": root["spec"]["prompt"],
                "model": root["spec"]["model"],
                "model_endpoint": request.model_endpoint,
                "api_key_env": self.config.get("api_key_env"),
                **{key: value for key, value in review.items() if key != "profile"},
            },
            review_profile=review["profile"],
            parent_max_turns=request.max_turns,
            # Internal ids: the roots' slots, and the slots the branches may fill, in order.
            root_slots=[internal_id(slot.sample_slot_id) for slot in request.sample_slots[: search.roots]],
            free_slots=[internal_id(slot.sample_slot_id) for slot in request.sample_slots[search.roots :]],
        )

    def get(self, group_id: str) -> dict[str, Any]:
        """The Miles result of the group ``group_id``. Raises KeyError for an unknown group.

        The result has no trajectories before the group is ready, after a cancellation request, or when the
        group failed after its runs again. The ``stop_reason`` of that failed group is its ``error``. The
        first call on a ready group exports the result and marks the group terminal. Later calls return it."""
        row = self.driver.ledger.get(internal_id(group_id))
        document = row["document"]
        request = MessageRequest.from_dict(document["message_request"])
        if request.rollout_job_id != group_id:
            raise KeyError(group_id)
        if "message_result" in document:
            stored: dict[str, Any] = document["message_result"]
            return stored
        result: dict[str, Any] = {
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
        if document.get("error") is not None:
            # A host was lost in each run (rerun.run_again). The samples that settled are only a part of
            # the group, so Miles gets none of them, and decides what to do with the group (ADR-0007).
            result.update(status="failed", stop_reason=document["error"])
        else:
            self._export(request, document, result)
        document["message_result"] = result
        self.driver.ledger.save(internal_id(group_id), document, terminal=True)
        return result

    def _export(self, request: "MessageRequest", document: dict[str, Any], result: dict[str, Any]) -> None:
        errors: list[str] = []
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
                # None if the grade failed, expired or gave no verdict. The trajectory is returned, marked
                # missing (zero_reason), with the 0.0 reward that the wire needs. The trainer leaves it out.
                verdict = grade_verdict(sample)
                origin = actor.get("origin") or output.get("training_origin") or {}
                parent = origin.get("job_id")
                search = sample.get("search")
                if search is None and document.get("search"):
                    # A root, labelled with the roots' outcome and counts. The trainer checks them
                    # against the roots that it receives.
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
                        # The trainer normalizes rewards within a group. Roots share the request's
                        # group, and branches group by their point and inserted turn.
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
                            # The share of the task's FAIL_TO_PASS tests that passed. The reward
                            # says whether every expected test passed.
                            "progress": None if verdict is None else verdict.get("progress"),
                            "origin": origin,
                            "provenance": output["training_provenance"],
                            "search": {key: value for key, value in search.items() if key != "group"}
                            if search
                            else None,
                            "logprob_context": "hint_free_messages",
                            "zero_reason": zero_reason(sample),
                            # The files of the task's hidden test patch that the graded diff changed
                            # too. None if the grade cannot say.
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

    def execution(self, group_id: str) -> dict[str, Any]:
        """The execution view (``Driver.get``) of the group behind the Miles group ``group_id``."""
        return self.driver.get(internal_id(group_id))

    def release(self, group_id: str) -> dict[str, Any]:
        """Release the Miles group ``group_id`` (``Driver.release``), and return its id and status.

        The status of a group that is still queued or running is ``cancelled``.
        """
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
    """One sample slot of a Miles request."""

    sample_slot_id: str
    sample_index: int

    @classmethod
    def from_dict(cls, value: object) -> "SampleSlot":
        if not isinstance(value, dict):
            raise ValueError("sample_slots entries must be objects")
        _reject_unknown_keys(value, {"sample_slot_id", "sample_index"}, "sample slot")
        return cls(
            _required_string(value.get("sample_slot_id"), "sample_slot_id"),
            _nonnegative_int(value.get("sample_index"), "sample_index"),
        )


@dataclass(frozen=True)
class MessageBudget:
    """The budget of a Miles request: the wall time of its execution phase, counted from submission."""

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
class MessageRequest:
    """A checked Miles rollout request. ``docs/RL_DRIVER.md`` (The Miles protocol) gives the fields."""

    rollout_job_id: str
    rollout_id: int
    prompt_group_id: str
    task_id: str
    image: str
    #: Text, or a list of chat messages as JSON objects.
    prompt: str | list[Any]
    model_endpoint: str
    sample_slots: tuple[SampleSlot, ...]
    max_samples: int
    minimum_returned_samples: int
    max_turns: int
    sampling_params: dict[str, Any]
    budgets: MessageBudget
    model: str | None = None
    finalization_timeout_seconds: float = 1800.0
    search: SearchSpec | None = None

    @classmethod
    def from_dict(cls, body: object) -> "MessageRequest":
        """The request in the JSON object ``body``. Raises ValueError for an invalid request."""
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

    def to_dict(self) -> dict[str, Any]:
        """The request as the JSON object that ``from_dict`` reads. Without a search, it has no ``search``."""
        value = asdict(self)
        value["sample_slots"] = list(value["sample_slots"])
        if value["search"] is None:
            del value["search"]
        return value

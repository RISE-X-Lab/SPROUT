"""What a search asks for, and the search document a group starts it from.

``SearchSpec`` is the shape of a search: how many roots, how many points in
each failed root, and how many branches run from each point in how many
rounds. ``search_document`` is the state of a group's search before its roots
finish. It holds one review job template for each review shape, so a
misconfigured review fails when the group is made, not when the roots finish.
``search_again`` gives that state back for a group that runs again from its
roots. docs/RL_DRIVER.md describes the search.
"""

import math
from copy import deepcopy
from dataclasses import asdict, dataclass, fields
from typing import Any

from sprout.runstore.specs import ReviewSpec


def _nonnegative_int(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


@dataclass(frozen=True)
class SearchSpec:
    """Hindsight branches after the roots: what a search asks for.

    How the ``roots`` came out (``outcome.search_outcome``) decides the review and the branches that
    follow (``outcome.SHAPES``). docs/RL_DRIVER.md gives the role of each field.
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
        """A checked ``SearchSpec`` from the ``search`` object of a request.

        Raises ValueError for a missing, unknown or invalid field.
        """
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
        """The most branches the search may make: ``points`` in each failed root, whatever the roots did.

        FF makes ``student_continuations + candidates * candidate_continuations * repair_rounds`` at a point
        in each of ``roots``. TF and FF_partial make ``tf_student_continuations`` in at most ``roots - 1``.
        """
        # Miles allocates a search request's slots by the same formula (docs/RL_DRIVER.md).
        return self.points * max(
            self.roots
            * (
                self.student_continuations
                + self.candidates * self.candidate_continuations * self.repair_rounds
            ),
            (self.roots - 1) * self.tf_student_continuations,
        )

    def search_seconds(self, grade_seconds: float) -> float:
        """How long in seconds the search may run after its roots: each round's review and branches,
        and the grades of every round but the last. The last round's grades belong to finalization."""
        rounds = self.repair_rounds if self.candidates else 1
        return rounds * (self.review_seconds + self.wall_time_seconds) + (rounds - 1) * grade_seconds

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def search_document(
    *,
    config: SearchSpec,
    review: dict[str, Any],
    review_profile: str,
    parent_max_turns: int,
    root_slots: list[str],
    free_slots: list[str],
) -> dict[str, Any]:
    """A group's ``search`` before its roots finish, with a review job template for each review shape.

    ``review`` holds the ``ReviewSpec`` fields that the search does not set. ``root_slots`` name the roots'
    samples, and branches fill ``free_slots`` in order. Raises ValueError or TypeError for an invalid review.
    """
    return _unstarted(
        config=config.to_dict(),
        review_templates=_review_templates(config=config, review=review, profile=review_profile),
        parent_max_turns=parent_max_turns,
        root_slots=list(root_slots),
        free_slots=list(free_slots),
    )


def search_again(search: dict[str, Any], *, branch_slots: list[str]) -> dict[str, Any]:
    """The ``search_document`` that ``search`` started as, for a group that runs again from its roots.

    ``branch_slots`` are the slots of the search's branches, in the order the search made them.
    """
    # A branch takes the first free slot, so the branches' slots came first among the free slots.
    return _unstarted(
        config=deepcopy(search["config"]),
        review_templates=deepcopy(search["review_templates"]),
        parent_max_turns=search["parent_max_turns"],
        root_slots=list(search["root_slots"]),
        free_slots=[*branch_slots, *search["free_slots"]],
    )


def _unstarted(
    *,
    config: dict[str, Any],
    review_templates: dict[str, dict[str, Any]],
    parent_max_turns: int,
    root_slots: list[str],
    free_slots: list[str],
) -> dict[str, Any]:
    return {
        "config": config,
        "state": "roots",
        "review": None,
        "points": [],
        "review_templates": review_templates,
        "parent_max_turns": parent_max_turns,
        "root_slots": root_slots,
        "free_slots": free_slots,
    }


def _review_templates(
    *, config: SearchSpec, review: dict[str, Any], profile: str
) -> dict[str, dict[str, Any]]:
    """One review job template for each shape that reviews, each lacking only its ``parents``.

    ``full`` writes the repair turns at each point, ``points_only`` picks the points alone (guidance
    ``none``), and ``chain``, made only for more than one repair round, reviews a round's failed repairs.
    """
    spec = {
        **review,
        "guidance": "assistant-turn" if config.candidates else "none",
        "points": config.points,
        "points_per_attempt": True,
        "candidates": config.candidates or 1,
        "timeout_s": config.review_seconds,
    }
    specs = {"full": spec, "points_only": {**spec, "guidance": "none", "candidates": 1}}
    if config.candidates and config.repair_rounds > 1:
        # A later round reviews each failed repair on its own: one point in it, one turn.
        specs["chain"] = {**spec, "points": 1, "candidates": 1}
    # A template gets its parents once the roots finish. A placeholder lets it be validated now.
    placeholder = {"name": "placeholder", "job_id": "x", "attempt_id": "x", "grade": {"resolved": False}}
    for shape_spec in specs.values():
        ReviewSpec(**{**shape_spec, "parents": [placeholder]}).validate()
    return {
        shape: {"kind": "review", "profile": profile, "max_infra_retries": 0, "spec": shape_spec}
        for shape, shape_spec in specs.items()
    }

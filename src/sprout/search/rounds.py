"""A group's search: label the roots, review the failed ones, branch where the review points, chain repairs.

The search document (``spec.search_document``) moves through the states
``roots``, ``reviewing``, ``branching``, ``chaining`` and ``done``. A review is
a Run Store job like any stage, submitted as the sample ``REVIEW_SLOT``, or as
``chain_review_slot(round)`` for a later repair round. Branches fill the free
slots in the order they are planned. A review that fails or proposes nothing
ends the search with what it has, and ``search.error`` (``search.chain_error``
for a later round) says why. docs/RL_DRIVER.md describes the search.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Iterator
from copy import deepcopy
from typing import Any

import httpx

from sprout.runstore.specs import JobSpec
from sprout.search.client import RunStoreClient
from sprout.search.group import initial_document, validate_request
from sprout.search.outcome import (
    SHAPES,
    done_reason,
    export_problem,
    grade_verdict,
    root_verdict,
    search_outcome,
    search_shape,
)
from sprout.search.stages import DONE, Stages

#: What the review is given of each failed root's grade. The last three describe the task's hidden
#: tests, and only the leak checks read them (docs/RL_DRIVER.md).
REVIEW_GRADE_KEYS = (
    "resolved",
    "progress",
    "test_statuses",
    "model_patch",
    "log",
    "reason",
    "error_output",
    "fail_to_pass",
    "test_patch_paths",
    "test_patch",
)

#: How a repair writes the reviewer's turn into its history: as the turn's
#: reasoning (``continuation.render_reasoning``), the form the policy samples.
REPAIR_RENDERING = "reasoning"

#: The slot that gives a search's review, phase ``review``, its idempotency key. Round r > 1 of
#: the repairs is reviewed as ``chain_review_slot(r)``.
REVIEW_SLOT = "search"


def chain_review_slot(round_no: int) -> str:
    """The slot that round ``round_no`` (2 or more) of the repairs is reviewed as."""
    return f"{REVIEW_SLOT}-r{round_no}"


class Search:
    """Advances the search of a group document, one call at a time."""

    def __init__(
        self, *, client: RunStoreClient, stages: Stages, context_key: str, stopping: threading.Event
    ) -> None:
        self._client = client
        self._stages = stages
        self._context_key = context_key
        self._stopping = stopping

    def advance(self, document: dict[str, Any]) -> None:
        """Move the document's search as far as its settled stages allow.

        Raises ``httpx.HTTPError``, ValueError or KeyError when a Run Store call fails or a review's plan
        is malformed. The search keeps its state, so the next call tries the same step again.
        """
        search = document["search"]
        if search["state"] == "done":
            return
        samples = document["samples"]
        roots = [sample for sample in samples if sample["sample_slot_id"] in search["root_slots"]]
        branches = [sample for sample in samples if sample["sample_slot_id"] not in search["root_slots"]]
        if search["state"] == "roots":
            if not _settled(roots):
                return
            self._review_roots(document, roots)
        if search["state"] == "reviewing":
            stage = search["review"]
            self._stages.advance(document, {"sample_slot_id": REVIEW_SLOT, "review": stage}, "review")
            if stage["state"] == "succeeded":
                self._plan_branches(document)
                return  # the branches settle on a later tick
            elif stage["state"] in DONE:
                search.update(state="done", error=stage.get("error") or f"review {stage['state']}")
                self._stages.save(document)
        if search["state"] == "branching" and _settled(branches):
            if not self._next_round(document):
                search["state"] = "done"
            self._stages.save(document)
        if search["state"] == "chaining":
            stage = search["chain_reviews"][-1]
            self._stages.advance(document, {"sample_slot_id": stage["slot"], "review": stage}, "review")
            if stage["state"] == "succeeded":
                self._plan_chain(document, stage)
            elif stage["state"] in DONE:
                search.update(state="done", chain_error=stage.get("error") or f"review {stage['state']}")
                self._stages.save(document)

    def _review_roots(self, document: dict[str, Any], roots: list[dict[str, Any]]) -> None:
        """Label the settled roots, then plan the review that their shape asks for, or end the search.

        The counts (planned, graded, resolved) give the label, and the label the
        shape. The search document keeps all three.
        """
        search = document["search"]
        verdicts = [root_verdict(sample) for sample in roots]
        counts = {
            "planned": search["config"]["roots"],
            "graded": sum(verdict is not None for verdict in verdicts),
            "resolved": sum(verdict is True for verdict in verdicts),
        }
        outcome = search_outcome(**counts)
        # A document without review_templates predates outcome-keyed review and has no
        # tf_student_continuations. Its search ends after the shape check.
        legacy = "review_templates" not in search
        shape = SHAPES[outcome] if legacy else search_shape(outcome, search["config"])
        search.update(root_counts=counts, outcome=outcome, shape=shape)
        if self._stages.cancelled(document) or shape == "none":
            search.update(state="done", error=done_reason(outcome, counts) if shape == "none" else None)
            self._stages.save(document)
            return
        if legacy:
            search.update(state="done", error="search document predates outcome-keyed review")
            self._stages.save(document)
            return
        failed = [sample for sample, verdict in zip(roots, verdicts, strict=True) if verdict is False]
        template = deepcopy(search["review_templates"][shape])
        template["spec"]["parents"] = _review_parents(failed)
        template["context"] = {
            self._context_key: {
                "rollout_job_id": document["rollout_job_id"],
                "phase": "review",
                "outcome": outcome,
            }
        }
        search["review"] = _review_stage(template)
        search["state"] = "reviewing"
        self._stages.save(document)

    def _plan_branches(self, document: dict[str, Any]) -> None:
        """Turn the review's points into branch samples, each in the next free slot.

        Under ``full``, each point gets ``student_continuations`` students and ``candidate_continuations``
        repairs of each repair turn. Under ``points_only``, it gets ``tf_student_continuations`` students.
        """
        search = document["search"]
        config = search["config"]
        # A document reviewed before shapes existed has none, and was reviewed in full.
        shape = search.get("shape", "full")
        students = config["student_continuations"] if shape == "full" else config["tf_student_continuations"]
        plan = search["review"].get("result") or {}
        free = list(search["free_slots"])
        roots = {
            sample["actor"]["job_id"]: sample
            for sample in document["samples"]
            if sample["sample_slot_id"] in search["root_slots"]
        }
        planned, skipped, points = [], [], []
        for index, point in enumerate(plan.get("points") or [], 1):
            root = roots.get(point["job_id"])
            recovery = None if root is None else self._recovery(point)
            if root is None or recovery is None:
                reason = "no available recovery point at that step" if root else "not a root of this group"
                skipped.append({"point": index, "step": point.get("step"), "reason": reason})
                continue
            budget = _branch_budget(
                parent_max_turns=search["parent_max_turns"],
                recovery=recovery,
                wall_time_seconds=config["wall_time_seconds"],
            )
            about = _point_about(index=index, point=point, recovery=recovery, search=search)
            points.append(
                {
                    **about,
                    "snapshot_id": recovery["snapshot_id"],
                    "why": point.get("why"),
                    "candidates": len(point.get("candidates") or []),
                }
            )
            wanted = _point_branches(
                point=point,
                about=about,
                students=students,
                repairs=config["candidate_continuations"] if shape == "full" else 0,
            )
            for continuation, annotation in wanted:
                if not free:
                    skipped.append({"point": index, "reason": "no free slot"})
                    continue
                planned.append(
                    {
                        "sample_slot_id": free.pop(0),
                        "branch": {
                            "job_id": point["job_id"],
                            "point_id": recovery["id"],
                            "continuation": continuation,
                            "budget": budget,
                        },
                        "grade": deepcopy(root["grade"]["template"]),
                        "search": annotation,
                    }
                )
        # Set after the loop. Then a failed read in the loop leaves the search as it was, and the next
        # call plans every point again.
        search.update(points=points, skipped=skipped)
        if not self._add_branches(document, planned, free):
            search["error"] = "the review proposed no branch that could be made"

    def _add_branches(self, document: dict[str, Any], planned: list[dict[str, Any]], free: list[str]) -> bool:
        """Add the planned branch samples to the document, dispatch them, and return whether there are any."""
        search = document["search"]
        if planned:
            body = validate_request(
                {
                    "rollout_job_id": document["rollout_job_id"],
                    "prompt_group_id": document["prompt_group_id"],
                    "samples": [{k: v for k, v in sample.items() if k != "search"} for sample in planned],
                }
            )
            key = document["affinity"]
            built = initial_document(body, context_key=self._context_key, affinity=key)["samples"]
            for sample, annotated in zip(built, planned, strict=True):
                sample["search"] = annotated["search"]
            document["samples"].extend(built)
        search.update(free_slots=free, state="branching" if planned else "done")
        self._stages.save(document)
        for sample in document["samples"][-len(planned) :] if planned else []:
            if self._stopping.is_set():
                break
            try:
                self._stages.advance(document, sample, "actor")
            except (httpx.HTTPError, ValueError, KeyError) as error:
                sample["actor"]["last_error"] = str(error)
                self._stages.save(document)
        return bool(planned)

    def _next_round(self, document: dict[str, Any]) -> bool:
        """Plan the review of the next repair round, and return whether there is one.

        Its parents are the round's graded, failed repairs at each point where a student was graded and
        nothing resolved by the trainer's measure (``_resolved``).
        """
        search = document["search"]
        config = search["config"]
        round_no = search.get("round", 1)
        if (
            search.get("shape") != "full"
            or round_no >= config.get("repair_rounds", 1)
            or self._stages.cancelled(document)
        ):
            return False
        parents = _chain_parents(document["samples"], points=search["points"], round_no=round_no)
        if not parents:
            search["chain_end"] = (
                f"after round {round_no}: no unresolved point with a failed repair to review"
            )
            return False
        template = deepcopy(search["review_templates"]["chain"])
        template["spec"]["round_no"] = round_no + 1
        template["spec"]["parents"] = _review_parents(parents)
        template["context"] = {
            self._context_key: {
                "rollout_job_id": document["rollout_job_id"],
                "phase": "review",
                "outcome": search.get("outcome"),
                "round": round_no + 1,
            }
        }
        stage = _review_stage(template, slot=chain_review_slot(round_no + 1), round=round_no + 1)
        search.setdefault("chain_reviews", []).append(stage)
        search.update(round=round_no + 1, state="chaining")
        return True

    def _plan_chain(self, document: dict[str, Any], stage: dict[str, Any]) -> None:
        """Turn a later round's review, ``stage``, into branches after the histories of its parent repairs.

        A chain branch belongs to its chain's root point, in the group ``repair:<point_id>:<n>``, with ``n``
        numbered on from earlier rounds. Its turn limit is what its parent had left at the chosen point.
        """
        search = document["search"]
        config = search["config"]
        round_no = stage["round"]
        free = list(search["free_slots"])
        repairs = {
            sample["actor"]["job_id"]: sample
            for sample in document["samples"]
            if (sample.get("search") or {}).get("kind") == "repair"
        }
        planned, skipped = [], []
        for point in (stage.get("result") or {}).get("points") or []:
            parent = repairs.get(point["job_id"])
            recovery = None if parent is None else self._recovery(point)
            if parent is None or recovery is None:
                reason = (
                    "no available recovery point at that step" if parent else "not a repair of this group"
                )
                skipped.append({"round": round_no, "step": point.get("step"), "reason": reason})
                continue
            inherited = parent["actor"]["submission"].get("budget") or {}
            budget = _branch_budget(
                parent_max_turns=inherited.get("max_turns", search["parent_max_turns"]),
                recovery=recovery,
                wall_time_seconds=config["wall_time_seconds"],
            )
            about = _chain_about(parent=parent, point=point, recovery=recovery, round_no=round_no)
            for c_index, candidate in enumerate(point.get("candidates") or [], 1):
                number = (round_no - 1) * config["candidates"] + c_index
                for _ in range(config["candidate_continuations"]):
                    if not free:
                        skipped.append({"round": round_no, "reason": "no free slot"})
                        break
                    planned.append(
                        {
                            "sample_slot_id": free.pop(0),
                            "branch": {
                                "job_id": point["job_id"],
                                "point_id": recovery["id"],
                                "continuation": _repair_continuation(candidate),
                                "budget": budget,
                            },
                            "grade": deepcopy(parent["grade"]["template"]),
                            "search": _chain_repair(
                                about=about, point=point, candidate=candidate, number=number, parent=parent
                            ),
                        }
                    )
        # Added after the loop. Then a failed read in the loop leaves the search as it was, and the next
        # call records each skip once.
        search.setdefault("skipped", []).extend(skipped)
        if not self._add_branches(document, planned, free):
            search["chain_error"] = f"round {round_no}'s review proposed no branch that could be made"

    def _recovery(self, point: dict[str, Any]) -> dict[str, Any] | None:
        """The available recovery point at the review point's tool step, or None."""
        return next(
            (
                recovery
                for recovery in self._client.points(point["job_id"], attempt_id=point["attempt_id"])
                if recovery.get("available") is True and recovery["tool_depth"] == point["step"]
            ),
            None,
        )


def _settled(samples: list[dict[str, Any]]) -> bool:
    return all(
        stage["state"] in DONE
        for sample in samples
        for stage in (sample["actor"], sample["grade"])
        if stage is not None
    )


def _review_parents(samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The review's parents: each sample's actor attempt and what the review reads of its grade."""
    return [
        {
            "name": sample["sample_slot_id"],
            "job_id": sample["actor"]["job_id"],
            "attempt_id": sample["actor"]["attempt_id"],
            "grade": {
                key: value for key, value in sample["grade"]["result"].items() if key in REVIEW_GRADE_KEYS
            },
        }
        for sample in samples
    ]


def _review_stage(template: dict[str, Any], **extra: object) -> dict[str, Any]:
    return {
        "state": "planned",
        "job_id": None,
        "attempt_id": None,
        "operation": "review",
        **extra,
        "submission": JobSpec.from_dict(template).validate(),
    }


def _branch_budget(
    *, parent_max_turns: int, recovery: dict[str, Any], wall_time_seconds: float
) -> dict[str, Any]:
    """A branch's budget: the turns its parent had left at the point, and a deadline of its own.

    The deadline is ``wall_time_seconds`` after the call, when the branch is planned.
    """
    return {
        "max_turns": max(1, parent_max_turns - recovery["message_step"]),
        "deadline_at": time.time() + wall_time_seconds,
    }


def _point_about(
    *, index: int, point: dict[str, Any], recovery: dict[str, Any], search: dict[str, Any]
) -> dict[str, Any]:
    """What every branch from a reviewed root point carries about the point and the roots."""
    # Every branch also carries the roots' label and counts: the trainer
    # checks each trajectory's against the roots it is sent.
    return {
        "point": index,
        "point_id": recovery["id"],
        "step": point["step"],
        "message_step": recovery["message_step"],
        "parent_job_id": point["job_id"],
        "outcome": search.get("outcome"),
        "root_counts": search.get("root_counts"),
    }


def _point_branches(
    *, point: dict[str, Any], about: dict[str, Any], students: int, repairs: int
) -> Iterator[tuple[dict[str, Any], dict[str, Any]]]:
    """The continuation and search annotation of each branch at a root point, in slot order.

    First ``students`` students with nothing added, then ``repairs`` repairs of each
    of the point's repair turns, numbered from 1.
    """
    for _ in range(students):
        yield {"guidance": "none"}, {"kind": "student", "group": f"student:{about['point_id']}", **about}
    for c_index, candidate in enumerate(point.get("candidates") or [], 1):
        for _ in range(repairs):
            yield (
                _repair_continuation(candidate),
                {
                    "kind": "repair",
                    "group": f"repair:{about['point_id']}:{c_index}",
                    **about,
                    "candidate": c_index,
                    "multiplicity": candidate.get("multiplicity", 1),
                    "samples": point.get("samples"),
                    "round": 1,
                    # The hidden test names that the turn gives away, as the review found them.
                    # None: the review had nothing to check against.
                    "leak_terms": candidate.get("leak_terms"),
                    "assistant_turn_rendering": REPAIR_RENDERING,
                    **_leak_judge(candidate),
                },
            )


def _chain_parents(
    samples: list[dict[str, Any]], *, points: list[dict[str, Any]], round_no: int
) -> list[dict[str, Any]]:
    """The round's failed repairs at each point that a student was graded at and nothing resolved."""
    parents = []
    for point in points:
        at = [
            sample for sample in samples if (sample.get("search") or {}).get("point_id") == point["point_id"]
        ]
        if any(_resolved(sample) for sample in at):
            continue
        # The trainer credits a repair only at a point where a student was graded.
        if not any(
            sample["search"]["kind"] == "student"
            and export_problem(sample) is None
            and grade_verdict(sample) is not None
            for sample in at
        ):
            continue
        parents += [
            sample
            for sample in at
            if sample["search"]["kind"] == "repair"
            and sample["search"].get("round", 1) == round_no
            and export_problem(sample) is None
            and grade_verdict(sample) is not None
        ]
    return parents


def _chain_about(
    *, parent: dict[str, Any], point: dict[str, Any], recovery: dict[str, Any], round_no: int
) -> dict[str, Any]:
    """What a chain branch carries: its root point's annotation, its round, and its own parent and point."""
    about = {
        key: parent["search"][key]
        for key in ("point", "point_id", "step", "message_step", "parent_job_id", "outcome", "root_counts")
    }
    about.update(
        round=round_no,
        chain_parent_job_id=point["job_id"],
        chain_point_id=recovery["id"],
        chain_step=point["step"],
    )
    return about


def _chain_repair(
    *,
    about: dict[str, Any],
    point: dict[str, Any],
    candidate: dict[str, Any],
    number: int,
    parent: dict[str, Any],
) -> dict[str, Any]:
    """The search annotation of a chain repair. Its ``leak_terms`` are those of every turn on the chain."""
    return {
        "kind": "repair",
        "group": f"repair:{about['point_id']}:{number}",
        **about,
        "candidate": number,
        "multiplicity": candidate.get("multiplicity", 1),
        "samples": point.get("samples"),
        "leak_terms": _merged_leak_terms(parent["search"].get("leak_terms"), candidate.get("leak_terms")),
        "assistant_turn_rendering": REPAIR_RENDERING,
        **_leak_judge(candidate, parent["search"].get("leak_judge")),
    }


def _repair_continuation(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "guidance": "assistant-turn",
        "assistant_turn": candidate["assistant_turn"],
        "assistant_turn_rendering": REPAIR_RENDERING,
    }


def _resolved(sample: dict[str, Any]) -> bool:
    """Whether a branch resolved the task in a way that the trainer credits.

    A repair that changed a file of the hidden test patch, or cannot say, did not.
    """
    verdict = grade_verdict(sample) if export_problem(sample) is None else None
    if verdict is None or verdict["resolved"] is not True:
        return False
    return sample["search"]["kind"] != "repair" or verdict.get("touched_test_paths") == []


def _merged_leak_terms(earlier: object, later: object) -> list[str] | None:
    """A chain's leak terms: every turn's, None when any turn had nothing to check against."""
    if not isinstance(earlier, list) or not isinstance(later, list):
        return None
    return earlier + [term for term in later if term not in earlier]


def _leak_judge(candidate: dict[str, Any], earlier: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """A repair's ``leak_judge`` entry: the check of each turn that the chain inserted, in round order.

    Each check is a ``worker.jobs.review.leak.judge.check_turn`` result. A turn is clean only when its
    ``leak`` is False. Empty when the candidate's review ran no judge.
    """
    if "leak_judge" not in candidate:
        return {}
    return {"leak_judge": list(earlier or []) + [candidate["leak_judge"]]}

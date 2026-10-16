"""One restartable coordinator; all agent and grader jobs execute elsewhere."""

from copy import deepcopy
import math
import threading
import time

import httpx

from sprout.runstore.specs import JobSpec
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.outcome import (
    SHAPES,
    done_reason,
    export_problem,
    grade_verdict,
    root_verdict,
    search_outcome,
    search_shape,
)
from sprout.rl_driver.specs import initial_document, submission_key, validate_request


DONE = {"succeeded", "failed", "cancelled", "skipped", "expired"}

#: What the review is given of each failed root's grade. The analyst reads the
#: first seven (``runstore.review.grade_of``); the last three, the task's hidden
#: tests, only the leak checks read -- ``methods.leak`` the names, so that every
#: repair turn's ``leak_terms`` is computed rather than None, and the review's
#: ``leak_judge`` the test patch itself.
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

#: The slot a search's review is submitted as, phase ``review``: its idempotency
#: key is ``submission_key(group, REVIEW_SLOT, "review")``. Round r > 1 of the
#: repairs is reviewed as ``chain_review_slot(r)``.
REVIEW_SLOT = "search"


def chain_review_slot(round_no: int) -> str:
    return f"{REVIEW_SLOT}-r{round_no}"


def _resolved(sample: dict) -> bool:
    """Whether a branch resolved the task in a way the trainer credits: a repair
    that changed a file of the hidden test patch (or cannot say) did not."""
    verdict = grade_verdict(sample) if export_problem(sample) is None else None
    if verdict is None or verdict["resolved"] is not True:
        return False
    return sample["search"]["kind"] != "repair" or verdict.get("touched_test_paths") == []


def _merged_leak_terms(earlier, later):
    """A chain's leak terms: every turn's, None when any turn had nothing to check against."""
    if not isinstance(earlier, list) or not isinstance(later, list):
        return None
    return earlier + [term for term in later if term not in earlier]


def _leak_judge(candidate: dict, earlier: list | None = None) -> dict:
    """A repair's ``leak_judge``, when its review ran one: the check of every
    turn the chain inserted (``methods.leak_judge.check_turn``), in round
    order. A turn is clean only when its ``leak`` is False; without a judge the
    key is absent, as from a review that predates it."""
    if "leak_judge" not in candidate:
        return {}
    return {"leak_judge": list(earlier or []) + [candidate["leak_judge"]]}


class Driver:
    def __init__(self, client, ledger: Ledger):
        self.client = client
        self.ledger = ledger
        self._tick_lock = threading.Lock()
        self.wakeup = threading.Event()
        self.stopping = threading.Event()

    def submit(
        self, body: dict, *, source_request: dict | None = None, extra_document: dict | None = None
    ) -> dict:
        request = validate_request(body)
        document = {**initial_document(request), **(extra_document or {})}
        row = self.ledger.create(request["rollout_job_id"], source_request or request, document)
        self.wakeup.set()
        return self._view(row)

    @staticmethod
    def _view(row: dict) -> dict:
        document = deepcopy(row["document"])
        document["cancel_requested"] = row["cancel_requested"]
        document["acknowledged_at"] = row["acknowledged_at"]
        if row["cancel_requested"] and not row["terminal"]:
            document["status"] = "cancelling"
        # Submission bodies may contain long prompts and are already retained
        # in the request ledger. Status/results expose the execution references.
        for sample in document["samples"]:
            for phase in ("actor", "grade"):
                stage = sample[phase]
                if stage:
                    stage.pop("submission", None)
                    stage.pop("template", None)
        search = document.get("search")
        if search:
            search.pop("review_templates", None)
            search.pop("review_template", None)  # a document from before outcome-keyed review
            for stage in [search.get("review"), *search.get("chain_reviews", [])]:
                if stage:
                    stage.pop("submission", None)
        return document

    def get(self, group_id: str) -> dict:
        return self._view(self.ledger.get(group_id))

    def release(self, group_id: str) -> dict:
        self.ledger.release(group_id)
        self.wakeup.set()
        view = self.get(group_id)
        return {key: view[key] for key in ("rollout_job_id", "status", "acknowledged_at")}

    def _save(self, document: dict) -> None:
        self.ledger.save(document["rollout_job_id"], document, terminal=document["ready"])

    def _cancelled(self, document: dict) -> bool:
        return self.ledger.get(document["rollout_job_id"])["cancel_requested"]

    def _dispatch(self, document: dict, sample: dict, phase: str) -> None:
        stage = sample[phase]
        if stage["state"] == "planned":
            if self._cancelled(document):
                stage["state"] = "cancelled"
                self._save(document)
                return
            stage["state"] = "submitting"
            self._save(document)  # intent is durable BEFORE a remote side effect
        key = submission_key(document["rollout_job_id"], sample["sample_slot_id"], phase)
        try:
            if stage["operation"] == "branch":
                branch = stage["submission"]
                job_id = self.client.branch(
                    branch["job_id"],
                    branch["point_id"],
                    idempotency_key=key,
                    continuation=branch["continuation"],
                    budget=branch.get("budget"),
                    **branch["overrides"],
                )
            else:
                job_id = self.client.submit(JobSpec.from_dict(stage["submission"]), key)
        except httpx.HTTPStatusError as error:
            if error.response.status_code in {400, 404, 409, 422} and not stage.get("submission_uncertain"):
                stage.update(state="failed", error=error.response.text)
                self._save(document)
                return
            stage["submission_uncertain"] = True
            raise
        except httpx.RequestError:
            stage["submission_uncertain"] = True
            raise
        stage.update(job_id=job_id, state="queued")
        stage.pop("submission_uncertain", None)
        stage.pop("last_error", None)
        self._save(document)

    def _poll(self, document: dict, stage: dict) -> None:
        job = self.client.get(stage["job_id"])
        if job["state"] in {"queued", "running"} and self._cancelled(document):
            # Nothing reads a cancelled group's results: a queued job is
            # cancelled, a running one stopped and its sandboxes reclaimed.
            self.client.stop(stage["job_id"])
            job = self.client.get(stage["job_id"])
        stage.update(
            state=job["state"],
            attempt_id=job.get("active_attempt"),
            phase=job.get("phase"),
            error=job.get("error"),
        )
        if job["state"] != "queued":
            # When it left the queue: a grade's wall time is its own, not the queue's.
            stage.setdefault("started_at", time.time())
        if job["state"] in DONE:
            stage["result"] = job.get("result")
        stage.pop("last_error", None)
        self._save(document)

    def _prepare_grade(self, document: dict, sample: dict) -> None:
        actor, grade = sample["actor"], sample["grade"]
        attempt = actor["attempt_id"]
        if not attempt:
            raise ValueError("Completed actor has no attempt identity")
        output = actor.get("result") or {}
        if output.get("final_snapshot_id"):
            # Every rollout ends with a final disk snapshot; it covers text-only
            # last turns and also permits scoring a run with no tool calls.
            point = {"id": None, "snapshot_id": output["final_snapshot_id"]}
        else:
            tools = self.client.all_tools(actor["job_id"], attempt_id=attempt)
            if not tools or any(tool.get("response") is None for tool in tools):
                raise ValueError("Cannot grade without a complete final tool boundary")
            final_depth = tools[-1]["depth"]
            if [tool["depth"] for tool in tools] != list(range(1, final_depth + 1)):
                raise ValueError("Final tool prefix has a gap")
            points = self.client.points(actor["job_id"], attempt_id=attempt)
            candidates = [
                point
                for point in points
                if point.get("available") is True and point["tool_depth"] == final_depth
            ]
            if not candidates:
                raise ValueError("Final message has no available recovery point; refusing an older snapshot")
            point = max(candidates, key=lambda item: item["message_step"])
        submission = deepcopy(grade["template"])
        submission["spec"]["snapshot_id"] = point["snapshot_id"]
        submission["context"]["rl_driver"] = {
            "rollout_job_id": document["rollout_job_id"],
            "sample_slot_id": sample["sample_slot_id"],
            "actor_job_id": actor["job_id"],
            "actor_attempt_id": attempt,
            "point_id": point["id"],
        }
        grade["submission"] = JobSpec.from_dict(submission).validate()
        grade["snapshot_id"] = point["snapshot_id"]
        self._save(document)

    def _advance(self, document: dict, sample: dict, phase: str) -> None:
        stage = sample[phase]
        if stage is None or stage["state"] in DONE:
            return
        wall = document.get("grade_wall_seconds")
        if phase == "grade" and wall is not None and time.time() >= stage.get("started_at", math.inf) + wall:
            # Its verdict is no longer read; the worker stops it and reclaims its sandboxes.
            stage.update(state="expired", error=f"Grading did not settle within {wall:g} s")
            self._save(document)
            try:
                self.client.stop(stage["job_id"])
            except httpx.HTTPError as error:
                stage["last_error"] = f"Could not stop the expired grade: {error}"
                self._save(document)
            return
        if not stage["job_id"]:
            if self._cancelled(document):
                if stage["state"] == "planned":
                    stage["state"] = "cancelled"
                else:
                    # A lost acknowledgement cannot be reconciled by creating
                    # a possibly NEW job after cancellation was requested.
                    stage["last_error"] = (
                        "Submission acknowledgement unknown; inspect Run Store using the idempotency key"
                    )
                    stage["idempotency_key"] = submission_key(
                        document["rollout_job_id"], sample["sample_slot_id"], phase
                    )
                self._save(document)
                return
            if phase == "grade" and "submission" not in stage:
                try:
                    self._prepare_grade(document, sample)
                except ValueError as error:
                    stage.update(state="failed", error=str(error))
                    self._save(document)
                    return
            self._dispatch(document, sample, phase)
        if stage["job_id"]:
            self._poll(document, stage)

    def tick(self) -> None:
        """Poll all owned groups once. A disconnected group does not block others."""
        if not self._tick_lock.acquire(blocking=False):
            return
        try:
            for group_id in self.ledger.active():
                if self.stopping.is_set():
                    break
                document = self.ledger.get(group_id)["document"]
                if document.get("deadline_at") is not None and time.time() >= document["deadline_at"]:
                    self.ledger.release(group_id)
                for sample in document["samples"]:
                    for phase in ("actor", "grade"):
                        if self.stopping.is_set():
                            return
                        stage = sample[phase]
                        if stage is None:
                            continue
                        if phase == "grade" and sample["actor"]["state"] != "succeeded":
                            if sample["actor"]["state"] in DONE:
                                stage.update(state="skipped", error="Actor did not succeed")
                            continue
                        try:
                            self._advance(document, sample, phase)
                        except (httpx.HTTPError, ValueError, KeyError) as error:
                            # Do not turn a polling failure into a new execution.
                            stage["last_error"] = str(error)
                            self._save(document)
                if document.get("search"):
                    # After the samples, so a stage that just settled counts this tick;
                    # branches planned here are dispatched here too.
                    try:
                        self._advance_search(document)
                    except (httpx.HTTPError, ValueError, KeyError) as error:
                        document["search"]["last_error"] = str(error)
                        self._save(document)
                stages = [
                    sample[phase]
                    for sample in document["samples"]
                    for phase in ("actor", "grade")
                    if sample[phase] is not None
                ]
                settled = not document.get("search") or document["search"]["state"] == "done"
                document["ready"] = settled and all(stage["state"] in DONE for stage in stages)
                if document["ready"]:
                    document["status"] = (
                        "cancelled"
                        if self._cancelled(document)
                        else "failed"
                        if any(stage["state"] != "succeeded" for stage in stages)
                        else "completed"
                    )
                else:
                    document["status"] = (
                        "quarantined"
                        if any(stage["state"] == "quarantined" for stage in stages)
                        else "running"
                    )
                self._save(document)
        finally:
            self._tick_lock.release()

    # ---- search: hindsight branches once the roots are graded -------------------

    def _advance_search(self, document: dict) -> None:
        """Roots graded -> review the failed ones -> branch where it says -> [next round] -> done.

        Once every root has settled, their counts -- planned, graded and
        resolved, a missing root in neither -- give the task its label
        (``outcome.search_outcome``), and the label the shape of what follows
        (``outcome.search_shape``): the full review, the points-only one, or
        nothing. The search document keeps all three, and the export labels
        every trajectory with the label and the counts. The review is a Run
        Store job like any other stage (``_advance`` submits and polls it
        under its own idempotency key), one per group whatever the shape. A
        review that fails or proposes nothing ends the search with the roots
        alone: the group still returns what it has, and ``search.error`` says
        what did not happen. With more than one repair round, the branches of
        a round settled, the points still unresolved are reviewed again
        (``_next_round``) and branched once more (``_plan_chain``); a later
        round's review that fails ends the search with what it has, noted as
        ``search.chain_error``.
        """
        search = document["search"]
        if search["state"] == "done":
            return
        samples = document["samples"]
        roots = [sample for sample in samples if sample["sample_slot_id"] in search["root_slots"]]
        branches = [sample for sample in samples if sample["sample_slot_id"] not in search["root_slots"]]

        def settled(group):
            return all(
                stage["state"] in DONE
                for sample in group
                for stage in (sample["actor"], sample["grade"])
                if stage is not None
            )

        if search["state"] == "roots":
            if not settled(roots):
                return
            verdicts = [root_verdict(sample) for sample in roots]
            counts = {
                "planned": search["config"]["roots"],
                "graded": sum(verdict is not None for verdict in verdicts),
                "resolved": sum(verdict is True for verdict in verdicts),
            }
            outcome = search_outcome(**counts)
            # A document from before outcome-keyed review has one template and no
            # tf_student_continuations; past the shape check it ends for want of templates.
            legacy = "review_templates" not in search
            shape = SHAPES[outcome] if legacy else search_shape(outcome, search["config"])
            search.update(root_counts=counts, outcome=outcome, shape=shape)
            if self._cancelled(document) or shape == "none":
                search.update(state="done", error=done_reason(outcome, counts) if shape == "none" else None)
                self._save(document)
                return
            if legacy:
                search.update(state="done", error="search document predates outcome-keyed review")
                self._save(document)
                return
            template = deepcopy(search["review_templates"][shape])
            template["spec"]["parents"] = [
                {
                    "name": sample["sample_slot_id"],
                    "job_id": sample["actor"]["job_id"],
                    "attempt_id": sample["actor"]["attempt_id"],
                    "grade": {
                        key: value
                        for key, value in sample["grade"]["result"].items()
                        if key in REVIEW_GRADE_KEYS
                    },
                }
                for sample, verdict in zip(roots, verdicts, strict=True)
                if verdict is False
            ]
            template["context"] = {
                "rl_driver": {
                    "rollout_job_id": document["rollout_job_id"],
                    "phase": "review",
                    "outcome": outcome,
                }
            }
            search["review"] = {
                "state": "planned",
                "job_id": None,
                "attempt_id": None,
                "operation": "review",
                "submission": JobSpec.from_dict(template).validate(),
            }
            search["state"] = "reviewing"
            self._save(document)
        if search["state"] == "reviewing":
            stage = search["review"]
            self._advance(document, {"sample_slot_id": REVIEW_SLOT, "review": stage}, "review")
            if stage["state"] == "succeeded":
                self._plan_branches(document)
                return  # the branches settle on a later tick
            elif stage["state"] in DONE:
                search.update(state="done", error=stage.get("error") or f"review {stage['state']}")
                self._save(document)
        if search["state"] == "branching" and settled(branches):
            if not self._next_round(document):
                search["state"] = "done"
            self._save(document)
        if search["state"] == "chaining":
            stage = search["chain_reviews"][-1]
            self._advance(document, {"sample_slot_id": stage["slot"], "review": stage}, "review")
            if stage["state"] == "succeeded":
                self._plan_chain(document, stage)
            elif stage["state"] in DONE:
                search.update(state="done", chain_error=stage.get("error") or f"review {stage['state']}")
                self._save(document)

    @staticmethod
    def _branch_budget(search: dict, recovery: dict) -> dict:
        """A branch's budget: the turns its parent had left at the point, and a deadline of its own.

        The deadline runs ``wall_time_seconds`` from now, when the branch is
        planned. Deferred (P3): the budget an unaided continuation has at the
        point -- a repair's inserted turn charged to its turns -- with the
        deadline counted from when the branch starts executing rather than
        from planning. That change belongs here; nothing else computes a
        branch's budget.
        """
        return {
            "max_turns": max(1, search["parent_max_turns"] - recovery["message_step"]),
            "deadline_at": time.time() + search["config"]["wall_time_seconds"],
        }

    def _plan_branches(self, document: dict) -> None:
        """Turn the review's points into branch samples, each in the next free slot.

        Under the ``full`` shape (FF), each point gets ``student_continuations``
        students with nothing added and, for each repair turn the review wrote
        there, ``candidate_continuations`` repairs that insert it as the turn's
        reasoning; under ``points_only`` (TF, FF_partial) it gets
        ``tf_student_continuations`` students and no repair, whatever the plan
        carries. A document reviewed before shapes existed has none, and was
        reviewed in full.
        """
        search = document["search"]
        config = search["config"]
        shape = search.get("shape", "full")
        students = config["student_continuations"] if shape == "full" else config["tf_student_continuations"]
        plan = search["review"].get("result") or {}
        free = list(search["free_slots"])
        roots = {
            sample["actor"]["job_id"]: sample
            for sample in document["samples"]
            if sample["sample_slot_id"] in search["root_slots"]
        }
        planned, skipped = [], []
        for index, point in enumerate(plan.get("points") or [], 1):
            root = roots.get(point["job_id"])
            recovery = next(
                (
                    p
                    for p in self.client.points(point["job_id"], attempt_id=point["attempt_id"])
                    if p.get("available") is True and p["tool_depth"] == point["step"]
                ),
                None,
            )
            if root is None or recovery is None:
                skipped.append(
                    {
                        "point": index,
                        "step": point.get("step"),
                        "reason": "no available recovery point at that step"
                        if root
                        else "not a root of this group",
                    }
                )
                continue
            budget = self._branch_budget(search, recovery)
            # Every branch also carries the roots' label and counts: the trainer
            # checks each trajectory's against the roots it is sent.
            about = {
                "point": index,
                "point_id": recovery["id"],
                "step": point["step"],
                "message_step": recovery["message_step"],
                "parent_job_id": point["job_id"],
                "outcome": search.get("outcome"),
                "root_counts": search.get("root_counts"),
            }
            search["points"].append(
                {
                    **about,
                    "snapshot_id": recovery["snapshot_id"],
                    "why": point.get("why"),
                    "candidates": len(point.get("candidates") or []),
                }
            )

            def branch(kind, continuation, group, **extra):
                if not free:
                    skipped.append({"point": index, "reason": "no free slot"})
                    return
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
                        "search": {"kind": kind, "group": group, **about, **extra},
                    }
                )

            for _ in range(students):
                branch("student", {"guidance": "none"}, f"student:{recovery['id']}")
            if shape == "full":
                for c_index, candidate in enumerate(point.get("candidates") or [], 1):
                    for _ in range(config["candidate_continuations"]):
                        branch(
                            "repair",
                            {
                                "guidance": "assistant-turn",
                                "assistant_turn": candidate["assistant_turn"],
                                "assistant_turn_rendering": REPAIR_RENDERING,
                            },
                            f"repair:{recovery['id']}:{c_index}",
                            candidate=c_index,
                            multiplicity=candidate.get("multiplicity", 1),
                            samples=point.get("samples"),
                            round=1,
                            # The hidden test names the turn gives away, as the review
                            # found them; None when it had nothing to check against.
                            leak_terms=candidate.get("leak_terms"),
                            assistant_turn_rendering=REPAIR_RENDERING,
                            **_leak_judge(candidate),
                        )
        search["skipped"] = skipped
        if not self._add_branches(document, planned, free):
            search["error"] = "the review proposed no branch that could be made"

    def _add_branches(self, document: dict, planned: list[dict], free: list[str]) -> bool:
        """Add planned branch samples to the document and dispatch them; whether there were any."""
        search = document["search"]
        if planned:
            body = validate_request(
                {
                    "rollout_job_id": document["rollout_job_id"],
                    "prompt_group_id": document["prompt_group_id"],
                    "samples": [{k: v for k, v in sample.items() if k != "search"} for sample in planned],
                }
            )
            built = initial_document(body)["samples"]
            for sample, annotated in zip(built, planned, strict=True):
                sample["search"] = annotated["search"]
            document["samples"].extend(built)
        search.update(free_slots=free, state="branching" if planned else "done")
        self._save(document)
        for sample in document["samples"][-len(planned) :] if planned else []:
            if self.stopping.is_set():
                break
            try:
                self._advance(document, sample, "actor")
            except (httpx.HTTPError, ValueError, KeyError) as error:
                sample["actor"]["last_error"] = str(error)
                self._save(document)
        return bool(planned)

    def _next_round(self, document: dict) -> bool:
        """Submit the next repair round's review, or say there is none.

        A point takes part when a student was graded there (the trainer credits
        a repair only then), none resolved, and no repair of any round resolved
        by the trainer's measure (``_resolved``). Its parents are its round's
        repairs that were graded and failed; each is reviewed on its own for
        one point in its history and one turn there.
        """
        search = document["search"]
        config = search["config"]
        round_no = search.get("round", 1)
        if (
            search.get("shape") != "full"
            or round_no >= config.get("repair_rounds", 1)
            or self._cancelled(document)
        ):
            return False
        parents = []
        for point in search["points"]:
            at = [
                sample
                for sample in document["samples"]
                if (sample.get("search") or {}).get("point_id") == point["point_id"]
            ]
            if any(_resolved(sample) for sample in at):
                continue
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
        if not parents:
            search["chain_end"] = (
                f"after round {round_no}: no unresolved point with a failed repair to review"
            )
            return False
        template = deepcopy(search["review_templates"]["chain"])
        template["spec"]["round_no"] = round_no + 1
        template["spec"]["parents"] = [
            {
                "name": sample["sample_slot_id"],
                "job_id": sample["actor"]["job_id"],
                "attempt_id": sample["actor"]["attempt_id"],
                "grade": {
                    key: value for key, value in sample["grade"]["result"].items() if key in REVIEW_GRADE_KEYS
                },
            }
            for sample in parents
        ]
        template["context"] = {
            "rl_driver": {
                "rollout_job_id": document["rollout_job_id"],
                "phase": "review",
                "outcome": search.get("outcome"),
                "round": round_no + 1,
            }
        }
        search.setdefault("chain_reviews", []).append(
            {
                "state": "planned",
                "job_id": None,
                "attempt_id": None,
                "operation": "review",
                "slot": chain_review_slot(round_no + 1),
                "round": round_no + 1,
                "submission": JobSpec.from_dict(template).validate(),
            }
        )
        search.update(round=round_no + 1, state="chaining")
        return True

    def _plan_chain(self, document: dict, stage: dict) -> None:
        """A later round's branches: each after its parent repair's history, at the point the review chose there.

        A chain branch belongs to the root point its chain started from (its
        ``point_id``, group ``repair:<point>:<candidate>``, the candidate
        numbered on from the earlier rounds'), and also names its own point
        (``chain_point_id``, ``chain_step``) and parent (``chain_parent_job_id``).
        Its budget is the turns its parent had left there; its ``leak_terms``
        are every turn's of the chain.
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
        planned, skipped = [], search.setdefault("skipped", [])
        for point in (stage.get("result") or {}).get("points") or []:
            parent = repairs.get(point["job_id"])
            recovery = (
                None
                if parent is None
                else next(
                    (
                        p
                        for p in self.client.points(point["job_id"], attempt_id=point["attempt_id"])
                        if p.get("available") is True and p["tool_depth"] == point["step"]
                    ),
                    None,
                )
            )
            if parent is None or recovery is None:
                skipped.append(
                    {
                        "round": round_no,
                        "step": point.get("step"),
                        "reason": "no available recovery point at that step"
                        if parent
                        else "not a repair of this group",
                    }
                )
                continue
            inherited = parent["actor"]["submission"].get("budget") or {}
            budget = {
                "max_turns": max(
                    1, inherited.get("max_turns", search["parent_max_turns"]) - recovery["message_step"]
                ),
                "deadline_at": time.time() + config["wall_time_seconds"],
            }
            about = {
                key: parent["search"][key]
                for key in (
                    "point",
                    "point_id",
                    "step",
                    "message_step",
                    "parent_job_id",
                    "outcome",
                    "root_counts",
                )
            }
            about.update(
                round=round_no,
                chain_parent_job_id=point["job_id"],
                chain_point_id=recovery["id"],
                chain_step=point["step"],
            )
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
                                "continuation": {
                                    "guidance": "assistant-turn",
                                    "assistant_turn": candidate["assistant_turn"],
                                    "assistant_turn_rendering": REPAIR_RENDERING,
                                },
                                "budget": budget,
                            },
                            "grade": deepcopy(parent["grade"]["template"]),
                            "search": {
                                "kind": "repair",
                                "group": f"repair:{about['point_id']}:{number}",
                                **about,
                                "candidate": number,
                                "multiplicity": candidate.get("multiplicity", 1),
                                "samples": point.get("samples"),
                                "leak_terms": _merged_leak_terms(
                                    parent["search"].get("leak_terms"), candidate.get("leak_terms")
                                ),
                                "assistant_turn_rendering": REPAIR_RENDERING,
                                **_leak_judge(candidate, parent["search"].get("leak_judge")),
                            },
                        }
                    )
        if not self._add_branches(document, planned, free):
            search["chain_error"] = f"round {round_no}'s review proposed no branch that could be made"

    def reconcile_submission(self, group_id: str, sample_id: str, phase: str, job_id: str) -> dict:
        """Attach a known accepted job after cancellation plus a lost HTTP reply.

        The upstream idempotency key proves it belongs to this exact submission.
        This operation never submits work or changes an execution's state.
        A search's review is the ``review`` phase of the slot ``REVIEW_SLOT``,
        a later repair round's that of ``chain_review_slot(round)``.
        """
        if phase not in {"actor", "grade", "review"}:
            raise ValueError("phase must be actor, grade or review")
        with self._tick_lock:
            document = self.ledger.get(group_id)["document"]
            if phase == "review":
                search = document.get("search") or {}
                stages = {
                    REVIEW_SLOT: search.get("review"),
                    **{stage["slot"]: stage for stage in search.get("chain_reviews", [])},
                }
                if sample_id not in stages:
                    raise KeyError(sample_id)
                stage = stages[sample_id]
            else:
                sample = next((s for s in document["samples"] if s["sample_slot_id"] == sample_id), None)
                if sample is None:
                    raise KeyError(sample_id)
                stage = sample[phase]
            if stage is None or stage["state"] != "submitting" or stage["job_id"]:
                raise ValueError("Only an unacknowledged submission can be reconciled")
            job = self.client.get(job_id)
            expected = submission_key(group_id, sample_id, phase)
            if job.get("idempotency_key") != expected:
                raise ValueError("Job does not match this submission's idempotency key")
            stage.update(
                job_id=job["id"],
                state=job["state"],
                attempt_id=job.get("active_attempt"),
                phase=job.get("phase"),
                error=job.get("error"),
            )
            if job["state"] in DONE:
                stage["result"] = job.get("result")
            stage.pop("last_error", None)
            stage.pop("submission_uncertain", None)
            self._save(document)
        self.wakeup.set()
        return self.get(group_id)

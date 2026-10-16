"""The review job: a failed rollout's hindsight, as the Run Store runs it.

Given the finished attempts of one task -- their journals, native histories
and grades, all on this host -- the model that ran them reads the failures
under SPROUT's analyst and reviewer prompts (``methods.branching``) and chooses
where a continuation is worth trying. With ``assistant-turn`` guidance it also
writes, for each chosen point, several independent candidate turns to insert
there; the prompt holds them to what the retained history shows. Nothing runs
here: the plan is what the RL driver turns into branch jobs.

Two stages share one set of analyses:

1. ``none`` review over every failed attempt picks the points (``points`` of
   them in total, across attempts; a point two attempts both failed at may be
   picked once). With ``points_per_attempt`` each failed attempt is reviewed
   on its own and contributes up to ``points``, so no failure goes unbranched
   because another looked more promising.
2. For each point, ``candidates`` independent ``assistant-turn`` reviews, each
   confined to that one point (``eligible``), at a temperature above zero so
   the proposals differ. Identical proposals -- the same commands -- are merged
   and their multiplicity kept: how many of the samples agreed is itself
   information about the teacher's distribution.

Each candidate is ``{assistant_turn, why, multiplicity, leak_terms}``. The
analyst has read the test log and the reviewer has read the analyst's report;
the actor has seen neither. ``leak_terms`` (``methods.leak``) lists the
hidden test names the turn mentions that h_s never showed. h_s is the history a continuation from that point keeps, with
hints removed, as training sees it. The hidden names come from two facts in
the parent's grade, ``fail_to_pass`` and ``test_patch_paths``, and neither
fact is ever shown to the analyst or the reviewer. ``leak_terms`` is ``[]``
for a clean turn and None when the grade has neither fact. Every candidate is
returned and run whatever its ``leak_terms``; whether it may be credited is
the trainer's decision.

The calls run side by side wherever none needs another's answer, at most
``max_concurrent_requests`` at a time, and the failed attempts are analysed
side by side. With ``points_per_attempt`` each failed attempt is then a
pipeline of its own -- its analysis, its point review, the candidate samples
at its points -- and the pipelines run side by side, so the eight calls of a
task whose two roots failed (a point each, two candidates) take three in a
row. Without it the one point review waits for every analysis, and the
samples at all its points then run together. The result is put together in
failed-attempt, point and sample order, never in the order the calls
finished. With one slot the calls run one after another in ``Call`` order:
every analysis, then every point review, then the samples, each stage in
failed-attempt, point and sample order -- the order the review made them in
before they ran side by side. Each call has its own receipts directory,
``reviews/<name>/``, and each review its own plan file, ``plan-<name>.json``,
so no file is named by which call came first.

A call that raises fails the review, with the error a review with one slot
fails with, whatever the timing: no call after the first failed one (in
``Call`` order) starts, those running are stopped -- they send no further
request, though one in flight runs its course -- and those before it carry
on, since one of them may fail first in that order. Once every running call
has returned, that first failure is raised with the others noted on it.
Interrupted -- the worker stops its child with SIGTERM -- the review stops
every call and leaves at once; the calls run on daemon threads, so a request
still in flight does not keep the child from exiting.

Runs in a worker child like a grade, with no sandbox; it reads the attempt
directories under ``artifact_root`` and talks to the model endpoint.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import heapq
import json
import logging
import os
from pathlib import Path
import queue
import threading
from typing import Any, Callable

from sprout.harness_agents.mini_swe.history import available_branch_points, training_messages
from sprout.methods.branching import (
    Attempt,
    BranchChoice,
    Grade,
    analyse,
    outcome_from_journal,
    review,
    selected_prefix,
)
from sprout.methods.review_transport import RequestStopped, ReviewTransport
from sprout.methods.leak import hidden_terms_of, leak_terms, visible_text
from sprout.methods.leak_judge import REVIEW_ATTEMPTS, REVIEW_TIMEOUT_S, HiddenText, JevLeakJudge, check_turn
from sprout.runstore.messages import clean_messages
from sprout.runstore.specs import ReviewSpec


def grade_of(record: dict) -> Grade:
    """SPROUT's view of a verdict, from what the grade job left behind.

    Only the keys named here are rendered. The grade's hidden-test facts
    (``fail_to_pass``, ``test_patch_paths``, ``touched_test_paths``, and the
    ``test_patch`` path) are kept for ``methods.leak`` and ``methods.leak_judge``
    and never reach the analyst.
    """
    patch_path = record.get("model_patch")
    patch = Path(patch_path).read_text(errors="replace") if patch_path and Path(patch_path).is_file() else ""
    lines = []
    if record.get("progress") is not None:
        lines.append("target tests passed (FAIL_TO_PASS): %.0f%%" % (100 * float(record["progress"])))
    if record.get("reason"):
        lines.append("reason: %s" % record["reason"])
    statuses = record.get("test_statuses") or {}
    failed = sorted(name for name, status in statuses.items() if status != "PASSED")
    if statuses:
        lines.append("test statuses: %d recorded, %d not passed" % (len(statuses), len(failed)))
        lines.extend("  %s: %s" % (name, statuses[name]) for name in failed[:40])
    log_path = record.get("log")
    if log_path and Path(log_path).is_file():
        tail = Path(log_path).read_text(errors="replace")[-20000:]
        lines.append("test log (tail):\n" + tail)
    if record.get("error_output"):
        lines.append("error output:\n" + str(record["error_output"])[-4000:])
    return Grade(
        resolved=record["resolved"],
        reward=float(record["resolved"]),
        patch=patch,
        detail="\n".join(lines),
        error=record.get("error"),
    )


def load_attempt(root: Path, parent: dict) -> Attempt:
    directory = root / parent["job_id"] / parent["attempt_id"]
    journal = directory / "trajectory.jsonl"
    if not journal.is_file():
        raise ValueError(f"Review parent {parent['name']} has no journal at {journal}")
    return Attempt(
        name=parent["name"],
        outcome=outcome_from_journal(journal, run_id=parent["name"]),
        grade=grade_of(parent["grade"]),
        hint=parent.get("hint") or "",
    )


def commands_of(turn: dict) -> tuple:
    return tuple(json.loads(call["function"]["arguments"])["command"] for call in turn["tool_calls"])


#: The stages of a review, in the order their calls start when they compete for a slot.
ANALYSIS, POINTS, CANDIDATE = 0, 1, 2


@dataclass(frozen=True, order=True)
class Call:
    """One stream of model requests in a review: a failed attempt's analysis,
    a point review, or one candidate sample at a point.

    Calls compare in the order a review with one slot makes them: by stage,
    since each unlocks more of the rest, then by failed attempt, point and
    sample. A review that fails, fails with the error of the first failed
    call in that order.
    """

    stage: int
    #: The failed attempt's position, from 1; 0 for the point review over
    #: every failed attempt and for the samples at its points.
    attempt: int
    #: The point's position among its point review's choices, and the sample's, from 1.
    point: int = 0
    sample: int = 0

    @property
    def name(self) -> str:
        """``reviews/<name>/`` holds the call's receipts and, but for an
        analysis, ``plan-<name>.json`` is its plan."""
        if self.stage == ANALYSIS:
            return f"analyst-{self.attempt}"
        if self.stage == POINTS:
            return f"points-{self.attempt}" if self.attempt else "points"
        point = f"{self.attempt}.{self.point}" if self.attempt else str(self.point)
        return f"point{point}-candidate{self.sample}"


class ReviewStopped(RequestStopped):
    """What a call is told when it asks the model for more after the review
    stopped it, and why: a call before it failed, or the review was
    interrupted. The review never fails with it while another error is known."""


class _Stop:
    """One call's stop. Once it is set, with why, the call asks the model for
    nothing more: ``event`` is its transport's ``stop``, so not even a retry
    inside the transport is sent."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason = ""

    def __call__(self, reason: str) -> None:
        if not self.event.is_set():
            self.reason = reason
            self.event.set()


def _analysis(spec: ReviewSpec, transport, attempt: Attempt) -> dict:
    points = available_branch_points(attempt.outcome.journal_path)
    return analyse(
        transport,
        spec.model,
        spec.problem,
        attempt,
        points,
        spec.points,
        spec.transcript_tokens,
        max_attempts=spec.max_attempts,
        prompt_chars=spec.prompt_chars,
    )


@dataclass(frozen=True)
class LeakFacts:
    """What a point's candidates are checked against: the hidden terms of the
    parent's grade (None without its facts), h_s as text, and the hidden text
    the Jev judge reads (None when no judge is configured or the grade names
    no test patch)."""

    terms: frozenset[str] | None
    visible: str
    hidden: HiddenText | None


def _point_review(
    spec: ReviewSpec, transport, scope: list[Attempt], reports: dict, plan_path: Path
) -> list[tuple[BranchChoice, LeakFacts | None]]:
    """Stage 1 over ``scope``: the points chosen, each with what its candidates' leak check reads."""
    choices = review(
        transport,
        spec.model,
        spec.problem,
        scope,
        reports,
        round_no=spec.round_no,
        width=spec.points,
        plan_path=plan_path,
        guidance_mode="none",
        count_mode="adaptive",
        reviewer_max_attempts=spec.max_attempts,
        prompt_chars=spec.prompt_chars,
    )
    return [
        (choice, _leak_facts(spec, choice) if spec.guidance == "assistant-turn" else None)
        for choice in choices
    ]


def _leak_facts(spec: ReviewSpec, choice: BranchChoice) -> LeakFacts:
    """The point's parent's hidden facts, and the text of h_s at the point."""
    parent = next(p for p in spec.parents if p["name"] == choice.base.name)
    terms = hidden_terms_of(parent["grade"])
    hidden = None
    if spec.leak_judge is not None:
        # The test patch the grade applied, and the verdict as the analyst read it.
        patch_path = parent["grade"].get("test_patch")
        if patch_path and Path(patch_path).is_file():
            hidden = HiddenText(
                Path(patch_path).read_text(errors="replace"),
                tuple(parent["grade"].get("fail_to_pass") or ()),
                choice.base.grade.detail,
            )
    # h_s. Hints are removed as in the training export: no policy is trained on
    # one, so a name that only a hint showed is still hidden.
    visible = (
        visible_text(
            clean_messages(
                training_messages(selected_prefix(choice.base.outcome.journal_path, choice.checkpoint))
            )
        )
        if terms or hidden
        else ""
    )
    return LeakFacts(terms, visible, hidden)


def _candidate(
    spec: ReviewSpec,
    transport,
    choice: BranchChoice,
    report: dict,
    plan_path: Path,
    facts: LeakFacts,
    judge_factory: Callable[[dict], Any] | None,
) -> BranchChoice | None:
    """Stage 2, one sample: an ``assistant-turn`` review confined to the point; None when it chose nothing.

    With ``leak_judge`` configured every turn the reviewer writes is checked
    (``leak_judge.check_turn``: ``methods.leak`` and Jev) before the plan is
    taken, and a flagged one is sent back, ``leak_judge.resamples`` times at
    most; the turn taken keeps its check either way.
    """
    turn_check, judge = None, None
    if spec.leak_judge is not None:
        judge = judge_factory(spec.leak_judge)

        def turn_check(proposal: BranchChoice) -> dict:
            return check_turn(
                proposal.assistant_turn,
                terms=facts.terms,
                visible=facts.visible,
                judge=judge,
                task_statement=spec.problem,
                hidden=facts.hidden,
            )

    try:
        found = review(
            transport,
            spec.model,
            spec.problem,
            [choice.base],
            {choice.base.name: report},
            round_no=spec.round_no,
            width=1,
            plan_path=plan_path,
            guidance_mode="assistant-turn",
            count_mode="fixed",
            reviewer_max_attempts=spec.max_attempts,
            prompt_chars=spec.prompt_chars,
            eligible={choice.base.name: {choice.checkpoint.step}},
            turn_check=turn_check,
            turn_resamples=(
                spec.leak_judge.get("resamples", LEAK_RESAMPLES) if spec.leak_judge is not None else 0
            ),
        )
    finally:
        if judge is not None and hasattr(judge, "close"):
            judge.close()
    return found[0] if found else None


#: How often a flagged turn goes back to the reviewer when ``leak_judge`` does not say.
LEAK_RESAMPLES = 2


def judge_key(config: dict) -> str:
    """The TypeSafe key a ``leak_judge`` config names, from the worker's environment."""
    variable = config.get("api_key_env", "TYPESAFE_API_KEY")
    key = os.environ.get(variable, "").strip()
    if not key:
        raise ValueError(f"leak_judge needs the TypeSafe key in the worker's ${variable}")
    return key


def jev_judge(config: dict) -> JevLeakJudge:
    """The Jev judge a review's ``leak_judge`` config names; its key is read on the worker."""
    # The review runs under its own deadline: a judge that cannot answer gives up
    # in about a minute and a half per request, and the turn stays unchecked.
    return JevLeakJudge(
        judge_key(config),
        timeout_s=REVIEW_TIMEOUT_S,
        max_attempts=REVIEW_ATTEMPTS,
        **{name: config[name] for name in ("model", "threshold") if name in config},
    )


def _merge_candidates(samples: list[BranchChoice | None], facts: LeakFacts, judged: bool) -> list[dict]:
    """A point's samples, in sample order, as its candidates.

    Proposals with the same commands merge into one candidate, which keeps the
    first proposal's turn and why. That kept turn is the one executed and
    trained, so it is the one checked for ``leak_terms``; with a judge it
    carries that turn's check as ``leak_judge``.
    """
    proposals: dict[tuple, dict] = {}
    for found in samples:
        if found is None:
            continue
        turn = found.assistant_turn
        key = commands_of(turn)
        if key in proposals:
            proposals[key]["multiplicity"] += 1
        else:
            proposals[key] = {"assistant_turn": turn, "why": found.why, "multiplicity": 1}
            if judged:
                proposals[key]["leak_judge"] = found.turn_check
    for candidate in proposals.values():
        candidate["leak_terms"] = (
            None
            if facts.terms is None
            else leak_terms(candidate["assistant_turn"], facts.terms, facts.visible)
        )
    return list(proposals.values())


class _Review:
    """One review job's calls: which can start, what each returned, and the points they make."""

    def __init__(
        self,
        spec: ReviewSpec,
        failed: list[Attempt],
        directory: Path,
        transport_factory: Callable[..., Any],
        judge_factory: Callable[[dict], Any] | None = None,
    ):
        self.spec, self.failed, self.directory = spec, failed, directory
        self.transport_factory, self.judge_factory = transport_factory, judge_factory
        self.transports: list = []
        #: The first failed call in ``Call`` order, once one has failed: no
        #: call after it starts, and those running are stopped.
        self.first_failed: Call | None = None
        self.analyses: dict[str, dict] = {}
        self.points: dict[Call, list[tuple[BranchChoice, LeakFacts | None]]] = {}
        self.samples: dict[Call, BranchChoice | None] = {}

    def run(self) -> list[dict]:
        """Makes every call, each once what it reads is in and a slot is free; returns the points.

        Calls start only from this thread, in ``Call`` order, at most
        ``max_concurrent_requests`` at a time, and each runs on a thread of
        its own without waiting for another, so no bound can deadlock. After
        a failure only calls before it in that order start; every running
        call is waited for, and then the first failure is raised with the
        others noted on it. Interrupted -- the worker stops its child with
        SIGTERM, raised here as KeyboardInterrupt -- the review stops every
        call and leaves at once, since a request in flight may last
        ``timeout_s``; failures already known are noted on the interrupt and
        logged.
        """
        slots = self.spec.max_concurrent_requests
        ready = [Call(ANALYSIS, position) for position in range(1, len(self.failed) + 1)]
        heapq.heapify(ready)
        running: dict[Call, tuple[threading.Thread, _Stop]] = {}
        returned: queue.SimpleQueue = queue.SimpleQueue()
        failures: list[tuple[Call, BaseException]] = []
        try:
            while True:
                while ready and len(running) < slots and self._may_start(ready[0]):
                    call = heapq.heappop(ready)
                    try:
                        self._launch(call, running, returned)
                    except Exception as error:
                        self._fail(failures, running, call, error)
                if not running:
                    break
                call, value, error = returned.get()
                running.pop(call)[0].join()
                if error is None:
                    try:
                        successors = self._finished(call, value)
                    except Exception as raised:
                        error = raised
                    else:
                        for successor in successors:
                            heapq.heappush(ready, successor)
                if error is not None:
                    self._fail(failures, running, call, error)
        except BaseException as error:
            interrupted = not isinstance(error, Exception)
            for _, stop in running.values():
                stop(
                    "the review was interrupted"
                    if interrupted
                    else f"the review failed: {type(error).__name__}: {error}"
                )
            self._note_lost(error, failures)
            if not interrupted:
                for thread, _ in running.values():
                    thread.join()
            raise
        if failures:
            raise self._first(failures)
        return self._selected()

    def _launch(self, call: Call, running: dict, returned: queue.SimpleQueue) -> None:
        """Starts the call on a daemon thread, which puts ``(call, value, error)`` on ``returned``."""
        stop = _Stop()
        work = self._start(call, stop)

        def target() -> None:
            try:
                value = work()
            except BaseException as error:
                returned.put((call, None, error))
            else:
                returned.put((call, value, None))

        thread = threading.Thread(target=target, name=f"sprout-review-{call.name}", daemon=True)
        # Registered first, so an interrupt that lands while the thread starts still stops it.
        running[call] = (thread, stop)
        try:
            thread.start()
        except Exception:
            del running[call]
            raise

    def _may_start(self, call: Call) -> bool:
        return self.first_failed is None or call < self.first_failed

    def _fail(self, failures: list, running: dict, call: Call, error: BaseException) -> None:
        """Records a failure; the first in ``Call`` order so far stops every running call after it."""
        failures.append((call, error))
        if isinstance(error, RequestStopped) or (self.first_failed is not None and self.first_failed < call):
            return
        self.first_failed = call
        for other, (_, stop) in running.items():
            if call < other:
                stop(f"review call {call.name} failed")

    @staticmethod
    def _real(failures: list[tuple[Call, BaseException]]) -> list[tuple[Call, BaseException]]:
        """The failures in ``Call`` order, but for calls that failed because the review stopped them."""
        return sorted(
            (failure for failure in failures if not isinstance(failure[1], RequestStopped)),
            key=lambda failure: failure[0],
        )

    @classmethod
    def _first(cls, failures: list[tuple[Call, BaseException]]) -> BaseException:
        """The error of the first failed call in ``Call`` order, the others noted on it.

        A call told to stop failed only because of another call or an
        interrupt, so it ranks last and is not noted.
        """
        real = cls._real(failures)
        (call, error), others = (real[0], real[1:]) if real else (min(failures, key=lambda f: f[0]), [])
        error.add_note(f"in review call {call.name}")
        for other, other_error in others:
            error.add_note(
                f"review call {other.name} failed too: {type(other_error).__name__}: {other_error}"
            )
        return error

    @classmethod
    def _note_lost(cls, error: BaseException, failures: list[tuple[Call, BaseException]]) -> None:
        """The failures an interrupt (or a fault of the review's own) cuts off, noted on it and logged."""
        for call, failure in cls._real(failures):
            error.add_note(f"review call {call.name} had failed: {type(failure).__name__}: {failure}")
            logging.getLogger(__name__).error(
                "Review call %s had failed when the review was cut off", call.name, exc_info=failure
            )

    def _transport(self, call: Call, stop: _Stop) -> Callable[[str, str], str]:
        """A transport of the call's own, numbering its receipts in
        ``reviews/<name>/``, that sends nothing once the call is stopped."""
        spec = self.spec
        transport = self.transport_factory(
            self.directory / "reviews" / call.name,
            model=spec.model,
            model_endpoint=spec.model_endpoint,
            api_key_env=spec.api_key_env,
            max_output_tokens=spec.max_output_tokens,
            timeout_s=spec.timeout_s,
            reasoning_effort=spec.reasoning_effort,
            temperature=spec.temperature,
            stop=stop.event,
        )
        self.transports.append(transport)

        def request(model: str, prompt: str) -> str:
            if not stop.event.is_set():
                try:
                    return transport(model, prompt)
                except RequestStopped:
                    if not stop.event.is_set():
                        raise
            raise ReviewStopped(f"review call {call.name} stopped: {stop.reason}")

        return request

    def _start(self, call: Call, stop: _Stop) -> Callable[[], Any]:
        """The call's work, for its thread. Its inputs are taken here, on the
        one thread that records results, so the calls share nothing that changes."""
        transport = self._transport(call, stop)
        if call.stage == ANALYSIS:
            return partial(_analysis, self.spec, transport, self.failed[call.attempt - 1])
        plan_path = self.directory / f"plan-{call.name}.json"
        if call.stage == POINTS:
            scope = [self.failed[call.attempt - 1]] if call.attempt else self.failed
            return partial(
                _point_review,
                self.spec,
                transport,
                scope,
                {attempt.name: self.analyses[attempt.name] for attempt in scope},
                plan_path,
            )
        choice, facts = self.points[Call(POINTS, call.attempt)][call.point - 1]
        return partial(
            _candidate,
            self.spec,
            transport,
            choice,
            self.analyses[choice.base.name],
            plan_path,
            facts,
            self.judge_factory,
        )

    def _finished(self, call: Call, value: Any) -> list[Call]:
        """Records what a call returned; returns the calls that can start now."""
        if call.stage == ANALYSIS:
            self.analyses[self.failed[call.attempt - 1].name] = value
            if self.spec.points_per_attempt:
                return [Call(POINTS, call.attempt)]
            # The one point review reads every analysis.
            return [Call(POINTS, 0)] if len(self.analyses) == len(self.failed) else []
        if call.stage == POINTS:
            self.points[call] = value
            if self.spec.guidance != "assistant-turn":
                return []
            return [
                Call(CANDIDATE, call.attempt, point, sample)
                for point in range(1, len(value) + 1)
                for sample in range(1, self.spec.candidates + 1)
            ]
        self.samples[call] = value
        return []

    def _selected(self) -> list[dict]:
        """The points in failed-attempt and choice order, each with its candidates."""
        spec = self.spec
        point_reviews = (
            [Call(POINTS, position) for position in range(1, len(self.failed) + 1)]
            if spec.points_per_attempt
            else [Call(POINTS, 0)]
        )
        selected = []
        for point_review in point_reviews:
            for point, (choice, leak) in enumerate(self.points[point_review], 1):
                parent = next(p for p in spec.parents if p["name"] == choice.base.name)
                entry = {
                    "base": choice.base.name,
                    "job_id": parent["job_id"],
                    "attempt_id": parent["attempt_id"],
                    "step": choice.checkpoint.step,
                    "snapshot_id": choice.checkpoint.snapshot_id,
                    "conversation_cut": choice.cut,
                    "why": choice.why,
                    "candidates": [],
                }
                if spec.guidance == "assistant-turn":
                    # Stage 2: the independent proposals for this one point.
                    samples = [
                        self.samples[Call(CANDIDATE, point_review.attempt, point, sample)]
                        for sample in range(1, spec.candidates + 1)
                    ]
                    entry["candidates"] = _merge_candidates(samples, leak, spec.leak_judge is not None)
                    entry["samples"] = spec.candidates
                selected.append(entry)
        return selected

    def reports(self) -> dict:
        """Every analysis, in failed-attempt order."""
        return {attempt.name: self.analyses[attempt.name] for attempt in self.failed}

    def requests(self) -> int:
        """How many requests the calls made, retries included. A transport the
        factory handed to several calls counts once."""
        return sum(transport.sequence for transport in {id(t): t for t in self.transports}.values())


def run_review(
    spec: dict,
    directory: Path,
    *,
    transport_factory: Callable[..., Any] = ReviewTransport,
    judge_factory: Callable[[dict], Any] | None = None,
) -> dict:
    """The review job. ``transport_factory`` is called once per call (``Call``),
    with the directory for that call's receipts, the model settings and
    ``stop``, an event set once the call's requests are no longer wanted;
    ``judge_factory`` (default :func:`jev_judge`) once per candidate call when
    the spec configures ``leak_judge``."""
    review_spec = ReviewSpec(**spec)
    review_spec.validate()
    if not review_spec.artifact_root:
        raise ValueError("The worker must supply artifact_root to a review")
    if review_spec.api_key_env is None:
        raise ValueError("A review needs api_key_env, the worker-side variable holding the endpoint key")
    if review_spec.leak_judge is not None and judge_factory is None:
        judge_key(review_spec.leak_judge)  # before any model call is paid for
    root = Path(review_spec.artifact_root)
    attempts = [load_attempt(root, parent) for parent in review_spec.parents]
    failed = [attempt for attempt in attempts if not attempt.grade.resolved]
    if not failed:
        return {
            "status": "completed",
            "guidance": review_spec.guidance,
            "points": [],
            "reports": {},
            "reason": "every attempt resolved the task; nothing to repair",
        }
    calls = _Review(review_spec, failed, directory, transport_factory, judge_factory or jev_judge)
    points = calls.run()
    return {
        "status": "completed",
        "guidance": review_spec.guidance,
        "points": points,
        "reports": calls.reports(),
        "analysed": [attempt.name for attempt in failed],
        "review_requests": calls.requests(),
    }

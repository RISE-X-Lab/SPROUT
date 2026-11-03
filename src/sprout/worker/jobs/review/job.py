"""The review job: hindsight over the finished attempts of one task, as a Run Store worker runs it.

The model reads each failed attempt (its journal, native history and grade, all
under ``artifact_root`` on this host) as the analyst and the reviewer of
``worker.jobs.review.branching``, and picks branch points in it. The spec's
``guidance`` says what the review writes at each point: ``candidates``
independent repair turns, each with its ``leak_terms`` (``assistant-turn``), a
hint (``user-hint``), or nothing (``none``). The job runs nothing in a sandbox.
Its caller turns the points into branch jobs.

The calls run side by side wherever no call needs another's answer, at most
``max_concurrent_requests`` at a time. The result is in failed-attempt, point
and sample order, whatever the order in which the calls finish. Each call has
its receipts directory ``reviews/<name>/``, and each call but an analysis has its
plan file ``plan-<name>.json``. A call that raises fails the review, with the
error of the first failed call in ``Call`` order. An interrupted review stops
every call and raises the interrupt at once. The Review section of
docs/RUNSTORE.md gives the details.
"""

from __future__ import annotations

import heapq
import json
import logging
import os
import queue
import threading
from collections.abc import Callable
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any

from sprout.contract.agents.history import training_messages
from sprout.contract.messages import clean_messages
from sprout.runstore.specs import ReviewSpec
from sprout.worker.agents.history import available_branch_points
from sprout.worker.jobs.review.branching import (
    Attempt,
    BranchChoice,
    Grade,
    analyse,
    outcome_from_journal,
    review,
    selected_prefix,
)
from sprout.worker.jobs.review.leak.judge import (
    REVIEW_ATTEMPTS,
    REVIEW_TIMEOUT_S,
    HiddenText,
    JevLeakJudge,
    check_turn,
)
from sprout.worker.jobs.review.leak.terms import hidden_terms_of, leak_terms, visible_text
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, Prompts, load_prompts
from sprout.worker.jobs.review.transport import RequestStopped, ReviewTransport


def grade_of(record: dict[str, Any]) -> Grade:
    """The ``Grade`` that the analyst reads, from ``record``, the result of a grade job.

    The analyst reads the patch, the progress, the reason, the test statuses, the test log and
    the error output. The hidden-test facts (``fail_to_pass``, ``test_patch_paths``,
    ``touched_test_paths`` and the ``test_patch`` path) never reach the analyst."""
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


def load_attempt(root: Path, parent: dict[str, Any]) -> Attempt:
    """The attempt that ``parent`` names, from its directory under ``root``.

    Raises ValueError when the attempt has no journal."""
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


def commands_of(turn: dict[str, Any]) -> tuple[str, ...]:
    """The commands of the tool calls of ``turn``, in call order."""
    return tuple(json.loads(call["function"]["arguments"])["command"] for call in turn["tool_calls"])


#: The stages of a review, in the order their calls start when they compete for a slot.
ANALYSIS, POINTS, CANDIDATE = 0, 1, 2


@dataclass(frozen=True, order=True)
class Call:
    """One stream of model requests in a review: an analysis, a point review, or a candidate sample.

    Calls compare in the order that a review with one slot makes them: by stage, then by failed
    attempt, point and sample. A review that fails raises the error of the first failed call in
    that order."""

    stage: int
    #: The failed attempt's position, from 1. It is 0 for the one point review over every
    #: failed attempt, and for the samples at its points.
    attempt: int
    #: The point's position among its point review's choices, and the sample's, from 1.
    point: int = 0
    sample: int = 0

    @property
    def name(self) -> str:
        """The call's name: ``reviews/<name>/`` holds its receipts, and ``plan-<name>.json`` is its plan.

        An analysis has no plan file."""
        if self.stage == ANALYSIS:
            return f"analyst-{self.attempt}"
        if self.stage == POINTS:
            return f"points-{self.attempt}" if self.attempt else "points"
        point = f"{self.attempt}.{self.point}" if self.attempt else str(self.point)
        return f"point{point}-candidate{self.sample}"


class ReviewStopped(RequestStopped):
    """Raised to a call that asks the model for more after the review stopped it, with the reason.

    The reason is a failed call before it, or an interrupt. The review never raises it while
    another error is known."""


class _Stop:
    """One call's stop. Once it is set, with its reason, the call asks the model for nothing more.

    ``event`` is the stop of the call's transport, so not even a retry inside the transport is sent."""

    def __init__(self) -> None:
        self.event = threading.Event()
        self.reason = ""

    def __call__(self, reason: str) -> None:
        if not self.event.is_set():
            self.reason = reason
            self.event.set()


#: The calls that run, each with its thread and its stop.
_Running = dict[Call, tuple[threading.Thread, _Stop]]
#: What a call's thread hands back: the call, its value or None, and its error or None.
_Returned = queue.SimpleQueue[tuple[Call, Any, BaseException | None]]


def _analysis(
    spec: ReviewSpec, transport: _CallTransport, attempt: Attempt, prompts: Prompts
) -> dict[str, Any]:
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
        prompts=prompts,
    )


@dataclass(frozen=True)
class LeakFacts:
    """What the leak checks of a point's candidates read.

    ``terms`` are the hidden terms of the parent's grade, None without its facts. ``visible`` is
    h_s as text. ``hidden`` is what the Jev judge reads, None without a judge or a test patch."""

    terms: frozenset[str] | None
    visible: str
    hidden: HiddenText | None


def _point_review(
    spec: ReviewSpec,
    transport: _CallTransport,
    scope: list[Attempt],
    reports: dict[str, dict[str, Any]],
    plan_path: Path,
    prompts: Prompts,
) -> list[tuple[BranchChoice, LeakFacts | None]]:
    """The points picked in ``scope``, each with what the leak checks of its candidates read.

    The leak facts are None unless the guidance is ``assistant-turn``. A ``user-hint`` review
    writes the hint at each point here. Any other review picks the points alone."""
    choices = review(
        transport,
        spec.model,
        spec.problem,
        scope,
        reports,
        round_no=spec.round_no,
        width=spec.points,
        plan_path=plan_path,
        guidance_mode="user-hint" if spec.guidance == "user-hint" else "none",
        count_mode=spec.count_mode,
        reviewer_max_attempts=spec.max_attempts,
        prompt_chars=spec.prompt_chars,
        prompts=prompts,
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
    transport: _CallTransport,
    choice: BranchChoice,
    report: dict[str, Any],
    plan_path: Path,
    facts: LeakFacts,
    judge_factory: Callable[[dict[str, Any]], JevLeakJudge],
    prompts: Prompts,
) -> BranchChoice | None:
    """One candidate sample: the ``assistant-turn`` review at the point, or None when it chose nothing.

    With ``leak_judge`` set, ``leak.judge.check_turn`` checks every turn that the reviewer writes
    before the plan is taken. A flagged turn goes back, ``leak_judge.resamples`` times at most.
    The turn taken keeps its check either way."""
    turn_check: Callable[[BranchChoice], dict[str, Any]] | None = None
    judge: JevLeakJudge | None = None
    if spec.leak_judge is not None:
        judge = judge_factory(spec.leak_judge)

        def turn_check(proposal: BranchChoice) -> dict[str, Any]:
            # ``review`` asks only about a choice with a turn.
            assert proposal.assistant_turn is not None
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
            prompts=prompts,
        )
    finally:
        if judge is not None and hasattr(judge, "close"):
            judge.close()
    return found[0] if found else None


#: How many times a flagged turn goes back to the reviewer when ``leak_judge`` names no ``resamples``.
LEAK_RESAMPLES = 2


def judge_key(config: dict[str, Any]) -> str:
    """The TypeSafe key that a ``leak_judge`` config names, from the worker's environment.

    Raises ValueError when the variable is unset or blank."""
    variable = config.get("api_key_env", "TYPESAFE_API_KEY")
    key = os.environ.get(variable, "").strip()
    if not key:
        raise ValueError(f"leak_judge needs the TypeSafe key in the worker's ${variable}")
    return key


def jev_judge(config: dict[str, Any]) -> JevLeakJudge:
    """The Jev judge that a review's ``leak_judge`` config names, with the key from ``judge_key``."""
    # The review runs under its own deadline: a judge that cannot answer gives up
    # in about a minute and a half per request, and the turn stays unchecked.
    return JevLeakJudge(
        judge_key(config),
        timeout_s=REVIEW_TIMEOUT_S,
        max_attempts=REVIEW_ATTEMPTS,
        **{name: config[name] for name in ("model", "threshold") if name in config},
    )


def _merge_candidates(
    samples: list[BranchChoice | None], facts: LeakFacts, judged: bool
) -> list[dict[str, Any]]:
    """A point's samples, in sample order, as its candidates.

    Proposals with the same commands merge into one candidate, which keeps the first proposal's
    turn and ``why``. That turn is the one executed and trained, so ``leak_terms`` checks it, and
    with a judge, the candidate's ``leak_judge`` is that turn's check."""
    proposals: dict[tuple[str, ...], dict[str, Any]] = {}
    for found in samples:
        if found is None:
            continue
        turn = found.assistant_turn
        # An assistant-turn review gives every choice its turn.
        assert turn is not None
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


class _CallTransport:
    """One call's requests: through its transport, and none once the call is stopped.

    With ``structured``, the analyst and reviewer ask through ``json`` for schema-constrained output.
    A request after the stop raises ``ReviewStopped``."""

    def __init__(self, transport: ReviewTransport, *, name: str, stop: _Stop, structured: bool) -> None:
        self._transport, self._name, self._stop = transport, name, stop
        self.supports_json_schema = structured

    def __call__(self, model: str, prompt: str) -> str:
        return self._send(lambda: self._transport(model, prompt))

    def json(self, model: str, prompt: str, *, schema: dict[str, Any], name: str) -> str:
        return self._send(lambda: self._transport.json(model, prompt, schema=schema, name=name))

    def json_overhead(self, schema: dict[str, Any], name: str) -> int:
        return self._transport.json_overhead(schema, name)

    def _send(self, request: Callable[[], str]) -> str:
        if not self._stop.event.is_set():
            try:
                return request()
            except RequestStopped:
                if not self._stop.event.is_set():
                    raise
        raise ReviewStopped(f"review call {self._name} stopped: {self._stop.reason}")


class _Review:
    """One review job's calls: which can start, what each returned, and the points they make."""

    def __init__(
        self,
        spec: ReviewSpec,
        failed: list[Attempt],
        directory: Path,
        transport_factory: Callable[..., ReviewTransport],
        judge_factory: Callable[[dict[str, Any]], JevLeakJudge] = jev_judge,
        prompts: Prompts = DEFAULT_PROMPTS,
    ) -> None:
        self.spec, self.failed, self.directory = spec, failed, directory
        self.transport_factory, self.judge_factory = transport_factory, judge_factory
        self.prompts = prompts
        self.transports: list[ReviewTransport] = []
        #: The first failed call in ``Call`` order, once one has failed: no
        #: call after it starts, and those running are stopped.
        self.first_failed: Call | None = None
        self.analyses: dict[str, dict[str, Any]] = {}
        self.points: dict[Call, list[tuple[BranchChoice, LeakFacts | None]]] = {}
        self.samples: dict[Call, BranchChoice | None] = {}

    def run(self) -> list[dict[str, Any]]:
        """Makes every call once its inputs are in and a slot is free, and returns the points.

        After a failure, only the calls before it in ``Call`` order start. Once the running calls
        return, the first failure is raised with the others noted on it. An interrupt stops every
        call and is raised at once, with the failures known so far noted on it and logged."""
        slots = self.spec.max_concurrent_requests
        ready = [Call(ANALYSIS, position) for position in range(1, len(self.failed) + 1)]
        heapq.heapify(ready)
        running: _Running = {}
        returned: _Returned = queue.SimpleQueue()
        failures: list[tuple[Call, BaseException]] = []
        try:
            # Calls start only from this thread, in ``Call`` order. Each runs on a thread of its
            # own and waits for no other call, so the bound on slots cannot deadlock.
            while True:
                while ready and len(running) < slots and self._may_start(ready[0]):
                    call = heapq.heappop(ready)
                    try:
                        self._launch(call, running, returned)
                    except Exception as error:
                        self._fail(failures, running, call, error)
                if not running:
                    break
                call, value, failure = returned.get()
                running.pop(call)[0].join()
                if failure is None:
                    try:
                        successors = self._finished(call, value)
                    except Exception as raised:
                        failure = raised
                    else:
                        for successor in successors:
                            heapq.heappush(ready, successor)
                if failure is not None:
                    self._fail(failures, running, call, failure)
        except BaseException as error:
            interrupted = not isinstance(error, Exception)
            for _, stop in running.values():
                stop(
                    "the review was interrupted"
                    if interrupted
                    else f"the review failed: {type(error).__name__}: {error}"
                )
            self._note_lost(error, failures)
            # An interrupt (the worker's SIGTERM, raised as KeyboardInterrupt) waits for no call:
            # a request in flight can last ``timeout_s``.
            if not interrupted:
                for thread, _ in running.values():
                    thread.join()
            raise
        if failures:
            raise self._first(failures)
        return self._selected()

    def _launch(self, call: Call, running: _Running, returned: _Returned) -> None:
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

    def _fail(
        self, failures: list[tuple[Call, BaseException]], running: _Running, call: Call, error: BaseException
    ) -> None:
        """Records a failure. The first in ``Call`` order so far stops every running call after it."""
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

        A call that the review stopped failed only because of another call or an interrupt. Its
        error ranks last and is not noted."""
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
        """Notes on ``error``, an interrupt or a fault of the review's own, each failure that it cuts off.

        Each of those failures is logged too."""
        for call, failure in cls._real(failures):
            error.add_note(f"review call {call.name} had failed: {type(failure).__name__}: {failure}")
            logging.getLogger(__name__).error(
                "Review call %s had failed when the review was cut off", call.name, exc_info=failure
            )

    def _transport(self, call: Call, stop: _Stop) -> _CallTransport:
        """A transport of the call's own, numbering its receipts in
        ``reviews/<name>/``, that sends nothing once the call is stopped."""
        spec = self.spec
        transport = self.transport_factory(
            self.directory / "reviews" / call.name,
            model=spec.model,
            model_endpoint=spec.model_endpoint,
            api_key_env=spec.api_key_env,
            max_output_tokens=spec.max_output_tokens,
            timeout_s=spec.request_timeout_s or spec.timeout_s,
            reasoning_effort=spec.reasoning_effort,
            temperature=spec.temperature,
            stop=stop.event,
            json_directive=self.prompts.transport_json_directive,
            supports_json_schema=spec.structured,
        )
        self.transports.append(transport)
        return _CallTransport(transport, name=call.name, stop=stop, structured=spec.structured)

    def _start(self, call: Call, stop: _Stop) -> Callable[[], Any]:
        """The call's work, for its thread. Its inputs are taken here, on the
        one thread that records results, so the calls share nothing that changes."""
        transport = self._transport(call, stop)
        if call.stage == ANALYSIS:
            return partial(_analysis, self.spec, transport, self.failed[call.attempt - 1], self.prompts)
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
                self.prompts,
            )
        choice, facts = self.points[Call(POINTS, call.attempt)][call.point - 1]
        # Candidate calls exist only under assistant-turn guidance, whose point review sets the facts.
        assert facts is not None
        return partial(
            _candidate,
            self.spec,
            transport,
            choice,
            self.analyses[choice.base.name],
            plan_path,
            facts,
            self.judge_factory,
            self.prompts,
        )

    def _finished(self, call: Call, value: Any) -> list[Call]:
        """Records what a call returned, and returns the calls that can start now."""
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

    def _selected(self) -> list[dict[str, Any]]:
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
                entry: dict[str, Any] = {
                    "base": choice.base.name,
                    "job_id": parent["job_id"],
                    "attempt_id": parent["attempt_id"],
                    "run_name": choice.run_name,
                    "step": choice.checkpoint.step,
                    "snapshot_id": choice.checkpoint.snapshot_id,
                    "conversation_cut": choice.cut,
                    "why": choice.why,
                    "candidates": [],
                }
                if spec.guidance == "user-hint":
                    entry["hint"] = choice.hint
                if spec.guidance == "assistant-turn":
                    # The independent proposals for this one point. Its point review set ``leak``.
                    assert leak is not None
                    samples = [
                        self.samples[Call(CANDIDATE, point_review.attempt, point, sample)]
                        for sample in range(1, spec.candidates + 1)
                    ]
                    entry["candidates"] = _merge_candidates(samples, leak, spec.leak_judge is not None)
                    entry["samples"] = spec.candidates
                selected.append(entry)
        return selected

    def reports(self) -> dict[str, dict[str, Any]]:
        """Every analysis, in failed-attempt order."""
        return {attempt.name: self.analyses[attempt.name] for attempt in self.failed}

    def requests(self) -> int:
        """How many requests the calls made, retries included. A transport the
        factory handed to several calls counts once."""
        return sum(transport.sequence for transport in {id(t): t for t in self.transports}.values())


def _prompts(record: dict[str, str] | None) -> Prompts:
    """The prompts a review reads: the packaged file, or the one ``record`` names.

    Raises ValueError when the named file no longer has the sha256 that ``record`` gives."""
    if record is None:
        return DEFAULT_PROMPTS
    prompts = load_prompts(Path(record["path"]))
    if prompts.sha256 != record["sha256"]:
        raise ValueError(f"{record['path']}: the prompts file changed after the review was submitted")
    return prompts


def run_review(
    spec: dict[str, Any],
    directory: Path,
    *,
    transport_factory: Callable[..., ReviewTransport] = ReviewTransport,
    judge_factory: Callable[[dict[str, Any]], JevLeakJudge] | None = None,
) -> dict[str, Any]:
    """The result of the review job ``spec``, which writes its receipts and plans under ``directory``.

    ``transport_factory`` makes one transport for each ``Call``, with the call's receipts directory,
    the model settings and ``stop``, which is set once the call's requests are no longer wanted.
    ``judge_factory`` (default ``jev_judge``) makes a judge for each candidate call under ``leak_judge``."""
    review_spec = ReviewSpec(**spec)
    review_spec.validate()
    prompts = _prompts(review_spec.prompts)
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
    calls = _Review(review_spec, failed, directory, transport_factory, judge_factory or jev_judge, prompts)
    points = calls.run()
    return {
        "status": "completed",
        "guidance": review_spec.guidance,
        "points": points,
        "reports": calls.reports(),
        "analysed": [attempt.name for attempt in failed],
        "review_requests": calls.requests(),
    }

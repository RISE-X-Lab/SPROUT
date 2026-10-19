"""A fake review model behind the transport factory of ``run_review``, and the review jobs that it serves.

``run_review`` makes one transport for each call, with the call's receipts directory ``reviews/<name>``.
``ReviewModel.factory`` makes that transport, and records its settings and its ``stop``. The model
answers each request by the stage that the call's name starts with:

- an analysis (``analyst-1``): ``ANALYSIS``
- a point review (``points``, ``points-1``): a point at each step that ``points`` lists for the call
- a candidate sample (``point1-candidate2``, ``point1.1-candidate2``): a turn that runs its proposal

``hold(call)`` runs before each answer, so that a test can order the calls. The model ignores the stop.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from tests.worker.jobs.review.recorded import record_attempt

ANALYSIS = {
    "failure_reason": "edited the wrong file",
    "lesson": "read the test first",
    "salvage": "the exploration",
    "branch_candidates": [{"step": 2, "why": "before the edit"}],
}
#: The hidden-test facts of a grade, which the leak checks read and no prompt shows.
HIDDEN = {
    "fail_to_pass": ["tests/test_hidden.py::test_secret_behaviour"],
    "test_patch_paths": ["tests/test_hidden.py"],
}


def parent(name: str, job_id: str, attempt_id: str, **grade: Any) -> dict[str, Any]:
    """A review parent: a failed attempt of 50% progress, unless ``grade`` says otherwise."""
    return {
        "name": name,
        "job_id": job_id,
        "attempt_id": attempt_id,
        "grade": {"resolved": False, "progress": 0.5, **grade},
    }


def review_spec(root: Path, parents: Sequence[dict[str, Any]], **changes: Any) -> dict[str, Any]:
    """A review job over ``parents``, whose attempts are under ``root``: two points, three samples at each."""
    spec = {
        "problem": "Fix the failing test.",
        "model": "fixture",
        "model_endpoint": "http://127.0.0.1:1",
        "api_key_env": "REVIEW_KEY",
        "guidance": "assistant-turn",
        "points": 2,
        "candidates": 3,
        "artifact_root": str(root),
        "parents": list(parents),
    }
    spec.update(changes)
    return spec


def failed_roots(root: Path, count: int, **grade: Any) -> list[dict[str, Any]]:
    """``count`` failed roots ``root-N``, each recorded under ``root/job-N/attempt-N``."""
    parents = []
    for number in range(1, count + 1):
        record_attempt(root / f"job-{number}" / f"attempt-{number}", run=f"root-{number}")
        parents.append(parent(f"root-{number}", f"job-{number}", f"attempt-{number}", **grade))
    return parents


def stage_of(call: str) -> str:
    """``analyst``, ``points`` or ``point``, which names a candidate sample."""
    match = re.match(r"[a-z]+", call)
    assert match is not None
    return match.group(0)


class ReviewModel:
    """Answers the calls of one review by their names, and records what each call sent and was given.

    ``points`` maps each point review to its points, as ``(base, step)``. ``proposals`` is the command of
    every sample, a map from a sample's name to its command, or a function of the name. ``fail`` maps a
    call to the error of each of its requests, and ``replies`` to its first replies, a text or an error."""

    def __init__(
        self,
        *,
        points: Mapping[str, Sequence[tuple[str, int]]],
        proposals: str | Mapping[str, str] | Callable[[str], str] = "pytest -q",
        hint: str | None = None,
        hold: Callable[[str], None] | None = None,
        fail: Mapping[str, BaseException] | None = None,
        replies: Mapping[str, Sequence[str | BaseException]] | None = None,
    ) -> None:
        self.points, self.proposals, self.hint, self.hold = points, proposals, hint, hold
        self.fail = dict(fail or {})
        self.replies = {call: list(texts) for call, texts in (replies or {}).items()}
        self.lock = threading.Lock()
        #: ``(call, kind, prompt)`` for each request, as it arrives. ``kind`` is ``text`` or ``json``.
        self.requests: list[tuple[str, str, str]] = []
        #: The calls in the order that the review made their transports, and the settings of each.
        self.made: list[str] = []
        self.settings: dict[str, dict[str, Any]] = {}
        #: The calls in the order that their requests returned or raised.
        self.answered: list[str] = []
        #: The thread that each call's requests run on.
        self.threads: dict[str, threading.Thread] = {}
        self.in_flight = self.most_in_flight = 0
        #: ``("made", call)`` and ``("finished", call)`` in the order that they happened.
        self.log: list[tuple[str, str]] = []
        self._events: dict[tuple[str, str], threading.Event] = {}

    @property
    def calls(self) -> list[str]:
        return [call for call, _, _ in self.requests]

    def prompts_of(self, call: str) -> list[str]:
        return [prompt for name, _, prompt in self.requests if name == call]

    def arrived(self, call: str) -> threading.Event:
        """Set once a request of ``call`` arrived: the call is in flight."""
        return self._event("arrived", call)

    def finished(self, call: str) -> threading.Event:
        """Set once a request of ``call`` returned or raised."""
        return self._event("finished", call)

    def made_transport(self, call: str) -> threading.Event:
        """Set once the review made the transport of ``call``, which it does when the call starts."""
        return self._event("made", call)

    def stop_of(self, call: str) -> threading.Event:
        """The stop that the review gave the transport of ``call``, once the call has started."""
        assert self.made_transport(call).wait(30), f"{call} did not start"
        stop: threading.Event = self.settings[call]["stop"]
        return stop

    def join(self, call: str) -> None:
        """Return once the thread of ``call`` has ended, after its first request arrived."""
        assert self.arrived(call).wait(30), f"{call} sent no request"
        self.threads[call].join(30)
        assert not self.threads[call].is_alive(), f"the thread of {call} did not end"

    def factory(self, directory: Path, **settings: Any) -> CallTransport:
        call = Path(directory).name
        with self.lock:
            self.made.append(call)
            self.log.append(("made", call))
            self.settings[call] = settings
        self.made_transport(call).set()
        return CallTransport(self, call)

    def request(self, call: str, kind: str, prompt: str) -> str:
        with self.lock:
            self.requests.append((call, kind, prompt))
            self.threads[call] = threading.current_thread()
            self.in_flight += 1
            self.most_in_flight = max(self.most_in_flight, self.in_flight)
        self.arrived(call).set()
        try:
            if self.hold is not None:
                self.hold(call)
            if call in self.fail:
                raise self.fail[call]
            if self.replies.get(call):
                reply = self.replies[call].pop(0)
                if isinstance(reply, BaseException):
                    raise reply
                return reply
            return self._answer(call)
        finally:
            with self.lock:
                self.in_flight -= 1
                self.answered.append(call)
                self.log.append(("finished", call))
            self.finished(call).set()

    def _answer(self, call: str) -> str:
        if stage_of(call) == "analyst":
            return json.dumps(ANALYSIS)
        if stage_of(call) == "points":
            branches = [
                {
                    "name": "point-%d" % step,
                    "base": base,
                    "branch_step": step,
                    "why": "a decision",
                    **self._hint(),
                }
                for base, step in self.points[call]
            ]
            return _fenced({"synthesis": "points", "branches": branches})
        base, step = self._point_of(call)
        return repair_plan(base, step, self._proposal(call))

    def _hint(self) -> dict[str, str]:
        return {} if self.hint is None else {"hint": self.hint}

    def _point_of(self, call: str) -> tuple[str, int]:
        match = re.fullmatch(r"point(?:(\d+)\.)?(\d+)-candidate\d+", call)
        assert match is not None, call
        review = "points" if match.group(1) is None else "points-" + match.group(1)
        return tuple(self.points[review][int(match.group(2)) - 1])  # type: ignore[return-value]

    def _proposal(self, call: str) -> str:
        if isinstance(self.proposals, str):
            return self.proposals
        if callable(self.proposals):
            return self.proposals(call)
        return self.proposals[call]

    def _event(self, kind: str, call: str) -> threading.Event:
        with self.lock:
            return self._events.setdefault((kind, call), threading.Event())


class CallTransport:
    """One call's transport. It counts its requests, as ``ReviewTransport.sequence`` counts its tries."""

    def __init__(self, model: ReviewModel, call: str) -> None:
        self.model, self.call, self.sequence = model, call, 0

    def __call__(self, model: str, prompt: str) -> str:
        self.sequence += 1
        return self.model.request(self.call, "text", prompt)

    def json(self, model: str, prompt: str, *, schema: dict[str, Any], name: str) -> str:
        self.sequence += 1
        return self.model.request(self.call, "json", prompt)

    def json_overhead(self, schema: dict[str, Any], name: str) -> int:
        return 0


def repair_plan(base: str, step: int, command: str) -> str:
    """A sample's reply: one branch at ``base`` and ``step`` whose turn runs ``command``."""
    repair = {"name": "repair", "base": base, "branch_step": step, "why": "check first"}
    repair.update(content="I will check the test before editing.", commands=[command])
    return _fenced({"synthesis": "repair", "branches": [repair]})


def _fenced(plan: dict[str, Any]) -> str:
    return "```branch-plan\n" + json.dumps(plan) + "\n```"

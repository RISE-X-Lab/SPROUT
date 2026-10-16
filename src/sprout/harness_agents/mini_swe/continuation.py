"""What a branch adds after its restored history: one definition for every entry point.

A branch is a closed model turn's snapshot plus mini's native history up to
that turn. ``history.index_mini`` finds both, for a Harbor trial's parent
journal (``reference_at``) and for a Run Store attempt (its recovery points)
alike. What the actor receives after that prefix is the branch's guidance:

``assistant-turn``
    an assistant turn written outside the actor -- text and at least one
    ``bash`` call -- that mini executes through the normal tool path before
    the actor's first request; its real results join the history
``user-hint``
    a user message carrying the hint, as ``render_branch_note`` words it
``none``
    nothing: the actor is queried with the prefix alone

An assistant-turn branch also says how the turn is written into the history,
its ``assistant_turn_rendering``: ``content`` (the default) keeps the text
where the reviewer wrote it; ``reasoning`` moves it into ``reasoning_content``
and leaves ``content`` empty (``render_reasoning``), the shape of a turn a
reasoning policy samples (thinking, then its calls), so training on the turn
does not teach the policy to leave its thinking empty. The RL driver's
repairs use it. The reviewer always writes the first form; only ``prepare``
renders it.

A Harbor branch trial (``harbor_plugin.branching.BranchMini``, what
``methods.evaluate --method sprout`` runs) and a Run Store branch job
(``POST /v1/jobs/{id}/branch``, which the RL driver's branch samples submit)
both turn a ``Continuation`` into mini's options with ``prepare``; neither
keeps its own copy of these rules.
"""

from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from dataclasses import dataclass

from sprout.harness_agents.mini_swe.history import validate_assistant_turn

GUIDANCE = ("assistant-turn", "user-hint", "none")
#: How an assistant-turn branch writes its turn: as given, or ``render_reasoning``.
RENDERINGS = ("content", "reasoning")
FIELDS = frozenset({"guidance", "assistant_turn", "hint", "assistant_turn_rendering"})


@dataclass(frozen=True)
class Continuation:
    guidance: str
    assistant_turn: dict | None = None
    hint: str | None = None
    assistant_turn_rendering: str = "content"

    def __post_init__(self) -> None:
        if self.guidance not in GUIDANCE:
            raise ValueError("Branch guidance must be assistant-turn, user-hint or none")
        if self.guidance == "assistant-turn" and not isinstance(self.assistant_turn, dict):
            raise ValueError("An assistant-turn branch needs its assistant_turn")
        if self.guidance != "assistant-turn" and self.assistant_turn is not None:
            raise ValueError("Only an assistant-turn branch carries an assistant turn")
        if self.guidance == "user-hint" and not (isinstance(self.hint, str) and self.hint.strip()):
            raise ValueError("A user-hint branch needs a non-empty hint")
        if self.guidance != "user-hint" and self.hint is not None:
            raise ValueError("Only a user-hint branch carries a hint")
        if self.assistant_turn_rendering not in RENDERINGS:
            raise ValueError("assistant_turn_rendering must be content or reasoning")
        if self.guidance != "assistant-turn" and self.assistant_turn_rendering != "content":
            raise ValueError("Only an assistant-turn branch has an assistant_turn_rendering")

    @classmethod
    def from_dict(cls, value: object) -> Continuation:
        if not isinstance(value, dict) or set(value) - FIELDS:
            raise ValueError("A continuation is an object with guidance and, for its mode, "
                             "assistant_turn (and optionally assistant_turn_rendering) or hint")
        return cls(value.get("guidance"), deepcopy(value.get("assistant_turn")), value.get("hint"),
                   value.get("assistant_turn_rendering", "content"))

    def to_dict(self) -> dict:
        body: dict = {"guidance": self.guidance}
        if self.assistant_turn is not None:
            body["assistant_turn"] = deepcopy(self.assistant_turn)
        if self.assistant_turn_rendering != "content":
            # The default is never written, so a body (and the request hash
            # and idempotency key derived from it) is what it was before.
            body["assistant_turn_rendering"] = self.assistant_turn_rendering
        if self.hint is not None:
            body["hint"] = self.hint
        return body

    def origin(self) -> dict:
        """What a branch's ``fork.origin`` records about its guidance."""
        record: dict = {"branch_guidance": self.guidance}
        if self.guidance == "assistant-turn":
            record["assistant_turn_source"] = "reviewer"
            if self.assistant_turn_rendering != "content":
                record["assistant_turn_rendering"] = self.assistant_turn_rendering
        if self.guidance == "user-hint":
            record.update(actor_hint=self.hint, hint_delivery="reviewer-direct")
        return record


def render_reasoning(turn: dict) -> dict:
    """The ``reasoning`` rendering of an assistant turn: its text as the turn's reasoning.

    ``content`` becomes ``reasoning_content``, verbatim, beside an empty
    ``content``; the calls are kept as they are. The input is not touched.
    """
    return {"role": "assistant", "content": "", "reasoning_content": turn["content"],
            "tool_calls": deepcopy(turn["tool_calls"])}


def render_branch_note(hint: str, *, commit_required: bool = False) -> str:
    """The user message a ``user-hint`` branch receives: the hint, verbatim."""
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError("branch hint must be a non-empty string")
    lines = [
        hint, "",
        "Continue the task directly, without acknowledging or recapping this message.",
        "Keep reasoning and actions focused on the code and observable behavior.",
        "Check uncertain claims against the actual implementation; do not invent",
        "prior observations or treat a suggested cause as already established.",
    ]
    if commit_required:
        lines.append("Commit your changes when finished.")
    return "\n".join(lines)


def prepare(continuation: Continuation, prefix: Sequence[dict], *, tools: list[dict] | None = None,
            commit_required: bool = False) -> tuple[str, dict]:
    """The prompt mini appends after ``prefix``, and the options it runs the branch with.

    ``prefix`` is the restored native history (``history.load_prefix``). The
    prompt is the rendered note for ``user-hint`` and empty for the other two
    modes, which mini reads as "append no user message". The options go into
    ``RunSpec.extra`` beside ``native_prefix``. An assistant turn is checked
    here, in the form its author writes, against the prefix's tool-call ids
    and, when the caller has it, the tool schema recorded at the branch point,
    and then rendered as the continuation asks; mini checks it once more
    against the tools it actually offers.
    """
    if continuation.guidance == "assistant-turn":
        history = [entry["message"] for entry in prefix if entry.get("type") == "mini.message"]
        turn = validate_assistant_turn(continuation.assistant_turn, history=history, tools=tools)
        if continuation.assistant_turn_rendering == "reasoning":
            turn = render_reasoning(turn)
        return "", {"assistant_turn": turn}
    if continuation.guidance == "none":
        return "", {"resume_without_hint": True}
    return render_branch_note(continuation.hint, commit_required=commit_required), {}

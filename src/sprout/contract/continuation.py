"""What a branch adds after its restored history: ``Continuation`` and ``prepare``, for every entry point.

A branch starts from a closed model turn's snapshot and mini's native history up to that turn.
Its ``guidance`` says what the actor receives after that prefix:

``assistant-turn``
    an assistant turn written outside the actor, with text and at least one ``bash`` call. mini
    runs it through the normal tool path before the actor's first request, and its real results
    join the history.
``user-hint``
    a user message that carries the hint, as ``render_branch_note`` words it.
``none``
    nothing. The actor is queried with the prefix alone.

``assistant_turn_rendering`` says how an assistant turn enters the history. ``content``, the
default, keeps the text where its author wrote it. ``reasoning`` moves the text to
``reasoning_content`` (``render_reasoning``), the shape of a turn that a reasoning policy samples.
Training on that turn then does not teach the policy to leave its thinking empty.
The "Branch" section of ``docs/RUNSTORE.md`` describes the Run Store request that carries it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from sprout.contract.mini_history import validate_assistant_turn

#: The values of ``Continuation.guidance``.
GUIDANCE = ("assistant-turn", "user-hint", "none")
#: How an assistant-turn branch writes its turn: as given, or ``render_reasoning``.
RENDERINGS = ("content", "reasoning")
#: The keys that a continuation object can have.
FIELDS = frozenset({"guidance", "assistant_turn", "hint", "assistant_turn_rendering"})


@dataclass(frozen=True)
class Continuation:
    """The guidance of one branch. Raises ``ValueError`` if the fields do not fit ``guidance``.

    Only an ``assistant-turn`` branch has an ``assistant_turn`` and can set ``assistant_turn_rendering``.
    Only a ``user-hint`` branch has a ``hint``, and the hint is not blank.
    """

    guidance: str
    assistant_turn: dict[str, Any] | None = None
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
        """The continuation in the JSON object ``value``. Raises ``ValueError`` if ``value`` is not one."""
        if not isinstance(value, dict) or set(value) - FIELDS:
            raise ValueError(
                "A continuation is an object with guidance and, for its mode, "
                "assistant_turn (and optionally assistant_turn_rendering) or hint"
            )
        return cls(
            # A missing guidance becomes "", which __post_init__ refuses like any unknown one.
            value.get("guidance", ""),
            deepcopy(value.get("assistant_turn")),
            value.get("hint"),
            value.get("assistant_turn_rendering", "content"),
        )

    def to_dict(self) -> dict[str, Any]:
        """The JSON object of this continuation, the inverse of ``from_dict``."""
        body: dict[str, Any] = {"guidance": self.guidance}
        if self.assistant_turn is not None:
            body["assistant_turn"] = deepcopy(self.assistant_turn)
        if self.assistant_turn_rendering != "content":
            # The default stays out of the body. The request hash and the idempotency key derive
            # from the body, so they stay the same as for a body that never had the field.
            body["assistant_turn_rendering"] = self.assistant_turn_rendering
        if self.hint is not None:
            body["hint"] = self.hint
        return body

    def origin(self) -> dict[str, Any]:
        """What a branch's ``fork.origin`` records about its guidance."""
        record: dict[str, Any] = {"branch_guidance": self.guidance}
        if self.guidance == "assistant-turn":
            record["assistant_turn_source"] = "reviewer"
            if self.assistant_turn_rendering != "content":
                record["assistant_turn_rendering"] = self.assistant_turn_rendering
        if self.guidance == "user-hint":
            record.update(actor_hint=self.hint, hint_delivery="reviewer-direct")
        return record


def render_reasoning(turn: Mapping[str, Any]) -> dict[str, Any]:
    """The ``reasoning`` rendering of an assistant turn: its text as the turn's reasoning.

    ``content`` becomes ``reasoning_content``, verbatim, beside an empty ``content``. The calls stay
    as they are. The function does not change ``turn``.
    """
    return {
        "role": "assistant",
        "content": "",
        "reasoning_content": turn["content"],
        "tool_calls": deepcopy(turn["tool_calls"]),
    }


def render_branch_note(hint: str, *, commit_required: bool = False) -> str:
    """The user message that a ``user-hint`` branch receives: ``hint`` verbatim, then fixed instructions.

    ``commit_required`` adds an instruction to commit. Raises ``ValueError`` if ``hint`` is blank.
    """
    if not isinstance(hint, str) or not hint.strip():
        raise ValueError("branch hint must be a non-empty string")
    lines = [
        hint,
        "",
        "Continue the task directly, without acknowledging or recapping this message.",
        "Keep reasoning and actions focused on the code and observable behavior.",
        "Check uncertain claims against the actual implementation; do not invent",
        "prior observations or treat a suggested cause as already established.",
    ]
    if commit_required:
        lines.append("Commit your changes when finished.")
    return "\n".join(lines)


def prepare(
    continuation: Continuation,
    prefix: Sequence[Mapping[str, Any]],
    *,
    tools: list[dict[str, Any]] | None = None,
    commit_required: bool = False,
) -> tuple[str, dict[str, Any]]:
    """The prompt that mini appends after ``prefix``, and the options for the branch's ``RunSpec.extra``.

    ``prefix`` holds the entries of the restored native history. The prompt is empty unless the guidance
    is ``user-hint``. An assistant turn must pass ``validate_assistant_turn`` before it is rendered.
    """
    # Continuation ties each field to its guidance: assistant_turn to assistant-turn, hint to user-hint.
    if continuation.assistant_turn is not None:
        history = [entry["message"] for entry in prefix if entry.get("type") == "mini.message"]
        turn = validate_assistant_turn(continuation.assistant_turn, history=history, tools=tools)
        if continuation.assistant_turn_rendering == "reasoning":
            turn = render_reasoning(turn)
        return "", {"assistant_turn": turn}
    if continuation.hint is not None:
        return render_branch_note(continuation.hint, commit_required=commit_required), {}
    return "", {"resume_without_hint": True}

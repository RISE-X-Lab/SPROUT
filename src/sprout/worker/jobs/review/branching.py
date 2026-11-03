"""The analyst and the reviewer of a review: what each reads, and how a plan becomes exact branches.

``analyse`` reports on one failed attempt. ``review`` reads the reports and picks branch points.
``prepare_branches`` checks each point of a plan against the parent's recorded history, and
resolves it to a ``BranchChoice``: a checkpoint, a native conversation cut and the guidance.
The functions read finished attempts from their journals, and start no attempt and no sandbox.
The prompts come from ``worker.jobs.review.prompts``. docs/RUNSTORE.md (Review requests)
describes how a request is held to its budget, and the plan file.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import Any, Protocol

from sprout.contract.agents.history import load_prefix, training_messages, validate_assistant_turn
from sprout.worker.agents.history import actor_tools_at, conversation_restore, reference_at
from sprout.worker.checkpoint.rollback import Checkpoint, load_checkpoints
from sprout.worker.jobs.review import structured_review
from sprout.worker.jobs.review.branch_plan import (
    PromptRejected,
    ReviewerPlanError,
    ReviewResponseTruncated,
    TurnRejected,
    extract_branch_plan,
    review_with_feedback,
)
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, Prompts
from sprout.worker.jobs.review.request_size import (
    DEFAULT_PROMPT_CHARS,
    FLOOR,
    PROMPT_HALVINGS,
    clip_text,
    fit_review_prompt,
)
from sprout.worker.run.journal import read_journal
from sprout.worker.run.runner import RunOutcome


@dataclass
class Grade:
    """A benchmark's verdict on one attempt, as Harbor's verifier reported it."""

    resolved: bool = False
    reward: float | None = None
    #: The diff the verifier graded, when the task collected one (model.patch).
    patch: str = ""
    #: The verifier's own words: its result record and the tail of its logs.
    detail: str = ""
    error: str | None = None

    def summary(self) -> str:
        """One line for a prompt: the grading error, or the reward and whether the attempt resolved."""
        if self.error:
            return "GRADING ERROR: %s" % self.error
        return "official Harbor reward=%s; resolved=%s" % (self.reward, self.resolved)


class Transport(Protocol):
    """What ``analyse`` and ``review`` send their requests through, as ``ReviewTransport`` does.

    A call returns the model's reply to ``prompt``. The functions call ``json`` and
    ``json_overhead`` only when the transport's ``supports_json_schema`` is True."""

    def __call__(self, model: str, prompt: str) -> str: ...

    def json(self, model: str, prompt: str, *, schema: dict[str, Any], name: str) -> str: ...

    def json_overhead(self, schema: dict[str, Any], name: str) -> int: ...


BRANCH_GUIDANCE_MODES = ("user-hint", "assistant-turn", "none")
DEFAULT_GUIDANCE = "assistant-turn"
#: The most rollouts that one evaluation may make for a task, the shared first rollout included.
MAX_ROLLOUTS = 128


def require_mini_parent(journal: Path) -> None:
    """Raises ValueError unless ``journal`` has runs, and each of them is a mini-swe-agent run."""
    slots = {e["slot"] for e in read_journal(journal) if e.get("type") == "run.started" and e.get("slot")}
    if slots not in ({"mini-swe-agent"}, {"opencollab-base"}):
        raise ValueError("This branch mode accepts only mini-swe-agent parent trajectories")


def selected_prefix(journal: Path, checkpoint: Checkpoint) -> list[dict[str, Any]]:
    """The native mini history up to ``checkpoint``. Raises ValueError when there is no exact prefix."""
    require_mini_parent(journal)
    session_id = checkpoint.session_ckpt
    reference = None if session_id is None else reference_at(journal, checkpoint.step, session_id)
    if reference is None:
        raise ValueError("No exact mini prefix at the selected checkpoint")
    return load_prefix(reference)


def reviewer_context(journal: Path, checkpoints: Mapping[int, Checkpoint]) -> dict[str, Any]:
    """An attempt's native history for the reviewer, up to its last step in ``checkpoints``.

    ``prefix_message_counts`` maps each of those steps to the number of messages that a branch
    there keeps. Raises ValueError when the history does not cover every step."""
    if not checkpoints:
        return {"messages": [], "prefix_message_counts": {}}
    prefix = selected_prefix(journal, checkpoints[max(checkpoints)])
    closed = {
        e["turn_id"]: e["step"] for e in read_journal(journal) if e.get("type") == "model.turn.completed"
    }
    count, cuts = 0, {}
    for entry in prefix:
        if entry["type"] in {"mini.message", "oc.message"} and entry["message"].get("role") != "exit":
            count += 1
        elif entry["type"] == "oc.context":
            count = len(entry["messages"])
        elif entry["type"] in {"mini.turn", "oc.turn"}:
            step = closed.get(entry["turn_id"])
            if step in checkpoints:
                cuts[str(step)] = count
    if set(cuts) != {str(step) for step in checkpoints}:
        raise ValueError("Reviewer mini history does not cover every eligible checkpoint")
    workspace = next(
        (e.get("workspace") for e in prefix if e["type"] in {"mini.session", "oc.session"}), None
    )
    return {
        "messages": training_messages(prefix),
        "prefix_message_counts": cuts,
        "tools": actor_tools_at(journal, max(checkpoints)),
        "workspace": workspace,
    }


#: The reviewer's feedback when ``turn_check`` refuses a turn. It names the branch, not what was
#: flagged: the flagged spans come from hidden tests, and naming them shows the reviewer more of them.
TURN_REJECTED = (
    "branch {branches}: the assistant turn states specifics -- test inputs, fixture values, "
    "expected outputs, exact messages or test names -- that the actor could not know from "
    "its history at that point. Rewrite the turn from the retained history and the task "
    "statement alone: what the actor has seen, and what it could check by running commands."
)


BRANCH_COUNT_MODES = ("adaptive", "fixed")


def branch_count_rule(mode: str, count: int, prompts: Prompts = DEFAULT_PROMPTS) -> str:
    """The ``{count_rule}`` of a reviewer prompt: ``count`` as a cap, or as the exact number.

    Raises ValueError for a ``mode`` other than ``adaptive`` and ``fixed``."""
    if mode == "adaptive":
        return prompts.reviewer_count_rule_adaptive.format(count=count)
    if mode == "fixed":
        return prompts.reviewer_count_rule_fixed.format(count=count)
    raise ValueError("unknown branch count mode: %r" % mode)


#: mini-swe-agent ends the episode when a command prints this line. ``prepare_branches`` refuses
#: a reviewer turn whose command carries it: the actor decides when to submit, not the reviewer.
SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def branch_run_name(round_no: int, index: int, direction: str | None) -> str:
    """The run name of branch ``index`` of round ``round_no``, with a slug of ``direction``."""
    slug = re.sub(r"[^a-z0-9]+", "-", str(direction or "b").lower())[:24]
    return "r%db%d-%s" % (round_no, index, slug)


def extract_json(text: str) -> dict[str, Any]:
    """The JSON object in ``text``: its first fenced block, else from its first ``{`` to its last ``}``.

    Raises ValueError when ``text`` holds no such object."""
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("analyst returned no JSON: %r" % text[:200])
        candidate = text[start : end + 1]
    # strict=False admits literal newlines inside strings. Long hints arrive with them, and
    # their intent is still clear.
    value: dict[str, Any] = json.loads(candidate, strict=False)
    return value


# --- journal -> transcript --------------------------------------------------

#: Characters per token in a transcript of JSON, code and test output. 3.2 is the pessimistic
#: end of what these journals measured, so a token budget never overshoots as characters.
CHARS_PER_TOKEN = 3.2

#: The caps on one transcript line, in characters: a tool result, a call's arguments and an
#: agent message. docs/RUNSTORE.md (Review requests) gives the measurements behind them.
RESULT_CHARS = 6000
ARG_CHARS = 4000
MESSAGE_CHARS = 2000
ELISION = "\n...[middle elided]...\n"


def render_transcript(
    journal_path: str | Path, token_budget: int = 100_000, char_budget: int | None = None
) -> tuple[str, int, int]:
    """The journal's transcript, one line per tool step, with its first and last step numbers.

    A step has the number of its checkpoint step, so the analyst names steps that checkpoints
    have. ``char_budget`` bounds the text exactly, and ``token_budget`` in tokens. A long line
    keeps its head and tail, and a long transcript loses its middle (docs/RUNSTORE.md)."""
    lines: list[str] = []
    step = 0
    for record in read_journal(journal_path):
        kind = record.get("type")
        if kind == "tool.started":
            step += 1
            args = json.dumps(record.get("args") or {}, ensure_ascii=False)
            lines.append("[%d] %s(%s)" % (step, record.get("name") or "?", clip_text(args, ARG_CHARS)))
        elif kind == "tool.finished" and lines:
            out = str(record.get("output") or "")
            lines[-1] += "  -> %s" % (clip_text(out, RESULT_CHARS) or record.get("status") or "")
        elif kind == "agent.message":
            text = str(record.get("text") or "").replace("\n", " ")
            if text.strip():
                lines.append("    (agent said: %s)" % clip_text(text, MESSAGE_CHARS))
    body = "\n".join(lines)
    budget = int(token_budget * CHARS_PER_TOKEN)
    if char_budget is not None:
        budget = max(0, min(budget, char_budget - len(ELISION)))
    if len(body) > budget:
        # Elide the middle, and give the tail two thirds: the late steps hold the failure under
        # diagnosis, and the early steps only show what the agent understood.
        head = budget // 3
        tail = budget - head
        body = body[:head] + ELISION + (body[-tail:] if tail else "")
    return body, 1, step


@dataclass
class Attempt:
    """One finished attempt, as the analyst and the reviewer read it."""

    name: str
    outcome: RunOutcome
    grade: Grade
    #: The hint that this attempt was given, "" for a root. A user-hint review shows it to the
    #: reviewer beside the attempt's report, so that the hint and its outcome are in one place.
    hint: str = ""
    round_no: int = 0
    assistant_turn: dict[str, Any] | None = None
    started_at: float | None = None
    validated_at: float | None = None
    seconds: float | None = None
    #: The final memory snapshot's fate ('kept', 'released', 'release_failed'), or None.
    memory_snapshot: str | None = None

    def verdict_text(self) -> str:
        """The grading verdict as the analyst reads it: the summary, the patch and the verifier's output."""
        return "%s\n\nPatch (%d lines):\n%s\n\nVerifier output:\n%s" % (
            self.grade.summary(),
            self.grade.patch.count("\n"),
            self.grade.patch[:40000],
            self.grade.detail[:20000],
        )


@dataclass(frozen=True)
class BranchChoice:
    """One branch point of a plan, resolved against its parent's recorded history."""

    run_name: str
    base: Attempt
    checkpoint: Checkpoint
    cut: str | None
    hint: str
    why: str
    assistant_turn: dict[str, Any] | None = None
    #: What ``review``'s ``turn_check`` said about ``assistant_turn``, None when no check ran.
    #: A turn keeps it when it passed, and when it was still flagged after the last resample.
    turn_check: dict[str, Any] | None = None


def prepare_branches(
    plan: dict[str, Any],
    *,
    limit: int,
    round_no: int,
    attempts: Mapping[str, Attempt],
    checkpoints: Mapping[str, Mapping[int, Checkpoint]],
    count_mode: str = "adaptive",
    guidance_mode: str = DEFAULT_GUIDANCE,
) -> list[BranchChoice]:
    """The branches of ``plan``, each resolved to its exact checkpoint and native cut.

    Raises ``ReviewerPlanError`` for a plan that the reviewer can correct. Raises ValueError for
    an unknown mode, or when a parent's history has no exact cut at a chosen step."""
    if count_mode not in BRANCH_COUNT_MODES:
        raise ValueError("unknown branch count mode: %r" % count_mode)
    if guidance_mode not in BRANCH_GUIDANCE_MODES:
        raise ValueError("unknown branch guidance mode: %r" % guidance_mode)
    branches = plan.get("branches") if isinstance(plan, dict) else None
    if not isinstance(branches, list):
        raise ReviewerPlanError("reviewer must return a branches list")
    if count_mode == "fixed" and len(branches) != limit:
        raise ReviewerPlanError(
            "fixed branch count requires exactly %d branches; reviewer returned %d" % (limit, len(branches))
        )
    if len(branches) > limit:
        raise ReviewerPlanError("reviewer returned %d branches above limit %d" % (len(branches), limit))
    choices: list[BranchChoice] = []
    cuts: dict[tuple[str, int], str | None] = {}
    for index, branch in enumerate(branches, 1):
        if not isinstance(branch, dict):
            raise ReviewerPlanError("branch %d is not an object" % index)
        base_name = branch.get("base")
        if not isinstance(base_name, str) or base_name not in attempts:
            raise ReviewerPlanError("branch %d names an unknown base %r" % (index, base_name))
        step = branch.get("branch_step")
        if type(step) is not int or step not in checkpoints.get(base_name, {}):
            raise ReviewerPlanError(
                "branch %d has no eligible checkpoint at %s:%r" % (index, base_name, step)
            )
        base = attempts[base_name]
        assistant_turn = None
        if guidance_mode == "none":
            if set(branch) - {"name", "base", "branch_step", "why"}:
                raise ReviewerPlanError("none branches may contain only point-selection fields, not guidance")
            selected_prefix(base.outcome.journal_path, checkpoints[base_name][step])
            hint = ""
        elif guidance_mode == "assistant-turn":
            if "hint" in branch:
                raise ReviewerPlanError("assistant-turn branches must not include a user hint")
            native = selected_prefix(base.outcome.journal_path, checkpoints[base_name][step])
            tools = actor_tools_at(base.outcome.journal_path, step)
            # The reviewer's JSON, of any type: validate_assistant_turn refuses all but a turn.
            proposed: Any = branch.get("assistant_turn")
            try:
                assistant_turn = validate_assistant_turn(
                    proposed,
                    history=training_messages(native),
                    tools=tools,
                    slot="opencollab-base" if native[0].get("type") == "oc.session" else "mini-swe-agent",
                )
            except ValueError as error:
                raise ReviewerPlanError(f"branch {index} ({base_name}@{step}): {error}") from error
            for call_no, call in enumerate(assistant_turn["tool_calls"], 1):
                if (
                    call["function"]["name"] == "submit_patch"
                    or SUBMISSION_MARKER in call["function"]["arguments"]
                ):
                    raise ReviewerPlanError(
                        f"branch {index} ({base_name}@{step}): a reviewer-authored turn never submits; "
                        f"command {call_no} contains {SUBMISSION_MARKER}"
                    )
            hint = ""
        else:
            if "assistant_turn" in branch:
                raise ReviewerPlanError("assistant_turn requires --branch-guidance assistant-turn")
            given = branch.get("hint")
            if not isinstance(given, str) or not given.strip():
                raise ReviewerPlanError("branch %d must supply a non-empty hint" % index)
            hint = given
        key = (base_name, step)
        if key not in cuts:
            session_id = checkpoints[base_name][step].session_ckpt
            cuts[key] = (
                None
                if session_id is None
                else conversation_restore(base.outcome.journal_path, step, session_id)
            )
        cut = cuts[key]
        if cut is None:
            raise ValueError("branch %d has no native conversation cut at %s:%d" % (index, base_name, step))
        choices.append(
            BranchChoice(
                branch_run_name(round_no, index, branch.get("name")),
                base,
                checkpoints[base_name][step],
                cut,
                hint,
                str(branch.get("why") or ""),
                assistant_turn,
            )
        )
    return choices


def outcome_from_journal(journal: Path, run_id: str = "parent") -> RunOutcome:
    """A ``RunOutcome`` rebuilt from what a finished run left in its journal."""
    status, error, final_text = "unknown", None, ""
    usage: dict[str, Any] = {}
    for record in read_journal(journal):
        kind = record.get("type")
        if kind == "run.finished":
            status = record.get("status") or status
            usage = record.get("usage") or {}
            error = record.get("error")
        elif kind == "run.result":
            final_text = record.get("text") or ""
    pairs = sum(1 for c in load_checkpoints(journal) if c.snapshot_id)
    return RunOutcome(
        run_id=run_id,
        journal_path=Path(journal),
        status=status,
        final_text=final_text,
        usage=usage,
        checkpoints=pairs,
        error=error,
    )


# --- analyst and reviewer -----------------------------------------------------


def analyst_prompt(
    problem: str,
    attempt: Attempt,
    points: Collection[int],
    width: int,
    token_budget: int,
    budget: int,
    prompts: Prompts = DEFAULT_PROMPTS,
) -> tuple[str, int, dict[str, Any]]:
    """The analyst prompt within ``budget`` characters, the number of steps, and the prompt's sizes.

    The transcript gets what the template, the problem, the verdict and room for one feedback
    leave. When that is less than a quarter of the budget, the verdict is clipped first: the
    transcript is the evidence. ``points`` are the checkpoint steps."""
    journal = attempt.outcome.journal_path
    _, lo, hi = render_transcript(journal, token_budget=token_budget)
    verdict = attempt.verdict_text()

    def render(transcript: str, verdict: str) -> str:
        return prompts.analyst_case.format(
            problem=problem[:20000],
            verdict=verdict,
            transcript=transcript,
            checkpoint_steps=json.dumps(sorted(s for s in points if s <= hi)),
            candidate_limit=width,
            lo=lo,
            hi=hi,
        )

    reserve = len(prompts.analyst_feedback.format(error="x" * 500))
    minimum = budget // 4
    room = budget - reserve - len(render("", verdict))
    if room < minimum:
        # 32: room for the elision marker clip_text inserts.
        verdict = clip_text(verdict, max(0, len(verdict) - (minimum - room) - 32), keep_newlines=True)
        room = budget - reserve - len(render("", verdict))
    transcript, _, _ = render_transcript(journal, token_budget=token_budget, char_budget=max(0, room))
    prompt = render(transcript, verdict)
    return (
        prompt,
        hi,
        {
            "budget": budget,
            "chars": len(prompt),
            "transcript_chars": len(transcript),
            "verdict_chars": len(verdict),
            "within_budget": len(prompt) + reserve <= budget,
        },
    )


def analyse(
    transport: Transport,
    model: str,
    problem: str,
    attempt: Attempt,
    points: Collection[int],
    width: int,
    token_budget: int,
    *,
    max_attempts: int = 3,
    prompt_chars: int = DEFAULT_PROMPT_CHARS,
    prompts: Prompts = DEFAULT_PROMPTS,
) -> dict[str, Any]:
    """One failed attempt's report under ``structured_review.ANALYST_SCHEMA``, with its ``prompt_budget``.

    An attempt without tool steps gets a fixed report and no request. An invalid reply goes back to
    the analyst with the validator's error. Raises ValueError when ``max_attempts`` replies were not
    valid. A refused request (HTTP 400) is sent again with half the budget, ``PROMPT_HALVINGS`` times."""
    from jsonschema import Draft202012Validator, ValidationError

    structured = getattr(transport, "supports_json_schema", False) is True
    overhead = (
        transport.json_overhead(structured_review.ANALYST_SCHEMA, "sprout_analysis") if structured else 0
    )
    budget, halvings = prompt_chars - overhead, 0
    prompt, hi, sizing = analyst_prompt(problem, attempt, points, width, token_budget, budget, prompts)
    if hi < 1:
        return {
            "failure_reason": "no tool steps recorded",
            "steps": 0,
            "lesson": "",
            "salvage": "nothing",
            "branch_candidates": [],
        }
    feedback, error, corrections = "", "", 0
    while corrections < max_attempts:
        try:
            text = (
                transport.json(
                    model, prompt + feedback, schema=structured_review.ANALYST_SCHEMA, name="sprout_analysis"
                )
                if structured
                else transport(model, prompt + feedback)
            )
        except PromptRejected:
            if halvings >= PROMPT_HALVINGS:
                raise
            # Halved, and in any case below what was just refused.
            halvings += 1
            budget = min(budget // 2, sizing["chars"] - 1)
            prompt, hi, sizing = analyst_prompt(
                problem, attempt, points, width, token_budget, budget, prompts
            )
            continue
        except ReviewResponseTruncated as exc:
            corrections += 1
            error = str(exc)[:500]
            feedback = prompts.analyst_feedback.format(error=error)
            continue
        # An invalid report is never repaired: the reviewer must not get one with a field missing.
        try:
            report = extract_json(text)
            Draft202012Validator(structured_review.ANALYST_SCHEMA).validate(report)
        except (ValueError, ValidationError) as exc:
            corrections += 1
            error = str(exc).splitlines()[0][:500]
            feedback = prompts.analyst_feedback.format(error=error)
            continue
        report["steps"] = hi
        report["prompt_budget"] = {
            **sizing,
            "halvings": halvings,
            **({"request_budget": prompt_chars, "structured_output_chars": overhead} if structured else {}),
        }
        return report
    raise ValueError("analyst report invalid after %d attempts: %s" % (max_attempts, error))


def review(
    transport: Transport,
    model: str,
    problem: str,
    attempts: list[Attempt],
    cases: dict[str, dict[str, Any]],
    *,
    round_no: int,
    width: int,
    plan_path: Path,
    guidance_mode: str = DEFAULT_GUIDANCE,
    count_mode: str = "adaptive",
    reviewer_max_attempts: int = 3,
    prompt_chars: int = DEFAULT_PROMPT_CHARS,
    prompts: Prompts = DEFAULT_PROMPTS,
    eligible: Mapping[str, Collection[int]] | None = None,
    turn_check: Callable[[BranchChoice], dict[str, Any] | None] | None = None,
    turn_resamples: int = 0,
) -> list[BranchChoice]:
    """The branches that the reviewer picks from the analyst reports ``cases``, each validated exactly.

    Returns [] when no step is eligible or no reply gives a valid plan. ``eligible`` narrows the
    steps of each attempt. A plan with a turn that ``turn_check`` flags (``leak`` true) goes back,
    ``turn_resamples`` times at most. ``plan_path`` gets the whole exchange (docs/RUNSTORE.md)."""
    points = _eligible_points(attempts, cases=cases, eligible=eligible)
    record: dict[str, Any] = {
        "reports": cases,
        "available_steps": {name: sorted(pairs) for name, pairs in points.items()},
        "branch_limit": width,
        "branch_count_mode": count_mode,
        "branch_guidance": guidance_mode,
        "reviewer_max_attempts": reviewer_max_attempts,
        "prompts": {"path": str(prompts.path), "sha256": prompts.sha256},
    }
    persist = partial(_write_plan, plan_path)
    if not any(points.values()):
        record.update(
            review=None,
            validation_error="no eligible snapshot/session pairs: no native "
            "conversation cut at a complete-turn boundary",
        )
        persist(record)
        return []
    reports = _reviewer_rows(attempts, cases=cases, points=points, guidance_mode=guidance_mode)
    template = {
        "user-hint": prompts.reviewer_user_hint,
        "assistant-turn": prompts.reviewer_assistant_turn,
        "none": prompts.reviewer_point,
    }[guidance_mode]
    schema = _review_schema(transport, guidance_mode=guidance_mode, width=width, count_mode=count_mode)
    record["structured_review"] = schema is not None
    overhead = transport.json_overhead(schema, "sprout_branches") if schema is not None else 0
    if schema is not None:
        record["structured_output"] = {
            "schema": schema,
            "request_budget": prompt_chars,
            "overhead_chars": overhead,
        }
    validate = _plan_validator(
        record,
        width=width,
        round_no=round_no,
        attempts=attempts,
        points=points,
        count_mode=count_mode,
        guidance_mode=guidance_mode,
        turn_check=turn_check,
        turn_resamples=turn_resamples,
    )
    fit, reduce = _prompt_fitter(
        record,
        template=template,
        problem=problem[:20000],
        reports=reports,
        count_rule=branch_count_rule(count_mode, width, prompts),
        budget=prompt_chars - overhead,
    )
    result: tuple[dict[str, Any], list[BranchChoice]] | None = review_with_feedback(
        _reviewer_request(transport, model=model, schema=schema),
        fit(),
        _reply_parser(
            guidance_mode,
            round_no=round_no,
            structured=schema is not None,
            tools_by_base={row["name"]: row.get("native_history", {}).get("tools", []) for row in reports},
        ),
        validate,
        max_attempts=reviewer_max_attempts,
        record=record,
        persist=persist,
        reduce=reduce,
        format_hint=prompts.reviewer_assistant_format_hint if guidance_mode == "assistant-turn" else "",
    )
    if result is None:
        return []
    plan, choices = result
    record["selected_branches"] = [_selected_branch(c, guidance_mode=guidance_mode) for c in choices]
    persist(record)
    return choices


def _eligible_points(
    attempts: list[Attempt],
    *,
    cases: dict[str, dict[str, Any]],
    eligible: Mapping[str, Collection[int]] | None,
) -> dict[str, dict[int, Checkpoint]]:
    """Each attempt's branch points up to the last step of its report, narrowed by ``eligible``."""
    from sprout.worker.agents.history import available_branch_points

    available = {a.name: available_branch_points(a.outcome.journal_path) for a in attempts}
    return {
        name: {
            step: pair
            for step, pair in pairs.items()
            if step <= cases.get(name, {}).get("steps", 0)
            and (eligible is None or step in eligible.get(name, ()))
        }
        for name, pairs in available.items()
    }


def _write_plan(plan_path: Path, record: dict[str, Any]) -> None:
    """Write ``record`` to ``plan_path`` as indented JSON, and create its directory first."""
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _reviewer_rows(
    attempts: list[Attempt],
    *,
    cases: dict[str, dict[str, Any]],
    points: Mapping[str, Mapping[int, Checkpoint]],
    guidance_mode: str,
) -> list[dict[str, Any]]:
    """The reviewer's row of each attempt: its report, its grade, its eligible steps and its guidance.

    A user-hint row has the hint that the attempt was given. Another row has the assistant turn
    that the attempt was given and its native history. Raises ValueError when ``reviewer_context`` does."""
    reports: list[dict[str, Any]] = []
    for attempt in attempts:
        # The analyst's sizing record describes its request, not the attempt, so the reviewer skips it.
        case = {key: value for key, value in cases.get(attempt.name, {}).items() if key != "prompt_budget"}
        row: dict[str, Any] = {
            **case,
            "name": attempt.name,
            "round": attempt.round_no,
            "grade": attempt.grade.summary(),
            "available_steps": sorted(points[attempt.name]),
        }
        if guidance_mode == "user-hint":
            row["hint_given"] = attempt.hint or None
        else:
            row.update(
                assistant_turn_given=attempt.assistant_turn,
                native_history=reviewer_context(attempt.outcome.journal_path, points[attempt.name]),
            )
        reports.append(row)
    return reports


def _review_schema(
    transport: Transport, *, guidance_mode: str, width: int, count_mode: str
) -> dict[str, Any] | None:
    """The wire schema of the reviewer's reply, or None when the request has no schema."""
    # A hint review answers in JSON, and the other modes in a labelled branch-plan fence. Only an
    # assistant-turn plan has a wire schema: its parser builds each turn and mints its call ids.
    structured = (
        guidance_mode == "assistant-turn" and getattr(transport, "supports_json_schema", False) is True
    )
    return structured_review.flat_review_schema(width, count_mode) if structured else None


def _reviewer_request(
    transport: Transport, *, model: str, schema: dict[str, Any] | None
) -> Callable[[str], str]:
    """The call that sends a prompt to the reviewer ``model``, under ``schema`` when it is not None."""
    return lambda text: (
        transport.json(model, text, schema=schema, name="sprout_branches")
        if schema is not None
        else transport(model, text)
    )


def _plan_validator(
    record: dict[str, Any],
    *,
    width: int,
    round_no: int,
    attempts: list[Attempt],
    points: Mapping[str, Mapping[int, Checkpoint]],
    count_mode: str,
    guidance_mode: str,
    turn_check: Callable[[BranchChoice], dict[str, Any] | None] | None,
    turn_resamples: int,
) -> Callable[[dict[str, Any]], list[BranchChoice]]:
    """The check of a reviewer plan: its exact branches, each marked with what ``turn_check`` said.

    The check raises ``TurnRejected`` for a flagged turn ``turn_resamples`` times at most, and
    counts each time in ``record``. Raises ValueError unless ``turn_resamples`` is a non-negative int."""
    if type(turn_resamples) is not int or turn_resamples < 0:
        raise ValueError("turn_resamples must be a non-negative integer")
    if turn_check is not None:
        record["turn_resamples"] = turn_resamples
    by_name = {a.name: a for a in attempts}
    rejections = 0

    def validate(plan: dict[str, Any]) -> list[BranchChoice]:
        nonlocal rejections
        choices = prepare_branches(
            plan,
            limit=width,
            round_no=round_no,
            attempts=by_name,
            checkpoints=points,
            count_mode=count_mode,
            guidance_mode=guidance_mode,
        )
        if turn_check is None:
            return choices
        checks = [turn_check(c) if c.assistant_turn is not None else None for c in choices]
        marked = [replace(c, turn_check=check) for c, check in zip(choices, checks, strict=True)]
        flagged = [index for index, check in enumerate(checks, 1) if check and check.get("leak") is True]
        if flagged and rejections < turn_resamples:
            rejections += 1
            record["turn_rejections"] = rejections
            raise TurnRejected(TURN_REJECTED.format(branches=", ".join(map(str, flagged))), checks, marked)
        return marked

    return validate


def _prompt_fitter(
    record: dict[str, Any],
    *,
    template: str,
    problem: str,
    reports: list[dict[str, Any]],
    count_rule: str,
    budget: int,
) -> tuple[Callable[[], str], Callable[[], str | None]]:
    """The ``fit`` and the ``reduce`` of a reviewer request. Each returns the prompt to send.

    ``fit`` fits ``reports`` to ``budget``. ``reduce`` lowers the budget after the endpoint refused
    a prompt. Each puts its fit and the refused fits in ``record["prompt_budget"]``."""
    refused: list[dict[str, Any]] = []

    def fit() -> str:
        prompt, sizing = fit_review_prompt(
            template, problem=problem, rows=reports, count_rule=count_rule, budget=budget
        )
        record["prompt_budget"] = {**sizing, "halvings": len(refused), "refused": refused}
        return prompt

    def reduce() -> str | None:
        """A smaller prompt after the endpoint refused one, or None after the last halving or at the floor.

        The budget halves and drops below the refused prompt's size, so a coarse stage ladder never
        sends the same prompt again. The plan keeps the refused fit: what it dropped is evidence."""
        nonlocal budget
        sent = record["prompt_budget"]
        if len(refused) >= PROMPT_HALVINGS or sent["stage"] == FLOOR:
            return None
        refused.append({key: value for key, value in sent.items() if key not in ("halvings", "refused")})
        budget = min(budget // 2, sent["chars_after"] - 1)
        return fit()

    return fit, reduce


def _reply_parser(
    guidance_mode: str,
    *,
    round_no: int,
    structured: bool,
    tools_by_base: dict[str, list[dict[str, Any]]] | None = None,
) -> Callable[[str], Any]:
    """The parser that reads the branch plan from a reviewer reply in ``guidance_mode``."""
    parsers: dict[str, Callable[[str], Any]] = {
        "user-hint": extract_json,
        "none": extract_branch_plan,
        "assistant-turn": partial(
            structured_review.extract_assistant_turn_plan,
            round_no=round_no,
            json_only=structured,
            tools_by_base=tools_by_base,
        ),
    }
    return parsers[guidance_mode]


def _selected_branch(choice: BranchChoice, *, guidance_mode: str) -> dict[str, Any]:
    """The plan file's entry of one selected branch."""
    return {
        "run_name": choice.run_name,
        "base": choice.base.name,
        "branch_step": choice.checkpoint.step,
        "snapshot_id": choice.checkpoint.snapshot_id,
        "conversation_cut": choice.cut,
        "branch_guidance": guidance_mode,
        **({"hint": choice.hint} if choice.hint else {}),
        **({"assistant_turn": choice.assistant_turn} if choice.assistant_turn is not None else {}),
        **({"turn_check": choice.turn_check} if choice.turn_check is not None else {}),
    }

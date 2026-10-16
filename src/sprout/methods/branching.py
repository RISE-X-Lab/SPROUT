"""Branching: what an analyst sees, what a reviewer may choose, and how a choice
becomes an exact branch.

Everything here is independent of how attempts run (``methods.evaluate`` runs
them as Harbor trials): the prompts (``prompts.yaml``, through
``methods.prompts``), the transcript renderer, the ``Grade`` and ``Attempt``
records the loop passes around, and ``prepare_branches``, which
turns a reviewer's plan into ``(snapshot, native cut, guidance)`` triples after
checking each one against the parent's recorded history.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from functools import partial
import json
from pathlib import Path
import re
from typing import Callable, List, Optional, Tuple

from sprout.harness.agents.mini_swe.history import (
    actor_tools_at, conversation_restore, load_prefix, reference_at, training_messages,
    validate_assistant_turn,
)
from sprout.harness_checkpoint.rollback import Checkpoint, load_checkpoints
from sprout.harness.core.journal import read_journal
from sprout.harness.orchestrator.run import RunOutcome
from sprout.methods.branch_plan import (
    PromptRejected, ReviewerPlanError, ReviewResponseTruncated, TurnRejected, extract_branch_plan,
    review_with_feedback,
)
from sprout.methods import structured_review
from sprout.methods.prompts import DEFAULT_PROMPTS, Prompts


@dataclass
class Grade:
    """A benchmark's verdict on one attempt, as Harbor's verifier reported it."""

    resolved: bool = False
    reward: Optional[float] = None
    #: The diff the verifier graded, when the task collected one (model.patch).
    patch: str = ""
    #: The verifier's own words: its result record and the tail of its logs.
    detail: str = ""
    error: Optional[str] = None

    def summary(self) -> str:
        if self.error:
            return "GRADING ERROR: %s" % self.error
        return "official Harbor reward=%s; resolved=%s" % (self.reward, self.resolved)


BRANCH_GUIDANCE_MODES = ("user-hint", "assistant-turn", "none")
DEFAULT_GUIDANCE = "assistant-turn"
# Total trajectories in a method evaluation, including the shared parent.
MAX_ROLLOUTS = 128


def require_mini_parent(journal) -> None:
    slots = {e["slot"] for e in read_journal(journal)
             if e.get("type") == "run.started" and e.get("slot")}
    if slots != {"mini-swe-agent"}:
        raise ValueError("This branch mode accepts only mini-swe-agent parent trajectories")


def selected_prefix(journal, checkpoint) -> list[dict]:
    require_mini_parent(journal)
    reference = reference_at(journal, checkpoint.step, checkpoint.session_ckpt)
    if reference is None:
        raise ValueError("No exact mini prefix at the selected checkpoint")
    return load_prefix(reference)


def reviewer_context(journal, checkpoints: dict) -> dict:
    """Supply each attempt once, with precise message offsets for eligible cuts."""
    if not checkpoints:
        return {"messages": [], "prefix_message_counts": {}}
    prefix = selected_prefix(journal, checkpoints[max(checkpoints)])
    closed = {e["turn_id"]: e["step"] for e in read_journal(journal)
              if e.get("type") == "model.turn.completed"}
    count, cuts = 0, {}
    for entry in prefix:
        if entry["type"] == "mini.message" and entry["message"].get("role") != "exit":
            count += 1
        elif entry["type"] == "mini.turn":
            step = closed.get(entry["turn_id"])
            if step in checkpoints:
                cuts[str(step)] = count
    if set(cuts) != {str(step) for step in checkpoints}:
        raise ValueError("Reviewer mini history does not cover every eligible checkpoint")
    workspace = next((e.get("workspace") for e in prefix if e["type"] == "mini.session"), None)
    return {"messages": training_messages(prefix), "prefix_message_counts": cuts,
            "tools": actor_tools_at(journal, max(checkpoints)), "workspace": workspace}


#: What the reviewer is told when ``turn_check`` refuses a turn. It names the
#: branch, never what was flagged: the flagged spans come from the hidden tests,
#: which the reviewer has not read, so naming them would hand it more of them.
TURN_REJECTED = (
    "branch {branches}: the assistant turn states specifics -- test inputs, fixture values, "
    "expected outputs, exact messages or test names -- that the actor could not know from "
    "its history at that point. Rewrite the turn from the retained history and the task "
    "statement alone: what the actor has seen, and what it could check by running commands."
)


BRANCH_COUNT_MODES = ("adaptive", "fixed")


def branch_count_rule(mode: str, count: int, prompts: Prompts = DEFAULT_PROMPTS) -> str:
    """The ``{count_rule}`` of a reviewer prompt: ``count`` as a cap, or as the exact number."""
    if mode == "adaptive":
        return prompts.reviewer_count_rule_adaptive.format(count=count)
    if mode == "fixed":
        return prompts.reviewer_count_rule_fixed.format(count=count)
    raise ValueError("unknown branch count mode: %r" % mode)


#: mini-swe-agent ends the episode when a command prints this line. The reviewer
#: prompt asks for a turn that probes, fixes and verifies and leaves that
#: decision to the actor, so the validator rejects a plan whose command carries
#: it instead of trusting the prompt alone.
SUBMISSION_MARKER = "COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT"


def branch_run_name(round_no: int, index: int, direction: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", str(direction or "b").lower())[:24]
    return "r%db%d-%s" % (round_no, index, slug)


def extract_json(text: str) -> dict:
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("analyst returned no JSON: %r" % text[:200])
        candidate = text[start:end + 1]
    # strict=False: long hints arrive with literal newlines inside strings --
    # invalid JSON, harmless intent.
    return json.loads(candidate, strict=False)


# --- journal -> transcript --------------------------------------------------
#: Roughly how many characters fit in a token here. Transcripts are JSON, code
#: and test output, which tokenize worse than prose -- 3.2 is the pessimistic end
#: of what was measured on these journals, so a token budget converts to a
#: character budget that will not overshoot.
CHARS_PER_TOKEN = 3.2

#: Per-line caps. Generous on purpose, and measured: tool RESULTS on a real run
#: have a median of ~1.9k characters and a max of ~17k, so the old 300-character
#: cap fed the analyst the first two lines of every test run and threw away the
#: assertion that explains the failure. Arguments are small (median ~112) -- the
#: one that matters is a `str_replace` payload, which is worth showing whole.
RESULT_CHARS = 6000
ARG_CHARS = 4000
MESSAGE_CHARS = 2000
ELISION = "\n...[middle elided]...\n"


def render_transcript(journal_path, token_budget: int = 100_000, char_budget: Optional[int] = None
                      ) -> "tuple[str, int, int]":
    """One line per tool step, numbered by the checkpoint step they map to.

    ``char_budget`` is an exact bound on the rendering (the analyst's request
    budget leaves it what the fixed parts do not take); ``token_budget`` the
    looser one in tokens.

    Numbered from the tool calls rather than from the checkpoints, because the
    analyst must name a step the *snapshot map* has -- and both are counted the
    same way (one per exec call, in order).

    The budget is spent per-line first and only then globally, because that is
    where the information was actually going: the old version elided the middle
    of long transcripts (which mattered rarely -- a 66-step run rendered to 34k
    characters) while truncating every tool result to 300 characters (which
    mattered always -- the failing assertion lives past that cut). A result long
    enough to be worth reading is kept head-and-tail, never head-only: a test
    run's verdict is at the END.
    """
    lines: List[str] = []
    step = 0
    for record in read_journal(journal_path):
        kind = record.get("type")
        if kind == "tool.started":
            step += 1
            args = json.dumps(record.get("args") or {}, ensure_ascii=False)
            lines.append("[%d] %s(%s)" % (step, record.get("name") or "?", _clip(args, ARG_CHARS)))
        elif kind == "tool.finished" and lines:
            out = str(record.get("output") or "")
            lines[-1] += "  -> %s" % (_clip(out, RESULT_CHARS)
                                      or record.get("status") or "")
        elif kind == "agent.message":
            text = str(record.get("text") or "").replace("\n", " ")
            if text.strip():
                lines.append("    (agent said: %s)"
                             % _clip(text, MESSAGE_CHARS))
    body = "\n".join(lines)
    budget = int(token_budget * CHARS_PER_TOKEN)
    if char_budget is not None:
        budget = max(0, min(budget, char_budget - len(ELISION)))
    if len(body) > budget:
        # Elide the middle, and give the TAIL two thirds: the late steps contain
        # the failure being diagnosed, the early ones only establish what was
        # understood. An even split looked fair and was not -- with a single step
        # rendering to several thousand characters, half a small budget did not
        # reach the end of step 1, so the last steps vanished entirely.
        head = budget // 3
        tail = budget - head
        body = body[:head] + ELISION + (body[-tail:] if tail else "")
    return body, 1, step


def _clip(text: str, limit: int, *, keep_newlines: bool = False) -> str:
    """Head AND tail when something is too long -- never head only.

    A test run's verdict is at the end of its output, so head-only truncation
    keeps the banner and drops the answer. Transcript lines are one per step,
    so newlines go; a verdict or a message keeps them.
    """
    if not keep_newlines:
        text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return "%s …[%d chars elided]… %s" % (text[:head], len(text) - limit,
                                          text[-tail:])


@dataclass
class Attempt:
    name: str
    outcome: RunOutcome
    grade: Grade
    #: The direction this attempt was given ("" for the parent). Analyses quote
    #: it so the reviewer can see hypothesis -> outcome in one place.
    hint: str = ""
    round_no: int = 0
    assistant_turn: Optional[dict] = None
    started_at: Optional[float] = None
    validated_at: Optional[float] = None
    seconds: Optional[float] = None
    #: The final memory snapshot's fate ('kept', 'released', 'release_failed'), or None.
    memory_snapshot: Optional[str] = None

    def verdict_text(self) -> str:
        """The grading verdict as the analysts see it."""
        return "%s\n\nPatch (%d lines):\n%s\n\nVerifier output:\n%s" % (
            self.grade.summary(), self.grade.patch.count("\n"),
            self.grade.patch[:40000], self.grade.detail[:20000])


@dataclass(frozen=True)
class BranchChoice:
    run_name: str
    base: Attempt
    checkpoint: Checkpoint
    cut: Optional[str]
    hint: str
    why: str
    assistant_turn: Optional[dict] = None
    #: What ``review``'s ``turn_check`` said about ``assistant_turn`` (None
    #: when no check ran): kept with a turn that passed it, and with one that
    #: was still flagged once the check's resamples were spent.
    turn_check: Optional[dict] = None


def prepare_branches(plan: dict, *, limit: int, round_no: int,
                     attempts: dict, checkpoints: dict,
                     count_mode: str = "adaptive",
                     guidance_mode: str = DEFAULT_GUIDANCE) -> List[BranchChoice]:
    """Resolve each selected point exactly before any continuation starts."""
    if count_mode not in BRANCH_COUNT_MODES:
        raise ValueError("unknown branch count mode: %r" % count_mode)
    if guidance_mode not in BRANCH_GUIDANCE_MODES:
        raise ValueError("unknown branch guidance mode: %r" % guidance_mode)
    branches = plan.get("branches") if isinstance(plan, dict) else None
    if not isinstance(branches, list):
        raise ReviewerPlanError("reviewer must return a branches list")
    if count_mode == "fixed" and len(branches) != limit:
        raise ReviewerPlanError("fixed branch count requires exactly %d branches; reviewer returned %d" %
                         (limit, len(branches)))
    if len(branches) > limit:
        raise ReviewerPlanError("reviewer returned %d branches above limit %d" % (len(branches), limit))
    choices = []
    cuts = {}
    for index, branch in enumerate(branches, 1):
        if not isinstance(branch, dict):
            raise ReviewerPlanError("branch %d is not an object" % index)
        base_name = branch.get("base")
        if not isinstance(base_name, str) or base_name not in attempts:
            raise ReviewerPlanError("branch %d names an unknown base %r" % (index, base_name))
        step = branch.get("branch_step")
        if type(step) is not int or step not in checkpoints.get(base_name, {}):
            raise ReviewerPlanError("branch %d has no eligible checkpoint at %s:%r" % (index, base_name, step))
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
            try:
                assistant_turn = validate_assistant_turn(
                    branch.get("assistant_turn"),
                    history=[e["message"] for e in native if e["type"] == "mini.message"], tools=tools)
            except ValueError as error:
                raise ReviewerPlanError(f"branch {index} ({base_name}@{step}): {error}") from error
            for call_no, call in enumerate(assistant_turn["tool_calls"], 1):
                if SUBMISSION_MARKER in call["function"]["arguments"]:
                    raise ReviewerPlanError(
                        f"branch {index} ({base_name}@{step}): a reviewer-authored turn never submits; "
                        f"command {call_no} contains {SUBMISSION_MARKER}")
            hint = ""
        else:
            if "assistant_turn" in branch:
                raise ReviewerPlanError("assistant_turn requires --branch-guidance assistant-turn")
            hint = branch.get("hint")
            if not isinstance(hint, str) or not hint.strip():
                raise ReviewerPlanError("branch %d must supply a non-empty hint" % index)
        key = (base_name, step)
        if key not in cuts:
            cuts[key] = conversation_restore(
                base.outcome.journal_path, step, checkpoints[base_name][step].session_ckpt)
        cut = cuts[key]
        if cut is None:
            raise ValueError("branch %d has no native conversation cut at %s:%d" %
                             (index, base_name, step))
        choices.append(BranchChoice(
            branch_run_name(round_no, index, branch.get("name")), base,
            checkpoints[base_name][step], cut, hint, str(branch.get("why") or ""), assistant_turn))
    return choices


def outcome_from_journal(journal: Path, run_id: str = "parent") -> RunOutcome:
    """A ``RunOutcome`` rebuilt from what a finished run left in its journal."""
    status, usage, error, final_text = "unknown", {}, None, ""
    for record in read_journal(journal):
        kind = record.get("type")
        if kind == "run.finished":
            status = record.get("status") or status
            usage = record.get("usage") or {}
            error = record.get("error")
        elif kind == "run.result":
            final_text = record.get("text") or ""
    pairs = sum(1 for c in load_checkpoints(journal) if c.snapshot_id)
    return RunOutcome(run_id=run_id, journal_path=Path(journal), status=status,
                      final_text=final_text, usage=usage, checkpoints=pairs,
                      error=error)


# --- the size of a request -----------------------------------------------------

#: The budget of one analyst or reviewer request, in characters. Four characters
#: per token is the rule of thumb, so 400k characters is about 100k prompt tokens:
#: with the 64k output tokens the runner asks for, that stays well inside the
#: model's context even where JSON tokenizes worse than prose. It is about half
#: the largest reviewer request known to have succeeded (a 136-step parent alone,
#: 746k characters) and a tenth of the one the endpoint refused with 400 Bad
#: Request (3.7M: the same parent plus four graded branches, each repeating the
#: parent's prefix up to its cut).
DEFAULT_PROMPT_CHARS = 400_000

#: How many times a request the endpoint refuses (HTTP 400) is retried with the
#: budget halved before the round gives up. The floor is the reports alone.
PROMPT_HALVINGS = 2


@dataclass(frozen=True)
class Window:
    """Which native messages a windowed history keeps: ``before`` messages up to
    each analyst candidate's cut and ``after`` beyond it, plus the last ``tail``
    and the first ``head`` (the system and task prompts)."""
    before: int
    after: int
    tail: int
    head: int = 2


@dataclass(frozen=True)
class Stage:
    name: str
    parent: Optional[Window]     # None: the parent's history is kept whole
    branches: Optional[Window]
    clip: Optional[int]          # tool results, reasoning and arguments longer than this are clipped


#: The reduction order of a reviewer request over budget, applied stage by stage
#: until it fits. The analyst reports are never reduced: they are what the
#: reviewer chooses by. Native histories are windowed first, branches before the
#: parent (a branch repeats the parent's prefix up to its cut, so it is the
#: redundant copy); then tool results, provider reasoning and call arguments in
#: the retained messages are clipped harder and the windows narrowed; the floor
#: is the reports alone: no native message at all, each history one marker (the
#: task text is in the prompt already, and the actor's system prompt does not
#: help choose a point). Every omitted run of messages becomes one marker that
#: says how many were omitted and where, so prefix_message_counts stay exact.
_WIDE, _NARROW, _NONE = Window(12, 4, 8), Window(6, 2, 4), Window(0, 0, 0, head=0)
STAGES = (
    Stage("branch histories windowed", None, _WIDE, None),
    Stage("parent history windowed", _WIDE, _WIDE, None),
    Stage("tool results, reasoning and arguments clipped to 6000", _WIDE, _WIDE, 6000),
    Stage("clipped to 2000", _WIDE, _WIDE, 2000),
    Stage("windows narrowed, clipped to 1000", _NARROW, _NARROW, 1000),
    Stage("reports alone", _NONE, _NONE, 0),
)
FLOOR = STAGES[-1].name


def omission_marker(first: int, last: int, total: int) -> dict:
    """The one entry that stands for the omitted messages at positions first..last
    (1-based) of a history ``total`` messages long. It names the position of the
    message that follows it -- or says that none does, when the run it stands
    for reaches the end of the history."""
    count = last - first + 1
    if last < total:
        where = ("positions %d-%d of the full history. prefix_message_counts count the full history "
                 "and are still exact; the next message is position %d." % (first, last, last + 1))
    else:
        where = ("positions %d-%d of the full history, through its end. prefix_message_counts count "
                 "the full history and are still exact." % (first, last))
    return {"role": "omitted", "omitted_messages": count, "positions": [first, last],
            "content": "[%d message%s omitted here to fit this request: %s]" % (
                count, "" if count == 1 else "s", where)}


def window_messages(messages: List[dict], cuts: List[int], window: Window) -> Tuple[List[dict], int]:
    """The messages a window keeps, markers in place of the rest; returns them and
    how many were omitted. ``cuts`` are prefix_message_counts: the first ``cut``
    messages are what a branch there retains."""
    total = len(messages)
    keep = set(range(min(window.head, total)))
    for cut in cuts:
        keep.update(range(max(0, cut - window.before), min(total, cut + window.after)))
    keep.update(range(max(0, total - window.tail), total))
    kept, omitted, position = [], 0, 0
    while position < total:
        if position in keep:
            kept.append(messages[position])
            position += 1
            continue
        start = position
        while position < total and position not in keep:
            position += 1
        kept.append(omission_marker(start + 1, position, total))
        omitted += position - start
    return kept, omitted


def clip_messages(messages: List[dict], limit: int) -> Tuple[List[dict], int]:
    """Tool results, provider reasoning and call arguments over ``limit`` kept
    head and tail, newlines and all (the reviewer reads them as the actor saw
    them); returns the messages and how many fields were clipped."""
    clipped, out = 0, []
    for message in messages:
        copy = dict(message)
        if copy.get("role") == "tool" and len(copy.get("content") or "") > limit:
            copy["content"] = _clip(copy["content"], limit, keep_newlines=True)
            clipped += 1
        if len(copy.get("reasoning_content") or "") > limit:
            copy["reasoning_content"] = _clip(copy["reasoning_content"], limit, keep_newlines=True)
            clipped += 1
        if copy.get("tool_calls"):
            calls = []
            for call in copy["tool_calls"]:
                arguments = call["function"]["arguments"]
                if len(arguments) > limit:
                    call = {**call, "function": {**call["function"],
                                                 "arguments": _clip(arguments, limit, keep_newlines=True)}}
                    clipped += 1
                calls.append(call)
            copy["tool_calls"] = calls
        out.append(copy)
    return out, clipped


def reduce_row(row: dict, window: Optional[Window], clip: Optional[int]) -> Tuple[dict, dict]:
    """One reviewer row at one stage: its native history windowed around the
    analyst's candidates and clipped; the report fields untouched."""
    native = row.get("native_history")
    if not isinstance(native, dict) or not native.get("messages") or (window is None and clip is None):
        return row, {}
    messages, counts = native["messages"], native.get("prefix_message_counts") or {}
    stats = {"messages": len(messages)}
    if window is not None:
        cuts = sorted({counts[str(c["step"])] for c in row.get("branch_candidates") or []
                       if str(c.get("step")) in counts})
        messages, stats["omitted"] = window_messages(messages, cuts, window)
        stats["retained"] = stats["messages"] - stats["omitted"]
    if clip:
        messages, stats["clipped"] = clip_messages(messages, clip)
    return {**row, "native_history": {**native, "messages": messages}}, stats


def fit_review_prompt(template: str, *, problem: str, rows: List[dict], count_rule: str,
                      budget: int) -> Tuple[str, dict]:
    """The reviewer prompt within ``budget`` characters, with the record of how.

    Unchanged when it fits. Otherwise the ``STAGES`` are applied in order until
    it does; the last one is sent even when it does not fit, and the record says
    so (``within_budget``). The record is the plan file's ``prompt_budget``.
    """
    def render(reports):
        return template.format(problem=problem, reports=json.dumps(reports, indent=1, ensure_ascii=False),
                               count_rule=count_rule)

    prompt = render(rows)
    record = {"budget": budget, "chars_before": len(prompt), "chars_after": len(prompt),
              "within_budget": len(prompt) <= budget, "stage": None, "attempts": {}}
    for stage in STAGES:
        if record["within_budget"]:
            break
        reduced, stats = [], {}
        for row in rows:
            window = stage.parent if row.get("round") == 0 else stage.branches
            reduced_row, row_stats = reduce_row(row, window, stage.clip)
            reduced.append(reduced_row)
            if row_stats:
                stats[row["name"]] = row_stats
        prompt = render(reduced)
        record.update(chars_after=len(prompt), within_budget=len(prompt) <= budget,
                      stage=stage.name, attempts=stats)
    return prompt, record


# --- analyst and reviewer -----------------------------------------------------


def analyst_prompt(problem: str, attempt: Attempt, points: dict, width: int,
                   token_budget: int, budget: int, prompts: Prompts = DEFAULT_PROMPTS) -> Tuple[str, int, dict]:
    """The analyst prompt within ``budget`` characters, the step count and the sizing.

    The transcript takes what the fixed parts (template, problem, verdict and
    room for one validation feedback) leave, elided in the middle as
    ``render_transcript`` does. When that would leave it less than a quarter of
    the budget, the verdict is clipped first: the transcript is the evidence.
    """
    journal = attempt.outcome.journal_path
    _, lo, hi = render_transcript(journal, token_budget=token_budget)
    verdict = attempt.verdict_text()

    def render(transcript, verdict):
        return prompts.analyst_case.format(
            problem=problem[:20000], verdict=verdict, transcript=transcript,
            checkpoint_steps=json.dumps(sorted(s for s in points if s <= hi)),
            candidate_limit=width, lo=lo, hi=hi)

    reserve = len(prompts.analyst_feedback.format(error="x" * 500))
    minimum = budget // 4
    room = budget - reserve - len(render("", verdict))
    if room < minimum:
        # 32: room for the elision marker _clip inserts.
        verdict = _clip(verdict, max(0, len(verdict) - (minimum - room) - 32), keep_newlines=True)
        room = budget - reserve - len(render("", verdict))
    transcript, _, _ = render_transcript(journal, token_budget=token_budget, char_budget=max(0, room))
    prompt = render(transcript, verdict)
    return prompt, hi, {"budget": budget, "chars": len(prompt), "transcript_chars": len(transcript),
                        "verdict_chars": len(verdict), "within_budget": len(prompt) + reserve <= budget}


def analyse(transport, model: str, problem: str, attempt: Attempt, points: dict,
            width: int, token_budget: int, *, max_attempts: int = 3,
            prompt_chars: int = DEFAULT_PROMPT_CHARS, prompts: Prompts = DEFAULT_PROMPTS) -> dict:
    """One failed attempt -> failure_reason, lesson, salvage, candidate steps.

    The report is validated against ``structured_review.ANALYST_SCHEMA``; an
    invalid reply is returned to the analyst with the validator's error, up to
    ``max_attempts`` times, and never repaired here -- the reviewer must not be
    handed a report with a field quietly missing. The request is held to
    ``prompt_chars``; one the endpoint refuses (HTTP 400) is retried with the
    budget halved, ``PROMPT_HALVINGS`` times at most. The sizing that produced
    the accepted report is returned as its ``prompt_budget``. ``prompts`` holds
    the analyst's template and feedback (``methods.prompts``).
    """
    from jsonschema import Draft202012Validator, ValidationError

    structured = getattr(transport, "supports_json_schema", False) is True
    overhead = (transport.json_overhead(structured_review.ANALYST_SCHEMA, "sprout_analysis")
                if structured else 0)
    budget, halvings = prompt_chars - overhead, 0
    prompt, hi, sizing = analyst_prompt(problem, attempt, points, width, token_budget, budget, prompts)
    if hi < 1:
        return {"failure_reason": "no tool steps recorded", "steps": 0,
                "lesson": "", "salvage": "nothing", "branch_candidates": []}
    feedback, error, corrections = "", "", 0
    while corrections < max_attempts:
        try:
            text = (transport.json(model, prompt + feedback, schema=structured_review.ANALYST_SCHEMA,
                                   name="sprout_analysis") if structured else transport(model, prompt + feedback))
        except PromptRejected:
            if halvings >= PROMPT_HALVINGS:
                raise
            # Halved, and in any case below what was just refused.
            halvings += 1
            budget = min(budget // 2, sizing["chars"] - 1)
            prompt, hi, sizing = analyst_prompt(problem, attempt, points, width, token_budget, budget, prompts)
            continue
        except ReviewResponseTruncated as exc:
            corrections += 1
            error = str(exc)[:500]
            feedback = prompts.analyst_feedback.format(error=error)
            continue
        try:
            report = extract_json(text)
            Draft202012Validator(structured_review.ANALYST_SCHEMA).validate(report)
        except (ValueError, ValidationError) as exc:
            corrections += 1
            error = str(exc).splitlines()[0][:500]
            feedback = prompts.analyst_feedback.format(error=error)
            continue
        report["steps"] = hi
        report["prompt_budget"] = {**sizing, "halvings": halvings,
                                  **({"request_budget": prompt_chars, "structured_output_chars": overhead}
                                     if structured else {})}
        return report
    raise ValueError("analyst report invalid after %d attempts: %s" % (max_attempts, error))


def review(transport, model: str, problem: str, attempts: List[Attempt], cases: dict,
           *, round_no: int, width: int, plan_path: Path, guidance_mode: str = DEFAULT_GUIDANCE,
           count_mode: str = "adaptive", reviewer_max_attempts: int = 3,
           prompt_chars: int = DEFAULT_PROMPT_CHARS, prompts: Prompts = DEFAULT_PROMPTS,
           eligible: Optional[dict] = None,
           turn_check: Optional[Callable[[BranchChoice], Optional[dict]]] = None,
           turn_resamples: int = 0) -> List[BranchChoice]:
    """The reviewer reads every analysis and picks branches; each is validated exactly.

    ``plan_path`` receives the whole exchange: eligible steps, every reviewer
    attempt with its validation error and size, the resolved selections, and
    under ``prompt_budget`` how the request was held to ``prompt_chars`` (see
    ``fit_review_prompt``). A request the endpoint refuses (HTTP 400) is retried
    with the budget halved, ``PROMPT_HALVINGS`` times at most, each try recorded:
    ``prompt_budget`` describes the request sent last and keeps the fits the
    endpoint refused before it under ``refused``, oldest first. ``prompts``
    supplies the template for ``guidance_mode`` and the count rule; the plan
    records its path and sha256 under ``prompts``, and under
    ``structured_review`` whether this request was schema-constrained (an
    assistant-turn review through a transport with ``supports_json_schema``).

    ``eligible`` (attempt name -> steps) narrows the points the reviewer may
    choose from; with one step it asks for a continuation at exactly that point,
    which is how several independent proposals for one point are collected.

    ``turn_check`` is asked about every assistant turn of a plan that passed
    validation; it returns a result (``leak`` true for a turn it refuses) or
    None. A plan with a refused turn goes back to the reviewer (``TURN_REJECTED``,
    not counted against ``reviewer_max_attempts``) up to ``turn_resamples``
    times; after that the plan is taken as it is, and so is the last refused
    plan when no attempt after it yields one (``taken_flagged`` in the plan).
    Either way every branch keeps its result in ``BranchChoice.turn_check``,
    and every refusal is in the plan.
    """
    from sprout.harness.agents.mini_swe.history import available_branch_points

    points = {a.name: available_branch_points(a.outcome.journal_path) for a in attempts}
    points = {name: {step: pair for step, pair in pairs.items()
                     if step <= cases.get(name, {}).get("steps", 0)
                     and (eligible is None or step in eligible.get(name, ()))}
              for name, pairs in points.items()}
    record = {"reports": cases, "available_steps": {name: sorted(pairs) for name, pairs in points.items()},
              "branch_limit": width, "branch_count_mode": count_mode,
              "branch_guidance": guidance_mode, "reviewer_max_attempts": reviewer_max_attempts,
              "prompts": {"path": str(prompts.path), "sha256": prompts.sha256}}

    def persist(value):
        plan_path.parent.mkdir(parents=True, exist_ok=True)
        plan_path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    if not any(points.values()):
        record.update(review=None, validation_error="no eligible snapshot/session pairs: no native "
                                                    "conversation cut at a complete-turn boundary")
        persist(record)
        return []
    reports = []
    for attempt in attempts:
        # The analyst's own sizing record is the controller's, not evidence.
        case = {key: value for key, value in cases.get(attempt.name, {}).items() if key != "prompt_budget"}
        row = {**case, "name": attempt.name, "round": attempt.round_no,
               "grade": attempt.grade.summary(), "available_steps": sorted(points[attempt.name])}
        if guidance_mode == "user-hint":
            row["hint_given"] = attempt.hint or None
        else:
            row.update(assistant_turn_given=attempt.assistant_turn,
                       native_history=reviewer_context(attempt.outcome.journal_path, points[attempt.name]))
        reports.append(row)
    template = {"user-hint": prompts.reviewer_user_hint, "assistant-turn": prompts.reviewer_assistant_turn,
                "none": prompts.reviewer_point}[guidance_mode]
    # Hint reviews answer in JSON; the other modes in a labelled branch-plan fence,
    # and only assistant turns are checked against the wire schema -- built by
    # the controller from each branch's content and commands, with call ids
    # minted from the round.
    structured = guidance_mode == "assistant-turn" and getattr(transport, "supports_json_schema", False) is True
    record["structured_review"] = structured
    schema = structured_review.flat_review_schema(width, count_mode) if structured else None
    overhead = transport.json_overhead(schema, "sprout_branches") if structured else 0
    parse = {"user-hint": extract_json, "none": extract_branch_plan,
             "assistant-turn": partial(structured_review.extract_assistant_turn_plan, round_no=round_no,
                                       json_only=structured)}[guidance_mode]
    format_hint = prompts.reviewer_assistant_format_hint if guidance_mode == "assistant-turn" else ""
    budget = prompt_chars - overhead
    if structured:
        record["structured_output"] = {"schema": schema, "request_budget": prompt_chars,
                                        "overhead_chars": overhead}

    def fit():
        prompt, sizing = fit_review_prompt(template, problem=problem[:20000], rows=reports,
                                           count_rule=branch_count_rule(count_mode, width, prompts), budget=budget)
        record["prompt_budget"] = {**sizing, "halvings": len(refused), "refused": refused}
        return prompt

    def reduce():
        """A smaller prompt after the endpoint refused one, or None at the floor.

        The budget is halved, and in any case set below what was just refused,
        so a coarse stage ladder never resends the same prompt. The refused fit
        stays in the plan: what it dropped is part of the record."""
        nonlocal budget
        sent = record["prompt_budget"]
        if len(refused) >= PROMPT_HALVINGS or sent["stage"] == FLOOR:
            return None
        refused.append({key: value for key, value in sent.items() if key not in ("halvings", "refused")})
        budget = min(budget // 2, sent["chars_after"] - 1)
        return fit()

    if type(turn_resamples) is not int or turn_resamples < 0:
        raise ValueError("turn_resamples must be a non-negative integer")
    rejections = 0

    def validate(plan):
        nonlocal rejections
        choices = prepare_branches(plan, limit=width, round_no=round_no,
                                   attempts={a.name: a for a in attempts}, checkpoints=points,
                                   count_mode=count_mode, guidance_mode=guidance_mode)
        if turn_check is None:
            return choices
        checks = [turn_check(c) if c.assistant_turn is not None else None for c in choices]
        marked = [replace(c, turn_check=check) for c, check in zip(choices, checks)]
        flagged = [index for index, check in enumerate(checks, 1) if check and check.get("leak") is True]
        if flagged and rejections < turn_resamples:
            rejections += 1
            record["turn_rejections"] = rejections
            raise TurnRejected(TURN_REJECTED.format(branches=", ".join(map(str, flagged))), checks, marked)
        return marked

    refused: List[dict] = []
    if turn_check is not None:
        record["turn_resamples"] = turn_resamples
    result = review_with_feedback(
        lambda text: (transport.json(model, text, schema=schema, name="sprout_branches")
                      if structured else transport(model, text)), fit(), parse, validate,
        max_attempts=reviewer_max_attempts, record=record, persist=persist, reduce=reduce,
        format_hint=format_hint)
    if result is None:
        return []
    plan, choices = result
    record["selected_branches"] = [
        {"run_name": c.run_name, "base": c.base.name, "branch_step": c.checkpoint.step,
         "snapshot_id": c.checkpoint.snapshot_id, "conversation_cut": c.cut,
         "branch_guidance": guidance_mode,
         **({"hint": c.hint} if c.hint else {}),
         **({"assistant_turn": c.assistant_turn} if c.assistant_turn is not None else {}),
         **({"turn_check": c.turn_check} if c.turn_check is not None else {})}
        for c in choices]
    persist(record)
    return choices

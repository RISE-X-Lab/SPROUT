"""The size of an analyst or reviewer request: its budget, and how a reviewer prompt is fitted to it.

``fit_review_prompt`` holds a reviewer prompt to a budget. It applies the ``STAGES`` in order, and
each stage windows and clips the native histories of the reviewer rows. ``clip_text`` keeps the
head and the tail of a long text. docs/RUNSTORE.md (Review requests) gives the measurements behind
the budget and the order of the stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

#: The budget of one analyst or reviewer request, in characters: about 100k prompt tokens.
#: docs/RUNSTORE.md (Review requests) gives the measurements behind it.
DEFAULT_PROMPT_CHARS = 400_000

#: How many times a request that the endpoint refuses (HTTP 400) is sent again with half the
#: budget, before the refusal counts as the request's failure.
PROMPT_HALVINGS = 2


@dataclass(frozen=True)
class Window:
    """Which native messages a windowed history keeps.

    It keeps ``before`` messages up to each analyst candidate's cut and ``after`` beyond it,
    the last ``tail``, and the first ``head``: the system and task prompts."""

    before: int
    after: int
    tail: int
    head: int = 2


@dataclass(frozen=True)
class Stage:
    """One reduction of a reviewer request, under the ``name`` that the plan file records.

    ``parent`` windows the round-0 history and ``branches`` every other one."""

    name: str
    parent: Window | None  # None: the parent's history is kept whole
    branches: Window | None
    clip: int | None  # tool results, reasoning and arguments longer than this are clipped


# The floor keeps no message, not even the first two: the reviewer prompt has the task text,
# and the actor's system prompt does not help to choose a point.
_WIDE, _NARROW, _NONE = Window(12, 4, 8), Window(6, 2, 4), Window(0, 0, 0, head=0)
#: The reductions of a reviewer request over budget, applied in order until it fits. The analyst
#: reports are never reduced. docs/RUNSTORE.md (Review requests) gives the order and the reasons.
STAGES = (
    Stage("branch histories windowed", None, _WIDE, None),
    Stage("parent history windowed", _WIDE, _WIDE, None),
    Stage("tool results, reasoning and arguments clipped to 6000", _WIDE, _WIDE, 6000),
    Stage("clipped to 2000", _WIDE, _WIDE, 2000),
    Stage("windows narrowed, clipped to 1000", _NARROW, _NARROW, 1000),
    Stage("reports alone", _NONE, _NONE, 0),
)
FLOOR = STAGES[-1].name


def clip_text(text: str, limit: int, *, keep_newlines: bool = False) -> str:
    """``text``, or its head and tail around a marker when it is longer than ``limit`` characters.

    The head and the tail together are ``limit`` characters. A test run's verdict is at the end
    of its output, so the tail always stays. Newlines become spaces unless ``keep_newlines``."""
    if not keep_newlines:
        text = text.replace("\n", " ")
    if len(text) <= limit:
        return text
    head = limit * 2 // 3
    tail = limit - head
    return "%s …[%d chars elided]… %s" % (text[:head], len(text) - limit, text[-tail:])


def omission_marker(first: int, last: int, total: int) -> dict[str, Any]:
    """The one message that stands for the omitted messages at positions ``first`` to ``last``.

    Positions count from 1 in a history ``total`` messages long. The marker names the position
    of the message after it, or says that the omitted run reaches the end of the history."""
    count = last - first + 1
    if last < total:
        where = (
            "positions %d-%d of the full history. prefix_message_counts count the full history "
            "and are still exact; the next message is position %d." % (first, last, last + 1)
        )
    else:
        where = (
            "positions %d-%d of the full history, through its end. prefix_message_counts count "
            "the full history and are still exact." % (first, last)
        )
    return {
        "role": "omitted",
        "omitted_messages": count,
        "positions": [first, last],
        "content": "[%d message%s omitted here to fit this request: %s]"
        % (count, "" if count == 1 else "s", where),
    }


def window_messages(
    messages: list[dict[str, Any]], cuts: list[int], window: Window
) -> tuple[list[dict[str, Any]], int]:
    """The messages that ``window`` keeps, a marker for each run of the rest, and the count omitted.

    ``cuts`` are prefix_message_counts: a branch at a cut keeps the first ``cut`` messages."""
    total = len(messages)
    keep = set(range(min(window.head, total)))
    for cut in cuts:
        keep.update(range(max(0, cut - window.before), min(total, cut + window.after)))
    keep.update(range(max(0, total - window.tail), total))
    kept: list[dict[str, Any]] = []
    omitted, position = 0, 0
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


def clip_messages(messages: list[dict[str, Any]], limit: int) -> tuple[list[dict[str, Any]], int]:
    """``messages`` with each tool result, provider reasoning and call argument over ``limit`` clipped.

    A clipped field keeps its head, its tail and its newlines, as the actor saw them. Returns the
    messages and how many fields were clipped."""
    clipped = 0
    out: list[dict[str, Any]] = []
    for message in messages:
        copy = dict(message)
        if copy.get("role") == "tool" and len(copy.get("content") or "") > limit:
            copy["content"] = clip_text(copy["content"], limit, keep_newlines=True)
            clipped += 1
        if len(copy.get("reasoning_content") or "") > limit:
            copy["reasoning_content"] = clip_text(copy["reasoning_content"], limit, keep_newlines=True)
            clipped += 1
        if copy.get("tool_calls"):
            calls = []
            for call in copy["tool_calls"]:
                arguments = call["function"]["arguments"]
                if len(arguments) > limit:
                    call = {
                        **call,
                        "function": {
                            **call["function"],
                            "arguments": clip_text(arguments, limit, keep_newlines=True),
                        },
                    }
                    clipped += 1
                calls.append(call)
            copy["tool_calls"] = calls
        out.append(copy)
    return out, clipped


def reduce_row(
    row: dict[str, Any], window: Window | None, clip: int | None
) -> tuple[dict[str, Any], dict[str, int]]:
    """A reviewer row reduced by one stage, and the counts of what the stage kept and dropped.

    The native history is windowed around the analyst's candidate steps and clipped. The report
    fields stay as they are. A row without native messages, or a stage that does neither, is
    returned unchanged with no counts."""
    native = row.get("native_history")
    if not isinstance(native, dict) or not native.get("messages") or (window is None and clip is None):
        return row, {}
    messages, counts = native["messages"], native.get("prefix_message_counts") or {}
    stats = {"messages": len(messages)}
    if window is not None:
        cuts = sorted(
            {
                counts[str(c["step"])]
                for c in row.get("branch_candidates") or []
                if str(c.get("step")) in counts
            }
        )
        messages, stats["omitted"] = window_messages(messages, cuts, window)
        stats["retained"] = stats["messages"] - stats["omitted"]
    if clip:
        messages, stats["clipped"] = clip_messages(messages, clip)
    return {**row, "native_history": {**native, "messages": messages}}, stats


def fit_review_prompt(
    template: str, *, problem: str, rows: list[dict[str, Any]], count_rule: str, budget: int
) -> tuple[str, dict[str, Any]]:
    """The reviewer prompt within ``budget`` characters, and the record of how it was fitted.

    A prompt that fits is unchanged. Otherwise the ``STAGES`` apply in order until it fits. The
    last stage is sent even when it does not fit, and the record's ``within_budget`` says so.
    The record is the plan file's ``prompt_budget``."""

    def render(reports: list[dict[str, Any]]) -> str:
        return template.format(
            problem=problem, reports=json.dumps(reports, indent=1, ensure_ascii=False), count_rule=count_rule
        )

    prompt = render(rows)
    record: dict[str, Any] = {
        "budget": budget,
        "chars_before": len(prompt),
        "chars_after": len(prompt),
        "within_budget": len(prompt) <= budget,
        "stage": None,
        "attempts": {},
    }
    for stage in STAGES:
        if record["within_budget"]:
            break
        reduced: list[dict[str, Any]] = []
        stats: dict[str, dict[str, int]] = {}
        for row in rows:
            window = stage.parent if row.get("round") == 0 else stage.branches
            reduced_row, row_stats = reduce_row(row, window, stage.clip)
            reduced.append(reduced_row)
            if row_stats:
                stats[row["name"]] = row_stats
        prompt = render(reduced)
        record.update(
            chars_after=len(prompt), within_budget=len(prompt) <= budget, stage=stage.name, attempts=stats
        )
    return prompt, record

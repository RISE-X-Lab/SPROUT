"""One trajectory as markdown or json, a branch stitched onto its ancestors.

A branch journal starts with ``fork.origin``; its history before that is the
parent's journal up to ``branch_step``. :func:`lineage` follows the origins up
(re-rooting a moved parent beside the branch), compares each parent's sha256
with what the branch recorded and REPORTS a difference -- a sweep or a released
memory snapshot appends a line and changes the hash -- and the renderers draw
the boundary by turn: a parent segment ends after the turn whose step is the
branch step, whatever position its checkpoint record has in the file.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

from sprout.inspect.journal import (
    Call,
    Capture,
    Trajectory,
    Turn,
    clip,
    minutes,
    read_trajectory,
    relocate,
    step_count,
)
from sprout.inspect.layout import receipt_identity

MAX_DEPTH = 16


@dataclass(frozen=True)
class Segment:
    trajectory: Trajectory
    cut_step: int | None  # None: rendered whole (the leaf)
    cut_turn: int | None  # the turn whose step == cut_step
    sha256_ok: bool | None
    recorded_sha256: str | None
    actual_sha256: str | None
    note: str | None


@dataclass(frozen=True)
class Lineage:
    segments: tuple[Segment, ...]  # root ancestor first, the requested journal last
    missing: tuple[str, ...]  # recorded parent paths that could not be found
    notes: tuple[str, ...]  # why the walk stopped early (a cycle, the depth cap)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def lineage(journal: Path, *, prefix: bool = True) -> Lineage:
    leaf = read_trajectory(Path(journal))
    segments = [Segment(leaf, None, None, None, None, None, None)]
    missing: list[str] = []
    notes: list[str] = []
    seen = {leaf.path.resolve()}
    current = leaf
    while prefix and current.origin is not None:
        origin = current.origin
        if len(segments) > MAX_DEPTH:
            notes.append("lineage deeper than %d journals: stopped at %s" % (MAX_DEPTH, current.run_id))
            break
        parent_path = relocate(origin.parent_journal, current.path)
        if parent_path is None:
            missing.append(origin.parent_journal)
            break
        if parent_path.resolve() in seen:
            notes.append(
                "fork.origin of %s names a journal already in this lineage (%s): cycle, stopped"
                % (current.run_id, parent_path)
            )
            break
        seen.add(parent_path.resolve())
        parent = read_trajectory(parent_path)
        actual = _sha256(parent_path)
        recorded = origin.parent_journal_sha256
        cut_turn = next((t.number for t in parent.turns if t.step == origin.branch_step), None)
        note = (
            None
            if cut_turn is not None
            else (
                "branch_step %d is not a turn boundary in this journal: rendering turns up to step %d"
                % (origin.branch_step, origin.branch_step)
            )
        )
        segments.insert(
            0,
            Segment(
                parent,
                origin.branch_step,
                cut_turn,
                None if recorded is None else actual == recorded,
                recorded,
                actual,
                note,
            ),
        )
        current = parent
    return Lineage(tuple(segments), tuple(missing), tuple(notes))


# --- shared pieces -----------------------------------------------------------------


def segment_turns(segment: Segment) -> list[Turn]:
    """The turns a segment renders: all of them for the leaf, up to the cut for a parent."""
    if segment.cut_step is None:
        return list(segment.trajectory.turns)
    return [t for t in segment.trajectory.turns if t.step is not None and t.step <= segment.cut_step]


def _turn_range(turns: list[Turn]) -> str:
    return "%d–%d" % (turns[0].number, turns[-1].number) if turns else "none"


def _step_range(turns: list[Turn]) -> str:
    steps = [c.step for t in turns for c in t.calls]
    return "%d–%d" % (min(steps), max(steps)) if steps else "none"


def _short(digest: str | None) -> str:
    return (digest or "")[:8] + "…" if digest else "unknown"


def checkpoint_line(capture: Capture | None) -> str | None:
    if capture is None:
        return None
    if not capture.snapshot_id or capture.reason == "failed":
        return "_no checkpoint for this turn (reason: %s)_" % capture.reason
    flags = [
        capture.reason if capture.reason != "clean" else "clean: unchanged since the previous capture",
        "disk-only" if capture.disk_only else "full",
    ]
    if capture.final:
        flags.append("final")
    if capture.uncertain:
        flags.append("uncertain")
    return "_checkpoint after step %d → snapshot `%s` (%s)_" % (
        capture.step,
        capture.snapshot_id,
        ", ".join(flags),
    )


def result_label(call: Call) -> str:
    if call.rejected:
        return "rejected (not executed)"
    if call.status is None:
        return "result: still running (no tool.finished)"
    parts = ["exit %s" % ("?" if call.exit_code is None else call.exit_code)]
    if call.timed_out:
        parts.append("timed out")
    if call.status != "ok":
        parts.append("status %s" % call.status)
    return "result (%s):" % ", ".join(parts)


def boundary(segment: Segment, child: Trajectory) -> dict:
    """What the cut between a parent segment and its branch says."""
    origin = child.origin
    parent = segment.trajectory
    rendered = segment_turns(segment)
    rest = [t for t in parent.turns if t not in rendered]
    return {
        "parent_run_id": parent.run_id,
        "branch_step": segment.cut_step,
        "cut_turn": segment.cut_turn,
        "snapshot_id": origin.snapshot_id if origin else None,
        "conversation_cut": origin.conversation_cut if origin else None,
        "branch_guidance": origin.branch_guidance if origin else None,
        "sha256_ok": segment.sha256_ok,
        "recorded_sha256": segment.recorded_sha256,
        "actual_sha256": segment.actual_sha256,
        "parent_turns_not_inherited": _turn_range(rest),
        "parent_steps_not_inherited": _step_range(rest),
        "note": segment.note,
    }


def _boundary_markdown(segment: Segment, child: Trajectory) -> list[str]:
    b = boundary(segment, child)
    turn = " (turn %d)" % b["cut_turn"] if b["cut_turn"] is not None else ""
    lines = [
        "---",
        "## ✂ BRANCH POINT — %s cut after step %s%s · snapshot `%s` · conversation cut `%s` · guidance %s"
        % (
            b["parent_run_id"],
            b["branch_step"],
            turn,
            b["snapshot_id"],
            b["conversation_cut"],
            b["branch_guidance"],
        ),
    ]
    if segment.sha256_ok is None:
        hash_note = "fork.origin recorded no parent sha256"
    elif segment.sha256_ok:
        hash_note = "parent journal sha256 %s matches fork.origin" % _short(segment.actual_sha256)
    else:
        hash_note = (
            "parent journal sha256 differs from fork.origin (recorded %s, file is %s): the parent was "
            "appended to or changed after the branch (a sweep or release marker does this); rendered anyway"
            % (_short(segment.recorded_sha256), _short(segment.actual_sha256))
        )
    lines.append(
        "parent steps %s (turns %s) are not part of this branch's history · %s"
        % (b["parent_steps_not_inherited"], b["parent_turns_not_inherited"], hash_note)
    )
    if segment.note:
        lines.append(segment.note)
    lines.append("---")
    return lines


def _missing_markdown(child: Trajectory, recorded: str) -> list[str]:
    origin = child.origin
    return [
        "---",
        "## ✂ BRANCH POINT — parent journal not found: %s (recorded; not found beside this journal "
        "either) · branch_step %s · snapshot %s · cut %s · guidance %s"
        % (recorded, origin.branch_step, origin.snapshot_id, origin.conversation_cut, origin.branch_guidance),
        "---",
    ]


def header_lines(segment: Segment, *, inherited: bool) -> list[str]:
    t = segment.trajectory
    turns = segment_turns(segment)
    actor = sum(1 for x in t.turns if x.source == "actor")
    reviewer = len(t.turns) - actor
    snapshots = sum(1 for c in t.captures if c.kind == "disk" and c.captured and c.snapshot_id)
    role = (
        "inherited prefix (turns %s, steps %s)" % (_turn_range(turns), _step_range(turns))
        if inherited
        else "this branch"
        if t.origin
        else "this run"
    )
    image = (
        "from snapshot `%s`" % t.image
        if t.origin and t.image
        else "image `%s`" % t.image
        if t.image
        else "image unknown"
    )
    if t.backend:
        where = "backend: %s · sandbox: `%s` · %s" % (t.backend, t.sandbox_id, image)
    else:
        backend, sandbox = receipt_identity(t.path)
        receipt = (
            "receipt says %s, sandbox `%s`" % (backend, sandbox)
            if backend or sandbox
            else "no receipt beside it"
        )
        where = "backend unknown (journal predates sandbox.created; %s) · %s" % (receipt, image)
    mins = minutes(t)
    lines = [
        "## %s — %s" % (t.run_id, role),
        "",
        "- journal: `%s`" % t.path,
        "- slot: `%s` %s · model: `%s`" % (t.slot or "unknown", t.slot_version or "", t.model or "unknown"),
        "- " + where,
        "- status: %s%s · turns: %d actor, %d reviewer · steps: %d · snapshots: %d%s"
        % (
            t.status or "open (no run.finished)",
            " — %s" % t.error if t.error else "",
            actor,
            reviewer,
            step_count(t),
            snapshots,
            " · %.1f min" % mins if mins is not None else "",
        ),
    ]
    if t.swept:
        lines.append("- ⚠ snapshots swept: ids no longer resolve (snapshots.swept in this journal)")
    if t.released:
        lines.append("- memory snapshot released: %s" % ", ".join(t.released))
    if t.uncertain:
        lines.append("- ⚠ execution.uncertain: a call's outcome is unknown; the rollout stopped there")
    return lines + [""]


def turn_lines(turn: Turn, *, max_output_chars: int) -> list[str]:
    if turn.assistant_turn is not None:
        title = (
            "### turn %d · REVIEWER-AUTHORED assistant turn (branch.assistant_turn; executed for real, "
            "not sampled from the model)" % turn.number
        )
    else:
        title = "### turn %d · %s" % (turn.number, turn.source)
    if turn.rejected:
        title += " · REJECTED (%s): no call executed" % (turn.rejected_source or "reason not recorded")
    elif turn.finished is None and not turn.calls:
        title += " · model call in flight"
    lines = [title, ""]
    if turn.started:
        lines += ["_started %s_" % turn.started, ""]
    if turn.thinking:
        lines += ["> thinking: " + turn.thinking.strip().replace("\n", "\n> "), ""]
    if turn.message:
        lines += [turn.message.strip(), ""]
    for call in turn.calls:
        lines += [
            "#### step %d · %s · call `%s`" % (call.step, call.name, call.call_id),
            "",
            "```bash",
            call.command.rstrip(),
            "```",
            "",
        ]
        if call.rejected:
            lines += ["**%s**" % result_label(call), ""]
            if call.output:
                lines += ["```", clip(call.output, max_output_chars).rstrip(), "```", ""]
        elif call.status is None:
            lines += ["**%s**" % result_label(call), ""]
        else:
            lines += [
                "**%s**" % result_label(call),
                "",
                "```",
                clip(call.output, max_output_chars).rstrip(),
                "```",
                "",
            ]
    checkpoint = checkpoint_line(turn.capture)
    if checkpoint:
        lines += [checkpoint, ""]
    return lines


def prompt_lines(l: Lineage) -> list[str]:
    root = l.segments[0].trajectory
    leaf = l.segments[-1].trajectory
    lines = ["## Prompt given to the agent", ""]
    if root.origin is not None and not root.task_prompt:
        # The root segment is itself a branch: its parent was not found, or not followed
        # (--no-prefix, a cycle, the depth cap); either way the prompt is in the parent.
        lines += [
            "_(parent journal not found; the task prompt is in the parent's run.started)_"
            if l.missing
            else "_(parent prefix not rendered: the task prompt is in the parent's run.started)_",
            "",
        ]
    elif root.task_prompt:
        lines += ["```", root.task_prompt.rstrip(), "```", ""]
    else:
        lines += ["_(empty)_", ""]
    if (
        leaf is not root
        and leaf.origin is not None
        and leaf.origin.branch_guidance == "user-hint"
        and leaf.task_prompt
    ):
        lines += [
            "## Branch note delivered as a user message",
            "",
            "```",
            leaf.task_prompt.rstrip(),
            "```",
            "",
        ]
    return lines


def closing_lines(t: Trajectory) -> list[str]:
    if t.status is None:
        torn = ", last line torn" if t.torn else ""
        return ["## Run still open — no run.finished; last event %s%s" % (t.last_ts, torn), ""]
    u = t.usage or {}
    lines = [
        "## Agent's closing message",
        "",
        t.final_text.rstrip() or "(empty)",
        "",
        "## Run finished",
        "",
        "- status: `%s`%s" % (t.status, " — %s" % t.error if t.error else ""),
        "- steps (tool calls): %d" % step_count(t)
        + (" · model calls: %d" % t.model_calls if t.model_calls is not None else ""),
        "- tokens: input %s · cached input %s · output %s · reasoning %s · cost: $%.2f"
        % (
            u.get("input_tokens"),
            u.get("cached_input_tokens"),
            u.get("output_tokens"),
            u.get("reasoning_output_tokens"),
            u.get("cost_usd") or 0,
        ),
    ]
    if t.swept:
        lines.append("- snapshots swept: the ids in this journal no longer resolve")
    return lines + [""]


def render_markdown(l: Lineage, *, max_output_chars: int) -> str:
    leaf = l.segments[-1].trajectory
    lines = ["# %s" % leaf.run_id, ""] + prompt_lines(l)
    if l.missing:
        lines += _missing_markdown(l.segments[0].trajectory, l.missing[0]) + [""]
    for note in l.notes:
        lines += ["_%s_" % note, ""]
    if len(l.segments) == 1 and leaf.origin is not None and not l.missing and not l.notes:
        # --no-prefix: say where the branch came from, once, before the leaf.
        origin = leaf.origin
        lines += [
            "---",
            "## ✂ BRANCH POINT — parent %s cut after step %d · snapshot `%s` · conversation cut `%s` "
            "· guidance %s (prefix not rendered)"
            % (
                origin.parent_journal,
                origin.branch_step,
                origin.snapshot_id,
                origin.conversation_cut,
                origin.branch_guidance,
            ),
            "---",
            "",
        ]
    for index, segment in enumerate(l.segments):
        lines += header_lines(segment, inherited=segment.cut_step is not None)
        for turn in segment_turns(segment):
            lines += turn_lines(turn, max_output_chars=max_output_chars)
        if index + 1 < len(l.segments):
            lines += _boundary_markdown(segment, l.segments[index + 1].trajectory) + [""]
    lines += closing_lines(leaf)
    return "\n".join(lines)


def _call_json(call: Call, max_output_chars: int) -> dict:
    return {
        "step": call.step,
        "call_id": call.call_id,
        "name": call.name,
        "command": call.command,
        "args": call.args,
        "status": call.status,
        "exit_code": call.exit_code,
        "timed_out": call.timed_out,
        "rejected": call.rejected,
        "output": clip(call.output, max_output_chars),
    }


def _capture_json(capture: Capture | None) -> dict | None:
    if capture is None:
        return None
    return {
        "snapshot_id": capture.snapshot_id,
        "step": capture.step,
        "reason": capture.reason,
        "captured": capture.captured,
        "disk_only": capture.disk_only,
        "final": capture.final,
        "uncertain": capture.uncertain,
        "branch_point": capture.branch_point,
    }


def _segment_json(segment: Segment, max_output_chars: int) -> dict:
    t = segment.trajectory
    return {
        "run_id": t.run_id,
        "journal": str(t.path),
        "inherited": segment.cut_step is not None,
        "cut_step": segment.cut_step,
        "cut_turn": segment.cut_turn,
        "backend": t.backend,
        "sandbox_id": t.sandbox_id,
        "image": t.image,
        "status": t.status,
        "error": t.error,
        "swept": t.swept,
        "turns": [
            {
                "turn": turn.number,
                "source": turn.source,
                "turn_id": turn.turn_id,
                "started": turn.started,
                "thinking": turn.thinking,
                "message": turn.message,
                "assistant_turn": turn.assistant_turn,
                "calls": [_call_json(c, max_output_chars) for c in turn.calls],
                "checkpoint": _capture_json(turn.capture),
                "closed": turn.closed,
                "rejected": turn.rejected,
                "rejected_source": turn.rejected_source,
            }
            for turn in segment_turns(segment)
        ],
    }


def render_json(l: Lineage, *, max_output_chars: int) -> dict:
    leaf = l.segments[-1].trajectory
    segments: list[dict] = []
    for index, segment in enumerate(l.segments):
        segments.append(_segment_json(segment, max_output_chars))
        if index + 1 < len(l.segments):
            segments.append({"boundary": boundary(segment, l.segments[index + 1].trajectory)})
    return {
        "journal": str(leaf.path),
        "run_id": leaf.run_id,
        "prompt": l.segments[0].trajectory.task_prompt,
        "prefix": len(l.segments) > 1 or leaf.origin is None,
        "lineage": [
            {
                "run_id": s.trajectory.run_id,
                "journal": str(s.trajectory.path),
                "cut_step": s.cut_step,
                "cut_turn": s.cut_turn,
                "sha256_ok": s.sha256_ok,
                "recorded_sha256": s.recorded_sha256,
                "actual_sha256": s.actual_sha256,
            }
            for s in l.segments
        ],
        "missing_parents": list(l.missing),
        "notes": list(l.notes),
        "segments": segments,
        "final_text": leaf.final_text,
        "status": leaf.status,
        "error": leaf.error,
        "usage": leaf.usage,
        "model_calls": leaf.model_calls,
        "tool_calls": leaf.tool_calls,
        "swept": leaf.swept,
        "torn": leaf.torn,
        "last_ts": leaf.last_ts,
    }

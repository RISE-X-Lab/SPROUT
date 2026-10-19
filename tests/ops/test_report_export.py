"""``ops.report.export``: one trajectory as markdown or json, with a branch stitched onto its parents."""

from __future__ import annotations

import json
import shutil

import pytest

from ops.report.export import lineage, render_json, render_markdown
from tests.ops.trees import (
    Step,
    Turn,
    actor,
    fork_origin,
    numbered,
    reviewer,
    sha256,
    write_journal,
    write_json,
)
from tests.support.paths import FIXTURES

REAL_OUTPUT = FIXTURES / "inspect" / "abs-module-cache-flags"


def headings(text):
    """The markdown headings of an export, in order."""
    return [line for line in text.splitlines() if line.startswith("#")]


def lines_starting(text, prefix):
    """The lines of ``text`` that start with ``prefix``."""
    return [line for line in text.splitlines() if line.startswith(prefix)]


def segments(chain, *fields):
    """Each segment of a lineage as a tuple of ``fields``: ``path`` and ``run_id`` are its trajectory's."""
    return [
        tuple(getattr(s.trajectory, f) if f in ("path", "run_id") else getattr(s, f) for f in fields)
        for s in chain.segments
    ]


def turn_numbers(segment):
    return [turn["turn"] for turn in segment["turns"]]


def write_events(path, events):
    """A journal of exactly ``events``, numbered in order."""
    path.write_text(
        "".join(
            json.dumps({"run_id": "r", "seq": seq, **event}) + "\n" for seq, event in enumerate(events, 1)
        )
    )
    return path


def parent_of_five_turns(tmp_path):
    return write_journal(
        tmp_path / "p" / "agent" / "trajectory.jsonl", run_id="t-parent", sandbox="par", turns=numbered(5)
    )


def branch_of(tmp_path, parent, *, step=3, sha="match", name="b", **journal):
    """A branch of ``parent`` after ``step``: a reviewer turn, then an actor turn."""
    origin = fork_origin(
        parent, step=step, snapshot_id="par-%04d" % step, sha=sha256(parent) if sha == "match" else sha
    )
    return write_journal(
        tmp_path / name / "agent" / "trajectory.jsonl",
        run_id="t-" + name,
        sandbox="brn",
        turns=journal.pop("turns", [reviewer("go test ./..."), actor("echo done")]),
        origin=origin,
        prompt=journal.pop("prompt", ""),
        image="par-%04d" % step,
        **journal,
    )


def test_a_branch_renders_its_parent_up_to_the_branch_step_then_its_own_turns(tmp_path):
    branch = branch_of(tmp_path, parent_of_five_turns(tmp_path), step=3)

    text = render_markdown(lineage(branch), max_output_chars=0)

    assert headings(text) == [
        "# t-b",
        "## Prompt given to the agent",
        "## t-parent — inherited prefix (turns 1–3, steps 1–3)",
        "### turn 1 · actor",
        "#### step 1 · shell · call `c1`",
        "### turn 2 · actor",
        "#### step 2 · shell · call `c2`",
        "### turn 3 · actor",
        "#### step 3 · shell · call `c3`",
        "## ✂ BRANCH POINT — t-parent cut after step 3 (turn 3) · snapshot `par-0003` · "
        "conversation cut `t3` · guidance assistant-turn",
        "## t-b — this branch",
        "### turn 1 · REVIEWER-AUTHORED assistant turn (branch.assistant_turn; executed for real, "
        "not sampled from the model)",
        "#### step 1 · shell · call `c1`",
        "### turn 2 · actor",
        "#### step 2 · shell · call `c2`",
        "## Agent's closing message",
        "## Run finished",
    ]


def test_the_branch_point_names_the_parent_steps_left_out_and_a_matching_parent_hash(tmp_path):
    parent = parent_of_five_turns(tmp_path)
    branch = branch_of(tmp_path, parent, step=3)

    text = render_markdown(lineage(branch), max_output_chars=0)

    assert lines_starting(text, "parent steps") == [
        "parent steps 4–5 (turns 4–5) are not part of this branch's history · "
        "parent journal sha256 %s… matches fork.origin" % sha256(parent)[:8]
    ]


def test_a_parent_appended_to_after_the_branch_is_rendered_anyway_with_both_hashes(tmp_path):
    """A sweep appends a marker to the parent, so its hash differs from the one that fork.origin recorded."""
    parent = parent_of_five_turns(tmp_path)
    branch = branch_of(tmp_path, parent, step=3)
    recorded = sha256(parent)
    swept = {"type": "snapshots.swept", "run_id": "sweep", "backend": "podman", "deleted": 5, "failed": 0}
    parent.write_text(parent.read_text() + json.dumps(swept) + "\n")

    text = render_markdown(lineage(branch), max_output_chars=0)

    assert lines_starting(text, "parent steps") == [
        "parent steps 4–5 (turns 4–5) are not part of this branch's history · parent journal sha256 "
        "differs from fork.origin (recorded %s…, file is %s…): the parent was appended to or changed "
        "after the branch (a sweep or release marker does this); rendered anyway"
        % (recorded[:8], sha256(parent)[:8])
    ]


def test_a_swept_journal_warns_in_its_header_that_its_snapshot_ids_no_longer_resolve(tmp_path):
    swept = {"type": "snapshots.swept", "run_id": "sweep", "backend": "podman", "deleted": 1, "failed": 0}
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(1), tail=(swept,))

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert lines_starting(text, "- ⚠") == [
        "- ⚠ snapshots swept: ids no longer resolve (snapshots.swept in this journal)"
    ]


def test_a_parent_whose_root_moved_is_found_beside_the_branch(tmp_path):
    root = tmp_path / "now"
    parent = write_journal(
        root / "baseline" / "attempt-1" / "trials" / "t-parent" / "agent" / "trajectory.jsonl",
        turns=numbered(4),
        sandbox="par",
    )
    origin = fork_origin(
        "/path/to/old/baseline/attempt-1/trials/t-parent/agent/trajectory.jsonl",
        step=2,
        snapshot_id="par-0002",
        sha=sha256(parent),
    )
    branch = write_journal(
        root / "sprout" / "trials" / "t-r1b1" / "agent" / "trajectory.jsonl",
        turns=[actor("ls -la")],
        origin=origin,
    )

    chain = lineage(branch)

    assert segments(chain, "path", "cut_step", "sha256_ok") == [(parent, 2, True), (branch, None, None)]


def test_a_missing_parent_is_reported_and_the_branch_is_still_rendered(tmp_path):
    gone = tmp_path / "nowhere" / "trials" / "t-parent" / "agent" / "trajectory.jsonl"
    branch = write_journal(
        tmp_path / "b" / "agent" / "trajectory.jsonl",
        turns=[reviewer("go test ./...")],
        origin=fork_origin(gone, step=3, snapshot_id="par-0003", sha="a" * 64),
        prompt="",
    )

    text = render_markdown(lineage(branch), max_output_chars=0)

    assert lines_starting(text, "## ✂") + lines_starting(text, "_(") == [
        "## ✂ BRANCH POINT — parent journal not found: %s (recorded; not found beside this journal either) "
        "· branch_step 3 · snapshot par-0003 · cut t3 · guidance assistant-turn" % gone,
        "_(parent journal not found; the task prompt is in the parent's run.started)_",
    ]


def test_a_fork_origin_that_names_its_own_journal_stops_the_lineage_as_a_cycle(tmp_path):
    path = tmp_path / "s" / "agent" / "trajectory.jsonl"
    write_journal(
        path,
        run_id="t-self",
        turns=[actor("ls -la")],
        origin=fork_origin(path, step=1, snapshot_id="x-0001", sha=None),
    )

    chain = lineage(path)

    assert (len(chain.segments), chain.notes) == (
        1,
        ("fork.origin of t-self names a journal already in this lineage (%s): cycle, stopped" % path,),
    )


def test_a_branch_of_a_branch_is_cut_at_each_ancestors_branch_step(tmp_path):
    grand = parent_of_five_turns(tmp_path)
    middle = branch_of(
        tmp_path,
        grand,
        step=3,
        name="mid",
        turns=[reviewer("go test ./..."), actor("echo mid"), actor("echo late")],
    )
    origin = fork_origin(middle, step=2, snapshot_id="brn-0002", sha=sha256(middle))
    leaf = write_journal(
        tmp_path / "leaf" / "agent" / "trajectory.jsonl",
        run_id="t-leaf",
        turns=[actor("ls -la")],
        origin=origin,
    )

    chain = lineage(leaf)

    assert segments(chain, "run_id", "cut_step", "cut_turn") == [
        ("t-parent", 3, 3),
        ("t-mid", 2, 2),
        ("t-leaf", None, None),
    ]


def test_without_the_prefix_a_branch_is_rendered_alone_after_a_note_of_its_origin(tmp_path):
    parent = parent_of_five_turns(tmp_path)
    branch = branch_of(tmp_path, parent, step=3)

    text = render_markdown(lineage(branch, prefix=False), max_output_chars=0)

    assert lines_starting(text, "## ✂") + lines_starting(text, "_(") + lines_starting(text, "## t-") == [
        "## ✂ BRANCH POINT — parent %s cut after step 3 · snapshot `par-0003` · conversation cut `t3` "
        "· guidance assistant-turn (prefix not rendered)" % parent,
        "_(parent prefix not rendered: the task prompt is in the parent's run.started)_",
        "## t-b — this branch",
    ]


def test_a_long_tool_result_keeps_its_head_and_tail_within_max_output_chars(tmp_path):
    journal = write_journal(
        tmp_path / "j.jsonl", turns=[Turn((Step("cat log.txt", "H" * 400 + "MIDDLE" + "T" * 400),))]
    )

    text = render_markdown(lineage(journal), max_output_chars=90)

    assert "H" * 60 + "\n… [716 chars elided] …\n" + "T" * 30 + "\n```" in text


def test_an_open_run_ends_with_its_last_event_and_its_model_call_in_flight(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2), status=None, open_turn=True, torn=True)

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert headings(text)[-2:] == [
        "### turn 3 · actor · model call in flight",
        "## Run still open — no run.finished; last event 2026-09-30T20:00:23.000Z, last line torn",
    ]


def test_a_user_hint_branch_shows_its_rendered_hint_as_a_user_message(tmp_path):
    parent = parent_of_five_turns(tmp_path)
    origin = fork_origin(
        parent,
        step=2,
        snapshot_id="par-0002",
        sha=sha256(parent),
        branch_guidance="user-hint",
        actor_hint="HINT",
    )
    branch = write_journal(
        tmp_path / "b" / "agent" / "trajectory.jsonl",
        turns=[actor("ls -la")],
        origin=origin,
        prompt="Rendered: HINT",
    )

    text = render_markdown(lineage(branch), max_output_chars=0)

    assert "## Branch note delivered as a user message\n\n```\nRendered: HINT\n```" in text


@pytest.mark.parametrize(
    "rejection,heading",
    [
        (
            {"source": "malformed_tool_calls"},
            "### turn 1 · actor · REJECTED (malformed_tool_calls): no call executed",
        ),
        ({}, "### turn 1 · actor · REJECTED (reason not recorded): no call executed"),
    ],
    ids=["with-source", "without-source"],
)
def test_a_rejected_turn_says_why_no_call_ran(tmp_path, rejection, heading):
    journal = write_events(
        tmp_path / "j.jsonl",
        [
            {"type": "run.started", "slot": "mini-swe-agent", "model": "m", "task_prompt": "P"},
            {"type": "turn.started", "turn": 1, "source": "actor", "ts": "2026-10-01T00:00:00Z"},
            {"type": "mini.turn.rejected", "turn_id": "t1", "executed": False, **rejection},
            {
                "type": "turn.completed",
                "turn": 1,
                "source": "actor",
                "rejected": True,
                "ts": "2026-10-01T00:00:01Z",
            },
            {"type": "run.finished", "status": "completed", "usage": {}},
        ],
    )

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert lines_starting(text, "### turn") == [heading]


def test_the_closing_section_gives_the_status_steps_model_calls_and_tokens(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2))

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert text.split("## Run finished\n\n")[1] == (
        "- status: `completed`\n"
        "- steps (tool calls): 2 · model calls: 2\n"
        "- tokens: input 200 · cached input 100 · output 20 · reasoning 10 · cost: $0.00\n"
    )


def test_a_journal_from_before_sandbox_created_names_the_backend_of_its_trial_receipt(tmp_path):
    trial = tmp_path / "trials" / "t-old"
    journal = write_journal(trial / "agent" / "trajectory.jsonl", turns=numbered(1), backend=None)
    write_json(trial / "t-old__env.agentenv.json", {"sandbox_id": "01a0-vm", "overlay": None})

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert lines_starting(text, "- backend") == [
        "- backend unknown (journal predates sandbox.created; receipt says microvm, sandbox `01a0-vm`) "
        "· image unknown"
    ]


def test_the_json_export_puts_a_boundary_between_the_parent_and_the_branch(tmp_path):
    parent = parent_of_five_turns(tmp_path)
    branch = branch_of(tmp_path, parent, step=3)

    document = render_json(lineage(branch), max_output_chars=0)

    parent_segment, boundary, leaf = document["segments"]
    assert (parent_segment["run_id"], turn_numbers(parent_segment), leaf["run_id"]) == (
        "t-parent",
        [1, 2, 3],
        "t-b",
    )
    assert boundary == {
        "boundary": {
            "parent_run_id": "t-parent",
            "branch_step": 3,
            "cut_turn": 3,
            "snapshot_id": "par-0003",
            "conversation_cut": "t3",
            "branch_guidance": "assistant-turn",
            "sha256_ok": True,
            "recorded_sha256": sha256(parent),
            "actual_sha256": sha256(parent),
            "parent_turns_not_inherited": "4–5",
            "parent_steps_not_inherited": "4–5",
            "note": None,
        }
    }


def test_the_real_branch_journal_exports_its_reviewer_turn_and_names_its_parent_as_not_found(tmp_path):
    """The fixture is a journal that a run wrote. Its parent's recorded path does not exist on any host."""
    journal = tmp_path / "trajectory.jsonl"
    shutil.copyfile(REAL_OUTPUT / "r1b1.trajectory.jsonl", journal)

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert lines_starting(text, "### turn") + lines_starting(text, "#### step 1 ") == [
        "### turn 1 · REVIEWER-AUTHORED assistant turn (branch.assistant_turn; executed for real, "
        "not sampled from the model)",
        "### turn 2 · actor",
        "### turn 3 · actor",
        "#### step 1 · shell · call `rev136-abs-test`",
    ]
    assert lines_starting(text, "## ✂") == [
        "## ✂ BRANCH POINT — parent journal not found: "
        "/path/to/ash-runs/qwen/deepswe-full-ce125c7d3b7c-podman/"
        "baseline/tasks/abs-module-cache-flags/baseline/attempt-1/trials/abs-module-cache-flags-parent/agent/"
        "trajectory.jsonl (recorded; not found beside this journal either) · branch_step 136 · snapshot "
        "sprout-b2867d643e67-0105 · cut bf5a28a1cb6144788f8c700d16ea7fd4 · guidance assistant-turn"
    ]


def test_a_turn_shows_its_thinking_its_message_its_calls_and_its_checkpoint_in_that_order(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2))

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert text.split("### turn 1 · actor\n\n")[1].split("### turn 2")[0] == (
        "_started 2026-09-30T20:00:04.000Z_\n\n"
        "> thinking: THINKING 1\n\n"
        "MESSAGE 1\n\n"
        "#### step 1 · shell · call `c1`\n\n"
        "```bash\necho ACTION_1\n```\n\n"
        "**result (exit 0):**\n\n"
        "```\nRESULT_1\n```\n\n"
        "_checkpoint after step 1 → snapshot `sbx-0001` (captured, disk-only)_\n\n"
    )


def test_the_checkpoint_after_the_last_turn_is_marked_final(tmp_path):
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2))

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert lines_starting(text, "_checkpoint") == [
        "_checkpoint after step 1 → snapshot `sbx-0001` (captured, disk-only)_",
        "_checkpoint after step 2 → snapshot `sbx-0002` (captured, disk-only, final)_",
    ]


def test_a_journal_whose_last_call_has_an_unknown_outcome_warns_in_its_header(tmp_path):
    uncertain = {"type": "execution.uncertain", "call_id": "c2", "step": 2, "kind": "runtime_timeout"}
    journal = write_journal(tmp_path / "j.jsonl", turns=numbered(2), tail=(uncertain,))

    text = render_markdown(lineage(journal), max_output_chars=0)

    assert lines_starting(text, "- ⚠") == [
        "- ⚠ execution.uncertain: a call's outcome is unknown; the rollout stopped there"
    ]

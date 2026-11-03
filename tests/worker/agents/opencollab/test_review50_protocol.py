# ruff: noqa: E402
"""The reviewed tool interface must execute and finish through native Single2."""

import json
import subprocess
from pathlib import Path

import pytest

pytest.importorskip("opencollab")

from sprout.contract.agents.history import load_prefix, training_messages
from sprout.contract.messages import clean_messages, export_tools
from sprout.worker.agents.history import available_branch_points, reference_at
from sprout.worker.agents.opencollab.review50 import build_tools, definition
from sprout.worker.run.journal import JournalWriter, read_journal
from tests.worker.agents.opencollab.test_opencollab import Model, call, response, run


def repository(root: Path) -> Path:
    root.mkdir()
    subprocess.run(["git", "init", "-q", str(root)], check=True)
    (root / "source.txt").write_text("original\n")
    subprocess.run(["git", "-C", str(root), "add", "."], check=True)
    # This is isolated test-fixture history, never a benchmark's history.
    subprocess.run(
        [
            "git",
            "-C",
            str(root),
            "-c",
            "user.name=Fixture",
            "-c",
            "user.email=fixture@example.invalid",
            "commit",
            "-qm",
            "fixture",
        ],
        check=True,
    )
    return root


def exported(journal):
    events = read_journal(journal)
    ref = next(e for e in events if e["type"] == "session.ref")
    rows = [json.loads(line) for line in Path(ref["transcript_path"]).read_text().splitlines()]
    return clean_messages(training_messages(rows)), events


def test_review50_file_tools_submission_and_closed_history(tmp_path):
    root = repository(tmp_path / "repo")
    model = Model(
        [
            response(call("read", "read_file", filepath="source.txt", start_line=1, end_line=1)),
            response(
                call("edit", "edit_file", filepath="source.txt", old_string="original", new_string="fixed")
            ),
            response(call("write", "write_file", filepath="new.txt", content="new file\n")),
            response(call("status", "get_status")),
            response(call("command", "run_command", command="printf verified")),
            response(call("submit", "submit_patch")),
        ]
    )
    result, _, journal = run(root, tmp_path / "run", model, protocol="review50-v1")
    assert result.status == "completed", result.error
    assert len(model.requests) == 6  # no extra model request needed to end
    messages, events = exported(journal)
    results = {m["tool_call_id"]: json.loads(m["content"]) for m in messages if m["role"] == "tool"}
    assert results["read"]["content"] == "original\n"
    assert results["edit"]["occurrences"] == 1
    assert results["write"]["size"] == 9
    assert results["status"]["grading"] == "pending official verifier"
    assert results["status"]["budget"]["agent_tool_calls"] == 4
    assert results["status"]["budget"]["max_tool_calls"] is None
    assert "new.txt" in results["status"]["patch"]["working_tree_status"]
    assert "source.txt" in results["status"]["patch"]["tracked_diff_stat"]
    assert results["command"]["stdout"] == "verified"
    assert results["submit"]["status"] == "submitted"
    assert results["submit"]["patch_chars"] > 0
    patch = subprocess.run(
        ["git", "-C", str(root), "diff", "HEAD"], check=True, text=True, capture_output=True
    ).stdout
    assert "new file" in patch and "+fixed" in patch
    assert export_tools(events) == definition()["tools"]
    assert messages[0]["content"] == definition()["system_prompt"]
    assert next(e for e in events if e["type"] == "opencollab.termination")["kind"] == "explicit_submission"
    assert available_branch_points(journal)


def test_review50_submission_rejects_later_calls_but_keeps_pairs(tmp_path):
    root = repository(tmp_path / "repo")
    model = Model(
        [
            response(
                call("s", "submit_patch"),
                call("later", "write_file", filepath="forbidden.txt", content="not executed"),
            )
        ]
    )
    result, _, journal = run(root, tmp_path / "run", model, protocol="review50-v1")
    assert result.status == "completed", result.error
    assert not (root / "forbidden.txt").exists()
    messages, _ = exported(journal)
    assert {m["tool_call_id"] for m in messages if m["role"] == "tool"} == {"s", "later"}


def test_failed_submission_is_not_a_successful_finish(tmp_path):
    root = tmp_path / "not-a-repo"
    root.mkdir()
    model = Model([response(call("s", "submit_patch")), response(content="No patch available.")])
    result, _, journal = run(root, tmp_path / "run", model, protocol="review50-v1")
    assert result.status == "completed", result.error
    messages, events = exported(journal)
    assert json.loads(next(m["content"] for m in messages if m["role"] == "tool"))["status"] == "error"
    assert next(e for e in events if e["type"] == "opencollab.termination")["kind"] == "final_response"
    assert len(model.requests) == 2


def test_review50_branch_inherits_protocol_and_can_submit(tmp_path):
    root = repository(tmp_path / "repo")
    result, _, journal = run(
        root,
        tmp_path / "parent",
        Model([response(call("r", "read_file", filepath="source.txt")), response(content="Done.")]),
        protocol="review50-v1",
    )
    assert result.status == "completed", result.error
    step, point = min(available_branch_points(journal).items())
    prefix = reference_at(journal, step, point.session_ckpt)
    assert load_prefix(prefix)[0]["tool_protocol"] == "review50-v1"
    reviewer = {
        "role": "assistant",
        "content": "I will make a focused correction.",
        "tool_calls": [
            call("review", "edit_file", filepath="source.txt", old_string="original", new_string="branch")
        ],
    }
    model = Model([response(call("s", "submit_patch"))])
    child, _, child_journal = run(
        root, tmp_path / "child", model, prefix=prefix, reviewer=reviewer, protocol="review50-v1"
    )
    assert child.status == "completed", child.error
    assert (root / "source.txt").read_text() == "branch\n"
    exported(child_journal)
    with pytest.raises(ValueError, match="protocol differs"):
        run(root, tmp_path / "wrong", Model([]), prefix=prefix)


def test_file_errors_and_repeated_edits_do_not_mutate_ambiguously(tmp_path):
    root = repository(tmp_path / "repo")
    (root / "source.txt").write_text("same same\n")
    result, _, journal = run(
        root,
        tmp_path / "run",
        Model(
            [
                response(
                    call(
                        "ambiguous",
                        "edit_file",
                        filepath="source.txt",
                        old_string="same",
                        new_string="different",
                    )
                ),
                response(call("outside", "read_file", filepath="../outside.txt")),
                response(content="Stopped."),
            ]
        ),
        protocol="review50-v1",
    )
    assert result.status == "completed", result.error
    messages, _ = exported(journal)
    assert all(json.loads(m["content"])["status"] == "error" for m in messages if m["role"] == "tool")
    assert (root / "source.txt").read_text() == "same same\n"


def test_compatibility_tools_have_independent_state():
    first, second = build_tools(), build_tools()
    assert [t.to_openai_schema() for t in first] == definition()["tools"]
    assert all(a is not b for a, b in zip(first, second, strict=True))


def test_status_uses_enforced_rollout_counts_and_does_not_invent_a_tool_limit(tmp_path):
    import time

    from sprout.worker.run.control import RunControl
    from sprout.worker.run.rollout import RolloutControls

    with JournalWriter(tmp_path / "controls.jsonl") as journal:
        controls = RolloutControls(
            {"model_endpoint": "http://127.0.0.1:1234", "deadline_at": time.time() + 30, "max_turns": 8},
            journal,
            RunControl(),
        )
        controls.prepare_model_request({})
        result = controls.status()
        assert result["model_calls"] == 1 and result["remaining_turns"] == 7
        assert result["runtime_tool_calls"] == 0 and result["max_tool_calls"] is None

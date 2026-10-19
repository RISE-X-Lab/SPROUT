"""``resolve_workspace`` fixes where mini works in the sandbox before any model action runs.

A stub session answers the probes, ``pwd -P`` and ``git rev-parse --show-toplevel``, by command and
working directory. A probe that it has no answer for fails, as outside a repository.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.agents.mini_swe.sandbox import resolve_workspace
from sprout.worker.run.journal import JournalWriter
from sprout.worker.tools.result import CommandOutcome, ToolResult

PWD = "pwd -P"
TOPLEVEL = "git rev-parse --show-toplevel"
NOT_A_REPOSITORY = ToolResult(
    False, "", outcome=CommandOutcome(exit_code=128, stderr="fatal: not a git repository")
)


def printed(text: str) -> ToolResult:
    return ToolResult(True, text + "\n")


class ProbeSession:
    """A sandbox session that answers each probe from ``answers``, keyed by command and working directory."""

    def __init__(self, answers: dict[tuple[str, str | None], ToolResult]) -> None:
        self.answers = answers

    def execute(self, name: str, args: dict[str, Any], timeout: float = 30) -> ToolResult:
        return self.answers.get((args["command"], args.get("working_dir")), NOT_A_REPOSITORY)


def resolve(tmp_path: Path, answers: dict[tuple[str, str | None], ToolResult], extra: dict[str, Any]) -> Any:
    with JournalWriter(tmp_path / "journal.jsonl") as journal:
        return resolve_workspace(ProbeSession(answers), extra, journal)


def prefix_reference(path: Path, workspace: dict[str, Any]) -> dict[str, Any]:
    """The reference of a native history at ``path`` that records ``workspace`` and one closed turn."""
    header = {
        "type": "mini.session",
        "format": "sprout-mini-v1",
        "mini_version": "2.4.6",
        "session_id": "parent",
        "workspace": workspace,
    }
    turn = {"type": "mini.turn", "turn_id": "turn-1", "call_ids": ["call-1"], "source": "actor"}
    data = (json.dumps(header) + "\n" + json.dumps(turn) + "\n").encode()
    path.write_bytes(data)
    return {"path": str(path), "sha256": hashlib.sha256(data).hexdigest()}


def test_without_a_configured_directory_mini_works_where_the_shell_starts(tmp_path):
    extra = {"native_home": "/path/to/native-home"}

    resolved = resolve(tmp_path, {(PWD, None): printed("/testbed")}, extra)

    assert resolved == {
        "native_home": "/path/to/native-home",
        "mini": {"environment": {"cwd": "/testbed"}},
        "mini_workspace": {"cwd": "/testbed", "repository_dir": None},
    }


def test_a_shell_that_starts_inside_a_repository_puts_mini_at_the_repository_root(tmp_path):
    answers = {
        (PWD, None): printed("/testbed/src"),
        (TOPLEVEL, "/testbed/src"): printed("/testbed"),
        (PWD, "/testbed"): printed("/testbed"),
    }

    resolved = resolve(tmp_path, answers, {})

    assert resolved["mini_workspace"] == {"cwd": "/testbed", "repository_dir": "/testbed"}


def test_a_configured_directory_inside_a_repository_stays_where_it_is(tmp_path):
    answers = {
        (PWD, "/testbed/src"): printed("/testbed/src"),
        (TOPLEVEL, "/testbed/src"): printed("/testbed"),
    }

    resolved = resolve(tmp_path, answers, {"mini": {"environment": {"cwd": "/testbed/src"}}})

    assert resolved["mini_workspace"] == {"cwd": "/testbed/src", "repository_dir": "/testbed"}


def test_a_branch_works_in_the_workspace_of_its_parent(tmp_path):
    reference = prefix_reference(tmp_path / "parent.jsonl", {"cwd": "/testbed/src", "repository_dir": None})
    answers = {(PWD, "/testbed/src"): printed("/testbed/src")}

    resolved = resolve(tmp_path, answers, {"native_prefix": reference})

    assert resolved["mini_workspace"] == {"cwd": "/testbed/src", "repository_dir": None}


def test_a_relative_configured_directory_is_refused(tmp_path):
    with pytest.raises(
        ValueError, match="^mini working directory must be an absolute path or omitted for discovery$"
    ):
        resolve(tmp_path, {}, {"mini": {"environment": {"cwd": "testbed"}}})


def test_a_probe_that_fails_is_refused_with_its_reason(tmp_path):
    failed = ToolResult(False, "", "failed to start shell: chdir /missing: no such file or directory")

    with pytest.raises(
        ValueError,
        match="^mini working-directory probe failed: failed to start shell: chdir /missing: no such file or "
        "directory$",
    ):
        resolve(tmp_path, {(PWD, "/missing"): failed}, {"mini": {"environment": {"cwd": "/missing"}}})


@pytest.mark.parametrize("output", ["testbed", "/testbed\n/other"], ids=["relative", "two-lines"])
def test_a_probe_that_prints_no_single_absolute_path_is_refused(tmp_path, output):
    with pytest.raises(ValueError, match="^mini working-directory probe returned an invalid absolute path"):
        resolve(tmp_path, {(PWD, None): printed(output)}, {})


def test_a_branch_cannot_name_another_directory_than_its_parents(tmp_path):
    reference = prefix_reference(tmp_path / "parent.jsonl", {"cwd": "/testbed", "repository_dir": None})
    extra = {"native_prefix": reference, "mini": {"environment": {"cwd": "/another-repository"}}}

    with pytest.raises(
        ValueError, match="^mini branch working directory differs from its inherited workspace$"
    ):
        resolve(tmp_path, {(PWD, "/another-repository"): printed("/another-repository")}, extra)


def test_a_branch_whose_sandbox_differs_from_its_parents_workspace_is_refused(tmp_path):
    """The restored snapshot must give the parent's repository too."""
    reference = prefix_reference(tmp_path / "parent.jsonl", {"cwd": "/testbed", "repository_dir": None})
    answers = {(PWD, "/testbed"): printed("/testbed"), (TOPLEVEL, "/testbed"): printed("/testbed")}

    with pytest.raises(
        ValueError, match="^mini restored workspace differs from the parent's recorded workspace$"
    ):
        resolve(tmp_path, answers, {"native_prefix": reference})

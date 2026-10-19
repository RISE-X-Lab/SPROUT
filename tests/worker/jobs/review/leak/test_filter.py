"""The RL filter over a sprout run root: which branches it reads, what is hidden, and which may enter RL.

``python -m sprout.worker.jobs.review.leak.judge`` builds its own Jev client, so these tests give it a
client on ``httpx.MockTransport`` in place of ``httpx.Client``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.worker.jobs.review.leak import judge as leak_judge
from sprout.worker.jobs.review.leak.judge import (
    HiddenText,
    main,
    patch_files,
    read_key,
    sprout_branches,
    task_hidden_tests,
)

PATCH = """diff --git a/tests/test_links.py b/tests/test_links.py
--- a/tests/test_links.py
+++ b/tests/test_links.py
@@ -1,2 +1,3 @@
 import links
+def test_balanced_parentheses():
+    assert convert("![Alt](<my image (1).png>)") == "![[my image (1).png|Alt]]"
"""
STATEMENT = "Convert Markdown links to wiki links."
TARGET = "tests/test_links.py::test_balanced_parentheses"
COPIES_THE_FIXTURE = 'echo "![Alt](<my image (1).png>)"'
NAMES_THE_TEST = "pytest " + TARGET


def write(path: Path, content: str | dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content if isinstance(content, str) else json.dumps(content))


def reviewer_turn(command: str) -> dict[str, Any]:
    call = {
        "id": "reviewer-r1-b1-c1",
        "type": "function",
        "function": {"name": "bash", "arguments": json.dumps({"command": command})},
    }
    return {"role": "assistant", "content": "Check the parenthesised case.", "tool_calls": [call]}


def sprout_run(root: Path, tasks: Path, *, command: str, reward: float = 1.0, inserted: bool = True) -> Path:
    """A run root with one task: its parent, graded before round 1, and one round-1 branch ``links-r1b1-fix``.

    The branch's history holds the reviewer's turn when ``inserted``."""
    write(tasks / "links" / "instruction.md", STATEMENT)
    write(tasks / "links" / "tests" / "test.patch", PATCH)
    write(tasks / "links" / "tests" / "config.json", {"f2p_node_ids": [TARGET]})
    parent = root / "baseline" / "trials" / "links-parent"
    write(parent / "verifier" / "test-stdout.txt", "FAILED " + TARGET)
    trial = root / "tasks" / "links" / "sprout" / "trials" / "links-r1b1-fix"
    write(trial / "result.json", {"verifier_result": {"rewards": {"reward": reward}}})
    history = [{"role": "system", "content": "s"}, {"role": "user", "content": STATEMENT}]
    turn = reviewer_turn(command) if inserted else {"role": "assistant", "content": "Mine.", "tool_calls": []}
    write(
        trial / "agent" / "mini-native" / "session.traj.json",
        {"messages": [*history, turn, {"role": "tool", "content": "ok"}]},
    )
    attempts = [
        {"name": "parent", "round": 0, "journal": str(parent / "agent" / "trajectory.jsonl")},
        {"name": "r1b1-fix", "round": 1, "journal": str(trial / "agent" / "trajectory.jsonl")},
    ]
    write(root / "tasks" / "links" / "sprout" / "summary.json", {"attempts": attempts})
    return trial


def test_a_branch_reads_as_its_reviewer_turn_with_what_the_actor_and_the_reviewer_could_read(tmp_path):
    trial = sprout_run(tmp_path / "run", tmp_path / "tasks", command=COPIES_THE_FIXTURE)

    (branch,) = sprout_branches(str(tmp_path / "run"), str(tmp_path / "tasks"))

    assert (branch.run, branch.task, branch.trial, branch.round, branch.resolved) == (
        "run",
        "links",
        str(trial),
        1,
        True,
    )
    assert branch.turn == reviewer_turn(COPIES_THE_FIXTURE)
    assert branch.visible == "s\n" + STATEMENT
    assert (branch.task_statement, branch.hidden) == (
        STATEMENT,
        HiddenText(PATCH, (TARGET,), "FAILED " + TARGET),
    )


@pytest.mark.parametrize(
    "reward, inserted, resolved_only, count",
    [(0.0, True, True, 0), (0.0, True, False, 1), (1.0, False, True, 0)],
    ids=["unresolved", "unresolved-with-all-branches", "no-reviewer-turn"],
)
def test_only_resolved_branches_with_a_reviewer_turn_are_read_by_default(
    tmp_path, reward, inserted, resolved_only, count
):
    sprout_run(
        tmp_path / "run", tmp_path / "tasks", command=COPIES_THE_FIXTURE, reward=reward, inserted=inserted
    )

    branches = list(
        sprout_branches(str(tmp_path / "run"), str(tmp_path / "tasks"), resolved_only=resolved_only)
    )

    assert len(branches) == count


def test_a_task_with_a_test_patch_hides_the_patch_and_its_fail_to_pass_ids(tmp_path):
    write(tmp_path / "tests" / "test.patch", PATCH)
    write(tmp_path / "tests" / "config.json", {"FAIL_TO_PASS": json.dumps([TARGET])})

    assert task_hidden_tests(tmp_path) == (PATCH, (TARGET,))


def test_a_task_without_a_patch_hides_each_text_file_of_its_tests_as_an_added_file(tmp_path):
    """Harbor's own files get no ``diff --git`` header, so their names are no leak terms."""
    write(tmp_path / "tests" / "test.sh", "pytest /tests/test_outputs.py")
    write(
        tmp_path / "tests" / "test_outputs.py", "def test_secret_output():\n    assert read() == 'SECRET-42'"
    )
    (tmp_path / "tests" / "blob.bin").write_bytes(b"\xff\xfe\x00binary")
    (tmp_path / "tests" / "huge.txt").write_text("x" * (leak_judge.MAX_TEST_FILE_BYTES + 1))

    patch, fail_to_pass = task_hidden_tests(tmp_path)

    assert patch == (
        "+++ b/tests/test.sh\n+pytest /tests/test_outputs.py\n"
        "diff --git a/tests/test_outputs.py b/tests/test_outputs.py\n+++ b/tests/test_outputs.py\n"
        "+def test_secret_output():\n+    assert read() == 'SECRET-42'"
    )
    assert (patch_files(patch), fail_to_pass) == (["tests/test_outputs.py"], ())


def test_a_task_without_tests_hides_nothing(tmp_path):
    assert task_hidden_tests(tmp_path / "missing") == ("", ())


def test_a_renamed_test_file_names_both_of_its_paths():
    patch = "diff --git a/tests/old_test.py b/tests/test_hidden_new.py\nrename from tests/old_test.py\n"

    assert patch_files(patch) == ["tests/old_test.py", "tests/test_hidden_new.py"]


class FakeJev:
    """Jev for the filter: answers each question with ``p``, or refuses every request with ``status``."""

    def __init__(self, *, p: float = 0.0, status: int = 200) -> None:
        self.p, self.status = p, status

    def __call__(self, request: httpx.Request) -> httpx.Response:
        if self.status != 200:
            return httpx.Response(self.status, json={"detail": "refused"})
        questions = json.loads(request.content)["questions"]
        answers = {name: {"type": "noul", "noul": self.p} for name in questions}
        return httpx.Response(
            200, json={"model": "jev-1.13.0", "usage": {"input_tokens": 1}, "answers": answers}
        )


@pytest.mark.parametrize(
    "command, jev, gate, eligible",
    [
        (COPIES_THE_FIXTURE, FakeJev(p=0.1), "both", True),
        (COPIES_THE_FIXTURE, FakeJev(p=0.95), "both", False),
        (COPIES_THE_FIXTURE, FakeJev(status=400), "both", False),
        (NAMES_THE_TEST, FakeJev(p=0.0), "both", False),
        (NAMES_THE_TEST, FakeJev(p=0.0), "jev", True),
    ],
    ids=["both-checks-clean", "jev-flags", "jev-fails", "a-name-leaks", "a-name-leaks-under-gate-jev"],
)
def test_a_branch_may_enter_rl_only_when_the_checks_ran_and_found_nothing(
    tmp_path, monkeypatch, command, jev, gate, eligible
):
    sprout_run(tmp_path / "run", tmp_path / "tasks", command=command)
    real_client = httpx.Client
    monkeypatch.setattr(
        leak_judge.httpx, "Client", lambda **settings: real_client(transport=httpx.MockTransport(jev))
    )
    monkeypatch.setenv("TYPESAFE_API_KEY", "typesafe-key")
    out = tmp_path / "judgements.jsonl"

    main([str(tmp_path / "run"), "--tasks-dir", str(tmp_path / "tasks"), "-o", str(out), "--gate", gate])

    (row,) = [json.loads(line) for line in out.read_text().splitlines()]
    assert row["eligible"] is eligible


def test_the_key_file_wins_over_the_environment(tmp_path, monkeypatch):
    (tmp_path / "key").write_text("from-file\n")
    monkeypatch.setenv("TYPESAFE_API_KEY", "from-env")

    assert read_key(str(tmp_path / "key")) == "from-file"


def test_without_a_key_file_the_key_comes_from_the_environment(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", " from-env ")

    assert read_key(None) == "from-env"


def test_without_a_key_the_filter_exits_naming_the_variable(monkeypatch):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)

    with pytest.raises(SystemExit, match="set TYPESAFE_API_KEY or pass --key-file"):
        read_key(None)

"""The ``swe-rebench-v2`` grade: the agent's diff, applied with the hidden tests to a pristine sandbox.

The sandboxes are the directory sandboxes of ``sandboxes.py``, so git, patch and the test command
run for real. The task image holds a repository at ``/project`` whose ``answer.txt`` says 0. The
hidden test patch adds ``check_answer.sh``, which passes ``test_answer`` when ``answer.txt`` says 1,
and always passes ``test_kept``.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

import pytest

from sprout.runstore.specs import GradeSpec
from sprout.worker.jobs.grade import dataset_rows, grade
from tests.worker.jobs.sandboxes import DirectorySandboxes

TEST_PATCH = (
    "diff --git a/check_answer.sh b/check_answer.sh\n"
    "new file mode 100644\n"
    "--- /dev/null\n"
    "+++ b/check_answer.sh\n"
    "@@ -0,0 +1,2 @@\n"
    '+[ "$(cat answer.txt)" = 1 ] && echo "PASSED test_answer" || echo "FAILED test_answer"\n'
    '+echo "PASSED test_kept"\n'
)
#: The log parser module of the grade, as the upstream ``log_parsers.py`` names its parsers.
PARSER = """import re


def parse(log):
    return {name: status for status, name in re.findall(r"^(PASSED|FAILED) (\\S+)$", log, re.M)}


NAME_TO_PARSER = {"fixture": parse}
"""
#: The lines of a file whose diff is larger than the 128 KiB that one command argument may be.
BULK = "".join("line %06d, only here to make the patch larger than a command line\n" % n for n in range(3200))
#: A hidden test patch that adds a file of ``BULK``.
BULK_PATCH = (
    "diff --git a/fixture.txt b/fixture.txt\nnew file mode 100644\n--- /dev/null\n+++ b/fixture.txt\n"
    "@@ -0,0 +1,3200 @@\n" + "".join("+" + line for line in BULK.splitlines(keepends=True))
)


def git(repository: Path, *argv: str) -> str:
    return subprocess.run(["git", *argv], cwd=repository, check=True, capture_output=True, text=True).stdout


def write_files(directory: Path, files: dict[str, str]) -> None:
    for name, text in files.items():
        (directory / name).write_text(text)


class Task:
    """A task in ``root``: the task image, the agent's final snapshot of it, and the grade's frozen inputs.

    Both images start equal. A test changes ``agent`` (the snapshot's ``/project``) before it grades."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.images = {
            "task-image": root / "images" / "task-image",
            "final-snapshot": root / "images" / "final",
        }
        self.pristine = self.images["task-image"] / "project"
        self.pristine.mkdir(parents=True)
        (self.images["task-image"] / "tmp").mkdir()
        git(self.pristine, "init", "-q")
        (self.pristine / "answer.txt").write_text("0\n")
        self.base = self.commit(self.pristine)
        shutil.copytree(self.images["task-image"], self.images["final-snapshot"], symlinks=True)
        self.agent = self.images["final-snapshot"] / "project"
        self.parser = root / "parser.py"
        self.parser.write_text(PARSER)
        self.revision = "sha256:" + hashlib.sha256(self.parser.read_bytes()).hexdigest()
        self.attempt = root / "attempt"
        self.attempt.mkdir()

    @staticmethod
    def commit(repository: Path) -> str:
        git(repository, "add", "-A")
        git(repository, "-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "c")
        return git(repository, "rev-parse", "HEAD").strip()

    def row(self, **changes: Any) -> dict[str, Any]:
        """The dataset row of the task, with ``changes``."""
        return {
            "instance_id": "owner__project-1",
            "repo": "owner/project",
            "base_commit": self.base,
            "test_patch": TEST_PATCH,
            "install_config": {"test_cmd": "sh check_answer.sh", "log_parser": "fixture"},
            "FAIL_TO_PASS": ["test_answer"],
            "PASS_TO_PASS": ["test_kept"],
            **changes,
        }

    def body(
        self, rows: list[dict[str, Any]], *, dataset: str = "dataset.json", **changes: Any
    ) -> dict[str, Any]:
        """The ``GradeSpec`` fields of the grade on the dataset ``rows``, frozen as a JSON list."""
        path = self.root / dataset
        path.write_text(json.dumps(rows))
        return {
            "benchmark": "swe-rebench-v2",
            "instance_id": "owner__project-1",
            "snapshot_id": "final-snapshot",
            "image": "task-image",
            "dataset_path": str(path),
            "dataset_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "parser_path": str(self.parser),
            "grader_revision": self.revision,
            "timeout_s": 60,
            **changes,
        }

    def facts(self, *, touched: list[str] | None = None) -> dict[str, Any]:
        """The fields of each completed grade of the task that do not depend on the verdict."""
        return {
            "model_patch": str(self.attempt / "model.patch"),
            "test_patch": str(self.attempt / "test.patch"),
            "grader_revision": self.revision,
            "test_patch_paths": ["check_answer.sh"],
            "touched_test_paths": touched or [],
            "fail_to_pass": ["test_answer"],
        }


@pytest.fixture
def task(tmp_path):
    return Task(tmp_path)


@pytest.fixture
def sandboxes(task):
    return DirectorySandboxes(task.root / "sandboxes", images=task.images)


def grade_of(
    task: Task, sandboxes: DirectorySandboxes, *, row: dict[str, Any] | None = None, **spec: Any
) -> dict:
    """The grade of the agent's final snapshot on the task, with the dataset row ``row``."""
    return grade(task.body([row or task.row()], **spec), task.attempt, session_factory=sandboxes)


@pytest.mark.parametrize(
    "answer, verdict",
    [
        ("1\n", {"resolved": True, "progress": 1.0, "expected_pass_share": 1.0, "test_answer": "PASSED"}),
        ("2\n", {"resolved": False, "progress": 0.0, "expected_pass_share": 0.5, "test_answer": "FAILED"}),
    ],
)
def test_the_diff_of_the_final_snapshot_is_graded_in_a_pristine_sandbox_of_the_task_image(
    task, sandboxes, answer, verdict
):
    (task.agent / "answer.txt").write_text(answer)

    result = grade_of(task, sandboxes)

    assert result == {
        "status": "completed",
        "resolved": verdict["resolved"],
        "progress": verdict["progress"],
        "expected_pass_share": verdict["expected_pass_share"],
        "test_statuses": {"test_answer": verdict["test_answer"], "test_kept": "PASSED"},
        "log": str(task.attempt / "swerebench-tests.log"),
        **task.facts(),
    }
    assert sandboxes.lifecycle == [
        ("create", "final-snapshot"),
        ("destroy", "sandbox-1"),
        ("create", "task-image"),
        ("destroy", "sandbox-2"),
    ]


def test_the_attempt_directory_keeps_the_patches_the_test_script_and_the_log(task, sandboxes):
    (task.agent / "answer.txt").write_text("1\n")

    grade_of(task, sandboxes)

    assert sorted(path.name for path in task.attempt.iterdir()) == [
        "eval.sh",
        "model.patch",
        "swerebench-tests.log",
        "test.patch",
    ]
    assert (task.attempt / "test.patch").read_text() == TEST_PATCH
    assert (task.attempt / "swerebench-tests.log").read_text() == "PASSED test_answer\nPASSED test_kept\n"


def test_an_agent_file_at_the_path_of_a_hidden_test_makes_the_test_patch_fail_to_apply(task, sandboxes):
    """As in the upstream harness, nothing resets the files that the test patch touches."""
    (task.agent / "answer.txt").write_text("1\n")
    (task.agent / "check_answer.sh").write_text('echo "PASSED test_answer"\n')

    result = grade_of(task, sandboxes)
    error_output = result.pop("error_output")

    assert result == {
        "status": "completed",
        "resolved": False,
        "progress": 0.0,
        "reason": "test_patch_apply_failed",
        **task.facts(touched=["check_answer.sh"]),
    }
    assert "check_answer.sh" in error_output


def test_a_diff_that_does_not_apply_to_the_task_image_scores_0(task, sandboxes):
    (task.agent / "answer.txt").write_text("1\n")
    (task.pristine / "answer.txt").unlink()
    task.commit(task.pristine)

    result = grade_of(task, sandboxes)
    error_output = result.pop("error_output")

    assert result == {
        "status": "completed",
        "resolved": False,
        "progress": 0.0,
        "reason": "model_patch_apply_failed",
        **task.facts(),
    }
    assert "answer.txt" in error_output


def test_a_diff_that_conflicts_with_the_task_image_scores_0_as_one_that_does_not_apply(task, sandboxes):
    (task.agent / "answer.txt").write_text("1\n")
    (task.pristine / "answer.txt").write_text("rewritten\nafter the base commit\n")
    task.commit(task.pristine)

    result = grade_of(task, sandboxes)

    assert (result["reason"], "test_statuses" in result) == ("model_patch_apply_failed", False)


def test_a_snapshot_whose_repository_gives_no_diff_scores_0_without_a_pristine_sandbox(task, sandboxes):
    """The agent's repository decides whether git can diff it, so a missing ``.git`` is the agent's 0."""
    shutil.rmtree(task.agent / ".git")

    result = grade_of(task, sandboxes)
    error_output = result.pop("error_output")

    assert result == {
        "status": "completed",
        "resolved": False,
        "progress": 0.0,
        "reason": "model_patch_unavailable",
        **task.facts(),
    }
    assert "not a git repository" in error_output
    assert (task.attempt / "model.patch").read_bytes() == b""
    assert sandboxes.lifecycle == [("create", "final-snapshot"), ("destroy", "sandbox-1")]


@pytest.mark.parametrize(
    "option, value",
    [("diff.noprefix", "true"), ("color.diff", "always"), ("diff.external", "false")],
)
def test_the_git_config_of_the_agent_does_not_change_the_graded_diff(task, sandboxes, option, value):
    (task.agent / "answer.txt").write_text("1\n")
    git(task.agent, "config", option, value)

    result = grade_of(task, sandboxes)

    assert (result["resolved"], result.get("reason")) == (True, None)


def test_a_repository_that_another_uid_owns_is_still_graded(task, sandboxes, monkeypatch):
    """git refuses such a repository ("dubious ownership"), and the refusal would score the agent 0."""
    (task.agent / "answer.txt").write_text("1\n")
    monkeypatch.setenv("GIT_TEST_ASSUME_DIFFERENT_OWNER", "1")
    refused = subprocess.run(["git", "status"], cwd=task.agent, capture_output=True)
    assert refused.returncode == 128, "git must refuse the repository of another owner"

    result = grade_of(task, sandboxes)

    assert (result["resolved"], result.get("reason")) == (True, None)


def test_tests_that_outrun_the_timeout_score_0_with_the_log_so_far(task, sandboxes):
    """The agent's code decides whether its tests finish, so a hang is the agent's 0, not a failed grade."""
    (task.agent / "answer.txt").write_text("1\n")
    row = task.row(install_config={"test_cmd": "echo started; sleep 30", "log_parser": "fixture"})

    result = grade_of(task, sandboxes, row=row, timeout_s=1)

    assert result == {
        "status": "completed",
        "resolved": False,
        "progress": 0.0,
        "reason": "tests_timed_out",
        "error_output": "the tests did not finish within 1 s",
        "log": str(task.attempt / "swerebench-tests.log"),
        **task.facts(),
    }
    assert (task.attempt / "swerebench-tests.log").read_text() == "started\n"


def test_tests_that_outrun_the_timeout_score_0_also_when_their_log_does_not_move(task, sandboxes):
    """The verdict does not need the log of a timed-out run, so its loss loses only the log."""
    (task.agent / "answer.txt").write_text("1\n")
    sandboxes.refuse("download", "tests.log")
    row = task.row(install_config={"test_cmd": "echo started; sleep 30", "log_parser": "fixture"})

    result = grade_of(task, sandboxes, row=row, timeout_s=1)

    assert result == {
        "status": "completed",
        "resolved": False,
        "progress": 0.0,
        "reason": "tests_timed_out",
        "error_output": "the tests did not finish within 1 s",
        **task.facts(),
    }


def test_a_test_runner_that_is_killed_scores_0_with_its_exit_code(task, sandboxes):
    """An OOM kill looks the same: the test script ends on SIGKILL, exit code 137."""
    (task.agent / "answer.txt").write_text("1\n")
    row = task.row(install_config={"test_cmd": "kill -KILL $$", "log_parser": "fixture"})

    result = grade_of(task, sandboxes, row=row)

    assert result == {
        "status": "completed",
        "resolved": False,
        "progress": 0.0,
        "reason": "test_runner_killed",
        "error_output": "exit code 137",
        "log": str(task.attempt / "swerebench-tests.log"),
        **task.facts(),
    }


@pytest.mark.parametrize(
    "agent_files, test_patch",
    [
        pytest.param({"answer.txt": "1\n", "bulk.txt": BULK}, TEST_PATCH, id="model-patch"),
        pytest.param({"answer.txt": "1\n"}, TEST_PATCH + BULK_PATCH, id="test-patch"),
    ],
)
def test_a_patch_larger_than_one_command_argument_moves_as_a_file(task, sandboxes, agent_files, test_patch):
    """Linux caps one ``sh -c`` argument at 128 KiB. Patches sent inside a command failed every such grade."""
    write_files(task.agent, agent_files)

    result = grade_of(task, sandboxes, row=task.row(test_patch=test_patch))

    assert (result["resolved"], result.get("reason")) == (True, None)


def test_a_diff_that_is_not_utf8_reaches_the_pristine_sandbox_byte_for_byte(task, sandboxes):
    (task.agent / "answer.txt").write_text("1\n")
    (task.agent / "latin.txt").write_bytes("café\n".encode("latin-1"))
    check = (
        "printf 'caf\\351\\n' | cmp -s - latin.txt && echo 'PASSED test_latin' || echo 'FAILED test_latin'"
    )
    row = task.row(
        install_config={"test_cmd": check, "log_parser": "fixture"},
        FAIL_TO_PASS=["test_latin"],
        PASS_TO_PASS=[],
    )

    result = grade_of(task, sandboxes, row=row)

    assert (result["resolved"], result["test_statuses"]) == (True, {"test_latin": "PASSED"})


@pytest.mark.parametrize(
    "direction, name, lifecycle",
    [
        ("download", "model.patch", [("create", "final-snapshot"), ("destroy", "sandbox-1")]),
        (
            "upload",
            "model.patch",
            [
                ("create", "final-snapshot"),
                ("destroy", "sandbox-1"),
                ("create", "task-image"),
                ("destroy", "sandbox-2"),
            ],
        ),  # noqa: E501
        (
            "upload",
            "test.patch",
            [
                ("create", "final-snapshot"),
                ("destroy", "sandbox-1"),
                ("create", "task-image"),
                ("destroy", "sandbox-2"),
            ],
        ),  # noqa: E501
        (
            "upload",
            "eval.sh",
            [
                ("create", "final-snapshot"),
                ("destroy", "sandbox-1"),
                ("create", "task-image"),
                ("destroy", "sandbox-2"),
            ],
        ),  # noqa: E501
        (
            "download",
            "tests.log",
            [
                ("create", "final-snapshot"),
                ("destroy", "sandbox-1"),
                ("create", "task-image"),
                ("destroy", "sandbox-2"),
            ],
        ),  # noqa: E501
    ],
)
def test_a_file_that_does_not_move_fails_the_grade_with_its_name_and_the_reason(
    task, sandboxes, direction, name, lifecycle
):
    """A grade that the grader could not make is not a 0: the job fails. Each sandbox is destroyed."""
    (task.agent / "answer.txt").write_text("1\n")
    sandboxes.refuse(direction, name)

    with pytest.raises(RuntimeError, match=re.escape(name) + ".*: PodmanError: refused by the test$"):
        grade_of(task, sandboxes)

    assert sandboxes.lifecycle == lifecycle


@pytest.mark.parametrize(
    "image, message",
    [
        (
            "snapshot_id",
            "Could not restore the SWE-rebench final snapshot: PodmanError: image absent not known",
        ),
        (
            "image",
            "Could not start a pristine SWE-rebench sandbox from the task image: "
            "PodmanError: image absent not known",
        ),
    ],
)
def test_a_sandbox_that_does_not_start_fails_the_grade_with_the_reason(task, sandboxes, image, message):
    (task.agent / "answer.txt").write_text("1\n")

    with pytest.raises(RuntimeError, match="^%s$" % re.escape(message)):
        grade_of(task, sandboxes, **{image: "absent"})


def test_each_test_command_runs_after_one_fails(task, sandboxes):
    (task.agent / "answer.txt").write_text("1\n")
    commands = ["echo 'FAILED test_first'; false", "sh check_answer.sh"]
    row = task.row(
        install_config={"test_cmd": commands, "log_parser": "fixture"}, FAIL_TO_PASS=["test_answer"]
    )

    result = grade_of(task, sandboxes, row=row)

    assert result["test_statuses"] == {"test_first": "FAILED", "test_answer": "PASSED", "test_kept": "PASSED"}


def test_an_install_config_that_the_dataset_stores_as_json_text_is_read(task, sandboxes):
    (task.agent / "answer.txt").write_text("1\n")
    row = task.row(install_config=json.dumps({"test_cmd": "sh check_answer.sh", "log_parser": "fixture"}))

    result = grade_of(task, sandboxes, row=row)

    assert (result["resolved"], result.get("reason")) == (True, None)


@pytest.mark.parametrize("verifier_network, egress", [("deny", [False, False]), ("allow", [True, True])])
def test_the_grading_sandboxes_start_offline_unless_the_verifier_may_reach_the_network(
    task, sandboxes, verifier_network, egress
):
    (task.agent / "answer.txt").write_text("1\n")

    grade_of(task, sandboxes, verifier_network=verifier_network, backend={"backend": "podman", "podman": {}})

    assert [session.allow_internet for session in sandboxes.sessions] == egress


@pytest.mark.parametrize("instances", [["owner__project-2"], ["owner__project-1", "owner__project-1"]])
def test_a_dataset_that_does_not_hold_the_task_exactly_once_fails_the_grade(task, sandboxes, instances):
    rows = [task.row(instance_id=instance) for instance in instances]

    with pytest.raises(ValueError, match="^Frozen dataset does not identify exactly one grading task$"):
        grade(task.body(rows), task.attempt, session_factory=sandboxes)

    assert sandboxes.lifecycle == []


@pytest.mark.parametrize(
    "frozen, message",
    [
        pytest.param("dataset_sha256", "^Frozen dataset checksum changed$", id="dataset"),
        pytest.param(
            "grader_revision", "^SWE-rebench parser revision differs from grader_revision$", id="parser"
        ),
    ],
)
def test_a_dataset_or_parser_whose_checksum_differs_fails_the_grade_before_a_sandbox_starts(
    task, sandboxes, frozen, message
):
    """The worker claims the grade by these checksums, so the file changed after the worker started."""
    body = task.body([task.row()], **{frozen: "sha256:" + "0" * 64})

    with pytest.raises(ValueError, match=message):
        grade(body, task.attempt, session_factory=sandboxes)

    assert sandboxes.lifecycle == []


@pytest.mark.parametrize(
    "name, text",
    [
        ("dataset.json", '[{"instance_id": "a"}, {"instance_id": "b"}]'),
        ("dataset.jsonl", '{"instance_id": "a"}\n\n{"instance_id": "b"}\n'),
    ],
)
def test_a_frozen_dataset_is_a_json_list_or_a_jsonl_file(tmp_path, name, text):
    (tmp_path / name).write_text(text)
    spec = GradeSpec(
        benchmark="swe-rebench-v2",
        dataset_path=str(tmp_path / name),
        dataset_sha256=hashlib.sha256(text.encode()).hexdigest(),
    )

    assert dataset_rows(spec) == [{"instance_id": "a"}, {"instance_id": "b"}]


def test_a_frozen_json_dataset_that_is_not_a_list_is_refused(tmp_path):
    (tmp_path / "dataset.json").write_text('{"instance_id": "a"}')
    spec = GradeSpec(
        benchmark="swe-rebench-v2",
        dataset_path=str(tmp_path / "dataset.json"),
        dataset_sha256=hashlib.sha256(b'{"instance_id": "a"}').hexdigest(),
    )

    with pytest.raises(ValueError, match="^Frozen dataset must contain a list of tasks$"):
        dataset_rows(spec)

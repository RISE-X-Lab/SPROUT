import hashlib
import os
from pathlib import Path
import re
import shlex
import signal
import stat
import subprocess
import sys
from types import SimpleNamespace

import pytest

from sprout.runstore.messages import export_messages, mark_hint
from sprout.runstore.grading import expected_pass_share, grade_swerebench, progress, resolved


def test_progress_is_the_share_of_target_tests_that_passed():
    task = {"FAIL_TO_PASS": ["t::a", "t::b"], "PASS_TO_PASS": ["t::c [12 ms]", "t::d"]}
    statuses = {"t::a": "PASSED", "t::b": "FAILED", "t::c": "PASSED", "t::d": "PASSED"}
    assert progress(statuses, task) == 0.5 and resolved(statuses, task) is False
    assert expected_pass_share(statuses, task) == 0.75, "the all-tests share is reported beside it"
    statuses["t::b"] = "PASSED"
    assert progress(statuses, task) == 1.0 and resolved(statuses, task) is True
    assert progress({}, task) == 0.0
    assert progress(statuses, {"FAIL_TO_PASS": [], "PASS_TO_PASS": []}) == 0.0, "no expected tests, no progress"


def test_progress_ignores_tests_that_already_passed_and_zeroes_a_regression():
    # cloudpickle-153's shape: 2 targets among 45 expected tests.
    task = {"FAIL_TO_PASS": ["t::f1", "t::f2"], "PASS_TO_PASS": [f"t::p{i}" for i in range(43)]}
    empty = {f"t::p{i}": "PASSED" for i in range(43)}
    assert expected_pass_share(empty, task) == 43 / 45
    assert progress(empty, task) == 0.0, "an empty patch makes no progress"
    one = {**empty, "t::f1": "PASSED"}
    assert progress(one, task) == 0.5
    assert progress({**one, "t::p7": "FAILED"}, task) == 0.0, "a broken PASS_TO_PASS test is no progress"
    assert progress({**one, "t::f2": "PASSED", "t::p7": "FAILED"}, task) == 0.0, "nor are fixed targets beside it"
    only_p2p = {"FAIL_TO_PASS": [], "PASS_TO_PASS": ["t::p0"]}
    assert progress({"t::p0": "PASSED"}, only_p2p) == 1.0 and progress({}, only_p2p) == 0.0


TEST_PATCH = (
    "diff --git a/test_answer.py b/test_answer.py\nnew file mode 100644\n"
    "--- /dev/null\n+++ b/test_answer.py\n@@ -0,0 +1,3 @@\n"
    "+from answer import VALUE\n+\n+def test_answer(): assert VALUE == 1\n"
)


class Grading:
    """The grader on a real repository; only the sandbox boundary is substituted.

    Every sandbox is this one checkout (a pristine one is first reset to the
    base commit) with a guest directory of its own under ``tmp_path``. Files
    cross only through ``upload_file`` / ``download_file``, which check that
    each goes between the attempt directory and the current sandbox's private
    directory. ``shell``, ``checked`` and ``shell_outcome`` run real bash, so a
    command too long for one argument fails as the runtime's would (E2BIG),
    and a command past its timeout has its process group killed, as there.
    """

    def __init__(self, tmp_path, monkeypatch):
        self.tmp_path = tmp_path
        self.repo, self.attempt = tmp_path / "repo", tmp_path / "attempt"
        self.repo.mkdir()
        self.attempt.mkdir()
        self.git("init", "-q")
        (self.repo / "answer.py").write_text("VALUE = 0\n")
        self.git("add", ".")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD")
        self.parser = tmp_path / "parser.py"
        self.parser.write_text(
            "import re\n"
            "def parse(text):\n"
            " return {name:status for name,status in re.findall(r'(test_answer.py::test_answer) (PASSED|FAILED)', text)}\n"
            "NAME_TO_PARSER={'fixture':parse}\n"
        )
        self.lifecycle, self.commands, self.uploads, self.downloads, self.guests = [], [], [], [], []
        #: ``(direction, file name)`` of a transfer the backend refuses.
        self.refuse = None
        monkeypatch.setattr("sprout.harness_sandbox.session.shell", self.shell)
        monkeypatch.setattr("sprout.harness_sandbox.session.checked", self.checked)
        monkeypatch.setattr("sprout.harness_sandbox.session.shell_outcome", self.shell_outcome)
        monkeypatch.setattr("sprout.runstore.grading.guest_directory", self.guest_directory)

    def git(self, *argv):
        return subprocess.check_output(["git", *argv], cwd=self.repo, text=True).strip()

    def grade(self, *, test_cmd=None, timeout_s=60, **row):
        spec = SimpleNamespace(
            parser_path=str(self.parser),
            grader_revision="sha256:" + hashlib.sha256(self.parser.read_bytes()).hexdigest(),
            snapshot_id="actor-final-snapshot", image="task-image",
            resources={"cpu": 2, "memory_mb": 4096}, timeout_s=timeout_s,
        )
        row = {
            "repo": "owner/repo", "base_commit": self.base, "test_patch": TEST_PATCH,
            # the grader pins PATH to the image's, so name this interpreter, which has pytest
            "install_config": {"test_cmd": test_cmd or shlex.quote(sys.executable) + " -m pytest -vv -rA test_answer.py",
                               "log_parser": "fixture"},
            "FAIL_TO_PASS": ["test_answer.py::test_answer"], "PASS_TO_PASS": [], **row,
        }
        return grade_swerebench(spec, row, {}, self.attempt, session_factory=lambda **_: Session(self))

    # --- the sandbox boundary -------------------------------------------------
    def guest_directory(self):
        self.guests.append(self.tmp_path / f"guest-{len(self.guests)}")
        return str(self.guests[-1])

    def guest_file(self, path):
        """``path`` is in the current sandbox's own directory, which only its owner can enter."""
        path = Path(path)
        assert self.guests and path.parent == self.guests[-1], path
        assert stat.S_IMODE(path.parent.stat().st_mode) == 0o700, "made with mkdir -m 700"
        return path

    def shell_outcome(self, session, text, timeout=120):
        self.commands.append(text)
        process = subprocess.Popen(["bash", "-c", text.replace("cd /repo", "cd " + str(self.repo))],
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, start_new_session=True)
        try:
            stdout, stderr = process.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate()
            return stdout.decode(errors="replace"), stderr.decode(errors="replace"), 128 + signal.SIGKILL, True
        code = process.returncode
        return stdout.decode(errors="replace"), stderr.decode(errors="replace"), code if code >= 0 else 128 - code, False

    def shell(self, session, text, timeout=120):
        stdout, stderr, code, timed_out = self.shell_outcome(session, text, timeout)
        assert not timed_out, text
        return stdout, stderr, code

    def checked(self, session, text, timeout=120):
        stdout, stderr, code = self.shell(session, text, timeout)
        assert code == 0, stderr
        return stdout


class Session:
    def __init__(self, grading):
        self.grading = grading
        self.transfer_error = ""

    def create(self, image, resources):
        self.grading.lifecycle.append(("create", image))
        if image == "task-image":       # a pristine VM: the repository as the image left it
            subprocess.run(["git", "reset", "-q", "--hard", self.grading.base], cwd=self.grading.repo, check=True)
            subprocess.run(["git", "clean", "-qfd"], cwd=self.grading.repo, check=True)
        return True

    def destroy(self):
        self.grading.lifecycle.append(("destroy",))

    def upload_file(self, source, destination):
        assert Path(source).parent == self.grading.attempt, source
        data = Path(source).read_bytes()
        self.grading.uploads.append((Path(destination).name, data))
        return self._move(("upload", Path(destination).name), data, self.grading.guest_file(destination))

    def download_file(self, source, destination, timeout=120.0):
        assert Path(destination).parent == self.grading.attempt, destination
        data = self.grading.guest_file(source).read_bytes()
        self.grading.downloads.append((Path(source).name, data))
        return self._move(("download", Path(source).name), data, Path(destination))

    def _move(self, transfer, data, target):
        if self.grading.refuse == transfer:
            self.transfer_error = "PodmanError: refused by the test"
            return False
        target.write_bytes(data)
        return True


@pytest.mark.parametrize("fix,agent_test,expected", [
    (False, "test_agent.py", "FAILED"),
    (True, "test_agent.py", "PASSED"),
    (True, "test_answer.py", "test_patch_apply_failed"),
])
def test_grader_applies_the_diff_to_a_fresh_image_like_the_official_harness(
        tmp_path, monkeypatch, fix, agent_test, expected):
    """Real git/test execution; only the VM filesystem boundary is substituted.

    Two VMs, as upstream: the final snapshot only yields the diff; a pristine
    checkout gets the diff, the hidden test patch and the test command. As in
    upstream's eval.py, nothing resets the files the test patch touches, so an
    agent test at the hidden test's path makes the test patch conflict.
    """
    grading = Grading(tmp_path, monkeypatch)
    repo = grading.repo
    # The agent's state: its own passing test, an untracked file, and (maybe)
    # the fix. Only the diff reaches the pristine VM.
    (repo / agent_test).write_text("def test_answer(): assert True\n")
    (repo / "installed.marker").write_text("side effect\n")
    if fix:
        (repo / "answer.py").write_text("VALUE = 1\n")
    result = grading.grade()
    if expected == "test_patch_apply_failed":
        assert result["resolved"] is False and result["reason"] == expected, result
        assert result["progress"] == 0.0
    else:   # the verdict comes from the hidden test actually running
        assert result["test_statuses"] == {"test_answer.py::test_answer": expected}, result
        assert result["resolved"] is (expected == "PASSED"), result
        assert result["progress"] == (1.0 if expected == "PASSED" else 0.0)
    # What every completed grade says about the hidden tests, an apply failure's included.
    assert result["model_patch"] == str(grading.attempt / "model.patch")
    assert result["test_patch_paths"] == ["test_answer.py"]
    assert result["fail_to_pass"] == ["test_answer.py::test_answer"]
    # The hidden tests as applied, for the review's leak judge: a path, like model_patch.
    assert result["test_patch"] == str(grading.attempt / "test.patch")
    assert "test_answer.py" in (grading.attempt / "test.patch").read_text()
    assert result["touched_test_paths"] == (["test_answer.py"] if agent_test == "test_answer.py" else [])
    assert grading.lifecycle == [("create", "actor-final-snapshot"), ("destroy",), ("create", "task-image"), ("destroy",)]
    diff = (grading.attempt / "model.patch").read_text()
    assert "installed.marker" in diff and ("VALUE = 1" in diff) is fix
    assert (repo / "answer.py").read_text() == ("VALUE = 1\n" if fix else "VALUE = 0\n")


@pytest.mark.parametrize("large", ["model patch", "test patch"])
def test_grader_moves_patches_larger_than_one_command_line(tmp_path, monkeypatch, large):
    """One ``sh -c`` argument holds at most 128 KiB (MAX_ARG_STRLEN), and
    patches once travelled base64-encoded inside one: every grade of a task
    whose diff or test patch passed ~96 KiB failed with E2BIG."""
    grading = Grading(tmp_path, monkeypatch)
    (grading.repo / "answer.py").write_text("VALUE = 1\n")
    bulk = [f"line {index:06d}, only here to make the patch larger than a command line\n" for index in range(3200)]
    test_patch = TEST_PATCH
    if large == "model patch":
        (grading.repo / "data.txt").write_text("".join(bulk))
    else:
        test_patch += ("diff --git a/fixture.txt b/fixture.txt\nnew file mode 100644\n--- /dev/null\n"
                       f"+++ b/fixture.txt\n@@ -0,0 +1,{len(bulk)} @@\n" + "".join("+" + line for line in bulk))
    result = grading.grade(test_patch=test_patch)
    assert result["resolved"] is True, result
    moved = dict(grading.uploads)["model.patch" if large == "model patch" else "test.patch"]
    assert len(moved) > 200 * 1024
    assert max(map(len, grading.commands)) < 4096, "no command carries a file"
    assert sorted(name for name, _ in grading.uploads) == ["eval.sh", "model.patch", "test.patch"]
    assert [name for name, _ in grading.downloads] == ["model.patch", "tests.log"]


def test_grader_moves_a_diff_that_is_not_utf8_byte_for_byte(tmp_path, monkeypatch):
    """A Latin-1 file makes a diff that is no UTF-8 text; it once failed to
    encode on its way into the pristine VM."""
    grading = Grading(tmp_path, monkeypatch)
    (grading.repo / "answer.py").write_text("VALUE = 1\n")
    (grading.repo / "latin.txt").write_bytes("café\n".encode("latin-1"))
    result = grading.grade()
    assert result["resolved"] is True, result
    downloaded, uploaded = dict(grading.downloads)["model.patch"], dict(grading.uploads)["model.patch"]
    assert b"+caf\xe9\n" in downloaded
    assert uploaded == downloaded == (grading.attempt / "model.patch").read_bytes()
    assert (grading.repo / "latin.txt").read_bytes() == b"caf\xe9\n", "the pristine checkout got the same bytes"


def test_a_diff_the_snapshot_cannot_produce_is_a_completed_zero(tmp_path, monkeypatch):
    """The agent's repository decides whether git can diff it (a removed
    ``.git``, a lost base commit): a 0 the agent caused, not a grading fault."""
    grading = Grading(tmp_path, monkeypatch)
    (grading.repo / "answer.py").write_text("VALUE = 1\n")
    result = grading.grade(base_commit="0" * 40)
    assert result["status"] == "completed" and result["resolved"] is False and result["progress"] == 0.0, result
    assert result["reason"] == "model_patch_unavailable" and "bad object 0000000000" in result["error_output"]
    assert result["model_patch"] == str(grading.attempt / "model.patch")
    assert (grading.attempt / "model.patch").read_bytes() == b"", "nothing was graded"
    assert result["test_patch"] == str(grading.attempt / "test.patch"), "a zero names the hidden tests too"
    assert result["test_patch_paths"] == ["test_answer.py"] and result["touched_test_paths"] == []
    assert result["fail_to_pass"] == ["test_answer.py::test_answer"]
    assert grading.lifecycle == [("create", "actor-final-snapshot"), ("destroy",)], "no pristine VM for no diff"
    assert grading.downloads == []


@pytest.mark.parametrize("direction,name", [
    ("download", "model.patch"), ("upload", "model.patch"), ("upload", "test.patch"),
    ("upload", "eval.sh"), ("download", "tests.log"),
])
def test_a_file_the_backend_does_not_move_fails_the_grade_naming_it(tmp_path, monkeypatch, direction, name):
    """A grade the grader could not make is no 0: the job fails, and its error
    says which file did not move and why."""
    grading = Grading(tmp_path, monkeypatch)
    (grading.repo / "answer.py").write_text("VALUE = 1\n")
    grading.refuse = (direction, name)
    with pytest.raises(RuntimeError, match=re.escape(name) + ".*: PodmanError: refused by the test$"):
        grading.grade()
    assert len(grading.lifecycle) % 2 == 0 and set(grading.lifecycle[1::2]) == {("destroy",)}, \
        "every sandbox is destroyed"


@pytest.mark.parametrize("test_cmd,timeout_s,refuse,reason,error", [
    ("sleep 30", 2, None, "tests_timed_out", "the tests did not finish within 2 s"),
    ("sleep 30", 2, ("download", "tests.log"), "tests_timed_out", "the tests did not finish within 2 s"),
    ("kill -KILL $$", 60, None, "test_runner_killed", "exit code 137"),   # how an OOM kill looks
])
def test_tests_that_hang_or_are_killed_are_a_completed_zero(tmp_path, monkeypatch, test_cmd, timeout_s, refuse,
                                                             reason, error):
    """The agent's code decides whether its tests finish: hanging them must
    not turn a 0 into a missing grade. A timed-out run's log is kept when it
    can be, and is not needed."""
    grading = Grading(tmp_path, monkeypatch)
    (grading.repo / "answer.py").write_text("VALUE = 1\n")
    grading.refuse = refuse
    result = grading.grade(test_cmd=test_cmd, timeout_s=timeout_s)
    assert result["status"] == "completed" and result["resolved"] is False and result["progress"] == 0.0, result
    assert result["reason"] == reason and result["error_output"].split("\n")[0] == error, result
    assert result.get("log") == (None if refuse else str(grading.attempt / "swerebench-tests.log"))
    assert "test_statuses" not in result and result["test_patch_paths"] == ["test_answer.py"]
    assert result["model_patch"] == str(grading.attempt / "model.patch")


def test_export_native_prefix_and_marked_branch_hint(tmp_path):
    import json

    home = tmp_path / "native-home"
    home.mkdir()
    history = [{"type": "mini.session", "format": "sprout-mini-v1", "mini_version": "2.4.6"}] + [
        {"type": "mini.message", "message": {"role": role, "content": text}}
        for role, text in [
            ("user", "original task"), ("assistant", "parent attempt"),
            ("user", mark_hint("private guidance")), ("assistant", "branch patch"),
        ]
    ]
    (home / "session-id.jsonl").write_text("".join(json.dumps(row) + "\n" for row in history))
    messages = export_messages(tmp_path, "session-id", "mini-swe-agent")
    assert [m["content"] for m in messages] == ["original task", "parent attempt", "branch patch"]

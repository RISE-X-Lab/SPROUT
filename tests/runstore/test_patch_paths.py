"""patch_paths: every file a patch names, read from what git itself writes.

The fixtures are made at test time: the model's side with the grader's own
diff command (``grading.DIFF``, ``core.quotepath=off``), and git's default
quoting for the test-patch side, since a dataset's patch was made by someone
else's git. Each test first asserts the fixture has the shape it is about, so
a git that wrote something else fails here instead of passing vacuously.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from sprout.worker.jobs.grade import APPLY, DIFF, GIT, GRADE_FACT_KEYS, patch_facts, patch_paths


class Repo:
    """A repository with one commit, whose work tree a test then changes."""

    def __init__(self, path: Path, files: dict) -> None:
        self.path = path
        path.mkdir()
        self.git("init", "-q")
        for name, data in files.items():
            self.write(name, data)
        self.git("add", "-A")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *argv: str) -> bytes:
        return subprocess.run(["git", *argv], cwd=self.path, check=True, capture_output=True).stdout

    def write(self, name: str, data: bytes) -> None:
        path = self.path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)

    def diff(self) -> str:
        """The work tree against the base commit, exactly as the grader takes it."""
        out = self.path.parent / "model.patch"
        subprocess.run(["bash", "-c", DIFF % (self.base, out)], cwd=self.path, check=True)
        return out.read_bytes().decode("utf-8", errors="replace")

    def default_diff(self) -> str:
        """The same change through git's defaults: non-ASCII names quoted in octal."""
        self.git("add", "-A")
        return self.git("diff", "--cached", "--binary", self.base).decode("utf-8", errors="replace")


@pytest.fixture
def repo(tmp_path):
    return lambda files: Repo(tmp_path / "repo", files)


def test_the_grader_trusts_a_repository_another_uid_owns(repo, tmp_path):
    """git refuses a repository owned by another uid ("dubious ownership"),
    and the grader would score that refusal as the agent's model_patch_unavailable.
    GIT_TEST_ASSUME_DIFFERENT_OWNER is git's own switch for faking the owner."""
    r = repo({"module.py": b"value = 1\n"})
    r.write("module.py", b"value = 2\n")
    # The switch only bites when no safe.directory whitelists the repository, and a CI
    # runner's system or global git config does (`safe.directory = *`), which once made
    # the plain call below exit 1 and this test fail there: so both configs are ignored.
    other_owner = {**__import__("os").environ, "GIT_TEST_ASSUME_DIFFERENT_OWNER": "1",
                   "GIT_CONFIG_SYSTEM": "/dev/null", "GIT_CONFIG_GLOBAL": "/dev/null"}
    plain = subprocess.run(["git", "diff", "--quiet"], cwd=r.path, env=other_owner, capture_output=True)
    assert plain.returncode not in (0, 1), "the switch must make plain git refuse the repository"
    out = tmp_path / "model.patch"
    subprocess.run(["bash", "-c", DIFF % (r.base, out)], cwd=r.path, env=other_owner, check=True)
    assert "+value = 2" in out.read_text()
    r.git("reset", "-q", "--hard", r.base)
    subprocess.run(["bash", "-c", APPLY % (out, out)], cwd=r.path, env=other_owner, check=True)
    assert (r.path / "module.py").read_bytes() == b"value = 2\n"
    assert GIT.count("safe.directory") == 1 and DIFF.count(GIT) == 2 and APPLY.startswith(GIT)


def test_a_rename_counts_its_old_and_new_name(repo):
    """A pure rename has no ---/+++ lines, and a header naming two paths is
    ambiguous once a name holds a space: the rename lines say both."""
    r = repo({"old.py": b"value = 1\n", "old name.py": b"x\n", "edited.py": b"".join(b"line %d\n" % i for i in range(20))})
    r.git("mv", "old.py", "new.py")
    r.git("mv", "old name.py", "new name.py")
    r.git("mv", "edited.py", "renamed.py")
    r.write("renamed.py", b"".join(b"line %d\n" % i for i in range(19)) + b"changed\n")
    diff = r.diff()
    assert "rename from old name.py\nrename to new name.py\n" in diff and "+++ b/renamed.py\n" in diff, diff
    assert patch_paths(diff) == ["edited.py", "new name.py", "new.py", "old name.py", "old.py", "renamed.py"]


def test_a_binary_section_is_read_from_its_header_only(repo):
    """``--binary`` writes base85 after ``GIT binary patch``; without it git
    writes one ``Binary files ... differ`` line. Either way only the header
    names the file."""
    r = repo({"blob.bin": b"\x00\x01" + bytes(range(256)) * 8})
    r.write("blob.bin", b"\x00\x02" + bytes(reversed(range(256))) * 8)
    r.write("added.bin", b"\x00\x03" * 64)
    diff = r.diff()
    assert diff.count("GIT binary patch") == 2 and "--- " not in diff, diff
    summary = r.git("diff", "--cached", r.base).decode()
    assert "Binary files /dev/null and b/added.bin differ" in summary, summary
    assert patch_paths(diff) == patch_paths(summary) == ["added.bin", "blob.bin"]


def test_a_name_with_a_space_ends_at_the_tab_git_writes_after_it(repo):
    r = repo({"docs/my notes.txt": b"x\n"})
    r.write("docs/my notes.txt", b"x\ny\n")
    diff = r.diff()
    assert "--- a/docs/my notes.txt\t\n+++ b/docs/my notes.txt\t\n" in diff, diff
    assert patch_paths(diff) == ["docs/my notes.txt"]


@pytest.mark.parametrize("made_by", ["grader", "git defaults"])
def test_quoted_and_raw_utf8_names_read_alike(repo, made_by):
    """git quotes a name holding a quote or a control character, and by
    default every non-ASCII byte as ``\\ooo`` -- bytes, which decode as UTF-8
    only together. The grader's diff keeps UTF-8 raw (``quotepath=off``)."""
    names = ["café/naïve.py", 'we"ird.py', "tab\there.py"]
    r = repo({name: b"x\n" for name in names})
    for name in names:
        r.write(name, b"x\ny\n")
    diff = r.diff() if made_by == "grader" else r.default_diff()
    assert '"a/we\\"ird.py"' in diff and '"a/tab\\there.py"' in diff, diff
    if made_by == "grader":
        assert "diff --git a/café/naïve.py b/café/naïve.py\n" in diff, diff
    else:
        assert '"a/caf\\303\\251/na\\303\\257ve.py"' in diff, diff
    assert patch_paths(diff) == sorted(names)


def test_a_deletion_a_mode_change_and_an_empty_file(repo):
    """/dev/null is no path; a mode change and an empty new file have no
    ---/+++ lines at all, only their header."""
    r = repo({"gone.py": b"x\n", "run.sh": b"echo\n"})
    (r.path / "gone.py").unlink()
    (r.path / "run.sh").chmod(0o755)
    r.write("empty.txt", b"")
    diff = r.diff()
    assert "+++ /dev/null\n" in diff and "old mode 100644\nnew mode 100755\n" in diff, diff
    assert "--- a/run.sh" not in diff and "--- a/empty.txt" not in diff and "--- /dev/null" not in diff, diff
    assert patch_paths(diff) == ["empty.txt", "gone.py", "run.sh"]


def test_a_hunk_line_that_reads_like_a_header_names_nothing(repo):
    """Removing the SQL comment ``-- note`` writes ``--- note`` inside the
    hunk, and adding ``++ added`` writes ``+++ added``."""
    r = repo({"schema.sql": b"select 1;\n-- note\nselect 2;\n"})
    r.write("schema.sql", b"select 1;\nselect 2;\n++ added\n")
    diff = r.diff()
    assert "\n--- note\n" in diff and "\n+++ added\n" in diff, diff
    assert patch_paths(diff) == ["schema.sql"]


@pytest.mark.skipif(shutil.which("diff") is None, reason="needs GNU diff")
def test_a_plain_unified_diff_is_read_by_its_file_triples(tmp_path):
    """No ``diff --git`` line: each file is a ---, +++, @@ triple, its names
    cut at the TAB before diff's timestamp; a trap inside a hunk is no triple."""
    for side, files in (("a", {"src/app.py": "one\n-- removed\ntwo\n"}),
                        ("b", {"src/app.py": "one\n++ added\ntwo\n", "tests/test_new.py": "def test(): pass\n"})):
        for name, text in files.items():
            (tmp_path / side / name).parent.mkdir(parents=True, exist_ok=True)
            (tmp_path / side / name).write_text(text)
    diff = subprocess.run(["diff", "-ruN", "a", "b"], cwd=tmp_path, capture_output=True, text=True).stdout
    assert "diff --git" not in diff and "\n--- removed\n+++ added\n two\n" in diff, diff
    assert "--- a/src/app.py\t" in diff and "--- a/tests/test_new.py\t" in diff, diff
    assert patch_paths(diff) == ["src/app.py", "tests/test_new.py"]


def test_patch_facts_say_what_the_hidden_tests_are_and_what_the_diff_touched():
    test_patch = (
        "diff --git a/tests/test_hidden.py b/tests/test_hidden.py\n--- a/tests/test_hidden.py\n"
        "+++ b/tests/test_hidden.py\n@@ -1 +1,2 @@\n x = 1\n+y = 2\n"
        "diff --git a/tests/data/case.json b/tests/data/case.json\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/tests/data/case.json\n@@ -0,0 +1 @@\n+{}\n"
    )
    model = (
        "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/tests/test_hidden.py b/tests/test_hidden.py\n--- a/tests/test_hidden.py\n"
        "+++ b/tests/test_hidden.py\n@@ -1 +1 @@\n-x = 1\n+x = 3\n"
    )
    row = {"test_patch": test_patch, "FAIL_TO_PASS": ["tests/test_hidden.py::test_y"], "PASS_TO_PASS": ["t::p"]}
    facts = patch_facts(model, row)
    assert tuple(facts) == GRADE_FACT_KEYS
    assert facts == {"test_patch_paths": ["tests/data/case.json", "tests/test_hidden.py"],
                     "touched_test_paths": ["tests/test_hidden.py"],
                     "fail_to_pass": ["tests/test_hidden.py::test_y"]}
    assert patch_facts("", row)["touched_test_paths"] == [], "no diff touches nothing"

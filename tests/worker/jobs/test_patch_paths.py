"""The paths that a patch names, which a grade reports as facts about the task's hidden tests.

Each patch comes from git at test time: the agent's side from the grader's own ``DIFF``, and a
dataset's test patch from git with its defaults, because someone else's git made it. Each test
first checks that git wrote the shape that the test is about, so that another git version fails
the test instead of passing it on a patch without that shape.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from sprout.worker.jobs.grade import DIFF, GRADE_FACT_KEYS, patch_facts, patch_paths


class Repository:
    """A git repository with one commit, whose work tree a test then changes."""

    def __init__(self, path: Path, files: dict[str, bytes]) -> None:
        self.path = path
        path.mkdir()
        self.git("init", "-q")
        self.write_files(files)
        self.git("add", "-A")
        self.git("-c", "user.name=Test", "-c", "user.email=test@example.invalid", "commit", "-qm", "base")
        self.base = self.git("rev-parse", "HEAD").decode().strip()

    def git(self, *argv: str) -> bytes:
        return subprocess.run(["git", *argv], cwd=self.path, check=True, capture_output=True).stdout

    def write(self, name: str, data: bytes) -> None:
        (self.path / name).parent.mkdir(parents=True, exist_ok=True)
        (self.path / name).write_bytes(data)

    def write_files(self, files: dict[str, bytes]) -> None:
        for name, data in files.items():
            self.write(name, data)

    def graded_diff(self) -> str:
        """The work tree against the base commit, as the grader takes it from a final snapshot."""
        patch = self.path.parent / "model.patch"
        subprocess.run(["sh", "-c", DIFF % (self.base, patch)], cwd=self.path, check=True)
        return patch.read_bytes().decode("utf-8", errors="replace")

    def default_diff(self, *options: str) -> str:
        """The same change as git writes it with its defaults, such as ``\\ooo`` for each non-ASCII byte."""
        self.git("add", "-A")
        return self.git("diff", "--cached", *options, self.base).decode("utf-8", errors="replace")


@pytest.fixture
def repository(tmp_path):
    return lambda files: Repository(tmp_path / "repository", files)


def numbered_lines(count: int) -> bytes:
    return b"".join(b"line %d\n" % number for number in range(count))


def test_a_rename_names_its_old_and_its_new_path(repository):
    """A pure rename has no ---/+++ lines. A header with a space in a name is ambiguous."""
    repo = repository({"old.py": b"value = 1\n", "old name.py": b"x\n", "edited.py": numbered_lines(20)})
    repo.git("mv", "old.py", "new.py")
    repo.git("mv", "old name.py", "new name.py")
    repo.git("mv", "edited.py", "renamed.py")
    repo.write("renamed.py", numbered_lines(19) + b"changed\n")

    diff = repo.graded_diff()

    assert "rename from old name.py\nrename to new name.py\n" in diff and "+++ b/renamed.py\n" in diff, diff
    assert patch_paths(diff) == ["edited.py", "new name.py", "new.py", "old name.py", "old.py", "renamed.py"]


def test_a_copy_names_its_source_and_its_copy(repository):
    repo = repository({"source.py": numbered_lines(20)})
    repo.write("copy.py", numbered_lines(20))

    diff = repo.default_diff("-C", "-C")

    assert "copy from source.py\ncopy to copy.py\n" in diff, diff
    assert patch_paths(diff) == ["copy.py", "source.py"]


def test_a_binary_file_is_named_by_its_header_only(repository):
    """``--binary`` writes base85 after ``GIT binary patch``. Without it, one ``Binary files`` line."""
    repo = repository({"blob.bin": b"\x00\x01" + bytes(range(256)) * 8})
    repo.write("blob.bin", b"\x00\x02" + bytes(reversed(range(256))) * 8)
    repo.write("added.bin", b"\x00\x03" * 64)

    graded, summary = repo.graded_diff(), repo.default_diff()

    assert graded.count("GIT binary patch") == 2 and "--- " not in graded, graded
    assert "Binary files /dev/null and b/added.bin differ" in summary, summary
    assert (patch_paths(graded), patch_paths(summary)) == (
        ["added.bin", "blob.bin"],
        ["added.bin", "blob.bin"],
    )


def test_a_name_with_a_space_ends_at_the_tab_that_git_writes_after_it(repository):
    repo = repository({"docs/my notes.txt": b"x\n"})
    repo.write("docs/my notes.txt", b"x\ny\n")

    diff = repo.graded_diff()

    assert "--- a/docs/my notes.txt\t\n+++ b/docs/my notes.txt\t\n" in diff, diff
    assert patch_paths(diff) == ["docs/my notes.txt"]


#: Names that git quotes: a quote, a tab, and non-ASCII letters unless ``core.quotepath`` is off.
NAMES = ["café/naïve.py", 'we"ird.py', "tab\there.py"]
ORIGINAL = {name: b"x\n" for name in NAMES}
CHANGED = {name: b"x\ny\n" for name in NAMES}


def test_a_name_that_the_grader_quotes_reads_as_the_file_name(repository):
    """The grader's diff keeps UTF-8 raw (``core.quotepath=off``), and quotes a quote and a tab."""
    repo = repository(ORIGINAL)
    repo.write_files(CHANGED)

    diff = repo.graded_diff()

    assert "diff --git a/café/naïve.py b/café/naïve.py\n" in diff, diff
    assert '"a/we\\"ird.py"' in diff and '"a/tab\\there.py"' in diff, diff
    assert patch_paths(diff) == sorted(NAMES)


def test_a_name_that_git_quotes_by_default_reads_as_the_file_name(repository):
    """By default git writes each non-ASCII byte as ``\\ooo``. The bytes decode as UTF-8 only together."""
    repo = repository(ORIGINAL)
    repo.write_files(CHANGED)

    diff = repo.default_diff()

    assert '"a/caf\\303\\251/na\\303\\257ve.py"' in diff and '"a/we\\"ird.py"' in diff, diff
    assert patch_paths(diff) == sorted(NAMES)


def test_a_deletion_a_mode_change_and_an_empty_file_are_named_without_dev_null(repository):
    """A mode change and an empty new file have no ---/+++ lines, only their header."""
    repo = repository({"gone.py": b"x\n", "run.sh": b"echo\n"})
    (repo.path / "gone.py").unlink()
    (repo.path / "run.sh").chmod(0o755)
    repo.write("empty.txt", b"")

    diff = repo.graded_diff()

    assert "+++ /dev/null\n" in diff and "old mode 100644\nnew mode 100755\n" in diff, diff
    assert "--- a/run.sh" not in diff and "--- a/empty.txt" not in diff and "--- /dev/null" not in diff, diff
    assert patch_paths(diff) == ["empty.txt", "gone.py", "run.sh"]


def test_a_hunk_line_that_reads_like_a_file_header_names_nothing(repository):
    """A removed ``-- note`` is ``--- note`` in the hunk, and an added ``++ added`` is ``+++ added``."""
    repo = repository({"schema.sql": b"select 1;\n-- note\nselect 2;\n"})
    repo.write("schema.sql", b"select 1;\nselect 2;\n++ added\n")

    diff = repo.graded_diff()

    assert "\n--- note\n" in diff and "\n+++ added\n" in diff, diff
    assert patch_paths(diff) == ["schema.sql"]


def test_a_plain_unified_diff_names_the_files_of_its_header_triples(tmp_path):
    """Without ``diff --git``, each file opens with ---, +++ and @@. A name ends at the TAB before a date."""
    (tmp_path / "a" / "src").mkdir(parents=True)
    (tmp_path / "b" / "src").mkdir(parents=True)
    (tmp_path / "b" / "tests").mkdir()
    (tmp_path / "a" / "src" / "app.py").write_text("one\n-- removed\ntwo\n")
    (tmp_path / "b" / "src" / "app.py").write_text("one\n++ added\ntwo\n")
    (tmp_path / "b" / "tests" / "test_new.py").write_text("def test(): pass\n")

    diff = subprocess.run(["diff", "-ruN", "a", "b"], cwd=tmp_path, capture_output=True, text=True).stdout

    assert "diff --git" not in diff and "\n--- removed\n+++ added\n two\n" in diff, diff
    assert "--- a/src/app.py\t" in diff and "--- a/tests/test_new.py\t" in diff, diff
    assert patch_paths(diff) == ["src/app.py", "tests/test_new.py"]


def test_the_grade_facts_name_the_hidden_tests_and_the_ones_that_the_agent_touched():
    test_patch = (
        "diff --git a/tests/test_hidden.py b/tests/test_hidden.py\n--- a/tests/test_hidden.py\n"
        "+++ b/tests/test_hidden.py\n@@ -1 +1,2 @@\n x = 1\n+y = 2\n"
        "diff --git a/tests/data/case.json b/tests/data/case.json\nnew file mode 100644\n"
        "--- /dev/null\n+++ b/tests/data/case.json\n@@ -0,0 +1 @@\n+{}\n"
    )
    model_patch = (
        "diff --git a/src/app.py b/src/app.py\n--- a/src/app.py\n+++ b/src/app.py\n@@ -1 +1 @@\n-a\n+b\n"
        "diff --git a/tests/test_hidden.py b/tests/test_hidden.py\n--- a/tests/test_hidden.py\n"
        "+++ b/tests/test_hidden.py\n@@ -1 +1 @@\n-x = 1\n+x = 3\n"
    )
    row = {
        "test_patch": test_patch,
        "FAIL_TO_PASS": ["tests/test_hidden.py::test_y"],
        "PASS_TO_PASS": ["t::p"],
    }

    facts = patch_facts(model_patch, row)

    assert facts == {
        "test_patch_paths": ["tests/data/case.json", "tests/test_hidden.py"],
        "touched_test_paths": ["tests/test_hidden.py"],
        "fail_to_pass": ["tests/test_hidden.py::test_y"],
    }


def test_each_fact_that_a_grade_adds_is_a_grade_fact_key():
    """The review hides each of ``GRADE_FACT_KEYS`` from its prompts. A new fact needs its key there."""
    row = {"test_patch": "", "FAIL_TO_PASS": [], "PASS_TO_PASS": []}

    facts = patch_facts("", row)

    assert sorted(facts) == sorted(GRADE_FACT_KEYS)

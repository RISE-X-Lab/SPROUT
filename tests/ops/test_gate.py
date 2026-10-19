"""The gates under ``gate/``: each check, run on a git repository of its own in ``tmp_path``.

A check finds the repository from its own path (``parents[2]``), so each test copies the check
into ``<repo>/gate/checks/`` and runs it there as a child process.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys

import pytest

from tests.support.paths import REPO


def environment(**overrides):
    """This process's environment with ``overrides`` and without the ``GIT_`` variables."""
    # A git hook sets GIT_DIR and GIT_INDEX_FILE, which would point the test's git at the outer repository.
    return {**{key: value for key, value in os.environ.items() if not key.startswith("GIT_")}, **overrides}


def repository(tmp_path, files, *, untracked=None):
    """A git repository with ``files`` (path -> text or bytes) in its index, and ``untracked`` beside them."""
    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q"], cwd=repo, check=True, env=environment())
    for name, content in {**files, **(untracked or {})}.items():
        path = repo / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if isinstance(content, bytes) else content.encode())
    subprocess.run(["git", "add", "--", *files], cwd=repo, check=True, env=environment())
    return repo


def run_check(repo, check, *, env=None):
    """Run ``gate/<check>`` of this checkout as the check of ``repo``: (exit code, stderr)."""
    script = repo / "gate" / check
    script.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(REPO / "gate" / check, script)
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=repo,
        capture_output=True,
        text=True,
        timeout=120,
        env=env or environment(),
    )
    return result.returncode, result.stderr


#: A home directory, joined from parts so that this file passes the check that it tests.
ALICE = "/".join(["", "home", "alice"])


def test_no_host_paths_fails_on_a_home_directory_in_a_tracked_file(tmp_path):
    repo = repository(tmp_path, {"docs/notes.md": "Run it:\n    cd %s/sprout && make\n" % ALICE})

    assert run_check(repo, "checks/no_host_paths.py") == (
        1,
        "docs/notes.md:2: %s -- use a placeholder under /path/to/\n" % ALICE,
    )


def test_no_host_paths_passes_a_nested_home_a_binary_file_and_an_untracked_file(tmp_path):
    repo = repository(
        tmp_path,
        {
            "a.md": "the store is /opt/aenv/home/snapshot-store and ~/sprout-btrfs\n",
            "b.bin": b"\0" + ALICE.encode(),
        },
        untracked={"c.md": ALICE},
    )

    assert run_check(repo, "checks/no_host_paths.py") == (0, "")


#: A docs/ tree that every rule of docs_layout.py accepts.
DOCS = {
    "README.md": "See [runs](docs/RUNS.md), [decisions](docs/decisions/README.md) and "
    "![logo](docs/figures/logo.svg).\n",
    "docs/RUNS.md": "Runs. ![plot](figures/plot.png)\n",
    "docs/figures/logo.svg": "<svg/>",
    "docs/figures/plot.png": b"\x89PNG",
    "docs/decisions/README.md": "- [0001](0001-store.md)\n",
    "docs/decisions/0001-store.md": "# 1. Store\n",
}


def test_docs_layout_passes_a_tree_that_keeps_every_rule(tmp_path):
    repo = repository(tmp_path, DOCS)

    assert run_check(repo, "checks/docs_layout.py") == (0, "")


@pytest.mark.parametrize(
    "extra,problem",
    [
        ({"src/diagram.png": b"\x89PNG"}, "src/diagram.png: an image outside docs/figures/"),
        ({"docs/figures/notes.md": "x\n"}, "docs/figures/notes.md: docs/figures/ holds images only"),
        (
            {"docs/figures/old.svg": "<svg/>"},
            "docs/figures/old.svg: referenced by neither README.md nor docs/*.md",
        ),
        ({"docs/data.json": "{}"}, "docs/data.json: docs/ outside docs/figures/ holds markdown only"),
        (
            {"docs/decisions/0002-gate.md": "# 2\n"},
            "docs/decisions/0002-gate.md: docs/decisions/README.md does not list it",
        ),
        ({"docs/ORPHAN.md": "x\n"}, "docs/ORPHAN.md: README.md does not link to it"),
    ],
    ids=[
        "image-outside-figures",
        "figures-not-an-image",
        "unreferenced-figure",
        "docs-not-markdown",
        "unlisted-adr",
        "unlinked-doc",
    ],
)
def test_docs_layout_fails_on_each_broken_rule(tmp_path, extra, problem):
    repo = repository(tmp_path, {**DOCS, **extra})

    assert run_check(repo, "checks/docs_layout.py") == (1, problem + "\n")


def python_lines(count):
    return "".join("x%d = %d\n" % (n, n) for n in range(count))


def files_in(package, count):
    """``count`` Python files in ``package``, __init__.py first."""
    return {"%s/%s.py" % (package, "__init__" if n == 0 else "m%d" % n): "" for n in range(count)}


def test_src_layout_passes_a_package_at_each_limit(tmp_path):
    repo = repository(
        tmp_path,
        {**files_in("src/sprout/a/b/c/d", 10), "src/sprout/long.py": python_lines(1000)},
    )

    assert run_check(repo, "checks/src_layout.py") == (0, "")


@pytest.mark.parametrize(
    "files,problem",
    [
        ({"src/sprout/a/b/c/d/e/x.py": ""}, "src/sprout/a/b/c/d/e/: 6 levels deep (max 5)"),
        (files_in("src/sprout/pkg", 11), "src/sprout/pkg/: 11 files (max 10)"),
        (
            {"src/sprout/schema.sql": "select 1;\n"},
            "src/sprout/schema.sql: not Python source; data lives in resources/",
        ),
        ({"src/sprout/long.py": python_lines(1001)}, "src/sprout/long.py: 1001 lines (max 1000)"),
    ],
    ids=["too-deep", "too-many-files", "not-python", "too-long"],
)
def test_src_layout_fails_one_past_each_limit(tmp_path, files, problem):
    repo = repository(tmp_path, files)

    assert run_check(repo, "checks/src_layout.py") == (1, problem + "\n")


def docstring(lines):
    return '"""' + "\n".join("line %d" % n for n in range(1, lines + 1)) + '\n"""\n'


@pytest.mark.parametrize(
    "name,text,code,problems",
    [
        (
            "ok.py",
            docstring(19)
            + "def f():\n    "
            + docstring(4).replace("\n", "\n    ")
            + "\n# one\n# two\nx = 1\n",
            0,
            "",
        ),
        ("module.py", docstring(20), 1, "module.py:1: file docstring is 21 lines (max 20)\n"),
        (
            "function.py",
            '"""M."""\n\n\ndef f():\n    ' + docstring(5).replace("\n", "\n    ") + "\n",
            1,
            "function.py:5: docstring is 6 lines (max 5)\n",
        ),
        (
            "comments.py",
            '"""M."""\n# one\n# two\n# three\nx = 1\n',
            1,
            "comments.py:2: 3 consecutive comment lines (max 2)\n",
        ),
        ("run.sh", "#!/bin/sh\n" + "# doc\n" * 20 + "set -eu\n" + "# a\n" * 5 + "f() {\n  :\n}\n", 0, ""),
        (
            "main.go",
            "// Package main.\npackage main\n\n" + "// f\n" * 6 + "func f() {}\n",
            1,
            "main.go:4: docstring is 6 lines (max 5)\n",
        ),
    ],
    ids=[
        "at-each-limit",
        "file-docstring-over",
        "function-docstring-over",
        "comment-run-over",
        "shell-blocks-at-limit",
        "go-block-over",
    ],
)
def test_comment_length_reports_each_comment_over_its_limit(tmp_path, name, text, code, problems):
    repo = repository(tmp_path, {name: text})

    assert run_check(repo, "checks/comment_length.py") == (code, problems)


def fake_go(tmp_path, tools):
    """A ``go`` first on PATH whose ``go run . --dump-schema`` prints a schema of ``tools``."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    schema = '{"tools": [%s]}' % ", ".join('{"name": "%s"}' % tool for tool in tools)
    (bin_dir / "go").write_text("#!/bin/sh\necho '%s'\n" % schema)
    (bin_dir / "go").chmod(0o755)
    return environment(PATH="%s:%s" % (bin_dir, os.environ["PATH"]))


ALLOWED = ["shell", "text_editor", "grep_files", "web_fetch", "web_search"]


def schema_file(tools):
    return {
        "runtime/schema/tools.json": '{"tools": [%s]}' % ", ".join('{"name": "%s"}' % tool for tool in tools)
    }


def test_runtime_tools_passes_when_the_runtime_and_its_schema_declare_the_allowed_tools(tmp_path):
    repo = repository(tmp_path, schema_file(ALLOWED))

    assert run_check(repo, "checks/runtime_tools.py", env=fake_go(tmp_path, ALLOWED)) == (0, "")


@pytest.mark.parametrize(
    "served,declared,where",
    [
        (ALLOWED + ["browser"], ALLOWED, "the runtime (--dump-schema)"),
        (ALLOWED, ALLOWED + ["browser"], "runtime/schema/tools.json"),
    ],
    ids=["served", "declared"],
)
def test_runtime_tools_fails_on_a_tool_outside_the_allowed_set(tmp_path, served, declared, where):
    repo = repository(tmp_path, schema_file(declared))

    code, stderr = run_check(repo, "checks/runtime_tools.py", env=fake_go(tmp_path, served))

    assert (code, stderr.split("\n")[0]) == (
        1,
        "runtime_tools: %s declares tools outside the allowed set: browser" % where,
    )


def test_runtime_tools_exits_2_without_go(tmp_path):
    repo = repository(tmp_path, schema_file(ALLOWED))
    (tmp_path / "empty").mkdir()

    code, stderr = run_check(repo, "checks/runtime_tools.py", env=environment(PATH=str(tmp_path / "empty")))

    assert (code, stderr) == (
        2,
        "runtime_tools: go not found; it is needed to ask the runtime what it serves\n",
    )


def gate_repository(tmp_path, checks):
    """A repository whose gate/checks/ holds ``checks`` (name -> (shell body, executable)) and gate/run.sh."""
    repo = repository(tmp_path, {"gate/run.sh": (REPO / "gate" / "run.sh").read_text()})
    (repo / "gate" / "checks" / "__pycache__").mkdir(parents=True)
    for name, (body, executable) in checks.items():
        path = repo / "gate" / "checks" / name
        path.write_text("#!/bin/sh\n" + body + "\n")
        path.chmod(0o755 if executable else 0o644)
    return repo


def run_gates(repo):
    result = subprocess.run(
        ["bash", "gate/run.sh"], cwd=repo, capture_output=True, text=True, timeout=60, env=environment()
    )
    return result.returncode, result.stdout, result.stderr


def test_run_sh_passes_when_every_check_passes(tmp_path):
    repo = gate_repository(tmp_path, {"a": ("exit 0", True), "b": ("exit 0", True)})

    assert run_gates(repo) == (0, "gate: pass  gate/checks/a\ngate: pass  gate/checks/b\n", "")


def test_run_sh_fails_on_a_failed_check_and_on_a_check_that_is_not_executable(tmp_path):
    """A check that cannot run must not read as one that passed."""
    repo = gate_repository(tmp_path, {"a": ("exit 0", True), "b": ("exit 3", True), "c": ("exit 0", False)})

    assert run_gates(repo) == (
        1,
        "gate: pass  gate/checks/a\n",
        "gate: FAIL  gate/checks/b\ngate: gate/checks/c is not executable\n",
    )

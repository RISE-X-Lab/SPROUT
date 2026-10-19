"""Each module imports, an import runs no call statement, and no tracked file holds a credential.

The modules are the Python files under ``src/``, ``examples/`` and ``ops/``. A file in a package is
named from the root of its package. A script, which is a file in a directory without
``__init__.py``, is imported by its stem from its own directory, as ``python <script>`` runs it.
"""

from __future__ import annotations

import ast
import json
import os
import re
import subprocess
import sys
from pathlib import Path

from tests.support.paths import REPO

SRC = REPO / "src"
ROOTS = (SRC, REPO / "examples", REPO / "ops")
#: Imports each module, and writes {module: error} to a file. A missing third-party module is no error.
IMPORT_PROBE = """
import importlib, json, sys
targets, ours, report = json.loads(sys.argv[1]), set(json.loads(sys.argv[2])), sys.argv[3]
failures = {}
for directory, name in targets:
    sys.path.insert(0, directory)
    try:
        importlib.import_module(name)
    except ModuleNotFoundError as error:
        if (error.name or "").partition(".")[0] in ours:
            failures[name] = "ModuleNotFoundError: %s" % error
    except BaseException as error:
        failures[name] = "%s: %s" % (type(error).__name__, error)
    finally:
        sys.path.remove(directory)
open(report, "w").write(json.dumps(failures))
"""
#: The key of a provider: OpenAI or DeepSeek (sk-), GitHub, AWS, Slack or Hugging Face.
PROVIDER_KEY = re.compile(
    r"(?<![\w-])(?:sk-[A-Za-z0-9_-]{20,}|ghp_[A-Za-z0-9]{36}|github_pat_\w{22,}|AKIA[0-9A-Z]{16}"
    r"|xox[abpr]-[A-Za-z0-9-]{10,}|hf_[A-Za-z0-9]{30,})"
)
#: A long opaque value assigned to a key, a token, a secret or a password.
ASSIGNED_SECRET = re.compile(r"""(?i)(?:api_?key|token|secret|password)["']?\s*[:=]\s*["']?[\w+/=.-]{32,}""")


def python_files() -> list[Path]:
    return sorted(path for root in ROOTS for path in root.rglob("*.py"))


def import_target(path: Path) -> tuple[str, str]:
    """The directory to put on ``sys.path`` for the module at ``path``, and the name to import it by."""
    if not (path.parent / "__init__.py").is_file():
        return str(path.parent), path.stem
    base = SRC if path.is_relative_to(SRC) else REPO
    parts = path.relative_to(base).with_suffix("").parts
    return str(base), ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def import_failures(*, home: Path) -> dict[str, str]:
    """Each module that fails to import in a new interpreter whose home is ``home``, with its error."""
    targets = [import_target(path) for path in python_files()]
    ours = sorted({name.partition(".")[0] for _, name in targets} | {"tests"})
    report = home / "failures.json"
    argv = [sys.executable, "-c", IMPORT_PROBE, json.dumps(targets), json.dumps(ours), str(report)]
    environment = {**os.environ, "HOME": str(home)}
    run = subprocess.run(argv, cwd=REPO, env=environment, capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, run.stderr
    return json.loads(report.read_text())


def is_main_guard(node: ast.stmt) -> bool:
    return (
        isinstance(node, ast.If)
        and isinstance(node.test, ast.Compare)
        and isinstance(node.test.left, ast.Name)
        and node.test.left.id == "__name__"
    )


def calls_in(path: Path) -> list[str]:
    """Each call statement that an import of ``path`` runs, outside a function and the ``__main__`` guard."""
    found, pending = [], list(ast.parse(path.read_text(encoding="utf-8")).body)
    while pending:
        node = pending.pop()
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) or is_main_guard(node):
            continue
        if isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            found.append(f"{path.relative_to(REPO)}:{node.lineno}")
        pending.extend(child for child in ast.iter_child_nodes(node) if isinstance(child, ast.stmt))
    return found


def calls_at_import_time() -> list[str]:
    """Each call statement that an import of a module runs, as ``<file>:<line>``."""
    return [call for path in python_files() for call in calls_in(path)]


def tracked_texts() -> dict[str, str]:
    """The text of each tracked file that is UTF-8, by its path in the repository."""
    listed = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, capture_output=True, text=True, check=True)
    texts = {}
    for name in filter(None, listed.stdout.split("\0")):
        try:
            texts[name] = (REPO / name).read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
    return texts


def literal_credentials() -> list[str]:
    """Each line of a tracked file that holds a provider key or a long secret value, as ``<file>:<line>``."""
    return [
        f"{name}:{number}"
        for name, text in tracked_texts().items()
        for number, line in enumerate(text.splitlines(), 1)
        if PROVIDER_KEY.search(line) or ASSIGNED_SECRET.search(line)
    ]


def test_every_module_imports(tmp_path):
    """A deleted or moved module leaves imports that name it. A missing third-party module is no
    failure, because CI does not install the ``docs`` extra."""
    failures = import_failures(home=tmp_path)

    assert failures == {}


def test_no_module_runs_a_call_when_it_is_imported():
    """A benchmark script once called ``main()`` without a ``__name__`` guard. An import sweep then
    started a run of 500 instances and spent real money before someone stopped it."""
    calls = calls_at_import_time()

    assert calls == []


def test_no_tracked_file_holds_a_literal_credential():
    """A real DeepSeek key once sat in two tracked configs, one push from being public. Credentials
    come from the environment or from a key file."""
    found = literal_credentials()

    assert found == []

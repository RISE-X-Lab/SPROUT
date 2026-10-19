"""The search package reaches the Run Store over HTTP only: it opens no database and runs no job.

A worker runs each job, and the Run Store's API is the only way to the records of the jobs
(``sprout.search.client``).
"""

from __future__ import annotations

import ast
from pathlib import Path

from tests.support.paths import REPO

#: The modules that would let the search reach the jobs' records or run a job itself.
FORBIDDEN = (
    "sprout.runstore.store",
    "sprout.runstore.index",
    "sprout.worker",
    "sprout_sandbox",
    "psycopg2",
    "subprocess",
)


def imported_modules(path: Path) -> set[str]:
    """Every module that the file imports, at any level of its code."""
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return modules


def forbidden_imports(directory: Path) -> list[tuple[str, str]]:
    """The file name and the module of each forbidden import in the Python files of ``directory``."""
    paths = sorted(directory.glob("*.py"))
    assert paths, "%s holds no Python file" % directory
    return [
        (path.name, module)
        for path in paths
        for module in sorted(imported_modules(path))
        if module.startswith(FORBIDDEN)
    ]


def test_no_module_of_the_search_package_imports_the_run_store_database_a_worker_or_a_sandbox():
    assert forbidden_imports(REPO / "src" / "sprout" / "search") == []

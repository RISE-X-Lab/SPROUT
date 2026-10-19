"""Imports point down the layers of ADR-0001 and ADR-0002, so each process loads only the code below it.

A layer is a package directly under ``src/sprout``. ``sprout_sandbox`` (the sandbox clients) and the
package root ``sprout`` (with ``sprout.resource``, the data file loader) are below every layer.
``ops`` is above every layer, and no row checks it. An import counts wherever a file writes it: in a
function, under ``TYPE_CHECKING``, or as a relative import.
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.paths import REPO

SRC = REPO / "src"
#: The top-level packages of this repository other than ``sprout``.
OTHER_PACKAGES = ("sprout_sandbox", "examples", "ops", "tests")
#: The packages below every layer.
LEAVES = {"sprout_sandbox", "sprout"}
#: Each package, and the packages of this repository that it may import besides itself.
LAYERS = {
    "sprout_sandbox": set(),
    "sprout": set(),
    "contract": set(),
    "runstore": LEAVES | {"contract"},
    "worker": LEAVES | {"contract", "runstore"},
    "search": LEAVES | {"contract", "runstore"},
    "rl_driver": LEAVES | {"contract", "runstore", "search"},
    "eval": LEAVES | {"contract", "runstore", "search", "worker"},
    "services": LEAVES | {"contract"},
    "examples": LEAVES | {"contract", "worker", "eval"},
}


def package_files(package: str) -> list[Path]:
    """The Python files of ``package``. For ``sprout``, only the modules directly in ``src/sprout``."""
    if package == "sprout":
        return sorted((SRC / "sprout").glob("*.py"))
    if package in OTHER_PACKAGES:
        root = SRC / package if (SRC / package).is_dir() else REPO / package
        return sorted(root.rglob("*.py"))
    return sorted((SRC / "sprout" / package).rglob("*.py"))


def packages_under(directory: Path) -> set[str]:
    return {init.parent.name for init in directory.glob("*/__init__.py")}


def module_parts(path: Path) -> list[str]:
    """The dotted name of the module at ``path``, as parts. A package's ``__init__.py`` is the package."""
    base = SRC if path.is_relative_to(SRC) else REPO
    parts = list(path.relative_to(base).with_suffix("").parts)
    return parts[:-1] if parts[-1] == "__init__" else parts


def imported_modules(path: Path) -> set[str]:
    """Each module that ``path`` imports, made absolute, with each name of a ``from`` import joined on."""
    package = module_parts(path) if path.name == "__init__.py" else module_parts(path)[:-1]
    found = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            base = package[: len(package) - node.level + 1] if node.level else []
            module = ".".join([*base, *([node.module] if node.module else [])])
            found.add(module)
            found.update(f"{module}.{alias.name}" for alias in node.names)
    return found


def package_of(module: str) -> str | None:
    """The package of this repository that ``module`` is in. None for the standard library and others."""
    top, _, rest = module.partition(".")
    if top == "sprout":
        layer = rest.partition(".")[0]
        return layer if layer and (SRC / "sprout" / layer / "__init__.py").is_file() else "sprout"
    return top if top in OTHER_PACKAGES else None


def imports_outside(*, files: list[Path], package: str, allowed: set[str]) -> list[str]:
    """Each import in ``files`` of a package outside ``allowed``, as ``<file> imports <package>``."""
    found = {
        f"{path.relative_to(REPO)} imports {package_of(module)}"
        for path in files
        for module in imported_modules(path)
        if package_of(module) not in allowed | {package, None}
    }
    return sorted(found)


def modules_loaded_by(statement: str) -> list[str]:
    """The modules that a new interpreter loads to run ``statement``, beyond the ones it loads at startup."""
    probe = (
        "import json, sys; before = set(sys.modules); exec(sys.argv[1]); "
        "print(json.dumps(sorted(set(sys.modules) - before)))"
    )
    argv = [sys.executable, "-c", probe, statement]
    run = subprocess.run(argv, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


@pytest.mark.parametrize("package,allowed", LAYERS.items(), ids=list(LAYERS))
def test_each_package_imports_only_the_packages_below_it(package, allowed):
    files = package_files(package)

    offenders = imports_outside(files=files, package=package, allowed=allowed)

    assert files != [], f"{package} has no Python files, so its row checks nothing"
    assert offenders == []


def test_each_package_under_src_sprout_has_a_row_in_the_layering_table():
    """A package without a row could import any layer, and no other test would see it."""
    packages = packages_under(SRC / "sprout")

    assert packages - LAYERS.keys() == set()


def test_importing_the_worker_package_loads_no_other_module():
    """``python -m ops.report`` loads three modules under ``sprout.worker``. The package root once
    re-exported the journal and the slot contract, so each of them also loaded the run control and httpx."""
    loaded = modules_loaded_by("import sprout.worker")

    assert loaded == ["sprout", "sprout.worker"]

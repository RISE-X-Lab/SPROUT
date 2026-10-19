"""The service manager imports only the standard library, PyYAML, ``sprout.contract`` and its own modules.

So ``python -m sprout.services`` runs before the optional extras are installed: it is what installs the
proxy's venv. ``tests/repo/test_layering.py`` forbids the other ``sprout`` layers. This test also forbids
each other package, such as ``sprout_sandbox`` or ``httpx``, in the services and in each module of
``sprout.contract`` that they load. An import inside a function counts.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from tests.support.paths import REPO

PACKAGE = REPO / "src" / "sprout" / "services"
CONTRACT = REPO / "src" / "sprout" / "contract"
OWN = ("sprout.services", "sprout.resource", "sprout.contract")


def imported_modules(path: Path) -> set[str]:
    """Each module that ``path`` imports by an absolute name, at any level of the file."""
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
            # `from sprout.contract import backend` imports the module sprout.contract.backend.
            modules.update("%s.%s" % (node.module, alias.name) for alias in node.names)
    return modules


def is_allowed(module: str) -> bool:
    top = module.split(".")[0]
    own = any(module == name or module.startswith(name + ".") for name in OWN)
    return top in sys.stdlib_module_names or top == "yaml" or own


def foreign_imports(files: list[Path]) -> list[str]:
    """``<file> imports <module>`` for each import in ``files`` of a module outside the allowed set."""
    assert files, "no modules to check"
    return [
        "%s imports %s" % (path.relative_to(REPO), module)
        for path in files
        for module in sorted(imported_modules(path))
        if not is_allowed(module)
    ]


def contract_file(module: str) -> Path | None:
    """The file of the ``sprout.contract`` module ``module``, or None for any other name."""
    file = CONTRACT / ("%s.py" % module.removeprefix("sprout.contract."))
    return file if module.startswith("sprout.contract.") and file.is_file() else None


def contract_files_loaded(files: list[Path]) -> list[Path]:
    """The file of each ``sprout.contract`` module that ``files`` import, directly or through each other."""
    found: set[Path] = set()
    pending = list(files)
    while pending:
        for module in imported_modules(pending.pop()):
            file = contract_file(module)
            if file is not None and file not in found:
                found.add(file)
                pending.append(file)
    return sorted(found)


def test_the_services_package_imports_only_the_standard_library_pyyaml_the_contract_and_itself():
    assert foreign_imports(sorted(PACKAGE.rglob("*.py"))) == []


def test_each_contract_module_that_the_services_load_imports_only_the_standard_library_and_pyyaml():
    """``sprout.contract.mini_history`` imports jsonschema, so the services must not import that module."""
    loaded = contract_files_loaded(sorted(PACKAGE.rglob("*.py")))

    assert foreign_imports(loaded) == []

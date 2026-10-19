"""The service manager imports only the standard library, PyYAML and its own modules.

So ``python -m sprout.services`` runs before the optional extras are installed: it is what installs the
proxy's venv. ``tests/repo/test_layering.py`` forbids the ``sprout`` layers. This test also forbids
each other package, such as ``sprout_sandbox`` or ``httpx``. An import inside a function counts.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from tests.support.paths import REPO

PACKAGE = REPO / "src" / "sprout" / "services"
OWN = ("sprout.services", "sprout.resource")


def imported_modules(path: Path) -> set[str]:
    """Each module that ``path`` imports by an absolute name, at any level of the file."""
    modules = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return modules


def is_allowed(module: str) -> bool:
    top = module.split(".")[0]
    own = any(module == name or module.startswith(name + ".") for name in OWN)
    return top in sys.stdlib_module_names or top == "yaml" or own


def foreign_imports(package: Path) -> list[str]:
    """``<file> imports <module>`` for each import of a module outside the allowed set."""
    files = sorted(package.glob("*.py"))
    assert files, "no modules in %s" % package
    return [
        "%s imports %s" % (path.name, module)
        for path in files
        for module in sorted(imported_modules(path))
        if not is_allowed(module)
    ]


def test_the_services_package_imports_only_the_standard_library_pyyaml_and_itself():
    assert foreign_imports(PACKAGE) == []

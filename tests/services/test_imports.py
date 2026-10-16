"""The service manager starts the other layers; it must not import them.

It has to work before the optional extras are installed (it is what installs
the proxy's venv) and must never inherit a fault from the code it is bringing
up. ``tests/harness/test_layering.py`` already forbids the ``sprout.*`` layers;
this pins the rest: the standard library, PyYAML and the package itself,
``sprout_sandbox`` included in the forbidden set.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

from sprout.services.spec import template_path

PACKAGE = Path(__file__).resolve().parents[2] / "src" / "sprout" / "services"


def _roots(path: Path) -> set[str]:
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            found.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            found.add(node.module)
    return found


def test_the_package_imports_only_the_stdlib_yaml_and_itself():
    files = sorted(PACKAGE.glob("*.py"))
    assert len(files) >= 8, "the package lost modules"
    offenders = []
    for path in files:
        for module in _roots(path):
            top = module.split(".")[0]
            if top in sys.stdlib_module_names or top == "yaml":
                continue
            if module in ("sprout.services", "sprout.resource") or module.startswith("sprout.services."):
                continue
            offenders.append("%s imports %s" % (path.name, module))
    assert not offenders, offenders
    assert template_path("services.yaml").is_file() and template_path("litellm.yaml").is_file()

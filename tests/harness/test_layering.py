"""Dependencies point down only; this pins the layering for every package.

    rl_driver, examples      protocol adapters and experiment drivers
          |
       runstore, inspect     queued rollout and grade jobs; the read-only CLI over run output
          |
       methods               evaluate / branching / policies / gate
          |
       harbor_plugin         Sprout inside a Harbor trial
          |
       harness, harness_*    the agent runtime (mini-swe-agent, sandbox, tools, checkpoints)

The layers live under ``src/sprout/``; ``examples`` and ``ops`` sit beside
``src``, and ``ops`` may use any layer. ``src/sprout_sandbox`` (the sandbox
clients) sits below all of them and imports nothing from ``sprout``. Lazy
imports inside functions count: a dependency is a dependency wherever it is
written.
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
import sys

REPO = Path(__file__).resolve().parents[2]

#: What each package may import from this repository (besides itself and sdk).
ALLOWED = {
    "harness": set(),
    "harbor_plugin": {"harness"},
    "methods": {"harness", "harbor_plugin"},
    "runstore": {"harness", "methods"},
    "inspect": {"harness", "harbor_plugin", "methods"},
    "rl_driver": {"harness", "runstore"},
    "examples": {"harness", "harbor_plugin", "methods"},
    # The service manager starts the layers above; it imports none of them
    # (stdlib + PyYAML), so it works before the extras are installed.
    "services": set(),
}
PACKAGES = set(ALLOWED) | {"ops"}


def package_dirs(package: str) -> list[Path]:
    if package in ("examples", "ops"):
        return [REPO / package]
    root = REPO / "src" / "sprout"
    # A layer is <layer>/ and its parts, one <layer>_<part>/ package each.
    return [root / package, *sorted(root.glob(f"{package}_*"))]


def _layer(module: str) -> str:
    parts = module.split(".")
    name = parts[1] if parts[0] == "sprout" and len(parts) > 1 else parts[0]
    return next((layer for layer in ALLOWED if name == layer or name.startswith(layer + "_")), name)


def _imports(path: Path) -> set[str]:
    roots = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            roots.update(_layer(alias.name) for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            roots.add(_layer(node.module))
    return roots & PACKAGES


def _sprout_imports(path: Path) -> set[str]:
    """Every ``sprout.*`` module a file imports, at any level."""
    modules = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            modules.add(node.module)
    return {m for m in modules if m == "sprout" or m.startswith("sprout.")}


def test_the_sdk_imports_nothing_from_sprout():
    """The harness drives the SDK; the SDK knows nothing of the harness. The
    one-line temptation is `from sprout.harness_sandbox.templates import
    RUNTIME_PATH` in the podman pool, which carries its own copy instead."""
    paths = sorted((REPO / "src" / "sprout_sandbox").rglob("*.py"))
    assert paths, "sprout_sandbox has no modules to check"
    offenders = [f"{path.relative_to(REPO)} imports {sorted(found)}"
                 for path in paths if (found := _sprout_imports(path))]
    assert not offenders, "\n".join(offenders)


def test_every_package_imports_only_the_layers_below_it():
    offenders = []
    for package, allowed in ALLOWED.items():
        paths = sorted(p for d in package_dirs(package) for p in d.rglob("*.py"))
        assert paths, f"{package} has no modules to check"
        for path in paths:
            bad = _imports(path) - allowed - {package}
            if bad:
                offenders.append(f"{path.relative_to(REPO)} imports {sorted(bad)}")
    assert not offenders, "\n".join(offenders)


def test_the_harness_package_root_imports_nothing():
    """`python -m sprout.inspect` is stdlib plus three harness modules. The
    package root once re-exported the journal and the slot contract, so any
    import under sprout.harness also loaded the run control and httpx."""
    probe = ("import json, sys, sprout.harness; print(json.dumps(sorted("
             "m for m in sys.modules if m.split('.')[0] in ('sprout', 'httpx'))))")
    loaded = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                            check=True, timeout=60).stdout
    assert json.loads(loaded) == ["sprout", "sprout.harness"]

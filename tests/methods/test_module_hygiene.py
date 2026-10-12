"""Importing a module must not run a benchmark.

`run_claude.py` called `main()` at module level with no `__name__` guard, so
`import swebench.run_claude` (since deleted) started a 500-instance run. Found by an import sweep
that spent real money before it was killed. The file is gone; this keeps the shape
of the mistake from coming back, and doubles as a check that no module is left with
a broken import after a deletion.
"""

from __future__ import annotations

import importlib
import pkgutil


from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: Where the code lives: the sprout package under src/, plus the examples beside it.
ROOTS = {"sprout": REPO / "src" / "sprout", "examples": REPO / "examples"}


def _modules():
    names = []
    for package, path in ROOTS.items():
        names.append(package)
        names += [m.name for m in pkgutil.walk_packages([str(path)], package + ".")]
    return names


def test_the_sweep_covers_every_layer():
    """It walked a deleted package for a while and passed by checking nothing."""
    assert len(_modules()) > 50


def test_the_sweep_reaches_the_service_manager():
    """`sprout.services` has a `__main__` of its own; the sweep must import it
    (and so prove the guard) rather than stop at the packages that existed first."""
    found = set(_modules())
    assert {"sprout.services", "sprout.services.__main__", "sprout.services.supervisor"} <= found


def test_every_module_imports_cleanly():
    """A deleted module leaves dangling imports behind; this finds them."""
    broken = {}
    for name in _modules():
        try:
            importlib.import_module(name)
        except ImportError as exc:          # a missing optional dep is not a defect
            if "No module named" not in str(exc):
                broken[name] = f"{type(exc).__name__}: {exc}"
        except Exception as exc:
            broken[name] = f"{type(exc).__name__}: {exc}"
    assert not broken, f"modules fail to import: {broken}"


def test_no_module_runs_anything_at_import_time():
    """A bare `main()` / `run()` call at module level, outside a __name__ guard."""
    import ast

    offenders = []
    for path in (p for root in ROOTS.values() for p in root.rglob("*.py")):
        tree = ast.parse(path.read_text())
        for node in tree.body:                      # module level only
            if not isinstance(node, ast.Expr) or not isinstance(node.value, ast.Call):
                continue
            func = node.value.func
            name = getattr(func, "id", None) or getattr(func, "attr", "")
            if name in ("main", "run", "run_batch", "cli"):
                offenders.append(f"{path}:{node.lineno} calls {name}()")
    assert not offenders, (
        "these run on import, so `import x` executes a benchmark: " + "; ".join(offenders))

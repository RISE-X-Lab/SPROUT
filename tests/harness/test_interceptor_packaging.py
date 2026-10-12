"""One file per interceptor — the structure, asserted.

An interceptor is the unit somebody replaces, so each gets its own module: a
caller writing different truncation, a different rendering, or a rule of their own
adds a file beside these rather than appending to one shared with the interceptors
they keep.

A directory apiece was the earlier convention. It was more structure than these
need -- guardrail was the only one with enough parts to fill one, and three
packages of two files each made the set look more elaborate than it is.

The list is **not a taxonomy**: these are the interceptors we happened to need, and
a new one does not have to fit the categories. What these tests pin is that each
stays independently replaceable, because that is what erodes quietly -- the next
interceptor gets appended to whichever file is already open.
"""

from __future__ import annotations

import importlib
import pathlib

SEATS_DIR = (pathlib.Path(__file__).resolve().parents[2]
             / "src" / "sprout" / "harness" / "tools" / "interceptors")

#: Every interceptor module and the class it must export. Adding an interceptor
#: means adding a line here -- deliberately, so the structural rules below cover
#: it too.
INTERCEPTORS = {
    "mutation": "MutationTracker",
    "shell_context": "ShellContext",
}

#: Not interceptors: the package's own re-exports.
NON_INTERCEPTOR_MODULES = {"__init__"}


def _modules():
    return sorted(p for p in SEATS_DIR.glob("*.py") if p.stem != "__init__")


def test_every_interceptor_is_its_own_module():
    """One file each, and no directories: a package here would be structure the
    interceptors do not need."""
    assert not [p for p in SEATS_DIR.iterdir()
                if p.is_dir() and not p.name.startswith("__")], \
        "an interceptor grew a directory again"
    stems = {p.stem for p in _modules()}
    assert stems == set(INTERCEPTORS) | (NON_INTERCEPTOR_MODULES - {"__init__"}), stems


def test_each_module_exports_its_interceptor():
    for module_name, cls in INTERCEPTORS.items():
        module = importlib.import_module("sprout.harness.tools.interceptors.%s" % module_name)
        assert hasattr(module, cls), "%s does not export %s" % (module_name, cls)


def test_no_interceptor_imports_another_interceptor():
    """Interceptors are peers. One reaching into another would make the pair a
    unit, and replacing either would mean understanding both."""
    offenders = []
    for path in _modules():
        if path.stem in NON_INTERCEPTOR_MODULES:
            continue
        others = set(INTERCEPTORS) - {path.stem}
        source = path.read_text()
        for other in others:
            if "interceptors.%s" % other in source or "from .%s" % other in source:
                offenders.append("%s imports %s" % (path.name, other))
    assert offenders == [], offenders


def test_interceptors_are_reachable_from_one_place():
    from sprout.harness.tools.interceptors import INTERCEPTORS, MutationTracker, ShellContext

    assert MutationTracker and ShellContext
    assert set(INTERCEPTORS) == {"shell_context"}, INTERCEPTORS


def test_declared_interceptors_are_built_by_name_and_unknown_names_are_refused():
    """A run declares its rules as data; a typo must not become a silently missing rule."""
    import pytest

    from sprout.harness.tools.interceptors import ShellContext, build_interceptors

    [built] = build_interceptors([{"name": "shell_context", "cwd": "/app", "env": {"A": "1"}, "user": "nobody"}])
    assert isinstance(built, ShellContext) and built.cwd == "/app" and built.env == {"A": "1"}
    with pytest.raises(ValueError, match="unknown interceptor 'nope'"):
        build_interceptors([{"name": "nope"}])
    with pytest.raises(ValueError, match="declared as"):
        build_interceptors(["shell_context"])
    assert build_interceptors(None) == []

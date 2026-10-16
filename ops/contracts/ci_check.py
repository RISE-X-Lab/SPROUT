#!/usr/bin/env python3
"""Assert that pinned upstreams still provide what our adapters depend on.

CI runs it on every push; run it before bumping a pinned version:

    python ops/contracts/ci_check.py                    # every contract
    python ops/contracts/ci_check.py --slot harbor      # one contract, by file stem

Contracts: mini-swe-agent (the only agent) and Harbor (the trial runner the
TB2.1 / HARD-51 plugins subclass). The checks are static and need no
credentials: exact package versions, module exports, class methods and model
fields, plus the event types the mini history reader handles.

Why this exists: the adapters are built on surfaces upstream does not promise to
keep (flag names, event fields, SDK kwargs). Without this, a rename shows up as
subtly wrong trajectories weeks later instead of a red build today.

Exit codes: 0 ok, 1 contract violation, 2 harness/setup error.
"""

from __future__ import annotations

import argparse
import importlib
import inspect
import sys
from pathlib import Path

CONTRACTS_DIR = Path(__file__).parent


def load_yaml(path: Path) -> dict:
    try:
        import yaml
    except ImportError:
        print("PyYAML required: pip install pyyaml", file=sys.stderr)
        raise SystemExit(2)
    return yaml.safe_load(path.read_text())


class Report:
    def __init__(self) -> None:
        self.failures: list = []
        self.skips: list = []
        self.checks = 0

    def ok(self, label: str) -> None:
        self.checks += 1
        print("  ok      %s" % label)

    def fail(self, label: str, detail: str = "") -> None:
        self.checks += 1
        self.failures.append((label, detail))
        print("  FAIL    %s%s" % (label, (" -- " + detail) if detail else ""))

    def skip(self, label: str, reason: str) -> None:
        self.skips.append((label, reason))
        print("  skip    %s (%s)" % (label, reason))


def check_normalizer_alignment(contract: dict, report: Report) -> None:
    """The history reader must still handle every event type the contract lists."""
    slot = contract_name(contract)
    module = contract.get("normalizer_module")
    if not module:
        return
    try:
        mod = importlib.import_module(module)
    except ImportError as exc:
        report.fail("import %s" % module, str(exc))
        return

    source = Path(mod.__file__).read_text()
    for etype in contract.get("event_types") or []:
        (report.ok if etype in source else report.fail)("%s handles event %s" % (slot, etype))


def check_python_api(contract: dict, report: Report) -> None:
    """Assert the SDK surface a protocol driver calls is still there.

    These are the names a version bump can remove silently: the driver would then
    fail at run time, mid-eval, rather than here.
    """
    spec = contract.get("python_api") or {}
    if not spec:
        return
    upstream = contract.get("upstream", {})
    if upstream.get("exact_version"):
        from importlib.metadata import PackageNotFoundError, version

        try:
            installed = version(upstream["package"])
        except PackageNotFoundError:
            installed = None
        expected = upstream["verified_versions"]["sdk"]
        (report.ok if installed == expected else report.fail)(
            f"{upstream['package']} == {expected}",
            **({} if installed == expected else {"detail": f"installed: {installed}"}),
        )
    module_name = spec.get("module")
    try:
        mod = importlib.import_module(module_name)
    except ImportError as exc:
        report.skip("%s python api" % contract_name(contract), "%s not installed (%s)" % (module_name, exc))
        return

    for name in spec.get("exports") or []:
        (report.ok if hasattr(mod, name) else report.fail)("%s exports %s" % (module_name, name))

    client_spec = spec.get("client") or {}
    if client_spec:
        try:
            client_mod = importlib.import_module(client_spec["module"])
            cls = getattr(client_mod, client_spec["cls"])
        except (ImportError, AttributeError) as exc:
            report.fail("%s.%s" % (client_spec.get("module"), client_spec.get("cls")), str(exc))
            cls = None
        if cls is not None:
            params = inspect.signature(cls.__init__).parameters
            for kwarg in client_spec.get("init_kwargs") or []:
                (report.ok if kwarg in params else report.fail)(
                    "%s.__init__ accepts %s" % (client_spec["cls"], kwarg)
                )
            for method in client_spec.get("methods") or []:
                (report.ok if callable(getattr(cls, method, None)) else report.fail)(
                    "%s.%s" % (client_spec["cls"], method)
                )


def check_classes(contract: dict, report: Report) -> None:
    """Each listed class still exists with the methods and model fields we use."""
    for entry in contract.get("classes") or []:
        label = "%s.%s" % (entry["module"], entry["cls"])
        try:
            cls = getattr(importlib.import_module(entry["module"]), entry["cls"])
        except (ImportError, AttributeError) as exc:
            report.fail(label, str(exc))
            continue
        report.ok(label)
        for method in entry.get("methods") or []:
            member = inspect.getattr_static(cls, method, None)
            present = isinstance(member, property) or callable(getattr(cls, method, None))
            (report.ok if present else report.fail)("%s.%s" % (label, method))
        known = set(getattr(cls, "model_fields", None) or getattr(cls, "__dataclass_fields__", None) or {})
        for field in entry.get("fields") or []:
            (report.ok if field in known else report.fail)("%s field %s" % (label, field))


def contract_name(contract: dict) -> str:
    return contract.get("name") or contract["slot"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--slot", action="append", help="limit to these slots")
    args = parser.parse_args(argv)

    paths = sorted(CONTRACTS_DIR.glob("*.yaml"))
    if args.slot:
        wanted = set(args.slot)
        paths = [p for p in paths if p.stem in wanted]
    if not paths:
        print("no contracts matched", file=sys.stderr)
        return 2

    report = Report()
    for path in paths:
        contract = load_yaml(path)
        print("\n== %s ==" % contract_name(contract))
        check_python_api(contract, report)
        check_classes(contract, report)
        check_normalizer_alignment(contract, report)

    print("\n%d checks, %d failures, %d skipped" % (report.checks, len(report.failures), len(report.skips)))
    if report.failures:
        print("\nAdapter drift detected. Update the adapter or the contract.")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

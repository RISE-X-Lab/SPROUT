#!/usr/bin/env python3
"""The sandbox runtime serves no tool beyond the allowed set.

The runtime's tools are the agent's whole action space: shell, text_editor,
grep_files, web_fetch, web_search. A new one changes what every run can do, so
it has to be a decision made here -- added to ALLOWED in the same commit -- not
something that arrives with a runtime change.

Both declarations are checked: what the runtime source actually serves
(`go run . --dump-schema` on the working tree) and the checked-in
runtime/schema/tools.json that the harness validates every call against.

Exit codes: 0 ok, 1 a tool outside the allowed set, 2 setup error.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / "runtime"
SCHEMA = RUNTIME / "schema" / "tools.json"

# In the order tools.All() serves them.
ALLOWED = ("shell", "text_editor", "grep_files", "web_fetch", "web_search")


def names(declaration: dict) -> list[str]:
    return [tool["name"] for tool in declaration["tools"]]


def served() -> list[str]:
    if shutil.which("go") is None:
        print("runtime_tools: go not found; it is needed to ask the runtime what it serves",
              file=sys.stderr)
        raise SystemExit(2)
    proc = subprocess.run(["go", "run", ".", "--dump-schema"], cwd=RUNTIME,
                          capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"runtime_tools: `go run . --dump-schema` failed:\n{proc.stderr}", file=sys.stderr)
        raise SystemExit(2)
    return names(json.loads(proc.stdout))


def main() -> int:
    declarations = {
        "the runtime (--dump-schema)": served(),
        str(SCHEMA.relative_to(REPO)): names(json.loads(SCHEMA.read_text(encoding="utf-8"))),
    }
    violations = [(where, extra) for where, tools in declarations.items()
                  if (extra := [t for t in tools if t not in ALLOWED])]
    for where, extra in violations:
        print(f"runtime_tools: {where} declares tools outside the allowed set: "
              f"{', '.join(extra)}", file=sys.stderr)
    if violations:
        print(f"  allowed: {', '.join(ALLOWED)}\n"
              f"  a new runtime tool must be added to ALLOWED in "
              f"{Path(__file__).resolve().relative_to(REPO)} and documented in README.md "
              f"(runtime toolset support)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

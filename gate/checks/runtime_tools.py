#!/usr/bin/env python3
"""The sandbox runtime declares no tool outside the allowed set.

The runtime's tools are the agent's whole action space, so a new tool changes what every run
can do. The commit that adds a tool to the runtime must also add it to ALLOWED, and to the
README section "Runtime toolset support".

The check reads two declarations. The first is what the runtime source serves, from
`go run . --dump-schema` on the working tree. The second is the checked-in
runtime/schema/tools.json, which the worker checks each tool call against.

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

# In the order that tools.All() in runtime/tools/ serves them.
ALLOWED = ("shell", "text_editor", "grep_files", "web_fetch", "web_search")


def names(declaration: dict) -> list[str]:
    """The tool names in a declaration in the format that ``--dump-schema`` prints."""
    return [tool["name"] for tool in declaration["tools"]]


def served() -> list[str]:
    """The tool names that the runtime source serves. Exits 2 if ``go`` is missing or the dump fails."""
    if shutil.which("go") is None:
        print("runtime_tools: go not found; it is needed to ask the runtime what it serves", file=sys.stderr)
        raise SystemExit(2)
    proc = subprocess.run(["go", "run", ".", "--dump-schema"], cwd=RUNTIME, capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"runtime_tools: `go run . --dump-schema` failed:\n{proc.stderr}", file=sys.stderr)
        raise SystemExit(2)
    return names(json.loads(proc.stdout))


def main() -> int:
    """Print each declared tool outside ALLOWED. Return 1 if there is one, and 0 if not."""
    declarations = {
        "the runtime (--dump-schema)": served(),
        str(SCHEMA.relative_to(REPO)): names(json.loads(SCHEMA.read_text(encoding="utf-8"))),
    }
    violations = [
        (where, extra)
        for where, tools in declarations.items()
        if (extra := [t for t in tools if t not in ALLOWED])
    ]
    for where, extra in violations:
        print(
            f"runtime_tools: {where} declares tools outside the allowed set: {', '.join(extra)}",
            file=sys.stderr,
        )
    if violations:
        print(
            f"  allowed: {', '.join(ALLOWED)}\n"
            f"  a new runtime tool must be added to ALLOWED in "
            f"{Path(__file__).resolve().relative_to(REPO)} and documented in runtime/README.md "
            f"(runtime toolset support)",
            file=sys.stderr,
        )
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

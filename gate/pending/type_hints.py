#!/usr/bin/env python3
"""Every function in src/ is fully typed, and mypy --strict agrees with the types.

The rules are pyproject.toml's [tool.mypy]: strict, so no parameter or return goes
unannotated, and warn_unreachable, so a None check on a value whose type cannot be
None is reported as dead code instead of kept as a guard.

Exit codes: 0 ok, 1 a type error, 2 mypy not installed.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    if importlib.util.find_spec("mypy") is None:
        print(f"type_hints: mypy is not installed for {sys.executable}; "
              f"pip install -e '.[dev]'", file=sys.stderr)
        return 2
    return subprocess.run([sys.executable, "-m", "mypy"], cwd=REPO).returncode


if __name__ == "__main__":
    sys.exit(main())

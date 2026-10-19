#!/usr/bin/env python3
"""Fail when mypy reports anything under the [tool.mypy] settings in pyproject.toml.

The settings check src/ in strict mode, so each parameter and each return needs an
annotation. They also set warn_unreachable, so mypy reports a None check on a value whose
type cannot be None as unreachable code.

Exit codes: 0 ok, 1 a type error, 2 mypy is not installed or cannot run.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def main() -> int:
    if importlib.util.find_spec("mypy") is None:
        print(
            f"type_hints: mypy is not installed for {sys.executable}; pip install -e '.[dev]'",
            file=sys.stderr,
        )
        return 2
    return subprocess.run([sys.executable, "-m", "mypy"], cwd=REPO).returncode


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Each tracked markdown document has at most MAX_LINES lines.

A longer document says more than its reader needs. Cut it, and draw complex logic as a figure.

Exit codes: 0 ok, 1 a document too long.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
MAX_LINES = 800


def main() -> int:
    """Print each document over MAX_LINES lines. Return 1 if there is one, and 0 if not."""
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", "*.md"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.split("\0")
    problems = []
    for name in filter(None, tracked):
        path = REPO / name
        if path.is_file() and (lines := len(path.read_text(encoding="utf-8").splitlines())) > MAX_LINES:
            problems.append(f"{name}: {lines} lines (max {MAX_LINES})")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

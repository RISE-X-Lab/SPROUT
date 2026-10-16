#!/usr/bin/env python3
"""The shape of src/: shallow packages of a few short source files.

- a package is at most MAX_DEPTH directories deep, counting src/sprout as the first;
- a package holds at most MAX_FILES files, __init__.py included;
- every file is Python source: data (sql, json, yaml) lives in resources/;
- a file is at most MAX_LINES lines.

Exit codes: 0 ok, 1 a rule broken.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
SRC = "src"
MAX_DEPTH = 5
MAX_FILES = 10
MAX_LINES = 1000


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", SRC], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.split("\0")
    problems = []
    files: Counter[PurePosixPath] = Counter()
    for name in filter(None, tracked):
        path = PurePosixPath(name).relative_to(SRC)
        files[path.parent] += 1
        if path.suffix != ".py":
            problems.append(f"{name}: not Python source; data lives in resources/")
        elif (lines := len((REPO / name).read_text(encoding="utf-8").splitlines())) > MAX_LINES:
            problems.append(f"{name}: {lines} lines (max {MAX_LINES})")
    for folder, count in sorted(files.items()):
        if len(folder.parts) > MAX_DEPTH:
            problems.append(f"{SRC}/{folder}/: {len(folder.parts)} levels deep (max {MAX_DEPTH})")
        if count > MAX_FILES:
            problems.append(f"{SRC}/{folder}/: {count} files (max {MAX_FILES})")
    for problem in sorted(problems):
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

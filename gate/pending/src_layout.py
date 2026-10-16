#!/usr/bin/env python3
"""The shape of src/: flat packages of a few short source files.

- a package holds no sub-package, except ALLOWED_NESTED (one folder per agent,
  and the tool interceptors);
- a package holds at most MAX_FILES files, __init__.py included;
- every file is Python source: data (sql, json, yaml) lives outside src/;
- a file is at most MAX_LINES lines.

Exit codes: 0 ok, 1 a rule broken.
"""

from __future__ import annotations

import subprocess
import sys
from collections import Counter
from fnmatch import fnmatch
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
SRC = "src"
ALLOWED_NESTED = ("sprout/harness_agents/*", "sprout/harness_tools/interceptors")
MAX_FILES = 10
MAX_LINES = 500


def nested(folder: PurePosixPath) -> bool:
    """Deeper than src/<root>/<package>/, and not one of ALLOWED_NESTED."""
    return len(folder.parts) > 2 and not any(
        fnmatch(str(folder), p) and len(folder.parts) == p.count("/") + 1 for p in ALLOWED_NESTED
    )


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "-z", "--", SRC], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.split("\0")
    problems = []
    files = Counter()
    for name in filter(None, tracked):
        path = PurePosixPath(name).relative_to(SRC)
        files[path.parent] += 1
        if path.suffix != ".py":
            problems.append(f"{name}: not Python source; data lives outside {SRC}/")
        elif (lines := len((REPO / name).read_text(encoding="utf-8").splitlines())) > MAX_LINES:
            problems.append(f"{name}: {lines} lines (max {MAX_LINES})")
    for folder, count in sorted(files.items()):
        if nested(folder):
            problems.append(f"{SRC}/{folder}/: a package inside a package")
        if count > MAX_FILES:
            problems.append(f"{SRC}/{folder}/: {count} files (max {MAX_FILES})")
    for problem in sorted(problems):
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

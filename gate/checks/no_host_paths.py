#!/usr/bin/env python3
"""No tracked file names a path in someone's home directory.

/home/<user>/... and /Users/<user>/... are one machine's layout: copied out of an
example they point nowhere, and a fixture holding them only reads right on the host
it was recorded on. Examples, docs and fixtures use placeholders under /path/to/.

Exit codes: 0 ok, 1 a home-directory path.
"""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

# At the start of a path, so /opt/x/home/y (a directory named home) is not one.
HOME_PATH = re.compile(r"(?<![\w.-])/(?:home|Users)/[A-Za-z0-9._-]+")


def main() -> int:
    tracked = subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True, text=True
    ).stdout.split("\0")
    failed = False
    for name in filter(None, tracked):
        path = REPO / name
        if not path.is_file():
            continue
        data = path.read_bytes()
        if b"\0" in data:
            continue  # binary
        for n, line in enumerate(data.decode("utf-8", "replace").splitlines(), 1):
            for match in HOME_PATH.finditer(line):
                failed = True
                print(f"{name}:{n}: {match.group()} -- use a placeholder under /path/to/", file=sys.stderr)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

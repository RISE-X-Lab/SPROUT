#!/usr/bin/env python3
"""Lint and format checks for every language in the repository.

- Python: `ruff check` and `ruff format --check`, configured in pyproject.toml.
- Go: `gofmt -l`, `go vet` and `staticcheck` over runtime/.
- Shell: `shellcheck` over every tracked shell script.

Exit codes: 0 ok, 1 a finding, 2 a tool is missing.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
RUNTIME = REPO / "runtime"
STATICCHECK = "honnef.co/go/tools/cmd/staticcheck@2025.1.1"


def shell_scripts() -> list[str]:
    tracked = subprocess.run(["git", "ls-files", "-z"], cwd=REPO, check=True,
                             capture_output=True, text=True).stdout.split("\0")
    return [name for name in tracked if name.endswith(".sh") or name == "gate/hooks/pre-commit"]


def main() -> int:
    missing = [tool for tool, found in (("ruff", importlib.util.find_spec("ruff")),
                                        ("go", shutil.which("go")), ("gofmt", shutil.which("gofmt")),
                                        ("shellcheck", shutil.which("shellcheck"))) if not found]
    if missing:
        print(f"lint: not installed: {', '.join(missing)} (pip install -e '.[dev]')", file=sys.stderr)
        return 2
    failed = False
    for argv, cwd in (([sys.executable, "-m", "ruff", "check", "--quiet", "."], REPO),
                      ([sys.executable, "-m", "ruff", "format", "--check", "--quiet", "."], REPO),
                      (["go", "vet", "./..."], RUNTIME),
                      (["go", "run", STATICCHECK, "./..."], RUNTIME),
                      (["shellcheck", *shell_scripts()], REPO)):
        failed |= subprocess.run(argv, cwd=cwd).returncode != 0
    # gofmt -l exits 0 whatever it finds: the file names it prints are the findings.
    unformatted = subprocess.run(["gofmt", "-l", "."], cwd=RUNTIME, capture_output=True, text=True).stdout
    for name in unformatted.split():
        print(f"runtime/{name}: not gofmt-formatted", file=sys.stderr)
    return 1 if failed or unformatted else 0


if __name__ == "__main__":
    sys.exit(main())

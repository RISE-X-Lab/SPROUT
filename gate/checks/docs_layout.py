#!/usr/bin/env python3
"""Images live in docs/figures/, docs/ is markdown, and nothing there is orphaned.

- an image anywhere but docs/figures/ fails, and docs/figures/ holds only images;
- every other file under docs/ is markdown that README.md links to;
- every image in docs/figures/ is referenced by README.md or a markdown file under docs/.

Exit codes: 0 ok, 1 a rule broken.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path, PurePosixPath

REPO = Path(__file__).resolve().parents[2]
FIGURES = PurePosixPath("docs/figures")
IMAGES = {".png", ".svg", ".jpg", ".jpeg", ".gif", ".webp", ".bmp", ".ico", ".tif", ".tiff"}


def referenced(target: PurePosixPath, documents: list[PurePosixPath]) -> bool:
    """Named by any document, relative to it or to the repository root."""
    for doc in documents:
        text = (REPO / doc).read_text(encoding="utf-8")
        if str(target) in text or os.path.relpath(target, doc.parent) in text:
            return True
    return False


def main() -> int:
    tracked = [PurePosixPath(p) for p in subprocess.run(
        ["git", "ls-files", "-z"], cwd=REPO, check=True, capture_output=True,
        text=True).stdout.split("\0") if p and (REPO / p).is_file()]
    docs = [p for p in tracked if p.parts[0] == "docs"]
    markdown = [p for p in docs if p.suffix == ".md"]
    readme = [PurePosixPath("README.md")]
    problems = []
    for path in tracked:
        image = path.suffix.lower() in IMAGES
        in_figures = path.parent == FIGURES
        if image and not in_figures:
            problems.append(f"{path}: an image outside {FIGURES}/")
        elif in_figures and not image:
            problems.append(f"{path}: {FIGURES}/ holds images only")
        elif in_figures and not referenced(path, readme + markdown):
            problems.append(f"{path}: referenced by neither README.md nor docs/*.md")
        elif path in docs and not in_figures and path.suffix != ".md":
            problems.append(f"{path}: docs/ outside {FIGURES}/ holds markdown only")
        elif path in markdown and not referenced(path, readme):
            problems.append(f"{path}: README.md does not link to it")
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())

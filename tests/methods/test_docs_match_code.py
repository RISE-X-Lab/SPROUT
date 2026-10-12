"""The documented claims have to match the code.

Prose cannot be checked, but a named file can be looked for. It had drifted: the
README once advertised a `swebench/agent.py` that had become a package and then
been deleted.
"""

from __future__ import annotations

import re
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]


# --------------------------------------------------------------------------- #
#  Secrets
# --------------------------------------------------------------------------- #

def test_no_config_carries_a_literal_secret():
    """A real DeepSeek key (`sk-bfe3fe…`) sat in two marathon configs and was one
    `git push` from being public -- caught by a pre-push scan, not by review.

    Credentials come from the environment or a key file. Every tracked config
    file is scanned; the heuristics are the shapes that actually appear: a
    provider-prefixed key, or a long opaque token that is not a placeholder.
    """
    import subprocess

    placeholder = re.compile(r"^(bench-key-|test-|dummy|fake|xxx|changeme|unused|<|\$|\{)", re.I)
    assignment = re.compile(r"""(?:api_key|token|secret)["']?\s*[:=]\s*["']?([^"'\s,]+)""", re.I)
    files = subprocess.run(["git", "ls-files", "*.yaml", "*.yml", "*.json", "*.toml", "*.sh"],
                           cwd=REPO, capture_output=True, text=True, check=True).stdout.split()
    assert len(files) > 10, "the scan found nothing to scan"
    offenders = []
    for name in files:
        for number, line in enumerate((REPO / name).read_text(errors="ignore").splitlines(), 1):
            match = assignment.search(line)
            if not match or placeholder.match(match.group(1)):
                continue
            value = match.group(1)
            if value.startswith(("sk-", "sk_", "ghp_", "xoxb-", "AKIA")) or len(value) >= 32:
                offenders.append("%s:%d" % (name, number))
    assert not offenders, (
        "literal credentials in %s -- source them from the environment instead"
        % ", ".join(offenders))


# --------------------------------------------------------------------------- #
#  The top-level README
# --------------------------------------------------------------------------- #
#  Nothing guarded its project tree, and it showed: it named a file that had become
#  a package and then been deleted, and a directory two moves after it went away.

TOP_README = REPO / "README.md"


def test_top_readme_project_tree_names_files_that_exist():
    """Nested entries belong to the directory above them, so the tree is walked
    rather than grepped -- a bare `slots/` is `harness/slots/`."""
    block = re.search(r"```\n\.\n(.*?)```", TOP_README.read_text(), re.S)
    assert block, "the README has no project tree"
    parent, missing = "", []
    for line in block.group(1).splitlines():
        entry = re.search(r"^(\s*)[├└]── ([\w/.-]+)", line)
        if not entry:
            continue
        indent, name = len(entry.group(1)), entry.group(2)
        if indent == 0:
            parent = name if name.endswith("/") else ""
            path = name
        else:
            path = parent + name
        if not (REPO / path.rstrip("/")).exists():
            missing.append(path)
    assert not missing, "the README tree names %s, which do not exist" % missing

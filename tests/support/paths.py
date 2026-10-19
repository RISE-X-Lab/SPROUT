"""Paths in the repository that tests read: its root, the shared fixtures and the runtime binary."""

from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
#: Data that tests of more than one area read. Each subdirectory is named after its format.
FIXTURES = REPO / "tests" / "fixtures"
#: A Harbor task: its agent writes /app/answer.txt, and its verifier rewards "ready\n" in it.
HARBOR_SHELL_TASK = FIXTURES / "harbor" / "shell-task"
#: The runtime binary that a large test mounts into each container. It is not in git.
RUNTIME_BIN = REPO / "runtime" / "sprout-runtime"

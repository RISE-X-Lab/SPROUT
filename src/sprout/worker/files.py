"""Atomic, durable JSON receipts: a reader sees the old file or the new one, never part of one."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sprout.runstore.specs import canonical


def write_json(path: Path, value: dict[str, Any]) -> None:
    """Replace ``path`` with ``value`` as canonical JSON, and return after it is on disk.

    The call creates the parent directories.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    # The rename is durable only after the directory entry is on disk too.
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path: Path) -> dict[str, Any] | None:
    """The JSON object that ``path`` holds, or None when ``path`` does not exist."""
    try:
        value: dict[str, Any] = json.loads(path.read_text())
    except FileNotFoundError:
        return None
    return value

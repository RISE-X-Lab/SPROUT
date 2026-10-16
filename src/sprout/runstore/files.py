"""Atomic, fsynced receipts and append-only trajectory ingestion."""

from __future__ import annotations

import json
import os
import re
from pathlib import Path

from sprout.runstore.specs import canonical


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w") as stream:
        stream.write(canonical(value) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)
    directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(directory)
    finally:
        os.close(directory)


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def failure_kind(result: dict) -> str | None:
    if result.get("status") == "completed":
        return None
    message = str(result.get("error") or "").lower()
    if any(marker in message for marker in ("checksum changed", "revision differs", "profile changed")):
        return "configuration"
    if result.get("status") in {"timeout", "step_limit", "cost_limit", "budget_exhausted"}:
        return "actor"
    if any(marker in message for marker in ("budget exhausted", "cost limit", "step limit", "maximum turns")):
        return "actor"
    if re.search(r"\b(?:502|503|504)\b", message) or any(
        marker in message
        for marker in (
            "transportclosederror",
            "connection reset",
            "connection refused",
            "stream stalled",
            "execution uncertain",
            "tool execution did not settle",
            "final snapshot failed",
            "readtimeout",
        )
    ):
        return "infrastructure"
    return result.get("failure_kind") or "actor"

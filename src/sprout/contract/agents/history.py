"""Dispatch native history operations without importing either agent SDK."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from sprout.contract import mini_history
from sprout.contract.agents import opencollab


def load_prefix(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Read a checked prefix, using its recorded native format."""
    with Path(reference["path"]).open("rb") as stream:
        header = json.loads(stream.readline())
    is_oc = header.get("format") == opencollab.FORMAT
    if reference.get("slot") is not None and ((reference["slot"] == opencollab.SLOT) != is_oc):
        raise ValueError("Native prefix slot differs from its recorded format")
    if is_oc:
        return opencollab.load_prefix(reference)
    return mini_history.load_prefix(reference)


def training_messages(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The conversation represented by a native transcript, including SDK context shaping."""
    if entries and entries[0].get("format") == opencollab.FORMAT:
        return opencollab.training_messages(entries)
    return mini_history.training_messages(entries)


def materialize(
    reference: Mapping[str, Any], cwd: Path, destination: Path, native_home: Path
) -> dict[str, Any]:
    """Copy a checked native prefix into an attempt-owned directory."""
    if reference["slot"] != opencollab.SLOT:
        return mini_history.materialize(reference, cwd, destination, native_home)
    data = mini_history.read_prefix(reference)
    load_prefix(reference)
    destination.mkdir(parents=True, exist_ok=False)
    path = destination / "prefix.jsonl"
    path.write_bytes(data)
    return {
        "native_prefix": {"path": str(path.resolve()), "sha256": reference["sha256"], "slot": opencollab.SLOT}
    }


def validate_assistant_turn(
    message: object,
    *,
    history: Sequence[Mapping[str, Any]] = (),
    tools: list[dict[str, Any]] | None = None,
    slot: str = "mini-swe-agent",
    allow_reasoning: bool = False,
) -> dict[str, Any]:
    """Validate a reviewer turn against the selected actor's tool contract."""
    if slot == opencollab.SLOT:
        if tools is None:
            raise ValueError("OpenCollab continuation requires recorded tool schemas")
        return opencollab.validate_assistant_turn(
            message, history=history, tools=tools, allow_reasoning=allow_reasoning
        )
    if slot != "mini-swe-agent":
        raise ValueError("Unsupported native history slot")
    return mini_history.validate_assistant_turn(
        message, history=history, tools=tools, allow_reasoning=allow_reasoning
    )

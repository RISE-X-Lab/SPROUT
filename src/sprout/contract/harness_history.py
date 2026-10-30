"""The Sprout harness's native history: its records, and the chat messages that training reads from them.

A history is one JSON line per record, and only grows:

- ``harness.session``: the header, with ``format``, ``session_id``, the ``workspace`` and the ``tools``.
- ``harness.message``: one Chat Completions message, with the assistant's ``reasoning_content``.
- ``harness.turn``: the end of one model response, with its ``turn_id`` and ``call_ids``. A
  ``rejected`` turn broke the format and ran nothing; a ``submitted`` turn ended the rollout.

The records mirror mini-swe-agent's (``sprout.contract.mini_history``), so the same prefix references
name a prefix of either history.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sprout.contract.mini_history import read_entries
from sprout.contract.mini_history import training_messages as _messages

#: The slot that writes this history, as ``RunSpec.slot`` and a prefix reference name it.
SLOT = "sprout-harness"
#: The ``format`` of a history's ``harness.session`` record.
FORMAT = "sprout-harness-v1"
SESSION, MESSAGE, TURN = "harness.session", "harness.message", "harness.turn"

#: The record that holds a chat message in the native history of each slot.
MESSAGE_RECORDS = {"mini-swe-agent": "mini.message", SLOT: MESSAGE}


def training_messages(entries: list[dict[str, Any]], *, slot: str = SLOT) -> list[dict[str, Any]]:
    """The chat messages of a native history of ``slot``, with only the keys that training reads.

    Raises ``ValueError`` for a slot with no native history, or a message whose content is not text."""
    if slot not in MESSAGE_RECORDS:
        raise ValueError("Unsupported native history slot %r" % slot)
    return _messages(entries, record=MESSAGE_RECORDS[slot])


def load_prefix(reference: Mapping[str, Any]) -> list[dict[str, Any]]:
    """The entries of the harness history prefix that ``reference`` names, checked for a branch to restore.

    Without ``byte_length`` the prefix is the whole file. Raises ``ValueError`` if the checksum differs,
    the format is not ``FORMAT``, or the prefix does not end at a turn that ran its calls."""
    with Path(reference["path"]).open("rb") as stream:
        data = stream.read(reference.get("byte_length", -1))
    if not data.endswith(b"\n") or hashlib.sha256(data).hexdigest() != reference["sha256"]:
        raise ValueError("harness native prefix checksum changed")
    entries = read_entries(data)
    header = next((entry for entry in entries if entry.get("type") == SESSION), {})
    if header.get("format") != FORMAT:
        raise ValueError("Unsupported harness native history format")
    last = entries[-1] if entries else {}
    if last.get("type") != TURN or last.get("rejected") or last.get("submitted"):
        raise ValueError("harness restoration requires a turn that ran its calls")
    return entries

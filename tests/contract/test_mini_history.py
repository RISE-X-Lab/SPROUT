"""mini's native history: its entries, its training messages, and the prefixes that a branch restores."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from sprout.contract.mini_history import (
    load_prefix,
    materialize,
    read_entries,
    read_prefix,
    training_messages,
    valid_prefix,
)
from tests.contract.builders import bash_call

SESSION = {
    "type": "mini.session",
    "format": "sprout-mini-v1",
    "mini_version": "2.4.6",
    "session_id": "session-1",
    "workspace": {"cwd": "/testbed", "repository_dir": None},
}
TASK = {"type": "mini.message", "message": {"role": "user", "content": "Fix the bug."}}
CALL = {
    "type": "mini.message",
    "message": {"role": "assistant", "content": "", "tool_calls": [bash_call("ls", "c1")]},
}
RESULT = {"type": "mini.message", "message": {"role": "tool", "tool_call_id": "c1", "content": "README\n"}}
TURN = {"type": "mini.turn", "turn_id": "t1", "call_ids": ["c1"], "source": "actor"}


def jsonl(*entries: dict[str, Any]) -> bytes:
    return b"".join(json.dumps(entry).encode() + b"\n" for entry in entries)


def write_history(path: Path, data: bytes) -> Path:
    path.write_bytes(data)
    return path


def reference(file: Path, data: bytes, **fields: Any) -> dict[str, Any]:
    """The reference of the first ``len(data)`` bytes of ``file``, with ``fields`` added or replaced."""
    return {"path": str(file), "byte_length": len(data), "sha256": hashlib.sha256(data).hexdigest(), **fields}


@pytest.mark.parametrize(
    "data, expected",
    [
        pytest.param(b'{"a": 1}\n\n  \n{"b": 2}\n', [{"a": 1}, {"b": 2}], id="blank-lines"),
        pytest.param(b'{"a": 1}\n{"b": ', [{"a": 1}], id="partial-last-line"),
        pytest.param(
            '{"text": "a\u2028b\u0085c"}\n'.encode(), [{"text": "a\u2028b\u0085c"}], id="unicode-line-breaks"
        ),
    ],
)
def test_each_complete_line_is_one_entry(data, expected):
    """A history that mini still writes can end in a partial line. A record keeps a tool's U+2028 verbatim."""
    assert read_entries(data) == expected


def test_the_training_messages_keep_only_the_chat_fields_of_each_message():
    entries = [
        SESSION,
        {
            "type": "mini.message",
            "message": {"role": "system", "content": "You are mini.", "extra": {"x": 1}},
        },
        TASK,
        {
            "type": "mini.message",
            "message": {
                "role": "assistant",
                "content": "Listing.",
                "reasoning_content": "Look first.",
                "tool_calls": [
                    {**bash_call("ls", "c1"), "index": 0, "function": {**bash_call("ls")["function"], "x": 1}}
                ],
                "extra": {"actions": [{"command": "ls"}]},
            },
        },
        {
            "type": "mini.message",
            "message": {"role": "tool", "tool_call_id": "c1", "content": "README\n", "extra": {}},
        },
        TURN,
        {
            "type": "mini.message",
            "message": {"role": "exit", "content": "", "extra": {"exit_status": "Submitted"}},
        },
    ]

    messages = training_messages(entries)

    assert messages == [
        {"role": "system", "content": "You are mini."},
        {"role": "user", "content": "Fix the bug."},
        {
            "role": "assistant",
            "content": "Listing.",
            "reasoning_content": "Look first.",
            "tool_calls": [bash_call("ls", "c1")],
        },
        {"role": "tool", "content": "README\n", "tool_call_id": "c1"},
    ]


def test_a_message_without_content_is_trained_on_empty_content():
    entries = [
        {
            "type": "mini.message",
            "message": {"role": "assistant", "content": None, "tool_calls": [bash_call()]},
        }
    ]

    messages = training_messages(entries)

    assert messages == [{"role": "assistant", "content": "", "tool_calls": [bash_call()]}]


def test_a_message_whose_content_is_not_text_has_no_training_messages():
    entries = [{"type": "mini.message", "message": {"role": "user", "content": [{"type": "image_url"}]}}]

    with pytest.raises(ValueError, match="^mini training requires text content$"):
        training_messages(entries)


def test_a_branch_restores_only_the_entries_of_its_prefix(tmp_path):
    prefix = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "native.jsonl", prefix + jsonl(CALL, RESULT))

    entries = load_prefix(reference(path, prefix))

    assert entries == [SESSION, TASK, CALL, RESULT, TURN]


def test_a_reference_without_a_length_names_the_whole_file(tmp_path):
    data = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "native.jsonl", data)

    entries = load_prefix({"path": str(path), "sha256": hashlib.sha256(data).hexdigest()})

    assert entries == [SESSION, TASK, CALL, RESULT, TURN]


def test_a_history_in_the_older_format_name_is_restored(tmp_path):
    """``ash-mini-v1`` is the name of the same format in histories that runs wrote before the rename."""
    data = jsonl({**SESSION, "format": "ash-mini-v1"}, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "native.jsonl", data)

    entries = load_prefix(reference(path, data))

    assert entries[0]["format"] == "ash-mini-v1"


@pytest.mark.parametrize(
    "data, cut, message",
    [
        pytest.param(
            jsonl(SESSION, TASK, CALL, RESULT, TURN), -1, "mini native prefix checksum changed", id="mid-line"
        ),
        pytest.param(
            jsonl(SESSION, TASK, CALL, RESULT),
            None,
            "mini restoration requires a closed model turn",
            id="open-turn",
        ),
        pytest.param(
            jsonl({**SESSION, "format": "other-v2"}, TASK, CALL, RESULT, TURN),
            None,
            "Unsupported mini native history version",
            id="other-format",
        ),
        pytest.param(
            jsonl({**SESSION, "mini_version": "2.5.0"}, TASK, CALL, RESULT, TURN),
            None,
            "Unsupported mini native history version",
            id="other-mini",
        ),
        pytest.param(
            jsonl(TASK, CALL, RESULT, TURN),
            None,
            "Unsupported mini native history version",
            id="no-session-header",
        ),
    ],
)
def test_a_prefix_that_a_branch_cannot_restore_is_refused(tmp_path, data, cut, message):
    """``cut`` drops bytes from the end of the referenced prefix."""
    prefix = data[:cut]
    path = write_history(tmp_path / "native.jsonl", data)

    with pytest.raises(ValueError, match="^%s$" % re.escape(message)):
        load_prefix(reference(path, prefix))


def test_a_prefix_whose_bytes_changed_is_refused(tmp_path):
    data = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "native.jsonl", data.replace(b"README", b"readme"))

    with pytest.raises(ValueError, match="^mini native prefix checksum changed$"):
        load_prefix(reference(path, data))


def test_the_bytes_of_a_prefix_are_the_first_byte_length_bytes_of_its_file(tmp_path):
    prefix = jsonl(SESSION, TASK)
    path = write_history(tmp_path / "native.jsonl", prefix + jsonl(CALL))

    data = read_prefix(reference(path, prefix))

    assert data == prefix


@pytest.mark.parametrize("length", [0, True, "12", 1.5])
def test_a_prefix_length_that_is_not_a_positive_integer_is_refused(tmp_path, length):
    prefix = jsonl(SESSION, TASK)
    path = write_history(tmp_path / "native.jsonl", prefix)

    with pytest.raises(ValueError, match="^Invalid native prefix length$"):
        read_prefix(reference(path, prefix, byte_length=length))


@pytest.mark.parametrize(
    "file_data, length_change, message",
    [
        pytest.param(jsonl(SESSION), 0, "Native prefix missing or truncated", id="file-shorter"),
        pytest.param(jsonl(SESSION, TASK), -1, "Native prefix missing or truncated", id="mid-line"),
        pytest.param(
            jsonl(SESSION, TASK).replace(b"Fix", b"Fox"), 0, "Native prefix checksum changed", id="changed"
        ),
    ],
)
def test_a_prefix_that_its_file_no_longer_holds_is_refused(tmp_path, file_data, length_change, message):
    prefix = jsonl(SESSION, TASK)
    path = write_history(tmp_path / "native.jsonl", file_data)
    named = reference(path, prefix)

    with pytest.raises(ValueError, match="^%s$" % re.escape(message)):
        read_prefix({**named, "byte_length": named["byte_length"] + length_change})


def test_a_prefix_of_a_missing_file_cannot_be_read(tmp_path):
    prefix = jsonl(SESSION, TASK)

    with pytest.raises(FileNotFoundError):
        read_prefix(reference(tmp_path / "absent.jsonl", prefix))


def test_a_readable_prefix_is_valid(tmp_path):
    prefix = jsonl(SESSION, TASK)
    path = write_history(tmp_path / "native.jsonl", prefix)

    assert valid_prefix(reference(path, prefix)) is True


@pytest.mark.parametrize(
    "fields",
    [
        pytest.param({"path": "/path/to/absent.jsonl"}, id="missing-file"),
        pytest.param({"sha256": hashlib.sha256(b"other\n").hexdigest()}, id="changed-bytes"),
        pytest.param({"path": None}, id="no-path"),
    ],
)
def test_a_prefix_that_cannot_be_read_is_invalid_and_raises_nothing(tmp_path, fields):
    prefix = jsonl(SESSION, TASK)
    path = write_history(tmp_path / "native.jsonl", prefix)

    assert valid_prefix(reference(path, prefix, **fields)) is False


def test_a_reference_without_a_checksum_is_invalid(tmp_path):
    prefix = jsonl(SESSION, TASK)
    path = write_history(tmp_path / "native.jsonl", prefix)

    assert valid_prefix({"path": str(path), "byte_length": len(prefix)}) is False


def test_materialize_copies_the_prefix_for_the_branch_and_names_the_copy(tmp_path):
    prefix = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "parent.jsonl", prefix + jsonl(CALL))
    destination = tmp_path / "attempt" / "restoration"

    extra = materialize(
        reference(path, prefix, slot="mini-swe-agent"),
        tmp_path / "cwd",
        destination,
        tmp_path / "native-home",
    )

    assert (extra, (destination / "prefix.jsonl").read_bytes()) == (
        {
            "native_prefix": {
                "path": str(destination / "prefix.jsonl"),
                "sha256": hashlib.sha256(prefix).hexdigest(),
            }
        },
        prefix,
    )


def test_the_copy_of_a_prefix_restores_the_entries_of_the_prefix(tmp_path):
    prefix = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "parent.jsonl", prefix + jsonl(CALL))
    extra = materialize(
        reference(path, prefix, slot="mini-swe-agent"),
        tmp_path / "cwd",
        tmp_path / "restoration",
        tmp_path / "native-home",
    )

    entries = load_prefix(extra["native_prefix"])

    assert entries == [SESSION, TASK, CALL, RESULT, TURN]


def test_a_prefix_is_not_copied_over_an_existing_destination(tmp_path):
    prefix = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "parent.jsonl", prefix)
    (tmp_path / "restoration").mkdir()

    with pytest.raises(FileExistsError):
        materialize(
            reference(path, prefix, slot="mini-swe-agent"),
            tmp_path / "cwd",
            tmp_path / "restoration",
            tmp_path / "native-home",
        )


def test_a_prefix_of_another_harness_is_not_materialized(tmp_path):
    prefix = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "parent.jsonl", prefix)

    with pytest.raises(ValueError, match="^Unsupported native history slot$"):
        materialize(
            reference(path, prefix, slot="codex"),
            tmp_path / "cwd",
            tmp_path / "restoration",
            tmp_path / "native-home",
        )


def test_a_changed_prefix_is_not_materialized(tmp_path):
    prefix = jsonl(SESSION, TASK, CALL, RESULT, TURN)
    path = write_history(tmp_path / "parent.jsonl", prefix.replace(b"README", b"readme"))

    with pytest.raises(ValueError, match="^Native prefix checksum changed$"):
        materialize(
            reference(path, prefix, slot="mini-swe-agent"),
            tmp_path / "cwd",
            tmp_path / "restoration",
            tmp_path / "native-home",
        )

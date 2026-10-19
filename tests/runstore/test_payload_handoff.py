"""The attempt payload: which payloads a worker may freeze, and its handoff to the child through a pipe.

The child reads its payload once, on stdin, and refuses a payload that is missing, truncated, too large,
late, or not the one that the worker froze.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Iterator
from typing import Any, BinaryIO

import pytest

from sprout.runstore.payload import (
    MAX_PAYLOAD_BYTES,
    PayloadUnavailable,
    checked_payload,
    encode_payload,
    receive_payload,
    send_payload,
)
from sprout.runstore.specs import canonical, digest


def payload(**changes: Any) -> dict[str, Any]:
    """A valid version-1 payload of the attempt ``attempt-1`` of ``job-1``, with ``changes`` over it."""
    return {
        "version": 1,
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "kind": "rollout",
        "effective_spec": {"prompt": "Write the answer."},
        "profile_config": {},
        "recovery": None,
        **changes,
    }


def payload_of_size(size: int) -> dict[str, Any]:
    """A valid payload whose canonical UTF-8 form has exactly ``size`` bytes, most of them non-ASCII."""
    base = len(canonical(payload(effective_spec={"prompt": ""})).encode())
    wide, narrow = divmod(size - base, 3)
    return payload(effective_spec={"prompt": "测" * wide + "x" * narrow})


@pytest.fixture
def pipe() -> Iterator[tuple[int, BinaryIO]]:
    """The read descriptor and the write stream of a new pipe. Both are closed after the test."""
    reader, writer = os.pipe()
    stream = os.fdopen(writer, "wb", buffering=0)
    yield reader, stream
    os.close(reader)
    stream.close()


def send_in_background(stream: BinaryIO, data: bytes) -> threading.Thread:
    """A thread that sends ``data`` to ``stream``, for a reader in the test's own thread."""
    thread = threading.Thread(target=send_payload, args=(stream, data), kwargs={"timeout_s": 10})
    thread.start()
    return thread


# Freezing a payload


@pytest.mark.parametrize(
    "changes",
    [{}, {"kind": "grade"}, {"kind": "review"}, {"recovery": {"point_id": "point-1"}}],
    ids=["rollout", "grade", "review", "with-a-recovery-point"],
)
def test_a_valid_payload_encodes_to_its_canonical_utf8_json(changes):
    assert encode_payload(payload(**changes), "job-1", "attempt-1") == canonical(payload(**changes)).encode()


def test_a_payload_keeps_an_env_reference_as_the_reference():
    """A credential reaches the child as a variable name. Its value never enters the database."""
    referenced = payload(profile_config={"worker_env": {"MODEL_API_KEY": {"$env": "SPROUT_MODEL_API_KEY"}}})

    assert json.loads(encode_payload(referenced, "job-1", "attempt-1")) == referenced


@pytest.mark.parametrize(
    "value,error",
    [
        (payload(version=2), "Unsupported attempt payload version"),
        ([payload()], "Unsupported attempt payload version"),
        (payload(job_id="job-2"), "Attempt payload identity mismatch"),
        (payload(attempt_id="attempt-2"), "Attempt payload identity mismatch"),
        (payload(kind="train"), "Invalid attempt payload kind"),
        (payload(effective_spec=None), "Attempt payload requires effective spec and profile"),
        (payload(profile_config=[]), "Attempt payload requires effective spec and profile"),
        (payload(recovery="point-1"), "Invalid attempt recovery metadata"),
        (payload(profile_config={"worker_env": {"MODEL_API_KEY": "sk-live"}}), "Use a worker-side $env"),
    ],
    ids=[
        "version-2",
        "not-an-object",
        "another-job",
        "another-attempt",
        "unknown-kind",
        "no-effective-spec",
        "profile-not-an-object",
        "recovery-not-an-object",
        "inline-credential",
    ],
)
def test_an_invalid_payload_is_refused(value, error):
    with pytest.raises(ValueError, match=error.replace("$", r"\$")):
        encode_payload(value, "job-1", "attempt-1")


def test_a_payload_of_exactly_the_size_limit_is_accepted():
    assert len(encode_payload(payload_of_size(MAX_PAYLOAD_BYTES), "job-1", "attempt-1")) == MAX_PAYLOAD_BYTES


def test_a_payload_one_byte_over_the_size_limit_is_refused():
    with pytest.raises(ValueError, match="Attempt payload exceeds 16 MiB"):
        encode_payload(payload_of_size(MAX_PAYLOAD_BYTES + 1), "job-1", "attempt-1")


# Checking a stored payload


def test_a_stored_payload_with_its_digest_passes_the_check():
    assert checked_payload(payload(), digest(payload()), "job-1", "attempt-1") == payload()


@pytest.mark.parametrize(
    "value,expected_hash,error",
    [
        (None, None, "Attempt has no frozen DB payload"),
        (payload(), digest(payload(kind="grade")), "Attempt payload checksum mismatch"),
        (payload(), "", "Attempt payload checksum mismatch"),
        (payload(job_id="job-2"), digest(payload(job_id="job-2")), "Attempt payload identity mismatch"),
    ],
    ids=["absent", "another-digest", "no-digest", "another-job"],
)
def test_a_stored_payload_that_fails_a_check_is_unavailable(value, expected_hash, error):
    with pytest.raises(PayloadUnavailable, match=error):
        checked_payload(value, expected_hash, "job-1", "attempt-1")


# The handoff through a pipe


def test_a_payload_at_the_size_limit_crosses_the_pipe_unchanged(pipe):
    """The payload is larger than a pipe buffer, so the handoff must not wait for the reader to finish."""
    reader, stream = pipe
    large = payload_of_size(MAX_PAYLOAD_BYTES)
    sender = send_in_background(stream, encode_payload(large, "job-1", "attempt-1"))

    received = receive_payload(reader, digest(large), "job-1", "attempt-1", timeout_s=10)
    sender.join()

    assert received == large


@pytest.mark.parametrize(
    "data,error",
    [
        (b"", json.JSONDecodeError),
        (b'{"version":', json.JSONDecodeError),
        (b"\xff", UnicodeDecodeError),
        (b"null", PayloadUnavailable),
        (b"[]", PayloadUnavailable),
        (b"{}", PayloadUnavailable),
    ],
    ids=["empty", "truncated", "not-utf8", "null", "not-an-object", "empty-object"],
)
def test_the_child_refuses_input_that_is_not_a_payload(pipe, data, error):
    reader, stream = pipe
    stream.write(data)
    stream.close()

    with pytest.raises(error):
        receive_payload(reader, digest(payload()), "job-1", "attempt-1", timeout_s=10)


def test_the_child_refuses_input_over_the_size_limit_before_it_parses_it(pipe):
    reader, stream = pipe
    sender = send_in_background(stream, b" " * (MAX_PAYLOAD_BYTES + 1))

    with pytest.raises(ValueError, match="Attempt payload exceeds 16 MiB"):
        receive_payload(reader, digest(payload()), "job-1", "attempt-1", timeout_s=10)
    sender.join()


def test_the_child_stops_waiting_for_a_payload_that_does_not_arrive_by_its_deadline(pipe):
    reader, _ = pipe

    with pytest.raises(TimeoutError, match="Attempt payload was not delivered before startup deadline"):
        receive_payload(reader, digest(payload()), "job-1", "attempt-1", timeout_s=0.05)


def test_the_worker_stops_sending_to_a_child_that_does_not_read_and_closes_the_pipe(pipe):
    _, stream = pipe

    with pytest.raises(TimeoutError, match="Attempt payload handoff timed out"):
        send_payload(stream, b"x" * (1024 * 1024), timeout_s=0.05)

    assert stream.closed

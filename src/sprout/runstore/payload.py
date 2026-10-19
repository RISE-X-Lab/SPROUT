"""The attempt payload: what a worker freezes for one attempt and hands its child process once, on stdin.

``encode_payload`` and ``checked_payload`` check a payload's version, identity, kind and checksum.
``send_payload`` and ``receive_payload`` move it through a pipe before a deadline, at most
``MAX_PAYLOAD_BYTES``. docs/RUNSTORE.md ("Failure handling") describes the frozen payload.
"""

from __future__ import annotations

import json
import os
import selectors
import time
from typing import Any, BinaryIO

from sprout.runstore.specs import canonical, digest, no_credentials

MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


class PayloadUnavailable(ValueError):
    """An attempt has no frozen payload, or its payload fails a check."""


def encode_payload(payload: dict[str, Any], job_id: str, attempt_id: str) -> bytes:
    """The canonical UTF-8 JSON of ``payload``.

    Raises ValueError unless it is a version-1 payload of ``job_id`` and ``attempt_id``, with a known kind,
    an effective spec and a profile, no inline credential, and at most ``MAX_PAYLOAD_BYTES``.
    """
    if not isinstance(payload, dict) or payload.get("version") != 1:
        raise ValueError("Unsupported attempt payload version")
    if payload.get("job_id") != job_id or payload.get("attempt_id") != attempt_id:
        raise ValueError("Attempt payload identity mismatch")
    if payload.get("kind") not in {"rollout", "grade", "review"}:
        raise ValueError("Invalid attempt payload kind")
    if not all(isinstance(payload.get(key), dict) for key in ("effective_spec", "profile_config")):
        raise ValueError("Attempt payload requires effective spec and profile")
    if payload.get("recovery") is not None and not isinstance(payload["recovery"], dict):
        raise ValueError("Invalid attempt recovery metadata")
    no_credentials(payload)
    data = canonical(payload).encode("utf-8")
    if len(data) > MAX_PAYLOAD_BYTES:
        raise ValueError("Attempt payload exceeds 16 MiB")
    return data


def checked_payload(
    payload: dict[str, Any] | None, expected_hash: str | None, job_id: str, attempt_id: str
) -> dict[str, Any]:
    """``payload``, if it passes ``encode_payload`` and its digest is ``expected_hash``.

    Raises PayloadUnavailable if ``payload`` is None or fails a check.
    """
    if payload is None:
        raise PayloadUnavailable("Attempt has no frozen DB payload")
    try:
        encode_payload(payload, job_id, attempt_id)
        if not expected_hash or digest(payload) != expected_hash:
            raise ValueError("Attempt payload checksum mismatch")
    except (ValueError, TypeError) as error:
        raise PayloadUnavailable(str(error)) from error
    return payload


def send_payload(stream: BinaryIO, data: bytes, *, timeout_s: float) -> None:
    """Write ``data`` to the pipe ``stream`` within ``timeout_s`` seconds, then close the stream.

    The stream is closed in every case. Raises TimeoutError if the reader does not take ``data`` in time.
    """
    deadline = time.monotonic() + timeout_s
    try:
        descriptor = stream.fileno()
        # A blocking write to a full pipe would wait past the deadline for a child that never reads.
        os.set_blocking(descriptor, False)
        with selectors.DefaultSelector() as selector:
            selector.register(descriptor, selectors.EVENT_WRITE)
            pending = memoryview(data)
            while pending:
                remaining = deadline - time.monotonic()
                if remaining <= 0 or not selector.select(remaining):
                    raise TimeoutError("Attempt payload handoff timed out")
                try:
                    written = os.write(descriptor, pending[:65536])
                except BlockingIOError:
                    continue
                pending = pending[written:]
    finally:
        stream.close()


def receive_payload(
    descriptor: int, expected_hash: str, job_id: str, attempt_id: str, *, timeout_s: float = 60
) -> dict[str, Any]:
    """Read the payload from ``descriptor`` up to its end, and return it as ``checked_payload`` does.

    Raises TimeoutError if the end does not come within ``timeout_s`` seconds, ValueError if the data
    exceeds ``MAX_PAYLOAD_BYTES`` or is not JSON, and PayloadUnavailable if a check fails.
    """
    deadline = time.monotonic() + timeout_s
    data = bytearray()
    with selectors.DefaultSelector() as selector:
        selector.register(descriptor, selectors.EVENT_READ)
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or not selector.select(remaining):
                raise TimeoutError("Attempt payload was not delivered before startup deadline")
            chunk = os.read(descriptor, min(65536, MAX_PAYLOAD_BYTES + 1 - len(data)))
            if not chunk:
                break
            data.extend(chunk)
            if len(data) > MAX_PAYLOAD_BYTES:
                raise ValueError("Attempt payload exceeds 16 MiB")
    return checked_payload(json.loads(data), expected_hash, job_id, attempt_id)

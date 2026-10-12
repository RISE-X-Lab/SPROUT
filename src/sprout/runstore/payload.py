"""Frozen attempt metadata and bounded, one-shot stdin transport."""

from __future__ import annotations

import json
import os
import selectors
import time
from typing import BinaryIO

from sprout.runstore.specs import canonical, digest, no_credentials


MAX_PAYLOAD_BYTES = 16 * 1024 * 1024


class PayloadUnavailable(ValueError):
    pass


def encode_payload(payload: dict, job_id: str, attempt_id: str) -> bytes:
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


def checked_payload(payload: dict | None, expected_hash: str | None,
                    job_id: str, attempt_id: str) -> dict:
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
    deadline = time.monotonic() + timeout_s
    try:
        descriptor = stream.fileno()
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


def receive_payload(descriptor: int, expected_hash: str, job_id: str, attempt_id: str,
                    *, timeout_s: float = 60) -> dict:
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

"""The sampling parameters a rollout request may carry, and their validation."""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any
from urllib.parse import urlsplit


def endpoint(value: str) -> str:
    """The API prefix of the URL ``value``, to which a caller appends ``/v1/chat/completions``.

    A trailing ``/`` or ``/v1`` is dropped. Raises ValueError unless ``value`` is an HTTP(S) URL
    with a host, and without credentials, a query or a fragment.
    """
    parts = urlsplit(value)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or parts.username
        or parts.password
        or parts.query
        or parts.fragment
    ):
        raise ValueError("Rollout endpoint must be HTTP(S), without credentials/query/fragment")
    value = value.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def validate(value: object) -> None:
    """Raise ValueError unless ``value`` is a dict of known sampling controls, each with a valid value.

    The output-token limits, when more than one is given, must be equal.
    """
    allowed = {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "max_new_tokens",
        "max_output_tokens",
        "stop",
        "reasoning_effort",
    }
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown message sampling controls")
    for key in ("temperature", "top_p"):
        if key in value:
            number = value[key]
            if (
                type(number) not in (int, float)
                or not math.isfinite(number)
                or number < 0
                or (key == "top_p" and not 0 < number <= 1)
            ):
                raise ValueError(f"Invalid {key}")
    if "top_k" in value and (type(value["top_k"]) is not int or value["top_k"] < -1):
        raise ValueError("top_k must be -1 or a nonnegative integer")
    if "reasoning_effort" in value and value["reasoning_effort"] not in {
        "none",
        "minimal",
        "low",
        "medium",
        "high",
        "xhigh",
        "max",
        "ultra",
    }:
        raise ValueError("Invalid reasoning_effort")
    limits = [value[k] for k in ("max_tokens", "max_new_tokens", "max_output_tokens") if k in value]
    if limits and any(type(n) is not int or n <= 0 or n != limits[0] for n in limits):
        raise ValueError("Output token limits must be positive and agree")
    stop = value.get("stop")
    if stop is not None and not isinstance(stop, str):
        if not isinstance(stop, list) or any(not isinstance(s, str) for s in stop):
            raise ValueError("stop must be text, a text list, or null")


def parameters(value: dict[str, Any]) -> dict[str, Any]:
    """The validated controls as Chat Completions request fields, with one ``max_tokens`` for the limits.

    Raises ValueError as ``validate`` does. ``value`` itself is not changed.
    """
    validate(value)
    result = deepcopy(value)
    limits = [result.pop(k) for k in ("max_tokens", "max_new_tokens", "max_output_tokens") if k in result]
    if limits:
        result["max_tokens"] = limits[0]
    return result

"""The one credential that a model endpoint takes: a bearer key named by an environment variable.

Each endpoint that Sprout calls is an OpenAI-compatible Chat Completions URL behind a
bearer key. The key is the provider's own, or the master key of a LiteLLM proxy for a
provider without such an endpoint (``docs/MODEL_ENDPOINTS.md``). Nothing here reads
the URL: there is no provider to recognise and no other credential to fall back on.
"""

from __future__ import annotations

import os


def bearer_key(key_env: str | None) -> str:
    """The value of the environment variable that ``key_env`` names.

    Raises ValueError if ``key_env`` is not a variable name, or if the variable is
    unset or empty. A caller never sends an anonymous request in place of a key.
    """
    if not isinstance(key_env, str) or not key_env.isidentifier():
        raise ValueError("Model key must name an environment variable")
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(f"Model key environment variable {key_env} is unset")
    return key


def bearer_headers(key: str | None) -> dict[str, str]:
    """The ``Authorization`` header for ``key``, or no header when ``key`` is empty or None."""
    return {"Authorization": "Bearer " + key} if key else {}

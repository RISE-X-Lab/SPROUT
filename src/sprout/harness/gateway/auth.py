"""The one credential a model endpoint takes: a bearer key named by an environment variable.

Every endpoint Sprout talks to is an OpenAI-compatible Chat Completions URL
behind a bearer key -- a provider's own, or a LiteLLM proxy's master key for the
providers that are not (``docs/MODEL_ENDPOINTS.md``). Nothing here looks at the
URL: there is no provider to recognise and no credential chain to fall back on.
"""

from __future__ import annotations

import os


def bearer_key(key_env: str) -> str:
    """The key in ``key_env``, or a ValueError before anything that would spend it.

    A misspelled or unset variable is an error, never an anonymous request: the
    check costs nothing and runs during configuration, before a trial starts.
    """
    if not isinstance(key_env, str) or not key_env.isidentifier():
        raise ValueError("Model key must name an environment variable")
    key = os.environ.get(key_env)
    if not key:
        raise ValueError(f"Model key environment variable {key_env} is unset")
    return key


def bearer_headers(key: str | None) -> dict[str, str]:
    return {"Authorization": "Bearer " + key} if key else {}

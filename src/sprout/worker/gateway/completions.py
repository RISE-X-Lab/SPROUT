"""The failure that a model endpoint can report in a Chat Completions body that it answered with 200.

OpenRouter answers 200 before its provider has finished. If the provider then fails, the only
choice of the body has ``finish_reason`` ``error``, and its ``error`` holds the provider's status,
such as 504 for an idle timeout. Each caller of an endpoint reads this status as the status of the
response, so that a transient failure is sent again.
"""

from __future__ import annotations

import re

#: How OpenAI-compatible servers (sglang, vLLM, OpenAI) word a request whose
#: prompt plus completion budget exceeds the model's context window.
CONTEXT_EXCEEDED = re.compile(r"maximum context length|context_length_exceeded|context length", re.I)


def choice_error_status(body: object) -> int | None:
    """The status in the ``error`` of the only choice of ``body``, if its ``finish_reason`` is ``error``.

    Returns None for each other body, and 502 for a failed choice without an integer status."""
    choices = body.get("choices") if isinstance(body, dict) else None
    if not isinstance(choices, list) or len(choices) != 1 or not isinstance(choices[0], dict):
        return None
    if choices[0].get("finish_reason") != "error":
        return None
    error = choices[0].get("error")
    code = error.get("code") if isinstance(error, dict) else None
    return code if isinstance(code, int) and not isinstance(code, bool) and 400 <= code < 600 else 502

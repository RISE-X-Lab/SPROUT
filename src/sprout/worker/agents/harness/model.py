"""The harness's model requests over an OpenAI-compatible Chat Completions endpoint: ``ModelClient``.

Each request asks for log-probs, whose token bytes are the raw text that the model sampled. A request
that fails for a transient reason is sent again a bounded number of times, within the rollout's
deadline. A request past the model's context window stops the rollout as a gradable one.
"""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import httpx

from sprout.worker.agents.harness.qwen_format import raw_completion
from sprout.worker.gateway.auth import bearer_headers
from sprout.worker.gateway.completions import CONTEXT_EXCEEDED, choice_error_status
from sprout.worker.gateway.server import _absorb_usage
from sprout.worker.run.control import RunControl, post
from sprout.worker.run.events import Usage
from sprout.worker.run.journal import JournalWriter

#: Statuses that a later identical request can pass. An endpoint answers 403 to refuse a request
#: for capacity, and the same request succeeds moments later.
TRANSIENT_STATUSES = frozenset({403, 408, 429, 500, 502, 503, 504})
#: How many times one request is sent at most.
ATTEMPTS = 12
#: The finish reasons of a complete choice.
FINISH_REASONS = frozenset({"stop", "tool_calls", "length"})


@dataclass(frozen=True)
class Completion:
    """One model response: its message, finish reason and usage, and the raw text the model sampled.

    ``raw_text`` is None when the endpoint returned no log-probs. ``body`` is the response without them."""

    message: dict[str, Any]
    finish_reason: str | None
    raw_text: str | None
    usage: Usage
    prompt_tokens: int
    completion_tokens: int
    body: dict[str, Any]


class ModelClient:
    """Sends the conversation to ``url`` as ``model``, within the run's ``control`` and ``deadline``."""

    def __init__(
        self,
        *,
        url: str,
        key: str,
        model: str | None,
        journal: JournalWriter,
        control: RunControl,
        deadline: float,
        parameters: Mapping[str, Any] | None = None,
        logprobs: bool = True,
    ) -> None:
        self.url, self.key, self.model = url, key, model
        self.journal, self.control, self.deadline = journal, control, deadline
        self.parameters = dict(parameters or {})
        self.logprobs = logprobs
        self.usage = Usage()

    def complete(
        self, messages: Sequence[Mapping[str, Any]], tools: Sequence[Mapping[str, Any]]
    ) -> Completion:
        """The model's response to ``messages`` with ``tools``.

        Raises ``RunAborted`` when the run stops, also for a spent deadline or context window, and
        ``httpx.HTTPError`` or ``ValueError`` for a failure that another request cannot fix."""
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": list(messages),
            "tools": list(tools),
            **self.parameters,
            "stream": False,
        }
        if self.logprobs:
            payload["logprobs"] = True
        body = self._post(payload)
        return self._completion(body)

    def _post(self, payload: dict[str, Any]) -> dict[str, Any]:
        """The JSON body of the first answer to ``payload`` that is not a transient failure."""
        for attempt in range(1, ATTEMPTS + 1):
            self.control.raise_if_stopped()
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                self.control.request_stop("harness wall-time budget exhausted", stop_reason="timeout")
                self.control.raise_if_stopped()
            try:
                response = post(
                    self.url,
                    json=payload,
                    timeout_s=min(900.0, remaining),
                    control=self.control,
                    headers=bearer_headers(self.key),
                )
                response.raise_for_status()
                body = response.json()
                failed = choice_error_status(body)
                if failed is None:
                    return dict(body)
                # A failed choice is not in the history, so its tokens are not the run's.
                raise httpx.HTTPStatusError(
                    "model endpoint answered 200 with a failed choice",
                    request=response.request,
                    response=httpx.Response(failed, content=response.content, request=response.request),
                )
            except (httpx.TransportError, TimeoutError) as error:
                if attempt == ATTEMPTS:
                    self.journal.emit("model.request.failed", error_type=type(error).__name__)
                    raise
                self._backoff(attempt, status=None, error_type=type(error).__name__)
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                if status == 400 and CONTEXT_EXCEEDED.search(error.response.text):
                    self.journal.emit(
                        "model.request.failed", status_code=status, error_type="context_length_exceeded"
                    )
                    self.control.request_stop(
                        "model context window exhausted", stop_reason="context_exhausted"
                    )
                    self.control.raise_if_stopped()
                if status not in TRANSIENT_STATUSES or attempt == ATTEMPTS:
                    self.journal.emit("model.request.failed", status_code=status)
                    raise
                self._backoff(attempt, status=status, error_type=None)
        raise AssertionError("unreachable: the last attempt returns or raises")

    def _backoff(self, attempt: int, *, status: int | None, error_type: str | None) -> None:
        delay = min(float(2 ** min(attempt, 5)), max(0.0, self.deadline - time.monotonic()))
        self.journal.emit(
            "model.request.retry", status_code=status, error_type=error_type, attempt=attempt, delay_s=delay
        )
        time.sleep(delay)
        self.control.raise_if_stopped()

    def _completion(self, body: dict[str, Any]) -> Completion:
        """The ``Completion`` of a response body. Raises ``ValueError`` unless it has one complete choice.

        A response that fails the check is journaled first, without its log-probs."""
        usage = Usage()
        _absorb_usage(body.get("usage"), usage)
        self.usage.add(usage)
        choices = body.get("choices")
        choice = choices[0] if isinstance(choices, list) and len(choices) == 1 else None
        message = choice.get("message") if isinstance(choice, dict) else None
        problem = (
            "the model response has no single choice"
            if not isinstance(choice, dict)
            else "the model response ended with finish_reason %r" % choice.get("finish_reason")
            if choice.get("finish_reason") not in FINISH_REASONS
            else "the model response has no assistant message"
            if not isinstance(message, dict) or message.get("role") != "assistant"
            else None
        )
        if problem is not None:
            stripped_body = {key: value for key, value in body.items() if key != "choices"}
            self.journal.emit(
                "model.response.invalid", error=problem, response=stripped_body, usage=usage.as_dict()
            )
            raise ValueError(problem)
        assert isinstance(choice, dict) and isinstance(message, dict)
        raw = raw_completion(choice)
        stripped = {**body, "choices": [{key: value for key, value in choice.items() if key != "logprobs"}]}
        native = body["usage"] if isinstance(body.get("usage"), dict) else {}
        return Completion(
            message=dict(message),
            finish_reason=choice.get("finish_reason"),
            raw_text=raw,
            usage=usage,
            prompt_tokens=int(native.get("prompt_tokens") or 0),
            completion_tokens=int(native.get("completion_tokens") or 0),
            body=stripped,
        )


def chat_url(base: str) -> str:
    """The Chat Completions URL of the OpenAI base URL ``base``, with or without its ``/v1``."""
    base = base.rstrip("/")
    return base + ("/chat/completions" if base.endswith("/v1") else "/v1/chat/completions")

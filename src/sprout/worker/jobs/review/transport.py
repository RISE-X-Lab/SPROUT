"""The analyst's and the reviewer's requests to a Chat Completions endpoint, each try in a receipt.

The transport is the same for every model. docs/RUNSTORE.md (Review requests) describes the
receipts and the retries.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any

import httpx

from sprout.contract.sampling import endpoint
from sprout.worker.gateway.auth import bearer_headers, bearer_key
from sprout.worker.gateway.completions import choice_error_status
from sprout.worker.jobs.review import structured_review
from sprout.worker.jobs.review.branch_plan import PromptRejected, ReviewResponseTruncated
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS

#: HTTP statuses that say nothing about the request: the gateway or its model was unavailable
#: or rate-limited for the moment. The transport sends the request again after one.
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
#: The seconds to wait before each further try after a transient reply, one try per entry.
#: A gateway that restarts or sheds load gets time to recover.
TRANSIENT_DELAYS_S = (10.0, 30.0, 60.0)


class RequestStopped(RuntimeError):
    """A try that the transport did not send, because its ``stop`` was set."""


class ReviewTransport:
    """Sends analyst and reviewer requests, and writes each try to a receipt in ``directory``.

    The receipts are ``review-NNN.json``, numbered in the order that the tries start. One
    instance is safe to share between threads. A retry names the earlier try of its own call.
    """

    def __init__(
        self,
        directory: str | Path,
        *,
        model: str,
        model_endpoint: str,
        api_key_env: str,
        max_output_tokens: int,
        timeout_s: float = 1800,
        reasoning_effort: str | None = None,
        temperature: float | None = None,
        stop: threading.Event | None = None,
        json_directive: str = DEFAULT_PROMPTS.transport_json_directive,
        supports_json_schema: bool = False,
    ) -> None:
        """Raises ValueError unless the output budget and timeout are positive and a temperature is 0..2.

        Once the caller sets ``stop``, each further try raises ``RequestStopped`` and sends nothing,
        and a wait before a retry ends at once. A request in flight runs to its end.
        ``supports_json_schema`` is the caller's word that the endpoint honours ``json_schema``."""
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.model_endpoint = endpoint(model_endpoint)
        self.supports_json_schema = bool(supports_json_schema)
        self.api_key_env = api_key_env
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        #: None leaves the temperature to the endpoint. Independent reviews of one point
        #: want it above zero, so that their proposals differ.
        self.temperature = temperature
        self.json_directive = json_directive
        #: The number of receipts written so far, one for each try.
        self.sequence = 0
        self._numbering = threading.Lock()
        self.stop = stop
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("review output budget must be positive")
        if timeout_s <= 0:
            raise ValueError("review timeout must be positive")
        if temperature is not None and (type(temperature) not in (int, float) or not 0 <= temperature <= 2):
            raise ValueError("review temperature must be between 0 and 2")

    @staticmethod
    def json_format(schema: dict[str, Any], name: str) -> dict[str, Any]:
        """The ``response_format`` that asks for strict JSON under ``schema``, named ``name``."""
        return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}

    def json_overhead(self, schema: dict[str, Any], name: str) -> int:
        """The characters that a ``json`` request adds to its prompt."""
        return len(self.json_directive) + len(json.dumps(self.json_format(schema, name)))

    def json(self, model: str, prompt: str, *, schema: dict[str, Any], name: str) -> str:
        """The model's reply to ``prompt``, constrained to ``schema``, as ``__call__`` returns it."""
        return self(model, prompt + self.json_directive, response_format=self.json_format(schema, name))

    def __call__(self, model: str, prompt: str, *, response_format: dict[str, Any] | None = None) -> str:
        """The model's reply to ``prompt``. Raises ValueError for a ``model`` other than ``self.model``.

        A try that times out is sent once more with half the output budget, and a transient reply
        is retried after each of ``TRANSIENT_DELAYS_S``. An HTTP 400 raises ``PromptRejected``. A
        reply cut at the token limit raises ``ReviewResponseTruncated``, and its receipt keeps it."""
        if model != self.model:
            raise ValueError("Review model differs from the frozen evaluation model")
        key = bearer_key(self.api_key_env)
        # A plan is a few kilobytes, so a try that runs to the timeout spent its budget on
        # reasoning. A smaller budget ends sooner, and the round still gets its review.
        budgets = [self.max_output_tokens, max(1, self.max_output_tokens // 2)]
        # The receipts this call has written, in order: each retry names the last.
        receipts: list[int] = []
        reason = None
        for attempt, max_tokens in enumerate(budgets):
            try:
                return self._request_through_outages(
                    model, prompt, key, max_tokens, receipts, reason, response_format=response_format
                )
            except httpx.TimeoutException:
                if attempt == len(budgets) - 1:
                    raise
                reason = "timeout"
        raise AssertionError("unreachable")

    def _request_through_outages(
        self,
        model: str,
        prompt: str,
        key: str,
        max_tokens: int,
        receipts: list[int],
        reason: str | None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """``_request``, sent again after each transient reply until ``TRANSIENT_DELAYS_S`` is spent.

        The last error propagates."""
        for delay in (*TRANSIENT_DELAYS_S, None):
            try:
                return self._request(
                    model, prompt, key, max_tokens, receipts, reason, response_format=response_format
                )
            except httpx.HTTPStatusError as error:
                if delay is None or error.response.status_code not in TRANSIENT_STATUSES:
                    raise
                reason = "http %d" % error.response.status_code
            except httpx.TimeoutException:
                raise
            except httpx.TransportError as error:
                if delay is None:
                    raise
                reason = type(error).__name__
            if self.stop is None:
                time.sleep(delay)
            else:
                self.stop.wait(delay)  # the next try sees the stop
        raise AssertionError("unreachable")

    def _request(
        self,
        model: str,
        prompt: str,
        key: str,
        max_tokens: int,
        receipts: list[int],
        reason: str | None = None,
        response_format: dict[str, Any] | None = None,
    ) -> str:
        """One try, in a receipt of its own. Once ``stop`` is set, raises ``RequestStopped`` unsent.

        ``receipts`` are the numbers of the call's earlier tries. The receipt names the last of
        them as the try that this one repeats, with ``reason``. This try's number joins them."""
        if self.stop is not None and self.stop.is_set():
            raise RequestStopped("the review stopped this transport; no further request was sent")
        retry_of = receipts[-1] if receipts else None
        with self._numbering:
            self.sequence += 1
            number = self.sequence
        receipts.append(number)
        receipt = self.directory / f"review-{number:03d}.json"
        started_at = time.time()
        request: dict[str, Any] = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
        }
        if response_format is not None:
            request["response_format"] = response_format
        record: dict[str, Any] = {"request": request, "started_at": started_at, "timeout_s": self.timeout_s}
        if retry_of is not None:
            record["retry_of"] = f"review-{retry_of:03d}.json"
            record["retry_reason"] = reason
        receipt.write_text(json.dumps(record, indent=2))
        try:
            value = self._complete(model, prompt, key, max_tokens, response_format=response_format)
            record["response"] = value
            text = self._text(value)
            # The receipt keeps a truncated reply in full. The caller's retry asks for a
            # complete reply and does not echo this one.
            record["text"] = text
            if value["choices"][0].get("finish_reason") == "length":
                raise ReviewResponseTruncated(text)
        except Exception as error:
            record["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            record["finished_at"] = time.time()
            record["seconds"] = record["finished_at"] - started_at
            receipt.write_text(json.dumps(record, indent=2))
        return text

    def _complete(
        self,
        model: str,
        prompt: str,
        key: str,
        max_tokens: int,
        response_format: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """The response body of one Chat Completions request. Raises ``PromptRejected`` on HTTP 400."""
        payload: dict[str, Any] = {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "stream": False,
        }
        if self.reasoning_effort is not None:
            payload["reasoning_effort"] = self.reasoning_effort
        if self.temperature is not None:
            payload["temperature"] = self.temperature
        if response_format is not None:
            payload["response_format"] = response_format
        response = httpx.post(
            self.model_endpoint + "/v1/chat/completions",
            json=payload,
            headers=bearer_headers(key),
            timeout=httpx.Timeout(self.timeout_s, connect=10),
        )
        try:
            response.raise_for_status()
        except httpx.HTTPStatusError as error:
            if error.response.status_code == 400:
                # The endpoint refused the request itself. In practice the prompt was past the
                # model's context, so the caller can retry with a smaller one.
                raise PromptRejected(
                    "model endpoint refused the request (HTTP 400 Bad Request): %s"
                    % error.response.text[:500]
                ) from error
            raise
        body: dict[str, Any] = response.json()
        failed = choice_error_status(body)
        if failed is not None:
            # The receipt keeps the provider's error through the message, and the status decides a retry.
            raise httpx.HTTPStatusError(
                "model endpoint answered 200 with a failed choice: %s" % body["choices"][0].get("error"),
                request=response.request,
                response=httpx.Response(failed, content=response.content, request=response.request),
            )
        return body

    def _text(self, value: dict[str, Any]) -> str:
        return structured_review.response_text(value)

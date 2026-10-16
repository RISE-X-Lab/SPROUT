"""Model-neutral Chat Completions transport for bounded benchmark reviews."""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path

import httpx

from sprout.contract.sampling import endpoint
from sprout.worker.gateway.auth import bearer_headers, bearer_key
from sprout.worker.jobs.review import structured_review
from sprout.worker.jobs.review.branch_plan import PromptRejected, ReviewResponseTruncated
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS

#: Replies that say nothing about the request: the gateway or the model behind
#: it was unavailable or rate-limited for the moment. One such reply once cost
#: a round (a lone 502 on a reviewer request, the branches never ran).
TRANSIENT_STATUSES = frozenset({408, 429, 500, 502, 503, 504})
#: How long to wait before each further try, so a gateway that is restarting
#: or shedding load gets time to recover; one try per entry.
TRANSIENT_DELAYS_S = (10.0, 30.0, 60.0)


class RequestStopped(RuntimeError):
    """A try the transport did not make, because its ``stop`` was set."""


class ReviewTransport:
    """Analyst and reviewer requests, each try a receipt ``review-NNN.json`` in
    ``directory``, numbered in the order the tries start.

    Safe to share between threads: a try takes its number under a lock, and a
    call's retries name the receipt that call wrote last, never one another
    thread wrote in between.

    ``stop``, when given, is set by the caller once its requests are no longer
    wanted: from then on every try -- a retry included -- raises
    ``RequestStopped`` instead of being sent, and a wait before a retry ends
    at once. A request already in flight runs its course. Without it (the
    Harbor review) nothing changes.

    ``supports_json_schema`` is the caller's word that the endpoint honours
    ``response_format: json_schema`` (``eval.evaluate --structured-review``);
    the analyst and reviewer then ask through ``json`` for schema-constrained
    output instead of a fenced plain-text plan. Off by default: a plain
    Chat Completions endpoint gets the same requests it always did.
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
    ):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.model = model
        self.model_endpoint = endpoint(model_endpoint)
        self.supports_json_schema = bool(supports_json_schema)
        self.api_key_env = api_key_env
        self.max_output_tokens = max_output_tokens
        self.timeout_s = timeout_s
        self.reasoning_effort = reasoning_effort
        #: Left to the endpoint's default unless set; several independent
        #: reviews of one point want it above zero, so their proposals differ.
        self.temperature = temperature
        self.json_directive = json_directive
        #: The receipts written so far, which is the number of requests made.
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
    def json_format(schema: dict, name: str) -> dict:
        return {"type": "json_schema", "json_schema": {"name": name, "strict": True, "schema": schema}}

    def json_overhead(self, schema: dict, name: str) -> int:
        return len(self.json_directive) + len(json.dumps(self.json_format(schema, name)))

    def json(self, model: str, prompt: str, *, schema: dict, name: str) -> str:
        return self(model, prompt + self.json_directive, response_format=self.json_format(schema, name))

    def __call__(self, model: str, prompt: str, *, response_format: dict | None = None) -> str:
        """One review: the model's text, or the error after every budget is spent.

        A request that times out is retried once with half the output budget:
        the plan the reviewer returns is a few kilobytes, so a request that runs
        to the timeout is one that spent the whole output budget on reasoning
        (64k tokens take about 30 minutes at the rates seen), and a smaller
        budget ends sooner rather than failing the round. A reply that is about
        the gateway rather than the request -- ``TRANSIENT_STATUSES``, or a
        connection that drops -- is retried with the same budget after each of
        ``TRANSIENT_DELAYS_S``. Each try is its own receipt; a retry names the
        one it repeats and why. Once ``stop`` is set, the next try raises
        ``RequestStopped`` instead.
        """
        if model != self.model:
            raise ValueError("Review model differs from the frozen evaluation model")
        key = bearer_key(self.api_key_env)
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
        response_format: dict | None = None,
    ) -> str:
        """``_request``, repeated after a transient gateway reply; the last error propagates."""
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
        response_format: dict | None = None,
    ) -> str:
        """One try, in a receipt of its own; ``receipts`` are the call's earlier tries, the last of
        which this one repeats (for ``reason``), and gain this one's number. Once ``stop``
        is set it raises ``RequestStopped`` instead, with no receipt and no request."""
        if self.stop is not None and self.stop.is_set():
            raise RequestStopped("the review stopped this transport; no further request was sent")
        retry_of = receipts[-1] if receipts else None
        with self._numbering:
            self.sequence += 1
            number = self.sequence
        receipts.append(number)
        receipt = self.directory / f"review-{number:03d}.json"
        started_at = time.time()
        request = {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "reasoning_effort": self.reasoning_effort,
            "temperature": self.temperature,
        }
        if response_format is not None:
            request["response_format"] = response_format
        record = {"request": request, "started_at": started_at, "timeout_s": self.timeout_s}
        if retry_of is not None:
            record["retry_of"] = f"review-{retry_of:03d}.json"
            record["retry_reason"] = reason
        receipt.write_text(json.dumps(record, indent=2))
        try:
            value = self._complete(model, prompt, key, max_tokens, response_format=response_format)
            record["response"] = value
            text = self._text(value)
            # The receipt keeps a truncated reply in full; the caller's retry asks
            # for a complete one without echoing it.
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
        self, model: str, prompt: str, key: str, max_tokens: int, response_format: dict | None = None
    ) -> dict:
        """One Chat Completions request; the response body."""
        payload = {
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
                # The request itself was refused -- a prompt past the model's
                # context, in practice. The caller may retry with a smaller one.
                raise PromptRejected(
                    "model endpoint refused the request (HTTP 400 Bad Request): %s"
                    % error.response.text[:500]
                ) from error
            raise
        return response.json()

    def _text(self, value: dict) -> str:
        return structured_review.response_text(value)

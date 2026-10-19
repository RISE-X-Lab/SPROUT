"""mini's model over an OpenAI-compatible chat/completions endpoint: ``ChatModel``.

A request that fails for a transient reason is retried a bounded number of times, within the
rollout's deadline. Each response is journaled with its usage before it is checked. A response
that breaks the tool-call envelope or the bash schema becomes feedback for another turn.
"""

import json
import os
import re
import time
from copy import deepcopy
from typing import Any

import httpx
import litellm
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from sprout.worker.agents.mini_swe.tools import malformed_feedback, malformed_tool_calls, mini_tool_schema
from sprout.worker.agents.slot import TaskSpec
from sprout.worker.gateway.auth import bearer_headers
from sprout.worker.gateway.completions import choice_error_status
from sprout.worker.gateway.server import _absorb_usage
from sprout.worker.run.control import RunControl, post
from sprout.worker.run.events import USAGE_UPDATED, Usage
from sprout.worker.run.journal import JournalWriter

#: How OpenAI-compatible servers (sglang, vLLM, OpenAI) word a request whose
#: prompt plus completion budget exceeds the model's context window.
CONTEXT_EXCEEDED = re.compile(r"maximum context length|context_length_exceeded|context length", re.I)


class ChatModel(LitellmModel):
    """mini's ``LitellmModel`` that posts each request to the run's endpoint, not through LiteLLM.

    ``OPENAI_BASE_URL`` and ``OPENAI_API_KEY`` come from ``task.env``, else from the process
    environment. ``__init__`` raises ``ValueError`` if there is no base URL.
    """

    # Upstream's retry loop retries nothing. _query retries transient failures a bounded number
    # of times, and then the error ends the attempt for queue-level recovery.
    abort_exceptions = [Exception, KeyboardInterrupt]  # type: ignore[list-item]  # upstream lists it too

    def __init__(
        self, task: TaskSpec, journal: JournalWriter, deadline: float, control: RunControl, **config: Any
    ) -> None:
        super().__init__(model_name=task.model, **config)
        self.journal, self.deadline, self.control = journal, deadline, control
        env = {**os.environ, **task.env}
        base = env.get("OPENAI_BASE_URL", "")
        if not base:
            raise ValueError("Set OPENAI_BASE_URL or configure the Sprout model gateway")
        self.url = base.rstrip("/") + (
            "/chat/completions" if base.rstrip("/").endswith("/v1") else "/v1/chat/completions"
        )
        self.key = env.get("OPENAI_API_KEY", "")
        self.usage = Usage()
        self.tools = mini_tool_schema(BASH_TOOL)

    def _query(self, messages: list[dict[str, Any]], **kwargs: Any) -> litellm.ModelResponse:
        self.control.raise_if_stopped()
        history_call_ids = {call["id"] for message in messages for call in message.get("tool_calls") or []}
        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "tools": self.tools,
            **self.config.model_kwargs,
            **kwargs,
            "stream": False,
        }
        # 403 is retried too. The endpoint answers 403 to refuse a request for capacity, and the
        # same request succeeds moments later.
        transient_statuses = {403, 408, 429, 500, 502, 503, 504}
        for attempt in range(1, 13):
            remaining = self.deadline - time.monotonic()
            if remaining <= 0:
                self.control.request_stop("mini wall-time budget exhausted", stop_reason="timeout")
                self.control.raise_if_stopped()
            try:
                response = post(
                    self.url,
                    json=payload,
                    timeout_s=min(600.0, remaining),
                    control=self.control,
                    headers=bearer_headers(self.key),
                )
                response.raise_for_status()
                failed = choice_error_status(response.json())
                if failed is None:
                    break
                # The status of a failed choice decides its retry, as the status of a response does. The
                # choice is not in the history, so its tokens are not the run's, like those of a 5xx.
                raise httpx.HTTPStatusError(
                    "model endpoint answered 200 with a failed choice",
                    request=response.request,
                    response=httpx.Response(failed, content=response.content, request=response.request),
                )
            # post() times out with the builtin TimeoutError, and httpx fails with a TransportError.
            # Both are request failures, retried within the deadline. Neither means a spent budget.
            except (httpx.TransportError, TimeoutError) as error:
                if attempt == 12:
                    self.journal.emit("model.request.failed", error_type=type(error).__name__)
                    raise
                delay = min(float(2 ** min(attempt, 5)), max(0.0, self.deadline - time.monotonic()))
                self.journal.emit(
                    "model.request.retry",
                    status_code=None,
                    error_type=type(error).__name__,
                    attempt=attempt,
                    delay_s=delay,
                )
                time.sleep(delay)
                self.control.raise_if_stopped()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                # A conversation past the model's context ends the rollout like a spent turn
                # budget. The verifier still grades the workspace.
                if status == 400 and CONTEXT_EXCEEDED.search(error.response.text):
                    self.journal.emit(
                        "model.request.failed", status_code=status, error_type="context_length_exceeded"
                    )
                    self.control.request_stop(
                        "model context window exhausted", stop_reason="context_exhausted"
                    )
                    self.control.raise_if_stopped()
                if status not in transient_statuses or attempt == 12:
                    self.journal.emit("model.request.failed", status_code=status)
                    raise
                delay = min(float(2 ** min(attempt, 5)), max(0.0, self.deadline - time.monotonic()))
                self.journal.emit("model.request.retry", status_code=status, attempt=attempt, delay_s=delay)
                time.sleep(delay)
                self.control.raise_if_stopped()
        raw = response.json()
        # A malformed completion can still consume tokens. Persist it before
        # validating its terminal choice so failed attempts remain measurable.
        self.journal.emit("raw.mini-swe-agent", response=raw)
        usage = Usage()
        _absorb_usage(raw.get("usage") if isinstance(raw, dict) else None, usage)
        self.usage.add(usage)
        self.journal.emit(USAGE_UPDATED, usage=usage.as_dict())
        if (
            not isinstance(raw, dict)
            or len(raw.get("choices", [])) != 1
            or raw["choices"][0].get("finish_reason") not in {"stop", "tool_calls", "length"}
        ):
            raise ValueError("mini model response is missing a valid terminal choice")
        message = raw["choices"][0].get("message", {})
        if message.get("role") != "assistant" or not isinstance(message.get("content") or "", str):
            raise ValueError("mini requires a text assistant completion")
        calls = message.get("tool_calls") or []
        if not isinstance(calls, list):
            raise ValueError("mini requires tool_calls to be a list")
        problems = malformed_tool_calls(calls, history_call_ids)
        if problems:
            # Like a schema error, this gets feedback and another turn. The check reads the wire dicts,
            # because litellm would invent a missing id, serialise object arguments and raise on the rest.
            raise FormatError(malformed_feedback(problems, message, raw))
        return litellm.ModelResponse(**raw)

    def _calculate_cost(self, response: litellm.ModelResponse) -> dict[str, float]:
        # The gateway's configured route owns authoritative pricing. Local
        # open-weight policies need no invented LiteLLM price registration.
        return {"cost": 0.0}

    def _parse_actions(self, response: litellm.ModelResponse) -> list[dict[str, Any]]:
        from jsonschema import Draft202012Validator

        # _query, or validate_assistant_turn for a reviewer's turn, has already checked the envelope.
        # Here each call's name and arguments meet the bash tool schema.
        calls = response.choices[0].message.tool_calls or []
        validator = Draft202012Validator(self.tools[0]["function"]["parameters"])
        errors: dict[str, str] = {}
        for call in calls:
            function = call.function
            if function.name != "bash":
                errors[call.id] = "Unknown tool. The available function is bash(command)."
                continue
            try:
                arguments = json.loads(function.arguments)
            except ValueError as error:
                errors[call.id] = f"Arguments are not valid JSON: {error}"
                continue
            problems = list(validator.iter_errors(arguments))
            if problems:
                errors[call.id] = "; ".join(error.message for error in problems)
        if errors:
            # DefaultAgent charges the response and asks again. CheckpointAgent turns this marker
            # into one tool-role error for each call, not a user hint.
            raise FormatError(
                {
                    "role": "user",
                    "content": "Tool schema validation failed. No calls in this response were executed. "
                    "Correct the arguments and submit new tool calls matching the tool schema.",
                    "extra": {
                        "interrupt_type": "FormatError",
                        "tool_schema_feedback": True,
                        "call_errors": errors,
                        "tools": deepcopy(self.tools),
                    },
                }
            )
        return super()._parse_actions(response)

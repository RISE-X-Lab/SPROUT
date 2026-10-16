"""mini's Model over an OpenAI-compatible chat/completions endpoint: bounded retry, usage
accounting, and the envelope and tool-schema checks that run before upstream parses actions.
"""

import json
import os
import re
import time
from copy import deepcopy

import httpx
import litellm
from minisweagent.exceptions import FormatError
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL

from sprout.worker.agents.mini_swe.tools import malformed_feedback, malformed_tool_calls, mini_tool_schema
from sprout.worker.agents.slot import TaskSpec
from sprout.worker.gateway.auth import bearer_headers
from sprout.worker.gateway.server import _absorb_usage
from sprout.worker.run.control import RunControl, post
from sprout.worker.run.events import USAGE_UPDATED, Usage
from sprout.worker.run.journal import JournalWriter

#: How OpenAI-compatible servers (sglang, vLLM, OpenAI) word a request whose
#: prompt plus completion budget exceeds the model's context window.
CONTEXT_EXCEEDED = re.compile(r"maximum context length|context_length_exceeded|context length", re.I)


class ChatModel(LitellmModel):
    # Transport and permanent HTTP errors end this attempt. A few transient
    # gateway statuses get a short bounded retry before queue-level recovery.
    abort_exceptions = [Exception, KeyboardInterrupt]

    def __init__(
        self, task: TaskSpec, journal: JournalWriter, deadline: float, control: RunControl, **config
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

    def _query(self, messages: list[dict], **kwargs) -> litellm.ModelResponse:
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
        # This OpenAI-compatible endpoint intermittently uses 403 for
        # admission/capacity rejection (the same request succeeds moments
        # later), so it is retryable here alongside ordinary gateway errors.
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
                break
            # ``asyncio.wait_for`` in worker.run.control raises the builtin
            # TimeoutError, while httpx raises TransportError subclasses. Both
            # are request-level failures and must be retried within the rollout
            # budget instead of being mislabeled as an exhausted actor budget.
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
                # A conversation that no longer fits the model's context ends
                # the rollout like an exhausted turn budget: the workspace is
                # still a valid attempt for the verifier to grade.
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
            # The same class of fault as a schema error, so the same treatment:
            # feedback and another turn, not the end of the run. It is checked on
            # the wire dicts because litellm would invent a missing id, serialise
            # object arguments and raise on the rest -- none of it feedback the
            # model can act on. DefaultAgent charges the response and retries.
            raise FormatError(malformed_feedback(problems, message, raw))
        return litellm.ModelResponse(**raw)

    def _calculate_cost(self, response: litellm.ModelResponse) -> dict:
        # The gateway's configured route owns authoritative pricing. Local
        # open-weight policies need no invented LiteLLM price registration.
        return {"cost": 0.0}

    def _parse_actions(self, response: litellm.ModelResponse) -> list[dict]:
        from jsonschema import Draft202012Validator

        # The envelope -- unique string ids, type, a named function with string
        # arguments -- was checked on the wire by _query, or for a reviewer's turn
        # by validate_assistant_turn; here the arguments meet the tool schema.
        calls = response.choices[0].message.tool_calls or []
        validator = Draft202012Validator(self.tools[0]["function"]["parameters"])
        errors = {}
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
            # DefaultAgent already charges this response and retries FormatError.
            # The marker is consumed below, producing tool-role errors, not a hint.
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

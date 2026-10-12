"""Execution controls for externally scheduled rollouts (Run Store, RL, eval loops).

The orchestrator consumes this contract; a queue/driver never runs the agent.
One rollout gets a model endpoint and sampling parameters applied to every
request, a per-trajectory turn limit (one turn = one model call), and a wall-time
deadline. Tool calls are counted but not limited.
"""

from copy import deepcopy
import math
import threading
import time
from urllib.parse import urlsplit

from sprout.harness.tools.pipeline import Continue, Reject, ToolInterceptor


def endpoint(value: str) -> str:
    """The API prefix callers append ``/v1/chat/completions`` to: a trailing ``/v1`` is dropped."""
    parts = urlsplit(value)
    if (parts.scheme not in {"http", "https"} or not parts.hostname or parts.username
            or parts.password or parts.query or parts.fragment):
        raise ValueError("Rollout endpoint must be HTTP(S), without credentials/query/fragment")
    value = value.rstrip("/")
    return value[:-3] if value.endswith("/v1") else value


def remaining_timeout(contract: dict, timeout_s: float) -> float:
    deadline = contract.get("deadline_at")
    if type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise ValueError("Rollout deadline must be finite")
    remaining = deadline - time.time()
    if remaining <= 0:
        raise ValueError("Rollout group wall-time budget expired before execution")
    return min(timeout_s, remaining)


class RolloutControls(ToolInterceptor):
    fail_mode = "closed"

    def __init__(self, contract: dict, journal, control):
        self.contract = deepcopy(contract)
        self.journal = journal
        self.control = control
        self.base_url = endpoint(contract["model_endpoint"])
        self.model_calls = self.tool_calls = 0
        self.lock = threading.Lock()
        self.timer = None
        self.deadline = contract["deadline_at"]
        if type(self.deadline) not in (float, int) or not math.isfinite(self.deadline):
            raise ValueError("Rollout deadline must be finite")
        if {"max_model_calls", "max_tool_calls"} & contract.keys():
            raise ValueError("Rollouts use per-trajectory max_turns, not model/tool call budgets")
        self.max_turns = contract.get("max_turns")
        if type(self.max_turns) is not int or self.max_turns <= 0:
            raise ValueError("Invalid max_turns")
        validate(contract.get("sampling_params", {}))

    def start(self):
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise ValueError("Rollout group wall-time budget expired before execution")
        self.timer = threading.Timer(remaining, lambda: self.control.request_stop(
            "rollout wall-time budget exhausted", stop_reason="timeout"))
        self.timer.daemon = True
        self.timer.start()

    def _stop_reason(self, limited: bool):
        """(reason, stop_reason) if this call may not proceed, else None."""
        if self.control.reason or time.time() >= self.deadline:
            return ("rollout wall-time budget exhausted or execution stopped",
                    None if self.control.reason else "timeout")
        if limited and self.model_calls >= self.max_turns:
            return "rollout turn budget exhausted", "max_turns_reached"
        return None

    def _admit(self, kind: str):
        with self.lock:
            refused = self._stop_reason(limited=kind == "model")
            if refused is None:
                setattr(self, kind + "_calls", getattr(self, kind + "_calls") + 1)
        if refused:
            reason, stop_reason = refused
            self.control.request_stop(reason, stop_reason=stop_reason)
            raise ValueError(reason)

    def prepare_model_request(self, payload: dict) -> dict:
        self._admit("model")
        self.journal.emit("rollout.model_tools", shape="chat/completions", tools=payload.get("tools", []))
        payload = {**payload, **parameters(self.contract.get("sampling_params", {}))}
        if self.contract.get("model"):
            payload["model"] = self.contract["model"]
        return payload

    def before(self, ctx):
        try:
            self._admit("tool")
            return Continue()
        except ValueError as error:
            return Reject(str(error))

    def finish(self):
        if self.timer:
            self.timer.cancel()
        self.journal.emit("rollout.usage", model_calls=self.model_calls, tool_calls=self.tool_calls)


def validate(value):
    allowed = {"temperature", "top_p", "top_k", "max_tokens", "max_new_tokens",
               "max_output_tokens", "stop", "reasoning_effort"}
    if not isinstance(value, dict) or set(value) - allowed:
        raise ValueError("Unknown message sampling controls")
    for key in ("temperature", "top_p"):
        if key in value:
            number = value[key]
            if (type(number) not in (int, float) or not math.isfinite(number)
                    or number < 0 or (key == "top_p" and not 0 < number <= 1)):
                raise ValueError(f"Invalid {key}")
    if "top_k" in value and (type(value["top_k"]) is not int or value["top_k"] < -1):
        raise ValueError("top_k must be -1 or a nonnegative integer")
    if ("reasoning_effort" in value
            and value["reasoning_effort"] not in {
                "none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"
            }):
        raise ValueError("Invalid reasoning_effort")
    limits = [value[k] for k in ("max_tokens", "max_new_tokens", "max_output_tokens") if k in value]
    if limits and any(type(n) is not int or n <= 0 or n != limits[0] for n in limits):
        raise ValueError("Output token limits must be positive and agree")
    stop = value.get("stop")
    if stop is not None and not isinstance(stop, str):
        if not isinstance(stop, list) or any(not isinstance(s, str) for s in stop):
            raise ValueError("stop must be text, a text list, or null")


def parameters(value):
    """The validated controls as Chat Completions request fields."""
    validate(value)
    result = deepcopy(value)
    limits = [result.pop(k) for k in ("max_tokens", "max_new_tokens", "max_output_tokens") if k in result]
    if limits:
        result["max_tokens"] = limits[0]
    return result

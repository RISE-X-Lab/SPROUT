"""Execution controls for externally scheduled rollouts (Run Store, RL, eval loops).

The orchestrator consumes this contract; a queue/driver never runs the agent.
One rollout gets a model endpoint and sampling parameters applied to every
request, a per-trajectory turn limit (one turn = one model call), and a wall-time
deadline. Tool calls are counted but not limited.
"""

import math
import threading
import time
from copy import deepcopy

from sprout.contract.sampling import endpoint, parameters, validate
from sprout.harness_tools.pipeline import Continue, Reject, ToolInterceptor


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
        self.timer = threading.Timer(
            remaining,
            lambda: self.control.request_stop("rollout wall-time budget exhausted", stop_reason="timeout"),
        )
        self.timer.daemon = True
        self.timer.start()

    def _stop_reason(self, limited: bool):
        """(reason, stop_reason) if this call may not proceed, else None."""
        if self.control.reason or time.time() >= self.deadline:
            return (
                "rollout wall-time budget exhausted or execution stopped",
                None if self.control.reason else "timeout",
            )
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

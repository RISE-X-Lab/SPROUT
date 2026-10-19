"""The execution limits of an externally scheduled rollout, from its rollout contract.

The Run Store, the RL driver and the evaluation loops schedule rollouts, but only the runner
runs the agent, and it applies the contract. A rollout gets one model endpoint, sampling
parameters applied to every model request, a turn limit for the trajectory (one turn is one
model call) and a wall-time deadline. Tool calls are counted but not limited.
"""

import math
import threading
import time
from copy import deepcopy
from typing import Any

from sprout.contract.sampling import endpoint, parameters, validate
from sprout.worker.run.control import RunControl
from sprout.worker.run.journal import JournalWriter
from sprout.worker.tools.pipeline import CallContext, Continue, Reject, ToolInterceptor, Verdict


def remaining_timeout(contract: dict[str, Any], timeout_s: float) -> float:
    """``timeout_s``, cut to the seconds left before the contract's ``deadline_at`` (Unix time).

    Raises ValueError if the deadline is missing or not a finite number, or has passed.
    """
    remaining = _deadline(contract) - time.time()
    if remaining <= 0:
        raise ValueError("Rollout group wall-time budget expired before execution")
    return min(timeout_s, remaining)


def _deadline(contract: dict[str, Any]) -> float:
    """The contract's ``deadline_at``. Raises ValueError if it is missing or not a finite number."""
    # The contract is JSON, so the deadline can be any value until it is checked.
    deadline: Any = contract.get("deadline_at")
    if type(deadline) not in (float, int) or not math.isfinite(deadline):
        raise ValueError("Rollout deadline must be finite")
    return float(deadline)


class RolloutControls(ToolInterceptor):
    """Applies one rollout contract to the model requests and the tool calls of a run.

    After the deadline or a stop, and for a model request past the turn limit, it stops the run
    through ``control``. The request then raises ValueError, and the tool call is rejected. A
    contract that it cannot apply raises ValueError in the constructor."""

    fail_mode = "closed"

    def __init__(self, contract: dict[str, Any], journal: JournalWriter, control: RunControl) -> None:
        self.contract = deepcopy(contract)
        self.journal = journal
        self.control = control
        self.base_url = endpoint(contract["model_endpoint"])
        self.model_calls = self.tool_calls = 0
        self.lock = threading.Lock()
        self.timer: threading.Timer | None = None
        self.deadline = _deadline(contract)
        if {"max_model_calls", "max_tool_calls"} & contract.keys():
            raise ValueError("Rollouts use per-trajectory max_turns, not model/tool call budgets")
        max_turns = contract.get("max_turns")
        if type(max_turns) is not int or max_turns <= 0:
            raise ValueError("Invalid max_turns")
        self.max_turns: int = max_turns
        validate(contract.get("sampling_params", {}))

    def start(self) -> None:
        """Arm the deadline: at ``deadline_at`` the run stops with ``stop_reason`` ``timeout``.

        Raises ValueError if the deadline has passed.
        """
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise ValueError("Rollout group wall-time budget expired before execution")
        self.timer = threading.Timer(
            remaining,
            lambda: self.control.request_stop("rollout wall-time budget exhausted", stop_reason="timeout"),
        )
        self.timer.daemon = True
        self.timer.start()

    def _stop_reason(self, limited: bool) -> tuple[str, str | None] | None:
        """(reason, stop_reason) if this call may not proceed, else None."""
        if self.control.reason or time.time() >= self.deadline:
            return (
                "rollout wall-time budget exhausted or execution stopped",
                None if self.control.reason else "timeout",
            )
        if limited and self.model_calls >= self.max_turns:
            return "rollout turn budget exhausted", "max_turns_reached"
        return None

    def _admit(self, kind: str) -> None:
        with self.lock:
            refused = self._stop_reason(limited=kind == "model")
            if refused is None:
                setattr(self, kind + "_calls", getattr(self, kind + "_calls") + 1)
        if refused:
            reason, stop_reason = refused
            self.control.request_stop(reason, stop_reason=stop_reason)
            raise ValueError(reason)

    def prepare_model_request(self, payload: dict[str, Any]) -> dict[str, Any]:
        """``payload`` with the contract's sampling parameters and model, after the turn is counted.

        Journals the request's tools as ``rollout.model_tools``. Raises ValueError, and stops the
        run, after the deadline or a stop, or past the turn limit.
        """
        self._admit("model")
        self.journal.emit("rollout.model_tools", shape="chat/completions", tools=payload.get("tools", []))
        payload = {**payload, **parameters(self.contract.get("sampling_params", {}))}
        if self.contract.get("model"):
            payload["model"] = self.contract["model"]
        return payload

    def before(self, ctx: CallContext) -> Verdict:
        """Count the tool call, or reject it after the deadline or a stop."""
        try:
            self._admit("tool")
            return Continue()
        except ValueError as error:
            return Reject(str(error))

    def finish(self) -> None:
        """Disarm the deadline, and journal the call counts as ``rollout.usage``."""
        if self.timer:
            self.timer.cancel()
        self.journal.emit("rollout.usage", model_calls=self.model_calls, tool_calls=self.tool_calls)

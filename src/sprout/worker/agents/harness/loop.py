"""The harness's turn loop: one model response and its one tool call a turn, until a stop condition.

The loop keeps mini's journal contract, so checkpoints, recovery points and the training export read
a harness run like a mini one:

- Each response is in the native history before its call runs; each turn closes after its result.
- The format check reads the raw text that the model sampled, never the server's parse of it.
- A rejected turn runs nothing. With ``on_invalid: feedback`` the model hears why and answers again;
  with ``terminate`` the rollout stops with the flaw's stop reason (Arctic-SWE, Table 18).
- ``submit`` runs nothing in the sandbox. Its call is journaled as skipped, so its turn is no
  branch point, and the final snapshot pairs with the last turn that ran a call.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sprout.worker.agents.harness.config import HarnessConfig
from sprout.worker.agents.harness.history import History
from sprout.worker.agents.harness.model import Completion, ModelClient
from sprout.worker.agents.harness.prompts import REPEAT_NOTE, feedback_message
from sprout.worker.agents.harness.qwen_format import (
    FormatViolation,
    Turn,
    check,
    render,
    structured_completion,
)
from sprout.worker.agents.harness.tools import bind, observation, runtime_call
from sprout.worker.run.events import TOOL_SKIPPED, TURN_COMPLETED, TURN_STARTED, USAGE_UPDATED
from sprout.worker.run.journal import JournalWriter

if TYPE_CHECKING:
    from sprout.worker.run.control import RunControl
    from sprout.worker.sandbox.image_shell import ImageShell
    from sprout.worker.tools.access import ToolAccess

#: The stop reason of each flaw that ends a training rollout. Every other flaw is ``format_invalid``.
STOP_REASONS = {"response_truncated": "response_truncated", "repetition": "repetition"}
#: What a ``submit`` call answers, as its tool result.
SUBMITTED = "Submitted. The repository will be graded as it is now."
#: The tools whose successful call changes a file, and so ends a run of repeated calls.
EDITS = frozenset({"str_replace", "write_file"})


@dataclass
class Session:
    """One harness rollout over ``tools``: the conversation, its history and its stop conditions."""

    journal: JournalWriter
    tools: ToolAccess
    client: ModelClient
    history: History
    config: HarnessConfig
    control: RunControl
    schemas: list[dict[str, Any]]
    messages: list[dict[str, Any]]
    cwd: str
    deadline: float
    image_shell: ImageShell | None = None
    turns: int = 0
    submitted: bool = False
    last_total_tokens: int = 0
    repeats: dict[str, tuple[int, str | None]] = field(default_factory=dict)

    @property
    def names(self) -> frozenset[str]:
        """The names of the tools that the model is offered."""
        return frozenset(tool["function"]["name"] for tool in self.schemas)

    def run(self) -> None:
        """Run turns until ``submit`` or a stop. Raises ``RunAborted`` for a stop."""
        for message in self.messages:
            self.history.message(message)
        while not self.submitted:
            self.control.raise_if_stopped()
            budget = self.config.max_total_tokens
            if budget and self.last_total_tokens >= budget:
                self.control.request_stop("trajectory token budget exhausted", stop_reason="max_total_tokens")
                self.control.raise_if_stopped()
            self.turn()

    def turn(self) -> None:
        """One model response, checked, and its call run, rejected or skipped."""
        turn_id = uuid4().hex
        self.turns += 1
        self.journal.emit(TURN_STARTED, turn=self.turns, source="actor")
        completion = self.client.complete(self.messages, self.schemas)
        self.journal.emit(USAGE_UPDATED, usage=completion.usage.as_dict())
        self.last_total_tokens = completion.prompt_tokens + completion.completion_tokens
        raw, source = completion.raw_text or "", "logprobs"
        try:
            if completion.raw_text is None:
                source = "structured"
                raw = structured_completion(completion.message)
            # A server that parses for us and sends no log-probs may send no reasoning either.
            parsed = check(
                raw, finish_reason=completion.finish_reason, require_reasoning=source == "logprobs"
            )
            arguments = bind(parsed.calls[0], self.schemas)
            canonical = render(parsed) == raw
            if not canonical and self.config.strict_roundtrip:
                raise FormatViolation(
                    "noncanonical_turn",
                    "Your response's whitespace differs from the required format. Write exactly one "
                    "newline before </think>, a blank line after it, and nothing after </tool_call>.",
                )
        except FormatViolation as violation:
            self._journal_raw(turn_id, completion, raw, source, violation=violation, canonical=None)
            self._reject(turn_id, raw, violation)
            return
        self._journal_raw(turn_id, completion, raw, source, violation=None, canonical=canonical)
        self._act(turn_id, parsed, arguments)

    def _journal_raw(
        self,
        turn_id: str,
        completion: Completion,
        raw: str,
        source: str,
        *,
        violation: FormatViolation | None,
        canonical: bool | None,
    ) -> None:
        verdict = {
            "valid": violation is None,
            "code": violation.code if violation else None,
            "canonical": canonical,
        }
        self.journal.emit(
            "raw.sprout-harness",
            turn_id=turn_id,
            response=completion.body,
            raw_text=raw,
            raw_source=source,
            finish_reason=completion.finish_reason,
            format=verdict,
        )

    def _reject(self, turn_id: str, raw: str, violation: FormatViolation) -> None:
        """Record a turn that broke the format. It runs nothing; ``terminate`` mode stops the rollout."""
        # Without </think> the whole text is reasoning: the reasoning block was never closed.
        reasoning, _, after = raw.partition("</think>")
        said = {"role": "assistant", "content": after.strip(), "reasoning_content": reasoning.strip()}
        self.history.message(said)
        self.messages.append(said)
        self._journal_said(said)
        self.journal.emit(
            "harness.turn.rejected",
            turn_id=turn_id,
            code=violation.code,
            message=violation.message,
            executed=False,
        )
        terminate = self.config.on_invalid == "terminate"
        if not terminate:
            feedback = {"role": "user", "content": feedback_message(violation.message)}
            self.history.message(feedback)
            self.messages.append(feedback)
        self.history.turn(turn_id, [], rejected=True, code=violation.code)
        self.journal.emit(TURN_COMPLETED, turn=self.turns, source="actor", rejected=True)
        if terminate:
            stop = STOP_REASONS.get(violation.code, "format_invalid")
            self.control.request_stop(
                "invalid turn (%s): %s" % (violation.code, violation.message), stop_reason=stop
            )
            self.control.raise_if_stopped()

    def _act(self, turn_id: str, parsed: Turn, arguments: dict[str, Any]) -> None:
        """Record the turn's call, then run it (or end the rollout for ``submit``) and record its result."""
        call = parsed.calls[0]
        call_id = "call_" + uuid4().hex
        said = {
            "role": "assistant",
            "content": parsed.content,
            "reasoning_content": parsed.reasoning,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {"name": call.name, "arguments": json.dumps(arguments, ensure_ascii=False)},
                }
            ],
        }
        self.history.message(said)
        self.messages.append(said)
        self._journal_said(said)
        if call.name == "submit":
            self.journal.emit(TOOL_SKIPPED, call_id=call_id, turn_id=turn_id, reason="submit")
            self._result(call_id, SUBMITTED)
            self.history.turn(turn_id, [call_id], submitted=True)
            self.journal.emit(TURN_COMPLETED, turn=self.turns, source="actor")
            self.submitted = True
            return
        runtime = runtime_call(
            call.name,
            arguments,
            cwd=self.cwd,
            remaining_s=self.deadline - time.monotonic(),
            image_shell=self.image_shell,
        )
        started = time.monotonic()
        result = self.tools.call(runtime.tool, runtime.args, call_id=call_id, turn_id=turn_id)
        text = observation(call.name, runtime, result, elapsed_s=time.monotonic() - started, tools=self.names)
        # A rollout limit can refuse the call and stop the run. Report that as the stop.
        self.control.raise_if_stopped()
        repeated = 0
        if call.name in EDITS and result.success:
            self.repeats.clear()
        else:
            repeated = self._repeated(call.name, arguments, text)
        self._result(
            call_id, text + (REPEAT_NOTE if repeated and self.config.on_invalid == "feedback" else "")
        )
        self.history.turn(turn_id, [call_id])
        step = self.journal.tool_calls()[-1]["step"]
        self.journal.emit(
            "model.turn.completed", turn_id=turn_id, call_ids=[call_id], step=step, source="actor"
        )
        self.tools.close_turn(turn_id)
        self.journal.emit(TURN_COMPLETED, turn=self.turns, source="actor")
        limit = self.config.repetition_limit
        if limit and repeated >= limit and self.config.on_invalid == "terminate":
            self.control.request_stop("repeated tool call with an unchanged result", stop_reason="repetition")
            self.control.raise_if_stopped()

    def _repeated(self, name: str, arguments: dict[str, Any], result: str) -> int:
        """How many times this exact call gave this exact result before, since the last successful edit.

        A call whose result changed, because a command edited files or time passed, starts again at 0."""
        key = json.dumps([name, arguments], sort_keys=True, ensure_ascii=False)
        count, last = self.repeats.get(key, (-1, None))
        count = count + 1 if last == result else 0
        self.repeats[key] = (count, result)
        return count

    def _result(self, call_id: str, text: str) -> None:
        message = {"role": "tool", "tool_call_id": call_id, "content": text}
        self.history.message(message)
        self.messages.append(message)

    def _journal_said(self, message: dict[str, Any]) -> None:
        """Journal what the model said in ``message``, for the readers of the trajectory."""
        if message.get("reasoning_content"):
            self.journal.emit("agent.thinking", text=message["reasoning_content"])
        if message.get("content"):
            self.journal.emit("agent.message", text=message["content"])

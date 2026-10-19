"""mini's agent loop with durable turns: ``CheckpointAgent``.

The agent keeps these guarantees:

- A branch continues an exact native history prefix.
- Each message is in the native history before any of its actions runs.
- A rejected response runs no command. The journal records it as a rejected turn.
- A turn closes only after every observation of its response is recorded.
"""

from copy import deepcopy
from typing import Any
from uuid import uuid4

import litellm
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import Submitted

from sprout.contract.mini_history import validate_assistant_turn
from sprout.worker.agents.mini_swe.history import History
from sprout.worker.agents.mini_swe.model import ChatModel
from sprout.worker.agents.mini_swe.sandbox import ToolEnvironment
from sprout.worker.agents.mini_swe.tools import tool_schema_feedback
from sprout.worker.run.control import RunControl
from sprout.worker.run.events import TOOL_SKIPPED, TURN_STARTED
from sprout.worker.run.journal import JournalWriter


class CheckpointAgent(DefaultAgent):
    """Upstream's ``DefaultAgent`` that writes each message to ``history`` and journals each turn."""

    model: ChatModel
    env: ToolEnvironment

    def __init__(
        self,
        model: ChatModel,
        env: ToolEnvironment,
        history: History,
        journal: JournalWriter,
        control: RunControl,
        *,
        assistant_turn: dict[str, Any] | None = None,
        resume_without_hint: bool = False,
        **kwargs: Any,
    ) -> None:
        """``assistant_turn`` is a reviewer's turn that runs before any model request.

        ``resume_without_hint`` continues the history prefix with no new user message. Each needs a
        prefix, and the two exclude each other. A violation raises ``ValueError``.
        """
        super().__init__(model, env, **kwargs)
        self.history, self.journal, self.control = history, journal, control
        self.initialized = False
        self.turn_id: str | None = None
        self.turn_count = 0
        self.turn_source = "actor"
        if type(resume_without_hint) is not bool:
            raise ValueError("resume_without_hint must be a boolean")
        if assistant_turn is not None and resume_without_hint:
            raise ValueError("Point-only continuation cannot include an assistant_turn")
        if (assistant_turn is not None or resume_without_hint) and not history.prefix_messages:
            raise ValueError("This branch mode requires an exact mini history prefix")
        self.pending_turn = (
            validate_assistant_turn(
                assistant_turn, history=history.prefix_messages, tools=model.tools, allow_reasoning=True
            )
            if assistant_turn is not None
            else None
        )
        self.omit_user_hint = assistant_turn is not None or resume_without_hint

    def add_messages(self, *messages: dict[str, Any]) -> list[dict[str, Any]]:
        """Add ``messages`` to the conversation and the native history, and return them.

        A rejection feedback (``extra`` with ``tool_schema_feedback``, ``malformed_tool_calls`` or a
        ``FormatError``) is recorded with the turn that it rejects.
        """
        if not self.initialized:
            self.initialized = True
            if self.history.prefix_messages:
                # The prefix replaces upstream's opening messages. The task follows as a user hint,
                # unless the branch continues the prefix without one.
                self.messages = deepcopy(self.history.prefix_messages)
                messages = (
                    ()
                    if self.omit_user_hint
                    else (self.model.format_message(role="user", content=self.extra_template_vars["task"]),)
                )
        for message in messages:
            extra = message.get("extra", {})
            response = extra.get("response")
            if extra.get("tool_schema_feedback") is True and isinstance(response, dict):
                self._schema_feedback(message, response)
                continue
            if isinstance(extra.get("malformed_tool_calls"), list):
                self._malformed_feedback(message)
                continue
            if extra.get("interrupt_type") == "FormatError" and isinstance(response, dict):
                self._format_feedback(message, response)
                continue
            super().add_messages(message)
            self.history.append({"type": "mini.message", "message": message})
            if message.get("role") == "assistant":
                self._journal_said(message)
        return list(messages)

    def _schema_feedback(self, feedback: dict[str, Any], response: dict[str, Any]) -> None:
        """Record a response whose calls failed the tool schema, with a tool-role error for each call.

        The response stays in the conversation. No call runs.
        """
        original = deepcopy(response["choices"][0]["message"])
        calls = original["tool_calls"]
        errors = feedback["extra"]["call_errors"]
        self.add_messages(original)
        observations = tool_schema_feedback(original, errors, feedback["extra"]["tools"])
        for call, observation in zip(calls, observations, strict=True):
            identifier = call["id"]
            self.journal.emit(
                "tool.rejected",
                call_id=identifier,
                turn_id=self.turn_id,
                name=(call.get("function") or {}).get("name"),
                executed=False,
                source="schema_validation",
                feedback=observation["content"],
            )
            self.add_messages(observation)
        ids = [call["id"] for call in calls]
        # A rejected turn closes in the conversation only. It has no tool step and no snapshot.
        self.history.append(
            {
                "type": "mini.turn",
                "turn_id": self.turn_id,
                "call_ids": ids,
                "rejected": True,
                "source": "schema_validation",
            }
        )
        self.journal.emit(
            "mini.turn.rejected",
            turn_id=self.turn_id,
            call_ids=ids,
            source="schema_validation",
            executed=False,
        )
        self.journal.emit("turn.completed", turn=self.turn_count, source="actor", rejected=True)

    def _malformed_feedback(self, feedback: dict[str, Any]) -> None:
        """Record a response with a malformed tool-call envelope as a rejected turn.

        No tool result can answer a call without a valid id, and a strict provider refuses the
        envelope if it is sent back. So only the user-role ``feedback`` enters the conversation.
        """
        problems = feedback["extra"]["malformed_tool_calls"]
        # The journal keeps what the model said, so an export does not show the turn as a blank step.
        self._journal_said(feedback["extra"]["response"]["choices"][0].get("message") or {})
        self.journal.emit(
            "mini.turn.rejected",
            turn_id=self.turn_id,
            source="malformed_tool_calls",
            executed=False,
            problems=problems,
        )
        self.journal.emit("turn.completed", turn=self.turn_count, source=self.turn_source, rejected=True)
        super().add_messages(feedback)
        self.history.append({"type": "mini.message", "message": feedback})

    def _format_feedback(self, feedback: dict[str, Any], response: dict[str, Any]) -> None:
        """Record upstream's format error, a response with no tool call, as a rejected turn.

        ``_query`` and ``_parse_actions`` check the rest first, so upstream raises no other format
        error. The response stays in the conversation as assistant context, before ``feedback``.
        """
        original = deepcopy(response["choices"][0]["message"])
        self.add_messages(original)
        self.journal.emit(
            "mini.turn.rejected",
            turn_id=self.turn_id,
            source="no_tool_calls",
            executed=False,
            feedback=feedback["content"],
        )
        self.journal.emit("turn.completed", turn=self.turn_count, source=self.turn_source, rejected=True)
        super().add_messages(feedback)
        self.history.append({"type": "mini.message", "message": feedback})

    def _journal_said(self, message: dict[str, Any]) -> None:
        """Journal what the model said in an assistant message: its text and reasoning."""
        if message.get("content"):
            self.journal.emit("agent.message", text=message["content"])
        if message.get("reasoning_content"):
            self.journal.emit("agent.thinking", text=message["reasoning_content"])

    def query(self) -> dict[str, Any]:
        """Start a turn, and return its assistant message: the pending reviewer's turn, else the model's.

        Raises ``RunAborted`` if the run was asked to stop.
        """
        self.control.raise_if_stopped()
        self.turn_id = uuid4().hex
        self.env.turn_id = self.turn_id
        self.turn_count += 1
        self.turn_source = "reviewer" if self.pending_turn is not None else "actor"
        self.journal.emit(TURN_STARTED, turn=self.turn_count, source=self.turn_source)
        if self.pending_turn is not None:
            message, self.pending_turn = self.pending_turn, None
            # mini's own parser reads the reviewer's turn. No model request runs, and the turn
            # adds no usage and no invented observation.
            response = litellm.ModelResponse(
                choices=[{"index": 0, "finish_reason": "tool_calls", "message": message}]
            )
            message["extra"] = {"actions": self.model._parse_actions(response), "source": "reviewer"}
            self.journal.emit(
                "branch.assistant_turn",
                source="reviewer",
                turn_id=self.turn_id,
                message={k: v for k, v in message.items() if k != "extra"},
            )
            self.add_messages(message)
            return message
        return super().query()

    def execute_actions(self, message: dict[str, Any]) -> list[dict[str, Any]]:
        """Run the actions of ``message`` in order, record each observation, and return them.

        Raises ``Submitted`` after the call that submits the task. The calls after it do not run. Each
        of them gets a not-executed result and a ``tool.skipped`` record, so the turn still closes.
        """
        actions = message.get("extra", {}).get("actions", [])
        # The model's entire response is already durable. Execute sequentially and persist each
        # real result, including COMPLETE_TASK_AND_SUBMIT_FINAL_OUTPUT.
        observed: list[dict[str, Any]] = []
        submitted: Submitted | None = None
        for index, action in enumerate(actions):
            output = self.env.execute(action)
            one = {**message, "extra": {**message["extra"], "actions": [action]}}
            observations = self.model.format_observation_messages(one, [output], self.get_template_vars())
            observed.extend(self.add_messages(*observations))
            try:
                # _check_finished reads only its output argument, so a ToolEnvironment can borrow it.
                LocalEnvironment._check_finished(self.env, output)  # type: ignore[arg-type]  # self unused
            except Submitted as end:
                submitted = end
                # Calls after the submission never run. Each gets a real tool result, so the history
                # stays paired for training export, and a tool.skipped record that settles the call.
                for skipped in actions[index + 1 :]:
                    self.journal.emit(
                        TOOL_SKIPPED,
                        call_id=skipped["tool_call_id"],
                        turn_id=self.turn_id,
                        reason="after_submit",
                    )
                    not_executed = {
                        "role": "tool",
                        "tool_call_id": skipped["tool_call_id"],
                        "content": "Not executed: the task was already submitted "
                        "by an earlier call in this response.",
                    }
                    observed.extend(self.add_messages(not_executed))
                break
        calls = [action["tool_call_id"] for action in actions]
        if calls and len(observed) == len(actions):
            # The whole response and every observation are durable: the turn is
            # closed, and ToolAccess may pair its end state with this prefix.
            self.history.append(
                {"type": "mini.turn", "turn_id": self.turn_id, "call_ids": calls, "source": self.turn_source}
            )
            self.journal.emit(
                "model.turn.completed",
                turn_id=self.turn_id,
                call_ids=calls,
                step=self.journal.tool_calls()[-1]["step"],
                source=self.turn_source,
            )
            # query() set this turn's id before the first action ran.
            assert self.turn_id is not None
            self.env.tools.close_turn(self.turn_id)
            self.journal.emit("turn.completed", turn=self.turn_count, source=self.turn_source)
        if submitted:
            raise submitted
        return observed

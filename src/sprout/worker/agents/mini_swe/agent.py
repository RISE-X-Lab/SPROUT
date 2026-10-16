"""Hooks around upstream DefaultAgent for durability: restore an exact prefix, persist messages
before execution, record rejected turns and close each turn once its observations are recorded.
"""

from copy import deepcopy
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
from sprout.worker.run.events import TURN_STARTED
from sprout.worker.run.journal import JournalWriter


class CheckpointAgent(DefaultAgent):
    def __init__(
        self,
        model: ChatModel,
        env: ToolEnvironment,
        history: History,
        journal: JournalWriter,
        control: RunControl,
        *,
        assistant_turn: dict | None = None,
        resume_without_hint: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(model, env, **kwargs)
        self.history, self.journal, self.control = history, journal, control
        self.initialized = False
        self.turn_id = None
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

    def add_messages(self, *messages: dict) -> list[dict]:
        if not self.initialized:
            self.initialized = True
            if self.history.prefix_messages:
                self.messages = deepcopy(self.history.prefix_messages)
                # The new mode continues the prefix directly, without a user hint.
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

    def _schema_feedback(self, feedback: dict, response: dict) -> None:
        """Persist real validation outcomes without pretending commands ran."""
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
        # Rejected turns close conversation only: no tool step or snapshot exists.
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

    def _malformed_feedback(self, feedback: dict) -> None:
        """A response whose tool-call envelope is malformed (``malformed_tool_calls``).

        No tool result can truthfully answer a call whose id is missing, blank,
        repeated or reused, and a strict provider refuses a malformed envelope
        echoed back in the next request -- so the rejection is a user message and
        the malformed response stays out of the conversation. The journal has the
        raw response (``raw.mini-swe-agent``), what the model said (so the turn
        is not a blank step in an export) and every problem.
        """
        problems = feedback["extra"]["malformed_tool_calls"]
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

    def _format_feedback(self, feedback: dict, response: dict) -> None:
        """mini's own format error: a response with no tool call at all.

        The envelope (``_query``) and the tool schema (``_parse_actions``) are
        checked before upstream's parser runs, so that is the only objection it
        can still raise. Upstream keeps the completion inside the feedback's
        extra field; it is preserved as actual assistant context as well, and the
        journal records the turn as rejected like the other two rejections.
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

    def _journal_said(self, message: dict) -> None:
        """Journal what the model said in an assistant message: its text and reasoning."""
        if message.get("content"):
            self.journal.emit("agent.message", text=message["content"])
        if message.get("reasoning_content"):
            self.journal.emit("agent.thinking", text=message["reasoning_content"])

    def query(self) -> dict:
        self.control.raise_if_stopped()
        self.turn_id = uuid4().hex
        self.env.turn_id = self.turn_id
        self.turn_count += 1
        self.turn_source = "reviewer" if self.pending_turn is not None else "actor"
        self.journal.emit(TURN_STARTED, turn=self.turn_count, source=self.turn_source)
        if self.pending_turn is not None:
            message, self.pending_turn = self.pending_turn, None
            # Use mini's real parser; no model request, usage or fake observation.
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

    def execute_actions(self, message: dict) -> list[dict]:
        actions = message.get("extra", {}).get("actions", [])
        # The model's entire response is already durable. Execute sequentially
        # and persist each real result, including COMPLETE_TASK_AND_SUBMIT.
        observed, submitted = [], None
        for index, action in enumerate(actions):
            output = self.env.execute(action)
            one = {**message, "extra": {**message["extra"], "actions": [action]}}
            observations = self.model.format_observation_messages(one, [output], self.get_template_vars())
            observed.extend(self.add_messages(*observations))
            try:
                LocalEnvironment._check_finished(self.env, output)
            except Submitted as end:
                submitted = end
                # Calls after the submission never run. Each still gets a real
                # tool result so the history stays paired for training export.
                for skipped in actions[index + 1 :]:
                    self.journal.emit(
                        "tool.rejected",
                        call_id=skipped["tool_call_id"],
                        turn_id=self.turn_id,
                        executed=False,
                        source="after_submit",
                    )
                    self.add_messages(
                        {
                            "role": "tool",
                            "tool_call_id": skipped["tool_call_id"],
                            "content": "Not executed: the task was already submitted "
                            "by an earlier call in this response.",
                        }
                    )
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
            self.env.tools.close_turn(self.turn_id)
            self.journal.emit("turn.completed", turn=self.turn_count, source=self.turn_source)
        if submitted:
            raise submitted
        return observed

"""Hooks around upstream DefaultAgent for Sprout execution, durability and limits.

The upstream run/step/query loop and tool parser/observation templates are used
directly. Hooks persist messages before execution, run each bash action through
the run's ToolAccess, close each turn once its observations are recorded,
restore an exact prefix and keep the submission command's actual result.
"""

from copy import deepcopy
import json
import math
import os
from pathlib import Path
import re
import shlex
import threading
import time
from uuid import uuid4

import httpx
import litellm
from minisweagent import package_dir
from minisweagent.agents.default import DefaultAgent
from minisweagent.environments.local import LocalEnvironment
from minisweagent.exceptions import FormatError, Submitted
from minisweagent.models.litellm_model import LitellmModel
from minisweagent.models.utils.actions_toolcall import BASH_TOOL
import yaml

from sprout.harness.agents.mini_swe.history import validate_assistant_turn
from sprout.harness.agents.mini_swe.history import History, load_prefix
from sprout.harness.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S, mini_tool_schema
from sprout.harness_core.control import GRADABLE_STOP_REASONS, RunAborted, RunControl
from sprout.harness_core.events import RUN_RESULT, TURN_STARTED, USAGE_UPDATED, Usage
from sprout.harness_core.control import post
from sprout.harness_core.journal import JournalWriter
from sprout.harness_core.slot import SlotResult, TaskSpec
from sprout.harness.tools.access import ToolAccess
from sprout.harness_gateway.server import _absorb_usage
from sprout.harness_gateway.auth import bearer_headers

#: How OpenAI-compatible servers (sglang, vLLM, OpenAI) word a request whose
#: prompt plus completion budget exceeds the model's context window.
CONTEXT_EXCEEDED = re.compile(r"maximum context length|context_length_exceeded|context length", re.I)


class ChatModel(LitellmModel):
    # Transport and permanent HTTP errors end this attempt. A few transient
    # gateway statuses get a short bounded retry before queue-level recovery.
    abort_exceptions = [Exception, KeyboardInterrupt]

    def __init__(self, task: TaskSpec, journal: JournalWriter, deadline: float,
                 control: RunControl, **config) -> None:
        super().__init__(model_name=task.model, **config)
        self.journal, self.deadline, self.control = journal, deadline, control
        env = {**os.environ, **task.env}
        base = env.get("OPENAI_BASE_URL", "")
        if not base:
            raise ValueError("Set OPENAI_BASE_URL or configure the Sprout model gateway")
        self.url = base.rstrip("/") + ("/chat/completions" if base.rstrip("/").endswith("/v1")
                                       else "/v1/chat/completions")
        self.key = env.get("OPENAI_API_KEY", "")
        self.usage = Usage()
        self.tools = mini_tool_schema(BASH_TOOL)

    def _query(self, messages: list[dict], **kwargs) -> litellm.ModelResponse:
        self.control.raise_if_stopped()
        history_call_ids = {call["id"] for message in messages
                            for call in message.get("tool_calls") or []}
        payload = {"model": self.config.model_name, "messages": messages, "tools": self.tools,
                   **self.config.model_kwargs, **kwargs, "stream": False}
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
                response = post(self.url, json=payload, timeout_s=min(600.0, remaining),
                                control=self.control, headers=bearer_headers(self.key))
                response.raise_for_status()
                break
            # ``asyncio.wait_for`` in harness_core.control raises the builtin
            # TimeoutError, while httpx raises TransportError subclasses. Both
            # are request-level failures and must be retried within the rollout
            # budget instead of being mislabeled as an exhausted actor budget.
            except (httpx.TransportError, TimeoutError) as error:
                if attempt == 12:
                    self.journal.emit("model.request.failed", error_type=type(error).__name__)
                    raise
                delay = min(float(2 ** min(attempt, 5)),
                            max(0.0, self.deadline - time.monotonic()))
                self.journal.emit("model.request.retry", status_code=None,
                                  error_type=type(error).__name__, attempt=attempt,
                                  delay_s=delay)
                time.sleep(delay)
                self.control.raise_if_stopped()
            except httpx.HTTPStatusError as error:
                status = error.response.status_code
                # A conversation that no longer fits the model's context ends
                # the rollout like an exhausted turn budget: the workspace is
                # still a valid attempt for the verifier to grade.
                if status == 400 and CONTEXT_EXCEEDED.search(error.response.text):
                    self.journal.emit("model.request.failed", status_code=status,
                                      error_type="context_length_exceeded")
                    self.control.request_stop("model context window exhausted",
                                              stop_reason="context_exhausted")
                    self.control.raise_if_stopped()
                if status not in transient_statuses or attempt == 12:
                    self.journal.emit("model.request.failed", status_code=status)
                    raise
                delay = min(float(2 ** min(attempt, 5)),
                            max(0.0, self.deadline - time.monotonic()))
                self.journal.emit("model.request.retry", status_code=status,
                                  attempt=attempt, delay_s=delay)
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
        if (not isinstance(raw, dict) or len(raw.get("choices", [])) != 1
                or raw["choices"][0].get("finish_reason") not in {"stop", "tool_calls", "length"}):
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
            raise FormatError({
                "role": "user",
                "content": "Tool schema validation failed. No calls in this response were executed. "
                           "Correct the arguments and submit new tool calls matching the tool schema.",
                "extra": {"interrupt_type": "FormatError", "tool_schema_feedback": True,
                          "call_errors": errors, "tools": deepcopy(self.tools)},
            })
        return super()._parse_actions(response)


class ToolEnvironment:
    """mini's environment over the run's ToolAccess: ``bash`` is the runtime's ``shell``."""

    def __init__(self, tools: ToolAccess, journal: JournalWriter, deadline: float,
                 control: RunControl, config: dict) -> None:
        self.tools, self.journal = tools, journal
        self.deadline, self.control = deadline, control
        self.config = config
        self.turn_id = None
        shell = next((tool for tool in tools.tools if tool["name"] == "shell"), None)
        if shell is None or not {"command", "working_dir", "timeout"} <= set(
                shell["inputSchema"].get("properties", {})):
            raise ValueError("mini requires Sprout shell(command, working_dir, timeout)")

    def execute(self, action: dict, cwd: str = "") -> dict:
        self.control.raise_if_stopped()
        call_id, command = action.get("tool_call_id"), action.get("command")
        if not isinstance(call_id, str) or not call_id or not isinstance(command, str):
            raise ValueError("Invalid mini bash action")
        env = self.config.get("env", {})
        if any(not re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", key) for key in env):
            raise ValueError("Invalid sandbox environment variable name")
        prefix = " ".join(shlex.quote(f"{key}={value}") for key, value in env.items())
        args = {"command": f"env {prefix} bash -c {shlex.quote(command)}",
                "working_dir": cwd or self.config["cwd"],
                "timeout": max(1, min(self.config.get("timeout", DEFAULT_COMMAND_TIMEOUT_S),
                                      math.ceil(self.deadline - time.monotonic())))}
        result = self.tools.call("shell", args, call_id=call_id, turn_id=self.turn_id)
        # A rollout limit refuses the call and stops the run; say so as a stop.
        self.control.raise_if_stopped()
        outcome = result.outcome
        if outcome is None or type(outcome.exit_code) is not int or outcome.running:
            raise ValueError("Sprout shell did not return a settled command outcome: %s"
                             % (result.error or result.output)[:500])
        # mini's completion sentinel and observation template consume streams.
        return {"output": outcome.stdout + outcome.stderr, "returncode": outcome.exit_code,
                "exception_info": "Command timed out" if outcome.timed_out else "",
                "extra": {"timed_out": outcome.timed_out}}

    def get_template_vars(self, **kwargs) -> dict:
        # No host environment or host uname enters the sandbox prompt.
        return {**self.config, "repository_dir": getattr(self, "repository_dir", None),
                "system": "Linux", "release": "", "version": "", "machine": "", **kwargs}

    def serialize(self) -> dict:
        return {"info": {"config": {"environment": self.config, "environment_type": "sprout-tool-access"}}}


class CheckpointAgent(DefaultAgent):
    def __init__(self, model: ChatModel, env: ToolEnvironment, history: History,
                 journal: JournalWriter, control: RunControl, *, assistant_turn: dict | None = None,
                 resume_without_hint: bool = False,
                 **kwargs) -> None:
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
        self.pending_turn = (validate_assistant_turn(assistant_turn, history=history.prefix_messages, tools=model.tools,
                                                     allow_reasoning=True)
                             if assistant_turn is not None else None)
        self.omit_user_hint = assistant_turn is not None or resume_without_hint

    def add_messages(self, *messages: dict) -> list[dict]:
        if not self.initialized:
            self.initialized = True
            if self.history.prefix_messages:
                self.messages = deepcopy(self.history.prefix_messages)
                # The new mode continues the prefix directly, without a user hint.
                messages = (() if self.omit_user_hint else (
                    self.model.format_message(role="user", content=self.extra_template_vars["task"]),))
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
            self.journal.emit("tool.rejected", call_id=identifier, turn_id=self.turn_id,
                              name=(call.get("function") or {}).get("name"), executed=False,
                              source="schema_validation", feedback=observation["content"])
            self.add_messages(observation)
        ids = [call["id"] for call in calls]
        # Rejected turns close conversation only: no tool step or snapshot exists.
        self.history.append({"type": "mini.turn", "turn_id": self.turn_id,
                             "call_ids": ids, "rejected": True, "source": "schema_validation"})
        self.journal.emit("mini.turn.rejected", turn_id=self.turn_id, call_ids=ids,
                          source="schema_validation", executed=False)
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
        self.journal.emit("mini.turn.rejected", turn_id=self.turn_id, source="malformed_tool_calls",
                          executed=False, problems=problems)
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
        self.journal.emit("mini.turn.rejected", turn_id=self.turn_id, source="no_tool_calls",
                          executed=False, feedback=feedback["content"])
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
            response = litellm.ModelResponse(choices=[{
                "index": 0, "finish_reason": "tool_calls", "message": message}])
            message["extra"] = {"actions": self.model._parse_actions(response), "source": "reviewer"}
            self.journal.emit("branch.assistant_turn", source="reviewer", turn_id=self.turn_id,
                              message={k: v for k, v in message.items() if k != "extra"})
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
                for skipped in actions[index + 1:]:
                    self.journal.emit("tool.rejected", call_id=skipped["tool_call_id"],
                                      turn_id=self.turn_id, executed=False, source="after_submit")
                    self.add_messages({"role": "tool", "tool_call_id": skipped["tool_call_id"],
                                       "content": "Not executed: the task was already submitted "
                                                  "by an earlier call in this response."})
                break
        calls = [action["tool_call_id"] for action in actions]
        if calls and len(observed) == len(actions):
            # The whole response and every observation are durable: the turn is
            # closed, and ToolAccess may pair its end state with this prefix.
            self.history.append({"type": "mini.turn", "turn_id": self.turn_id,
                                 "call_ids": calls, "source": self.turn_source})
            self.journal.emit("model.turn.completed", turn_id=self.turn_id, call_ids=calls,
                              step=self.journal.tool_calls()[-1]["step"], source=self.turn_source)
            self.env.tools.close_turn(self.turn_id)
            self.journal.emit("turn.completed", turn=self.turn_count, source=self.turn_source)
        if submitted:
            raise submitted
        return observed


def run(task: TaskSpec, journal: JournalWriter, tools: ToolAccess,
        home: Path, session_id: str) -> SlotResult:
    config = yaml.safe_load((package_dir / "config/mini.yaml").read_text())
    overrides = task.extra.get("mini", {})
    if not isinstance(overrides, dict) or set(overrides) - {"agent", "environment", "model"}:
        raise ValueError("mini config accepts agent/environment/model overrides")
    for section, value in overrides.items():
        config[section] = {**config.get(section, {}), **value}
    # Admission and pricing belong to Sprout. No hidden upstream step/cost budget.
    agent_config = {**config["agent"], "step_limit": 0, "cost_limit": 0,
                    "wall_time_limit_seconds": 0, "output_path": home / f"{session_id}.traj.json"}
    for key in ("step_limit", "cost_limit", "wall_time_limit_seconds", "output_path"):
        if key in overrides.get("agent", {}):
            raise ValueError(f"Use Sprout rollout controls instead of mini agent.{key}")
    env_config = {"cwd": None, "timeout": DEFAULT_COMMAND_TIMEOUT_S, **config.get("environment", {})}
    if set(env_config) - {"cwd", "timeout", "env"}:
        raise ValueError("mini environment only accepts cwd, timeout and env")
    if (not isinstance(env_config["cwd"], str) or not env_config["cwd"].startswith("/")
            or type(env_config["timeout"]) is not int or env_config["timeout"] <= 0
            or not isinstance(env_config.get("env", {}), dict)
            or any(not isinstance(k, str) or not isinstance(v, str)
                   for k, v in env_config.get("env", {}).items())):
        raise ValueError("mini requires a discovered or explicit absolute sandbox cwd, positive timeout and text env")
    workspace = task.extra.get("mini_workspace") or {"cwd": env_config["cwd"], "repository_dir": None}
    if workspace["cwd"] != env_config["cwd"]:
        raise ValueError("mini prompt workspace and execution working directory differ")
    agent_config["system_template"] += (
        "\n\n<workspace>\n"
        "Default working directory for every bash call: {{cwd}}\n"
        "{% if repository_dir %}Repository root: {{repository_dir}}\n{% endif %}"
        "Each call runs in a new shell. Changes made by cd or export do not persist across calls.\n"
        "</workspace>"
    )
    model_config = dict(config["model"])
    # This is an OpenAI-compatible wire adapter, not LiteLLM provider selection.
    model_config.pop("model_name", None)
    model_kwargs = model_config.pop("model_kwargs", {})
    model_kwargs.pop("drop_params", None)
    if set(model_kwargs) - {"temperature", "top_p", "top_k", "max_tokens", "stop", "parallel_tool_calls",
                            "reasoning_effort"}:
        raise ValueError("Unsupported mini model_kwargs; parameters must reach the model unchanged")
    model_config["model_kwargs"] = model_kwargs
    control = task.control or RunControl()
    deadline = time.monotonic() + task.timeout_s
    inherited = task.extra.get("native_prefix")
    prefix = load_prefix(inherited) if inherited else ()
    if prefix:
        parent = next(entry for entry in prefix if entry.get("type") == "mini.session")
        if parent.get("workspace") is not None and parent["workspace"] != workspace:
            raise ValueError("mini branch workspace differs from its inherited history")
    assistant_turn = task.extra.get("assistant_turn")
    resume_without_hint = task.extra.get("resume_without_hint", False)
    if type(resume_without_hint) is not bool:
        raise ValueError("resume_without_hint must be a boolean")
    if assistant_turn is not None and resume_without_hint:
        raise ValueError("Point-only continuation cannot include an assistant_turn")
    if (assistant_turn is not None or resume_without_hint) and not prefix:
        raise ValueError("This branch mode requires an exact mini history prefix")
    if assistant_turn is not None:
        # As continuation.prepare left it: possibly rendered, its text as reasoning_content.
        validate_assistant_turn(assistant_turn, history=[
            entry["message"] for entry in prefix if entry.get("type") == "mini.message"], allow_reasoning=True)
    history = History(home / f"{session_id}.jsonl", session_id, prefix=prefix, workspace=workspace)
    model, agent = None, None
    status, error, final_text = "error", None, ""
    journal.emit("run.started", slot="mini-swe-agent", slot_version="2.4.6", model=task.model,
                 task_prompt=task.prompt, cwd=task.cwd,
                 **({"prompt_primer": task.extra["prompt_primer"]} if task.extra.get("prompt_primer") else {}))
    journal.emit("session.ref", native_session_id=session_id, transcript_path=str(history.path.resolve()))
    if inherited:
        # Make standalone branch indexing aware of the inherited call ids too.
        journal.emit("mini.restored", inherited_native={
            **inherited, "slot": "mini-swe-agent",
            "byte_length": inherited.get("byte_length", Path(inherited["path"]).stat().st_size)})
    timer = threading.Timer(task.timeout_s, lambda: control.request_stop(
        "mini wall-time budget exhausted", stop_reason="timeout"))
    timer.daemon = True
    timer.start()
    try:
        model = ChatModel(task, journal, deadline, control, **model_config)
        journal.emit("rollout.model_tools", shape="chat/completions", tools=model.tools)
        env = ToolEnvironment(tools, journal, deadline, control, env_config)
        env.repository_dir = workspace["repository_dir"]
        agent = CheckpointAgent(model, env, history, journal, control,
                                assistant_turn=assistant_turn, resume_without_hint=resume_without_hint,
                                **agent_config)
        info = agent.run(task.prompt)
        final_text = info.get("submission", "")
        status = "completed" if info.get("exit_status") == "Submitted" else "error"
        error = None if status == "completed" else info.get("exit_status", "mini did not submit")
    except RunAborted as exc:
        # Exhausting a sampling budget leaves a valid workspace to grade.
        status = ("timeout" if control.stop_reason in GRADABLE_STOP_REASONS
                  else "killed")
        error = str(exc) or control.reason
    except (TimeoutError, KeyboardInterrupt) as exc:
        control.request_stop("mini wall-time budget exhausted", stop_reason="timeout")
        status, error = "timeout", str(exc) or control.reason
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        timer.cancel()
        history.close()
    usage = model.usage.as_dict() if model else {}
    journal.emit(RUN_RESULT, text=final_text)
    journal.emit("run.finished", status=status, error=error, usage=usage)
    return SlotResult(status=status, final_text=final_text, usage=usage,
                      native_session_id=session_id, error=error)


def malformed_tool_calls(calls: list, history_ids: set) -> list[dict]:
    """The structural faults of a response's raw tool calls, one record per faulty call.

    A fault is a call that is not an object; a ``type`` other than ``function``
    (a missing or null one is read as ``function``, the wire default, as litellm
    does); no function object; a missing or blank name; arguments that are
    neither a JSON string nor an object (litellm serialises an object, and the
    schema check then reads it); an id that is missing, blank, not a string,
    repeated in the response or already used in the conversation. The records
    are what the journal keeps and what the model is shown.
    """
    ids = [call.get("id") for call in calls if isinstance(call, dict)]
    problems = []
    for index, call in enumerate(calls):
        if not isinstance(call, dict):
            problems.append({"index": index, "id": None, "error": "the tool call is not an object"})
            continue
        faults = []
        if call.get("type") not in (None, "function"):
            faults.append('type must be "function"')
        function = call.get("function")
        if not isinstance(function, dict):
            faults.append("function must be an object with name and arguments")
        else:
            name = function.get("name")
            if not isinstance(name, str) or not name.strip():
                faults.append("function.name is missing or blank; the only tool is bash")
            if not isinstance(function.get("arguments"), (str, dict)):
                faults.append("function.arguments must be a JSON-encoded string")
        identifier = call.get("id")
        if not isinstance(identifier, str) or not identifier.strip():
            faults.append("id is missing or blank")
        elif ids.count(identifier) > 1:
            faults.append("id repeats another call's id in this response")
        elif identifier in history_ids:
            faults.append("id was already used earlier in this conversation")
        if faults:
            problems.append({"index": index, "id": identifier if isinstance(identifier, str) else None,
                             "error": "; ".join(faults)})
    return problems


def _head(value, limit: int) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else "%s …[%d more chars]" % (text[:limit], len(text) - limit)


def malformed_feedback(problems: list[dict], message: dict, raw: dict) -> dict:
    """The user-role rejection of a response with malformed tool calls.

    The rejected calls are quoted as data, head-clipped, so the model sees what
    it sent beside what was wrong with it; the instruction names the shape every
    call must have. ``extra`` carries the problems for the journal and the raw
    response, as upstream does for its own format errors.
    """
    calls = message.get("tool_calls") or []
    rejected = []
    for problem in problems:
        call = calls[problem["index"]]
        function = call.get("function") if isinstance(call, dict) else None
        function = function if isinstance(function, dict) else {}
        rejected.append({**problem, "name": function.get("name"),
                         "arguments": _head(function.get("arguments"), 2000)})
    content = (
        "Tool call error: the previous response was rejected before execution because "
        "%d of its %d tool call(s) were malformed. None of them ran, and the response is "
        "not part of the conversation.\n\n<rejected_response>\n%s\n</rejected_response>\n\n"
        "Resend the response as one assistant message: your reasoning as text, then "
        "tool_calls in which every call has a unique non-empty string id not used earlier "
        'in this conversation, type "function", function.name "bash" and function.arguments '
        "as a JSON-encoded string with the single field command, for example "
        '{"command": "ls -la"}.'
        % (len(problems), len(calls), json.dumps(
            {"content": _head(message.get("content") or "", 1000), "rejected_tool_calls": rejected},
            ensure_ascii=False, indent=1)))
    return {"role": "user", "content": content,
            "extra": {"interrupt_type": "FormatError", "malformed_tool_calls": problems, "response": raw}}


def tool_schema_feedback(message: dict, errors: dict[str, str], tools: list[dict]) -> list[dict]:
    observations = []
    for call in message["tool_calls"]:
        identifier = call["id"]
        error = {
            "type": "tool_schema_error" if identifier in errors else "tool_batch_rejected",
            "message": errors.get(identifier, "Another call in this response failed schema validation."),
        }
        content = json.dumps({
            "error": error, "executed": False,
            "instruction": "No calls in this response were executed. Correct and resend the tool calls.",
            "available_tools": tools,
        }, ensure_ascii=False)
        observations.append({"role": "tool", "tool_call_id": identifier, "content": content})
    return observations

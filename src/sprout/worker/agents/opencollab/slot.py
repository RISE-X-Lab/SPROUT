"""Drive the real Single2 session loop through Sprout-owned collaborators."""

from __future__ import annotations

import asyncio
import os
import threading
import time
from dataclasses import replace
from importlib.metadata import version
from pathlib import Path
from typing import Any
from uuid import uuid4

from opencollab.adapters.llm import LLMClient
from opencollab.adapters.llm.types import LLMResponse
from opencollab.adapters.llm.types import Usage as OCUsage
from opencollab.application.session_run import GenerationTimeoutError
from opencollab.bootstrap.agent_profiles import resolve_agent_profile
from opencollab.bootstrap.runtime_context import build_workspace_safety_policy
from opencollab.bootstrap.session_factory import build_session
from opencollab.domain.agent import Agent

from sprout.contract.agents.opencollab import load_prefix, training_messages, validate_assistant_turn
from sprout.worker.agents.opencollab.environment import SproutEnvironment
from sprout.worker.agents.opencollab.history import History
from sprout.worker.agents.slot import AgentSlot, SlotResult, SlotStatus, TaskSpec
from sprout.worker.gateway.server import _absorb_usage
from sprout.worker.run.control import RunAborted, RunControl
from sprout.worker.run.events import Usage
from sprout.worker.run.journal import JournalWriter
from sprout.worker.tools.access import ToolAccess


class RecordedModel:
    """Preserve native provider adaptation, recording usage and branch injection."""

    def __init__(
        self, client: Any, journal: Any, control: Any, pending: Any = None, client_factory: Any = None
    ) -> None:
        self.client, self.journal, self.control = client, journal, control
        self.owner_loop = asyncio.get_running_loop()
        self.client_factory = client_factory
        self.pending = pending
        self.usage = Usage()
        self.last_usage: dict[str, Any] = {}
        self.source = "actor"

    def __getattr__(self, name: Any) -> Any:
        return getattr(self.client, name)

    async def complete(self, messages: Any, **kwargs: Any) -> Any:
        self.control.raise_if_stopped()
        self.source = "actor"
        if self.pending is not None and kwargs.get("tools"):
            turn, self.pending = self.pending, None
            self.source = "reviewer"
            self.last_usage = {}
            self.journal.emit("branch.assistant_turn", source="reviewer", message=turn)
            return LLMResponse(
                content=turn["content"],
                tool_calls=turn["tool_calls"],
                usage=OCUsage(),
                reasoning=turn.get("reasoning_content"),
                finish_reason="tool_calls",
            )
        # Native read-time compaction runs in its own short-lived event loop.
        # HTTP connection pools must be created and closed on that same loop.
        if self.client_factory and asyncio.get_running_loop() is not self.owner_loop:
            async with self.client_factory() as client:
                response = await client.complete(messages, **kwargs)
        else:
            response = await self.client.complete(messages, **kwargs)
        # Persist token consumption before the loop validates or executes it.
        raw = response.usage.raw_usage or {
            "prompt_tokens": response.usage.input_tokens,
            "completion_tokens": response.usage.output_tokens,
            "total_tokens": response.usage.total_tokens,
        }
        usage = Usage()
        _absorb_usage(raw, usage)
        self.usage.add(usage)
        self.last_usage = usage.as_dict()
        self.journal.emit(
            "raw.opencollab-base",
            response={
                "usage": raw,
                "choices": [
                    {
                        "message": {
                            "role": "assistant",
                            "content": response.content,
                            "tool_calls": response.tool_calls,
                        },
                        "finish_reason": response.finish_reason,
                    }
                ],
            },
        )
        self.journal.emit("usage.updated", usage=self.last_usage)
        if response.content:
            self.journal.emit("agent.message", text=response.content)
        if response.reasoning:
            self.journal.emit("agent.thinking", text=response.reasoning)
        self.control.raise_if_stopped()
        return response


class TurnBridge:
    def __init__(self, history: Any, tools: Any, env: Any, model: Any, journal: Any, control: Any) -> None:
        self.history, self.tools, self.env, self.model = history, tools, env, model
        self.journal, self.control = journal, control
        self.session: Any = None
        self.error: Exception | None = None

    async def emit(self, event: Any) -> Any:
        try:
            self.journal.emit("opencollab.event", native_event_type=event.type, data=event.data)
            if event.type == "step_start":
                self.env.turn_id = uuid4().hex
                self.env.call_ids = []
                self.journal.emit("turn.started", turn=event.data["step"])
            elif event.type == "tool_start":
                # Native session has appended the complete assistant response;
                # make that durable before any sandbox side effect begins.
                self.history.persist_messages(self.session.messages)
            elif event.type == "step_end":
                if self.env.failure:
                    raise RuntimeError("Single2 sandbox transport failed") from self.env.failure
                self.history.persist_messages(self.session.messages)
                _, metadata = self.session._snapshot_for_save()
                metadata["session_state"]["phase"] = "precheck"
                metadata["session_state"]["pending_events"] = []
                metadata["session_state"]["pending_step_latency"] = None
                self.history.append(
                    {
                        "type": "oc.turn",
                        "turn_id": self.env.turn_id,
                        "runtime_call_ids": list(self.env.call_ids),
                        "source": self.model.source,
                        "session_snapshot": metadata,
                    }
                )
                if self.env.call_ids:
                    self.journal.emit(
                        "model.turn.completed",
                        turn_id=self.env.turn_id,
                        call_ids=list(self.env.call_ids),
                        source=self.model.source,
                        step=self.journal.tool_calls()[-1]["step"],
                    )
                    self.tools.close_turn(self.env.turn_id)
                self.journal.emit(
                    "turn.completed",
                    turn=event.data["step"],
                    source=self.model.source,
                    usage=self.model.last_usage,
                )
        except Exception as error:  # noqa: BLE001 - checkpoint failure aborts its owner
            # EventBus isolates subscriber failures. Fail the owning rollout
            # explicitly so a lost history/checkpoint is never silently accepted.
            self.error = error
            cause = error.__cause__
            fields = {"error_type": type(error).__name__, "error": str(error)}
            if cause is not None:
                fields.update(cause_type=type(cause).__name__, cause=str(cause))
            self.journal.emit("opencollab.bridge.failed", **fields)
            if self.env.failure is not None:
                detail = "%s: %s" % (type(self.env.failure).__name__, self.env.failure)
                self.control.request_stop(
                    "Single2 sandbox transport failed: " + detail, stop_reason="infrastructure_error"
                )
            else:
                self.control.request_stop(
                    "Single2 checkpoint bridge failed: " + str(error), stop_reason="infrastructure_error"
                )


class OpenCollabSlot(AgentSlot):
    name = "opencollab-base"

    def version(self) -> str:
        return version("opencollab")

    def run(self, task: TaskSpec, journal: JournalWriter, tools: ToolAccess) -> SlotResult:
        if tools is None or not task.model:
            raise ValueError("OpenCollab requires ToolAccess and an explicit model")
        return asyncio.run(self._run(task, journal, tools))

    async def _run(self, task: Any, journal: JournalWriter, tools: ToolAccess) -> SlotResult:
        control = task.control or RunControl()
        started_at = time.monotonic()
        deadline = started_at + task.timeout_s
        profile = resolve_agent_profile("base")
        if profile.name != "single2":
            raise ValueError("This experiment pins Base to the Single2 implementation")
        workspace = task.extra["mini_workspace"]
        prefix = load_prefix(task.extra["native_prefix"]) if task.extra.get("native_prefix") else []
        protocol = task.extra.get("opencollab_tool_protocol", "single2")
        if protocol not in {"single2", "review50-v1"}:
            raise ValueError("Unknown OpenCollab tool protocol")
        if prefix and prefix[0].get("tool_protocol", "single2") != protocol:
            raise ValueError("OpenCollab branch tool protocol differs from its parent")
        if prefix and prefix[0]["workspace"] != workspace:
            raise ValueError("Single2 branch workspace differs from the parent's")
        inherited_messages = training_messages(prefix)
        if protocol == "review50-v1":
            from sprout.worker.agents.opencollab.review50 import build_tools, definition

            profile = replace(profile, system_prompt=definition()["system_prompt"])
            native_tools = build_tools()
        else:
            native_tools = profile.resolve_tools("coding")
        # Use the runtime deadline; SDK timers can cancel a healthy queued command.
        for tool in native_tools:
            tool.disable_outer_timeout = True
        schemas = [tool.to_openai_schema() for tool in native_tools]
        pending = task.extra.get("assistant_turn")
        if pending is not None:
            if not prefix or task.extra.get("resume_without_hint"):
                raise ValueError("Reviewer continuation requires an exact prefix")
            pending = validate_assistant_turn(
                pending, history=inherited_messages, tools=schemas, allow_reasoning=True
            )
        if task.extra.get("resume_without_hint") and not prefix:
            raise ValueError("Point continuation requires an exact prefix")
        session_id = uuid4().hex
        home = Path(task.extra.get("native_home") or journal.path.parent / "opencollab-native")
        home.mkdir(parents=True, exist_ok=True)
        history = History(
            home / f"{session_id}.jsonl", session_id, workspace, prefix, tool_protocol=protocol, tools=schemas
        )
        settings = dict(task.extra.get("mini", {}).get("model", {}).get("model_kwargs", {}))
        wire_env = {**os.environ, **task.env}

        def client_factory() -> Any:
            return LLMClient(
                model=task.model,
                provider="openai",
                wire_protocol="chat_completions",
                base_url=wire_env.get("OPENAI_BASE_URL"),
                api_key=wire_env.get("OPENAI_API_KEY"),
                max_retries=5,
                request_timeout=task.timeout_s,
                stream_chat=False,
                first_event_timeout=None,
                stream_idle_timeout=None,
                provider_error_time_budget=min(300, task.timeout_s),
            )

        injected = task.extra.get("opencollab_llm")
        client = injected or client_factory()
        model = RecordedModel(
            client, journal, control, pending, client_factory=None if injected else client_factory
        )
        env = SproutEnvironment(tools, workspace["cwd"], control, deadline, image_shell=task.image_shell)
        env.rollout_controls = task.extra.get("rollout_controls")
        bridge = TurnBridge(history, tools, env, model, journal, control)
        agent = Agent(
            name="single2",
            system_prompt=profile.system_prompt,
            tools=list(native_tools),
            model=task.model,
            provider="openai",
            max_tokens_per_step=settings.get("max_tokens", 64000),
            reasoning_effort=settings.get("reasoning_effort", "high"),
            temperature=settings.get("temperature", 0.0),
            top_p=settings.get("top_p"),
        )
        session = build_session(
            agent=agent,
            env=env,
            llm=model,
            event_sink=bridge,
            max_budget_tokens=None,
            max_steps=None,
            agent_profile=profile,
            llm_timeout=None,
            safety_policy=build_workspace_safety_policy(env),
        )
        bridge.session = session
        journal.emit(
            "run.started",
            slot=self.name,
            slot_version=self.version(),
            agent_profile=profile.name,
            tool_protocol=protocol,
            model=task.model,
            task_prompt=task.prompt,
            cwd=task.cwd,
            actor_timeout_s=task.timeout_s,
            generation_timeout_s=None,
        )
        journal.emit("session.ref", native_session_id=session_id, transcript_path=str(history.path.resolve()))
        journal.emit("rollout.model_tools", shape="chat/completions", tools=schemas)
        timer = threading.Timer(
            task.timeout_s,
            lambda: control.request_stop("OpenCollab wall-time exhausted", stop_reason="timeout"),
        )
        timer.daemon = True
        timer.start()
        status: SlotStatus = "error"
        error: str | None = None
        text = ""
        wall_timeout = asyncio.timeout_at(deadline)
        try:
            if prefix:
                restored = home / f"{session_id}.restore.json"
                session.store.save(str(restored), inherited_messages, meta=prefix[-1]["session_snapshot"])
                session.restore(str(restored))
                # Keep inherited state/history, grant only NEW turns to this run.
                journal.emit(
                    "opencollab.restored",
                    inherited_native=task.extra["native_prefix"],
                    inherited_steps=session.step_count,
                    inherited_tokens=session.used_tokens,
                )
                if not pending and not task.extra.get("resume_without_hint") and task.prompt:
                    await session.add_user_message(task.prompt)
            else:
                await session.add_user_message(task.prompt)
            # The outer context identifies its own deadline; an inner provider
            # TimeoutError must never masquerade as the whole rollout deadline.
            async with wall_timeout:
                text = await session.run_loop()
            if env.failure:
                raise RuntimeError("Single2 sandbox transport failed") from env.failure
            if bridge.error:
                raise bridge.error
            control.raise_if_stopped()
            if session.phase.value == "error":
                raise RuntimeError(session.state.terminal_reason or "Single2 session failed")
            status = "completed"
        except RunAborted as exc:
            if env.failure or bridge.error:
                status, error = "error", str(env.failure or bridge.error)
            else:
                status = "timeout" if control.stop_reason in {"timeout", "max_turns_reached"} else "killed"
                error = str(exc) or control.reason
        except GenerationTimeoutError as exc:
            error = f"{type(exc).__name__}: {exc}"
            control.request_stop(error, stop_reason="infrastructure_error")
            status = "error"
        except TimeoutError as exc:
            if wall_timeout.expired():
                control.request_stop("OpenCollab wall-time exhausted", stop_reason="timeout")
                status, error = "timeout", control.reason
            else:
                error = f"{type(exc).__name__}: {exc}"
                control.request_stop(error, stop_reason="infrastructure_error")
                status = "error"
        except Exception as exc:  # noqa: BLE001 - the slot returns a durable outcome
            if (
                control.stop_reason in {"timeout", "max_turns_reached"}
                and not env.failure
                and not bridge.error
            ):
                status, error = "timeout", control.reason
            else:
                status, error = "error", f"{type(exc).__name__}: {exc}"
        finally:
            timer.cancel()
            history.persist_messages(session.messages)
            session.save(str(home / f"{session_id}.session.json"))
            history.close()
            await session.aclose()
            close = getattr(client, "close", None) or getattr(client, "aclose", None)
            if close:
                result = close()
                if asyncio.iscoroutine(result):
                    await result
        usage = model.usage.as_dict()
        journal.emit("run.result", text=text)
        submission = next(
            (tool.submission_receipt for tool in native_tools if getattr(tool, "submission_receipt", None)),
            None,
        )
        journal.emit(
            "opencollab.termination",
            tool_protocol=protocol,
            submitted=submission is not None,
            submission_receipt=submission,
            kind=(
                "explicit_submission"
                if submission
                else "final_response"
                if status == "completed"
                else control.stop_reason or status
            ),
        )
        termination = (
            "completed"
            if status == "completed"
            else "model_request_budget"
            if status == "timeout" and control.stop_reason == "max_turns_reached"
            else "actor_deadline"
            if status == "timeout"
            else "infrastructure_error"
        )
        journal.emit(
            "run.finished",
            status=status,
            error=error,
            usage=usage,
            termination_kind=termination,
            stop_reason=("infrastructure_error" if env.failure or bridge.error else control.stop_reason),
            elapsed_s=time.monotonic() - started_at,
            actor_timeout_s=task.timeout_s,
        )
        return SlotResult(
            status=status, final_text=text, usage=dict(usage), native_session_id=session_id, error=error
        )

"""One mini session for a ``TaskSpec``: ``run``.

``run`` loads upstream's ``mini.yaml`` with the task's overrides, runs the agent within
``task.timeout_s`` and returns a ``SlotResult``. The native history goes under ``home``.
"""

import threading
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

import yaml
from minisweagent import package_dir

from sprout.contract.mini_history import load_prefix, validate_assistant_turn
from sprout.worker.agents.mini_swe.agent import CheckpointAgent
from sprout.worker.agents.mini_swe.history import History
from sprout.worker.agents.mini_swe.model import ChatModel
from sprout.worker.agents.mini_swe.sandbox import ToolEnvironment, Workspace
from sprout.worker.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S
from sprout.worker.agents.slot import SlotResult, SlotStatus, TaskSpec
from sprout.worker.run.control import GRADABLE_STOP_REASONS, RunAborted, RunControl
from sprout.worker.run.events import RUN_RESULT
from sprout.worker.run.journal import JournalWriter
from sprout.worker.tools.access import ToolAccess


def run(task: TaskSpec, journal: JournalWriter, tools: ToolAccess, home: Path, session_id: str) -> SlotResult:
    """Run ``task`` with mini through ``tools``, and return its result.

    A stop for a gradable reason (``GRADABLE_STOP_REASONS``) gives ``timeout``, another stop ``killed``.
    Raises ``ValueError`` for an invalid configuration or branch request, before the run starts.
    """
    config = _mini_config(task.extra.get("mini", {}))
    # Admission and pricing belong to Sprout. No hidden upstream step/cost budget.
    agent_config = {
        **config["agent"],
        "step_limit": 0,
        "cost_limit": 0,
        "wall_time_limit_seconds": 0,
        "output_path": home / f"{session_id}.traj.json",
    }
    env_config = _environment_config(config.get("environment", {}))
    # task.extra holds JSON knobs: resolve_workspace's mini_workspace, and a branch's native_prefix
    # and assistant_turn. load_prefix and validate_assistant_turn check the last two.
    workspace = _workspace(cast(Workspace | None, task.extra.get("mini_workspace")), cwd=env_config["cwd"])
    agent_config["system_template"] += _WORKSPACE_PROMPT
    model_config = _model_config(config["model"])
    control = task.control or RunControl()
    deadline = time.monotonic() + task.timeout_s
    inherited = cast(dict[str, Any] | None, task.extra.get("native_prefix"))
    prefix = _inherited_prefix(inherited, workspace=workspace)
    assistant_turn = cast(dict[str, Any] | None, task.extra.get("assistant_turn"))
    resume_without_hint = _checked_continuation(
        assistant_turn=assistant_turn,
        resume_without_hint=task.extra.get("resume_without_hint", False),
        prefix=prefix,
    )
    history = History(home / f"{session_id}.jsonl", session_id, prefix=prefix, workspace=workspace)
    model: ChatModel | None = None
    status: SlotStatus = "error"
    final_text = ""
    error: str | None = None
    _emit_started(
        journal,
        model=task.model,
        prompt=task.prompt,
        cwd=task.cwd,
        session_id=session_id,
        transcript=history.path,
        inherited=inherited,
    )
    timer = _wall_time_timer(timeout_s=task.timeout_s, control=control)
    try:
        model = ChatModel(task, journal, deadline, control, **model_config)
        journal.emit("rollout.model_tools", shape="chat/completions", tools=model.tools)
        env = ToolEnvironment(tools, journal, deadline, control, env_config, image_shell=task.image_shell)
        env.repository_dir = workspace["repository_dir"]
        agent = CheckpointAgent(
            model,
            env,
            history,
            journal,
            control,
            assistant_turn=assistant_turn,
            resume_without_hint=resume_without_hint,
            **agent_config,
        )
        info = agent.run(task.prompt)
        final_text = info.get("submission", "")
        status = "completed" if info.get("exit_status") == "Submitted" else "error"
        error = None if status == "completed" else info.get("exit_status", "mini did not submit")
    except RunAborted as exc:
        # Exhausting a sampling budget leaves a valid workspace to grade.
        status = "timeout" if control.stop_reason in GRADABLE_STOP_REASONS else "killed"
        error = str(exc) or control.reason
    except (TimeoutError, KeyboardInterrupt) as exc:
        control.request_stop("mini wall-time budget exhausted", stop_reason="timeout")
        status, error = "timeout", str(exc) or control.reason
    except Exception as exc:
        status, error = "error", f"{type(exc).__name__}: {exc}"
    finally:
        timer.cancel()
        history.close()
    usage: dict[str, object] = dict(model.usage.as_dict()) if model else {}
    journal.emit(RUN_RESULT, text=final_text)
    journal.emit("run.finished", status=status, error=error, usage=usage)
    return SlotResult(
        status=status, final_text=final_text, usage=usage, native_session_id=session_id, error=error
    )


def _mini_config(overrides: object) -> dict[str, Any]:
    """Upstream's ``mini.yaml``, with ``overrides`` merged into its sections.

    Raises ``ValueError`` for an override of another section, or of a limit that Sprout sets."""
    config: dict[str, Any] = yaml.safe_load((package_dir / "config/mini.yaml").read_text())
    if not isinstance(overrides, dict) or set(overrides) - {"agent", "environment", "model"}:
        raise ValueError("mini config accepts agent/environment/model overrides")
    for section, value in overrides.items():
        config[section] = {**config.get(section, {}), **value}
    for key in ("step_limit", "cost_limit", "wall_time_limit_seconds", "output_path"):
        if key in overrides.get("agent", {}):
            raise ValueError(f"Use Sprout rollout controls instead of mini agent.{key}")
    return config


def _environment_config(environment: dict[str, Any]) -> dict[str, Any]:
    """The environment section with its defaults. Raises ``ValueError`` unless it is valid for the sandbox."""
    env_config = {"cwd": None, "timeout": DEFAULT_COMMAND_TIMEOUT_S, **environment}
    if set(env_config) - {"cwd", "timeout", "env"}:
        raise ValueError("mini environment only accepts cwd, timeout and env")
    if (
        not isinstance(env_config["cwd"], str)
        or not env_config["cwd"].startswith("/")
        or type(env_config["timeout"]) is not int
        or env_config["timeout"] <= 0
        or not isinstance(env_config.get("env", {}), dict)
        or any(not isinstance(k, str) or not isinstance(v, str) for k, v in env_config.get("env", {}).items())
    ):
        raise ValueError(
            "mini requires a discovered or explicit absolute sandbox cwd, positive timeout and text env"
        )
    return env_config


def _workspace(given: Workspace | None, *, cwd: str) -> Workspace:
    """``given``, else ``cwd`` with no repository. Raises ``ValueError`` if ``given`` is at another cwd."""
    workspace: Workspace = given or {"cwd": cwd, "repository_dir": None}
    if workspace["cwd"] != cwd:
        raise ValueError("mini prompt workspace and execution working directory differ")
    return workspace


#: The workspace section that ``run`` appends to mini's system template.
_WORKSPACE_PROMPT = (
    "\n\n<workspace>\n"
    "Default working directory for every bash call: {{cwd}}\n"
    "{% if repository_dir %}Repository root: {{repository_dir}}\n{% endif %}"
    "Each call runs in a new shell. Changes made by cd or export do not persist across calls.\n"
    "</workspace>"
)


def _model_config(model: dict[str, Any]) -> dict[str, Any]:
    """The model section as ``ChatModel`` takes it. Raises ``ValueError`` for an unsupported model kwarg."""
    model_config = dict(model)
    # This is an OpenAI-compatible wire adapter, not LiteLLM provider selection.
    model_config.pop("model_name", None)
    model_kwargs = model_config.pop("model_kwargs", {})
    model_kwargs.pop("drop_params", None)
    if set(model_kwargs) - {
        "temperature",
        "top_p",
        "top_k",
        "max_tokens",
        "stop",
        "parallel_tool_calls",
        "reasoning_effort",
    }:
        raise ValueError("Unsupported mini model_kwargs; parameters must reach the model unchanged")
    model_config["model_kwargs"] = model_kwargs
    return model_config


def _inherited_prefix(inherited: dict[str, Any] | None, *, workspace: Workspace) -> Sequence[dict[str, Any]]:
    """The native history that ``inherited`` names, or (). Raises ``ValueError`` for another workspace."""
    prefix = load_prefix(inherited) if inherited else ()
    if prefix:
        parent = next(entry for entry in prefix if entry.get("type") == "mini.session")
        if parent.get("workspace") is not None and parent["workspace"] != workspace:
            raise ValueError("mini branch workspace differs from its inherited history")
    return prefix


def _checked_continuation(
    *, assistant_turn: dict[str, Any] | None, resume_without_hint: object, prefix: Sequence[dict[str, Any]]
) -> bool:
    """``resume_without_hint`` once the branch's continuation is valid. Raises ``ValueError`` if it is not."""
    if type(resume_without_hint) is not bool:
        raise ValueError("resume_without_hint must be a boolean")
    if assistant_turn is not None and resume_without_hint:
        raise ValueError("Point-only continuation cannot include an assistant_turn")
    if (assistant_turn is not None or resume_without_hint) and not prefix:
        raise ValueError("This branch mode requires an exact mini history prefix")
    if assistant_turn is not None:
        # sprout.contract.continuation.prepare can have rendered it, with its text as reasoning_content.
        validate_assistant_turn(
            assistant_turn,
            history=[entry["message"] for entry in prefix if entry.get("type") == "mini.message"],
            allow_reasoning=True,
        )
    return resume_without_hint


def _emit_started(
    journal: JournalWriter,
    *,
    model: str | None,
    prompt: str,
    cwd: str,
    session_id: str,
    transcript: Path,
    inherited: dict[str, Any] | None,
) -> None:
    """Record the start of the run, its native session, and the history that a branch inherits."""
    journal.emit(
        "run.started",
        slot="mini-swe-agent",
        slot_version="2.4.6",
        model=model,
        task_prompt=prompt,
        cwd=cwd,
    )
    journal.emit("session.ref", native_session_id=session_id, transcript_path=str(transcript.resolve()))
    if inherited:
        # Branch-point indexing of this journal alone reads the inherited history and its call ids here.
        journal.emit(
            "mini.restored",
            inherited_native={
                **inherited,
                "slot": "mini-swe-agent",
                "byte_length": inherited.get("byte_length", Path(inherited["path"]).stat().st_size),
            },
        )


def _wall_time_timer(*, timeout_s: float, control: RunControl) -> threading.Timer:
    """A started daemon timer that stops ``control`` for a timeout after ``timeout_s`` seconds."""
    timer = threading.Timer(
        timeout_s, lambda: control.request_stop("mini wall-time budget exhausted", stop_reason="timeout")
    )
    timer.daemon = True
    timer.start()
    return timer

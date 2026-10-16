"""One mini session for a TaskSpec: merge mini.yaml with the task's overrides, wire the model,
environment and agent, enforce the wall-time budget and map the outcome to a SlotResult.
"""

import threading
import time
from pathlib import Path

import yaml
from minisweagent import package_dir

from sprout.contract.mini_history import load_prefix, validate_assistant_turn
from sprout.worker.agents.mini_swe.agent import CheckpointAgent
from sprout.worker.agents.mini_swe.history import History
from sprout.worker.agents.mini_swe.model import ChatModel
from sprout.worker.agents.mini_swe.sandbox import ToolEnvironment
from sprout.worker.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S
from sprout.worker.agents.slot import SlotResult, TaskSpec
from sprout.worker.run.control import GRADABLE_STOP_REASONS, RunAborted, RunControl
from sprout.worker.run.events import RUN_RESULT
from sprout.worker.run.journal import JournalWriter
from sprout.worker.tools.access import ToolAccess


def run(task: TaskSpec, journal: JournalWriter, tools: ToolAccess, home: Path, session_id: str) -> SlotResult:
    config = yaml.safe_load((package_dir / "config/mini.yaml").read_text())
    overrides = task.extra.get("mini", {})
    if not isinstance(overrides, dict) or set(overrides) - {"agent", "environment", "model"}:
        raise ValueError("mini config accepts agent/environment/model overrides")
    for section, value in overrides.items():
        config[section] = {**config.get(section, {}), **value}
    # Admission and pricing belong to Sprout. No hidden upstream step/cost budget.
    agent_config = {
        **config["agent"],
        "step_limit": 0,
        "cost_limit": 0,
        "wall_time_limit_seconds": 0,
        "output_path": home / f"{session_id}.traj.json",
    }
    for key in ("step_limit", "cost_limit", "wall_time_limit_seconds", "output_path"):
        if key in overrides.get("agent", {}):
            raise ValueError(f"Use Sprout rollout controls instead of mini agent.{key}")
    env_config = {"cwd": None, "timeout": DEFAULT_COMMAND_TIMEOUT_S, **config.get("environment", {})}
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
        validate_assistant_turn(
            assistant_turn,
            history=[entry["message"] for entry in prefix if entry.get("type") == "mini.message"],
            allow_reasoning=True,
        )
    history = History(home / f"{session_id}.jsonl", session_id, prefix=prefix, workspace=workspace)
    model, agent = None, None
    status, error, final_text = "error", None, ""
    journal.emit(
        "run.started",
        slot="mini-swe-agent",
        slot_version="2.4.6",
        model=task.model,
        task_prompt=task.prompt,
        cwd=task.cwd,
        **({"prompt_primer": task.extra["prompt_primer"]} if task.extra.get("prompt_primer") else {}),
    )
    journal.emit("session.ref", native_session_id=session_id, transcript_path=str(history.path.resolve()))
    if inherited:
        # Make standalone branch indexing aware of the inherited call ids too.
        journal.emit(
            "mini.restored",
            inherited_native={
                **inherited,
                "slot": "mini-swe-agent",
                "byte_length": inherited.get("byte_length", Path(inherited["path"]).stat().st_size),
            },
        )
    timer = threading.Timer(
        task.timeout_s, lambda: control.request_stop("mini wall-time budget exhausted", stop_reason="timeout")
    )
    timer.daemon = True
    timer.start()
    try:
        model = ChatModel(task, journal, deadline, control, **model_config)
        journal.emit("rollout.model_tools", shape="chat/completions", tools=model.tools)
        env = ToolEnvironment(tools, journal, deadline, control, env_config)
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
    usage = model.usage.as_dict() if model else {}
    journal.emit(RUN_RESULT, text=final_text)
    journal.emit("run.finished", status=status, error=error, usage=usage)
    return SlotResult(
        status=status, final_text=final_text, usage=usage, native_session_id=session_id, error=error
    )

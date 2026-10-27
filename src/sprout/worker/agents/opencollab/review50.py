"""Reviewed-data tool compatibility using OpenCollab's native execution loop."""

from __future__ import annotations

import json
import shlex
import time
from copy import deepcopy
from pathlib import Path
from typing import Any, cast

from opencollab.adapters.tools.base import Tool

from sprout.resource import resource


def definition() -> dict[str, Any]:
    return cast(dict[str, Any], json.loads(resource("opencollab/review50_protocol.json").read_text()))


class ReviewTool(Tool):  # type: ignore[misc]  # SDK base has no published typing.
    """Keep the recorded tool names and parameter meanings at the SDK boundary."""

    def __init__(self, schema: dict[str, Any], started: float) -> None:
        function = deepcopy(schema["function"])
        self.name = function["name"]
        self.description = function["description"]
        self.parameters = function["parameters"]
        self.started = started
        self.submission_receipt = None

    async def execute_with_runtime(self, params: Any, runtime: Any) -> Any:
        env = runtime.environment
        env.agent_tool_calls = getattr(env, "agent_tool_calls", 0) + 1
        if self.name == "run_command":
            if runtime.safety_policy is not None:
                runtime.safety_policy.check_cmd(params["command"])
            result = await env.exec_cmd(params["command"], timeout=300)
            runtime.observations.record_execution(exit_code=result.returncode, timed_out=result.timed_out)
            if result.returncode or result.timed_out:
                return json.dumps(
                    {
                        "status": "error",
                        "error_type": "CommandError",
                        "error_message": (result.stdout + result.stderr)[:5000],
                        "returncode": result.returncode,
                        "timed_out": result.timed_out,
                    }
                )
            return json.dumps(
                {
                    "status": "ok",
                    "stdout": result.stdout[:5000],
                    "stderr": result.stderr[:5000],
                    "is_truncated": result.stdout_truncated
                    or result.stderr_truncated
                    or len(result.stdout) > 5000
                    or len(result.stderr) > 5000,
                }
            )
        source = Path(__file__).with_name("review50_sandbox.py").read_text()
        command = (
            "python3 -c "
            + shlex.quote(source)
            + " "
            + shlex.quote(self.name)
            + " "
            + shlex.quote(json.dumps(params))
        )
        result = await env.exec_cmd(command, timeout=300)
        if result.returncode or result.timed_out or result.stdout_truncated:
            raise RuntimeError("Review50 sandbox helper failed: " + result.stderr[:1000])
        answer = json.loads(result.stdout)
        if self.name == "get_status":
            controls = getattr(env, "rollout_controls", None)
            budget = (
                controls.status()
                if controls is not None
                else {
                    "model_calls": None,
                    "max_turns": None,
                    "remaining_turns": None,
                    "runtime_tool_calls": None,
                    "max_tool_calls": None,
                    "remaining_seconds": max(0, env.deadline - time.monotonic()),
                }
            )
            answer.update(
                elapsed_seconds=time.monotonic() - self.started,
                remaining_seconds=budget["remaining_seconds"],
                budget={**budget, "agent_tool_calls": env.agent_tool_calls},
                runtime_calls_in_turn=len(env.call_ids),
                grading="pending official verifier",
            )
        if self.name in {"write_file", "edit_file"}:
            runtime.observations.record_write(
                completed=answer["status"] == "ok", changed=None, path=params["filepath"]
            )
        if self.name == "submit_patch" and answer["status"] == "submitted":
            # This is OpenCollab's own submission fact, not a replacement stop loop.
            runtime.observations.record_submission(json.dumps(answer))
            self.submission_receipt = answer
        return json.dumps(answer, ensure_ascii=False)


def build_tools() -> tuple[Tool, ...]:
    started = time.monotonic()
    return tuple(ReviewTool(schema, started) for schema in definition()["tools"])

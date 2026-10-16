"""Host mini-swe-agent over Harbor's Sprout task sandbox (microVM or podman)."""

from __future__ import annotations

import math
import time

from sprout.contract.run_spec import RunSpec
from sprout.contract.sampling import endpoint
from sprout.eval.harbor.agentenv_agent import AgentENVAgent
from sprout.worker.agents.mini_swe.slot import MiniSweSlot
from sprout.worker.agents.mini_swe.tools import DEFAULT_COMMAND_TIMEOUT_S, mini_tool_primer
from sprout.worker.gateway.auth import bearer_key


class AgentENVMini(AgentENVAgent):
    def __init__(
        self,
        *args,
        inference_endpoint: str,
        api_key_env: str | None = None,
        actor_timeout_s: float = 36000,
        max_output_tokens: int = 64000,
        max_turns: int = 300,
        reasoning_effort: str | None = None,
        final_memory_snapshot: bool = False,
        command_timeout_s: int = DEFAULT_COMMAND_TIMEOUT_S,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.final_memory_snapshot = bool(final_memory_snapshot)
        self.inference_endpoint = endpoint(inference_endpoint)
        if not api_key_env or not api_key_env.isidentifier():
            raise ValueError("mini requires an API key environment variable name")
        if not math.isfinite(actor_timeout_s) or actor_timeout_s <= 0:
            raise ValueError("mini actor timeout must be finite and positive")
        if type(max_output_tokens) is not int or max_output_tokens <= 0:
            raise ValueError("mini max_output_tokens must be positive")
        if type(max_turns) is not int or max_turns <= 0:
            raise ValueError("mini max_turns must be positive")
        if type(command_timeout_s) is not int or command_timeout_s <= 0:
            raise ValueError("mini command_timeout_s must be a positive number of seconds")
        self.api_key_env = api_key_env
        #: Seconds the runtime gives each of the agent's commands before killing
        #: it. The model's bash tool has no timeout argument, so this is the one
        #: limit every command runs under; a run records it so a branch runs
        #: under its parent's.
        self.command_timeout_s = command_timeout_s
        self.actor_timeout_s = float(actor_timeout_s)
        self.max_output_tokens = max_output_tokens
        self.max_turns = max_turns
        self.reasoning_effort = reasoning_effort

    @staticmethod
    def name() -> str:
        return "sprout-agentenv-mini-swe"

    def version(self) -> str | None:
        return MiniSweSlot().version()

    async def setup(self, environment) -> None:
        await super().setup(environment)
        if not self.model_name:
            raise ValueError("mini requires an explicit model")
        bearer_key(self.api_key_env)

    def _instruction(self, instruction, environment) -> str:
        # The same tool description the DeepSWE actor receives, generated from
        # the pinned mini tool contract rather than written separately.
        workdir = environment.task_env_config.workdir or environment.image_workdir
        return instruction + "\n\n" + mini_tool_primer(workdir)

    def _make_spec(self, prompt, workspace, journal, environment) -> RunSpec:
        # Harbor owns the task timeout. This finite Sprout deadline also bounds
        # model traffic if Harbor loses its cancellation path.
        return RunSpec(
            prompt=prompt,
            slot="mini-swe-agent",
            model=self.model_name,
            cwd=str(workspace.resolve()),
            run_id=self.logs_dir.parent.name,
            journal_path=journal,
            timeout_s=self.actor_timeout_s,
            keep_sandbox=True,
            checkpoint_enabled=environment.checkpoint_mode != "none",
            final_memory_snapshot=self.final_memory_snapshot,
            backend=environment.backend,
            interceptors=[environment.shell_context()],
            sandbox_image=environment.image,
            extra={
                "prompt_primer": "mini_tool_primer",
                "mini": {
                    "model": {
                        "model_kwargs": {
                            "max_tokens": self.max_output_tokens,
                            **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
                        }
                    },
                    "environment": {"timeout": self.command_timeout_s},
                },
                "rollout_contract": {
                    "message_export": True,
                    "model_endpoint": self.inference_endpoint,
                    "api_key_env": self.api_key_env,
                    "model": self.model_name,
                    "deadline_at": time.time() + self.actor_timeout_s,
                    "max_turns": self.max_turns,
                    "sampling_params": {
                        "max_new_tokens": self.max_output_tokens,
                        **({"reasoning_effort": self.reasoning_effort} if self.reasoning_effort else {}),
                    },
                },
            },
        )

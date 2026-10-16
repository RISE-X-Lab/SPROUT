"""Harbor environment lifecycle backed by an AgentENV microVM."""

from __future__ import annotations

import asyncio
import json
import subprocess
import urllib.request

from sprout.harbor_plugin.environment import SandboxEnvironment
from sprout.harness_sandbox.backends import default_aenv_server_url, resolve_microvm_endpoint
from sprout.harness_sandbox.templates import find_regctl

MIN_OCI_DISK_MB = 64 * 1024


class AgentENVEnvironment(SandboxEnvironment):
    def __init__(
        self,
        *args,
        runtime_bin: str,
        server_url: str | None = None,
        api_key_file: str | None = None,
        sandbox_ttl: int = 36000,
        **kwargs,
    ):
        # The same server the cleanup client assumes: a memory snapshot this
        # trial takes is released through it (methods.evaluate).
        self.server_url = server_url or default_aenv_server_url()
        self.api_key_file = api_key_file
        self.sandbox_ttl = sandbox_ttl
        super().__init__(*args, runtime_bin=runtime_bin, exec_timeout_s=sandbox_ttl, **kwargs)

    @staticmethod
    def type() -> str:
        return "agentenv"

    def _validate_definition(self) -> None:
        super()._validate_definition()
        if self.sandbox_ttl < 30000:
            raise ValueError("sandbox_ttl must cover an 8-hour actor plus setup/collection")
        disk = self.task_env_config.storage_mb
        if disk is not None and (disk < 1024 or disk % 1024):
            raise ValueError("AgentENV disk size must be at least 1024 MiB and divisible by 1024")

    def _backend_section(self) -> dict:
        section = {
            "server_url": self.server_url,
            "runtime_bin": self.runtime_bin,
            "from_image": True,
            "image_env": True,
            "sandbox_ttl": self.sandbox_ttl,
            "allow_internet": self._allow_internet,
            "request_timeout": 900,
        }
        if self.api_key_file:
            section["api_key_file"] = self.api_key_file
        return {"backend": "microvm", "microvm": section}

    def _resources(self) -> dict:
        return {
            "cpu": self._effective_cpus or 2,
            "memory_mb": self._effective_memory_mb or 1024,
            "disk_size_mb": max(self._effective_storage_mb or MIN_OCI_DISK_MB, MIN_OCI_DISK_MB),
        }

    async def _prepare_image(self, force_build: bool) -> tuple[str, dict]:
        if force_build:
            raise ValueError("Sprout boots the task's prebuilt image; it does not build one")
        return await asyncio.to_thread(
            self._pin_with_regctl, self.task_env_config.docker_image or self.overlay.base
        )

    @staticmethod
    def _pin_with_regctl(image: str) -> tuple[str, dict]:
        regctl = find_regctl()
        if regctl is None:
            raise RuntimeError("regctl is required to resolve OCI configuration")
        digest = subprocess.check_output(
            [str(regctl), "image", "digest", image], text=True, timeout=120
        ).strip()
        image = image.split("@", 1)[0] + "@" + digest
        config = json.loads(
            subprocess.check_output(
                [str(regctl), "image", "config", image, "--format", "{{json .Config}}"],
                text=True,
                timeout=120,
            )
        )
        return image, config

    async def _set_network(self, allow: bool) -> None:
        _, key = resolve_microvm_endpoint(self.backend["microvm"])
        payload = {"allow_internet_access": allow, "allowOut": [], "denyOut": []}
        request = urllib.request.Request(
            f"{self.server_url.rstrip('/')}/sandboxes/{self.session.sandbox_id}/network",
            data=json.dumps(payload).encode(),
            method="PUT",
            headers={"X-API-KEY": key, "Content-Type": "application/json"},
        )

        def update() -> None:
            with urllib.request.urlopen(request, timeout=30) as response:
                if response.status not in (200, 204):
                    raise RuntimeError(f"AgentENV network update returned {response.status}")

        await asyncio.to_thread(update)

"""Harbor's ``BaseEnvironment`` over one Sprout sandbox, for Harbor's own agents and verifier.

``SandboxEnvironment`` answers each call that Harbor makes (start, exec, upload, download,
the network policy, stop) through a :class:`~sprout.worker.sandbox.session.SandboxSession`.
A backend subclass decides how the image is pinned, the session's backend section, the
sandbox's shape and how egress changes. Harbor's trials on it are the reference that the
worker's ``harbor`` grade must equal (ADR-0003). ``docs/HARBOR_PLUGIN.md`` gives the steps
of a start and the tasks that the environment refuses.
"""

from __future__ import annotations

import asyncio
import json
import shlex
import tarfile
import tempfile
from pathlib import Path
from typing import Any
from uuid import uuid4

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.task.config import NetworkMode, NetworkPolicy

from sprout.contract.backend import backend_name
from sprout.worker.jobs.harbor.overlay import ImageOverlay, overlay_for, replay
from sprout.worker.sandbox.session import SandboxSession
from sprout.worker.tools.interceptors import shell_command

# --- the environment ------------------------------------------------------------


class SandboxEnvironment(BaseEnvironment):  # type: ignore[misc]  # Harbor ships no types: its base is Any.
    """Harbor's environment over one :class:`SandboxSession`. It implements every call that Harbor makes.

    A backend subclass implements ``type()``, ``_backend_section()``, ``_prepare_image()``, ``_resources()``
    and ``_set_network()``. It can extend ``_validate_definition()`` and ``_receipt_extra()``.
    """

    def __init__(self, *args: Any, runtime_bin: str, exec_timeout_s: int, **kwargs: Any) -> None:
        self.runtime_bin = str(Path(runtime_bin).resolve())
        #: The default bound on a command Harbor runs without a timeout of its own.
        self.exec_timeout_s = exec_timeout_s
        self.session: SandboxSession | None = None
        self.image_user = "root"
        self.image_workdir = "/"
        self.image_env: dict[str, str] = {}
        self.image_config: dict[str, Any] = {}
        self.backend: dict[str, object] = {}
        self.image = ""
        self.overlay: ImageOverlay | None = None
        super().__init__(*args, **kwargs)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, dynamic_network_policy=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    # --- what a backend decides ----------------------------------------------

    def _backend_section(self) -> dict[str, object]:
        """The session's backend config: ``{"backend": <name>, <name>: {...}}``."""
        raise NotImplementedError

    async def _prepare_image(self, force_build: bool) -> tuple[str, dict[str, Any]]:
        """The image to create the sandbox from, pinned to a digest, and its OCI config.

        ``start()`` reads the OCI keys ``User``, ``WorkingDir`` and ``Env`` from the config.
        """
        raise NotImplementedError

    def _resources(self) -> dict[str, int]:
        """The sandbox's shape for :meth:`SandboxSession.create`."""
        raise NotImplementedError

    async def _set_network(self, allow: bool) -> None:
        """Give the running sandbox egress if ``allow``, and take it away otherwise."""
        raise NotImplementedError

    # --- admission -----------------------------------------------------------

    def _validate_definition(self) -> None:
        if not Path(self.runtime_bin).is_file():
            raise ValueError(f"Missing sprout-runtime: {self.runtime_bin}")
        for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
            if (self.environment_dir / name).exists():
                raise ValueError(
                    f"{self.type()} multi-service composition is not implemented; no Docker fallback"
                )
        if self.task_env_config.mcp_servers:
            raise ValueError(f"Task-side MCP networking requires a {self.type()} service adapter")
        if not self.task_env_config.docker_image and not (self.environment_dir / "Dockerfile").is_file():
            raise ValueError("Task needs a docker_image or environment/Dockerfile")
        # overlay_for raises ValueError for an overlay that a sandbox cannot replay, before a sandbox exists.
        self.overlay = overlay_for(self.environment_dir, self.task_env_config.docker_image)
        if not self.task_env_config.docker_image and self.overlay is None:
            raise ValueError(
                "A task without a docker_image needs a Dockerfile a sandbox can replay; "
                "Sprout boots prebuilt images and never builds one"
            )
        for mount in self._mounts:
            if mount.get("target") not in {
                "/logs/agent",
                "/logs/verifier",
                "/logs/artifacts",
                "/logs/user-agent",
            }:
                raise ValueError(f"{self.type()} cannot reproduce host mount {mount.get('target')}")

    # --- lifecycle -----------------------------------------------------------

    async def _apply_network_policy(self, network_policy: NetworkPolicy) -> None:
        if network_policy.network_mode not in {NetworkMode.PUBLIC, NetworkMode.NO_NETWORK}:
            raise ValueError(f"{self.type()} supports only public and no-network policies")
        if self.session is None or not self.session.sandbox_id:
            raise RuntimeError("Cannot update network policy before sandbox start")
        await self._set_network(network_policy.network_mode == NetworkMode.PUBLIC)

    @property
    def _allow_internet(self) -> bool:
        """Whether the sandbox starts with egress: the baseline policy that Harbor gave."""
        return bool(self._network_policy.network_mode == NetworkMode.PUBLIC)

    async def start(self, force_build: bool = False) -> None:
        self.image, config = await self._prepare_image(force_build)
        self.image_config = config
        self.image_user = config.get("User") or "root"
        self.image_workdir = config.get("WorkingDir") or "/"
        self.image_env = dict(value.split("=", 1) for value in config.get("Env") or [] if "=" in value)
        self.backend = self._backend_section()
        self.session = SandboxSession(backend=self.backend, quiet=True)
        resources = self._resources()
        try:
            creation = asyncio.create_task(asyncio.to_thread(self.session.create, self.image, resources))
            try:
                created = await asyncio.shield(creation)
            except asyncio.CancelledError:
                # Let the creation finish, so that the stop() below destroys what it made.
                await asyncio.shield(creation)
                raise
            if not created:
                raise RuntimeError(self.session.create_error)
            await self._checked(
                "mkdir -p /logs/agent /logs/user-agent /logs/verifier /logs/artifacts && "
                "chmod 777 /logs/agent /logs/user-agent /logs/verifier /logs/artifacts"
            )
            self._write_receipt(resources)
            await self._populate()
        except BaseException:
            await self.stop(delete=True)
            raise

    def _running_session(self) -> SandboxSession:
        """The session of the started sandbox. Raises RuntimeError before ``start()`` or after ``stop()``."""
        if self.session is None:
            raise RuntimeError(f"{self.type()} environment has not started")
        return self.session

    def _write_receipt(self, resources: dict[str, int]) -> None:
        """Write ``<session_id>.<type>.json`` into the trial directory: what this sandbox is."""
        record = {
            "backend": backend_name(self.backend),
            "sandbox_id": self._running_session().sandbox_id,
            "image": self.image,
            "image_config": self.image_config,
            "requested_disk_size_mb": self._effective_storage_mb,
            "resources": resources,
            "network": self._network_policy.model_dump(mode="json"),
            "overlay": (
                {"copies": [[str(s), d] for s, d in self.overlay.copies], "runs": list(self.overlay.runs)}
                if self.overlay
                else None
            ),
            **self._receipt_extra(),
        }
        (self.trial_paths.trial_dir / f"{self.session_id}.{self.type()}.json").write_text(
            json.dumps(record, indent=2)
        )

    def _receipt_extra(self) -> dict[str, object]:
        """What a backend adds to the receipt beyond the common fields."""
        return {}

    async def _populate(self) -> None:
        """Put the task's files in the fresh sandbox: an overlay's COPY/RUN, or environment/."""
        if self.overlay is not None:
            await asyncio.to_thread(
                replay,
                self._running_session(),
                self.overlay,
                cwd=self.image_workdir,
                env={**self.image_env, **(self._merge_env(None) or {})},
                timeout_s=int(self.task_env_config.build_timeout_sec),
            )
            return
        await self._upload_environment_dir_after_start()

    async def stop(self, delete: bool = True) -> None:
        if self.session is None:
            return
        session, pool = self.session, self.session._pool
        try:
            # delete=False releases the sandbox from the pool, because closing a pool destroys what it holds.
            # A failed destroy still closes the pool below, and its error names what is left.
            await asyncio.to_thread(
                session._loop.drive, session._destroy_async() if delete else session._release_async()
            )
        finally:
            if pool is not None:
                await asyncio.to_thread(session._loop.drive, pool.close())
            self.session = None

    # --- commands ------------------------------------------------------------

    def command_args(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        user: str | int | None = None,
    ) -> dict[str, str]:
        """The ``shell`` tool's arguments that run ``command`` as Harbor's ``exec`` asks.

        Unless the call names them, the command runs in the task's workdir or the image's ``WorkingDir``,
        as the image's ``User``. Its environment is the image's ``Env`` with Harbor's ``env`` on top.
        """
        effective_user = self._resolve_user(user)
        return {
            "command": shell_command(
                command,
                cwd or self.task_env_config.workdir or self.image_workdir,
                {**self.image_env, **(self._merge_env(env) or {})},
                self.image_user if effective_user is None else effective_user,
            ),
            "working_dir": "/",
        }

    async def exec(
        self,
        command: str,
        cwd: str | None = None,
        env: dict[str, str] | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        session = self._running_session()
        timeout = timeout_sec or self.exec_timeout_s
        arguments = {
            **self.command_args(command, cwd, env, user),
            "timeout": timeout,
            "max_output_bytes": 16 * 1024 * 1024,
        }
        pending = asyncio.create_task(asyncio.to_thread(session.execute, "shell", arguments, timeout + 60))
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            # Cancel the command on the session's loop, and wait until its thread returns.
            loop = session._loop.get()
            loop.call_soon_threadsafe(lambda: [task.cancel() for task in asyncio.all_tasks(loop)])
            try:
                await asyncio.shield(pending)
            except BaseException:
                pass
            raise
        if result.outcome is not None:
            outcome = result.outcome
            if outcome.timed_out:
                raise TimeoutError(f"Environment command timed out after {timeout}s")
            if outcome.exit_code is None:
                raise RuntimeError("Environment command has no exit status")
            answer = ExecResult(stdout=outcome.stdout, stderr=outcome.stderr, return_code=outcome.exit_code)
        elif result.success:
            answer = ExecResult(stdout=result.output, stderr="", return_code=0)
        else:
            raise RuntimeError(result.error or result.output)
        callback = self._output_callback()
        if callback:
            await callback(answer.stdout or "", "stdout")
            await callback(answer.stderr or "", "stderr")
        return answer

    async def _checked(self, command: str) -> None:
        result = await self.exec(command, cwd="/", user="root", timeout_sec=300)
        if result.return_code:
            raise RuntimeError(result.stderr or result.stdout)

    # --- files ---------------------------------------------------------------

    async def upload_file(self, source_path: Path | str, target_path: str) -> None:
        await self._checked(f"mkdir -p {shlex.quote(str(Path(target_path).parent))}")
        if not await asyncio.to_thread(self._running_session().upload_file, source_path, target_path):
            raise RuntimeError(f"Failed to upload {source_path}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        if not await asyncio.to_thread(self._running_session().download_file, source_path, target_path):
            raise RuntimeError(f"Failed to download {source_path}")

    async def upload_dir(self, source_dir: Path | str, target_dir: str) -> None:
        remote = f"/tmp/sprout-harbor-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "files.tar.gz"
            with tarfile.open(archive, "w:gz") as stream:
                stream.add(source_dir, arcname=".")
            try:
                await self.upload_file(archive, remote)
                await self._checked(
                    f"mkdir -p {shlex.quote(target_dir)} && tar -xzf {remote} -C {shlex.quote(target_dir)}"
                )
            finally:
                await self._checked(f"rm -f {remote}")

    async def download_dir(self, source_dir: str, target_dir: Path | str) -> None:
        remote = f"/tmp/sprout-harbor-{uuid4().hex}.tar.gz"
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary) / "files.tar.gz"
            try:
                await self._checked(f"tar -czf {remote} -C {shlex.quote(source_dir)} .")
                await self.download_file(remote, archive)
                Path(target_dir).mkdir(parents=True, exist_ok=True)
                with tarfile.open(archive, "r:gz") as stream:
                    stream.extractall(target_dir, filter="data")
            finally:
                await self._checked(f"rm -f {remote}")

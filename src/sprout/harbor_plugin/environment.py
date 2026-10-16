"""A Harbor environment that is a Sprout sandbox: what every backend's shares.

Harbor asks an environment for a handful of things -- start, exec, upload,
download, a network policy, stop -- and Sprout answers every one of them
through a :class:`~sprout.worker.sandbox.session.SandboxSession`, whichever
backend the session's config names. :class:`SandboxEnvironment` is that
answer; a backend subclass (``agentenv.AgentENVEnvironment`` for a microVM,
``podman.PodmanEnvironment`` for a rootless container) adds only what differs:
how the task image is pinned and its OCI config read, the backend section the
session is built from, the shape a sandbox gets, and how egress is toggled.

The agent classes (``agentenv_agent``, ``agentenv_mini``, ``branching``) are
written against this class alone: the session they borrow, the backend dict a
branch inherits, the pinned image, the image's user and working directory,
``shell_context()``, ``retain_reason``, ``checkpoint_mode`` and
``actor_control``. Nothing in them names a backend.
"""

from __future__ import annotations

import asyncio
import json
import re
import shlex
import tarfile
import tempfile
from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

from harbor.environments.base import BaseEnvironment, ExecResult
from harbor.environments.capabilities import EnvironmentCapabilities, EnvironmentResourceCapabilities
from harbor.models.task.config import NetworkMode

from sprout.worker.sandbox.backends import backend_name
from sprout.worker.sandbox.session import SandboxSession
from sprout.worker.tools.interceptors import shell_command

CHECKPOINT_MODES = ("disk_only", "none")

_INSTRUCTION = re.compile(r"^([A-Za-z]+)\s+(.*)$")


# --- tests/Dockerfile as an overlay on the task image -----------------------------


@dataclass(frozen=True)
class ImageOverlay:
    """A Dockerfile that only adds files and commands on top of the task image.

    Harbor builds a separate verifier from ``tests/Dockerfile``. Sprout boots
    OCI images, not Dockerfiles, so the same result is produced by booting the
    FROM image and replaying the COPY and RUN steps in it -- no registry and no
    build. Anything a replay could not honour faithfully (ENV, WORKDIR, USER, a
    second stage) is refused, so the verifier never runs in an environment
    that differs from the one the task author built.
    """

    base: str
    copies: tuple[tuple[Path, str], ...]
    runs: tuple[str, ...]


def dockerfile_base(path: Path) -> str | None:
    for line in _dockerfile_lines(path.read_text()):
        word, rest = line.split(None, 1) if " " in line else (line, "")
        if word.upper() == "FROM":
            return rest.split()[0]
    return None


def _dockerfile_lines(text: str) -> list[str]:
    lines, pending = [], ""
    for raw in text.splitlines():
        stripped = raw.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.endswith("\\"):
            pending += stripped[:-1] + " "
            continue
        lines.append((pending + stripped).strip())
        pending = ""
    if pending:
        lines.append(pending.strip())
    return lines


def parse_overlay(path: Path) -> ImageOverlay:
    base, copies, runs = None, [], []
    for line in _dockerfile_lines(path.read_text()):
        match = _INSTRUCTION.match(line)
        word, rest = (match.group(1).upper(), match.group(2)) if match else (line.upper(), "")
        if word == "FROM":
            if base is not None:
                raise ValueError(f"{path}: multi-stage Dockerfiles are not replayed")
            base = rest.split()[0]
        elif word == "COPY":
            parts = shlex.split(rest)
            if len(parts) != 2 or parts[0].startswith("--"):
                raise ValueError(f"{path}: only `COPY <src> <dst>` is replayed, got {line!r}")
            source = path.parent / parts[0]
            if not source.exists():
                raise ValueError(f"{path}: COPY source {parts[0]} is missing")
            copies.append((source, parts[1]))
        elif word == "RUN":
            runs.append(rest)
        else:
            raise ValueError(
                f"{path}: {word} is not replayed in a sandbox; refusing to "
                f"verify in an environment that differs from the task's"
            )
    if base is None:
        raise ValueError(f"{path}: no FROM")
    return ImageOverlay(base, tuple(copies), tuple(runs))


def overlay_for(environment_dir: Path, docker_image: str | None) -> ImageOverlay | None:
    """The overlay the directory's Dockerfile describes, if a sandbox can replay it.

    With a ``docker_image``, a Dockerfile FROM another image is the recipe that
    produced that prebuilt image (DeepSWE's ``environment/Dockerfile``): the image
    already holds its result, so it is not replayed; a Dockerfile FROM the image
    itself must be replayable, or the verifier would run in an environment that
    differs from the task's. Without a ``docker_image`` (Harbor's separate
    verifier config for DeepSWE names only resources), a replayable Dockerfile
    is booted from its FROM image; anything else would need a build, which
    Sprout does not do.
    """
    path = environment_dir / "Dockerfile"
    if not path.is_file():
        return None
    if docker_image and dockerfile_base(path) != docker_image:
        return None
    try:
        return parse_overlay(path)
    except ValueError:
        if docker_image:
            raise
        return None


# --- the environment ------------------------------------------------------------


class SandboxEnvironment(BaseEnvironment):
    """Harbor's environment over one :class:`SandboxSession`; a backend subclass picks the pool.

    A subclass provides ``type()``, ``_backend_section()``, ``_prepare_image()``,
    ``_resources()`` and ``_set_network()``, and extends ``_validate_definition``
    with the limits its backend has. Everything Harbor calls is here.
    """

    def __init__(
        self, *args, runtime_bin: str, checkpoint_mode: str = "disk_only", exec_timeout_s: int, **kwargs
    ):
        self.runtime_bin = str(Path(runtime_bin).resolve())
        self.checkpoint_mode = checkpoint_mode
        #: The default bound on a command Harbor runs without a timeout of its own.
        self.exec_timeout_s = exec_timeout_s
        self.session: SandboxSession | None = None
        self.image_user = "root"
        self.image_workdir = "/"
        self.image_env: dict[str, str] = {}
        self.image_config: dict = {}
        self.backend: dict = {}
        self.image = ""
        self.actor_control = None
        #: Set while the sandbox holds state nothing else has: ``stop()`` then
        #: releases the handle instead of destroying the sandbox (``harness
        #: reap`` remains), and records why.
        self.retain_reason: str | None = None
        self.overlay: ImageOverlay | None = None
        super().__init__(*args, **kwargs)

    @property
    def capabilities(self) -> EnvironmentCapabilities:
        return EnvironmentCapabilities(disable_internet=True, dynamic_network_policy=True)

    @classmethod
    def resource_capabilities(cls) -> EnvironmentResourceCapabilities:
        return EnvironmentResourceCapabilities(cpu_limit=True, memory_limit=True)

    # --- what a backend decides ----------------------------------------------

    def _backend_section(self) -> dict:
        """The session's backend config: ``{"backend": <name>, <name>: {...}}``."""
        raise NotImplementedError

    async def _prepare_image(self, force_build: bool) -> tuple[str, dict]:
        """The image to create the sandbox from, digest-pinned, and its OCI config."""
        raise NotImplementedError

    def _resources(self) -> dict:
        """The sandbox's shape for :meth:`SandboxSession.create`."""
        raise NotImplementedError

    async def _set_network(self, allow: bool) -> None:
        """Give the running sandbox egress or take it away."""
        raise NotImplementedError

    # --- admission -----------------------------------------------------------

    def _validate_definition(self) -> None:
        if not Path(self.runtime_bin).is_file():
            raise ValueError(f"Missing sprout-runtime: {self.runtime_bin}")
        if self.checkpoint_mode not in CHECKPOINT_MODES:
            raise ValueError("checkpoint_mode must be disk_only or none")
        for name in ("docker-compose.yaml", "docker-compose.yml", "compose.yaml", "compose.yml"):
            if (self.environment_dir / name).exists():
                raise ValueError(
                    f"{self.type()} multi-service composition is not implemented; no Docker fallback"
                )
        if self.task_env_config.mcp_servers:
            raise ValueError(f"Task-side MCP networking requires a {self.type()} service adapter")
        if not self.task_env_config.docker_image and not (self.environment_dir / "Dockerfile").is_file():
            raise ValueError("Task needs a docker_image or environment/Dockerfile")
        self.overlay = overlay_for(self.environment_dir, self.task_env_config.docker_image)  # refuses early
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

    async def _apply_network_policy(self, network_policy) -> None:
        if network_policy.network_mode not in {NetworkMode.PUBLIC, NetworkMode.NO_NETWORK}:
            raise ValueError(f"{self.type()} supports only public and no-network policies")
        if self.session is None or not self.session.sandbox_id:
            raise RuntimeError("Cannot update network policy before sandbox start")
        await self._set_network(network_policy.network_mode == NetworkMode.PUBLIC)

    @property
    def _allow_internet(self) -> bool:
        """What the sandbox starts with: the baseline policy Harbor handed over."""
        return self._network_policy.network_mode == NetworkMode.PUBLIC

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

    def _write_receipt(self, resources: dict) -> None:
        """``<session_id>.<type>.json`` in the trial directory: what this sandbox is.

        ``methods.evaluate`` reads it back to tie a graded trial to its sandbox
        and to hand a branch the image and config its parent ran on.
        """
        record = {
            "backend": backend_name(self.backend),
            "sandbox_id": self.session.sandbox_id,
            "image": self.image,
            "image_config": self.image_config,
            "requested_disk_size_mb": self._effective_storage_mb,
            "resources": resources,
            "network": self._network_policy.model_dump(mode="json"),
            "checkpoint_mode": self.checkpoint_mode,
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

    def _receipt_extra(self) -> dict:
        """What a backend adds to the receipt beyond the common fields."""
        return {}

    async def _populate(self) -> None:
        """Put the task's files in the fresh sandbox: an overlay's COPY/RUN, or environment/."""
        if self.overlay is not None:
            for source, destination in self.overlay.copies:
                if source.is_dir():
                    await self.upload_dir(source, destination)
                else:
                    await self.upload_file(source, destination)
            for command in self.overlay.runs:
                result = await self.exec(
                    command,
                    cwd=self.image_workdir,
                    user="root",
                    timeout_sec=int(self.task_env_config.build_timeout_sec),
                )
                if result.return_code:
                    raise RuntimeError(f"overlay RUN failed ({command}): {result.stderr or result.stdout}")
            return
        await self._upload_environment_dir_after_start()

    async def stop(self, delete: bool = True) -> None:
        if self.actor_control is not None:
            self.actor_control.request_stop("Harbor is stopping this trial")
        if self.session is None:
            return
        session, pool = self.session, self.session._pool
        if delete and self.retain_reason:
            # The sandbox holds state nothing else has (its final snapshot
            # failed). Release the handle; destroying it would lose the only copy.
            (self.trial_paths.trial_dir / f"{self.session_id}.retained-sandbox.json").write_text(
                json.dumps({"sandbox_id": session.sandbox_id, "reason": self.retain_reason}, indent=2)
            )
            delete = False
        try:
            # A kept sandbox is released from the pool before the pool closes:
            # closing destroys every sandbox the pool still holds. A destroy
            # that fails half-way still closes the pool; the error names what
            # is left and how `harness reap` reclaims it.
            await asyncio.to_thread(
                session._loop.drive, session._destroy_async() if delete else session._release_async()
            )
        finally:
            if pool is not None:
                await asyncio.to_thread(session._loop.drive, pool.close())
            self.session = None

    # --- commands ------------------------------------------------------------

    def shell_context(self) -> dict:
        """The interceptor declaration that runs the agent's commands the way Harbor would."""
        user = self._resolve_user(None)
        return {
            "name": "shell_context",
            "cwd": self.task_env_config.workdir or self.image_workdir,
            "env": {**self.image_env, **(self._merge_env(None) or {})},
            "user": self.image_user if user is None else user,
        }

    def command_args(
        self, command: str, cwd: str | None = None, env: dict | None = None, user: str | int | None = None
    ) -> dict:
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
        env: dict | None = None,
        timeout_sec: int | None = None,
        user: str | int | None = None,
    ) -> ExecResult:
        if self.session is None:
            raise RuntimeError(f"{self.type()} environment has not started")
        timeout = timeout_sec or self.exec_timeout_s
        arguments = {
            **self.command_args(command, cwd, env, user),
            "timeout": timeout,
            "max_output_bytes": 16 * 1024 * 1024,
        }
        pending = asyncio.create_task(
            asyncio.to_thread(self.session.execute, "shell", arguments, timeout + 60)
        )
        try:
            result = await asyncio.shield(pending)
        except asyncio.CancelledError:
            loop = self.session._loop.get()
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
        if not await asyncio.to_thread(self.session.upload_file, source_path, target_path):
            raise RuntimeError(f"Failed to upload {source_path}")

    async def download_file(self, source_path: str, target_path: Path | str) -> None:
        Path(target_path).parent.mkdir(parents=True, exist_ok=True)
        if not await asyncio.to_thread(self.session.download_file, source_path, target_path):
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

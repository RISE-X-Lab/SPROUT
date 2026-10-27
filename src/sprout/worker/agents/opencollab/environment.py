"""OpenCollab's environment port over Sprout ToolAccess, never host execution."""

from __future__ import annotations

import asyncio
import base64
import hashlib
import math
import posixpath
import shlex
import time
from contextlib import suppress
from typing import Any
from uuid import uuid4

from opencollab.adapters._env_base import ENV_FILE_WRITE_LIMIT_BYTES
from opencollab.adapters._env_docker import DockerEnvironment
from opencollab.adapters.env import Environment, ExecResult

from sprout.worker.run.control import RunAborted


class SproutEnvironment(Environment):  # type: ignore[misc]  # SDK base has no published typing.
    process_isolated = True
    # These implementations use exec_cmd only, and retain native output limits.
    read_text_range = DockerEnvironment.read_text_range
    write_temp_file = DockerEnvironment.write_temp_file
    remove_file = DockerEnvironment.remove_file

    def __init__(
        self, tools: Any, workspace: Any, control: Any, deadline: Any, *, image_shell: Any = None
    ) -> None:
        super().__init__()
        self.tools, self.workspace = tools, workspace
        self.control, self.deadline = control, deadline
        self.image_shell = image_shell
        self.turn_id: str | None = None
        self.call_ids: list[str] = []
        self.failure: BaseException | None = None
        self._temporary_files: set[str] = set()

    async def exec_cmd(self, cmd: str, timeout: float = 120.0) -> ExecResult:
        self._ensure_active()
        self.control.raise_if_stopped()
        if not self.turn_id:
            raise RuntimeError("OpenCollab execution requires an active model turn")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("command timeout must be positive and finite")
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            self.control.request_stop("OpenCollab deadline exhausted", stop_reason="timeout")
            self.control.raise_if_stopped()
        identifier = uuid4().hex
        arguments = {
            "command": "bash -c " + shlex.quote(cmd),
            "working_dir": self.workspace,
            "timeout": max(1, math.ceil(min(timeout, remaining))),
        }
        if self.image_shell is not None:
            arguments["command"] = self.image_shell.command(arguments["command"], working_dir=self.workspace)
        worker = asyncio.create_task(
            asyncio.to_thread(
                self.tools.call,
                "shell",
                arguments,
                call_id=identifier,
                turn_id=self.turn_id,
            )
        )
        try:
            # Shield the serialized runtime operation until its own timeout settles it.
            result = await asyncio.shield(worker)
            self.call_ids.append(identifier)
            outcome = result.outcome
            if outcome is None or type(outcome.exit_code) is not int or outcome.running:
                raise RuntimeError("OpenCollab command did not return a settled outcome")
            self.control.raise_if_stopped()
            return ExecResult(
                outcome.exit_code,
                outcome.stdout,
                outcome.stderr,
                stdout_truncated=outcome.stdout_truncated,
                stderr_truncated=outcome.stderr_truncated,
                timed_out=outcome.timed_out,
            )
        except asyncio.CancelledError:
            # Drain a cancelled operation before another model turn may begin.
            while not worker.done():
                try:
                    await asyncio.shield(worker)
                except asyncio.CancelledError:
                    continue
                except Exception:
                    break
            try:
                worker.result()
            except Exception as error:  # the cancelled call really lost transport
                self.failure = error
            else:
                self.call_ids.append(identifier)
            raise
        except RunAborted:
            # A rollout deadline or controller stop that arrives after the
            # command settled does not make the sandbox transport uncertain.
            raise
        except BaseException as error:
            # OpenCollab's tools can convert exceptions to feedback. Retain the
            # transport failure so that cannot turn it into a benchmark failure.
            self.failure = error
            raise

    async def write_file(self, path: str, content: str) -> None:
        raw = content.encode("utf-8")
        if len(raw) > ENV_FILE_WRITE_LIMIT_BYTES:
            raise ValueError("file exceeds the native OpenCollab write limit")
        target = posixpath.normpath(path)
        if not target.startswith("/") or target == "/":
            raise ValueError("file path must be absolute and non-root")
        parent, quoted = shlex.quote(posixpath.dirname(target)), shlex.quote(target)
        temporary = shlex.quote(
            posixpath.join(
                posixpath.dirname(target),
                ".sprout-write-" + uuid4().hex,
            )
        )

        async def checked(command: str, phase: str) -> None:
            result = await self.exec_cmd(command)
            if result.returncode != 0:
                raise OSError(
                    f"sandbox file write failed ({phase}, exit={result.returncode}): " + result.stderr
                )

        # Bound each shell argument below Linux execve limits, even for large files.
        owned = False
        try:
            await checked(
                f"mkdir -p -- {parent} && (umask 077; set -C; : > {temporary})",
                "create stage",
            )
            owned = True
            for offset in range(0, len(raw), 32 * 1024):
                payload = base64.b64encode(raw[offset : offset + 32 * 1024]).decode("ascii")
                await checked(
                    f"printf '%s' {shlex.quote(payload)} | base64 -d >> {temporary}",
                    "append chunk",
                )
            digest = hashlib.sha256(raw).hexdigest()
            await checked(
                f'test "$(wc -c < {temporary})" -eq {len(raw)} && '
                f"test \"$(sha256sum -- {temporary} | cut -d ' ' -f 1)\" = {digest} && "
                f"if [ -f {quoted} ]; then chmod --reference={quoted} {temporary}; fi && "
                f"mv -f -- {temporary} {quoted}",
                "verify and commit",
            )
            owned = False
        finally:
            # Never mask cancellation or a transport failure; never bypass a
            # stopped rollout to clean up. Harbor ultimately owns the sandbox.
            if owned and self.failure is None:
                with suppress(Exception, asyncio.CancelledError):
                    await self.exec_cmd(f"rm -f -- {temporary}")

    async def read_file(self, path: str) -> str:
        """Read through ToolAccess without borrowing private Docker transport."""
        result = await self.exec_cmd(f"cat -- {shlex.quote(path)}")
        if result.returncode != 0:
            raise FileNotFoundError(result.stderr)
        if result.stdout_truncated:
            raise OSError(f"sandbox read exceeded capture limit for {path}")
        return str(result.stdout)

    async def cleanup(self) -> None:
        # Harbor owns the VM. Never delete it, or a restore point, here.
        pass

"""``MicroVMPool``, the microVMs of an AgentENV server, and ``Snapshot``, which every pool returns."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

from .backends import GatewayBackend
from .sandbox import Sandbox


@dataclass(frozen=True)
class Snapshot:
    """A persistent snapshot, with the layer counts that a caller needs to manage its chain.

    ``rootfs_layers`` is inherited: a sandbox started from this snapshot can add layers to the chain,
    but can never compact the layers that it inherited. ``None`` means that the backend does not say,
    or does not stack layers."""

    id: str
    rootfs_layers: int | None = None
    memory_layers: int | None = None
    #: True when this snapshot carries no memory image, so sandboxes created
    #: from it cold-boot instead of resuming.
    disk_only: bool = False
    #: Whether the server reports this snapshot's delta as empty. ``None`` when the server does not say.
    delta_empty: bool | None = None

    @classmethod
    def from_api(cls, body: dict[str, Any], *, disk_only: bool = False) -> Snapshot:
        """The snapshot that an AgentENV snapshot or squash response ``body`` describes."""
        return cls(
            id=body["snapshotID"],
            rootfs_layers=body.get("rootfsLayerCount"),
            memory_layers=body.get("memoryLayerCount"),
            disk_only=disk_only,
            delta_empty=body.get("deltaEmpty") if isinstance(body.get("deltaEmpty"), bool) else None,
        )


class MicroVMPool:
    """Sandboxes as Firecracker microVMs, provisioned by an AgentENV server over its HTTP API.

    The template or image must already start sprout-runtime on ``runtime_port``. Calls reach it through
    AgentENV's proxy. A disk snapshot takes tens of milliseconds, and a VM can start from any snapshot."""

    #: AgentENV's proxy routing headers (it also accepts E2B-compatible aliases).
    SANDBOX_ID_HEADER = "x-agentenv-sandbox-id"
    TARGET_PORT_HEADER = "x-agentenv-target-port"
    #: The port of AgentENV's file service in the guest, reached through the same proxy. It does not use
    #: the runtime, so it can install the runtime, or stage binary files that a tool call cannot carry.
    ENVD_PORT = "49983"

    def __init__(
        self,
        server_url: str,
        runtime_port: int = 3000,
        api_key: str = "",
        request_timeout: float = 120,
        sandbox_ttl: int = 300,
        allow_internet: bool | None = None,
    ) -> None:
        """A pool on the AgentENV server at ``server_url``, such as ``http://127.0.0.1:8000``.

        ``runtime_port`` is the runtime's port in each VM, and ``sandbox_ttl`` each VM's time to live, in
        seconds. ``allow_internet=False`` denies all egress from each VM that the pool creates, and
        ``None`` keeps the server's default. The host reaches the runtime in either case."""
        self.server_url = server_url.rstrip("/")
        self.runtime_port = runtime_port
        # Every create sends the TTL. AgentENV's own default pauses the VM after 15 s, which is too
        # short for an agent turn.
        self.sandbox_ttl = sandbox_ttl
        self.allow_internet = allow_internet
        # The server checks X-API-KEY. It does not accept a Bearer Authorization header.
        headers = {"X-API-KEY": api_key} if api_key else {}
        self._client = httpx.AsyncClient(timeout=request_timeout, headers=headers)
        #: Live sandboxes by id.
        self._sandboxes: dict[str, Sandbox] = {}

    # --- Capabilities ---
    # The session asks these before each optional operation. Its test doubles can answer no.

    def supports_snapshot(self) -> bool:
        return True

    def supports_cold_start(self) -> bool:
        return True

    def supports_upload(self) -> bool:
        return True

    def supports_download(self) -> bool:
        return True

    # --- Host files ---

    async def upload_file(
        self, sandbox: Sandbox, source: str | Path, destination: str, timeout: float | None = None
    ) -> None:
        """Copy the host file ``source`` to ``destination`` in the sandbox, without the runtime.

        The tool surface cannot carry a fixture: ``shell`` takes a command, ``text_editor`` writes text,
        and fixtures are often binary and several megabytes. Raises ``httpx.HTTPError`` on a failure."""
        source = Path(source)
        sid = sandbox.require_id()
        headers = {self.SANDBOX_ID_HEADER: sid, self.TARGET_PORT_HEADER: self.ENVD_PORT}
        with open(source, "rb") as handle:
            resp = await self._client.post(
                f"{self.server_url}/files",
                params={"path": destination},
                headers=headers,
                files={"file": (source.name, handle, "application/octet-stream")},
                # Not the client's default: these are megabytes over a proxy, and the
                # client's default suits short control calls.
                timeout=timeout or 300.0,
            )
        resp.raise_for_status()

    async def download_file(
        self, sandbox: Sandbox, source: str, destination: str | Path, timeout: float | None = None
    ) -> None:
        """Copy the guest file ``source`` to the host path ``destination``, without the runtime.

        ``destination`` changes only after the whole file has arrived. Raises ``httpx.HTTPError`` on a
        failure."""
        import tempfile

        destination = Path(destination)
        headers = {self.SANDBOX_ID_HEADER: sandbox.require_id(), self.TARGET_PORT_HEADER: self.ENVD_PORT}
        temporary = None
        try:
            async with self._client.stream(
                "GET",
                f"{self.server_url}/files",
                params={"path": source},
                headers=headers,
                timeout=timeout or 120.0,
            ) as response:
                response.raise_for_status()
                with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as handle:
                    temporary = Path(handle.name)
                    async for chunk in response.aiter_bytes():
                        handle.write(chunk)
                temporary.replace(destination)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)

    # --- Lifecycle ---

    def list(self) -> list[Sandbox]:
        """Sandboxes this pool currently holds."""
        return list(self._sandboxes.values())

    async def spawn(self, image: str, env: dict[str, str] | None = None, agent_id: str = "") -> Sandbox:
        """Start a microVM from the template or snapshot id ``image``, and return a registered handle.

        An OCI image reference goes to :meth:`spawn_from_image` instead. The template or the snapshot
        fixes the VM's CPUs, memory and disk. Every call through the handle is attributed to
        ``agent_id``. Raises ``ValueError`` if ``image`` is empty."""
        if not image:
            raise ValueError("spawn needs a template or snapshot id")
        payload: dict[str, Any] = {
            "templateID": image,
            "timeout": self.sandbox_ttl,
            "autoPause": True,
            # AgentENV resumes a sandbox that paused at TTL expiry on its next proxied call. A sandbox
            # that is kept for a grade after the run needs this.
            "autoResume": {"enabled": True},
        }
        if env:
            payload["envVars"] = env
        if self.allow_internet is not None:
            # AgentENV 758d575 reads only snake_case on POST /sandboxes, only camelCase on /sandboxes-cold,
            # and ignores unknown keys. Drop snake_case once its NewSandbox model has the camelCase rename.
            payload["allowInternetAccess"] = self.allow_internet
            payload["allow_internet_access"] = self.allow_internet

        resp = await self._client.post(f"{self.server_url}/sandboxes", json=payload)
        resp.raise_for_status()
        return self.attach(_sandbox_id(resp.json()), agent_id)

    async def spawn_from_image(
        self, image: str, agent_id: str = "", resources: dict[str, Any] | None = None
    ) -> Sandbox:
        """Cold-start a microVM from the OCI image reference ``image``, and return a registered handle.

        The image must contain the runtime, started by the template's startup command. ``resources``
        can set ``cpu``, ``memory_mb`` and ``disk_size_mb``. A cache miss can take tens of seconds. The
        pool does not pin a tag, so a caller that records the image pins its digest first."""
        payload: dict[str, Any] = {
            "image": image,
            "timeout": self.sandbox_ttl,
            "autoPause": True,
            "autoResume": {"enabled": True},
        }
        if resources:
            if "cpu" in resources:
                payload["cpuCount"] = int(resources["cpu"])
            if "memory_mb" in resources:
                payload["memoryMB"] = int(resources["memory_mb"])
            if "disk_size_mb" in resources:
                payload["diskSizeMB"] = int(resources["disk_size_mb"])
        if self.allow_internet is not None:
            payload["allowInternetAccess"] = self.allow_internet
        resp = await self._client.post(f"{self.server_url}/sandboxes-cold", json=payload)
        resp.raise_for_status()
        return self.attach(_sandbox_id(resp.json()), agent_id)

    async def destroy(self, *sandboxes: Sandbox) -> None:
        """Delete each of ``sandboxes`` that this pool holds, and drop its handle.

        A sandbox that the server no longer has is not an error."""
        for sb in sandboxes:
            sid = sb._container_id
            if not sid or sid not in self._sandboxes:
                continue
            response = await self._client.delete(f"{self.server_url}/sandboxes/{sid}")
            if response.status_code != 404:
                response.raise_for_status()
            del self._sandboxes[sid]
            sb._container_id = None

    async def forget(self, *sandboxes: Sandbox) -> None:
        """Drop the handles on ``sandboxes`` and leave their VMs running. ``close`` then leaves them alone.

        This is for a sandbox that must outlive its pool, such as one whose final snapshot failed and that
        so holds the only copy of the rollout. ``python -m sprout.worker reap`` or a person removes it."""
        for sb in sandboxes:
            sid = sb._container_id
            if sid and sid in self._sandboxes:
                del self._sandboxes[sid]
                sb._container_id = None
                await sb.backend.close()

    async def destroy_all(self) -> None:
        """Tear down everything this pool created."""
        await self.destroy(*self.list())
        self._sandboxes.clear()

    async def close(self) -> None:
        """Destroy every sandbox that the pool holds, then close its HTTP client."""
        await self.destroy_all()
        await self._client.aclose()

    # --- Snapshots ---

    async def snapshot(
        self, sandbox: Sandbox, *, name: str | None = None, disk_only: bool = False
    ) -> Snapshot:
        """Publish the sandbox's current state as a snapshot. The sandbox keeps running.

        ``spawn(image=<snapshot id>)`` starts a new sandbox from it. ``disk_only`` leaves out the VM state
        and memory, so the cost is about the bytes written since the previous snapshot. A sandbox from a
        disk-only snapshot cold-boots: the template's startup command runs again, and no process is left."""
        sid = sandbox.require_id()
        payload: dict[str, Any] = {}
        if name:
            payload["name"] = name
        if disk_only:
            payload["diskOnly"] = True
        resp = await self._client.post(f"{self.server_url}/sandboxes/{sid}/snapshots", json=payload)
        resp.raise_for_status()
        return Snapshot.from_api(resp.json(), disk_only=disk_only)

    async def squash(self, snapshot: Snapshot | str, *, name: str | None = None) -> Snapshot:
        """A one-layer snapshot that is equivalent to ``snapshot``.

        The source snapshot does not change and stays usable. A chain that is already flat comes back
        with the same id, so a call is always safe. A bare id gives a result that is not ``disk_only``."""
        snapshot_id = snapshot.id if isinstance(snapshot, Snapshot) else snapshot
        disk_only = snapshot.disk_only if isinstance(snapshot, Snapshot) else False
        payload: dict[str, Any] = {}
        if name:
            payload["name"] = name
        resp = await self._client.post(f"{self.server_url}/snapshots/{snapshot_id}/squash", json=payload)
        resp.raise_for_status()
        return Snapshot.from_api(resp.json(), disk_only=disk_only)

    # --- Handles ---

    def handle(self, sandbox_id: str, agent_id: str = "") -> Sandbox:
        """A new handle on the existing sandbox ``sandbox_id``, which the pool does not register.

        A handle's HTTP client binds to the first event loop that uses it. A caller on another event
        loop takes its own handle here."""
        sb = Sandbox(
            backend=GatewayBackend(
                self.server_url,
                sandbox_id,
                sandbox_id_header=self.SANDBOX_ID_HEADER,
                target_port=self.runtime_port,
                target_port_header=self.TARGET_PORT_HEADER,
            ),
            agent_id=agent_id,
        )
        sb._container_id = sandbox_id
        return sb

    def attach(self, sandbox_id: str, agent_id: str = "") -> Sandbox:
        """A registered handle on the AgentENV sandbox ``sandbox_id``. The pool destroys it on ``close``.

        A process that created a sandbox can lend it to another process. That process attaches to it to
        serve tool calls, while the owner still takes its snapshots."""
        sb = self.handle(sandbox_id, agent_id=agent_id)
        self._sandboxes[sandbox_id] = sb
        return sb


def _sandbox_id(body: dict[str, Any]) -> str:
    """The sandbox id an AgentENV create response reports."""
    sid = body.get("sandboxID")
    if not sid:
        raise RuntimeError(f"no sandbox id in response: {body}")
    return str(sid)

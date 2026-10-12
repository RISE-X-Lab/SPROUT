from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .backends import GatewayBackend
from .sandbox import Sandbox


@dataclass(frozen=True)
class Snapshot:
    """A persistent snapshot, plus the facts needed to manage its chain.

    A snapshot is a stack of layers: the base image, then one layer per
    capture. `rootfs_layers` matters to callers because it is *inherited* --
    a sandbox started from this snapshot can never compact those layers, only
    add to them. Two decisions read it:

    - the count dropping between consecutive snapshots of one sandbox means
      the server compacted the chain, so that snapshot is a compact base and
      the sandbox should be replaced by one started from it (`re-board`);
    - a high count at branch time means children would inherit a deep prefix,
      so the snapshot is worth squashing first.
    """

    id: str
    rootfs_layers: int | None = None
    memory_layers: int | None = None
    #: True when this snapshot carries no memory image, so sandboxes created
    #: from it cold-boot instead of resuming.
    disk_only: bool = False
    delta_empty: bool | None = None

    @classmethod
    def from_api(cls, body: dict[str, Any], *, disk_only: bool = False) -> "Snapshot":
        return cls(
            id=body["snapshotID"],
            rootfs_layers=body.get("rootfsLayerCount"),
            memory_layers=body.get("memoryLayerCount"),
            disk_only=disk_only,
            delta_empty=body.get("deltaEmpty") if isinstance(body.get("deltaEmpty"), bool) else None,
        )


class MicroVMPool:
    """Sandboxes as Firecracker microVMs, provisioned by an AgentENV server.

    Sprout does not manage Firecracker itself; that is an entire storage stack
    (layered block devices, memory snapshots) and AgentENV already provides it
    behind an HTTP API. This pool is a client of that API: it asks for VMs from
    a template or snapshot (or cold-starts them from an OCI image) and reaches
    the runtime inside them through AgentENV's per-node proxy, which routes on
    a sandbox id and a target port.

    The template image must already contain sprout-runtime, started on
    `runtime_port`. That is why the runtime is a single static binary with no
    dependencies: it drops into any OCI image.

    What this buys is the state operations: publishing a running VM's disk as a
    snapshot takes tens of milliseconds, and a fresh VM starts from any snapshot.

    Reaching the runtime only through a proxy has one consequence worth knowing:
    AgentENV gives an upstream 30 s to produce response headers
    (`response_header_timeout_ms`), and a tool call produces none until its
    command finishes. Runtimes from before mid-2026 therefore lose any call
    longer than that -- a 504 with the command still running inside the VM, so
    the work happens and the result is discarded. Current runtimes send headers
    up front and heartbeat while the command runs, which keeps the connection
    alive for as long as the call takes; there is nothing to configure.
    """

    #: AgentENV's proxy routing headers (it also accepts E2B-compatible aliases).
    SANDBOX_ID_HEADER = "x-agentenv-sandbox-id"
    TARGET_PORT_HEADER = "x-agentenv-target-port"
    #: The backend's own file service inside the guest. Reached through the
    #: same proxy as the runtime, with a different target port -- which is the
    #: point: it does not involve the runtime, so it can put files in a
    #: sandbox whose runtime is not the thing being talked to (installing the
    #: runtime itself, staging binary fixtures a tool call cannot carry).
    ENVD_PORT = "49983"

    def __init__(self, server_url: str, runtime_port: int = 3000, api_key: str = "",
                 request_timeout: float = 120, sandbox_ttl: int = 300,
                 allow_internet: "bool | None" = None):
        """
        Args:
            server_url: AgentENV server, e.g. "http://127.0.0.1:8000".
            allow_internet: Sent as ``allowInternetAccess`` on every sandbox
                this pool creates (template spawn and cold start alike).
                ``False`` denies all egress from the VM -- the host still
                reaches the runtime, since that is inbound. ``None`` leaves
                the server's default (internet on). A benchmark that declares
                ``network_mode = "no-network"`` needs ``False``; nothing in
                the payload can be inferred, so it is explicit here.
            runtime_port: Port sprout-runtime listens on inside the VM.
            api_key: AgentENV API key. Sent as `X-API-KEY`; that is the header
                the server validates (a Bearer Authorization header is not).
            sandbox_ttl: Sandbox time-to-live in seconds. AgentENV's own
                default is 15s, after which the VM auto-pauses -- far too
                short for an agent turn, so every create sends this instead.
        """
        self.server_url = server_url.rstrip("/")
        self.runtime_port = runtime_port
        self.sandbox_ttl = sandbox_ttl
        self.allow_internet = allow_internet
        headers = {"X-API-KEY": api_key} if api_key else {}
        self._client = httpx.AsyncClient(timeout=request_timeout, headers=headers)
        #: Live sandboxes by id.
        self._sandboxes: dict[str, Sandbox] = {}

    # --- Capabilities ---
    #
    # The session asks before each optional operation instead of assuming, and
    # its test doubles answer differently; the checkpointer treats a session
    # that cannot snapshot as one with nothing to record.

    def supports_snapshot(self) -> bool:
        return True

    def supports_cold_start(self) -> bool:
        return True

    def supports_upload(self) -> bool:
        return True

    def supports_download(self) -> bool:
        return True

    # --- Host files ---

    async def upload_file(self, sandbox: Sandbox, source, destination: str,
                          timeout: float | None = None) -> None:
        """Copy a host file into the sandbox, bypassing the runtime.

        The tool surface cannot carry this: `shell` takes a command, not bytes,
        and fixtures are routinely binary and multi-megabyte. The backend's file
        service takes the bytes directly.
        """
        from pathlib import Path
        source = Path(source)
        sid = sandbox.require_id()
        headers = {self.SANDBOX_ID_HEADER: sid,
                   self.TARGET_PORT_HEADER: self.ENVD_PORT}
        with open(source, "rb") as handle:
            resp = await self._client.post(
                f"{self.server_url}/files", params={"path": destination},
                headers=headers,
                files={"file": (source.name, handle,
                                "application/octet-stream")},
                # Not the client's default: these are megabytes over a
                # proxy, and the client is tuned for short control calls.
                timeout=timeout or 300.0)
        resp.raise_for_status()

    async def download_file(self, sandbox: Sandbox, source: str, destination,
                            timeout: float | None = None) -> None:
        """Stream a guest file to an atomic host destination, outside the runtime."""
        from pathlib import Path
        import tempfile

        destination = Path(destination)
        headers = {self.SANDBOX_ID_HEADER: sandbox.require_id(),
                   self.TARGET_PORT_HEADER: self.ENVD_PORT}
        temporary = None
        try:
            async with self._client.stream(
                    "GET", f"{self.server_url}/files", params={"path": source},
                    headers=headers, timeout=timeout or 120.0) as response:
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

    async def spawn(self, image: str, env: dict[str, str] | None = None,
                    agent_id: str = "") -> Sandbox:
        """Start a microVM from a template or snapshot and attach to its runtime.

        ``image`` names a template whose image already runs sprout-runtime, or a
        snapshot id; an OCI image reference goes through :meth:`spawn_from_image`
        instead. The VM's shape (CPUs, memory, disk) is the template's or the
        snapshot's -- only a cold start can choose one. ``agent_id`` binds the
        returned handle's identity, so every call made through it is attributed
        to that agent.
        """
        if not image:
            raise ValueError("spawn needs a template or snapshot id")
        payload: dict = {
            "templateID": image,
            "timeout": self.sandbox_ttl,
            "autoPause": True,
            # A sandbox that auto-paused at TTL expiry is resumed transparently
            # by its next proxied call; a sandbox kept alive for grading after
            # the run relies on it.
            "autoResume": {"enabled": True},
        }
        if env:
            payload["envVars"] = env
        if self.allow_internet is not None:
            # Both spellings on purpose. Measured against AgentENV 758d575
            # (2026-09-03): POST /sandboxes only honours the snake_case key
            # (its NewSandbox model lacks the camelCase rename) while
            # /sandboxes-cold only honours camelCase; unknown keys are ignored,
            # so sending both is what works on every server version.
            payload["allowInternetAccess"] = self.allow_internet
            payload["allow_internet_access"] = self.allow_internet

        resp = await self._client.post(f"{self.server_url}/sandboxes", json=payload)
        resp.raise_for_status()
        return self.attach(_sandbox_id(resp.json()), agent_id)

    async def spawn_from_image(self, image: str, agent_id: str = "",
                               resources: dict | None = None) -> Sandbox:
        """Cold-start a microVM from an OCI image reference.

        :meth:`spawn` starts from a snapshot or template, which is what a
        replay or a re-board needs. This is the other entry: a benchmark names
        its environment with an image reference, and only this path accepts
        one. The image must already contain the runtime, started by the
        template's startup command -- the same requirement as any template.

        A cold start on a cache miss pulls and converts image layers and can
        take tens of seconds. A tag can be re-pushed, so a caller that must say
        which image it ran against pins the digest before asking (the Harbor
        environment does, with regctl) rather than trusting the name.
        """
        payload: dict = {
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
        resp = await self._client.post(f"{self.server_url}/sandboxes-cold",
                                       json=payload)
        resp.raise_for_status()
        return self.attach(_sandbox_id(resp.json()), agent_id)

    async def destroy(self, *sandboxes: Sandbox) -> None:
        """Tear down specific sandboxes."""
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
        """Drop the handles and leave the VMs running.

        For a sandbox that must outlive its pool -- one whose final snapshot
        failed holds the only copy of the rollout, and Harbor's ``stop`` keeps
        it for ``harness reap`` or a person. ``close`` destroys every sandbox
        the pool still holds, which a forgotten one no longer is.
        """
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
        await self.destroy_all()
        await self._client.aclose()

    # --- Snapshots ---

    async def snapshot(self, sandbox: Sandbox, *, name: str | None = None,
                       disk_only: bool = False) -> Snapshot:
        """Publish the sandbox's current state; the sandbox keeps running.

        This is the checkpoint primitive a rollout uses per step, and
        `spawn(image=<snapshot id>)` later starts a fresh sandbox from it.
        `disk_only` skips the VM state and memory image, which is what makes a
        per-step checkpoint cheap: the cost is roughly the bytes written since
        the previous capture, and the guest filesystem is synced first so
        recent writes are on the virtual disk. Sandboxes created from a
        disk-only snapshot cold-boot (processes are gone, the template's
        startup command is re-run) instead of resuming.
        """
        sid = sandbox.require_id()
        payload: dict = {}
        if name:
            payload["name"] = name
        if disk_only:
            payload["diskOnly"] = True
        resp = await self._client.post(
            f"{self.server_url}/sandboxes/{sid}/snapshots", json=payload)
        resp.raise_for_status()
        return Snapshot.from_api(resp.json(), disk_only=disk_only)

    async def squash(self, snapshot: Snapshot | str, *,
                     name: str | None = None) -> Snapshot:
        """Flatten a snapshot's chain into an equivalent one-layer snapshot.

        A sandbox started from a snapshot inherits its layers as a prefix it
        can never compact, so branching repeatedly from deep snapshots pays a
        growing cost. Squashing hands out a one-layer equivalent instead; the
        source snapshot is untouched and stays usable. Already-flat chains
        come back unchanged (the same id), so this is safe to call blindly.
        """
        snapshot_id = snapshot.id if isinstance(snapshot, Snapshot) else snapshot
        disk_only = snapshot.disk_only if isinstance(snapshot, Snapshot) else False
        payload: dict = {}
        if name:
            payload["name"] = name
        resp = await self._client.post(
            f"{self.server_url}/snapshots/{snapshot_id}/squash", json=payload)
        resp.raise_for_status()
        return Snapshot.from_api(resp.json(), disk_only=disk_only)

    # --- Handles ---

    def handle(self, sandbox_id: str, agent_id: str = "") -> Sandbox:
        """A fresh, unregistered handle on an existing sandbox.

        A handle's HTTP client binds to the first event loop that uses it, so
        whoever calls from another loop takes its own handle from here.
        """
        sb = Sandbox(backend=GatewayBackend(
            self.server_url, sandbox_id,
            sandbox_id_header=self.SANDBOX_ID_HEADER,
            target_port=self.runtime_port,
            target_port_header=self.TARGET_PORT_HEADER,
        ), agent_id=agent_id)
        sb._container_id = sandbox_id
        return sb

    def attach(self, sandbox_id: str, agent_id: str = "") -> Sandbox:
        """Wrap an AgentENV sandbox id as a Sandbox reachable via the proxy.

        Public because a caller that created a sandbox may lend it to another
        process, which serves tool calls into a sandbox the owner still
        snapshots. Registered: this pool destroys it on close.
        """
        sb = self.handle(sandbox_id, agent_id=agent_id)
        self._sandboxes[sandbox_id] = sb
        return sb


def _sandbox_id(body: dict) -> str:
    """The sandbox id an AgentENV create response reports."""
    sid = body.get("sandboxID")
    if not sid:
        raise RuntimeError(f"no sandbox id in response: {body}")
    return str(sid)

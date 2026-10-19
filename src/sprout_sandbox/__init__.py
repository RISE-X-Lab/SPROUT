"""Sandboxes from two pools: AgentENV microVMs (``MicroVMPool``) and rootless podman (``PodmanPool``).

A ``Sandbox`` handle calls the tools of the runtime in one sandbox, and gets a ``ToolResult`` for each call.
"""

from .backends import GatewayBackend, HttpBackend
from .btrfs_store import BtrfsStore
from .podman_pool import PodmanPool
from .pool import MicroVMPool, Snapshot
from .result import ToolResult
from .sandbox import Sandbox

__all__ = [
    "ToolResult",
    "GatewayBackend",
    "HttpBackend",
    "Sandbox",
    "MicroVMPool",
    "PodmanPool",
    "BtrfsStore",
    "Snapshot",
]

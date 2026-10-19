"""Harbor's environment on a rootless podman container whose rootfs is a btrfs snapshot.

``PodmanEnvironment`` boots the task image with ``sprout-runtime`` inside, on the session's
``podman`` backend (``sprout_sandbox.podman_pool``). Its sandboxes live in a btrfs store that
``ops/btrfs_store.sh`` makes once, as root. Everything else runs as the user. Harbor's phase
policies change the egress of the running container (``SandboxSession.set_network``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

from sprout.eval.harbor.environment import SandboxEnvironment
from sprout_sandbox.podman_cli import default_podman_bin
from sprout_sandbox.podman_images import CONFIG_KEYS, digest_reference, image_info

#: The bound in seconds on a command that Harbor runs without a timeout of its own: the eight-hour
#: actor budget plus time for setup and collection.
EXEC_TIMEOUT_S = 36000


class PodmanEnvironment(SandboxEnvironment):
    """``SandboxEnvironment`` on the btrfs ``store``, with ``runtime_bin`` in every container.

    ``type()`` is ``podman``. ``podman_bin`` defaults to ``default_podman_bin()``. The constructor
    raises ValueError if ``store`` is not a directory.
    """

    def __init__(
        self, *args: Any, runtime_bin: str, store: str, podman_bin: str | None = None, **kwargs: Any
    ) -> None:
        self.store = str(Path(store).expanduser().resolve())
        self.podman_bin = podman_bin or default_podman_bin()
        super().__init__(*args, runtime_bin=runtime_bin, exec_timeout_s=EXEC_TIMEOUT_S, **kwargs)

    @staticmethod
    def type() -> str:
        return "podman"

    def _validate_definition(self) -> None:
        super()._validate_definition()
        if not Path(self.store).is_dir():
            raise ValueError(
                f"podman store does not exist: {self.store} (sudo ops/btrfs_store.sh "
                "<mount point> <image file> <size>)"
            )

    def _backend_section(self) -> dict[str, object]:
        return {
            "backend": "podman",
            "podman": {
                "store": self.store,
                "runtime_bin": self.runtime_bin,
                "podman_bin": self.podman_bin,
                # The baseline policy. Harbor's phases change the egress of the running container.
                "allow_internet": self._allow_internet,
            },
        }

    def _resources(self) -> dict[str, int]:
        # No disk: the rootfs is a subvolume of the store and has none to size.
        return {"cpu": self._effective_cpus or 2, "memory_mb": self._effective_memory_mb or 1024}

    def _receipt_extra(self) -> dict[str, object]:
        # A branch must run on the store that holds its parent's snapshots.
        return {"store": self.store}

    async def _prepare_image(self, force_build: bool) -> tuple[str, dict[str, Any]]:
        """The task image pinned to the digest that podman pulled, and its OCI config.

        Both come from one ``image_info`` record. Raises ValueError if ``force_build``.
        """
        if force_build:
            raise ValueError("Sprout boots the task's prebuilt image; it does not build one")
        image = self.task_env_config.docker_image
        if not image:
            # _validate_definition admits a task without a docker_image only with an overlay.
            assert self.overlay is not None
            image = self.overlay.base
        info = await image_info(image, podman_bin=self.podman_bin, cwd=self.store)
        config = info.get("Config") or {}
        return digest_reference(image, info), {key: config.get(key) for key in CONFIG_KEYS}

    async def _set_network(self, allow: bool) -> None:
        await asyncio.to_thread(self._running_session().set_network, allow)

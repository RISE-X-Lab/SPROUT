"""Harbor environment lifecycle backed by a rootless podman container on a btrfs store.

The same trial as on a microVM -- the task image booted with sprout-runtime
inside, the tests/Dockerfile overlay replayed, a separate verifier, a disk
snapshot after every model turn -- on a machine without the snapshot service.
The session's ``podman`` backend does the work (``sprout_sandbox.podman``):
the image's rootfs is extracted once into the store, every sandbox is a
writable btrfs snapshot of it, a checkpoint is a read-only snapshot taken while
the container runs, and a branch is a container started from one. Egress is a
network membership the pool toggles, so Harbor's per-phase policy is
``session.set_network``. Everything runs as the user; the store is the one
thing made as root (``ops/btrfs_store.sh``).
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from sprout.harbor_plugin.environment import SandboxEnvironment
from sprout_sandbox.podman import CONFIG_KEYS, default_podman_bin, digest_reference, image_info

#: A command Harbor runs without a timeout of its own may take this long: the
#: eight-hour actor budget plus setup and collection, as on the microVM.
EXEC_TIMEOUT_S = 36000


class PodmanEnvironment(SandboxEnvironment):
    def __init__(self, *args, runtime_bin: str, store: str, podman_bin: str | None = None, **kwargs):
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

    def _backend_section(self) -> dict:
        return {
            "backend": "podman",
            "podman": {
                "store": self.store,
                "runtime_bin": self.runtime_bin,
                "podman_bin": self.podman_bin,
                # The baseline policy; Harbor's phases toggle the running container.
                "allow_internet": self._allow_internet,
            },
        }

    def _resources(self) -> dict:
        # No disk: the rootfs is a subvolume of the store and has none to size.
        return {"cpu": self._effective_cpus or 2, "memory_mb": self._effective_memory_mb or 1024}

    def _receipt_extra(self) -> dict:
        # A branch must run on the store that holds its parent's snapshots.
        return {"store": self.store}

    async def _prepare_image(self, force_build: bool) -> tuple[str, dict]:
        """The task image pinned to the digest podman pulled, and its OCI config.

        Both from ``podman image inspect`` (pulling first when the image is
        not local), rootless and without regctl; the pool keys its extracted
        base on the same record's image id.
        """
        if force_build:
            raise ValueError("Sprout boots the task's prebuilt image; it does not build one")
        image = self.task_env_config.docker_image or self.overlay.base
        info = await image_info(image, podman_bin=self.podman_bin, cwd=self.store)
        config = info.get("Config") or {}
        return digest_reference(image, info), {key: config.get(key) for key in CONFIG_KEYS}

    async def _set_network(self, allow: bool) -> None:
        await asyncio.to_thread(self.session.set_network, allow)

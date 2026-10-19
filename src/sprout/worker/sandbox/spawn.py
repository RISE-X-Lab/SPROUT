"""Starting a sandbox in the configured backend from the name of an image.

The name becomes a per-image template built on demand, a cold start from an OCI reference with the
task's shape, or a spawn from a template or a snapshot.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

from sprout.contract.backend import backend_name
from sprout.worker.sandbox.backends import resolve_microvm_endpoint, spawn_takes_shape
from sprout.worker.sandbox.template_recipe import (
    DEFAULT_RUNTIME_PORT,
    REGCTL_DEPS_GLOB,
    TemplateError,
    ensure_ripgrep,
    find_regctl,
)
from sprout.worker.sandbox.templates import TemplateBuilder
from sprout_sandbox import MicroVMPool, PodmanPool, Sandbox


def builder_from_backend(backend: dict[str, Any]) -> TemplateBuilder | None:
    """The builder of per-image templates for a microvm ``backend`` that names a ``runtime_bin``, else None.

    Without ``runtime_bin`` there is nothing to install, and an image name is spawned as it is. Raises
    ``TemplateError`` without a server URL, or with ``image_env`` and no regctl. May download ripgrep once."""
    # Resolve the backend and the endpoint as `build_pool` does, defaults and environment included.
    # Otherwise a config that spawns sandboxes fails here, with an error that blames the image.
    if backend_name(backend) != "microvm":
        return None
    section = backend.get("microvm") or {}
    runtime_bin = section.get("runtime_bin") or backend.get("runtime_bin")
    if not runtime_bin:
        return None
    server_url, api_key = resolve_microvm_endpoint(section)
    if not server_url:
        raise TemplateError(
            "microvm.runtime_bin is set but no microvm.server_url and no "
            "AENV_SERVER_URL: a per-image template has to be built somewhere"
        )
    image_env = bool(section.get("image_env"))
    regctl = find_regctl() if image_env else None
    if image_env and regctl is None:
        raise TemplateError(
            "microvm.image_env is set but no regctl was found (SPROUT_REGCTL, PATH, "
            f"or {REGCTL_DEPS_GLOB}); the image's ENV cannot be read"
        )
    return TemplateBuilder(
        server_url=str(server_url).rstrip("/"),
        api_key=api_key,
        runtime_bin=Path(str(runtime_bin)),
        ripgrep_bin=ensure_ripgrep(),
        runtime_port=int(section.get("runtime_port", DEFAULT_RUNTIME_PORT)),
        request_timeout=float(section.get("request_timeout", 120.0)),
        image_env=image_env,
        regctl_bin=regctl,
    )


async def spawn_sandbox(
    pool: MicroVMPool | PodmanPool,
    backend: dict[str, Any],
    image: str,
    resources: dict[str, Any] | None,
    note: Callable[[str, str], None],
) -> Sandbox:
    """A sandbox started in ``pool`` from ``image``, by the path that ``backend`` names.

    ``note`` receives the progress line for a template that has to be resolved.
    """
    # The microVM backend starts from templates, not images. With a runtime_bin, the builder
    # makes one template per image on demand, and reuses it.
    builder = builder_from_backend(backend)
    if builder is not None:
        note("template", "resolving for %s" % image)
        image = builder.template_for(image, resources)
    # The config, not the string, says whether `image` is an OCI reference or a snapshot id.
    # A guess from the string would one day take a template tag for an image tag.
    backend_section = backend.get(backend_name(backend), {}) or {}
    # A builder's name is a template, so it never cold-starts, even with `from_image`. Podman
    # takes the shape on every spawn, from a reference and a snapshot id alike.
    shaped = builder is None and (
        spawn_takes_shape(backend) or bool(backend_section.get("from_image") or backend.get("from_image"))
    )
    if shaped and pool.supports_cold_start():
        return await pool.spawn_from_image(image, resources=resources)
    # No resources: a microVM template or snapshot fixes its shape, and the pool rejects the argument.
    # The builder put the task's shape into the template, and a resumed run keeps its machine.
    return await pool.spawn(image=image)

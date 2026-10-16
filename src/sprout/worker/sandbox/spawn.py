"""How an image name becomes a running sandbox in the configured backend: a
per-image template built on demand, a cold start from an OCI reference with the
task's shape, or a spawn from a template or snapshot."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from sprout.worker.sandbox.backends import backend_name, resolve_microvm_endpoint, spawn_takes_shape
from sprout.worker.sandbox.template_recipe import (
    DEFAULT_RUNTIME_PORT,
    REGCTL_DEPS_GLOB,
    TemplateError,
    ensure_ripgrep,
    find_regctl,
)
from sprout.worker.sandbox.templates import TemplateBuilder
from sprout_sandbox import MicroVMPool, PodmanPool, Sandbox


def builder_from_backend(backend: dict) -> TemplateBuilder | None:
    """A builder for a microvm backend that asks for per-image templates.

    Returns ``None`` unless the backend names a ``runtime_bin``: without the
    binary there is nothing to install, so the harness must be pointed at
    pre-built templates instead.
    """
    # Resolved exactly as `build_pool` resolves it, default backend and environment
    # included. They used to disagree: the pool fell back to AENV_SERVER_URL /
    # AENV_API_KEY while this required both in the config, so a configuration that
    # spawned sandboxes fine failed here -- reported as "could not create a
    # sandbox", which points at the image rather than at the missing setting.
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
    backend: dict,
    image: str,
    resources: dict | None,
    note: Callable[[str, str], None],
) -> Sandbox:
    """A sandbox started in ``pool`` from ``image``, by the path ``backend`` names.

    ``note`` receives the progress line for a template that has to be resolved.
    """
    # A caller names its environment with an image; the microVM backend
    # starts from templates. Build one per image on demand (cached across
    # instances) when the config says how to get the runtime into it.
    builder = builder_from_backend(backend)
    if builder is not None:
        note("template", "resolving for %s" % image)
        image = builder.template_for(image, resources)
    # Two entries, and the config says which one this caller's image names
    # are for. A benchmark names its environment with an OCI image
    # reference, which only the microVM cold-start path accepts; a replay
    # or a re-board hands over a snapshot id, which only its snapshot path
    # accepts. Guessing from the string would eventually mistake a
    # template tag for an image tag. A builder-resolved name is a template
    # by construction, so it must never take the cold-start path even when
    # `from_image` is set. The podman pool has one path for a reference
    # and a snapshot id alike, and it takes the shape every time (a
    # subvolume carries none), so both go through it with the resources.
    backend_section = backend.get(backend_name(backend), {}) or {}
    shaped = builder is None and (
        spawn_takes_shape(backend) or bool(backend_section.get("from_image") or backend.get("from_image"))
    )
    if shaped and pool.supports_cold_start():
        return await pool.spawn_from_image(image, resources=resources)
    # No resources here on purpose: a microVM template or snapshot
    # has its shape baked in (only a cold start can choose one), and
    # the pool rejects the argument rather than ignoring it. The
    # task's shape reached the template above; a resumed run
    # continues in the machine it was running on.
    return await pool.spawn(image=image)

"""How a session starts its sandbox: the spawn path of each backend, and the builder of microVM templates.

``SandboxSession.create`` spawns on a ``StubPool`` here (``use_pool``). A microVM template comes from
a scripted AgentENV API (``agentenv_stub``). ``ensure_ripgrep`` finds its binary in the cache under
``$HOME``, so nothing is downloaded.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import sprout.worker.sandbox.template_recipe as recipe
from sprout.worker.sandbox.session import SandboxSession
from sprout.worker.sandbox.spawn import builder_from_backend
from sprout.worker.sandbox.template_recipe import TemplateError
from tests.worker.sandbox.pools import StubPool

IMAGE = "swebench/sweb.eval.x86_64.django__django-11848:latest"
SHAPE = {"cpu": 2, "memory_mb": 2048}


@pytest.fixture
def runtime_bin(tmp_path: Path) -> Path:
    path = tmp_path / "sprout-runtime"
    path.write_bytes(b"runtime v1")
    return path


@pytest.fixture
def cached_ripgrep(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """The ripgrep binary that an earlier download left in the cache of ``$HOME``."""
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.delenv("AENV_SERVER_URL", raising=False)
    binary = tmp_path / ".cache" / "sprout-swebench" / ("rg-%s" % recipe.RIPGREP_VERSION)
    binary.parent.mkdir(parents=True)
    binary.write_bytes(b"rg v14")
    return binary


# --- the spawn path of each backend -----------------------------------------------------------


@pytest.mark.parametrize(
    "backend, image, resources, spawned",
    [
        (None, IMAGE, None, ("spawn_from_image", IMAGE, None)),
        ({"backend": "podman"}, IMAGE, SHAPE, ("spawn_from_image", IMAGE, SHAPE)),
        (
            {"backend": "podman"},
            "sprout-0123456789ab-0003",
            SHAPE,
            ("spawn_from_image", "sprout-0123456789ab-0003", SHAPE),
        ),
        ({"backend": "microvm"}, "snap-7", SHAPE, ("spawn", "snap-7")),
        (
            {"backend": "microvm", "microvm": {"from_image": True}},
            IMAGE,
            SHAPE,
            ("spawn_from_image", IMAGE, SHAPE),
        ),
    ],
    ids=["default", "podman-image", "podman-snapshot", "microvm-snapshot", "microvm-from-image"],
)
def test_a_sandbox_takes_the_shape_of_its_task_on_each_spawn_path_that_can_choose_one(
    use_pool, backend, image, resources, spawned
):
    """A subvolume carries no shape, so a podman branch takes it again. A microVM snapshot has its own."""
    pool = use_pool(StubPool())

    SandboxSession(quiet=True, backend=backend).create(image, resources)

    assert pool.operations == [spawned]


def test_a_microvm_with_a_runtime_binary_spawns_from_the_template_of_its_image_and_shape(
    use_pool, agentenv_stub, runtime_bin, cached_ripgrep
):
    """The template has the shape built in, so the spawn passes none."""
    name = "sprout-swebench-cbf3bb78d36b48d94cf98012"
    agentenv_stub.answer("GET", "/templates/aliases/" + name, (200, {"templateID": "tid"}))
    agentenv_stub.answer("GET", "/templates/tid/builds/tid/status", (200, {"status": "ready"}))
    pool = use_pool(StubPool())
    backend = {
        "backend": "microvm",
        "microvm": {"server_url": agentenv_stub.url, "runtime_bin": str(runtime_bin), "from_image": True},
    }

    SandboxSession(quiet=True, backend=backend).create(IMAGE, SHAPE)

    assert pool.operations == [("spawn", name)]


# --- the template builder of a backend --------------------------------------------------------


@pytest.mark.parametrize(
    "backend",
    [
        {"runtime_bin": "/path/to/sprout-runtime"},
        {"backend": "podman", "podman": {"runtime_bin": "/path/to/sprout-runtime"}},
        {"backend": "microvm"},
    ],
    ids=["default-backend", "podman", "microvm-without-a-runtime-binary"],
)
def test_no_template_is_built_without_a_microvm_runtime_binary_to_install(backend):
    """A podman container mounts the runtime. Without a binary, a microVM name is spawned as it is."""
    assert builder_from_backend(backend) is None


def test_the_builder_of_a_microvm_backend_takes_its_endpoint_port_and_ripgrep(runtime_bin, cached_ripgrep):
    builder = builder_from_backend(
        {
            "backend": "microvm",
            "microvm": {
                "server_url": "http://server:8000/",
                "api_key": "key",
                "runtime_bin": str(runtime_bin),
                "runtime_port": 4000,
                "request_timeout": 30,
            },
        }
    )

    assert builder is not None
    assert (
        builder.server_url,
        builder.api_key,
        builder.runtime_bin,
        builder.runtime_port,
        builder.request_timeout,
        builder.ripgrep_bin,
        builder.image_env,
    ) == ("http://server:8000", "key", runtime_bin, 4000, 30.0, cached_ripgrep, False)


def test_a_runtime_binary_beside_the_section_and_a_server_in_the_environment_make_a_builder(
    runtime_bin, cached_ripgrep, monkeypatch
):
    """The endpoint resolves as for the pool, so a config that spawns sandboxes also builds templates."""
    monkeypatch.setenv("AENV_SERVER_URL", "http://bench:18000")

    builder = builder_from_backend({"backend": "microvm", "runtime_bin": str(runtime_bin)})

    assert (builder.server_url, builder.runtime_bin) == ("http://bench:18000", runtime_bin)


def test_a_runtime_binary_without_a_server_is_refused(runtime_bin, cached_ripgrep):
    with pytest.raises(
        TemplateError, match="^microvm.runtime_bin is set but no microvm.server_url and no AENV_SERVER_URL"
    ):
        builder_from_backend({"backend": "microvm", "microvm": {"runtime_bin": str(runtime_bin)}})


def test_image_env_without_a_regctl_is_refused_before_a_template_is_built(
    runtime_bin, cached_ripgrep, monkeypatch, tmp_path
):
    monkeypatch.delenv("SPROUT_REGCTL", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path / "bin"))
    monkeypatch.setattr(recipe, "REGCTL_DEPS_GLOB", str(tmp_path / "deps" / "*" / "regctl"))
    backend = {
        "backend": "microvm",
        "microvm": {"server_url": "http://x", "runtime_bin": str(runtime_bin), "image_env": True},
    }

    with pytest.raises(TemplateError, match="^microvm.image_env is set but no regctl was found"):
        builder_from_backend(backend)


def test_image_env_builds_templates_with_the_hosts_regctl(runtime_bin, cached_ripgrep, monkeypatch, tmp_path):
    regctl = tmp_path / "regctl"
    regctl.write_text("#!/bin/sh\n")
    regctl.chmod(0o755)
    monkeypatch.setenv("SPROUT_REGCTL", str(regctl))
    backend = {
        "backend": "microvm",
        "microvm": {"server_url": "http://x", "runtime_bin": str(runtime_bin), "image_env": True},
    }

    builder = builder_from_backend(backend)

    assert (builder.image_env, builder.regctl_bin) == (True, regctl)

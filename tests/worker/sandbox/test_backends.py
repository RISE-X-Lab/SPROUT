"""``build_pool`` and the edits of a backend section: egress, the microVM lease and the AgentENV endpoint.

A podman pool checks its store's mount at construction. So each podman test runs with the fake
``btrfs`` and ``findmnt`` on PATH (``fake_btrfs_store``) and a dummy runtime binary.
"""

from __future__ import annotations

import asyncio
import copy
import math
from pathlib import Path

import pytest

from sprout.contract.backend import BackendError
from sprout.worker.sandbox.backends import (
    build_pool,
    default_aenv_server_url,
    resolve_microvm_endpoint,
    with_egress,
    with_sandbox_budget,
)
from sprout_sandbox import MicroVMPool, PodmanPool


@pytest.fixture(autouse=True)
def no_agentenv_in_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """The host's own AgentENV endpoint and key must not reach a test."""
    monkeypatch.delenv("AENV_SERVER_URL", raising=False)
    monkeypatch.delenv("AENV_API_KEY", raising=False)


def set_environment(monkeypatch: pytest.MonkeyPatch, variables: dict[str, str]) -> None:
    for name, value in variables.items():
        monkeypatch.setenv(name, value)


# --- the microVM pool -----------------------------------------------------------------------


def test_a_microvm_pool_is_built_from_its_section():
    pool = build_pool(
        {
            "backend": "microvm",
            "microvm": {
                "server_url": "http://127.0.0.1:8000/",
                "runtime_port": 4000,
                "sandbox_ttl": 900,
                "allow_internet": False,
            },
        }
    )

    assert isinstance(pool, MicroVMPool)
    assert (pool.server_url, pool.runtime_port, pool.sandbox_ttl, pool.allow_internet) == (
        "http://127.0.0.1:8000",
        4000,
        900,
        False,
    )


def test_a_microvm_section_that_names_only_its_server_gets_the_defaults():
    """``allow_internet`` None keeps the server's default."""
    pool = build_pool({"backend": "microvm", "microvm": {"server_url": "http://127.0.0.1:8000"}})

    assert (pool.runtime_port, pool.sandbox_ttl, pool.allow_internet) == (3000, 600, None)


def test_a_microvm_section_without_a_server_takes_the_one_in_aenv_server_url(monkeypatch):
    monkeypatch.setenv("AENV_SERVER_URL", "http://bench:18000")

    pool = build_pool({"backend": "microvm"})

    assert pool.server_url == "http://bench:18000"


async def spawn_and_close(pool: MicroVMPool) -> None:
    await pool.spawn("template-1")
    await pool.close()


def test_the_microvm_pool_authenticates_with_the_key_of_its_section(agentenv_stub, tmp_path, monkeypatch):
    """AgentENV checks ``X-API-KEY``. A key in a file keeps it out of a config that reaches a commit."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "aenv-key").write_text("secret-from-file\n")
    agentenv_stub.answer("POST", "/sandboxes", (201, {"sandboxID": "sb-1"}))
    pool = build_pool(
        {"backend": "microvm", "microvm": {"server_url": agentenv_stub.url, "api_key_file": "~/aenv-key"}}
    )

    asyncio.run(spawn_and_close(pool))

    assert agentenv_stub.paths() == ["POST /sandboxes", "DELETE /sandboxes/sb-1"]
    assert agentenv_stub.header_values("x-api-key") == {"secret-from-file"}


@pytest.mark.parametrize(
    "section, environment, endpoint",
    [
        (
            {},
            {"AENV_SERVER_URL": "http://from-env:8000", "AENV_API_KEY": "key-from-env"},
            ("http://from-env:8000", "key-from-env"),
        ),
        (
            {"server_url": "http://named:1"},
            {"AENV_SERVER_URL": "http://from-env:8000"},
            ("http://named:1", ""),
        ),
        (
            {"server_url": "http://x", "api_key_file": "~/aenv-key"},
            {"AENV_API_KEY": "key-from-env"},
            ("http://x", "secret-from-file"),
        ),
        (
            {"server_url": "http://x", "api_key": "inline", "api_key_file": "~/aenv-key"},
            {},
            ("http://x", "inline"),
        ),
        ({}, {}, ("", "")),
    ],
    ids=["environment", "named-server", "file-before-environment", "inline-before-file", "nothing"],
)
def test_the_endpoint_comes_from_the_section_then_a_key_file_then_the_environment(
    tmp_path, monkeypatch, section, environment, endpoint
):
    """An empty server URL lets each caller raise its own error."""
    monkeypatch.setenv("HOME", str(tmp_path))
    (tmp_path / "aenv-key").write_text("secret-from-file\n")
    set_environment(monkeypatch, environment)

    assert resolve_microvm_endpoint(section) == endpoint


@pytest.mark.parametrize(
    "environment, url",
    [({"AENV_SERVER_URL": "http://bench:18000/"}, "http://bench:18000/"), ({}, "http://127.0.0.1:8000")],
    ids=["named", "agentenvs-own-default"],
)
def test_the_assumed_agentenv_server_is_aenv_server_url_or_else_port_8000(monkeypatch, environment, url):
    """The Harbor environment and the reaper assume one server. They once assumed two ports."""
    set_environment(monkeypatch, environment)

    assert default_aenv_server_url() == url


# --- the podman pool --------------------------------------------------------------------------


def test_a_podman_pool_is_built_from_its_section(fake_btrfs_store, fake_runtime_bin, tmp_path):
    pool = build_pool(
        {
            "backend": "podman",
            "podman": {
                "store": str(fake_btrfs_store),
                "runtime_bin": str(fake_runtime_bin),
                "podman_bin": "/path/to/podman",
                "allow_internet": False,
                "socket_dir": str(tmp_path / "sockets"),
            },
        }
    )

    assert isinstance(pool, PodmanPool)
    assert (pool.store.path, pool.runtime_bin, pool.podman_bin, pool.allow_internet, pool.socket_dir) == (
        fake_btrfs_store,
        fake_runtime_bin,
        "/path/to/podman",
        False,
        tmp_path / "sockets",
    )


def test_a_config_that_names_no_backend_gets_a_podman_pool_with_the_defaults(
    fake_btrfs_store, fake_runtime_bin, monkeypatch
):
    """The podman that ``ops/setup_podman.sh`` installs off PATH is named by ``$SPROUT_PODMAN_BIN``."""
    monkeypatch.setenv("SPROUT_PODMAN_BIN", "/path/to/podman/bin/podman")

    pool = build_pool({"podman": {"store": str(fake_btrfs_store), "runtime_bin": str(fake_runtime_bin)}})

    assert isinstance(pool, PodmanPool)
    assert (pool.podman_bin, pool.socket_dir, pool.allow_internet) == (
        "/path/to/podman/bin/podman",
        fake_btrfs_store / "sockets",
        None,
    )


def test_the_runtime_binary_can_be_named_beside_the_section(fake_btrfs_store, fake_runtime_bin):
    """The CLI passes ``--runtime-bin`` at the top level, as the template builder reads it."""
    pool = build_pool(
        {
            "backend": "podman",
            "runtime_bin": str(fake_runtime_bin),
            "podman": {"store": str(fake_btrfs_store)},
        }
    )

    assert pool.runtime_bin == fake_runtime_bin


def test_a_tilde_in_a_podman_path_is_the_home_directory(
    fake_btrfs_store, fake_runtime_bin, monkeypatch, tmp_path
):
    """A YAML config names ``~/sprout-btrfs`` where a shell would have expanded it."""
    monkeypatch.setenv("HOME", str(tmp_path))

    pool = build_pool(
        {"podman": {"store": "~/store", "runtime_bin": "~/sprout-runtime", "socket_dir": "~/sockets"}}
    )

    assert (pool.store.path, pool.runtime_bin, pool.socket_dir) == (
        fake_btrfs_store,
        fake_runtime_bin,
        tmp_path / "sockets",
    )


def test_a_microvm_section_takes_each_setting_that_it_documents(tmp_path):
    """A section refuses a key that it does not know, and so every run that names it. The CLI's
    section names ``from_image`` and ``runtime_bin``. ``image_user`` and ``image_env`` run Harbor tasks."""
    (tmp_path / "aenv-key").write_text("key\n")
    section = {
        "server_url": "http://127.0.0.1:8000",
        "runtime_port": 3000,
        "api_key": "key",
        "api_key_file": str(tmp_path / "aenv-key"),
        "request_timeout": 30,
        "sandbox_ttl": 600,
        "from_image": True,
        "runtime_bin": "/path/to/sprout-runtime",
        "allow_internet": True,
        "image_env": True,
        "image_user": True,
    }

    assert isinstance(build_pool({"backend": "microvm", "microvm": section}), MicroVMPool)


def test_a_podman_section_takes_each_setting_that_it_documents(fake_btrfs_store, fake_runtime_bin, tmp_path):
    section = {
        "store": str(fake_btrfs_store),
        "runtime_bin": str(fake_runtime_bin),
        "podman_bin": "podman",
        "allow_internet": True,
        "socket_dir": str(tmp_path / "sockets"),
        "image_env": True,
        "image_user": True,
    }

    assert isinstance(build_pool({"backend": "podman", "podman": section}), PodmanPool)


# --- misconfiguration ---------------------------------------------------------------------------


def podman(store: Path, runtime_bin: Path, **settings: object) -> dict[str, object]:
    return {"backend": "podman", "podman": {"store": str(store), "runtime_bin": str(runtime_bin), **settings}}


@pytest.mark.parametrize(
    "config, message",
    [
        (lambda store, binary: {}, r"^backend 'podman' needs podman\.runtime_bin: "),
        (
            lambda store, binary: {"podman": {"store": str(store)}},
            r"^backend 'podman' needs podman\.runtime_bin: ",
        ),
        (
            lambda store, binary: {"podman": {"runtime_bin": str(binary)}},
            r"^backend 'podman' needs podman\.store: .*sudo ops/btrfs_store\.sh ",
        ),
        (
            lambda store, binary: podman(store, "/path/to/absent/sprout-runtime"),
            r"^backend 'podman': runtime_bin does not exist: /path/to/absent/sprout-runtime ",
        ),
        (
            lambda store, binary: podman("/path/to/absent-store", binary),
            r"^backend 'podman': store does not exist: /path/to/absent-store ",
        ),
        (
            lambda store, binary: podman(store, binary, allow_internt=False),
            r"^unknown podman setting\(s\): allow_internt; known: allow_internet, image_env, image_user, "
            r"podman_bin, runtime_bin, socket_dir, store$",
        ),
        (
            lambda store, binary: podman(store, binary, network="lab"),
            r"^unknown podman setting\(s\): network; ",
        ),
        (
            lambda store, binary: {"podman": "runtime/sprout-runtime"},
            r"^config key 'podman' must be a mapping of settings$",
        ),
    ],
    ids=[
        "no-config",
        "no-runtime-binary",
        "no-store",
        "runtime-binary-missing",
        "store-missing",
        "mistyped-setting",
        "shared-network-setting",
        "section-not-a-mapping",
    ],
)
def test_a_podman_misconfiguration_names_the_setting_instead_of_failing_later(
    fake_btrfs_store, fake_runtime_bin, config, message
):
    """A mistyped ``allow_internt: false`` would run a no-network task online. No network is shared now."""
    with pytest.raises(BackendError, match=message):
        build_pool(config(fake_btrfs_store, fake_runtime_bin))


def test_a_store_that_is_not_on_btrfs_is_refused_with_the_mount_that_it_is_on(
    fake_btrfs_store, fake_runtime_bin, fake_findmnt
):
    fake_findmnt.report(fstype="ext4", options="rw,relatime")

    with pytest.raises(
        BackendError, match=r"^backend 'podman': store .* is on ext4 mounted with rw,relatime; "
    ):
        build_pool(podman(fake_btrfs_store, fake_runtime_bin))


@pytest.mark.parametrize(
    "config, message",
    [
        (
            {"backend": "microvm"},
            r"^backend 'microvm' needs microvm\.server_url \(or AENV_SERVER_URL\), e\.g\. http://127\.0\.0\.1:8000$",
        ),
        (
            {"backend": "microvm", "microvm": {"server_url": "http://x", "sandbox_tll": 900}},
            r"^unknown microvm setting\(s\): sandbox_tll; known: ",
        ),
        (
            {"backend": "microvm", "microvm": "http://x"},
            r"^config key 'microvm' must be a mapping of settings$",
        ),
        (
            {
                "backend": "microvm",
                "microvm": {"server_url": "http://x", "api_key_file": "/path/to/absent-key"},
            },
            r"^api_key_file not found: /path/to/absent-key$",
        ),
        ({"backend": "docker"}, r"^unknown backend 'docker'; the backends are 'microvm', 'podman'$"),
    ],
    ids=["no-server", "mistyped-setting", "section-not-a-mapping", "key-file-missing", "unknown-backend"],
)
def test_a_microvm_misconfiguration_names_the_setting_instead_of_failing_later(config, message):
    """A missing server would read as a host that is down, and a missing key file as a bad key."""
    with pytest.raises(BackendError, match=message):
        build_pool(config)


# --- edits of a section ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "config, allow, expected",
    [
        (
            {"backend": "microvm", "microvm": {"server_url": "http://x"}},
            False,
            {"backend": "microvm", "microvm": {"server_url": "http://x", "allow_internet": False}},
        ),
        (
            {"backend": "podman", "podman": {"store": "/path/to/store", "allow_internet": False}},
            True,
            {"backend": "podman", "podman": {"store": "/path/to/store", "allow_internet": True}},
        ),
        ({}, False, {"podman": {"allow_internet": False}}),
    ],
    ids=["microvm", "podman-overridden", "default-backend"],
)
def test_with_egress_sets_whether_the_sandboxes_start_connected_and_leaves_the_given_config(
    config, allow, expected
):
    given = copy.deepcopy(config)

    edited = with_egress(config, allow=allow)

    assert (edited, config) == (expected, given)


def test_the_sandbox_budget_leaves_a_podman_config_as_it_is():
    """A container has no lease. It lives until it is removed."""
    config = {"backend": "podman", "podman": {"store": "/path/to/store"}}

    assert with_sandbox_budget(config, 1800) == {"backend": "podman", "podman": {"store": "/path/to/store"}}


@pytest.mark.parametrize(
    "section, budget, ttl",
    [
        ({"server_url": "http://x"}, 1800, 2400),
        ({"server_url": "http://x", "sandbox_ttl": 0}, 1800, 2400),
        ({"server_url": "http://x", "sandbox_ttl": 900}, 1800, 2400),
        ({"server_url": "http://x", "sandbox_ttl": 7200}, 1800, 7200),
        ({"server_url": "http://x", "sandbox_ttl": 900.5}, 1, 901),
        ({"server_url": "http://x"}, 0.5, 601),
    ],
    ids=["no-lease", "lease-0", "shorter-lease", "longer-lease", "fractional-lease", "fractional-budget"],
)
def test_the_microvm_lease_covers_the_budget_plus_ten_minutes(section, budget, ttl):
    """``max(ceil(sandbox_ttl), ceil(budget) + 600)``: provisioning and teardown take the 600 s."""
    config = {"backend": "microvm", "microvm": section}

    assert with_sandbox_budget(config, budget) == {
        "backend": "microvm",
        "microvm": {**section, "sandbox_ttl": ttl},
    }


@pytest.mark.parametrize(
    "budget", [0, -1, math.inf, math.nan, True, "1800"], ids=["0", "negative", "inf", "nan", "bool", "str"]
)
def test_a_budget_that_is_not_a_finite_positive_number_is_refused(budget):
    with pytest.raises(BackendError, match="^VM lifetime requires a finite positive operation budget$"):
        with_sandbox_budget({"backend": "microvm"}, budget)


@pytest.mark.parametrize("ttl", [-1, math.inf, True, "900"], ids=["negative", "inf", "bool", "str"])
def test_a_lease_that_is_not_a_finite_non_negative_number_is_refused(ttl):
    with pytest.raises(BackendError, match="^microvm.sandbox_ttl must be a finite non-negative number$"):
        with_sandbox_budget({"backend": "microvm", "microvm": {"sandbox_ttl": ttl}}, 1800)

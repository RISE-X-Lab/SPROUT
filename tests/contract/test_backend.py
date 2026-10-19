"""The backend section of a run's config: which backend runs it, and where its snapshots live."""

from __future__ import annotations

import re
from pathlib import Path

import pytest

from sprout.contract.backend import (
    PODMAN_KEYS,
    BackendError,
    backend_name,
    backend_section,
    default_podman_store,
    expanded_path,
    snapshot_store,
)


def test_the_default_store_is_the_one_that_the_environment_names(monkeypatch):
    monkeypatch.setenv("SPROUT_PODMAN_STORE", "/path/to/store")

    assert default_podman_store() == "/path/to/store"


def test_without_a_store_in_the_environment_the_default_store_is_sprout_btrfs_in_the_home(monkeypatch):
    """``ops/setup_podman.sh`` makes ``~/sprout-btrfs``."""
    monkeypatch.setenv("HOME", "/path/to/home")
    monkeypatch.delenv("SPROUT_PODMAN_STORE", raising=False)

    assert default_podman_store() == "/path/to/home/sprout-btrfs"


def test_an_empty_store_variable_names_no_store(monkeypatch):
    monkeypatch.setenv("HOME", "/path/to/home")
    monkeypatch.setenv("SPROUT_PODMAN_STORE", "")

    assert default_podman_store() == "/path/to/home/sprout-btrfs"


@pytest.mark.parametrize(
    "config, name",
    [
        ({}, "podman"),
        ({"backend": None}, "podman"),
        ({"backend": ""}, "podman"),
        ({"backend": "  MicroVM "}, "microvm"),
        ({"backend": "Podman"}, "podman"),
    ],
    ids=["absent", "null", "empty", "padded-mixed-case", "capitalized"],
)
def test_the_backend_name_is_podman_unless_the_config_names_another(config, name):
    assert backend_name(config) == name


def test_the_snapshots_of_a_podman_run_live_in_its_store_with_the_tilde_expanded(monkeypatch):
    """A config file names ``~`` where a shell would expand it."""
    monkeypatch.setenv("HOME", "/path/to/home")

    assert (
        snapshot_store({"backend": "podman", "podman": {"store": "~/sprout-btrfs"}})
        == "/path/to/home/sprout-btrfs"
    )


def test_a_relative_store_is_resolved_against_the_working_directory(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    assert snapshot_store({"podman": {"store": "stores/a"}}) == str(tmp_path.resolve() / "stores" / "a")


@pytest.mark.parametrize(
    "config",
    [
        {
            "backend": "microvm",
            "microvm": {"server_url": "http://127.0.0.1:3000"},
            "podman": {"store": "/path/to/s"},
        },
        {"backend": "podman", "podman": {}},
    ],
    ids=["microvm", "podman-without-store"],
)
def test_a_run_without_a_podman_store_has_no_snapshot_store(config):
    """A microVM's snapshots are AgentENV records, not files on this host."""
    assert snapshot_store(config) is None


def test_a_podman_section_with_an_unknown_key_has_no_snapshot_store():
    with pytest.raises(BackendError, match="^unknown podman setting\\(s\\): stor; known: "):
        snapshot_store({"podman": {"stor": "/path/to/store"}})


@pytest.mark.parametrize(
    "path, expanded",
    [("~/runs", "/path/to/home/runs"), (Path("~/runs"), "/path/to/home/runs")],
    ids=["text", "path-object"],
)
def test_a_tilde_in_a_config_path_is_the_home_directory(monkeypatch, path, expanded):
    monkeypatch.setenv("HOME", "/path/to/home")

    assert expanded_path(path) == expanded


@pytest.mark.parametrize("config", [{}, {"podman": None}], ids=["absent", "null"])
def test_a_missing_section_is_empty(config):
    assert backend_section(config, "podman", PODMAN_KEYS) == {}


def test_the_section_is_a_copy_of_the_one_in_the_config():
    config = {"podman": {"store": "/path/to/store"}}

    backend_section(config, "podman", PODMAN_KEYS)["store"] = "/path/to/other"

    assert config == {"podman": {"store": "/path/to/store"}}


def test_a_section_that_is_not_a_mapping_is_refused():
    with pytest.raises(BackendError, match="^config key 'podman' must be a mapping of settings$"):
        backend_section({"podman": "/path/to/store"}, "podman", PODMAN_KEYS)


def test_a_section_with_unknown_keys_is_refused_with_the_unknown_and_the_known_keys():
    """A mistyped key must not leave a setting at its default without a word."""
    with pytest.raises(
        BackendError,
        match="^%s$"
        % re.escape("unknown microvm setting(s): server, ttl; known: runtime_port, server_url, the_ttl"),
    ):
        backend_section(
            {"microvm": {"ttl": 60, "server": "x", "server_url": "y"}},
            "microvm",
            {"server_url", "runtime_port", "the_ttl"},
        )

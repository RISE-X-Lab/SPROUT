"""The Run Store's config file, its ``$env`` references, the merge of profile defaults, podman snapshots."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from sprout.runstore.config import merge, podman_snapshot_present, read_config, referenced_env, resolve
from tests.support.paths import REPO

#: The example config that docs/RUNSTORE.md starts the services with.
EXAMPLE_CONFIG = REPO / "resources" / "runstore" / "config.example.json"


def config_file(directory: Path, body: object) -> Path:
    path = directory / "runstore.json"
    path.write_text(json.dumps(body))
    return path


# References


def test_resolve_replaces_each_env_reference_in_mappings_and_lists(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "key-value")
    monkeypatch.setenv("REGION", "region-value")

    resolved = resolve({"model": {"api_key": {"$env": "MODEL_KEY"}}, "zones": [{"$env": "REGION"}, "fixed"]})

    assert resolved == {"model": {"api_key": "key-value"}, "zones": ["region-value", "fixed"]}


def test_a_mapping_with_more_keys_than_env_is_not_a_reference(monkeypatch):
    monkeypatch.setenv("MODEL_KEY", "key-value")

    assert resolve({"$env": "MODEL_KEY", "default": "x"}) == {"$env": "MODEL_KEY", "default": "x"}


def test_resolve_raises_key_error_for_an_unset_variable(monkeypatch):
    monkeypatch.delenv("UNSET_MODEL_KEY", raising=False)

    with pytest.raises(KeyError, match="UNSET_MODEL_KEY"):
        resolve({"api_key": {"$env": "UNSET_MODEL_KEY"}})


def test_referenced_env_names_each_variable_in_mappings_and_lists():
    value = {
        "model": {"api_key": {"$env": "MODEL_KEY"}},
        "zones": [{"$env": "REGION"}, {"$env": "MODEL_KEY"}],
    }

    assert referenced_env(value) == {"MODEL_KEY", "REGION"}


# Merging profile defaults


def test_merge_lays_the_updates_over_the_base_key_by_key_where_both_are_mappings():
    base = {"slot": "mini-swe-agent", "backend": {"backend": "podman", "podman": {"store": "/path/to/a"}}}
    updates = {"backend": {"podman": {"store": "/path/to/b", "image_user": True}}, "model": "policy"}

    assert merge(base, updates) == {
        "slot": "mini-swe-agent",
        "backend": {"backend": "podman", "podman": {"store": "/path/to/b", "image_user": True}},
        "model": "policy",
    }


@pytest.mark.parametrize(
    "base,updates,expected",
    [
        ({"backend": {"backend": "podman"}}, {"backend": "microvm"}, {"backend": "microvm"}),
        ({"backend": "microvm"}, {"backend": {"backend": "podman"}}, {"backend": {"backend": "podman"}}),
        ({"extra": {"a": 1}}, {"extra": None}, {"extra": None}),
    ],
    ids=["value-over-a-mapping", "mapping-over-a-value", "null-over-a-mapping"],
)
def test_merge_replaces_a_value_where_either_side_is_not_a_mapping(base, updates, expected):
    assert merge(base, updates) == expected


def test_merge_changes_neither_input():
    base = {"backend": {"podman": {"store": "/path/to/a"}}}
    updates = {"backend": {"podman": {"store": "/path/to/b"}}}

    merge(base, updates)

    assert (base, updates) == (
        {"backend": {"podman": {"store": "/path/to/a"}}},
        {"backend": {"podman": {"store": "/path/to/b"}}},
    )


# The config file


def test_read_config_returns_a_config_with_profiles_and_an_artifact_root(tmp_path):
    body = {"artifact_root": "/path/to/artifacts", "profiles": {"mini": {"env": {"KEY": {"$env": "KEY"}}}}}

    assert read_config(config_file(tmp_path, body)) == body


@pytest.mark.parametrize(
    "body,error",
    [
        (
            {
                "artifact_root": "/path/to/artifacts",
                "profiles": {"mini": {"worker_env": {"API_KEY": "sk-live"}}},
            },
            r"Use a worker-side \$env reference instead of inline credentials",
        ),
        ({"artifact_root": "/path/to/artifacts"}, "Worker config requires profiles and artifact_root"),
        (
            {"artifact_root": "/path/to/artifacts", "profiles": []},
            "Worker config requires profiles and artifact_root",
        ),
        ({"profiles": {}}, "Worker config requires profiles and artifact_root"),
        ({"artifact_root": "", "profiles": {}}, "Worker config requires profiles and artifact_root"),
    ],
    ids=[
        "inline-credential",
        "no-profiles",
        "profiles-not-an-object",
        "no-artifact-root",
        "blank-artifact-root",
    ],
)
def test_read_config_refuses_a_config_that_a_worker_cannot_use(tmp_path, body, error):
    with pytest.raises(ValueError, match=error):
        read_config(config_file(tmp_path, body))


def test_the_example_config_is_a_valid_config():
    assert set(read_config(EXAMPLE_CONFIG)["profiles"]) == {
        "mini",
        "harness",
        "grade",
        "mini-microvm",
        "grade-microvm",
        "review",
    }


# Podman snapshots


@pytest.fixture
def podman_store(tmp_path: Path) -> Path:
    """A store whose ``snapshots/`` holds the snapshot ``sprout-1-3``, a file and a ``a\\b`` directory."""
    store = tmp_path / "store"
    (store / "snapshots" / "sprout-1-3").mkdir(parents=True)
    (store / "snapshots" / "a\\b").mkdir()
    (store / "snapshots" / "sprout-1-file").write_text("")
    return store


@pytest.mark.parametrize(
    "snapshot_id,present",
    [("sprout-1-3", True), ("sprout-1-4", False), ("sprout-1-file", False)],
    ids=["subvolume", "absent", "file"],
)
def test_a_podman_snapshot_is_present_while_its_directory_is_in_the_store(podman_store, snapshot_id, present):
    assert podman_snapshot_present(str(podman_store), snapshot_id) is present


@pytest.mark.parametrize(
    "snapshot_id",
    ["", ".", "..", "../snapshots/sprout-1-3", "sprout-1-3/", "a\\b", None, 7],
    ids=["empty", "dot", "dot-dot", "relative-path", "trailing-slash", "backslash", "null", "number"],
)
def test_a_snapshot_id_that_is_not_a_plain_directory_name_is_never_present(podman_store, snapshot_id):
    """Each of these ids names an existing directory, or is not text. None of them is a snapshot."""
    assert podman_snapshot_present(str(podman_store), snapshot_id) is False

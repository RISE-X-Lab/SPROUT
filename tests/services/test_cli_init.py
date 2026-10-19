"""``init``: the config files from the packaged templates with this host's values, and the secrets.

``init`` never overwrites a file or a secret, so a second run keeps each edit. It prints no secret value.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from sprout.services import __main__ as cli
from sprout.services import secrets
from sprout.services.spec import load_spec, template_path
from tests.services.helpers import write_spec
from tests.support.paths import RUNTIME_BIN

#: The secrets that the packaged services.yaml names.
TEMPLATE_SECRETS = ["db-password", "litellm-master-key", "rl-driver-token", "runstore-token"]


def names_in(directory: Path) -> list[str]:
    return sorted(path.name for path in directory.iterdir())


def values_at_any_depth(data: Any, key: str) -> set[Any]:
    """Each value of ``key`` in ``data``, in a mapping or a list at any depth."""
    if isinstance(data, dict):
        found = {data[key]} if key in data else set()
        return found.union(*(values_at_any_depth(value, key) for value in data.values()))
    if isinstance(data, list):
        return set().union(*(values_at_any_depth(item, key) for item in data))
    return set()


def secrets_in(text: str, paths) -> list[str]:
    """The name of each template secret whose value appears in ``text``."""
    return [name for name in TEMPLATE_SECRETS if secrets.require(paths, name) in text]


def test_init_on_a_new_host_writes_the_config_files_from_the_templates(paths):
    code = cli.main(["init"])

    assert (
        code,
        names_in(paths.config),
        paths.spec.read_text(),
        (paths.config / "litellm.yaml").read_text(),
    ) == (
        0,
        ["litellm.yaml", "rl_driver.json", "runstore.json", "services.yaml"],
        template_path("services.yaml").read_text(),
        template_path("litellm.yaml").read_text(),
    )


def test_init_makes_each_secret_that_services_yaml_names(paths):
    cli.main(["init"])

    assert names_in(paths.secrets) == TEMPLATE_SECRETS


def test_init_never_overwrites_a_config_file_or_a_secret(paths):
    cli.main(["init"])
    (paths.config / "litellm.yaml").write_text("model_list: []\n")
    password = secrets.require(paths, "db-password")

    code = cli.main(["init"])

    assert (code, (paths.config / "litellm.yaml").read_text(), secrets.require(paths, "db-password")) == (
        0,
        "model_list: []\n",
        password,
    )


def test_init_fills_runstore_json_with_the_artifact_root_the_runtime_binary_and_the_interpreter_of_this_host(
    paths,
):
    """``runstore init`` refuses a volatile artifact root. A profile's jobs run on its interpreter."""
    write_spec(paths, {"proxy": {"kind": "process", "command": ["true"]}}, python="/path/to/python")

    cli.main(["init"])

    config = json.loads((paths.config / "runstore.json").read_text())
    assert (
        config["artifact_root"],
        values_at_any_depth(config, "runtime_bin"),
        values_at_any_depth(config["profiles"], "python"),
    ) == (str(paths.state / "runstore-artifacts"), {str(RUNTIME_BIN)}, {"/path/to/python"})


def test_init_points_the_rl_driver_at_a_ledger_under_state_and_at_tokens_that_its_service_receives(paths):
    cli.main(["init"])

    config = json.loads((paths.config / "rl_driver.json").read_text())
    spec = load_spec(paths.spec)
    received = set(dict(spec.exports_for(spec.get("rl-driver"))))
    assert (config["ledger"], {config["runstore_token_env"], config["driver_token_env"]} - received) == (
        str(paths.state / "rl-driver" / "groups.sqlite3"),
        set(),
    )


def test_init_prints_no_secret_value(paths, capsys):
    cli.main(["init"])

    assert secrets_in(capsys.readouterr().out, paths) == []


def podman_stores(config: dict) -> dict[str, Any]:
    """The ``store`` of each podman backend section of ``config``, by ``<profile>.<section>``."""
    return {
        "%s.%s" % (name, section): profile[section]["backend"]["podman"].get("store")
        for name, profile in config["profiles"].items()
        for section in ("run_defaults", "grade_defaults")
        if profile.get(section, {}).get("backend", {}).get("backend") == "podman"
    }


def test_init_names_the_store_of_sprout_podman_store_in_each_podman_section(paths, monkeypatch):
    """The template's store is an example path, and each job of a podman profile failed on it."""
    monkeypatch.setenv("SPROUT_PODMAN_STORE", "/path/to/the-store-of-this-host")

    cli.main(["init"])

    config = json.loads((paths.config / "runstore.json").read_text())
    assert podman_stores(config) == {
        "mini.run_defaults": "/path/to/the-store-of-this-host",
        "grade.grade_defaults": "/path/to/the-store-of-this-host",
    }


def test_without_sprout_podman_store_init_names_sprout_btrfs_in_the_home_directory(paths, monkeypatch):
    """The worker's doctor and each run use this default, which ``ops/setup_podman.sh`` makes."""
    monkeypatch.delenv("SPROUT_PODMAN_STORE", raising=False)

    cli.main(["init"])

    config = json.loads((paths.config / "runstore.json").read_text())
    store = str(Path(os.environ["HOME"]) / "sprout-btrfs")
    assert podman_stores(config) == {"mini.run_defaults": store, "grade.grade_defaults": store}

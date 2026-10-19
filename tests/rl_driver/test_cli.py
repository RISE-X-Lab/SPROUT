"""``python -m sprout.rl_driver --config FILE``: the address and the tokens of the driver, and its ledger.

Each test replaces ``uvicorn.run``, where the command hands over its app, with a spy. The spy serves
nothing: the test calls the app it caught with ``TestClient``. docs/RL_DRIVER.md ("Start and
configuration") describes the fields of the config.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import uvicorn
from fastapi.testclient import TestClient

from sprout.rl_driver import __main__ as command
from sprout.rl_driver.ledger import Conflict, Ledger
from tests.support.paths import REPO

SHIPPED_CONFIG = REPO / "resources" / "rl_driver" / "config.example.json"


class Served:
    """A spy on ``uvicorn.run``: it keeps the app and the options of each call."""

    def __init__(self) -> None:
        self.calls: list[tuple[Any, dict[str, Any]]] = []

    def __call__(self, app: Any, **options: Any) -> None:
        self.calls.append((app, options))


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Served:
    """The spy in place of ``uvicorn.run``, with both tokens of the default config in the environment."""
    spy = Served()
    monkeypatch.setattr(uvicorn, "run", spy)
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "runstore-token")
    monkeypatch.setenv("SPROUT_RL_DRIVER_TOKEN", "driver-token")
    return spy


def start(monkeypatch: pytest.MonkeyPatch, config: Path) -> None:
    """Run the command as ``rl_driver --config <config>``."""
    monkeypatch.setattr(sys, "argv", ["rl_driver", "--config", str(config)])
    command.main()


def write_config(directory: Path, **fields: Any) -> Path:
    """A config file in ``directory`` whose ledger is ``groups.sqlite3`` beside it, with ``fields``."""
    path = directory / "rl_driver.json"
    path.write_text(json.dumps({"ledger": "groups.sqlite3", **fields}))
    return path


def test_the_driver_serves_on_its_documented_default_address(served, monkeypatch, tmp_path):
    start(monkeypatch, write_config(tmp_path))

    ((_, options),) = served.calls
    assert options == {"host": "127.0.0.1", "port": 11001}


@pytest.mark.parametrize(
    "fields, bound_to",
    [
        ({}, "http://127.0.0.1:18110"),
        ({"runstore_url": "http://runstore.test:18110/"}, "http://runstore.test:18110"),
    ],
)
def test_the_ledger_is_bound_to_the_configured_run_store_url(served, monkeypatch, tmp_path, fields, bound_to):
    start(monkeypatch, write_config(tmp_path, **fields))

    ledger = Ledger(tmp_path / "groups.sqlite3")
    ledger.bind(bound_to)
    with pytest.raises(Conflict):
        ledger.bind("http://runstore.test:9999")


def test_a_relative_ledger_path_is_found_beside_the_config_and_not_in_the_working_directory(
    served, monkeypatch, tmp_path
):
    (tmp_path / "config").mkdir()
    (tmp_path / "elsewhere").mkdir()
    config = write_config(tmp_path / "config", ledger="state/groups.sqlite3")
    monkeypatch.chdir(tmp_path / "elsewhere")

    start(monkeypatch, config)

    created = (tmp_path / "config" / "state" / "groups.sqlite3").is_file()
    assert (created, list((tmp_path / "elsewhere").iterdir())) == (True, [])


@pytest.mark.parametrize(
    "config_fields, headers, status",
    [
        pytest.param({}, {}, 401, id="default-env-no-token"),
        pytest.param({}, {"Authorization": "Bearer driver-token"}, 200, id="default-env"),
        pytest.param(
            {"driver_token_env": "MY_DRIVER_TOKEN"}, {"Authorization": "Bearer my-token"}, 200, id="own-env"
        ),
        pytest.param(
            {"driver_token_env": "MY_DRIVER_TOKEN"},
            {"Authorization": "Bearer driver-token"},
            401,
            id="own-env-other-token",
        ),
        pytest.param({"driver_token_env": None}, {}, 200, id="no-authentication"),
    ],
)
def test_the_served_app_takes_the_callers_token_from_the_configured_variable(
    served, monkeypatch, tmp_path, config_fields, headers, status
):
    monkeypatch.setenv("MY_DRIVER_TOKEN", "my-token")
    start(monkeypatch, write_config(tmp_path, **config_fields))
    ((app, _),) = served.calls

    with TestClient(app) as http:
        response = http.get("/health", headers=headers)

    assert response.status_code == status


def test_the_shipped_config_starts_the_driver_with_its_miles_section_and_its_ledger_under_home(
    served, monkeypatch, tmp_path
):
    """docs/RL_DRIVER.md starts the driver with this file as it is."""
    monkeypatch.setenv("HOME", str(tmp_path))

    start(monkeypatch, SHIPPED_CONFIG)

    ((app, _),) = served.calls
    with TestClient(app) as http:
        health = http.get("/health").json()
    assert (health, (tmp_path / "sprout-runs" / "rl-driver" / "groups.sqlite3").is_file()) == (
        {"role": "driver", "miles_configured": True},
        True,
    )


ALLOWED = "['driver_token_env', 'ledger', 'miles', 'poll_interval_s', 'runstore_token_env', 'runstore_url']"
URL_ERROR = "runstore_url must be an HTTP(S) URL without embedded credentials/query/fragment"


@pytest.mark.parametrize(
    "config, message",
    [
        pytest.param(
            {"ledger": "g.sqlite3", "port": 1}, "Config fields must be drawn from " + ALLOWED, id="field"
        ),
        pytest.param(None, "Config fields must be drawn from " + ALLOWED, id="not-an-object"),
        pytest.param({}, "'ledger'", id="no-ledger"),
        pytest.param({"ledger": "g.sqlite3", "runstore_url": "ftp://runstore.test"}, URL_ERROR, id="scheme"),
        pytest.param({"ledger": "g.sqlite3", "runstore_url": "http://"}, URL_ERROR, id="no-host"),
        pytest.param(
            {"ledger": "g.sqlite3", "runstore_url": "http://u:p@runstore.test"}, URL_ERROR, id="userinfo"
        ),
        pytest.param(
            {"ledger": "g.sqlite3", "runstore_url": "http://runstore.test/?a=1"}, URL_ERROR, id="query"
        ),
        pytest.param(
            {"ledger": "g.sqlite3", "runstore_url": "http://runstore.test/#a"}, URL_ERROR, id="fragment"
        ),
        pytest.param(
            {"ledger": "g.sqlite3", "poll_interval_s": 0},
            "poll_interval_s must be finite and positive",
            id="no-interval",
        ),
        pytest.param(
            {"ledger": "g.sqlite3", "poll_interval_s": True},
            "poll_interval_s must be finite and positive",
            id="bool-interval",
        ),
        pytest.param(
            {"ledger": "g.sqlite3", "poll_interval_s": "1"},
            "poll_interval_s must be finite and positive",
            id="text-interval",
        ),
        pytest.param(
            {"ledger": "g.sqlite3", "poll_interval_s": float("inf")},
            "poll_interval_s must be finite and positive",
            id="endless-interval",
        ),
    ],
)
def test_an_invalid_config_stops_the_command_with_a_usage_error(
    served, monkeypatch, capsys, tmp_path, config, message
):
    path = tmp_path / "rl_driver.json"
    path.write_text(json.dumps(config))

    with pytest.raises(SystemExit) as stopped:
        start(monkeypatch, path)

    assert (stopped.value.code, capsys.readouterr().err.splitlines()[-1], served.calls) == (
        2,
        "rl_driver: error: " + message,
        [],
    )


@pytest.mark.parametrize("variable", ["SPROUT_RUNSTORE_TOKEN", "SPROUT_RL_DRIVER_TOKEN"])
def test_a_token_variable_that_is_not_set_stops_the_command_with_a_usage_error(
    served, monkeypatch, capsys, tmp_path, variable
):
    monkeypatch.delenv(variable)

    with pytest.raises(SystemExit) as stopped:
        start(monkeypatch, write_config(tmp_path))

    assert (stopped.value.code, capsys.readouterr().err.splitlines()[-1]) == (
        2,
        f"rl_driver: error: '{variable}'",
    )


@pytest.mark.parametrize("variable", ["SPROUT_RUNSTORE_TOKEN", "SPROUT_RL_DRIVER_TOKEN"])
def test_an_empty_token_stops_the_command_with_a_usage_error(served, monkeypatch, capsys, tmp_path, variable):
    monkeypatch.setenv(variable, "")

    with pytest.raises(SystemExit) as stopped:
        start(monkeypatch, write_config(tmp_path))

    assert (stopped.value.code, capsys.readouterr().err.splitlines()[-1]) == (
        2,
        "rl_driver: error: Configured tokens must be nonempty",
    )


def test_a_ledger_bound_to_another_run_store_stops_the_command_with_a_usage_error(
    served, monkeypatch, capsys, tmp_path
):
    Ledger(tmp_path / "groups.sqlite3").bind("http://runstore-a.test:18110")

    with pytest.raises(SystemExit) as stopped:
        start(monkeypatch, write_config(tmp_path, runstore_url="http://runstore-b.test:18110"))

    assert (stopped.value.code, capsys.readouterr().err.splitlines()[-1]) == (
        2,
        "rl_driver: error: This ledger belongs to a different Run Store URL",
    )


def test_an_invalid_miles_section_stops_the_command_with_a_usage_error(served, monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit) as stopped:
        start(monkeypatch, write_config(tmp_path, miles={"workers": 4}))

    assert (stopped.value.code, capsys.readouterr().err.splitlines()[-1]) == (
        2,
        "rl_driver: error: Invalid Miles adapter config",
    )


@pytest.mark.parametrize(
    "run_defaults, message",
    [
        pytest.param(["mini-swe-agent"], "run_defaults must be a JSON object", id="run_defaults"),
        pytest.param({"extra": []}, "run_defaults.extra must be a JSON object", id="extra"),
    ],
)
def test_miles_run_defaults_or_their_extra_that_are_not_an_object_stop_the_command_with_a_usage_error(
    served, monkeypatch, capsys, tmp_path, run_defaults, message
):
    """The adapter merges both into each root rollout, so the error must come at the start."""
    with pytest.raises(SystemExit) as stopped:
        start(monkeypatch, write_config(tmp_path, miles={"run_defaults": run_defaults}))

    assert (stopped.value.code, capsys.readouterr().err.splitlines()[-1], served.calls) == (
        2,
        "rl_driver: error: " + message,
        [],
    )


def test_the_driver_loads_no_code_that_runs_a_job_or_opens_the_run_stores_database():
    """The driver submits and polls jobs over HTTP. Workers run the jobs. The Run Store owns its database."""
    probe = (
        "import sys, sprout.rl_driver.__main__\n"
        "loaded = [name for name in sys.modules if name.startswith(tuple(sys.argv[1:]))]\n"
        "print(sorted(loaded))"
    )
    forbidden = [
        "sprout.worker",
        "sprout_sandbox",
        "sprout.runstore.store",
        "sprout.runstore.index",
        "psycopg2",
        "swebench",
        "docker",
    ]

    completed = subprocess.run([sys.executable, "-c", probe, *forbidden], capture_output=True, text=True)

    assert (completed.returncode, completed.stdout, completed.stderr) == (0, "[]\n", "")

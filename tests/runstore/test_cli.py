"""``python -m sprout.runstore``: ``init`` creates the schema, and ``serve`` serves the API of a config.

Each test runs the command in a child process. The tests of ``init`` and ``serve`` use the PostgreSQL test
database (``SPROUT_RUNSTORE_TEST_DSN``), and skip without it.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from uuid import uuid4

import httpx
import pytest

from sprout.runstore.store import Store
from tests.runstore.builders import rollout_spec
from tests.support.runstore import TEST_DSN_ENV
from tests.support.waiting import wait_until

TOKEN = "fixture-token"
#: A profile whose defaults show that ``serve`` read it.
PROFILE = {"python": "/path/to/python", "run_defaults": {"timeout_s": 100}}


def config_file(directory: Path, *, artifact_root: str = "/path/to/artifacts") -> Path:
    path = directory / "runstore.json"
    path.write_text(json.dumps({"artifact_root": artifact_root, "profiles": {"mini": PROFILE}}))
    return path


def runstore_command(*args: str) -> list[str]:
    return [sys.executable, "-m", "sprout.runstore", *args]


def child_env(**variables: str) -> dict[str, str]:
    """The environment of the test with ``variables``, and without a DSN or a token of its own."""
    inherited = {key: value for key, value in os.environ.items() if not key.startswith("SPROUT_RUNSTORE_")}
    return {**inherited, **variables}


@pytest.fixture
def empty_schema_dsn() -> Iterator[str]:
    """The DSN of a new schema with no tables in the test database. The schema is dropped after the test."""
    dsn = os.environ.get(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(
            "set %s to an isolated PostgreSQL database, such as dbname=sprout_runstore_test" % TEST_DSN_ENV
        )
    pytest.importorskip("psycopg2", reason="install the runstore extra: pip install -e '.[runstore]'")
    schema = "fixture_" + uuid4().hex
    with Store(dsn).transaction() as cursor:
        cursor.execute("CREATE SCHEMA %s" % schema)
    yield dsn + " options='-c search_path=%s'" % schema
    with Store(dsn).transaction() as cursor:
        cursor.execute("DROP SCHEMA %s CASCADE" % schema)


def free_port() -> int:
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        return int(listener.getsockname()[1])


def usage_error(stderr: str) -> str:
    """The message of the usage error that argparse writes as the last line of ``stderr``."""
    return stderr.splitlines()[-1].partition("error: ")[2]


def answers(url: str) -> bool:
    try:
        httpx.get(url, timeout=1)
    except httpx.TransportError:
        return False
    return True


def test_init_creates_the_schema_in_the_database_that_the_environment_names(empty_schema_dsn, tmp_path):
    result = subprocess.run(
        runstore_command("init", "--config", str(config_file(tmp_path))),
        env=child_env(SPROUT_RUNSTORE_DSN=empty_schema_dsn),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (result.returncode, Store(empty_schema_dsn).list_jobs()) == (0, []), result.stderr


def test_init_refuses_an_artifact_root_that_the_os_empties_on_reboot_before_it_opens_the_database(tmp_path):
    """No DSN is set: a command that opened the database first would fail with a KeyError instead."""
    result = subprocess.run(
        runstore_command(
            "init", "--config", str(config_file(tmp_path, artifact_root="/tmp/sprout-runstore"))
        ),
        env=child_env(),
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert (result.returncode, usage_error(result.stderr)) == (
        2,
        "/tmp/sprout-runstore is under /tmp, which the OS empties on reboot -- "
        "a journal is the only record a run leaves",
    )


@pytest.fixture
def served_url(run_store: Store, tmp_path: Path) -> Iterator[str]:
    """The URL of ``python -m sprout.runstore serve`` over ``run_store``, with the token ``TOKEN``."""
    pytest.importorskip("uvicorn", reason="install the runstore extra: pip install -e '.[runstore]'")
    port = free_port()
    url = "http://127.0.0.1:%d" % port
    server = subprocess.Popen(
        runstore_command("serve", "--config", str(config_file(tmp_path)), "--port", str(port)),
        env=child_env(SPROUT_RUNSTORE_DSN=run_store.dsn, SPROUT_RUNSTORE_TOKEN=TOKEN),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until(lambda: answers(url), timeout_s=30, what="the Run Store API on port %d" % port)
        yield url
    finally:
        server.terminate()
        server.wait(timeout=30)


def test_serve_serves_the_api_with_the_token_of_the_environment_and_the_profiles_of_the_config(served_url):
    response = httpx.post(
        served_url + "/v1/jobs",
        json={"kind": "rollout", "spec": rollout_spec(), "profile": "mini"},
        headers={"Authorization": "Bearer " + TOKEN, "Idempotency-Key": "key-1"},
        timeout=10,
    )

    assert (response.status_code, response.json()["request"]["spec"]) == (202, rollout_spec(timeout_s=100))

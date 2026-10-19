"""``fresh_run_store`` gives each test a Run Store of its own on the PostgreSQL test database."""

from __future__ import annotations

import os

import pytest

from sprout.runstore.specs import JobSpec
from sprout.runstore.store import Store
from tests.support.runstore import TEST_DSN_ENV, fresh_run_store


@pytest.fixture
def dsn() -> str:
    value = os.environ.get(TEST_DSN_ENV)
    if not value:
        pytest.skip("set %s to an isolated PostgreSQL database" % TEST_DSN_ENV)
    pytest.importorskip("psycopg2")
    return value


def rollout() -> JobSpec:
    return JobSpec("rollout", {"prompt": "fixture"})


def current_schema(store: Store) -> str:
    with store.transaction() as cursor:
        cursor.execute("SELECT current_schema() AS name")
        return cursor.fetchone()["name"]


def schema_exists(dsn: str, schema: str) -> bool:
    with Store(dsn).transaction() as cursor:
        cursor.execute("SELECT count(*) AS found FROM pg_namespace WHERE nspname = %s", (schema,))
        return cursor.fetchone()["found"] == 1


def test_two_fresh_run_stores_do_not_see_each_others_jobs(dsn):
    with fresh_run_store(dsn) as first, fresh_run_store(dsn) as second:
        first.submit(rollout(), "only-in-the-first")

        assert second.list_jobs() == []


def test_a_second_store_on_the_dsn_of_a_fresh_run_store_sees_its_jobs(dsn):
    with fresh_run_store(dsn) as store:
        job = store.submit(rollout(), "fixture")

        assert Store(store.dsn).get(job["id"]) == store.get(job["id"])


def test_the_schema_of_a_fresh_run_store_is_dropped_when_the_block_exits(dsn):
    with fresh_run_store(dsn) as store:
        schema = current_schema(store)
        existed = schema_exists(dsn, schema)

    assert (existed, schema_exists(dsn, schema)) == (True, False)

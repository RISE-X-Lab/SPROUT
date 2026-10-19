"""A Run Store on the isolated PostgreSQL test database, in a schema of its own for each test.

``$SPROUT_RUNSTORE_TEST_DSN`` names the database, for example ``dbname=sprout_runstore_test``. A test
that uses ``run_store`` skips when the variable is not set, or when psycopg2 is not installed. Do not
set the variable to a production database: each test creates a schema in it, and drops it again.
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from uuid import uuid4

import pytest

from sprout.runstore.store import Store

#: The variable that names the PostgreSQL test database.
TEST_DSN_ENV = "SPROUT_RUNSTORE_TEST_DSN"


@contextmanager
def fresh_run_store(dsn: str) -> Iterator[Store]:
    """A Run Store in a new, initialized schema of the database ``dsn``, dropped when the block exits.

    ``Store(store.dsn)`` opens the same schema, so a second store sees the same jobs, as after a restart."""
    schema = "fixture_" + uuid4().hex
    admin = Store(dsn)
    with admin.transaction() as cursor:
        cursor.execute("CREATE SCHEMA %s" % schema)
    try:
        store = Store(dsn + " options='-c search_path=%s'" % schema)
        store.initialize()
        yield store
    finally:
        with admin.transaction() as cursor:
            cursor.execute("DROP SCHEMA %s CASCADE" % schema)


@pytest.fixture
def run_store() -> Iterator[Store]:
    """A Run Store of its own for one test (``fresh_run_store``). Skips without the test database."""
    dsn = os.environ.get(TEST_DSN_ENV)
    if not dsn:
        pytest.skip(
            "set %s to an isolated PostgreSQL database, such as dbname=sprout_runstore_test" % TEST_DSN_ENV
        )
    pytest.importorskip("psycopg2", reason="install the runstore extra: pip install -e '.[runstore]'")
    with fresh_run_store(dsn) as store:
        yield store


def _session(worker_id: str) -> str:
    """The session in which ``registered`` registers ``worker_id``."""
    return "session-" + worker_id


def registered(store: Store, *worker_ids: str, host: str = "host-1") -> None:
    """Register each worker of ``worker_ids`` on ``host`` in a session of its own, as at its start.

    A claim reads the host and the grades of the claiming worker from its registration.
    """
    for worker_id in worker_ids:
        store.register_worker(worker_id, host=host, session=_session(worker_id))


def stopped(store: Store, worker_id: str) -> None:
    """Mark the row that ``registered`` gave ``worker_id`` stopped, as the worker does when it stops."""
    store.worker_stopped(worker_id, _session(worker_id))

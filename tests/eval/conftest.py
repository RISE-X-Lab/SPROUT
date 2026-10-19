"""The Run Store of a gate or an evaluation under test: the test's own, served over HTTP on a local port."""

from collections.abc import Iterator

import pytest

from sprout.runstore.index import Index
from sprout.runstore.store import Store
from sprout.search.client import RunStoreClient

#: The bearer token of the test's Run Store API.
RUNSTORE_TOKEN = "fixture-token"


@pytest.fixture
def run_store_client(run_store: Store) -> Iterator[RunStoreClient]:
    """A client of the HTTP API over ``run_store``. Skips without the test database or uvicorn.

    Every recovery point that the fake worker publishes is available: its snapshot counts as kept."""
    pytest.importorskip("uvicorn", reason="install the runstore extra: pip install -e '.[runstore]'")
    from sprout.runstore.api import create_app
    from tests.support.http import serving

    index = Index(run_store, snapshot_valid=lambda point: True)
    with serving(create_app(run_store, RUNSTORE_TOKEN, index=index)) as url:
        client = RunStoreClient(url, RUNSTORE_TOKEN)
        try:
            yield client
        finally:
            client.close()

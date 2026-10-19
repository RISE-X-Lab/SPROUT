"""``RunStoreClient.all_tools`` reads every page of an attempt's tool calls, and stops on a stuck cursor.

A stub Run Store answers the pages of ``GET /v1/jobs/{id}/tools`` from a script, then empty pages.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import httpx
import pytest

from sprout.search.client import RunStoreClient


class Pages:
    """A stub of the tool-call route: each request gets the next scripted page, then ``[]``."""

    def __init__(self, *pages: list[dict[str, Any]]) -> None:
        self._pages = list(pages)
        #: The ``after`` of each request, in order.
        self.cursors: list[int] = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.cursors.append(int(request.url.params["after"]))
        return httpx.Response(200, json=self._pages.pop(0) if self._pages else [])


@pytest.fixture
def client() -> Iterator[RunStoreClient]:
    """A ``RunStoreClient`` whose transport the test sets."""
    opened = RunStoreClient("http://runstore", "fixture")
    yield opened
    opened.close()


def tools(*depths: int) -> list[dict[str, Any]]:
    return [{"depth": depth, "response": {"output": "ok"}} for depth in depths]


def connect(client: RunStoreClient, pages: Pages) -> None:
    client.http.close()
    client.http = httpx.Client(base_url="http://runstore", transport=httpx.MockTransport(pages))


def test_all_tools_reads_each_page_after_the_last_depth_until_an_empty_page(client):
    pages = Pages(tools(1, 2), tools(3))
    connect(client, pages)

    calls = client.all_tools("job-0", attempt_id="attempt-job-0")

    assert (calls, pages.cursors) == (tools(1, 2, 3), [0, 2, 3])


def test_all_tools_refuses_a_page_that_does_not_move_the_cursor(client):
    """Without the check, a Run Store that answers the same page again would hold the coordinator forever."""
    connect(client, Pages(tools(1, 2), tools(2)))

    with pytest.raises(ValueError, match="Run Store tool cursor did not advance"):
        client.all_tools("job-0", attempt_id="attempt-job-0")

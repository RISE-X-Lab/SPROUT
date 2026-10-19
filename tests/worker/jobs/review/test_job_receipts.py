"""A review through its real transport to an endpoint on 127.0.0.1: the receipts of each call, and its stop.

Each test is medium. The analyst and the reviewer read the short templates of ``plain_prompts``, so the
endpoint tells the requests apart by their first word, and the two roots by their progress.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.worker.jobs.review.job import run_review
from tests.worker.jobs.review.endpoint import HOLD, completion, scripted_endpoint
from tests.worker.jobs.review.plain_prompts import prompts_record, reports_in
from tests.worker.jobs.review.review_model import ANALYSIS, failed_roots, repair_plan, review_spec


@pytest.fixture(autouse=True)
def review_key(monkeypatch):
    monkeypatch.setenv("REVIEW_KEY", "review-secret")


def two_roots(tmp_path: Path, url: str, **changes: Any) -> dict[str, Any]:
    """Two failed roots, of 50% and 0% progress, with one point each and two samples at each point."""
    root = tmp_path / "artifacts"
    parents = failed_roots(root, 2)
    parents[1]["grade"]["progress"] = 0.0
    spec = review_spec(root, parents, model_endpoint=url, points=1, points_per_attempt=True, candidates=2)
    spec.update(prompts=prompts_record(tmp_path), max_output_tokens=1000, **changes)
    return spec


def reply(prompt: str) -> tuple[int, dict[str, Any]]:
    """The review model's answer: a report, a point at the first step offered, or a turn at that point."""
    if prompt.startswith("PROBLEM "):
        return 200, completion(json.dumps(ANALYSIS))
    (row,) = reports_in(prompt)
    if prompt.startswith("POINTS "):
        branch = {"name": "point", "base": row["name"], "branch_step": row["available_steps"][0], "why": "w"}
        return 200, completion(
            "```branch-plan\n" + json.dumps({"synthesis": "s", "branches": [branch]}) + "\n```"
        )
    return 200, completion(repair_plan(row["name"], row["available_steps"][0], "pytest -q"))


def receipts(directory: Path) -> dict[str, dict[str, Any]]:
    return {
        str(path.relative_to(directory / "reviews")): json.loads(path.read_text())
        for path in sorted((directory / "reviews").rglob("review-*.json"))
    }


def slow_samples(*, root_2_first: threading.Event | None = None) -> Any:
    """``reply``, but each sample's first try is held until it times out.

    With ``root_2_first``, root 1's analysis is answered once root 2's point review has asked."""

    def respond(body: dict[str, Any]) -> Any:
        prompt = body["messages"][0]["content"]
        if prompt.startswith("TURNS ") and body["max_tokens"] == 1000:
            return HOLD
        if root_2_first is not None and prompt.startswith("PROBLEM ") and "FAIL_TO_PASS): 50%" in prompt:
            assert root_2_first.wait(30)
        if (
            root_2_first is not None
            and prompt.startswith("POINTS ")
            and reports_in(prompt)[0]["name"] == "root-2"
        ):
            root_2_first.set()
        return reply(prompt)

    return respond


def test_each_call_numbers_its_receipts_in_its_own_directory_whatever_order_the_calls_return_in(tmp_path):
    """Each sample's first try times out, and its retry names that try.

    With eight slots, root 2 returns first."""
    with scripted_endpoint(slow_samples()) as endpoint:
        spec = two_roots(tmp_path, endpoint.url, request_timeout_s=0.2)
        one = run_review({**spec, "max_concurrent_requests": 1}, tmp_path / "one")
    with scripted_endpoint(slow_samples(root_2_first=threading.Event())) as endpoint:
        eight = run_review(
            {**spec, "model_endpoint": endpoint.url, "max_concurrent_requests": 8}, tmp_path / "eight"
        )

    first, second = receipts(tmp_path / "one"), receipts(tmp_path / "eight")
    samples = [f"point{a}.1-candidate{s}" for a in (1, 2) for s in (1, 2)]
    assert (
        sorted(second)
        == sorted(first)
        == sorted(
            [f"{call}/review-001.json" for call in ("analyst-1", "analyst-2", "points-1", "points-2")]
            + [f"{call}/review-00{n}.json" for call in samples for n in (1, 2)]
        )
    )
    assert {name: r["request"] for name, r in second.items()} == {
        name: r["request"] for name, r in first.items()
    }
    assert [
        (second[f"{c}/review-002.json"]["retry_of"], second[f"{c}/review-002.json"]["retry_reason"])
        for c in samples
    ] == [("review-001.json", "timeout")] * 4
    assert (json.dumps(eight), one["review_requests"]) == (json.dumps(one), 2 + 2 + 4 * 2)


def test_a_failure_ends_the_pause_before_the_retry_of_a_running_call_and_sends_no_retry(tmp_path):
    """Root 2's first try comes back 502 while root 1's analysis is refused.

    Stopped, root 2 waits out no pause."""
    out = threading.Event()

    def answer(body: dict[str, Any]) -> Any:
        prompt = body["messages"][0]["content"]
        if "FAIL_TO_PASS): 50%" in prompt:
            assert out.wait(30)
            return 401, {"error": "Unauthorized"}
        out.set()
        return 502, {"error": "Bad Gateway"}

    with scripted_endpoint(answer) as endpoint:
        spec = two_roots(tmp_path, endpoint.url)
        with pytest.raises(httpx.HTTPStatusError, match="401 Unauthorized") as raised:
            run_review(spec, tmp_path / "review")

    written = receipts(tmp_path / "review")
    assert (raised.value.__notes__, sorted(written)) == (
        ["in review call analyst-1"],
        ["analyst-1/review-001.json", "analyst-2/review-001.json"],
    )
    assert (
        written["analyst-2/review-001.json"]["error"].startswith("HTTPStatusError"),
        len(endpoint.requests),
    ) == (True, 2)

"""``ReviewTransport`` against an endpoint on 127.0.0.1: the request, its receipts, its retries and its stop.

Each test is medium: the transport sends real HTTP. A stop that records each pause before a retry, and
does not wait it out, stands in for the 10, 30 and 60 seconds between tries.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import httpx
import pytest

from sprout.worker.jobs.review.branch_plan import PromptRejected, ReviewResponseTruncated
from sprout.worker.jobs.review.structured_review import ANALYST_SCHEMA
from sprout.worker.jobs.review.transport import TRANSIENT_DELAYS_S, RequestStopped, ReviewTransport
from tests.worker.jobs.review.endpoint import DROP, HOLD, completion, scripted_endpoint


class Pauses(threading.Event):
    """A stop that is never set. It records each pause before a retry instead of waiting it out."""

    def __init__(self) -> None:
        super().__init__()
        self.pauses: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.pauses.append(timeout)
        return False


class Watched(threading.Event):
    """A stop that says when the transport starts to wait on it, and then waits."""

    def __init__(self) -> None:
        super().__init__()
        self.waiting = threading.Event()
        self.pauses: list[float | None] = []

    def wait(self, timeout: float | None = None) -> bool:
        self.pauses.append(timeout)
        self.waiting.set()
        return super().wait(timeout)


@pytest.fixture(autouse=True)
def review_key(monkeypatch):
    monkeypatch.setenv("REVIEW_KEY", "review-secret")


def transport(directory: Path, url: str, **settings: Any) -> ReviewTransport:
    defaults: dict[str, Any] = {"model": "fixture", "api_key_env": "REVIEW_KEY", "max_output_tokens": 4000}
    return ReviewTransport(directory, model_endpoint=url, **{**defaults, **settings})


def receipts(directory: Path) -> dict[str, dict[str, Any]]:
    return {path.name: json.loads(path.read_text()) for path in sorted(directory.glob("review-*.json"))}


def ok(text: str = "plan") -> tuple[int, dict[str, Any]]:
    return 200, completion(text)


def in_threads(shared: ReviewTransport, prompts: list[str]) -> list[str | None]:
    """The reply to each of ``prompts``, each sent through ``shared`` from a thread of its own."""
    answers: list[str | None] = [None] * len(prompts)

    def send(index: int) -> None:
        answers[index] = shared("fixture", prompts[index])

    threads = [threading.Thread(target=send, args=(index,), daemon=True) for index in range(len(prompts))]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(30)
    return answers


def test_a_reply_is_the_text_of_the_one_message_of_a_chat_completions_request(tmp_path):
    with scripted_endpoint(lambda body: ok("the plan")) as endpoint:
        text = transport(tmp_path, endpoint.url + "/v1", reasoning_effort="high", temperature=0.8)(
            "fixture", "Pick."
        )

    (request,) = endpoint.requests
    assert (text, request.path, request.authorization) == (
        "the plan",
        "/v1/chat/completions",
        "Bearer review-secret",
    )
    assert request.body == {
        "model": "fixture",
        "messages": [{"role": "user", "content": "Pick."}],
        "max_tokens": 4000,
        "stream": False,
        "reasoning_effort": "high",
        "temperature": 0.8,
    }


def test_without_a_reasoning_effort_or_a_temperature_the_request_leaves_them_to_the_endpoint(tmp_path):
    with scripted_endpoint(lambda body: ok()) as endpoint:
        transport(tmp_path, endpoint.url)("fixture", "prompt")

    assert set(endpoint.requests[0].body) == {"model", "messages", "max_tokens", "stream"}


def test_each_try_is_a_receipt_of_the_request_and_the_reply_without_the_key(tmp_path):
    with scripted_endpoint(lambda body: ok("the plan")) as endpoint:
        transport(tmp_path, endpoint.url, timeout_s=90, temperature=0.5)("fixture", "prompt")

    ((name, receipt),) = receipts(tmp_path).items()
    assert (name, set(receipt)) == (
        "review-001.json",
        {"request", "started_at", "timeout_s", "response", "text", "finished_at", "seconds"},
    )
    assert (receipt["request"], receipt["timeout_s"], receipt["text"], receipt["response"]) == (
        {
            "model": "fixture",
            "prompt": "prompt",
            "max_tokens": 4000,
            "reasoning_effort": None,
            "temperature": 0.5,
        },
        90,
        "the plan",
        completion("the plan"),
    )
    assert "review-secret" not in (tmp_path / name).read_text()


def test_a_model_other_than_the_frozen_one_is_refused_before_a_receipt(tmp_path):
    with scripted_endpoint(lambda body: ok()) as endpoint:
        with pytest.raises(ValueError, match="Review model differs from the frozen evaluation model"):
            transport(tmp_path, endpoint.url)("other-model", "prompt")

    assert (endpoint.requests, receipts(tmp_path)) == ([], {})


def test_an_unset_key_is_refused_before_a_receipt(tmp_path, monkeypatch):
    monkeypatch.delenv("REVIEW_KEY")

    with scripted_endpoint(lambda body: ok()) as endpoint:
        with pytest.raises(ValueError, match="REVIEW_KEY is unset"):
            transport(tmp_path, endpoint.url)("fixture", "prompt")

    assert (endpoint.requests, receipts(tmp_path)) == ([], {})


def test_a_try_that_times_out_is_sent_once_more_with_half_the_output_budget(tmp_path):
    """A try that runs to the timeout spent its budget on reasoning. A smaller budget ends sooner."""
    with scripted_endpoint(lambda body: HOLD if body["max_tokens"] == 64000 else ok()) as endpoint:
        text = transport(tmp_path, endpoint.url, max_output_tokens=64000, timeout_s=0.2)("fixture", "prompt")

    first, second = receipts(tmp_path).values()
    assert (text, [r.body["max_tokens"] for r in endpoint.requests]) == ("plan", [64000, 32000])
    assert first["error"].startswith("ReadTimeout")
    assert (second["retry_of"], second["retry_reason"]) == ("review-001.json", "timeout")


def test_a_second_timeout_raises(tmp_path):
    with scripted_endpoint(lambda body: HOLD) as endpoint:
        with pytest.raises(httpx.ReadTimeout):
            transport(tmp_path, endpoint.url, timeout_s=0.2)("fixture", "prompt")

    assert (len(endpoint.requests), list(receipts(tmp_path))) == (2, ["review-001.json", "review-002.json"])


@pytest.mark.parametrize("status", [408, 429, 500, 502, 503, 504])
def test_a_transient_status_is_retried_after_each_pause_with_the_same_budget(tmp_path, status):
    replies = [(status, {"error": "busy"})] * 3 + [ok()]
    stop = Pauses()

    with scripted_endpoint(lambda body: replies.pop(0)) as endpoint:
        text = transport(tmp_path, endpoint.url, stop=stop)("fixture", "prompt")

    assert (text, stop.pauses) == ("plan", list(TRANSIENT_DELAYS_S))
    assert [r.body["max_tokens"] for r in endpoint.requests] == [4000] * 4
    assert [(r.get("retry_of"), r.get("retry_reason")) for r in receipts(tmp_path).values()] == [
        (None, None),
        ("review-001.json", "http %d" % status),
        ("review-002.json", "http %d" % status),
        ("review-003.json", "http %d" % status),
    ]


def test_a_failed_choice_in_a_200_reply_is_retried_as_its_status_and_its_receipt_keeps_the_error(tmp_path):
    """OpenRouter answers 200 first. Its provider's idle timeout then ends the only choice as a 504."""
    failed = completion(None, finish_reason="error")
    failed["choices"][0]["error"] = {"code": 504, "message": "Upstream idle timeout exceeded"}
    replies = [(200, failed), ok()]

    with scripted_endpoint(lambda body: replies.pop(0)) as endpoint:
        text = transport(tmp_path, endpoint.url, stop=Pauses())("fixture", "prompt")

    first, second = receipts(tmp_path).values()
    assert (text, "Upstream idle timeout exceeded" in first["error"], second["retry_reason"]) == (
        "plan",
        True,
        "http 504",
    )


def test_a_dropped_connection_is_retried(tmp_path):
    replies = [DROP, ok()]

    with scripted_endpoint(lambda body: replies.pop(0)) as endpoint:
        text = transport(tmp_path, endpoint.url, stop=Pauses())("fixture", "prompt")

    assert (text, receipts(tmp_path)["review-002.json"]["retry_reason"]) == ("plan", "RemoteProtocolError")


def test_a_transient_status_after_the_last_pause_raises(tmp_path):
    stop = Pauses()

    with scripted_endpoint(lambda body: (502, {"error": "Bad Gateway"})) as endpoint:
        with pytest.raises(httpx.HTTPStatusError, match="502 Bad Gateway"):
            transport(tmp_path, endpoint.url, stop=stop)("fixture", "prompt")

    assert (len(endpoint.requests), len(receipts(tmp_path)), len(stop.pauses)) == (4, 4, 3)


def test_a_400_says_the_endpoint_refused_the_prompt_and_is_not_retried(tmp_path):
    """Only a smaller prompt can succeed, which the caller decides."""
    stop = Pauses()

    with scripted_endpoint(lambda body: (400, {"error": "context length exceeded"})) as endpoint:
        with pytest.raises(PromptRejected, match=r"HTTP 400 Bad Request\): .*context length exceeded"):
            transport(tmp_path, endpoint.url, stop=stop)("fixture", "prompt")

    (receipt,) = receipts(tmp_path).values()
    assert (len(endpoint.requests), stop.pauses, receipt["error"].startswith("PromptRejected: ")) == (
        1,
        [],
        True,
    )


def test_another_client_error_raises_without_a_retry(tmp_path):
    stop = Pauses()

    with scripted_endpoint(lambda body: (401, {"error": "Unauthorized"})) as endpoint:
        with pytest.raises(httpx.HTTPStatusError, match="401 Unauthorized"):
            transport(tmp_path, endpoint.url, stop=stop)("fixture", "prompt")

    assert (len(endpoint.requests), stop.pauses) == (1, [])


def test_a_reply_cut_at_the_token_limit_raises_with_its_text_and_its_receipt_keeps_it(tmp_path):
    with scripted_endpoint(
        lambda body: (200, completion('{"branches": [', finish_reason="length"))
    ) as endpoint:
        with pytest.raises(ReviewResponseTruncated) as truncated:
            transport(tmp_path, endpoint.url)("fixture", "prompt")

    (receipt,) = receipts(tmp_path).values()
    assert (truncated.value.text, receipt["text"], len(endpoint.requests)) == (
        '{"branches": [',
        '{"branches": [',
        1,
    )
    assert receipt["error"].startswith("ReviewResponseTruncated: ")


def test_a_stopped_transport_sends_nothing_and_writes_no_receipt(tmp_path):
    stop = threading.Event()
    stop.set()

    with scripted_endpoint(lambda body: ok()) as endpoint:
        with pytest.raises(RequestStopped):
            transport(tmp_path, endpoint.url, stop=stop)("fixture", "prompt")

    assert (endpoint.requests, receipts(tmp_path)) == ([], {})


def test_a_stop_during_the_pause_before_a_retry_ends_the_pause_and_sends_no_retry(tmp_path):
    stop, outcome = Watched(), {}

    def call() -> None:
        try:
            sent("fixture", "prompt")
        except RequestStopped as error:
            outcome["error"] = error

    with scripted_endpoint(lambda body: (502, {"error": "Bad Gateway"})) as endpoint:
        sent = transport(tmp_path, endpoint.url, stop=stop)
        caller = threading.Thread(target=call, daemon=True)
        caller.start()
        assert stop.waiting.wait(15), "the transport pauses on its stop before a retry"
        stop.set()
        caller.join(15)

    assert (caller.is_alive(), isinstance(outcome.get("error"), RequestStopped)) == (False, True)
    assert (stop.pauses, len(endpoint.requests), list(receipts(tmp_path))) == ([10.0], 1, ["review-001.json"])


def test_a_try_that_times_out_after_the_stop_is_not_sent_again(tmp_path):
    stop = threading.Event()

    def hold_and_stop(body: dict[str, Any]) -> str:
        stop.set()
        return HOLD

    with scripted_endpoint(hold_and_stop) as endpoint:
        with pytest.raises(RequestStopped):
            transport(tmp_path, endpoint.url, stop=stop, timeout_s=0.2)("fixture", "prompt")

    (receipt,) = receipts(tmp_path).values()
    assert (len(endpoint.requests), receipt["error"].startswith("ReadTimeout")) == (1, True)


def test_calls_that_share_a_transport_number_their_receipts_apart_and_each_retry_names_its_own_try(tmp_path):
    """Every first try is out before any is answered. A counter read without the lock named the wrong try."""
    calls = 6
    together = threading.Barrier(calls, timeout=30)
    lock, tried = threading.Lock(), set()

    def answer(body: dict[str, Any]) -> tuple[int, dict[str, Any]]:
        prompt = body["messages"][0]["content"]
        with lock:
            first = prompt not in tried
            tried.add(prompt)
        if first:
            together.wait()
            return (502 if int(prompt.split()[-1]) % 2 else 429), {"error": "busy"}
        return ok("re: " + prompt)

    with scripted_endpoint(answer) as endpoint:
        shared = transport(tmp_path, endpoint.url, stop=Pauses())
        answers = in_threads(shared, [f"prompt {n}" for n in range(calls)])

    written = receipts(tmp_path)
    retries = {r["retry_of"]: r for r in written.values() if "retry_of" in r}
    assert (answers, sorted(written), shared.sequence) == (
        [f"re: prompt {n}" for n in range(calls)],
        ["review-%03d.json" % n for n in range(1, 2 * calls + 1)],
        2 * calls,
    )
    assert {first: written[first]["request"]["prompt"] for first in retries} == {
        first: retry["request"]["prompt"] for first, retry in retries.items()
    }
    assert sorted(retries) == ["review-%03d.json" % n for n in range(1, calls + 1)]


def test_a_structured_request_appends_the_directive_and_asks_for_strict_json_and_its_overhead_is_what_it_adds(
    tmp_path,
):
    with scripted_endpoint(lambda body: ok("{}")) as endpoint:
        sent = transport(tmp_path, endpoint.url, json_directive="\nJSON ONLY.")
        text = sent.json("fixture", "prompt", schema=ANALYST_SCHEMA, name="sprout_analysis")

    body = endpoint.requests[0].body
    assert (text, body["messages"][0]["content"]) == ("{}", "prompt\nJSON ONLY.")
    assert body["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "sprout_analysis", "strict": True, "schema": ANALYST_SCHEMA},
    }
    added = len(body["messages"][0]["content"]) - len("prompt") + len(json.dumps(body["response_format"]))
    assert sent.json_overhead(ANALYST_SCHEMA, "sprout_analysis") == added
    assert receipts(tmp_path)["review-001.json"]["request"]["response_format"] == body["response_format"]


@pytest.mark.parametrize(
    "settings, error",
    [
        ({"max_output_tokens": 0}, "review output budget must be positive"),
        ({"max_output_tokens": 1.5}, "review output budget must be positive"),
        ({"timeout_s": 0}, "review timeout must be positive"),
        ({"temperature": -0.1}, "review temperature must be between 0 and 2"),
        ({"temperature": 2.1}, "review temperature must be between 0 and 2"),
        ({"temperature": "0.5"}, "review temperature must be between 0 and 2"),
    ],
)
def test_a_transport_with_an_unusable_setting_is_refused(tmp_path, settings, error):
    with pytest.raises(ValueError, match=error):
        transport(tmp_path, "http://127.0.0.1:1", **settings)

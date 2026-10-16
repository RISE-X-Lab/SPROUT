"""ReviewTransport under threads: shared between them, and stopped by another thread."""

import json
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

from sprout.worker.jobs.review import transport as review_transport
from sprout.worker.jobs.review.transport import RequestStopped, ReviewTransport

ENDPOINT = "http://127.0.0.1:18252"


def test_concurrent_calls_number_their_receipts_apart_and_each_retry_names_its_own_try(tmp_path, monkeypatch):
    """Every call's first try is in flight before any is answered; then each
    fails -- by timing out (odd calls) or with a 502 (even ones) -- and is
    tried again. A counter shared without a lock named, for each retry,
    whichever receipt another thread had written last."""
    calls = 6
    transport = ReviewTransport(tmp_path, model="fixture", model_endpoint=ENDPOINT,
                                api_key_env="TEST_KEY", max_output_tokens=4000)
    monkeypatch.setenv("TEST_KEY", "k")
    monkeypatch.setattr(review_transport.time, "sleep", lambda seconds: None)
    together = threading.Barrier(calls, timeout=30)
    lock, tried = threading.Lock(), set()
    request = httpx.Request("POST", ENDPOINT + "/v1/chat/completions")

    def post(url, **kwargs):
        prompt = kwargs["json"]["messages"][0]["content"]
        with lock:
            first = prompt not in tried
            tried.add(prompt)
        if first:
            together.wait()   # a transport that serialised its calls would break this
            if int(prompt.split()[-1]) % 2:
                raise httpx.ReadTimeout("timed out")
            return httpx.Response(502, request=request, text="Bad Gateway")
        return httpx.Response(200, request=request, json={"choices": [{"message": {"content": "re: " + prompt}}]})

    monkeypatch.setattr(review_transport.httpx, "post", post)
    answers = {}
    threads = [threading.Thread(target=lambda i=i: answers.__setitem__(i, transport("fixture", f"prompt {i}")))
               for i in range(calls)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(60)
    assert answers == {i: f"re: prompt {i}" for i in range(calls)}
    receipts = {path.name: json.loads(path.read_text()) for path in tmp_path.glob("review-*.json")}
    assert sorted(receipts) == [f"review-{n:03d}.json" for n in range(1, 2 * calls + 1)]
    assert transport.sequence == 2 * calls
    retries = [receipt for receipt in receipts.values() if "retry_of" in receipt]
    firsts = sorted(name for name, receipt in receipts.items() if "retry_of" not in receipt)
    assert sorted(retry["retry_of"] for retry in retries) == firsts and len(firsts) == calls
    for retry in retries:
        first = receipts[retry["retry_of"]]
        timed_out = int(first["request"]["prompt"].split()[-1]) % 2
        assert retry["request"]["prompt"] == first["request"]["prompt"], "a retry names its own call's try"
        assert first["error"].startswith("ReadTimeout" if timed_out else "HTTPStatusError")
        assert (retry["retry_reason"], retry["request"]["max_tokens"]) == (
            ("timeout", 2000) if timed_out else ("http 502", 4000))
        assert retry["text"] == "re: " + first["request"]["prompt"]


class Watched(threading.Event):
    """A stop that says when the transport begins to wait on it, and for how long."""

    def __init__(self):
        super().__init__()
        self.waits, self.waiting = [], threading.Event()

    def wait(self, timeout=None):
        self.waits.append(timeout)
        self.waiting.set()
        return super().wait(timeout)


def test_a_stopped_transport_tries_nothing_more_and_its_backoff_ends_at_once(tmp_path, monkeypatch):
    """Given a stop, the transport makes no try once it is set -- not a first
    one, not the retry after a 502, not the half-budget retry after a timeout
    -- and writes no receipt for it, and a wait before a retry ends the moment
    the stop is set, an hour-long one included."""
    monkeypatch.setenv("TEST_KEY", "k")
    slept = []
    monkeypatch.setattr(review_transport, "time", SimpleNamespace(time=time.time, sleep=slept.append))
    monkeypatch.setattr(review_transport, "TRANSIENT_DELAYS_S", (3600.0, 3600.0, 3600.0))
    request = httpx.Request("POST", ENDPOINT + "/v1/chat/completions")

    def transport(name, stop):
        return ReviewTransport(tmp_path / name, model="fixture", model_endpoint=ENDPOINT,
                               api_key_env="TEST_KEY", max_output_tokens=4000, stop=stop)

    def receipts(name):
        return sorted(path.name for path in (tmp_path / name).glob("review-*.json"))

    stopped = Watched()
    stopped.set()
    monkeypatch.setattr(review_transport.httpx, "post", lambda url, **kwargs: pytest.fail("a request was sent"))
    with pytest.raises(RequestStopped):
        transport("before", stopped)("fixture", "prompt")
    assert receipts("before") == []

    # A 502, and the stop set while the transport waits to try again.
    tries = []

    def gateway_down(url, **kwargs):
        tries.append(kwargs["json"]["max_tokens"])
        return httpx.Response(502, request=request, text="Bad Gateway")

    monkeypatch.setattr(review_transport.httpx, "post", gateway_down)
    stop, outcome = Watched(), {}
    down = transport("down", stop)

    def call():
        try:
            down("fixture", "prompt")
        except BaseException as error:
            outcome["error"] = error

    thread = threading.Thread(target=call, daemon=True)
    thread.start()
    assert stop.waiting.wait(30), "the transport waits on its stop before a retry"
    stop.set()
    thread.join(30)
    assert not thread.is_alive(), "the hour-long wait ended when the stop was set"
    assert isinstance(outcome["error"], RequestStopped) and tries == [4000] and stop.waits == [3600.0]
    assert receipts("down") == ["review-001.json"]

    # A timeout, the stop set while that try was out: no retry at half the budget.
    stop, tries = Watched(), []

    def timed_out(url, **kwargs):
        tries.append(kwargs["json"]["max_tokens"])
        stop.set()
        raise httpx.ReadTimeout("timed out")

    monkeypatch.setattr(review_transport.httpx, "post", timed_out)
    with pytest.raises(RequestStopped):
        transport("slow", stop)("fixture", "prompt")
    assert tries == [4000] and receipts("slow") == ["review-001.json"]
    assert json.loads((tmp_path / "slow" / "review-001.json").read_text())["error"].startswith("ReadTimeout")
    assert slept == [], "with a stop the transport waits on it, never in time.sleep"

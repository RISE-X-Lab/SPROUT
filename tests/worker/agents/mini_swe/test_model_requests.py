"""mini's model requests: where they go, what they carry, and what a failed request does to the run.

The ``backoff`` fixture (conftest.py) records the waits between attempts instead of sleeping them.
"""

from __future__ import annotations

import socket
from typing import Any

import pytest

from sprout.worker.gateway import GatewayServer, ModelRoute, RoutingTable
from sprout.worker.run.journal import JournalWriter
from tests.support.http import chat_completion, chat_server
from tests.worker.agents.mini_swe.rollouts import (
    BASH_TOOL,
    SUBMIT,
    ShellSandbox,
    mini_task,
    needs_mini,
    run_mini,
    run_slot,
)

pytestmark = needs_mini

#: sglang's refusal of a prompt and a completion budget past the context window.
SGLANG = "Requested token count exceeds the model's maximum context length of 262144 tokens."
#: OpenAI's code for the same refusal.
OPENAI = "This model's context is too long (context_length_exceeded)."
#: vLLM's wording of it.
VLLM = "The prompt (262203 tokens) is longer than the model's context length (262144 tokens)."


def refusal(status: int, message: str = "refused") -> tuple[int, dict[str, object]]:
    return (status, {"error": {"message": message, "type": "invalid_request_error"}})


def with_finish_reason(reply: dict[str, Any], finish_reason: str) -> dict[str, Any]:
    reply["choices"][0]["finish_reason"] = finish_reason
    return reply


@pytest.mark.parametrize("base_path", ["", "/v1", "/v1/"])
def test_the_request_goes_to_chat_completions_under_the_base_url_with_or_without_v1(tmp_path, base_path):
    rollout = run_mini(tmp_path, [chat_completion(SUBMIT)], base_path=base_path)

    assert (rollout.result.status, len(rollout.requests)) == ("completed", 1)


def test_a_run_without_a_base_url_is_an_error(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)

    rollout = run_mini(tmp_path, [chat_completion(SUBMIT)], base_path=None)

    assert (rollout.result.status, rollout.result.error, rollout.requests) == (
        "error",
        "ValueError: Set OPENAI_BASE_URL or configure the Sprout model gateway",
        [],
    )


def test_a_request_carries_the_bash_tool_and_the_sampling_parameters_unchanged(tmp_path):
    sampling = {
        "temperature": 0.25,
        "top_p": 0.9,
        "top_k": 20,
        "max_tokens": 300,
        "stop": ["</done>"],
        "parallel_tool_calls": True,
        "reasoning_effort": "high",
    }

    rollout = run_mini(
        tmp_path, [chat_completion(SUBMIT)], extra={"mini": {"model": {"model_kwargs": sampling}}}
    )

    request = rollout.requests[0]
    assert request == {
        "model": "fixture",
        "messages": request["messages"],
        "tools": [BASH_TOOL],
        **sampling,
        "stream": False,
    }


def test_a_run_reaches_its_model_through_the_gateway_with_the_slot_token_of_its_task(tmp_path):
    """The runner points the agent at the gateway with ``GatewayServer.env_for``."""
    with chat_server([chat_completion(SUBMIT)]) as server:
        table = RoutingTable({"default": ModelRoute(base_url=server.url)})
        token = table.mint("agent-1")
        with (
            JournalWriter(tmp_path / "gateway.jsonl") as journal,
            GatewayServer(table, journal=journal) as gateway,
        ):
            task = mini_task(base_url=None, home=tmp_path / "native-home", env=gateway.env_for(token))
            rollout = run_slot(task, ShellSandbox(tmp_path / "sandbox"), tmp_path / "journal.jsonl")

    assert (rollout.result.status, len(server.requests)) == ("completed", 1)


@pytest.mark.parametrize("status", [403, 408, 429, 500, 502, 503, 504])
def test_a_transient_status_is_retried_after_a_wait(tmp_path, status):
    """An endpoint answers 403 to refuse a request for capacity, and the request succeeds moments later."""
    rollout = run_mini(tmp_path, [refusal(status), chat_completion(SUBMIT)])

    assert (rollout.result.status, rollout.events("model.request.retry")) == (
        "completed",
        [{"type": "model.request.retry", "status_code": status, "attempt": 1, "delay_s": 2.0}],
    )


@pytest.mark.parametrize("code, status", [(504, 504), (None, 502)], ids=["its-code", "no-code"])
def test_a_failed_choice_in_a_200_body_is_retried_as_its_status(tmp_path, code, status):
    """OpenRouter answers 200 first. Its provider's idle timeout then ends the only choice as a 504."""
    failed = with_finish_reason(chat_completion(content=""), "error")
    failed["choices"][0]["error"] = {"code": code, "message": "Upstream idle timeout exceeded"}

    rollout = run_mini(tmp_path, [failed, chat_completion(SUBMIT)])

    assert (rollout.result.status, rollout.events("model.request.retry")) == (
        "completed",
        [{"type": "model.request.retry", "status_code": status, "attempt": 1, "delay_s": 2.0}],
    )


@pytest.mark.parametrize("status", [400, 401, 404])
def test_a_status_that_no_retry_can_change_ends_the_run_at_once(tmp_path, status):
    rollout = run_mini(tmp_path, [refusal(status), chat_completion(SUBMIT)])

    assert (rollout.result.status, len(rollout.requests), rollout.events("model.request.failed")) == (
        "error",
        1,
        [{"type": "model.request.failed", "status_code": status}],
    )


@pytest.mark.parametrize("message", [SGLANG, OPENAI, VLLM], ids=["sglang", "openai", "vllm"])
def test_a_refusal_for_the_context_length_ends_the_rollout_gradable(tmp_path, message):
    """A conversation past the context window ends like a spent turn budget. The workspace is graded."""
    rollout = run_mini(tmp_path, [refusal(400, message), chat_completion(SUBMIT)])

    assert (
        rollout.result.status,
        rollout.result.error,
        rollout.control.stop_reason,
        len(rollout.requests),
    ) == (
        "timeout",
        "model context window exhausted",
        "context_exhausted",
        1,
    )
    assert rollout.events("model.request.failed") == [
        {"type": "model.request.failed", "status_code": 400, "error_type": "context_length_exceeded"}
    ]


@pytest.mark.parametrize(
    "failures, status",
    [(11, "completed"), (12, "error")],
    ids=["twelfth-attempt-succeeds", "all-twelve-fail"],
)
def test_a_request_has_twelve_attempts(tmp_path, failures, status):
    rollout = run_mini(tmp_path, [refusal(503)] * failures + [chat_completion(SUBMIT)])

    assert (rollout.result.status, len(rollout.requests)) == (status, 12)


def test_a_request_that_cannot_connect_is_retried_until_its_attempts_are_spent(tmp_path):
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        closed = "http://127.0.0.1:%d" % listener.getsockname()[1]
    task = mini_task(base_url=closed, home=tmp_path / "native-home")

    rollout = run_slot(task, ShellSandbox(tmp_path / "sandbox"), tmp_path / "journal.jsonl")

    retries = rollout.events("model.request.retry")
    assert (rollout.result.status, len(retries), retries[:1], rollout.events("model.request.failed")) == (
        "error",
        11,
        [
            {
                "type": "model.request.retry",
                "status_code": None,
                "error_type": "ConnectError",
                "attempt": 1,
                "delay_s": 2.0,
            }
        ],
        [{"type": "model.request.failed", "error_type": "ConnectError"}],
    )


def test_the_wait_between_attempts_doubles_up_to_32_seconds(tmp_path, backoff):
    run_mini(tmp_path, [refusal(503)] * 6 + [chat_completion(SUBMIT)])

    assert backoff == [2.0, 4.0, 8.0, 16.0, 32.0, 32.0]


def test_a_response_without_a_valid_terminal_choice_is_an_error_whose_usage_still_counts(tmp_path):
    """A malformed completion can still consume tokens."""
    rollout = run_mini(tmp_path, [with_finish_reason(chat_completion(SUBMIT), "content_filter")])

    assert (rollout.result.status, rollout.result.error, rollout.result.usage["input_tokens"]) == (
        "error",
        "ValueError: mini model response is missing a valid terminal choice",
        11,
    )
    assert len(rollout.events("raw.mini-swe-agent")) == 1

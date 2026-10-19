"""``chat_server`` answers an OpenAI client with the scripted replies, and keeps each request."""

from __future__ import annotations

from typing import Any

import httpx

from tests.support.http import CHAT_PATH, chat_completion, chat_server


def post(url: str, body: dict[str, Any], *, path: str = CHAT_PATH) -> httpx.Response:
    return httpx.post(url + path, json=body, timeout=10)


def tool_call_ids(response: httpx.Response) -> list[str]:
    return [call["id"] for call in response.json()["choices"][0]["message"]["tool_calls"]]


def test_the_replies_come_in_the_order_of_the_script():
    script = [chat_completion("ls", ids=("first",)), chat_completion("pwd", ids=("second",))]

    with chat_server(script) as server:
        first, second = post(server.url, {"turn": 1}), post(server.url, {"turn": 2})

    assert (tool_call_ids(first), tool_call_ids(second)) == (["first"], ["second"])


def test_the_server_keeps_the_body_of_each_request_in_arrival_order():
    with chat_server([chat_completion("ls"), chat_completion("pwd")]) as server:
        post(server.url, {"turn": 1})
        post(server.url, {"turn": 2})

    assert server.requests == [{"turn": 1}, {"turn": 2}]


def test_tool_calls_without_ids_are_numbered_across_the_replies_in_send_order():
    """Tool call ids must be unique across the history of a rollout (sprout.contract.mini_history)."""
    with chat_server([chat_completion("ls", "pwd"), chat_completion("cat answer")]) as server:
        first, second = post(server.url, {"turn": 1}), post(server.url, {"turn": 2})

    assert (tool_call_ids(first), tool_call_ids(second)) == (["call-1", "call-2"], ["call-3"])


def test_a_status_and_body_reply_answers_with_that_status_and_body():
    refusal = {"error": {"message": "rate limited"}}

    with chat_server([(429, refusal)]) as server:
        response = post(server.url, {"turn": 1})

    assert (response.status_code, response.json()) == (429, refusal)


def test_a_request_after_the_last_reply_gets_a_500_that_says_so():
    with chat_server([]) as server:
        response = post(server.url, {"turn": 1})

    assert (response.status_code, response.json()) == (
        500,
        {"error": {"message": "the chat script has no reply left"}},
    )


def test_a_request_to_another_path_gets_a_404_and_is_not_kept():
    with chat_server([chat_completion("ls")]) as server:
        response = post(server.url, {"turn": 1}, path="/v1/completions")

    assert (response.status_code, server.requests) == (404, [])

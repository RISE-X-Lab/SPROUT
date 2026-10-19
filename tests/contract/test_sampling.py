"""The sampling controls that a rollout request may carry, and the model endpoint that it names."""

from __future__ import annotations

import re

import pytest

from sprout.contract.sampling import endpoint, parameters, validate


@pytest.mark.parametrize(
    "value, prefix",
    [
        ("https://api.example.com/v1", "https://api.example.com"),
        ("https://api.example.com/v1/", "https://api.example.com"),
        ("http://127.0.0.1:30000/", "http://127.0.0.1:30000"),
        ("https://gateway.example.com/openai/v1", "https://gateway.example.com/openai"),
        ("https://gateway.example.com/v1beta", "https://gateway.example.com/v1beta"),
    ],
)
def test_the_endpoint_is_the_prefix_before_v1_chat_completions(value, prefix):
    assert endpoint(value) == prefix


@pytest.mark.parametrize(
    "value",
    [
        "ftp://models.example.com",
        "http://",
        "https://key@api.example.com",
        "https://:secret@api.example.com",
        "https://api.example.com/v1?key=secret",
        "https://api.example.com/v1#chat",
    ],
    ids=["ftp", "no-host", "user", "password", "query", "fragment"],
)
def test_an_endpoint_that_is_not_a_plain_http_url_is_refused(value):
    """A credential in the URL would reach the journal and the stored request."""
    with pytest.raises(
        ValueError, match="^Rollout endpoint must be HTTP\\(S\\), without credentials/query/fragment$"
    ):
        endpoint(value)


@pytest.mark.parametrize(
    "controls",
    [
        {},
        {"temperature": 0},
        {"temperature": 2.5},
        {"top_p": 1},
        {"top_p": 0.05},
        {"top_k": -1},
        {"top_k": 0},
        {"max_tokens": 1},
        {"max_tokens": 300, "max_new_tokens": 300, "max_output_tokens": 300},
        {"stop": "END"},
        {"stop": ["END", "STOP"]},
        {"stop": []},
        {"stop": None},
    ],
    ids=str,
)
def test_controls_at_and_inside_their_limits_are_accepted(controls):
    assert validate(controls) is None


@pytest.mark.parametrize("effort", ["none", "minimal", "low", "medium", "high", "xhigh", "max", "ultra"])
def test_each_reasoning_effort_that_a_model_api_names_is_accepted(effort):
    assert validate({"reasoning_effort": effort}) is None


@pytest.mark.parametrize(
    "controls, error",
    [
        pytest.param(["temperature"], "Unknown message sampling controls", id="names-not-an-object"),
        pytest.param({"stop_token_ids": [7]}, "Unknown message sampling controls", id="token-ids"),
        pytest.param({"temperature": -0.1}, "Invalid temperature", id="negative-temperature"),
        pytest.param({"temperature": float("nan")}, "Invalid temperature", id="nan-temperature"),
        pytest.param({"temperature": float("inf")}, "Invalid temperature", id="infinite-temperature"),
        pytest.param({"temperature": True}, "Invalid temperature", id="boolean-temperature"),
        pytest.param({"top_p": 0}, "Invalid top_p", id="zero-top-p"),
        pytest.param({"top_p": 1.01}, "Invalid top_p", id="top-p-over-one"),
        pytest.param({"top_k": -2}, "top_k must be -1 or a nonnegative integer", id="top-k-below-minus-one"),
        pytest.param({"top_k": 20.0}, "top_k must be -1 or a nonnegative integer", id="float-top-k"),
        pytest.param({"top_k": True}, "top_k must be -1 or a nonnegative integer", id="boolean-top-k"),
        pytest.param({"reasoning_effort": "extreme"}, "Invalid reasoning_effort", id="unknown-effort"),
        pytest.param({"max_tokens": 0}, "Output token limits must be positive and agree", id="zero-tokens"),
        pytest.param(
            {"max_new_tokens": True}, "Output token limits must be positive and agree", id="boolean-tokens"
        ),
        pytest.param(
            {"max_output_tokens": 300.0}, "Output token limits must be positive and agree", id="float-tokens"
        ),
        pytest.param(
            {"max_tokens": 300, "max_new_tokens": 200},
            "Output token limits must be positive and agree",
            id="disagree",
        ),
        pytest.param(
            {"max_tokens": 300, "max_new_tokens": 300, "max_output_tokens": 200},
            "Output token limits must be positive and agree",
            id="third-disagrees",
        ),
        pytest.param({"stop": 7}, "stop must be text, a text list, or null", id="number-stop"),
        pytest.param({"stop": ["END", 7]}, "stop must be text, a text list, or null", id="mixed-stop"),
    ],
)
def test_a_control_outside_its_limits_is_refused(controls, error):
    with pytest.raises(ValueError, match="^%s$" % re.escape(error)):
        validate(controls)


@pytest.mark.parametrize(
    "controls, fields",
    [
        ({"temperature": 0.8, "max_new_tokens": 300}, {"temperature": 0.8, "max_tokens": 300}),
        ({"max_output_tokens": 300}, {"max_tokens": 300}),
        ({"max_tokens": 300, "max_new_tokens": 300, "max_output_tokens": 300}, {"max_tokens": 300}),
        (
            {"top_k": 20, "top_p": 0.95, "stop": ["END"], "reasoning_effort": "high"},
            {"top_k": 20, "top_p": 0.95, "stop": ["END"], "reasoning_effort": "high"},
        ),
    ],
    ids=["max-new-tokens", "max-output-tokens", "all-three-limits", "other-controls"],
)
def test_the_request_fields_carry_the_output_limit_once_as_max_tokens(controls, fields):
    assert parameters(controls) == fields


def test_the_request_fields_leave_the_controls_as_they_were():
    controls = {"temperature": 0.8, "max_new_tokens": 300}

    parameters(controls)

    assert controls == {"temperature": 0.8, "max_new_tokens": 300}


def test_the_request_fields_of_refused_controls_are_refused():
    with pytest.raises(ValueError, match="^Unknown message sampling controls$"):
        parameters({"stop_token_ids": [7]})

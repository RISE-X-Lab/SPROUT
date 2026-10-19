"""A model endpoint takes one credential: the bearer key in the environment variable that a run names."""

from __future__ import annotations

import pytest

from sprout.worker.gateway.auth import bearer_headers, bearer_key


def test_the_key_is_the_value_of_the_named_variable(monkeypatch):
    monkeypatch.setenv("SPROUT_TEST_MODEL_KEY", "provider-secret")

    key = bearer_key("SPROUT_TEST_MODEL_KEY")

    assert key == "provider-secret"


@pytest.mark.parametrize(
    "prepare",
    [
        lambda monkeypatch: monkeypatch.delenv("SPROUT_TEST_MODEL_KEY", raising=False),
        lambda monkeypatch: monkeypatch.setenv("SPROUT_TEST_MODEL_KEY", ""),
    ],
    ids=["unset", "empty"],
)
def test_an_unset_or_empty_variable_is_an_error_not_an_anonymous_request(monkeypatch, prepare):
    prepare(monkeypatch)

    with pytest.raises(ValueError, match="^Model key environment variable SPROUT_TEST_MODEL_KEY is unset$"):
        bearer_key("SPROUT_TEST_MODEL_KEY")


@pytest.mark.parametrize("name", [None, "", "QWEN-API-KEY", "$KEY", "MODEL KEY", "1KEY"])
def test_a_key_must_name_an_environment_variable(name):
    with pytest.raises(ValueError, match="^Model key must name an environment variable$"):
        bearer_key(name)


@pytest.mark.parametrize(
    "key, headers",
    [
        ("provider-secret", {"Authorization": "Bearer provider-secret"}),
        ("", {}),
        (None, {}),
    ],
    ids=["key", "empty", "none"],
)
def test_the_authorization_header_carries_the_key_and_is_absent_without_one(key, headers):
    """A local server without a key gets a request with no ``Authorization`` header at all."""
    assert bearer_headers(key) == headers

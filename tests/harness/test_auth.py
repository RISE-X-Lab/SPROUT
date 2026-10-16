"""The bearer key is the only model credential, and it is checked before it is spent."""

import pytest

from sprout.harness_gateway.auth import bearer_headers, bearer_key


def test_bearer_key_reads_the_named_variable(monkeypatch):
    monkeypatch.setenv("SPROUT_TEST_MODEL_KEY", "provider-secret")
    assert bearer_key("SPROUT_TEST_MODEL_KEY") == "provider-secret"
    assert bearer_headers(bearer_key("SPROUT_TEST_MODEL_KEY")) == {"Authorization": "Bearer provider-secret"}


@pytest.mark.parametrize("value", [None, ""])
def test_bearer_key_raises_on_an_unset_or_empty_variable(monkeypatch, value):
    """No credential chain to fall back on: an unset key is an error, not an anonymous request."""
    if value is None:
        monkeypatch.delenv("SPROUT_TEST_MODEL_KEY", raising=False)
    else:
        monkeypatch.setenv("SPROUT_TEST_MODEL_KEY", value)
    with pytest.raises(ValueError, match="Model key environment variable SPROUT_TEST_MODEL_KEY is unset"):
        bearer_key("SPROUT_TEST_MODEL_KEY")


@pytest.mark.parametrize("name", ["", "not a name", "QWEN-API-KEY", "$KEY", None])
def test_bearer_key_requires_an_environment_variable_name(name):
    with pytest.raises(ValueError, match="must name an environment variable"):
        bearer_key(name)


def test_bearer_headers_are_empty_without_a_key():
    """``harness run`` by hand may talk to a local server with no key at all."""
    assert bearer_headers(None) == {} and bearer_headers("") == {}

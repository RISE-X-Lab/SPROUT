"""The gateway's lookups: the route of a model name, the price of a request, and the slot of a token."""

from __future__ import annotations

from collections.abc import Callable

import pytest

from sprout.worker.gateway.routing import ModelRoute, RoutingTable, SlotToken
from sprout.worker.run.events import Usage

#: USD per million tokens. Each class of token has its own rate.
PRICING = {"input": 3.0, "output": 15.0, "cache_read": 0.3, "cache_write": 3.75}


def route(name: str) -> ModelRoute:
    return ModelRoute(base_url="http://%s.invalid" % name)


def test_cached_input_is_priced_at_the_cache_read_rate_and_not_also_as_input():
    """``cached_input_tokens`` is a subset of ``input_tokens``, so only the rest pays the input rate."""
    usage = Usage(input_tokens=1000, output_tokens=100, cached_input_tokens=400, cache_creation_tokens=200)

    cost = ModelRoute(base_url="http://up.invalid", pricing=PRICING).price(usage)

    # 600 uncached * 3 + 100 * 15 + 400 * 0.3 + 200 * 3.75, per million tokens.
    assert cost == pytest.approx(0.00417)


def test_the_key_of_a_route_wins_over_its_key_variable(monkeypatch):
    monkeypatch.setenv("SPROUT_TEST_ROUTE_KEY", "from-the-environment")
    configured = ModelRoute(
        base_url="http://up.invalid", api_key="configured", api_key_env="SPROUT_TEST_ROUTE_KEY"
    )

    assert configured.resolve_key() == "configured"


def test_the_key_variable_of_a_route_is_read_when_the_key_is_needed(monkeypatch):
    """A route made before its variable is set still finds the key."""
    monkeypatch.delenv("SPROUT_TEST_ROUTE_KEY", raising=False)
    lazy = ModelRoute(base_url="http://up.invalid", api_key_env="SPROUT_TEST_ROUTE_KEY")

    monkeypatch.setenv("SPROUT_TEST_ROUTE_KEY", "set-later")

    assert lazy.resolve_key() == "set-later"


#: The ways in which the key variable of a route holds no key.
NO_KEY: dict[str, Callable[[pytest.MonkeyPatch], None]] = {
    "unset": lambda environment: environment.delenv("SPROUT_TEST_ROUTE_KEY", raising=False),
    "empty": lambda environment: environment.setenv("SPROUT_TEST_ROUTE_KEY", ""),
}


@pytest.mark.parametrize("no_key", NO_KEY.values(), ids=NO_KEY.keys())
def test_a_route_whose_key_variable_holds_no_key_refuses_to_give_one(monkeypatch, no_key):
    """A request without the key that the route names would reach the provider as an anonymous one."""
    no_key(monkeypatch)
    route = ModelRoute(base_url="http://up.invalid", api_key_env="SPROUT_TEST_ROUTE_KEY")

    with pytest.raises(ValueError, match="^Model key environment variable SPROUT_TEST_ROUTE_KEY is unset$"):
        route.resolve_key()


@pytest.mark.parametrize(
    "budget_usd, spent_usd, over",
    [(None, 1000.0, False), (2.0, 1.99, False), (2.0, 2.0, True), (2.0, 2.01, True)],
    ids=["no-budget", "below", "at", "above"],
)
def test_a_token_is_over_budget_once_its_spend_reaches_the_budget(budget_usd, spent_usd, over):
    token = SlotToken(
        token="sprout-slot-fixture", agent_id="agent", budget_usd=budget_usd, spent_usd=spent_usd
    )

    assert token.over_budget() is over


@pytest.mark.parametrize(
    "model, expected",
    [("checkpoint-42", "checkpoint-42"), ("unmapped-model", "default"), (None, "default")],
    ids=["named", "unmapped", "no-model"],
)
def test_a_model_without_its_own_route_takes_the_default_route(model, expected):
    routes = {"default": route("default"), "checkpoint-42": route("named")}

    chosen = RoutingTable(routes).route_for(model)

    assert chosen is routes[expected]


def test_without_a_default_route_an_unmapped_model_has_no_route():
    table = RoutingTable({"checkpoint-42": route("named")})

    with pytest.raises(KeyError, match="no route for model 'unmapped-model' and no default route"):
        table.route_for("unmapped-model")


@pytest.mark.parametrize(
    "presented",
    ["{token}", "Bearer {token}", "bearer {token}", "  Bearer   {token}  "],
    ids=["bare", "bearer", "lower-case", "padded"],
)
def test_a_minted_token_is_found_with_or_without_its_bearer_prefix(presented):
    table = RoutingTable()
    token = table.mint("agent-a", run_id="run-1", budget_usd=5.0)

    found = table.lookup(presented.format(token=token.token))

    assert found is token


@pytest.mark.parametrize("presented", [None, "Bearer ", "sprout-slot-never-minted"])
def test_a_credential_that_no_mint_returned_finds_no_slot(presented):
    table = RoutingTable()
    table.mint("agent-a")

    assert table.lookup(presented) is None


def test_two_mints_for_one_agent_give_two_tokens():
    """Attribution and budgets are per slot, so a token never stands for two slots."""
    table = RoutingTable()

    first, second = table.mint("agent-a"), table.mint("agent-a")

    assert first.token != second.token
    assert [table.lookup(first.token), table.lookup(second.token)] == [first, second]


def test_a_minted_token_carries_24_random_bytes():
    """An agent that could guess another slot's token could spend that slot's budget."""
    token = RoutingTable().mint("agent-a").token

    assert (token[: len("sprout-slot-")], len(token) - len("sprout-slot-")) == ("sprout-slot-", 32)


def test_charges_add_up_on_the_token():
    table = RoutingTable()
    token = table.mint("agent-a", budget_usd=1.0)

    table.charge(token, 0.25)
    table.charge(token, 0.5)

    assert token.spent_usd == pytest.approx(0.75)

"""The two lookups of the gateway: model routes and per-slot tokens.

``RoutingTable.route_for`` maps the model name in a request to a ``ModelRoute``: an
upstream base URL, its credential and an optional model rename. A table with only a
``default`` route sends every request to one upstream, whatever model it names.

``RoutingTable.lookup`` maps the credential that an agent presents to a ``SlotToken``:
an ``agent_id`` and a budget. The caller mints one token per slot, so attribution and
the budget do not depend on what the agent says about itself.
"""

from __future__ import annotations

import secrets
import threading
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from sprout.worker.gateway.auth import bearer_key

if TYPE_CHECKING:
    from sprout.worker.run.events import Usage

DEFAULT_ROUTE = "default"


@dataclass
class ModelRoute:
    """The upstream that the requests for one model name go to, with its credential and prices."""

    base_url: str
    #: The upstream key. The agent never sees it, only its own slot token.
    api_key: str | None = None
    #: The environment variable that holds the upstream key, when ``api_key`` is unset.
    api_key_env: str | None = None
    #: The model name that the gateway sends upstream in place of the requested one.
    upstream_model: str | None = None
    #: USD per million tokens, keyed ``input``, ``output``, ``cache_read`` and ``cache_write``.
    #: Without it, a budget binds only when the provider reports a cost (``docs/MODEL_ENDPOINTS.md``).
    pricing: dict[str, float] = field(default_factory=dict)

    def price(self, usage: Usage) -> float:
        """The USD cost of one request's ``usage`` at this route's prices. 0.0 without pricing.

        The ``input`` rate applies to uncached input tokens only. ``cached_input_tokens``
        is a subset of ``input_tokens``, and the ``cache_read`` rate applies to it.
        """
        if not self.pricing:
            return 0.0
        rate = {k: float(v) / 1_000_000 for k, v in self.pricing.items()}
        uncached = max(0, usage.input_tokens - usage.cached_input_tokens)
        return (
            uncached * rate.get("input", 0.0)
            + usage.output_tokens * rate.get("output", 0.0)
            + usage.cached_input_tokens * rate.get("cache_read", 0.0)
            + usage.cache_creation_tokens * rate.get("cache_write", 0.0)
        )

    def resolve_key(self) -> str | None:
        """The upstream key: ``api_key``, else the value of ``api_key_env``, else None.

        Raises ValueError if ``api_key_env`` names an unset or empty variable. A request never goes out
        without the key that its route names. A route without either key sends none."""
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return bearer_key(self.api_key_env)
        return None


@dataclass
class SlotToken:
    """A minted credential that identifies one slot, with its budget."""

    token: str
    agent_id: str
    run_id: str | None = None
    #: The budget in USD, or None for no limit. The gateway checks it before it forwards a request.
    budget_usd: float | None = None
    spent_usd: float = 0.0

    def over_budget(self) -> bool:
        """True when the token has a budget and ``spent_usd`` has reached it."""
        return self.budget_usd is not None and self.spent_usd >= self.budget_usd


class RoutingTable:
    """The routes and the slot tokens of one gateway. Each method is safe to call from many threads."""

    def __init__(self, routes: dict[str, ModelRoute] | None = None) -> None:
        self._routes: dict[str, ModelRoute] = dict(routes or {})
        self._tokens: dict[str, SlotToken] = {}
        self._lock = threading.Lock()

    # --- routes ------------------------------------------------------------
    def route_for(self, model: str | None) -> ModelRoute:
        """The route of ``model``, else the ``default`` route. Raises KeyError if neither exists."""
        if model and model in self._routes:
            return self._routes[model]
        if DEFAULT_ROUTE not in self._routes:
            raise KeyError("no route for model %r and no default route" % model)
        return self._routes[DEFAULT_ROUTE]

    # --- slot tokens -------------------------------------------------------
    def mint(
        self,
        agent_id: str,
        *,
        run_id: str | None = None,
        budget_usd: float | None = None,
    ) -> SlotToken:
        """A new slot token for ``agent_id``, which ``lookup`` finds from now on."""
        token = SlotToken(
            token="sprout-slot-" + secrets.token_urlsafe(24),
            agent_id=agent_id,
            run_id=run_id,
            budget_usd=budget_usd,
        )
        with self._lock:
            self._tokens[token.token] = token
        return token

    def lookup(self, presented: str | None) -> SlotToken | None:
        """The slot token that ``presented`` names, with or without a ``Bearer`` prefix, or None."""
        if not presented:
            return None
        candidate = presented.strip()
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()
        with self._lock:
            return self._tokens.get(candidate)

    def charge(self, token: SlotToken, cost_usd: float) -> None:
        """Add ``cost_usd`` to the ``spent_usd`` of ``token``."""
        with self._lock:
            token.spent_usd += cost_usd

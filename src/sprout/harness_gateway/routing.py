"""Model routing + per-slot tokens.

Two lookups, deliberately separate:

- ``RoutingTable`` maps the model name an agent asks for to an upstream
  (base URL + credential + optional rename). The orchestrator builds one route
  per run from the rollout contract (``harness_orchestrator.run``): whatever
  name the agent asks for, the request goes to the contract's endpoint as the
  contract's model, with the contract's credential.
- ``SlotToken`` maps the credential an agent presents to an ``agent_id`` and a
  budget. The orchestrator mints one per slot, so attribution and enforcement do
  not depend on the agent cooperating.
"""

from __future__ import annotations

import os
import secrets
import threading
from dataclasses import dataclass, field

DEFAULT_ROUTE = "default"


@dataclass
class ModelRoute:
    """Where a requested model actually goes."""

    base_url: str
    #: Literal key, or ``api_key_env`` to read one from the environment. Keeping
    #: the credential here means the agent never holds a provider key -- it only
    #: ever sees its own slot token.
    api_key: str | None = None
    api_key_env: str | None = None
    #: Rewrite the model name on the way out (agent asks X, upstream wants Y).
    upstream_model: str | None = None
    #: USD per million tokens, e.g. ``{"input": 3.0, "output": 15.0,
    #: "cache_read": 0.30, "cache_write": 3.75}``. The provider's responses
    #: carry token counts but no price, so without this the gateway can count
    #: everything and charge nothing -- ``budget_usd`` then never binds unless
    #: the provider reports a cost in its usage block (OpenRouter does).
    pricing: dict[str, float] = field(default_factory=dict)

    def price(self, usage) -> float:
        """USD for one request's usage, 0.0 when this route has no pricing.

        ``input`` rates apply to uncached input only: providers bill cache reads
        at their own (much lower) rate, and ``cached_input_tokens`` is a subset
        of ``input_tokens``.
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
        if self.api_key:
            return self.api_key
        if self.api_key_env:
            return os.environ.get(self.api_key_env)
        return None


@dataclass
class SlotToken:
    """A minted credential identifying one slot, with its budget."""

    token: str
    agent_id: str
    run_id: str | None = None
    #: Hard ceiling in USD; None means unlimited. Enforced before forwarding.
    budget_usd: float | None = None
    spent_usd: float = 0.0

    def over_budget(self) -> bool:
        return self.budget_usd is not None and self.spent_usd >= self.budget_usd


class RoutingTable:
    """Routes + slot tokens. Thread-safe; the server is multi-threaded."""

    def __init__(self, routes: dict[str, ModelRoute] | None = None) -> None:
        self._routes: dict[str, ModelRoute] = dict(routes or {})
        self._tokens: dict[str, SlotToken] = {}
        self._lock = threading.Lock()

    # --- routes ------------------------------------------------------------
    def route_for(self, model: str | None) -> ModelRoute:
        """Exact model match, else ``default``; KeyError when neither exists."""
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
        if not presented:
            return None
        candidate = presented.strip()
        if candidate.lower().startswith("bearer "):
            candidate = candidate[7:].strip()
        with self._lock:
            return self._tokens.get(candidate)

    def charge(self, token: SlotToken, cost_usd: float) -> None:
        with self._lock:
            token.spent_usd += cost_usd

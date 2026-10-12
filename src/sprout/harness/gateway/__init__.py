"""Inference gateway: the model seam.

mini-swe-agent's model traffic reaches the provider through here, which buys:

1. **Model swap** -- point the agent at a different endpoint (another provider,
   a local vLLM, an RL checkpoint being trained) without touching the agent.
2. **Wire-level tap** -- what the model actually received and returned, with
   exact token counts, independent of what the agent reports.
3. **Budget hard-stop** -- refuse requests once a run is over budget.
4. **Rollout policy** -- a run's sampling parameters, turn limit and deadline,
   applied to every request (``harness.orchestrator.rollout``).

Not a policy engine for tools: tool calls are governed on the ToolAccess path.
The agent is wired with ``OPENAI_BASE_URL`` and a per-slot ``OPENAI_API_KEY``.
"""

from sprout.harness.gateway.routing import ModelRoute, RoutingTable, SlotToken
from sprout.harness.gateway.server import GatewayServer

__all__ = ["ModelRoute", "RoutingTable", "SlotToken", "GatewayServer"]

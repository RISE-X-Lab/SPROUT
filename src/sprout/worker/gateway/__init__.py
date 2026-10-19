"""The inference gateway: the HTTP endpoint between an agent and its model provider.

An agent reaches its model through the gateway, with ``OPENAI_BASE_URL`` and a
per-slot ``OPENAI_API_KEY`` (``GatewayServer.env_for``). Without a change to the
agent, the gateway gives a run these properties:

- The run chooses the upstream endpoint, model and credential (``RoutingTable``).
- The journal records the model, status, token counts and cost of each request,
  as the provider reported them, not as the agent reports them.
- The gateway forwards no request for a slot that has spent its budget.
- A request policy, such as ``sprout.worker.run.rollout.RolloutControls``, applies
  the run's sampling parameters, turn limit and deadline to each request.

The gateway does not govern tool calls. ``ToolAccess`` does.
"""

from sprout.worker.gateway.routing import ModelRoute, RoutingTable, SlotToken
from sprout.worker.gateway.server import GatewayServer

__all__ = ["ModelRoute", "RoutingTable", "SlotToken", "GatewayServer"]

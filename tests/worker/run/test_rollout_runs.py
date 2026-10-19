"""A rollout under a contract: a gateway between mini and the model applies the contract to each request.

mini runs on a ``DirectorySession`` and talks to a scripted model endpoint on a local port.
"""

from __future__ import annotations

from sprout.contract.messages import export_tools
from sprout.worker.jobs.rollout import execute
from sprout.worker.run.journal import read_journal
from sprout.worker.run.runner import Runner
from tests.support.http import chat_completion, chat_server
from tests.worker.run.directory_session import DirectorySession
from tests.worker.run.rollouts import SUBMIT, events, needs_mini, rollout_contract, rollout_spec
from tests.worker.run.upstream import provider

pytestmark = needs_mini

CONTRACT_FIELDS = ("model", "temperature", "max_tokens", "top_k", "stop")


def test_each_model_request_reaches_the_endpoint_of_the_contract_with_its_model_and_sampling_parameters(
    tmp_path, sandbox: DirectorySession
):
    with chat_server([chat_completion("printf ready > answer"), chat_completion(SUBMIT)]) as server:
        Runner().run(rollout_spec(tmp_path, server.url))

    assert [{key: request[key] for key in CONTRACT_FIELDS} for request in server.requests] == [
        {"model": "contract-model", "temperature": 0.4, "max_tokens": 7, "top_k": 11, "stop": ["stop-here"]},
        {"model": "contract-model", "temperature": 0.4, "max_tokens": 7, "top_k": 11, "stop": ["stop-here"]},
    ]


def test_the_request_past_the_turn_limit_never_reaches_the_endpoint_and_the_rollout_can_still_be_graded(
    tmp_path, sandbox: DirectorySession
):
    replies = [
        chat_completion("printf a > answer"),
        chat_completion("printf b > answer"),
        chat_completion(SUBMIT),
    ]
    with chat_server(replies) as server:
        outcome = Runner().run(
            rollout_spec(tmp_path, server.url, contract=rollout_contract(server.url, max_turns=2))
        )

    assert (len(server.requests), outcome.status, outcome.stop_reason, outcome.error) == (
        2,
        "timeout",
        "max_turns_reached",
        "rollout turn budget exhausted",
    )
    assert sandbox.snapshots[outcome.final_snapshot_id] == {"answer": b"b"}


def test_tool_calls_are_counted_and_not_limited_by_the_turn_limit(tmp_path, sandbox: DirectorySession):
    replies = [
        chat_completion("printf 1 > one", "printf 2 > two", "printf 3 > three"),
        chat_completion(SUBMIT),
    ]
    with chat_server(replies) as server:
        outcome = Runner().run(
            rollout_spec(tmp_path, server.url, contract=rollout_contract(server.url, max_turns=2))
        )

    [usage] = events(outcome, "rollout.usage")
    assert (outcome.status, usage["model_calls"], usage["tool_calls"]) == ("completed", 2, 4)
    assert sorted(path.name for path in sandbox.root.iterdir()) == ["one", "three", "two"]


def test_the_journal_holds_the_tools_that_the_model_was_offered_for_the_training_export(
    tmp_path, sandbox: DirectorySession
):
    with chat_server([chat_completion(SUBMIT)]) as server:
        outcome = Runner().run(rollout_spec(tmp_path, server.url))

    assert export_tools(read_journal(outcome.journal_path)) == server.requests[0]["tools"]


def test_a_budget_without_a_contract_keeps_the_agent_away_from_a_provider_that_it_could_reach(
    tmp_path, monkeypatch, sandbox: DirectorySession
):
    """The gateway of such a run has no route, so it refuses each request, and the ceiling holds."""
    with chat_server([chat_completion(SUBMIT)]) as server:
        monkeypatch.setenv("OPENAI_BASE_URL", server.url)
        spec = rollout_spec(tmp_path, server.url, budget_usd=1.0)
        del spec.extra["rollout_contract"]

        outcome = Runner().run(spec)

    assert (outcome.status, server.requests) == ("error", [])


def test_a_contract_that_names_no_key_sends_no_credential_to_its_endpoint(
    tmp_path, monkeypatch, sandbox: DirectorySession
):
    """An endpoint must never get the credential of another provider, nor the token of the slot."""
    monkeypatch.setenv("OPENAI_API_KEY", "other-provider-secret")
    with provider([chat_completion(SUBMIT, ids=("call-1",))]) as upstream:
        outcome = Runner().run(rollout_spec(tmp_path, upstream.url))

    assert (outcome.status, ["authorization" in request.headers for request in upstream.requests]) == (
        "completed",
        [False],
    )


def test_a_contract_whose_key_variable_is_unset_fails_the_run_before_its_first_model_request(
    tmp_path, monkeypatch, sandbox: DirectorySession
):
    """Without the check, the agent would meet the provider's 401 in the middle of its run."""
    monkeypatch.delenv("UPSTREAM_API_KEY", raising=False)
    with provider([chat_completion(SUBMIT, ids=("call-1",))]) as upstream:
        spec = rollout_spec(
            tmp_path, upstream.url, contract=rollout_contract(upstream.url, api_key_env="UPSTREAM_API_KEY")
        )

        outcome = Runner().run(spec)

    assert (outcome.status, outcome.error, upstream.requests) == (
        "error",
        "ValueError: Model key environment variable UPSTREAM_API_KEY is unset",
        [],
    )


def queued_rollout(url: str) -> dict:
    """The request of a queued rollout whose profile points the agent at ``http://must-not-bypass-gateway``."""
    return {
        "kind": "rollout",
        "attempt_id": "attempt-1",
        "effective_spec": {
            "prompt": "Write the answer.",
            "model": "fixture",
            "timeout_s": 30,
            "sandbox_image": "image",
            "extra": {"rollout_contract": rollout_contract(url, api_key_env="UPSTREAM_API_KEY")},
        },
        "profile_config": {
            "worker_env": {"UPSTREAM_API_KEY": {"$env": "RS_PROVIDER"}},
            "env": {"OPENAI_BASE_URL": "http://must-not-bypass-gateway"},
        },
    }


def test_a_queued_rollout_reaches_its_model_only_through_the_gateway_with_the_key_of_its_profile(
    tmp_path, monkeypatch, sandbox: DirectorySession
):
    """The agent's own environment cannot redirect it, and only the key that the contract names travels."""
    monkeypatch.setenv("RS_PROVIDER", "provider-fixture-secret")
    replies = [
        chat_completion("printf ready > answer", ids=("call-1",)),
        chat_completion(SUBMIT, ids=("call-2",)),
    ]
    with provider(replies) as upstream:
        result = execute(queued_rollout(upstream.url), tmp_path)

    assert (result["status"], [request.headers["authorization"] for request in upstream.requests]) == (
        "completed",
        ["Bearer provider-fixture-secret", "Bearer provider-fixture-secret"],
    )

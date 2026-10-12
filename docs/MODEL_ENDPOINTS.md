# Model endpoints

Sprout speaks one model protocol: OpenAI Chat Completions at
`<--model-endpoint>/v1/chat/completions` (a trailing `/v1` on the endpoint is
accepted), authenticated with the bearer key named by `--model-key-env`. The
actor's requests reach it through the run's gateway, the analyst's and
reviewer's through `ReviewTransport`, the BPO/Shepherd selectors' through
`policies.ModelClient`; all three send the same shape. There is no
provider-specific code: nothing is recognised from the URL, and an unset key is
an error before the first trial, never an anonymous request.

## What an endpoint has to provide

- **Chat Completions with function tools**, non-streaming: `tools`, assistant
  `tool_calls` with string `id`s, and `tool` messages carrying `tool_call_id`.
  The gateway refuses `stream: true`; mini never asks for it.
- **`reasoning_effort` passed through** (`--reasoning-effort`), and whatever
  reasoning the provider returns on an assistant message (`reasoning_content`,
  `provider_specific_fields`, thinking blocks) accepted back when that message
  is replayed on the next turn -- mini's native history keeps the message as
  received, and a branch replays it from a checkpoint.
- **`max_tokens`** as the output budget.
- **`usage`** with `prompt_tokens`, `completion_tokens`, and the splits the
  journal records: `prompt_tokens_details.cached_tokens` and
  `completion_tokens_details.reasoning_tokens`. A `cost` field (OpenRouter) is
  taken as authoritative for the budget; without one, `budget_usd` binds only
  through a route's `pricing`.
- **Errors as HTTP statuses**: a 400 whose text names the context length ends
  the rollout as `context_exhausted`; 408/429/5xx and dropped connections are
  retried by each transport with its own receipts.
- Optionally, **`response_format: {"type": "json_schema", ...}`** with
  `strict: true`, for `--structured-review`.

## Providers that are not that: the LiteLLM proxy

Providers that offer no such endpoint -- a native messages API, a model reached
only through a provider SDK, a service tier the Chat Completions route cannot
select -- are put behind a LiteLLM proxy, which presents each as a plain alias
on one local URL:

```bash
python -m sprout.services init              # seeds ~/.config/sprout/litellm.yaml (aliases luna, sonnet5, kimi-k3; fill in each route) and the master key
python -m sprout.services up proxy          # uv venv with litellm[proxy]==1.103.0, detached, waits for /health/liveliness
eval "$(python -m sprout.services env)"     # OPENAI_BASE_URL=http://127.0.0.1:4000/v1, OPENAI_API_KEY and SPROUT_LITELLM_KEY
python -m examples.qwen.run ... --model luna \
    --model-endpoint http://127.0.0.1:4000/v1 --model-key-env SPROUT_LITELLM_KEY --structured-review
```

The analyst and reviewer alone can go behind the proxy while the actor stays on
its own endpoint: `--review-model <alias> --review-endpoint
http://127.0.0.1:4000/v1 --review-key-env SPROUT_LITELLM_KEY` (and
`--review-effort`) name a second endpoint under the same contract; each defaults
to the actor's `--model`, `--model-endpoint`, `--model-key-env` and
`--reasoning-effort`.

`src/sprout/services/templates/litellm.yaml` (copied to `~/.config/sprout/litellm.yaml`
by `init`, never overwritten) explains each setting: one route per alias in LiteLLM's
`<provider>/<model id>` notation, with whatever the provider takes beside it (a
region, an `api_base`), `cache_control_injection_points` on the system and user
messages for Sonnet 5 (its provider caches only up to an explicit cache point, and
mini sets none), `service_tier: priority` for Kimi K3, `drop_params: false` (a field a model does not
take is refused, not silently dropped), `request_timeout: 3900` (above Sprout's
largest per-request timeout, `--review-timeout`'s default 3600 s, so a long review
hits Sprout's own timeout and its half-budget retry rather than a 408 from the proxy;
raise the two together), `num_retries: 0` (Sprout retries with its own receipts) and
the master key read from the environment -- a secret `init` generates once
(`~/.local/state/sprout/services/secrets/litellm-master-key`) and `up proxy` passes
as `LITELLM_MASTER_KEY`. Provider credentials come from the environment of the shell
that runs `up proxy`, which the proxy inherits; `unset_env` in `services.yaml` names
the variables removed from it first (the template removes `FORCE_COLOR`, which would
colour the log; add a provider variable that must not reach the proxy). `up proxy`
starts nothing when `/health/liveliness` already answers on the port (a proxy started
by hand is left alone rather than contended), reports the proxy up only once it
answers, and shows the log's last lines when it exits first; `docs/SERVICES.md` has
the rest (`status`, `logs proxy -f`, `restart proxy` after an alias edit).

**What the smoke verified (2026-10-09, litellm 1.103.0).** Through
`http://127.0.0.1:4000/v1`, each of `luna` (effort high), `sonnet5` (high) and
`kimi-k3` (max) ran `methods.evaluate --method sprout --max-turns 3 --rounds 1
--branches 2 --branch-count-mode fixed` on one DeepSWE task: a 3-turn root, the
analyst, the reviewer and two 3-turn branches, all exiting 0 with
`branch_schedule_complete: true`. Tool calls, `reasoning_effort`, reasoning fed
back across turns and into the branches, `usage` with cache-write and cache-read
token details (prompt-cache hits on the actor's requests) and Kimi's priority
tier all worked. `response_format: json_schema` was verified by a probe request
on each alias; the smoke's own reviews ran in the default plain-text mode. No
request of the smoke ran long enough to meet `request_timeout`, so that setting
is reasoned from Sprout's timeouts, not measured.

## `--structured-review`

Off, the analyst and reviewer answer in a labelled fenced block, parsed by the
controller (the behaviour every non-proxy run has had). On, `methods.evaluate`
builds its `ReviewTransport` with `supports_json_schema` and both are asked for
schema-constrained JSON (`response_format: json_schema`, `strict: true`, the
schemas in `methods/structured_review.py`); the schema's size is reserved from
`--prompt-budget-chars` and recorded with the plan. It is an explicit switch
because only the caller knows whether the endpoint honours `json_schema`;
behind the proxy it is recommended, as it removes the fence-parsing failures.
`summary.json` records `structured_review`, each `plan-roundN.json` records
whether that review was schema-constrained, and `examples/qwen/run.py` passes
the flag to every `methods.evaluate` and records it in the manifest.

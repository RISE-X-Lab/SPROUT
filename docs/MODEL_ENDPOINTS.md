# Model endpoints

Sprout calls each model through one protocol: OpenAI Chat Completions, with a bearer key. An
endpoint that serves this protocol with the features of
[What an endpoint has to provide](#what-an-endpoint-has-to-provide) can serve Sprout directly. Put
a provider without such an endpoint behind the LiteLLM proxy, which the services CLI runs. Read
this document before you point an experiment at a model.

Sprout has no provider-specific code. No caller recognises a provider from the URL, and no caller
falls back to another credential. The leak judge is the only model call outside this protocol.
It calls the API of TypeSafe.

## Three callers, one protocol

Three callers send the same request shape, each from its own process. The actor and the reviews
run on the worker, and the selectors run in the evaluation process. The figure shows each caller
with the flags that set its URL and its key.

<p align="center">
  <img src="figures/model-endpoints-request-paths.svg" width="1000" alt="Three callers send Chat Completions requests to one endpoint. In the attempt process of a rollout job on the worker host, mini's ChatModel, the actor, sends each request with its slot token to the run's gateway, sprout.worker.gateway. The gateway sends it on to the URL of --model-endpoint, with the key in the variable that --model-key-env names. In the attempt process of a review job, ReviewTransport sends the analyst's and the reviewer's requests to the URL of --review-endpoint, with the key in the variable that --review-key-env names. In the sprout.eval.evaluate process, policies.ModelClient sends the BPO and Shepherd selectors' requests to the URL of --model-endpoint, with the key in the variable that --model-key-env names. The endpoint is one of two: a provider's own Chat Completions endpoint, or the proxy service, LiteLLM at http://127.0.0.1:4000/v1. The proxy sends each request on to the route of its alias, a provider route in the provider's own API."/>
</p>
<p align="center"><sub>The three callers of a model, the flags that set their URL and their key, and the two kinds of endpoint.</sub></p>

The callers are these:

| Caller | Process | URL | Key variable |
|---|---|---|---|
| The actor: mini's `ChatModel`, through the gateway of the run | The attempt process of a rollout job, on the worker | `--model-endpoint` | `--model-key-env` |
| The analyst and the reviewer: `ReviewTransport` | The attempt process of a review job, on the worker | `--review-endpoint`, default `--model-endpoint` | `--review-key-env`, default `--model-key-env` |
| The BPO and Shepherd selectors: `policies.ModelClient` | The `sprout.eval.evaluate` process | `--model-endpoint` | `--model-key-env` |

The actor sends each request to the gateway with a slot token. For each request, the gateway
does these steps:

1. It checks the slot token. An unknown token gets HTTP 401.
2. It applies the rollout contract. A request after the deadline or past the turn limit gets
   HTTP 400.
3. It sends the request to the endpoint with the real key.

The agent never sees the real key, and the gateway never forwards the slot token.

A run of `python -m sprout.worker run` without `extra.rollout_contract` and without a
`budget_usd` starts no gateway. Then `ChatModel` posts to `OPENAI_BASE_URL` with
`OPENAI_API_KEY` itself.

Each caller retries some failures and records each try:

| Caller | Time for one try | Failures that it retries | Record |
|---|---|---|---|
| Actor | 600 s to the gateway. The gateway gives the upstream request 900 s. | HTTP 403, 408, 429, 500, 502, 503 and 504, transport errors and timeouts. Up to 12 tries within the rollout deadline, with a back-off of up to 32 s. | `model.request.retry` and `model.request.failed` events in the journal |
| Analyst and reviewer | `--review-timeout`, default 3600 s | HTTP 408, 429, 500, 502, 503 and 504, and transport errors: 3 more tries, after 10, 30 and 60 s. A timeout: one more try with half the output budget. | One `review-NNN.json` receipt for each try |
| Selectors | 1800 s | HTTP 408, 409, 425, 429, 500, 502, 503 and 504, timeouts, transport errors, a body without exactly one choice, and a failed choice. Up to 8 tries, with a back-off of up to 30 s. | One audit file for each request, with each try |

## The key of each request

Each caller reads its key from the environment variable that its flag names. The variable must
hold the key in the process that sends the request. The callers check the key in these ways:

- `sprout.eval.evaluate` refuses a `--model-key-env` or a `--review-key-env` that is not one
  variable name. It exits with status 2 before any job exists.
- For `--method bpo` or `shepherd`, `sprout.eval.evaluate` reads the key in its own process
  before the first job. If the variable is unset or empty, it raises `ValueError`.
  `examples.qwen.run` makes the same check when its `--method` includes a selector.
- `ReviewTransport` reads the key on the worker at the start of each call. If the variable is
  unset or empty, it raises `ValueError` and sends nothing.
- The runner reads the key on the worker when the rollout starts, before the agent runs. If the
  variable is unset or empty, the rollout fails with `ValueError` and sends no model request. The
  gateway reads the key again for each request of the actor. If the variable is unset or empty
  then, the gateway answers 502 and forwards nothing.
- A contract without `api_key_env` sends no `Authorization` header. A local server without a key
  needs this.

The attempt process of a job gets only these variables from the worker:

- the variables in `CHILD_ENV_PASSTHROUGH` of `src/sprout/worker/attempts.py`, for example
  `PATH` and `SPROUT_PODMAN_BIN`
- each variable that the profile or the spec of the job references as `{"$env": "<NAME>"}`

So the profile of the job in `runstore.json` must reference the key variable, for example in
its `worker_env` ([Run Store](RUNSTORE.md)). The worker itself must have the variable. If it
does not, the attempt ends with an error before its process starts. The profiles of the packaged
template reference `SPROUT_MODEL_API_KEY`. For the key of the proxy, add this entry to the
`worker_env` of each profile that calls the model, for example `mini` and `review`:

```json
"SPROUT_LITELLM_KEY": {"$env": "SPROUT_LITELLM_KEY"}
```

The `runstore-worker` service does not depend on `proxy`. So the worker has
`SPROUT_LITELLM_KEY` only if the shell that started it had the variable. For example, run
`eval "$(python -m sprout.services env)"` before `python -m sprout.services up runstore-worker`.

## What an endpoint has to provide

An endpoint must serve `POST <endpoint>/v1/chat/completions` with these features:

- **A plain URL.** The flag holds an HTTP or HTTPS URL with a host. The callers drop a trailing
  `/` or `/v1`. The URL has none of these:
  - credentials
  - a query
  - a fragment
- **Chat Completions with function tools, without streaming.** Each caller sends
  `stream: false`, and the gateway refuses `stream: true` with HTTP 400. The endpoint takes these
  parts of a request:
  - `tools`
  - assistant `tool_calls` with string `id`s
  - `tool` messages with `tool_call_id`
- **`reasoning_effort` passed through.** The flags are `--reasoning-effort` and
  `--review-effort`. When the next turn replays an assistant message, the endpoint must accept
  back the reasoning that it returned on that message. The native history of mini keeps each
  message as it arrived, and a branch replays it from a checkpoint. The reasoning takes these
  forms, for example:
  - `reasoning_content`
  - `provider_specific_fields`
  - thinking blocks
- **`max_tokens` as the output budget.** The flags are `--max-output-tokens` and
  `--review-output-tokens`.
- **`usage` with token counts.** The journal records these counts:
  - `prompt_tokens`
  - `completion_tokens`
  - `prompt_tokens_details.cached_tokens`
  - `completion_tokens_details.reasoning_tokens`
- **Errors as HTTP statuses.** A review that gets a 400 halves its prompt budget and tries again,
  at most 2 times. A 400 ends the rollout with the stop reason `context_exhausted` if its text
  matches one of these:
  - `maximum context length`
  - `context_length_exceeded`
  - `context length`

  The verifier still grades the workspace.

  OpenRouter answers 200 before its provider finishes. If the provider then fails, the only
  choice of the reply has `finish_reason` `error`. The actor and the review read such a reply as
  the status in the `code` of the choice's `error`, or as 502 without one. For example, a provider
  idle timeout is a 504, which they retry.
- **For `--structured-review`:** `response_format: {"type": "json_schema", ...}` with
  `strict: true`.
- **For `--method bpo`:** `logprobs: true` with `top_logprobs`. The reply must hold the log
  probabilities of its content tokens.

The gateway takes the cost of a request from `usage`. If `usage` has `cost` or `cost_usd`, the
gateway uses that value, as OpenRouter reports it. Otherwise the gateway prices the request with
the `pricing` of its route. The worker builds the route of a rollout without `pricing`. So a run's
`budget_usd` binds only when the endpoint reports a cost. If it does not, the gateway journals one
`budget_unenforceable` event.

## Providers without such an endpoint: the LiteLLM proxy

Put a provider behind the `proxy` service if it has no Chat Completions endpoint with these
features. The proxy presents each model as an alias on one local URL, with one key. These are
examples of such providers:

- a provider with a native messages API only
- a model that only a provider SDK reaches
- a service tier that the Chat Completions route cannot select

To run an experiment through the proxy, do these steps:

1. Run `python -m sprout.services init`. The command writes `~/.config/sprout/litellm.yaml` with
   three aliases, and it makes the master key.
   [The settings of `litellm.yaml`](#the-settings-of-litellmyaml) names the aliases.
2. In `~/.config/sprout/litellm.yaml`, set the route of each alias that you use. If you leave a
   route as the placeholder, the proxy logs `Error creating deployment` and starts anyway. The
   proxy then does not know that alias.
3. Export the credentials of each provider in your shell. The proxy inherits them.
4. Run `python -m sprout.services up proxy`. The first time, the command makes a venv with
   `litellm[proxy]==1.103.0`. It starts the proxy detached and waits for `/health/liveliness`.
5. Run `eval "$(python -m sprout.services env)"`. The shell gets these variables:
   - `OPENAI_BASE_URL=http://127.0.0.1:4000/v1`
   - `OPENAI_API_KEY`, the master key
   - `SPROUT_LITELLM_KEY`, the master key
6. Give the worker the key, as [The key of each request](#the-key-of-each-request) says.
7. Run the experiment against the proxy:

   ```bash
   python -m examples.qwen.run ... --model luna \
       --model-endpoint http://127.0.0.1:4000/v1 --model-key-env SPROUT_LITELLM_KEY \
       --structured-review
   ```

The analyst and the reviewer can go behind the proxy while the actor stays on its own endpoint.
These flags name a second endpoint under the same contract. Each flag defaults to the actor's
flag:

- `--review-model <alias>`, default `--model`
- `--review-endpoint http://127.0.0.1:4000/v1`, default `--model-endpoint`
- `--review-key-env SPROUT_LITELLM_KEY`, default `--model-key-env`
- `--review-effort`, default `--reasoning-effort`

The actor that continues each branch always uses `--model`.

### The settings of `litellm.yaml`

`init` copies `resources/services/litellm.yaml` to `~/.config/sprout/litellm.yaml` and never
overwrites it. The comments of the file explain each setting:

| Setting | Value in the template | Reason |
|---|---|---|
| `model_name` | `luna`, `sonnet5`, `kimi-k3` | The alias that `--model` names. |
| `litellm_params.model` | `<provider>/<model id>` | The route, in the notation of LiteLLM. Add what the provider takes beside it, for example a region or an `api_base`. |
| `cache_control_injection_points` of `sonnet5` | The system message and the last user message | The provider of Sonnet 5 caches only up to an explicit cache point, and mini sets none. The experiments saw 98% cache hits on the actor's requests with these two points. |
| `service_tier` of `kimi-k3` | `priority` | Kimi K3 is served in tiers. The experiments measured on `priority`. |
| `drop_params` | `false` | The proxy refuses a field that a model does not take, and does not drop it. A dropped `reasoning_effort` or `response_format` would change the experiment without a sign. |
| `request_timeout` | `3900` | Above the largest timeout that Sprout gives one request. The next paragraph gives the reason. |
| `num_retries` | `0` | Sprout retries with its own receipts. A retry inside the proxy would hide a provider error from the journal. |
| `master_key` | `os.environ/LITELLM_MASTER_KEY` | The generated secret `~/.local/state/sprout/services/secrets/litellm-master-key`. `up proxy` passes it as `LITELLM_MASTER_KEY`. The file holds no secret. |

Keep `request_timeout` above `--review-timeout`, so that Sprout's own timeout fires first. The
default of `--review-timeout` is 3600 s, the largest timeout that Sprout gives one request. The
selectors wait 1800 s, and the gateway waits 900 s for the actor.

If a review request times out in Sprout, `ReviewTransport` sends it once more with half the output
budget. A 408 from the proxy counts as a transient fault instead. `ReviewTransport` then retries it
three times with the same budget, and then the round fails. The value stays finite, so the proxy
abandons an upstream call whose client has given up. If you raise `--review-timeout`, raise
`request_timeout` with it.

### How `up proxy` acts

`up proxy` acts by these rules:

- If `/health/liveliness` already answers on port 4000, `up proxy` starts nothing. A proxy that
  you started by hand stays as it is.
- `up proxy` reports the proxy as up only after it answers.
- If the proxy exits before it answers, `up proxy` prints the last lines of its log.
- `unset_env` in `services.yaml` names the variables that the CLI removes from the environment of
  the proxy. The template removes `FORCE_COLOR`, which would colour the log. Add each provider
  variable that must not reach the proxy.

[Services](SERVICES.md) gives the other commands, for example these:

- `status`
- `logs proxy -f`
- `restart proxy`, after you edit an alias

## What the smoke verified (2026-10-09, litellm 1.103.0)

The smoke on 2026-10-09 ran the SPROUT method through the proxy on three aliases, and each run
completed. Through `http://127.0.0.1:4000/v1`, each alias ran this evaluation on one DeepSWE
task:

```bash
python -m sprout.eval.evaluate --method sprout --max-turns 3 --rounds 1 --branches 2 \
    --branch-count-mode fixed ...
```

The aliases ran with these reasoning efforts:

| Alias | Effort |
|---|---|
| `luna` | `high` |
| `sonnet5` | `high` |
| `kimi-k3` | `max` |

Each run had these parts:

- a 3-turn root
- the analyst
- the reviewer
- two 3-turn branches

Each run exited 0 with `branch_schedule_complete: true`. These features worked:

- tool calls
- `reasoning_effort`
- reasoning fed back across turns and into the branches
- `usage` with cache-write and cache-read token details, with prompt-cache hits on the actor's
  requests
- the priority tier of Kimi

A probe request on each alias verified `response_format: json_schema`. The reviews of the smoke
ran in the default plain-text mode. No request of the smoke ran long enough to meet
`request_timeout`. So the value of that setting comes from Sprout's timeouts, not from a
measurement.

## `--structured-review`

`--structured-review` asks the analyst and the reviewer for JSON under a schema, instead of a
fenced block of plain text. Use it behind the proxy, because it removes the failures of fence
parsing. The flag is explicit, because only you know whether the endpoint honours strict
`json_schema`. Do not set it for an endpoint that does not.

Without the flag, the reviews need no `response_format` from the endpoint. Every run without the
proxy has used this plain-text mode. A reply that the parser refuses goes back to the model with
the error, up to `--reviewer-max-attempts` tries.

The flag changes these requests:

| Request | Without the flag | With the flag |
|---|---|---|
| The analyst | Plain text. The review job takes the first fenced JSON block, else the text from the first `{` to the last `}`. | `response_format` `json_schema` with `strict: true`, under `ANALYST_SCHEMA`, named `sprout_analysis` |
| The reviewer, `assistant-turn` guidance | One fenced `branch-plan` block | `response_format` `json_schema` with `strict: true`, under `flat_review_schema(...)`, named `sprout_branches` |
| The reviewer, `user-hint` guidance | JSON in plain text | The same as without the flag |
| The reviewer, `none` guidance | One fenced `branch-plan` block | The same as without the flag |

The schemas are in `src/sprout/worker/jobs/review/structured_review.py`. With the flag, these
actions occur:

1. `sprout.eval.evaluate` gives each review job `structured: true`.
2. The review job gives the transport of each call `supports_json_schema`.
3. The review job subtracts the size of the schema and of its JSON directive from
   `--prompt-budget-chars`.

These files record the flag:

- `summary.json` records `structured_review`. It is true only for `--method sprout` with the
  flag.
- Each review plan records `structured_review`, which says whether that review was
  schema-constrained. With a schema, the plan also records `structured_output`, with these keys:
  - `schema`
  - `request_budget`
  - `overhead_chars`
- With a schema, each analyst report records `request_budget` and `structured_output_chars` in
  its `prompt_budget`.
- `examples/qwen/run.py` passes the flag to each `sprout.eval.evaluate` that it runs, and it
  records `structured_review` in the manifest.

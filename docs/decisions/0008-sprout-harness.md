# ADR-0008: A native harness slot checks the raw model output and scores invalid rollouts -1

- **Status:** Accepted
- **Date:** 2026-10-30
- **Revert cost:** the harness defines the actor's tools, prompt and reward contract of every
  experiment that uses it. Rollouts and trained checkpoints made under it cannot be compared with
  mini-swe-agent runs for free.

## Context

Sprout's only model-driven slot is mini-swe-agent with one `bash` tool. It trusts the server's
tool-call parser, which silently repairs malformed turns, and it trains rejected turns like any
other turn. Arctic-SWE (arXiv 2610.24914) shows that this lets format flaws be reinforced until a
long RL run collapses, and that checking the **raw** sampled output, ending invalid rollouts and
scoring them below honest failures keeps training stable.

## Decision

We add a second slot, `sprout-harness` (agent `harness`), written for RL: five native
tools with friendly arguments and results (user), a strict check of the raw model output that
reproduces the paper's Table 6, and the paper's stop and reward contract (Table 18) with reward
-1 for invalid trajectories (user). The first use is an overfit experiment: Qwen3.5-4B on about
150 SWE-rebench V2 tasks, trained through Miles, until the training reward rises (user).

## Goals

1. Qwen3.5-4B rolls out end to end through the Run Store with the harness, on podman sandboxes.
2. The raw-output check rejects all 13 malformed turns of the paper's Table 6, plus the harness
   preferences of Appendix I.3 (non-empty reasoning, one call per turn, no prose between the
   reasoning block and the call).
3. Evaluation mode gives feedback and continues; training mode ends the rollout (Table 5).
4. The RL driver gives an invalid trajectory reward -1 without grading it, and in training mode an
   unfinished one, which a budget ended before `submit`, 0 without grading it (Tables 5 and 18).
   Miles trains on non-search GRPO groups with those rewards.
5. An overfit run on ~150 tasks whose training reward rises over many steps.

## Non-goals (user: no review and no RAFO for now)

- Review jobs, reviewer-inserted turns, RAFO and search groups for harness parents.
- BPO and the other branch methods of `sprout.eval` for harness parents.
- Raw checks for API models (no raw output is available there; the check degrades to structure).
- CISPO's importance weights with rollout log-probs: Miles re-tokenizes Sprout's messages and
  recomputes log-probs, so the trainer has no behavior probabilities on this path.

## Anti-goals

- Executing a turn that a parser repaired. The harness executes only its own strict parse.
- Changing mini-swe-agent's behavior. Every mini test keeps passing.

## Tools (model-facing)

All tools are native Chat Completions functions with flat schemas. Parameters are strings or
integers only: the strict check renders non-string values with the chat template, and arrays would
make a semantically equal call fail on whitespace. Relative paths resolve against the task's
working directory, and every result names the absolute path.

| Tool | Arguments | Result |
|---|---|---|
| `execute_bash` | `command`, optional `timeout` (s, default 60, at most 600) | exit code, duration, output (head and tail kept past 10,000 chars, with the omitted count); a timeout says so and shows the partial output |
| `read_file` (`full` only) | `file_path`, optional `offset` (1-based line, default 1), `limit` (lines, default 400) | numbered lines, the range shown, whether the file continues and the `offset` to continue from; clear errors for a missing file or a directory |
| `write_file` (`full` only) | `file_path`, `content` | created or overwritten, line and byte counts (and the previous size when overwritten) |
| `str_replace` | `file_path`, `old_string`, `new_string` | on success the replaced line range and a numbered snippet around the edit; on failure why (not found, or found N times at which lines) and what to do |
| `submit` | none | ends the rollout; the workspace is graded as it is |

The default tool set is the paper's three: `execute_bash`, `str_replace` and `submit`. Its system
message says how to read files with commands (`grep -n`, `sed -n 'START,ENDp'`, and that output past
10,000 characters keeps only its ends), and the task message and failure hints name commands in
place of the file tools. `extra.harness.toolset: full` offers all five, as the first overfit run
did. That run showed why the default dropped two: `write_file` made 0.1% of the calls late in the
run, and `read_file` returned 5k characters a call, about a quarter of a trajectory's tokens, while
the model already read with `grep -n` and `sed -n` (see the experiment's log).

Each tool call is one ToolAccess call, so checkpoints and journal steps stay one per call:

- `execute_bash` runs as the runtime's `shell` (as the image's user when `image_user` is on).
- `read_file` runs as `text_editor view`, which the mutation tracker counts as read-only: a turn
  that only reads reuses the previous snapshot.
- `write_file` and `str_replace` run as `text_editor write` and `str_replace`. The runtime gains:
  a snippet and line numbers in `str_replace` results, created/overwritten and sizes in `write`
  results, and new files and directories that take the owner of their nearest existing parent (the
  runtime runs as root; without this, files that `write_file` creates would be root's when the
  agent's own commands run as the image's user).
- `submit` makes no sandbox call. Its turn closes with a `tool.skipped` record, so it is not a
  branch point; the final snapshot pairs with the last closed turn.

## Raw output and the format check

**Raw text.** The harness requests `logprobs` (no top log-probs). SGLang returns every output
token, with its exact bytes recovered from the token id, whatever the server's parsers did. The
concatenated bytes, minus a final `<|im_end|>`, are the raw completion that follows the generation
prompt `<|im_start|>assistant\n<think>\n`. Without log-probs (API models) the harness rebuilds a
canonical text from the structured fields and marks the turn `raw: unavailable`.

**Grammar (Qwen3.5/3.6 `qwen3_coder` XML).** The canonical completion of a turn with one call:

```
{reasoning}\n</think>\n\n[{content}\n\n]<tool_call>\n<function={name}>\n
(<parameter={key}>\n{value}\n</parameter>\n)*</function>\n</tool_call><|im_end|>
```

**Checks**, in order; the first failure names the turn's `format_error`:

| Check | Table 6 flaws it rejects |
|---|---|
| exactly one `\n</think>\n\n` closes the reasoning; no `<think>` or `</think>` elsewhere | no `</think>`, second `</think>`, extra `<think>` |
| no `<tool_call>`, `<function=` or `<parameter=` inside the reasoning | call inside reasoning |
| exactly one `<tool_call>` block, closed, nothing after `</tool_call>`, nothing nested | text after `</tool_call>`, unclosed, nested, two calls |
| inside the block: `<function=NAME>` with a closing `>`, each `<parameter=K>` closed by `</parameter>`, then `</function>`; one function | no `</parameter>`, no `</function>`, no `>` after `<function=`, two functions, no `<function=` |
| no parameter repeats | duplicate parameter |
| the name is a tool; arguments meet its schema (required present, no extras, integers in canonical decimal) | (harness) |
| reasoning is not blank; no prose between `</think>` and `<tool_call>` | (harness, Appendix I.3) |
| `finish_reason` is `stop` | truncation (a stop condition of its own) |

The check is strict on purpose: a tag string such as `</parameter>` or `<tool_call>` anywhere in
the reasoning or in an argument value is a flaw too, so the agent cannot write those strings
literally.

A round-trip diagnostic renders the parsed turn with the template and compares it with the raw
bytes. By default a mismatch is journaled, not rejected (`strict_roundtrip: false`): the paper's
check is structural, and whitespace-only mismatches are measured first.

## Turn loop, modes and stop conditions

`extra.harness.on_invalid` is `feedback` (evaluation, default) or `terminate` (training).

| Stop condition | Trigger | Training reward | Evaluation |
|---|---|---|---|
| `format_invalid` | a check fails | -1, not graded | feedback message, the turn counts, continue |
| `response_truncated` | `finish_reason: length` | -1, not graded | feedback, continue |
| `repetition` | the same call with the same result `repetition_limit` more times since the last successful edit (off by default, as in the paper's main run) | -1, not graded | a note, continue |
| `submitted` | `submit` | graded 0/1 | same |
| `max_turns_reached`, `timeout`, `max_total_tokens` | budgets (the last: prompt plus completion tokens of the latest request past `max_total_tokens`, default 131072) | 0, not graded: only a submission scores (set `require_submit: false` to grade it) | graded |
| `context_exhausted` | the server's context window | graded | graded |

A rejected turn in evaluation mode keeps the assistant message (with its raw text) and a user-role
feedback that quotes the error; it is a message step but not a branch point, as in mini.

## Integration with Sprout

- **Journal:** the mini contract (`run.started`, `session.ref`, `rollout.model_tools`,
  `turn.started`, `agent.message`/`agent.thinking`, ToolAccess calls, `model.turn.completed`,
  `close_turn`, `turn.completed`, `tool.skipped`, `run.result`, `run.finished`), plus
  `harness.turn.rejected` and a compact `raw.sprout-harness` per response (raw text, finish
  reason, usage, format verdict; never the log-probs).
- **Native history** `sprout-harness-v1` at `<native_home>/<session_id>.jsonl`:
  `harness.session`, `harness.message` (Chat Completions messages with `reasoning_content`),
  `harness.turn` (`turn_id`, `call_ids`, `source`, `rejected`/`submitted`). The mini indexer and
  the export become parametrized by a transcript spec (record types, header check, slot), so
  recovery points, salvage and `message_export` work for both slots.
- **Allow-lists:** `RunSpec` slot/agent, `runstore.specs`, `runner._agent_slot`, the worker CLI and
  the RL driver's `run_defaults` check accept `sprout-harness`/`harness`.
- **Rewards:** a contract constant `INVALID_STOP_REASONS`. A rollout that stops for one is a
  successful actor with `status: truncated` and that `stop_reason`; the coordinator skips its
  grade, and the driver exports reward -1 with `zero_reason: null`. Miles's `StopReason` gains the
  three reasons; non-search GRPO uses the reward as given. The invalid turn is exported as the last
  assistant message (reasoning and content from the raw text, no tool calls), so the -1 reaches
  the tokens that caused it.

## Overfit experiment

- **Data:** `PrimeIntellect/SWE-rebench-V2-Filtered-Easy-Verified` (prepared on this host, 150
  candidates with the smallest image per repo; the 96 already pulled, since the disk holds no more
  images). Like the paper's 160-task pool, keep the tasks that Qwen3.5-4B solves at least once in
  four rollouts, run in evaluation mode: 92% of the base model's trajectories break the format at
  least once (77% of first flaws are prose before the call), so a training-mode filter would keep
  only the shortest tasks.
- **Rollouts:** temperature 1.0, top-p 1, top-k -1, 60 s command timeout, 10k-char output bound;
  turn and token budgets scaled to two GPUs: 60 turns, 16k tokens a response, 56k a trajectory
  (so a trajectory fits the trainer's 80k tokens).
- **Training:** Miles colocated on the 2 RTX PRO 6000 GPUs, GRPO with group-centered advantages
  (`--disable-grpo-std-normalization`), no KL, no entropy term, AdamW lr 1e-6 with beta2 0.98,
  epsilon 1e-15, no weight decay and clipping at 1.0; 16 tasks of 5 rollouts an update (the
  paper's overfit run takes 32 of 5), token mean over the batch, one optimizer step per rollout
  batch (so the PPO ratio is 1 and the step is a plain policy gradient), invalid rollouts at -1
  kept in the batch, groups whose rewards are all equal left out of it.
- **Metrics:** mean reward and resolve rate per step, invalid rate by stop condition, turns and
  tokens per trajectory, entropy; success is a rising training reward on the pool.

## Left to the builder

Prompt wording (from mini-swe-agent-plus's SWE-bench template, rewritten for these tools, with the
paper's workflow hint to run a relevant test before submitting), feedback texts, default limits,
file layout within the gate's limits (10 files a package, 1000 lines a file).

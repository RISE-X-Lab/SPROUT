"""The journal event model, version 2: the envelope, the event types and ``Usage``.

One append-only stream holds the whole trajectory of a run: what the agent said, its tool
calls, its turns and session, and the checkpoints that pair them.

:class:`~sprout.worker.run.journal.JournalWriter` adds this envelope to each event::

    {"v": 2, "type": "<event type>", "ts": "...Z", "seq": N,
     "run_id": ..., "agent_id": ..., "sandbox_id": ..., **payload}

Conventions:

- Event types are dotted and lowercase: ``<domain>.<what>``.
- Payload keys are snake_case and JSON-safe.
- A slot-native payload with no mapping yet goes verbatim into a ``raw.<slot>`` event, never
  dropped. For example, mini's model responses are ``raw.mini-swe-agent`` events.
- Framework-private additions go under ``extensions.*``.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

JOURNAL_SCHEMA_VERSION = 2

# --- run lifecycle ---------------------------------------------------------
RUN_STARTED = "run.started"  # {slot, slot_version, model, task_prompt, cwd, config}
RUN_FINISHED = "run.finished"  # {status, usage, error}
RUN_RESULT = "run.result"  # {text, native}  final answer as reported by the agent

# --- agent output ----------------------------------------------------------
AGENT_MESSAGE = "agent.message"  # {text, partial?, part_id?}
AGENT_THINKING = "agent.thinking"  # {text, part_id?}

# --- tools -----------------------------------------------------------------
TOOL_STARTED = "tool.started"  # {call_id, name, args, parent_call_id?}
TOOL_FINISHED = "tool.finished"  # {call_id, name?, status: ok|error, output, exit_code?, duration_ms?}
#: A call of a model response that the agent did not run. It has no step, and it settles the call
#: for the turn ``turn_id`` only. ``reason`` is ``after_submit``: an earlier call submitted the task.
TOOL_SKIPPED = "tool.skipped"  # {call_id, turn_id, reason}

# --- turn / usage ----------------------------------------------------------

# A turn is one model call and the tool calls of its response. No tool call is in flight
# between ``turn.completed`` and the next ``turn.started``.
TURN_STARTED = "turn.started"  # {turn?}
#: A turn boundary, and only that. A consumer ends a step at each one, so a report that is
#: not a boundary goes in ``usage.updated``.
TURN_COMPLETED = "turn.completed"  # {turn?, usage?: Usage.as_dict()}
#: Interim accounting. A report with ``cumulative`` true is the thread's total so far and
#: replaces the earlier reports, so a consumer must not add it to them.
USAGE_UPDATED = "usage.updated"  # {usage: Usage.as_dict(), cumulative?: bool}

# --- session & rollback ----------------------------------------------------
SESSION_REF = "session.ref"  # {native_session_id, transcript_path?, capabilities?}
CHECKPOINT_CAPTURED = "checkpoint.captured"  # {step, snapshot_id, reason, disk_only, session_ckpt?}
#: A full (memory) snapshot taken after the final disk one, when the run asked for it.
#: It is not a branch point, and a benchmark can release it after grading.
SNAPSHOT_MEMORY = "snapshot.memory"  # {step, turn_id, snapshot_id, captured, after}

# --- sandbox ---------------------------------------------------------------

#: The sandbox this run owns and the backend that made it. ``sprout.worker sweep`` reads
#: ``backend`` and ``store`` (podman only) to delete the run's snapshots where they live.
SANDBOX_CREATED = "sandbox.created"  # {sandbox_id, backend, image, store?}


@dataclass
class Usage:
    """Normalized token and cost accounting of model calls.

    Each dimension has its own field, because one total loses the cache and reasoning costs.
    ``cached_input_tokens`` counts the input tokens read from a cache (ATIF ``cached_tokens``).
    ``cache_creation_tokens`` is Anthropic's cache-write count, and 0 for other providers."""

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: Usage) -> None:
        """Add each count of ``other`` to this one."""
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict[str, int | float]:
        """The fields as a JSON-safe dict."""
        return asdict(self)

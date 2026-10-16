"""Unified journal event model (v2).

One append-only stream holds a run's whole trajectory: what the agent said, its
tool calls, turns and session, and the checkpoints that pair them. Version 1
had the tool events alone.

Envelope (added by JournalWriter, see harness_core/journal.py)::

    {"v": 2, "type": "<event type>", "ts": "...Z", "seq": N,
     "run_id": ..., "agent_id": ..., "sandbox_id": ..., **payload}

Conventions:
- Event types are dotted, lowercase: ``<domain>.<what>``.
- Payload keys are snake_case, JSON-safe.
- Slot-native payloads that have no mapping yet are preserved verbatim under
  ``raw.<slot>`` events -- never silently dropped (mini's model responses are
  ``raw.mini-swe-agent``).
- Framework-private additions go under ``extensions.*`` (v1 convention kept).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

JOURNAL_SCHEMA_VERSION = 2

# --- run lifecycle ---------------------------------------------------------
RUN_STARTED = "run.started"          # {slot, slot_version, model, task_prompt, cwd, config}
RUN_FINISHED = "run.finished"        # {status, usage, error}
RUN_RESULT = "run.result"            # {text, native}  final answer as reported by the agent

# --- agent output ----------------------------------------------------------
AGENT_MESSAGE = "agent.message"      # {text, partial?, part_id?}
AGENT_THINKING = "agent.thinking"    # {text, part_id?}

# --- tools (aligned with v1 tool.started/tool.finished) --------------------
TOOL_STARTED = "tool.started"        # {call_id, name, args, parent_call_id?}
TOOL_FINISHED = "tool.finished"      # {call_id, name?, status: ok|error, output, exit_code?, duration_ms?}

# --- turn / usage ----------------------------------------------------------
# One disk snapshot per model turn, taken once the turn has closed
# (harness/tools/access.py); a closed turn with its snapshot is a branch point
# (harness_checkpoint/rollback.py): between turn.completed and the next
# turn.started there is no in-flight call.
TURN_STARTED = "turn.started"        # {turn?}
#: A turn boundary, and *only* that. Anything that is not a boundary must not use
#: it -- a mid-turn usage report emitted here once produced a checkpoint per
#: token update.
TURN_COMPLETED = "turn.completed"    # {turn?, usage?: Usage.as_dict()}
#: Interim accounting. ``cumulative`` marks a total-for-the-thread figure that
#: replaces earlier ones; without it a consumer summing every report multiplies
#: a run's real token count by the number of updates.
USAGE_UPDATED = "usage.updated"      # {usage: Usage.as_dict(), cumulative?: bool}

# --- session & rollback ----------------------------------------------------
SESSION_REF = "session.ref"          # {native_session_id, transcript_path?, capabilities?}
CHECKPOINT_CAPTURED = "checkpoint.captured"  # {step, snapshot_id, reason, disk_only, session_ckpt?}
#: A full (memory) snapshot taken after the final disk one, when the run asked
#: for it. Not a branch point; a benchmark may release it after grading.
SNAPSHOT_MEMORY = "snapshot.memory"          # {step, turn_id, snapshot_id, captured, after}

# --- sandbox ---------------------------------------------------------------
#: The sandbox this run owns and the backend that made it. ``harness sweep``
#: reads ``backend``: a podman run's snapshot ids are btrfs subvolume names in
#: ``store`` (recorded for podman only), and sending them to AgentENV would
#: count every 404 as a deletion.
SANDBOX_CREATED = "sandbox.created"          # {sandbox_id, backend, image, store?}


@dataclass
class Usage:
    """Normalized token/cost accounting.

    Dimensions are kept separate on purpose (providers disagree on the
    breakdown; collapsing to a single total loses cache/reasoning economics).
    ``cached_input_tokens`` counts tokens read from cache (a subset of input,
    ATIF ``cached_tokens`` semantics). ``cache_creation_tokens`` is the
    Anthropic cache-write count; 0 elsewhere.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    cache_creation_tokens: int = 0
    reasoning_output_tokens: int = 0
    cost_usd: float = 0.0

    def add(self, other: "Usage") -> None:
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cached_input_tokens += other.cached_input_tokens
        self.cache_creation_tokens += other.cache_creation_tokens
        self.reasoning_output_tokens += other.reasoning_output_tokens
        self.cost_usd += other.cost_usd

    def as_dict(self) -> dict:
        return asdict(self)

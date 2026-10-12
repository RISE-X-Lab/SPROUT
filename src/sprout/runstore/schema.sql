CREATE TABLE IF NOT EXISTS rs_jobs (
    id text PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    request_hash text NOT NULL,
    request jsonb NOT NULL,
    kind text NOT NULL CHECK (kind IN ('rollout', 'grade', 'review')),
    state text NOT NULL DEFAULT 'queued'
        CHECK (state IN ('queued', 'running', 'quarantined', 'succeeded', 'failed', 'cancelled')),
    phase text NOT NULL DEFAULT 'queued',
    attempt_count integer NOT NULL DEFAULT 0,
    max_attempts integer NOT NULL CHECK (max_attempts BETWEEN 1 AND 3),
    active_attempt text,
    lease_token text,
    lease_until timestamptz,
    worker_id text,
    ready_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    result jsonb,
    error text,
    created_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    updated_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS rs_jobs_ready ON rs_jobs (ready_at, created_at) WHERE state = 'queued';
CREATE TABLE IF NOT EXISTS rs_attempts (
    id text PRIMARY KEY,
    job_id text NOT NULL REFERENCES rs_jobs(id),
    number integer NOT NULL,
    worker_id text NOT NULL,
    lease_token text NOT NULL UNIQUE,
    state text NOT NULL DEFAULT 'running',
    execution jsonb NOT NULL DEFAULT '{}',
    result jsonb,
    error text,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    finished_at timestamptz,
    UNIQUE(job_id, number)
);
ALTER TABLE rs_attempts ADD COLUMN IF NOT EXISTS payload jsonb;
ALTER TABLE rs_attempts ADD COLUMN IF NOT EXISTS payload_hash text;
-- Set by a stop request; the worker running the job reads it at each heartbeat.
ALTER TABLE rs_jobs ADD COLUMN IF NOT EXISTS stop_requested_at timestamptz;
-- A database made before review jobs existed allows two kinds; re-stating the
-- constraint is idempotent under the advisory lock initialize() holds.
ALTER TABLE rs_jobs DROP CONSTRAINT IF EXISTS rs_jobs_kind_check;
ALTER TABLE rs_jobs ADD CONSTRAINT rs_jobs_kind_check CHECK (kind IN ('rollout', 'grade', 'review'));
CREATE TABLE IF NOT EXISTS rs_events (
    attempt_id text NOT NULL REFERENCES rs_attempts(id),
    seq bigint NOT NULL,
    event jsonb NOT NULL,
    PRIMARY KEY (attempt_id, seq)
);
CREATE TABLE IF NOT EXISTS rs_prefix_nodes (
    node text PRIMARY KEY,
    scope text NOT NULL,
    parent text NOT NULL,
    depth integer NOT NULL,
    call jsonb NOT NULL
);
CREATE INDEX IF NOT EXISTS rs_prefix_parent ON rs_prefix_nodes(scope, parent);
CREATE TABLE IF NOT EXISTS rs_tools (
    attempt_id text NOT NULL REFERENCES rs_attempts(id),
    depth integer NOT NULL,
    call_id text NOT NULL,
    call jsonb NOT NULL,
    response jsonb,
    prefix_node text REFERENCES rs_prefix_nodes(node),
    PRIMARY KEY(attempt_id, depth),
    UNIQUE(attempt_id, call_id)
);
CREATE INDEX IF NOT EXISTS rs_tools_prefix ON rs_tools(prefix_node);
CREATE TABLE IF NOT EXISTS rs_recoveries (
    id text PRIMARY KEY,
    attempt_id text NOT NULL REFERENCES rs_attempts(id),
    message_step integer NOT NULL,
    tool_depth integer NOT NULL,
    prefix_node text NOT NULL REFERENCES rs_prefix_nodes(node),
    snapshot_id text NOT NULL,
    native jsonb NOT NULL,
    valid boolean NOT NULL DEFAULT true,
    invalid_reason text,
    UNIQUE(attempt_id, message_step)
);
CREATE INDEX IF NOT EXISTS rs_recoveries_depth ON rs_recoveries(attempt_id, tool_depth);

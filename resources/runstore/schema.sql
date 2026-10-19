CREATE TABLE IF NOT EXISTS rs_jobs (
    id text PRIMARY KEY,
    idempotency_key text NOT NULL UNIQUE,
    request_hash text NOT NULL,
    request jsonb NOT NULL,
    kind text NOT NULL CHECK (kind IN ('rollout', 'grade', 'review')),
    state text NOT NULL DEFAULT 'queued'
        CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled')),
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
-- When the worker of a lost attempt deleted the attempt's local data, or found no attempt directory.
-- NULL while the data can still be on the host. The worker looks up these attempts once a minute.
ALTER TABLE rs_attempts ADD COLUMN IF NOT EXISTS data_deleted_at timestamptz;
CREATE INDEX IF NOT EXISTS rs_attempts_undeleted ON rs_attempts (worker_id)
    WHERE state = 'lost' AND data_deleted_at IS NULL;
-- Set by a stop request; the worker running the job reads it at each heartbeat.
ALTER TABLE rs_jobs ADD COLUMN IF NOT EXISTS stop_requested_at timestamptz;
-- A database made before review jobs existed allows two kinds; re-stating the
-- constraint is idempotent under the advisory lock initialize() holds.
ALTER TABLE rs_jobs DROP CONSTRAINT IF EXISTS rs_jobs_kind_check;
ALTER TABLE rs_jobs ADD CONSTRAINT rs_jobs_kind_check CHECK (kind IN ('rollout', 'grade', 'review'));
-- A grade of a kept sandbox: only bound_worker claims it, and only before keep_until.
ALTER TABLE rs_jobs ADD COLUMN IF NOT EXISTS bound_worker text;
ALTER TABLE rs_jobs ADD COLUMN IF NOT EXISTS keep_until timestamptz;
CREATE INDEX IF NOT EXISTS rs_jobs_kept ON rs_jobs (keep_until) WHERE state = 'queued' AND bound_worker IS NOT NULL;
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
-- A database made before ADR-0006 allows a state that no job takes now, quarantined. Re-stating
-- the constraint is idempotent under the advisory lock initialize() holds.
ALTER TABLE rs_jobs DROP CONSTRAINT IF EXISTS rs_jobs_state_check;
ALTER TABLE rs_jobs ADD CONSTRAINT rs_jobs_state_check
    CHECK (state IN ('queued', 'running', 'succeeded', 'failed', 'cancelled'));
-- One row for each worker, <host>/<name>. The session that registered last holds it. The worker
-- is live while the row is not stopped and seen_at, its last heartbeat, is less than 600 s old.
CREATE TABLE IF NOT EXISTS rs_workers (
    worker_id text PRIMARY KEY,
    host text NOT NULL,
    session text NOT NULL,
    started_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    seen_at timestamptz NOT NULL DEFAULT clock_timestamp(),
    -- {"grades": [{"dataset_sha256": ..., "grader_revision": ...}, ...]}: the checksums of the files of
    -- the worker's grade profiles. A claim gives a swe-rebench-v2 grade only to a worker with its pair.
    capabilities jsonb NOT NULL DEFAULT '{}'
);
-- {"pid": ..., "start_ticks": ..., "boot_id": ...}: the process of the session, NULL if unknown. A new
-- session on the row's host takes the row at once when that process is gone from the host.
ALTER TABLE rs_workers ADD COLUMN IF NOT EXISTS process jsonb;
-- A worker that stops sets stopped and seen_at, and keeps its row (ADR-0006). The next session takes the
-- row at once. Its host is lost when no row of the host, stopped or not, has a seen_at in the last 600 s.
ALTER TABLE rs_workers ADD COLUMN IF NOT EXISTS stopped boolean NOT NULL DEFAULT false;
-- The affinity key of each job (ADR-0007): the jobs of one key run on one host. A job stored before
-- the column takes its own id as its key, as a new job that names no other job does.
ALTER TABLE rs_jobs ADD COLUMN IF NOT EXISTS affinity text;
UPDATE rs_jobs SET affinity = id WHERE affinity IS NULL;
ALTER TABLE rs_jobs ALTER COLUMN affinity SET NOT NULL;
CREATE INDEX IF NOT EXISTS rs_jobs_affinity ON rs_jobs (affinity) WHERE state IN ('queued', 'running');
-- The host of each affinity key: the host of the worker that claimed a job of the key first. A key
-- stays bound, also after its host is lost.
CREATE TABLE IF NOT EXISTS rs_affinity (
    key text PRIMARY KEY,
    host text NOT NULL,
    bound_at timestamptz NOT NULL DEFAULT clock_timestamp()
);
CREATE INDEX IF NOT EXISTS rs_affinity_host ON rs_affinity (host);
-- A request stored before ADR-0006 dropped the profile digest names it, and JobSpec has no such field.
-- Its request_hash still covers the key, so a resubmission under its idempotency key is a Conflict.
UPDATE rs_jobs SET request = request - 'profile_hash' WHERE request ? 'profile_hash';

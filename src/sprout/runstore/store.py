"""The Run Store's state in PostgreSQL: workers, jobs, attempts, leases, payloads and journals.

A worker claims a queued job and then owns it through a lease token. Each write for an attempt
checks the token first and raises ``Fenced`` when the lease is no longer live. Each worker
registers, and its heartbeats renew the leases of its running jobs. Once a worker is dead and a
lease expired, ``settle_dead_workers`` settles the job. Each job has an affinity key, and only
the host that its key is bound to claims it. Once that host is lost, the job fails as ``data_lost``.
The schema is ``resources/runstore/schema.sql``. docs/RUNSTORE.md describes the service.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from contextlib import contextmanager
from datetime import datetime
from typing import TYPE_CHECKING, Any
from uuid import uuid4

from sprout.resource import resource
from sprout.runstore.payload import checked_payload, encode_payload
from sprout.runstore.specs import JobSpec, canonical, digest

if TYPE_CHECKING:
    from psycopg2.extras import RealDictCursor

#: The result of a bound grade whose ``keep_until`` passed before a claim or a sweep: its sandbox is gone.
_SANDBOX_LOST = {"reward": 0.0, "reason": "sandbox_lost"}
#: How long a worker stays live after its last heartbeat, in seconds. Each heartbeat moves the end of
#: the leases of the worker's running jobs to as long after it. ADR-0006 sets the value.
WORKER_LIVE_S = 600
#: An SQL condition: the worker row ``w`` had a heartbeat in WORKER_LIVE_S. A stop counts as a heartbeat.
_SEEN = f"w.seen_at > clock_timestamp() - {WORKER_LIVE_S} * interval '1 second'"
#: An SQL condition: the worker row ``w`` is live. It is not stopped, and it had a heartbeat in WORKER_LIVE_S.
_LIVE = f"NOT w.stopped AND {_SEEN}"
#: An SQL condition: the worker in ``{column}`` is dead. It has no live row.
_DEAD = f"NOT EXISTS (SELECT 1 FROM rs_workers w WHERE w.worker_id={{column}} AND {_LIVE})"
#: An SQL condition: a worker whose grade pairs are the JSON array in the parameter can run the job ``j``.
#: A ``harbor`` grade reads its task directory, which no checksum pins, so it needs no pair.
_GRADABLE = """(j.kind<>'grade' OR j.request->'spec'->>'benchmark'='harbor'
    OR %s::jsonb @> jsonb_build_array(jsonb_build_object('dataset_sha256',j.request->'spec'->'dataset_sha256',
        'grader_revision',j.request->'spec'->'grader_revision')))"""
#: An SQL condition: the host in ``{column}`` is lost. No worker row of it, stopped or not, had a
#: heartbeat in WORKER_LIVE_S (ADR-0007).
_HOST_LOST = f"NOT EXISTS (SELECT 1 FROM rs_workers w WHERE w.host={{column}} AND {_SEEN})"


class Conflict(ValueError):
    """A request contradicts what the Run Store already holds."""


class Fenced(Conflict):
    """The lease token of an attempt no longer owns a live lease on its job."""


def _keep_until(rollout: dict[str, Any]) -> datetime:
    """The deadline of the sandbox that a finished rollout job kept, from its ``kept_sandbox`` result.

    Raises Conflict if the rollout has not finished, and ValueError if it kept no sandbox.
    """
    if rollout["state"] in ("queued", "running"):
        raise Conflict("The in_place_job rollout has not finished")
    kept = (rollout["result"] or {}).get("kept_sandbox")
    if not isinstance(kept, dict):
        raise ValueError("The in_place_job rollout kept no sandbox")
    deadline: datetime | None
    try:
        deadline = datetime.fromisoformat(kept["keep_until"])
    except (KeyError, TypeError, ValueError):
        deadline = None
    if deadline is None or deadline.tzinfo is None:
        raise ValueError("The kept sandbox of the in_place_job rollout has no UTC keep_until")
    return deadline


def _data_lost(host: str) -> dict[str, Any]:
    """The result of a job whose affinity key is bound to the lost ``host``, which held the job's data."""
    return {
        "status": "error",
        "failure_kind": "infrastructure",
        "reason": "data_lost",
        "host": host,
        "error": f"Host {host} is lost: none of its workers reported a heartbeat in {WORKER_LIVE_S} s",
    }


def _named_jobs(cursor: RealDictCursor, request: JobSpec) -> list[str]:
    """The ids of the jobs that ``request`` names: a branch's parent, its ``in_place_job`` or its parents.

    Raises KeyError for an unknown ``parent_point``.
    """
    if request.parent_point is not None:
        cursor.execute(
            "SELECT a.job_id FROM rs_recoveries r JOIN rs_attempts a ON a.id=r.attempt_id WHERE r.id=%s",
            (request.parent_point,),
        )
        point = cursor.fetchone()
        if point is None:
            raise KeyError(request.parent_point)
        return [point["job_id"]]
    if request.kind == "grade" and request.spec.get("in_place_job") is not None:
        return [request.spec["in_place_job"]]
    if request.kind == "review":
        return sorted({parent["job_id"] for parent in request.spec["parents"]})
    return []


def _affinity(cursor: RealDictCursor, request: JobSpec) -> str | None:
    """The affinity key of ``request``: its own, else the one key of the jobs that it names. None: neither.

    Raises KeyError for an unknown named job or point. Raises ValueError for named jobs of two keys, for an
    own key that is not theirs, and for a grade from a ``snapshot_id`` without a key."""
    named = _named_jobs(cursor, request)
    if not named:
        if request.affinity is None and request.kind == "grade" and request.spec.get("snapshot_id"):
            raise ValueError("A grade from a snapshot_id needs the affinity key of its rollout")
        return request.affinity
    cursor.execute("SELECT id,affinity FROM rs_jobs WHERE id=ANY(%s)", (named,))
    keys = {row["id"]: row["affinity"] for row in cursor.fetchall()}
    unknown = [job_id for job_id in named if job_id not in keys]
    if unknown:
        raise KeyError(unknown[0])
    if request.affinity is not None:
        # Another key could bind to another host, where the data of the named jobs does not exist.
        if any(key not in (None, request.affinity) for key in keys.values()):
            raise ValueError("The affinity of a request must be the key of the jobs that it names")
        return request.affinity
    if len(set(keys.values())) > 1:
        raise ValueError(
            "The jobs that the request names have different affinity keys, so no host holds all their data"
        )
    return next(iter(keys.values()), None)


def _one_row(cursor: RealDictCursor) -> dict[str, Any]:
    """The row of a statement that always returns one, by the schema's constraints and the row locks."""
    row = cursor.fetchone()
    if row is None:
        raise LookupError("The statement returned no row")
    return dict(row)


class Store:
    """The Run Store in the PostgreSQL database that ``dsn`` names.

    Each method runs in a transaction of its own. A row is a dict of its columns. A method that takes
    a lease ``token`` raises ``Fenced`` unless the token owns the job's live lease.
    """

    def __init__(self, dsn: str) -> None:
        self.dsn = dsn

    @contextmanager
    def transaction(self) -> Iterator[RealDictCursor]:
        """A cursor in a new transaction, committed when the block exits and rolled back when it raises.

        Each statement in it times out after 5 seconds.
        """
        import psycopg2
        from psycopg2.extras import RealDictCursor

        # Fail a connection to an unreachable database within about 10 s. tcp_user_timeout is in ms.
        connection = psycopg2.connect(
            self.dsn,
            connect_timeout=5,
            keepalives=1,
            keepalives_idle=5,
            keepalives_interval=2,
            keepalives_count=2,
            tcp_user_timeout=10000,
        )
        try:
            with connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    cursor.execute("SET LOCAL statement_timeout = '5s'")
                    yield cursor
        finally:
            connection.close()

    def initialize(self) -> None:
        """Create the schema, or bring an existing one up to date. Concurrent and repeated calls are safe."""
        with self.transaction() as cursor:
            # Serializes concurrent initializations. Any key works if every initialize() uses the same one.
            cursor.execute("SELECT pg_advisory_xact_lock(82951734)")
            cursor.execute(resource("runstore/schema.sql").read_text())

    def submit(self, request: JobSpec, idempotency_key: str) -> dict[str, Any]:
        """Queue ``request`` under ``idempotency_key``, and return its job. A repeat returns the first job.

        The ``affinity`` is the request's, else the one key of the jobs that it names, else the job's id. A
        grade from a ``snapshot_id`` needs its own. Raises Conflict for an idempotency key of another request
        or an unfinished ``in_place_job``, KeyError for an unknown job or point, and ValueError if invalid."""
        body = request.validate()
        if not idempotency_key or len(idempotency_key) > 256:
            raise ValueError("An idempotency key of 1..256 characters is required")
        job_id = uuid4().hex
        with self.transaction() as cursor:
            bound_worker, keep_until = self._binding(cursor, request)
            affinity = _affinity(cursor, request) or job_id
            cursor.execute(
                """INSERT INTO rs_jobs
                (id,idempotency_key,request_hash,request,kind,max_attempts,bound_worker,keep_until,affinity)
                VALUES (%s,%s,%s,%s::jsonb,%s,%s,%s,%s,%s)
                ON CONFLICT(idempotency_key) DO NOTHING""",
                (
                    job_id,
                    idempotency_key,
                    digest(body),
                    canonical(body),
                    request.kind,
                    request.max_infra_retries + 1,
                    bound_worker,
                    keep_until,
                    affinity,
                ),
            )
            cursor.execute("SELECT * FROM rs_jobs WHERE idempotency_key=%s", (idempotency_key,))
            job = _one_row(cursor)
            if job["request_hash"] != digest(body):
                raise Conflict("Idempotency key already names a different request")
            return job

    @staticmethod
    def _binding(cursor: RealDictCursor, request: JobSpec) -> tuple[str | None, datetime | None]:
        """The worker and the deadline that bind a grade with ``in_place_job``, else ``(None, None)``."""
        rollout_id = request.spec.get("in_place_job") if request.kind == "grade" else None
        if rollout_id is None:
            return None, None
        # The worker of the last claim ran the attempt that kept the sandbox.
        cursor.execute("SELECT state,result,worker_id FROM rs_jobs WHERE id=%s", (rollout_id,))
        rollout = cursor.fetchone()
        if rollout is None:
            raise KeyError(rollout_id)
        return rollout["worker_id"], _keep_until(rollout)

    def get(self, job_id: str) -> dict[str, Any]:
        """The job ``job_id``. Raises KeyError if there is none."""
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rs_jobs WHERE id=%s", (job_id,))
            row = cursor.fetchone()
            if row is None:
                raise KeyError(job_id)
            return dict(row)

    def list_jobs(self, state: str | None = None, limit: int = 100) -> list[dict[str, Any]]:
        """The newest jobs first, only those in ``state`` when it is given, at most ``limit`` (1 to 1000)."""
        with self.transaction() as cursor:
            cursor.execute(
                """SELECT * FROM rs_jobs WHERE (%s IS NULL OR state=%s)
                ORDER BY created_at DESC,id LIMIT %s""",
                (state, state, min(1000, max(1, limit))),
            )
            return [dict(row) for row in cursor.fetchall()]

    def attempts(self, job_id: str) -> list[dict[str, Any]]:
        """The attempts of ``job_id``, in the order of their numbers."""
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rs_attempts WHERE job_id=%s ORDER BY number", (job_id,))
            return [dict(row) for row in cursor.fetchall()]

    def register_worker(
        self,
        worker_id: str,
        *,
        host: str,
        session: str,
        process: dict[str, Any] | None = None,
        grades: Sequence[tuple[str, str]] = (),
        replaces: str | None = None,
    ) -> None:
        """Give the row of the worker ``worker_id`` on ``host`` to ``session``, and renew its jobs' leases.

        The row keeps ``process``, the session's ``process_identity``, and ``grades``, its pairs of
        ``(dataset_sha256, grader_revision)``. The session takes over the jobs of earlier sessions. Raises
        Conflict if the row is live in another session, unless that session is ``replaces`` on ``host``."""
        capabilities = {
            "grades": [{"dataset_sha256": rows, "grader_revision": parser} for rows, parser in grades]
        }
        with self.transaction() as cursor:
            cursor.execute(
                """INSERT INTO rs_workers(worker_id,host,session,process,capabilities)
                VALUES(%s,%s,%s,%s::jsonb,%s::jsonb)
                ON CONFLICT(worker_id) DO UPDATE SET host=EXCLUDED.host,session=EXCLUDED.session,
                process=EXCLUDED.process,capabilities=EXCLUDED.capabilities,stopped=false,
                started_at=clock_timestamp(),seen_at=clock_timestamp()
                WHERE rs_workers.session=EXCLUDED.session
                OR rs_workers.stopped OR rs_workers.seen_at <= clock_timestamp() - %s * interval '1 second'
                OR (rs_workers.session=%s AND rs_workers.host=EXCLUDED.host)""",
                (
                    worker_id,
                    host,
                    session,
                    None if process is None else canonical(process),
                    canonical(capabilities),
                    WORKER_LIVE_S,
                    replaces,
                ),
            )
            if cursor.rowcount != 1:
                raise Conflict(f"Worker {worker_id} is live in another session")
            self._renew_leases(cursor, worker_id)

    def worker(self, worker_id: str) -> dict[str, Any] | None:
        """The row of the worker ``worker_id``, with its ``host``, ``session``, ``process`` and ``stopped``.

        None if the worker never registered.
        """
        with self.transaction() as cursor:
            cursor.execute("SELECT * FROM rs_workers WHERE worker_id=%s", (worker_id,))
            row = cursor.fetchone()
            return None if row is None else dict(row)

    def worker_heartbeat(self, worker_id: str, session: str) -> bool:
        """Record a heartbeat of the worker, renew the leases of its running jobs, and return True.

        Each lease then ends ``WORKER_LIVE_S`` after the heartbeat. Returns False, and changes nothing,
        if ``session`` does not hold the worker's row, or if the row is stopped.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """UPDATE rs_workers SET seen_at=clock_timestamp()
                WHERE worker_id=%s AND session=%s AND NOT stopped""",
                (worker_id, session),
            )
            if cursor.rowcount != 1:
                return False
            self._renew_leases(cursor, worker_id)
            return True

    def running_jobs(self, worker_id: str) -> list[dict[str, Any]]:
        """The running jobs of the worker ``worker_id``, with their lease tokens, oldest first."""
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT * FROM rs_jobs WHERE state='running' AND worker_id=%s ORDER BY created_at,id",
                (worker_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def lost_attempts(self, worker_id: str) -> list[dict[str, Any]]:
        """The lost attempts of the worker ``worker_id`` whose local data is not recorded as deleted.

        Each one is its ``id``, ``job_id`` and ``process``, oldest first. ``note_lost_data_deleted`` records
        the deletion."""
        with self.transaction() as cursor:
            cursor.execute(
                """SELECT id,job_id,execution->'process' AS process FROM rs_attempts
                WHERE worker_id=%s AND state='lost' AND data_deleted_at IS NULL ORDER BY started_at,id""",
                (worker_id,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def note_lost_data_deleted(self, attempt_id: str, *, log: str) -> None:
        """Record that the local data of the lost attempt ``attempt_id`` is deleted.

        ``log``, text from the attempt's log or empty, goes at the end of the attempt's error."""
        with self.transaction() as cursor:
            cursor.execute(
                """UPDATE rs_attempts SET error=concat(error,%s),data_deleted_at=clock_timestamp()
                WHERE id=%s AND state='lost'""",
                (log, attempt_id),
            )

    def worker_stopped(self, worker_id: str, session: str) -> None:
        """Mark the worker's row stopped if ``session`` holds it, with the stop as its last heartbeat.

        The worker is then dead, and another session can register at once. The row stays, so the host of
        the worker is lost only ``WORKER_LIVE_S`` after the stop.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """UPDATE rs_workers SET stopped=true,seen_at=clock_timestamp()
                WHERE worker_id=%s AND session=%s""",
                (worker_id, session),
            )

    @staticmethod
    def _renew_leases(cursor: RealDictCursor, worker_id: str) -> None:
        cursor.execute(
            """UPDATE rs_jobs SET lease_until=clock_timestamp() + %s * interval '1 second'
            WHERE state='running' AND worker_id=%s""",
            (WORKER_LIVE_S, worker_id),
        )

    def claim(
        self, worker_id: str, *, kinds: tuple[str, ...] = ("rollout", "grade", "review")
    ) -> dict[str, Any] | None:
        """Lease the next queued job of ``kinds`` to the live worker ``worker_id`` for ``WORKER_LIVE_S``.

        None: no job to claim now. The job's affinity key binds to the worker's host, if it is unbound. The
        filters and the order of the jobs are in docs/RUNSTORE.md ("Claims"). Raises Conflict if the worker
        is not live."""
        with self.transaction() as cursor:
            worker = self._live_worker(cursor, worker_id)
            self._settle_expired_grades(cursor)
            # A keep_until can pass after the settlement, so the select checks it again.
            cursor.execute(
                f"""SELECT j.*,a.host AS bound_host FROM rs_jobs j LEFT JOIN rs_affinity a ON a.key=j.affinity
                WHERE j.state='queued' AND j.ready_at <= clock_timestamp()
                AND j.attempt_count < j.max_attempts AND j.kind=ANY(%s) AND (j.bound_worker IS NULL
                    OR (j.bound_worker=%s AND j.keep_until > clock_timestamp()))
                AND (a.host IS NULL OR a.host=%s) AND {_GRADABLE}
                ORDER BY j.bound_worker IS NULL,a.host IS NULL,j.ready_at,j.created_at,j.id
                FOR UPDATE OF j SKIP LOCKED LIMIT 1""",
                (list(kinds), worker_id, worker["host"], canonical(worker["grades"])),
            )
            job = cursor.fetchone()
            if job is None or not self._bind(cursor, job, worker["host"]):
                return None
            attempt_id, token = uuid4().hex, uuid4().hex
            # A released attempt keeps its number but not its count, so attempt_count can lag the numbers.
            cursor.execute(
                """INSERT INTO rs_attempts(id,job_id,number,worker_id,lease_token)
                VALUES(%s,%s,(SELECT COALESCE(MAX(number),0)+1 FROM rs_attempts WHERE job_id=%s),%s,%s)""",
                (attempt_id, job["id"], job["id"], worker_id, token),
            )
            cursor.execute(
                """UPDATE rs_jobs SET state='running',phase='claimed',
                attempt_count=attempt_count+1,active_attempt=%s,lease_token=%s,
                lease_until=clock_timestamp() + %s * interval '1 second',worker_id=%s,
                updated_at=clock_timestamp() WHERE id=%s RETURNING *""",
                (attempt_id, token, WORKER_LIVE_S, worker_id, job["id"]),
            )
            return _one_row(cursor)

    @staticmethod
    def _live_worker(cursor: RealDictCursor, worker_id: str) -> dict[str, Any]:
        """The ``host`` and the ``grades`` of the live worker ``worker_id``. Raises Conflict if not live."""
        cursor.execute(
            f"SELECT host,capabilities->'grades' AS grades FROM rs_workers w WHERE worker_id=%s AND {_LIVE}",
            (worker_id,),
        )
        worker = cursor.fetchone()
        if worker is None:
            raise Conflict(
                f"Worker {worker_id} is not live: it stopped, or it has no heartbeat in {WORKER_LIVE_S} s"
            )
        return dict(worker)

    @staticmethod
    def _bind(cursor: RealDictCursor, job: dict[str, Any], host: str) -> bool:
        """Whether the affinity key of the selected ``job`` is bound to ``host``, once an unbound key is."""
        if job["bound_host"] is not None:
            return True
        # A claim on another host can bind the key first. The insert then waits for it to commit.
        cursor.execute(
            "INSERT INTO rs_affinity(key,host) VALUES(%s,%s) ON CONFLICT(key) DO NOTHING",
            (job["affinity"], host),
        )
        cursor.execute("SELECT host FROM rs_affinity WHERE key=%s", (job["affinity"],))
        return bool(_one_row(cursor)["host"] == host)

    @staticmethod
    def _settle_expired_grades(cursor: RealDictCursor) -> None:
        """Finish each queued bound grade whose ``keep_until`` has passed as a ``sandbox_lost`` grade."""
        # SKIP LOCKED keeps concurrent claims and sweeps from waiting for each other. A row that
        # another of them holds is settled or claimed by that one.
        cursor.execute(
            """WITH expired AS (
                SELECT id FROM rs_jobs WHERE state='queued' AND bound_worker IS NOT NULL
                AND keep_until <= clock_timestamp() FOR UPDATE SKIP LOCKED)
            UPDATE rs_jobs SET state='succeeded',phase='succeeded',result=%s::jsonb,error=NULL,
            updated_at=clock_timestamp() WHERE id IN (SELECT id FROM expired)""",
            (canonical(_SANDBOX_LOST),),
        )

    def settle_dead_workers(self) -> list[str]:
        """Settle the jobs of dead workers and of lost hosts, and return their ids.

        First, each queued bound grade past its ``keep_until`` succeeds as ``sandbox_lost``, as at a claim.
        Then a queued job of a lost host fails as ``data_lost``, a grade in a kept sandbox too. A running
        job of a dead worker is settled once its lease expired. A grade bound to a dead worker waits."""
        with self.transaction() as cursor:
            # Settle these before the lost hosts: an expired grade scores 0, also on a lost host (ADR-0006).
            self._settle_expired_grades(cursor)
            settled = self._fail_queued_of_lost_hosts(cursor)
            cursor.execute(
                f"""SELECT * FROM rs_jobs WHERE state='running' AND lease_until <= clock_timestamp()
                AND {_DEAD.format(column="rs_jobs.worker_id")}
                ORDER BY lease_until,id FOR UPDATE SKIP LOCKED LIMIT 100"""
            )
            jobs = [dict(row) for row in cursor.fetchall()]
            for job in jobs:
                self._settle_lost(cursor, job)
            return settled + [job["id"] for job in jobs]

    def _fail_queued_of_lost_hosts(self, cursor: RealDictCursor) -> list[str]:
        """Fail as ``data_lost`` each queued job whose key is bound to a lost host, and return their ids."""
        cursor.execute(
            f"""SELECT j.id,a.host FROM rs_jobs j JOIN rs_affinity a ON a.key=j.affinity
            WHERE j.state='queued' AND {_HOST_LOST.format(column="a.host")}
            ORDER BY j.id FOR UPDATE OF j SKIP LOCKED"""
        )
        jobs = [dict(row) for row in cursor.fetchall()]
        for job in jobs:
            self._end_job(cursor, job, state="failed", result=_data_lost(job["host"]), phase="data_lost")
        return [job["id"] for job in jobs]

    def _settle_lost(self, cursor: RealDictCursor, job: dict[str, Any]) -> None:
        """End the active attempt of ``job``, whose worker is dead, as ``lost``, and settle the job.

        A job of a lost host fails as ``data_lost``. Another job runs again from its spec, or fails, as
        ``finish`` with ``rerun`` decides. A grade in a kept sandbox stays bound to its worker.
        """
        result = {
            "status": "error",
            "failure_kind": "infrastructure",
            "error": f"Worker {job['worker_id']} stopped reporting heartbeats",
        }
        cursor.execute(
            f"SELECT host FROM rs_affinity a WHERE key=%s AND {_HOST_LOST.format(column='a.host')}",
            (job["affinity"],),
        )
        lost = cursor.fetchone()
        if lost is not None:
            # Only a worker on that host can claim a job of the key, and the host is lost (ADR-0007).
            self._end_attempt(cursor, job, state="lost", result=result)
            self._end_job(cursor, job, state="failed", result=_data_lost(lost["host"]), phase="data_lost")
            return
        self._rerun_or_fail(cursor, job, result, last="lost")

    def freeze_payload(self, job_id: str, token: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Freeze ``payload`` for the job's active attempt, and return it as the database stores it.

        Freezing the same payload again returns it. Raises Conflict if another payload is frozen, and
        ValueError if the payload fails ``encode_payload`` or its kind is not the job's.
        """
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            data = encode_payload(payload, job_id, job["active_attempt"])
            if payload["kind"] != job["kind"]:
                raise ValueError("Attempt payload kind differs from job")
            # Hash the payload in the form that jsonb stores, so that a later read hashes the same.
            cursor.execute("SELECT %s::jsonb AS payload", (data.decode("utf-8"),))
            frozen: dict[str, Any] = _one_row(cursor)["payload"]
            fingerprint = digest(frozen)
            cursor.execute(
                "SELECT payload,payload_hash FROM rs_attempts WHERE id=%s", (job["active_attempt"],)
            )
            current = _one_row(cursor)
            if current["payload"] is not None or current["payload_hash"] is not None:
                checked_payload(current["payload"], current["payload_hash"], job_id, job["active_attempt"])
                if current["payload_hash"] != fingerprint:
                    raise Conflict("Attempt payload is already frozen")
            else:
                cursor.execute(
                    "UPDATE rs_attempts SET payload=%s::jsonb,payload_hash=%s WHERE id=%s",
                    (canonical(frozen), fingerprint, job["active_attempt"]),
                )
            return frozen

    def payload(self, job_id: str, attempt_id: str) -> dict[str, Any]:
        """The frozen payload of the attempt ``attempt_id`` of ``job_id``.

        Raises KeyError if the job has no such attempt, and PayloadUnavailable if the payload is absent
        or fails its checks.
        """
        with self.transaction() as cursor:
            cursor.execute(
                "SELECT payload,payload_hash FROM rs_attempts WHERE job_id=%s AND id=%s", (job_id, attempt_id)
            )
            row = cursor.fetchone()
            if row is None:
                raise KeyError(attempt_id)
            return checked_payload(row["payload"], row["payload_hash"], job_id, attempt_id)

    @staticmethod
    def _fence(cursor: RealDictCursor, job_id: str, token: str) -> dict[str, Any]:
        """The job, locked for the transaction, if ``token`` owns its live lease. Raises Fenced if not."""
        cursor.execute(
            """SELECT * FROM rs_jobs WHERE id=%s AND lease_token=%s
            AND state='running' AND lease_until > clock_timestamp() FOR UPDATE""",
            (job_id, token),
        )
        job = cursor.fetchone()
        if job is None:
            raise Fenced("Attempt no longer owns a live lease")
        return dict(job)

    def progress(
        self, job_id: str, token: str, *, phase: str | None = None, execution: dict[str, Any] | None = None
    ) -> bool:
        """Record the progress of the job's active attempt, and return whether someone asked the job to stop.

        ``phase`` replaces the job's phase, and ``execution`` is merged into the attempt's. The lease does
        not change: the worker's heartbeat renews it. A stop request is a ``request_stop`` call.
        """
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            cursor.execute(
                "UPDATE rs_jobs SET phase=COALESCE(%s,phase),updated_at=clock_timestamp() WHERE id=%s",
                (phase, job_id),
            )
            if execution is not None:
                cursor.execute(
                    "UPDATE rs_attempts SET execution=execution || %s::jsonb WHERE id=%s",
                    (canonical(execution), job["active_attempt"]),
                )
            return job["stop_requested_at"] is not None

    def append_events(self, job_id: str, token: str, events: list[dict[str, Any]]) -> None:
        """Append ``events`` to the journal of the job's active attempt, in one transaction.

        An event appended again changes nothing. Raises ValueError if an event's ``seq`` is not a positive
        int, and Conflict if a ``seq`` already names another event.
        """
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            for event in events:
                seq = event.get("seq")
                if type(seq) is not int or seq < 1:
                    raise ValueError("Journal sequence must be positive")
                cursor.execute(
                    """INSERT INTO rs_events(attempt_id,seq,event) VALUES(%s,%s,%s::jsonb)
                    ON CONFLICT(attempt_id,seq) DO NOTHING""",
                    (job["active_attempt"], seq, canonical(event)),
                )
                if cursor.rowcount == 0:
                    cursor.execute(
                        "SELECT event FROM rs_events WHERE attempt_id=%s AND seq=%s",
                        (job["active_attempt"], seq),
                    )
                    if _one_row(cursor)["event"] != event:
                        raise Conflict("Journal sequence was rewritten")

    def events(self, attempt_id: str, after: int = 0, limit: int = 1000) -> list[dict[str, Any]]:
        """The journal events of an attempt whose ``seq`` is above ``after``, in order, at most ``limit``."""
        with self.transaction() as cursor:
            cursor.execute(
                """SELECT event FROM rs_events WHERE attempt_id=%s AND seq>%s
                ORDER BY seq LIMIT %s""",
                (attempt_id, after, min(10000, max(1, limit))),
            )
            return [row["event"] for row in cursor.fetchall()]

    def finish(
        self,
        job_id: str,
        token: str,
        result: dict[str, Any],
        *,
        state: str = "succeeded",
        rerun: bool = False,
    ) -> None:
        """End the job's active attempt with ``result``, in ``state``: succeeded or failed.

        With ``rerun``, the failed attempt is lost: the job runs again from its spec while attempts remain
        and no stop was requested. Else it fails, with each counted attempt's error in ``attempt_errors``.
        """
        if state not in {"succeeded", "failed"}:
            raise ValueError("Invalid terminal outcome")
        if rerun and (state != "failed" or result.get("failure_kind") != "infrastructure"):
            raise ValueError("Only an infrastructure failure runs again")
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            if rerun:
                self._rerun_or_fail(cursor, job, result, last="failed")
                return
            self._end_attempt(cursor, job, state=state, result=result)
            self._end_job(cursor, job, state=state, result=result)

    @staticmethod
    def _end_attempt(
        cursor: RealDictCursor, job: dict[str, Any], *, state: str, result: dict[str, Any]
    ) -> None:
        cursor.execute(
            """UPDATE rs_attempts SET state=%s,result=%s::jsonb,error=%s,finished_at=clock_timestamp()
            WHERE id=%s""",
            (state, canonical(result), result.get("error"), job["active_attempt"]),
        )

    @staticmethod
    def _end_job(
        cursor: RealDictCursor,
        job: dict[str, Any],
        *,
        state: str,
        result: dict[str, Any],
        phase: str | None = None,
    ) -> None:
        """Finish ``job`` in ``state`` with ``result``. Its phase is ``phase``, else ``state``."""
        cursor.execute(
            """UPDATE rs_jobs SET state=%s,phase=%s,result=%s::jsonb,error=%s,
            lease_until=NULL,lease_token=NULL,updated_at=clock_timestamp() WHERE id=%s""",
            (state, phase or state, canonical(result), result.get("error"), job["id"]),
        )

    def _rerun_or_fail(
        self, cursor: RealDictCursor, job: dict[str, Any], result: dict[str, Any], *, last: str
    ) -> None:
        """Queue the job to run again after a backoff, or fail it when no attempt is left or a stop was asked.

        A rerun ends the attempt as ``lost`` and forgets its records. Otherwise the attempt ends in ``last``.
        A grade in a kept sandbox without an attempt left succeeds as ``sandbox_lost`` with reward 0.
        """
        if job["attempt_count"] < job["max_attempts"] and job["stop_requested_at"] is None:
            self._end_attempt(cursor, job, state="lost", result=result)
            self._forget_attempt(cursor, job["active_attempt"])
            cursor.execute(
                """UPDATE rs_jobs SET state='queued',phase='retry_wait',result=%s::jsonb,error=%s,
                lease_until=NULL,lease_token=NULL,updated_at=clock_timestamp(),
                ready_at=clock_timestamp()+%s*interval '1 second' WHERE id=%s""",
                (canonical(result), result.get("error"), 30 * job["attempt_count"], job["id"]),
            )
            return
        self._end_attempt(cursor, job, state=last, result=result)
        if job["bound_worker"] is not None and job["stop_requested_at"] is None:
            # A grade in a kept sandbox has no other sandbox to run in, so its lost sandbox scores 0.
            self._end_job(cursor, job, state="succeeded", result=_SANDBOX_LOST)
            return
        cursor.execute(
            "SELECT error FROM rs_attempts WHERE job_id=%s AND state<>'released' ORDER BY number",
            (job["id"],),
        )
        errors = [row["error"] for row in cursor.fetchall()]
        self._end_job(cursor, job, state="failed", result={**result, "attempt_errors": errors})

    @staticmethod
    def _forget_attempt(cursor: RealDictCursor, attempt_id: str) -> None:
        """Delete the attempt's events, tool calls and recovery points, and the prefix nodes only it has."""
        cursor.execute("DELETE FROM rs_events WHERE attempt_id=%s", (attempt_id,))
        cursor.execute("DELETE FROM rs_recoveries WHERE attempt_id=%s", (attempt_id,))
        # A node is the digest of its scope and calls, so another attempt can reach the same one.
        cursor.execute(
            """WITH removed AS (DELETE FROM rs_tools WHERE attempt_id=%s RETURNING prefix_node)
            DELETE FROM rs_prefix_nodes node WHERE node.node IN (SELECT prefix_node FROM removed)
            AND NOT EXISTS (SELECT 1 FROM rs_tools t WHERE t.prefix_node=node.node AND t.attempt_id<>%s)
            AND NOT EXISTS (SELECT 1 FROM rs_recoveries r WHERE r.prefix_node=node.node)""",
            (attempt_id, attempt_id),
        )

    def release(self, job_id: str, token: str, result: dict[str, Any]) -> None:
        """End the job's active attempt as ``released`` with ``result``, and queue the job, ready at once.

        The attempt does not count against ``max_attempts``, and its events, tool calls and recovery points
        are deleted, as for a lost attempt. If a stop was requested, the job is cancelled instead.
        """
        with self.transaction() as cursor:
            job = self._fence(cursor, job_id, token)
            self._end_attempt(cursor, job, state="released", result=result)
            self._forget_attempt(cursor, job["active_attempt"])
            # A stop request cancels a queued job, as request_stop() does.
            next_state, phase = (
                ("cancelled", "cancelled") if job["stop_requested_at"] else ("queued", "released")
            )
            cursor.execute(
                """UPDATE rs_jobs SET state=%s,phase=%s,attempt_count=attempt_count-1,
                lease_until=NULL,lease_token=NULL,ready_at=clock_timestamp(),
                updated_at=clock_timestamp() WHERE id=%s""",
                (next_state, phase, job_id),
            )

    def cancel_queued(self, job_id: str) -> None:
        """Cancel a queued job. Raises Conflict if the job is not queued."""
        with self.transaction() as cursor:
            cursor.execute(
                """UPDATE rs_jobs SET state='cancelled',phase='cancelled',
                updated_at=clock_timestamp() WHERE id=%s AND state='queued'""",
                (job_id,),
            )
            if cursor.rowcount != 1:
                raise Conflict("Only a queued job can be cancelled; a running one needs a stop request")

    def request_stop(self, job_id: str) -> dict[str, Any]:
        """Ask a job to stop, and return the job after the request. Raises KeyError for an unknown job.

        A queued job is cancelled, and a finished job stays. A running job records the first request:
        ``progress`` returns True, and no ``finish``, ``release`` or sweep queues the job again.
        """
        with self.transaction() as cursor:
            cursor.execute(
                """UPDATE rs_jobs SET state='cancelled',phase='cancelled',
                updated_at=clock_timestamp() WHERE id=%s AND state='queued'""",
                (job_id,),
            )
            cursor.execute(
                """UPDATE rs_jobs SET stop_requested_at=COALESCE(stop_requested_at, clock_timestamp()),
                updated_at=clock_timestamp() WHERE id=%s AND state='running'""",
                (job_id,),
            )
        return self.get(job_id)

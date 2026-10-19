"""The task gate: an ``oracle`` and a ``nop`` rollout of each task, each graded by the task's own verifier.

``check_record`` and ``already_ok`` are small tests. ``run_gate`` runs through the test's own Run
Store, whose jobs the fake worker of ``tests.eval.worker`` settles between two looks of the gate. The
tasks are copies of the Harbor fixture task.
"""

import json
import re
import shutil
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from sprout.eval import gate
from sprout.runstore.store import Store
from sprout.search.client import RunStoreClient
from tests.eval.worker import Worker, final_snapshot_id
from tests.support.paths import HARBOR_SHELL_TASK

IMAGE_BACKEND = {
    "podman": {"image_env": True, "image_user": True},
    "microvm": {"image_env": True, "image_user": True},
}


def stage(state: str, job_id: str, **fields: Any) -> dict[str, Any]:
    return {"state": state, "job_id": job_id, **fields}


def settled(actor: dict[str, Any], grade: dict[str, Any]) -> dict[str, Any]:
    """The settled group document of one check, under the task's key ``gate-key``."""
    return {
        "affinity": "gate-key",
        "ready": True,
        "samples": [{"sample_slot_id": "nop", "actor": actor, "grade": grade}],
    }


@pytest.mark.parametrize(
    "actor, grade, record",
    [
        (
            stage("succeeded", "job-0"),
            stage("succeeded", "job-1", result={"reward": 0.0, "reason": None}),
            {"reward": 0.0, "reason": None, "ok": True},
        ),
        (
            stage("succeeded", "job-0"),
            stage("succeeded", "job-1", result={"reward": 1.0, "reason": None}),
            {"reward": 1.0, "reason": None, "ok": False},
        ),
        (
            stage("succeeded", "job-0"),
            stage("succeeded", "job-1", result={"reward": 0.0, "reason": "sandbox_lost"}),
            {
                "reward": 0.0,
                "reason": "sandbox_lost",
                "ok": False,
                "error": "the verifier wrote no reward: sandbox_lost",
            },
        ),
        (
            stage("succeeded", "job-0"),
            stage("failed", "job-1", error="verifier crashed"),
            {"ok": False, "error": "the grade failed: verifier crashed"},
        ),
        (
            stage("failed", "job-0", error="mini did not submit"),
            stage("skipped", None, error="Actor did not succeed"),
            {"ok": False, "error": "the rollout failed: mini did not submit"},
        ),
        (
            stage("failed", "job-0", error="Host host-1 is lost", result={"reason": "data_lost"}),
            stage("skipped", None, error="Actor did not succeed"),
            {"ok": False, "reason": "data_lost", "error": "the rollout failed: Host host-1 is lost"},
        ),
    ],
    ids=[
        "expected-reward",
        "wrong-reward",
        "no-reward-written",
        "grade-failed",
        "rollout-failed",
        "data-lost",
    ],
)
def test_a_check_passes_only_on_the_expected_reward_that_the_verifier_wrote(
    actor: dict[str, Any], grade: dict[str, Any], record: dict[str, Any]
) -> None:
    """A reward of 0 without the verifier's word, as in a lost kept sandbox, proves nothing about a nop.

    A job of a lost host is a check without a verdict, for the reason ``data_lost``."""
    assert gate.check_record(settled(actor, grade), task="shell-task", mode="nop") == {
        "task": "shell-task",
        "mode": "nop",
        "expected": 0.0,
        "affinity": "gate-key",
        "rollout_job": "job-0",
        "grade_job": grade["job_id"],
        **record,
    }


def test_a_rerun_skips_exactly_the_checks_that_a_line_of_the_log_records_as_passed(tmp_path: Path) -> None:
    """A line that is not a JSON object, such as the torn last line of a killed gate, counts for nothing."""
    log = tmp_path / "gate.jsonl"
    lines = [
        {"task": "a", "mode": "oracle", "ok": True},
        {"task": "a", "mode": "nop", "ok": False},
        {"task": "b", "mode": "nop", "ok": True},
        ["not", "a", "record"],
    ]
    log.write_text("\n".join(json.dumps(line) for line in lines) + '\n{"task": "c", "mode": "or')

    assert gate.already_ok(log) == {("a", "oracle"), ("b", "nop")}


def test_a_gate_without_a_log_has_passed_no_check(tmp_path: Path) -> None:
    assert gate.already_ok(tmp_path / "gate.jsonl") == set()


@pytest.mark.parametrize("extra", [["--workers", "0"], ["--poll-interval", "0"]])
def test_a_gate_without_workers_or_with_no_pause_between_looks_is_refused(extra: list[str]) -> None:
    with pytest.raises(SystemExit):
        gate.parse_args(["--tasks-dir", "/path/to/tasks", "--output", "/path/to/gate", *extra])


def test_a_gate_without_a_run_store_token_fails_before_it_writes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPROUT_RUNSTORE_TOKEN", raising=False)

    with pytest.raises(ValueError, match=r"^\$SPROUT_RUNSTORE_TOKEN holds no Run Store token$"):
        gate.main(["--tasks-dir", str(tmp_path / "tasks"), "--output", str(tmp_path / "gate")])

    assert not (tmp_path / "gate").exists()


# --- gate runs through the Run Store ----------------------------------------------------------


def tasks(root: Path, *names: str, separate: tuple[str, ...] = (), imageless: tuple[str, ...] = ()) -> Path:
    """Copies of the fixture task in ``root/tasks``, each named after its directory.

    A task in ``separate`` verifies in a sandbox of its own (tests/Dockerfile).
    A task in ``imageless`` has no image."""
    for name in names:
        task = Path(shutil.copytree(HARBOR_SHELL_TASK, root / "tasks" / name))
        toml = (task / "task.toml").read_text()
        if name in separate:
            toml = (
                toml.replace("[verifier]\n", '[verifier]\nenvironment_mode = "separate"\n')
                + "\n[verifier.environment]\n"
            )
            (task / "tests" / "Dockerfile").write_text("FROM python:3.11-slim\n")
        if name in imageless:
            toml = toml.replace('docker_image = "python:3.11-slim"\n', "")
        (task / "task.toml").write_text(toml)
    return root / "tasks"


def scores(
    table: dict[tuple[str, str], tuple[float, str | None]] | None = None,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """The verdict of each check: ``table`` maps (task, mode) to (reward, reason).

    Any other check scores what it expects."""

    def verdict(record: dict[str, Any]) -> dict[str, Any]:
        task, mode = record["rollout_job_id"].rsplit("-", 1)
        reward, reason = (table or {}).get((task, mode), (gate.EXPECTED[mode], None))
        return {"reward": reward, "reason": reason}

    return verdict


def expected_verdict(record: dict[str, Any]) -> dict[str, Any]:
    """The verdict that each check expects, by its mode: the id of the sample of the check's group."""
    return {"reward": gate.EXPECTED[record["sample_slot_id"]], "reason": None}


def run(tmp_path: Path, client: RunStoreClient, worker: Worker, *extra: str) -> int:
    """``python -m sprout.eval.gate`` of ``tmp_path/tasks`` into ``tmp_path/gate``, with ``extra``."""
    args = gate.parse_args(
        ["--tasks-dir", str(tmp_path / "tasks"), "--output", str(tmp_path / "gate"), *extra]
    )
    return gate.run_gate(args, client=client, pause=worker)


def worker(store: Store, tmp_path: Path, **settings: Any) -> Worker:
    return Worker(store, artifacts=tmp_path / "artifacts", context_key="gate", **settings)


def records(tmp_path: Path) -> list[dict[str, Any]]:
    """The lines of ``gate.jsonl``."""
    return [json.loads(line) for line in (tmp_path / "gate" / "gate.jsonl").read_text().splitlines()]


def jobs(store: Store) -> list[dict[str, Any]]:
    """The jobs of ``store`` in the order of their submission."""
    return list(reversed(store.list_jobs(limit=1000)))


def checks(store: Store) -> list[str]:
    """Each job of ``store`` in submission order, as its check, and ``grade`` after the check of a grade."""
    return [
        job["request"]["context"]["gate"]["rollout_job_id"] + (" grade" if job["kind"] == "grade" else "")
        for job in jobs(store)
    ]


def test_the_gate_reports_a_task_unless_its_oracle_scores_1_and_its_nop_scores_0(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """A verifier that passes the untouched task measures nothing."""
    tasks(tmp_path, "good", "bad")

    code = run(
        tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores({("bad", "nop"): (1.0, None)}))
    )

    assert code == 1
    assert [(r["task"], r["mode"], r["reward"], r["ok"]) for r in records(tmp_path)] == [
        ("bad", "oracle", 1.0, True),
        ("good", "oracle", 1.0, True),
        ("bad", "nop", 1.0, False),
        ("good", "nop", 0.0, True),
    ]
    summary = json.loads((tmp_path / "gate" / "summary.json").read_text())
    assert (summary["failed_tasks"], summary["ok"], summary["tasks"]) == (
        ["bad"],
        ["bad/oracle", "good/nop", "good/oracle"],
        2,
    )


def test_a_gate_whose_checks_all_pass_exits_0(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    tasks(tmp_path, "good")

    assert run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores())) == 0


@pytest.mark.parametrize(
    "settings, error",
    [
        (
            {"grade": scores({("task", "nop"): (0.0, "sandbox_lost")})},
            "the verifier wrote no reward: sandbox_lost",
        ),
        ({"failing": frozenset({"nop"})}, "the rollout failed: mini did not submit"),
    ],
    ids=["lost-sandbox", "failed-rollout"],
)
def test_a_nop_check_that_the_verifier_did_not_score_fails(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, settings: dict[str, Any], error: str
) -> None:
    tasks(tmp_path, "task")

    code = run(tmp_path, run_store_client, worker(run_store, tmp_path, **settings), "--mode", "nop")

    (record,) = records(tmp_path)
    assert (code, record["ok"], record["error"]) == (1, False, error)


def test_each_check_runs_its_agent_without_checkpoints_and_grades_it_where_the_verifier_runs(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """A verifier that shares the agent's environment grades the kept sandbox, another the final snapshot."""
    tasks(tmp_path, "shared", "separate", separate=("separate",))
    options = ["--mode", "oracle", "--profile", "actors", "--grade-profile", "graders"]

    run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores()), *options)

    separate_rollout, shared_rollout, separate_grade, shared_grade = jobs(run_store)
    task = str((tmp_path / "tasks" / "separate").resolve())
    assert (separate_rollout["request"]["profile"], separate_rollout["request"]["spec"]) == (
        "actors",
        {
            "prompt": "Write `ready` followed by a newline into `/app/answer.txt`.\n",
            "agent": "oracle",
            "harbor_task": task,
            "sandbox_image": "python:3.11-slim",
            "timeout_s": 60.0,
            "checkpoint_enabled": False,
            "keep_sandbox": False,
            "backend": IMAGE_BACKEND,
            "sandbox_resources": {"cpu": 1, "memory_mb": 512},
        },
    )
    assert (separate_grade["request"]["profile"], separate_grade["request"]["spec"]) == (
        "graders",
        {
            "benchmark": "harbor",
            "task_dir": task,
            "timeout_s": 60.0,
            "backend": IMAGE_BACKEND,
            "snapshot_id": final_snapshot_id(separate_rollout["id"]),
        },
    )
    assert shared_grade["request"]["spec"]["in_place_job"] == shared_rollout["id"]


@pytest.mark.parametrize(
    "flag, image",
    [
        ("--no-image-env", {"image_env": False, "image_user": True}),
        ("--no-image-user", {"image_env": True, "image_user": False}),
    ],
)
def test_each_image_flag_turns_its_setting_off_in_the_rollout_and_its_grade(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, flag: str, image: dict[str, bool]
) -> None:
    tasks(tmp_path, "task")

    run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores()), "--mode", "oracle", flag)

    rollout, grade = jobs(run_store)
    assert [rollout["request"]["spec"]["backend"], grade["request"]["spec"]["backend"]] == [
        {"podman": image, "microvm": image},
        {"podman": image, "microvm": image},
    ]


@pytest.mark.parametrize(
    "workers, order",
    [
        (
            "1",
            [
                "a-oracle",
                "a-oracle grade",
                "a-nop",
                "a-nop grade",
                "b-oracle",
                "b-oracle grade",
                "b-nop",
                "b-nop grade",
            ],
        ),
        (
            "2",
            [
                "a-oracle",
                "b-oracle",
                "a-oracle grade",
                "b-oracle grade",
                "a-nop",
                "b-nop",
                "a-nop grade",
                "b-nop grade",
            ],
        ),
    ],
)
def test_the_checks_of_a_task_run_one_after_the_other_and_workers_tasks_at_a_time(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient, workers: str, order: list[str]
) -> None:
    """The nop starts after the oracle's grade settles, so two sandboxes of one task never run at once."""
    tasks(tmp_path, "a", "b")

    run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores()), "--workers", workers)

    assert checks(run_store) == order


def test_the_checks_of_one_task_carry_one_key_and_the_checks_of_two_tasks_two_keys(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """ADR-0007: a gate run makes one key for each task, and the record of each check names its key."""
    tasks(tmp_path, "a", "b")

    run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores()))

    rollout_a, rollout_b = jobs(run_store)[:2]
    key_a, key_b = rollout_a["affinity"], rollout_b["affinity"]
    assert (re.fullmatch("gate-[0-9a-f]{32}", key_a) is not None, key_a != key_b) == (True, True), key_a
    assert [
        (check, job["affinity"]) for check, job in zip(checks(run_store), jobs(run_store), strict=True)
    ] == [
        ("a-oracle", key_a),
        ("b-oracle", key_b),
        ("a-oracle grade", key_a),
        ("b-oracle grade", key_b),
        ("a-nop", key_a),
        ("b-nop", key_b),
        ("a-nop grade", key_a),
        ("b-nop grade", key_b),
    ]
    assert [(r["task"], r["mode"], r["affinity"]) for r in records(tmp_path)] == [
        ("a", "oracle", key_a),
        ("b", "oracle", key_b),
        ("a", "nop", key_a),
        ("b", "nop", key_b),
    ]


def test_a_check_whose_host_is_lost_runs_again_under_the_tasks_next_key_and_so_do_its_later_checks(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """The lost host keeps the task's key, so no later job of the task carries it.

    The group of the earlier run stays, beside the groups of the run again."""
    tasks(tmp_path, "task")
    pause = worker(run_store, tmp_path, grade=expected_verdict, losing=["oracle"])

    code = run(tmp_path, run_store_client, pause)

    first = jobs(run_store)[0]["affinity"]
    again = first + "-r1"
    pairs = [(check, job["affinity"]) for check, job in zip(checks(run_store), jobs(run_store), strict=True)]
    assert (code, pairs) == (
        0,
        [
            ("task-oracle", first),
            ("task-oracle-r1", again),
            ("task-oracle-r1 grade", again),
            ("task-nop-r1", again),
            ("task-nop-r1 grade", again),
        ],
    )
    assert [(r["mode"], r["affinity"], r["ok"]) for r in records(tmp_path)] == [
        ("oracle", again, True),
        ("nop", again, True),
    ]
    assert sorted(path.name for path in (tmp_path / "gate" / "groups").glob("*/*.json")) == [
        "task-nop-r1.json",
        "task-oracle-r1.json",
        "task-oracle.json",
    ]


def test_a_check_whose_third_run_loses_its_host_fails_with_the_reason_data_lost(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """The checks of a task run again at most 2 times in one gate run, as a task evaluation does."""
    tasks(tmp_path, "task")
    pause = worker(run_store, tmp_path, grade=expected_verdict, losing=["nop"] * 3)

    code = run(tmp_path, run_store_client, pause, "--mode", "nop")

    (record,) = records(tmp_path)
    first = jobs(run_store)[0]["affinity"]
    assert (
        code,
        record["affinity"],
        record["ok"],
        record["reason"],
        record["error"],
        len(jobs(run_store)),
    ) == (
        1,
        first + "-r2",
        False,
        "data_lost",
        "the rollout failed: Host host-1 is lost: none of its workers reported a heartbeat in 600 s",
        3,
    )


def test_a_rerun_runs_again_only_the_checks_that_did_not_pass_and_as_new_jobs(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    """Each run submits jobs of its own, so the failed check never reads its old result."""
    tasks(tmp_path, "task")
    run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores({("task", "nop"): (1.0, None)})))
    before = len(jobs(run_store))

    code = run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores()))

    assert (code, checks(run_store)[before:]) == (0, ["task-nop", "task-nop grade"])
    assert json.loads((tmp_path / "gate" / "summary.json").read_text())["failed_tasks"] == []


def test_a_task_that_no_rollout_can_run_fails_its_checks_and_the_others_go_on(
    tmp_path: Path, run_store: Store, run_store_client: RunStoreClient
) -> None:
    tasks(tmp_path, "imageless", "task", imageless=("imageless",))

    code = run(tmp_path, run_store_client, worker(run_store, tmp_path, grade=scores()))

    imageless = tmp_path / "tasks" / "imageless"
    refusal = f"ValueError: {imageless}: a rollout starts from the task's docker_image, and it names none"
    assert (code, [(r["task"], r["mode"], r["ok"], r.get("error")) for r in records(tmp_path)]) == (
        1,
        [
            ("imageless", "oracle", False, refusal),
            ("imageless", "nop", False, refusal),
            ("task", "oracle", True, None),
            ("task", "nop", True, None),
        ],
    )


@pytest.mark.parametrize(
    "extra, expected",
    [
        (lambda tmp_path: [], ["a", "b", "c"]),
        (lambda tmp_path: ["--task", "c", "--task", "a"], ["c", "a"]),
        (lambda tmp_path: ["--selection", str(tmp_path / "selection.json")], ["b"]),
    ],
    ids=["every-task", "named-tasks", "a-selection"],
)
def test_the_gate_checks_the_named_tasks_or_else_every_task_directory(
    tmp_path: Path,
    run_store: Store,
    run_store_client: RunStoreClient,
    extra: Callable[[Path], list[str]],
    expected: list[str],
) -> None:
    """A directory without a task.toml is no task. ``--selection`` is a JSON list of task directory names."""
    tasks(tmp_path, "c", "a", "b")
    (tmp_path / "tasks" / "notes").mkdir()
    (tmp_path / "selection.json").write_text(json.dumps(["b"]))
    pause = worker(run_store, tmp_path, grade=scores())

    code = run(tmp_path, run_store_client, pause, "--mode", "nop", "--workers", "3", *extra(tmp_path))

    assert (code, [record["task"] for record in records(tmp_path)]) == (0, expected)

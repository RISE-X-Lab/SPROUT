"""``ops/prepull_images.py``: each task image is pulled and its base extracted before an evaluation runs.

The script runs as a child process against the fake ``podman``, ``btrfs`` and ``findmnt`` of
``tests/support``, which the child finds first on PATH.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys

from tests.support.paths import REPO

SCRIPT = REPO / "ops" / "prepull_images.py"


def write_task(tasks, name, environment):
    """A Harbor task directory whose task.toml has the ``[environment]`` lines given."""
    (tasks / name).mkdir(parents=True)
    (tasks / name / "task.toml").write_text("[environment]\n" + environment)


def prepull(*args):
    return subprocess.run(
        [sys.executable, str(SCRIPT), *map(str, args)], capture_output=True, text=True, cwd=REPO, timeout=120
    )


def records(state):
    return [json.loads(line) for line in state.read_text().splitlines()]


def outcomes(state):
    """Each record as (image, backend, ok, base)."""
    return [
        (record["image"], record.get("backend"), record["ok"], record.get("base"))
        for record in records(state)
    ]


def commands(fake_podman):
    """The first argv word of each call to the fake podman, sorted."""
    return sorted(call[0] for call in fake_podman.calls())


def bases(state):
    """The base of each record, or None for a record of a failure."""
    return [record.get("base") for record in records(state)]


def base_of(store, reference):
    """Where the fake podman's image id of ``reference`` puts its base in ``store``."""
    return str(store / "bases" / hashlib.sha256(reference.encode()).hexdigest())


def test_each_task_image_is_pulled_and_extracted_into_the_store_without_starting_a_container(
    fake_podman, fake_btrfs_store, fake_runtime_bin, tmp_path
):
    tasks = tmp_path / "tasks"
    write_task(tasks, "a", 'docker_image = "public.ecr.aws/example/swe:task-a"\n')
    write_task(tasks, "b", 'docker_image = "python:3.11-slim"\n')
    write_task(tasks, "c", "")
    state = tmp_path / "state" / "prepull.jsonl"

    result = prepull(
        "--store",
        fake_btrfs_store,
        "--tasks-dir",
        tasks,
        "--runtime-bin",
        fake_runtime_bin,
        "--state",
        state,
        "--workers",
        1,
    )

    assert result.returncode == 0, result.stdout + result.stderr
    assert outcomes(state) == [
        (
            "public.ecr.aws/example/swe:task-a",
            "podman",
            True,
            base_of(fake_btrfs_store, "public.ecr.aws/example/swe:task-a"),
        ),
        ("python:3.11-slim", "podman", True, base_of(fake_btrfs_store, "docker.io/library/python:3.11-slim")),
    ]
    # export and unshare tar start together, so their order varies from run to run.
    assert commands(fake_podman) == [
        "create",
        "create",
        "export",
        "export",
        "image",
        "image",
        "rm",
        "rm",
        "unshare",
        "unshare",
    ]


def test_a_rerun_pulls_only_the_images_without_an_ok_record_of_the_same_backend(
    fake_podman, fake_btrfs_store, fake_runtime_bin, tmp_path
):
    """A record without a backend is from the microvm backend, which wrote no backend key."""
    tasks = tmp_path / "tasks"
    write_task(tasks, "a", 'docker_image = "registry.example/a:1"\n')
    write_task(tasks, "b", 'docker_image = "registry.example/b:1"\n')
    write_task(tasks, "c", 'docker_image = "registry.example/c:1"\n')
    state = tmp_path / "prepull.jsonl"
    state.write_text(
        json.dumps({"image": "registry.example/a:1", "ok": True})
        + "\n"
        + json.dumps({"image": "registry.example/b:1", "ok": True, "backend": "podman"})
        + "\n"
        + json.dumps({"image": "registry.example/c:1", "ok": False, "backend": "podman", "error": "x"})
        + "\n"
    )

    result = prepull(
        "--store",
        fake_btrfs_store,
        "--tasks-dir",
        tasks,
        "--runtime-bin",
        fake_runtime_bin,
        "--state",
        state,
        "--workers",
        1,
    )

    assert result.stdout.splitlines()[0] == "images 3 total, 1 already pulled for podman, 2 to pull"
    assert outcomes(state)[3:] == [
        ("registry.example/a:1", "podman", True, base_of(fake_btrfs_store, "registry.example/a:1")),
        ("registry.example/c:1", "podman", True, base_of(fake_btrfs_store, "registry.example/c:1")),
    ]


def test_a_failed_pull_is_recorded_and_the_script_exits_1(
    fake_podman, fake_btrfs_store, fake_runtime_bin, tmp_path
):
    tasks = tmp_path / "tasks"
    write_task(tasks, "a", 'docker_image = "registry.example/a:1"\n')
    fake_podman.unpulled("registry.example/a:1")
    fake_podman.fail("pull")
    state = tmp_path / "prepull.jsonl"

    result = prepull(
        "--store", fake_btrfs_store, "--tasks-dir", tasks, "--runtime-bin", fake_runtime_bin, "--state", state
    )

    assert (result.returncode, records(state), result.stdout.splitlines()[-1]) == (
        1,
        [
            {
                "image": "registry.example/a:1",
                "backend": "podman",
                "ok": False,
                "error": "podman pull registry.example/a:1 failed (exit 125): Error: simulated pull failure",
            }
        ],
        "re-run this script to retry the failures (state is per-image).",
    )


def test_the_store_defaults_to_sprout_podman_store(
    fake_podman, fake_btrfs_store, fake_runtime_bin, tmp_path, monkeypatch
):
    tasks = tmp_path / "tasks"
    write_task(tasks, "a", 'docker_image = "registry.example/a:1"\n')
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(fake_btrfs_store))
    state = tmp_path / "prepull.jsonl"

    result = prepull("--tasks-dir", tasks, "--runtime-bin", fake_runtime_bin, "--state", state)

    assert (result.returncode, bases(state)) == (0, [base_of(fake_btrfs_store, "registry.example/a:1")])

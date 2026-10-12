"""``ops/prepull_images.py``: the pre-pull step consumes the backends' cache-filling paths.

Run as a subprocess against the fake ``podman``/``btrfs``/``findmnt`` from
``tests/conftest.py`` (installed on PATH by the ``fake_podman`` and ``store`` fixtures and inherited
by the child), so what is asserted is the wiring: task images are collected
from ``task.toml`` files, each is prepared through ``PodmanPool.prepare`` with
no container started, and the state file records the backend so a re-run
skips only what THIS backend has.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

from tests.conftest import recorded_calls

SCRIPT = Path(__file__).resolve().parents[2] / "ops" / "prepull_images.py"
IMAGES = ("public.ecr.aws/example/swe:task-a", "docker.io/library/python:3.11-slim")


def write_tasks(root: Path) -> None:
    for n, image in enumerate(IMAGES):
        task = root / f"task-{n}"
        task.mkdir(parents=True)
        (task / "task.toml").write_text(f'[environment]\ndocker_image = "{image}"\n')
    (root / "task-2").mkdir()
    (root / "task-2" / "task.toml").write_text("[environment]\n")  # no image: skipped


def run(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True,
                          cwd=SCRIPT.parents[1], timeout=120)


def test_the_podman_mode_prepares_every_task_image_through_the_pool(fake_podman, store, runtime_bin, tmp_path):
    tasks = tmp_path / "tasks"
    write_tasks(tasks)
    state = tmp_path / "state" / "prepull.jsonl"
    # An older microvm record for one image must not count for podman.
    state.parent.mkdir()
    state.write_text(json.dumps({"image": IMAGES[0], "ok": True}) + "\n")

    result = run("--backend", "podman", "--store", str(store), "--tasks-dir", str(tasks),
                 "--runtime-bin", str(runtime_bin), "--state", str(state), "--workers", "2")
    assert result.returncode == 0, result.stdout + result.stderr
    assert "images 2 total, 0 already pulled for podman, 2 to pull" in result.stdout
    assert "done: 2 ok, 0 failed" in result.stdout

    records = [json.loads(line) for line in state.read_text().splitlines()][1:]
    assert sorted(r["image"] for r in records) == sorted(IMAGES)
    for record in records:
        digest = hashlib.sha256(record["image"].encode()).hexdigest()
        assert record["ok"] and record["backend"] == "podman"
        assert record["base"] == str(store / "bases" / digest) and Path(record["base"]).is_dir()
    podman = recorded_calls(store.parent, "podman")
    assert sorted(c[-1] for c in podman if c[0] == "create") == sorted(IMAGES), "one extraction per image"
    assert not [c for c in podman if c[0] in ("run", "start")], "no container was started"

    again = run("--backend", "podman", "--store", str(store), "--tasks-dir", str(tasks),
                "--runtime-bin", str(runtime_bin), "--state", str(state))
    assert again.returncode == 0 and "2 already pulled for podman, 0 to pull" in again.stdout


def test_podman_is_the_default_and_takes_the_default_store(fake_podman, store, runtime_bin, tmp_path,
                                                           monkeypatch):
    tasks = tmp_path / "tasks"
    write_tasks(tasks)
    state = tmp_path / "prepull.jsonl"
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(store))
    result = run("--tasks-dir", str(tasks), "--runtime-bin", str(runtime_bin), "--state", str(state))
    assert result.returncode == 0, result.stdout + result.stderr
    records = [json.loads(line) for line in state.read_text().splitlines()]
    assert len(records) == 2 and all(r["backend"] == "podman" for r in records)
    assert all(Path(r["base"]).parent == store / "bases" for r in records), "bases land in the default store"

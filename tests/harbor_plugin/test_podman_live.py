"""A real Harbor trial on the podman backend: the gate on the fixture task.

``eval.gate --backend podman`` runs Harbor's own ``oracle`` and ``nop``
agents on ``PodmanEnvironment`` -- the image pinned and extracted through
podman, a shaped offline container, the ``tests/`` upload, the verifier's
reward read back -- and the oracle must score 1 and doing nothing 0. What is
left afterwards is checked too: Harbor deletes both environments, so no
container, rootfs subvolume or socket directory may remain (the extracted base
may, and stays for the next run). Also: Harbor's phase policy toggles the
container's egress; a retained stop leaves the container and its rootfs in
place; and on an image whose USER is not root, Harbor's root steps and an
overlay's RUN run as root while the agent's commands run as that user.

Skipped under the same conditions as the other podman live tests
(``mount_problem`` decides whether the store will do). Run the file alone.

    pytest tests/harbor_plugin/test_podman_live.py -v -m slow
"""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest

from tests.conftest import default_podman_bin, podman_store_in_use, sandbox_networks

pytest.importorskip("harbor")

from harbor.models.task.config import EnvironmentConfig, NetworkMode, NetworkPolicy
from harbor.models.trial.paths import TrialPaths
from sprout.eval.harbor.podman import PodmanEnvironment
from sprout.eval import gate
from sprout_sandbox.btrfs_store import BtrfsStore, mount_problem
from sprout_sandbox.podman_images import CONFIG_KEYS, image_info

REPO = Path(__file__).resolve().parents[2]
RUNTIME_BIN = REPO / "runtime" / "sprout-runtime"
STORE = Path(os.environ.get("SPROUT_PODMAN_STORE", "~/sprout-btrfs")).expanduser()
#: The podman the pool runs: $SPROUT_PODMAN_BIN (the one ops/setup_podman.sh installed), else PATH.
PODMAN = default_podman_bin()
TASKS = Path(__file__).resolve().parent / "agentenv_fixtures"
IMAGE = "docker.io/library/python:3.11-slim"

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(shutil.which(default_podman_bin()) is None,
                       reason="needs podman on PATH or in $SPROUT_PODMAN_BIN"),
    pytest.mark.skipif(shutil.which("btrfs") is None, reason="needs btrfs-progs on PATH"),
    pytest.mark.skipif(not RUNTIME_BIN.exists(),
                       reason="needs runtime/sprout-runtime "
                              "(cd runtime && go generate ./... && go build -o sprout-runtime .)"),
    pytest.mark.skipif(STORE.is_dir() and podman_store_in_use(STORE) is not None,
                       reason="the store is in use by a run (%s); the fixtures here remove every container "
                              "and snapshot that appears during a test" % podman_store_in_use(STORE)),
    pytest.mark.skipif(not STORE.is_dir() or mount_problem(STORE) is not None,
                       reason="needs a btrfs store with user_subvol_rm_allowed at SPROUT_PODMAN_STORE"),
]


def _containers() -> set[str]:
    return set(subprocess.run([PODMAN, "ps", "-a", "--format", "{{.Names}}"],
                              capture_output=True, text=True, check=True).stdout.split())


def _entries(kind: str) -> set[str]:
    directory = STORE / kind
    return {p.name for p in directory.iterdir()} if directory.is_dir() else set()


def test_the_gate_passes_on_podman_and_the_trials_leave_nothing_behind(tmp_path, capsys):
    before = {kind: _entries(kind) for kind in ("sandboxes", "snapshots", "sockets")}
    containers, networks = _containers(), sandbox_networks()
    code = gate.main(["--tasks-dir", str(TASKS), "--task", "shell-task", "--output", str(tmp_path / "gate"),
                      "--runtime-bin", str(RUNTIME_BIN), "--backend", "podman", "--store", str(STORE),
                      "--workers", "1"])
    out = capsys.readouterr().out
    leaked = _containers() - containers
    if leaked:
        subprocess.run([PODMAN, "rm", "-f", *leaked], capture_output=True)
    leaked_networks = sandbox_networks() - networks
    if leaked_networks:
        subprocess.run([PODMAN, "network", "rm", *leaked_networks], capture_output=True)
    assert code == 0, out
    records = {row["mode"]: row for row in map(json.loads, (tmp_path / "gate" / "gate.jsonl").read_text().splitlines())}
    assert records["oracle"]["reward"] == 1.0 and records["nop"]["reward"] == 0.0, records
    for mode in ("oracle", "nop"):
        trial = Path(records[mode]["trial"])
        receipts = sorted(trial.glob("*.podman.json"))
        assert receipts, "the environment wrote no receipt"
        for receipt in map(json.loads, (p.read_text() for p in receipts)):
            assert receipt["backend"] == "podman" and receipt["store"] == str(STORE.resolve())
            assert "@sha256:" in receipt["image"], "the image is pinned to the digest podman pulled"
            assert receipt["resources"] == {"cpu": 1, "memory_mb": 512}, "the task's shape reached the container"
            assert receipt["network"]["network_mode"] == "public"
    assert not leaked, "trials left containers behind: %s" % sorted(leaked)
    assert not leaked_networks, "trials left networks behind: %s" % sorted(leaked_networks)
    assert {kind: _entries(kind) - before[kind] for kind in before} == {
        "sandboxes": set(), "snapshots": set(), "sockets": set()}


def test_harbors_phase_policy_toggles_the_containers_egress(tmp_path):
    """The task's `[agent] network_mode = "no-network"` reaches the container
    through `set_network_policy` -> `SandboxSession.set_network` -> `podman
    network disconnect`: a connect attempt inside fails offline and succeeds
    once the public phase policy is applied, with the runtime answering
    throughout over its Unix socket."""
    before = _entries("sandboxes")
    env = PodmanEnvironment(
        environment_dir=TASKS / "shell-task" / "environment", environment_name="shell-task",
        session_id="phases__env", trial_paths=TrialPaths(trial_dir=tmp_path),
        task_env_config=EnvironmentConfig(docker_image="python:3.11-slim", cpus=1, memory_mb=512),
        runtime_bin=str(RUNTIME_BIN), store=str(STORE),
        network_policy=NetworkPolicy(network_mode=NetworkMode.NO_NETWORK),
        phase_network_policies=[NetworkPolicy(network_mode=NetworkMode.PUBLIC)])
    probe = ("python3 -c \"import socket; socket.create_connection(('1.1.1.1', 53), timeout=3)\" "
             "&& echo online || echo offline")

    async def phases():
        await env.start()
        try:
            offline = (await env.exec(probe, timeout_sec=30)).stdout.strip()
            await env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.PUBLIC))
            online = (await env.exec(probe, timeout_sec=30)).stdout.strip()
            await env.set_network_policy(NetworkPolicy(network_mode=NetworkMode.NO_NETWORK))
            again = (await env.exec(probe, timeout_sec=30)).stdout.strip()
            return offline, online, again
        finally:
            await env.stop(delete=True)

    assert asyncio.run(phases()) == ("offline", "online", "offline")
    assert _entries("sandboxes") == before, "stop left the rootfs behind"


def test_a_retained_stop_leaves_the_container_and_its_rootfs_running(tmp_path):
    """A sandbox whose final snapshot failed is the only copy of the rollout.
    Harbor still calls stop(delete=True); the environment writes the receipt
    and releases the sandbox from the pool before closing it -- closing a pool
    destroys every sandbox it still holds, which once took the retained one
    with it, receipt and all."""
    env = PodmanEnvironment(
        environment_dir=TASKS / "shell-task" / "environment", environment_name="shell-task",
        session_id="retained__env", trial_paths=TrialPaths(trial_dir=tmp_path),
        task_env_config=EnvironmentConfig(docker_image="python:3.11-slim", cpus=1, memory_mb=512),
        runtime_bin=str(RUNTIME_BIN), store=str(STORE))

    async def retained():
        await env.start()
        sid = env.session.sandbox_id
        await env.exec("echo the-only-copy > /tmp/result")
        env.retain_reason = "snapshot_pending"
        await env.stop(delete=True)
        return sid

    sid = asyncio.run(retained())
    try:
        assert env.session is None
        assert json.loads((tmp_path / "retained__env.retained-sandbox.json").read_text()) == {
            "sandbox_id": sid, "reason": "snapshot_pending"}
        assert sid in _containers(), "the container is still there"
        assert subprocess.run([PODMAN, "inspect", "--format", "{{.State.Status}}", sid],
                              capture_output=True, text=True).stdout.strip() == "running"
        assert (STORE / "sandboxes" / sid).is_dir() and (STORE / "sandboxes" / (sid + ".json")).is_file()
        assert (STORE / "sockets" / sid / "runtime.sock").exists()
        assert (STORE / "sandboxes" / sid / "tmp" / "result").read_text() == "the-only-copy\n", \
            "the rollout's result is on the rootfs, for the person who comes back for it"
    finally:
        subprocess.run([PODMAN, "rm", "-f", sid], capture_output=True)
        subprocess.run([PODMAN, "network", "rm", sid], capture_output=True)
        store = BtrfsStore(STORE)
        asyncio.run(store.discard_rootfs(store.sandboxes / sid))
        shutil.rmtree(STORE / "sockets" / sid, ignore_errors=True)
    assert sid not in _containers() and not (STORE / "sandboxes" / sid).exists()
    assert sid not in sandbox_networks()


USER_IMAGE = "localhost/sprout-test-harbor-user:latest"


class UnpinnedPodmanEnvironment(PodmanEnvironment):
    """The real environment on a locally built image, which has no registry
    digest to pin to; everything after the pinning is the real thing."""

    async def _prepare_image(self, force_build: bool) -> tuple[str, dict]:
        image = self.task_env_config.docker_image
        info = await image_info(image, podman_bin=self.podman_bin)
        return image, {key: (info.get("Config") or {}).get(key) for key in CONFIG_KEYS}


@pytest.fixture
def user_image(tmp_path: Path):
    subprocess.run([PODMAN, "build", "-q", "-t", USER_IMAGE, "-f", "-", str(tmp_path)],
                   input="FROM %s\nUSER nobody\nWORKDIR /tmp\n" % IMAGE, text=True, check=True, capture_output=True)
    yield USER_IMAGE
    store = BtrfsStore(STORE)
    digest = asyncio.run(image_info(USER_IMAGE))["Id"]
    asyncio.run(store.discard(store.bases / digest))
    subprocess.run([PODMAN, "rmi", "-f", USER_IMAGE], capture_output=True)


def test_a_non_root_image_runs_harbors_root_steps_as_root_and_the_agents_commands_as_its_user(tmp_path, user_image):
    """The image's USER is nobody. start() lays out /logs as root, the
    tests/Dockerfile overlay's RUN executes as root, and a command without an
    explicit user runs as nobody in the image's WORKDIR -- the microVM's
    semantics. With the runtime itself running as the image's user none of the
    root steps could succeed and the trial failed at `mkdir /logs`."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test.sh").write_text("#!/bin/bash\necho 1 > /logs/verifier/reward.txt\n")
    (tests / "Dockerfile").write_text("FROM %s\nCOPY test.sh /tests/test.sh\n"
                                      "RUN chmod +x /tests/test.sh && touch /usr/local/root-only && id -u > /tmp/run-as\n"
                                      % user_image)
    before = _entries("sandboxes")
    env = UnpinnedPodmanEnvironment(
        environment_dir=tests, environment_name="user-task", session_id="user__verifier__grade",
        trial_paths=TrialPaths(trial_dir=tmp_path),
        task_env_config=EnvironmentConfig(docker_image=user_image, cpus=1, memory_mb=512),
        runtime_bin=str(RUNTIME_BIN), store=str(STORE))
    assert env.overlay is not None and env.overlay.runs

    async def trial():
        await env.start()
        try:
            as_agent = await env.exec("id -u && pwd && cat /tmp/run-as && ls /tests/test.sh /usr/local/root-only")
            as_root = await env.exec("id -u", user="root")
            logs = await env.exec("stat -c '%a %U' /logs/verifier")
            return as_agent, as_root, logs
        finally:
            await env.stop(delete=True)

    as_agent, as_root, logs = asyncio.run(trial())
    assert env.image_user == "nobody" and env.image_workdir == "/tmp"
    assert (as_agent.return_code, as_agent.stdout) == (0, "65534\n/tmp\n0\n/tests/test.sh\n/usr/local/root-only\n"), \
        "the agent's command ran as nobody in /tmp; the overlay's RUN had run as root"
    assert (as_root.return_code, as_root.stdout) == (0, "0\n")
    assert logs.stdout.strip() == "777 root", "start() laid out /logs as root"
    assert _entries("sandboxes") == before, "stop left the rootfs behind"

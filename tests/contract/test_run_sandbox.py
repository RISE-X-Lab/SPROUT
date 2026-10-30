"""The sandbox that a ``RunSpec`` gets: its image, its resources, and the lease of a microVM.

A microVM lives until its lease ends, so its lease must outlive the run's ``timeout_s``. A podman
container has no lease: it lives until it is removed. Each run here uses the ``nop`` agent, which calls
no tool, on the fake sandbox backend of ``fake_sandbox.py``.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Iterator
from contextlib import contextmanager
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

import pytest

from sprout.contract.run_spec import RunSpec
from sprout.worker.jobs.rollout import execute
from sprout.worker.run.runner import Runner
from sprout.worker.sandbox.backends import build_pool


def nop_spec(tmp_path, **fields: Any) -> RunSpec:
    """A ``nop`` run from the template ``ready-template``, with ``fields``."""
    return RunSpec(
        prompt="Do nothing.",
        agent="nop",
        sandbox_image="ready-template",
        journal_path=tmp_path / "trajectory.jsonl",
        **fields,
    )


@contextmanager
def agentenv() -> Iterator[tuple[str, list[dict[str, Any]]]]:
    """The URL of a stub AgentENV server, and the body of each sandbox it is asked to create."""
    created: list[dict[str, Any]] = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format: str, *args: Any) -> None:
            pass

        def do_POST(self) -> None:
            created.append(json.loads(self.rfile.read(int(self.headers["Content-Length"]))))
            self._answer(200, {"sandboxID": "vm-1", "templateID": created[-1]["templateID"]})

        def do_DELETE(self) -> None:
            self._answer(204, None)

        def _answer(self, status: int, body: dict[str, Any] | None) -> None:
            data = json.dumps(body).encode() if body is not None else b""
            self.send_response(status)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield "http://127.0.0.1:%d" % server.server_port, created
    finally:
        server.shutdown()
        server.server_close()
        thread.join(5)


def test_the_sandbox_starts_from_the_image_of_the_spec_with_its_resources(tmp_path, sandboxes):
    Runner(out_dir=tmp_path).run(nop_spec(tmp_path, sandbox_resources={"cpu": 2, "memory_mb": 12288}))

    assert [(session.image, session.resources) for session in sandboxes.sessions] == [
        ("ready-template", {"cpu": 2, "memory_mb": 12288})
    ]


def test_a_run_of_another_slot_than_mini_swe_agent_fails_before_a_sandbox_exists(tmp_path, sandboxes):
    outcome = Runner(out_dir=tmp_path).run(nop_spec(tmp_path, slot="codex"))

    assert (outcome.status, outcome.error, sandboxes.sessions) == (
        "error",
        "ValueError: unknown slot 'codex'; the slots are mini-swe-agent and sprout-harness",
        [],
    )


@pytest.mark.parametrize("network, allow", [("public", True), ("no-network", False)])
def test_the_sandbox_of_a_harbor_task_starts_on_the_network_of_its_agent_phase(
    tmp_path, sandboxes, network, allow
):
    task_dir = tmp_path / "task"
    (task_dir / "tests").mkdir(parents=True)
    (task_dir / "task.toml").write_text(
        '[environment]\ndocker_image = "image"\n[agent]\nnetwork_mode = "%s"\n' % network
    )
    (task_dir / "instruction.md").write_text("Fix the bug.\n")
    (task_dir / "tests" / "test.sh").write_text("#!/bin/bash\n")

    Runner(out_dir=tmp_path).run(nop_spec(tmp_path, harbor_task=str(task_dir)))

    assert sandboxes.sessions[0].backend == {"podman": {"allow_internet": allow}}


def test_a_kept_sandbox_outlives_its_run(tmp_path, sandboxes):
    """A grade can run in it, or a person can inspect it."""
    Runner(out_dir=tmp_path).run(nop_spec(tmp_path, keep_sandbox=True))

    assert [session.destroyed for session in sandboxes.sessions] == [False]


def test_the_lease_of_a_kept_microvm_covers_the_time_that_a_queued_run_keeps_it(tmp_path, sandboxes):
    """1800 s of budget, the 1200 s of ``KEEP_SANDBOX_S`` (ADR-0004), and the margin of 600 s."""
    request = {
        "kind": "rollout",
        "job_id": "job-1",
        "attempt_id": "attempt-1",
        "profile_config": {},
        "effective_spec": {
            "prompt": "Do nothing.",
            "agent": "nop",
            "sandbox_image": "ready-template",
            "timeout_s": 1800,
            "keep_sandbox": True,
            "backend": {"backend": "microvm", "microvm": {"server_url": "http://127.0.0.1:3000"}},
        },
    }
    (tmp_path / "attempt").mkdir()

    execute(request, tmp_path / "attempt")

    assert sandboxes.sessions[0].backend["microvm"]["sandbox_ttl"] == 3600


@pytest.mark.parametrize(
    "budget, own, lease",
    [
        (1800, None, 2400),
        (1800, 600, 2400),
        (1800, 7200, 7200),
        (3600, None, 4200),
        (10800, 600, 11400),
        (1.2, None, 602),
        (125.5, None, 726),
    ],
)
def test_a_microvm_run_gets_a_lease_of_its_budget_plus_600_s_unless_its_own_lease_is_longer(
    tmp_path, sandboxes, budget, own, lease
):
    """A budget with a fraction of a second rounds up."""
    microvm = {
        "server_url": "http://127.0.0.1:3000",
        "allow_internet": False,
        **({"sandbox_ttl": own} if own else {}),
    }

    Runner(out_dir=tmp_path).run(
        nop_spec(tmp_path, timeout_s=budget, backend={"backend": "microvm", "microvm": microvm})
    )

    assert sandboxes.sessions[0].backend == {
        "backend": "microvm",
        "microvm": {"server_url": "http://127.0.0.1:3000", "allow_internet": False, "sandbox_ttl": lease},
    }


def test_the_lease_of_a_run_leaves_its_spec_as_it_was(tmp_path, sandboxes):
    backend = {"backend": "microvm", "microvm": {"server_url": "http://127.0.0.1:3000"}}
    spec = nop_spec(tmp_path, timeout_s=1800, backend=deepcopy(backend))

    Runner(out_dir=tmp_path).run(spec)

    assert (spec.timeout_s, spec.backend) == (1800, backend)


@pytest.mark.parametrize("backend", [{}, {"backend": "podman"}], ids=["default", "named"])
def test_a_podman_run_gets_no_lease(tmp_path, sandboxes, backend):
    Runner(out_dir=tmp_path).run(nop_spec(tmp_path, timeout_s=1800, backend=backend))

    assert sandboxes.sessions[0].backend == backend


@pytest.mark.parametrize("budget", [None, 0, -1, True, float("inf"), float("nan")])
def test_a_microvm_run_without_a_finite_positive_budget_fails_before_a_sandbox_exists(
    tmp_path, sandboxes, budget
):
    spec = nop_spec(
        tmp_path, timeout_s=budget, backend={"backend": "microvm", "microvm": {"server_url": "http://x"}}
    )

    outcome = Runner(out_dir=tmp_path).run(spec)

    assert (outcome.status, outcome.error, sandboxes.sessions) == (
        "error",
        "BackendError: VM lifetime requires a finite positive operation budget",
        [],
    )


def test_the_lease_of_a_run_is_the_timeout_of_its_vm_at_agentenv(tmp_path, sandboxes):
    """AgentENV pauses the VM at the end of its lease, and resumes it at the next call."""
    with agentenv() as (url, created):
        Runner(out_dir=tmp_path).run(
            nop_spec(tmp_path, timeout_s=1800, backend={"backend": "microvm", "microvm": {"server_url": url}})
        )
        pool = build_pool(sandboxes.sessions[0].backend)

        async def spawn() -> None:
            await pool.spawn("ready-template")
            await pool.close()

        asyncio.run(spawn())

    assert [(body["templateID"], body["timeout"], body["autoPause"]) for body in created] == [
        ("ready-template", 2400, True)
    ]

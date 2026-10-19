"""``python -m sprout.worker run``: the run spec that the flags make, and what is refused before a run.

The ``Runner`` of the CLI is replaced by ``RecordingRunner``, which starts no sandbox. It records each
spec that it gets, and returns the outcome that the test gives it.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pytest

from sprout.contract.run_spec import RunSpec
from sprout.worker import cli
from sprout.worker.run.runner import RunOutcome

RUN = [
    "run",
    "--sandbox-image",
    "docker.io/library/python:3.11-slim",
    "--cwd",
    "/path/to/work",
    "fix the failing test",
]
COMPLETED = RunOutcome("run-1", Path("/path/to/runs/run-1.jsonl"), "completed")


@pytest.fixture
def runner(monkeypatch) -> Callable[[RunOutcome], list[RunSpec]]:
    """Replace the CLI's ``Runner`` with one that returns the given outcome. Give the specs that it got."""

    def install(outcome: RunOutcome) -> list[RunSpec]:
        specs: list[RunSpec] = []

        class RecordingRunner:
            def __init__(self, **options: object) -> None:
                pass

            def run(self, spec: RunSpec) -> RunOutcome:
                specs.append(spec)
                return outcome

        monkeypatch.setattr(cli, "Runner", RecordingRunner)
        return specs

    return install


def test_the_flags_of_a_run_reach_its_run_spec(runner):
    specs = runner(COMPLETED)
    flags = [
        "--model", "my-model", "--timeout", "900", "--agent-id", "agent-7", "--run-id", "run-7",
        "--journal", "/path/to/runs/run-7.jsonl", "--keep-sandbox", "--extra", '{"max_turns": 5}',
        "--runtime-bin", "/path/to/sprout-runtime", "--store", "/path/to/store",
    ]  # fmt: skip

    code = cli.main([*RUN, *flags])

    assert (code, specs) == (
        0,
        [
            RunSpec(
                prompt="fix the failing test",
                cwd="/path/to/work",
                model="my-model",
                timeout_s=900.0,
                agent_id="agent-7",
                run_id="run-7",
                journal_path="/path/to/runs/run-7.jsonl",
                sandbox_image="docker.io/library/python:3.11-slim",
                keep_sandbox=True,
                backend={
                    "backend": "podman",
                    "podman": {"store": "/path/to/store", "runtime_bin": "/path/to/sprout-runtime"},
                },
                runtime_bin="/path/to/sprout-runtime",
                extra={"max_turns": 5},
            )
        ],
    )


def test_podman_is_the_default_backend_and_its_store_defaults_to_sprout_podman_store(runner, monkeypatch):
    specs = runner(COMPLETED)
    monkeypatch.setenv("SPROUT_PODMAN_STORE", "/path/to/store")

    cli.main([*RUN, "--runtime-bin", "/path/to/sprout-runtime"])

    assert [spec.backend for spec in specs] == [
        {"backend": "podman", "podman": {"store": "/path/to/store", "runtime_bin": "/path/to/sprout-runtime"}}
    ]


@pytest.mark.parametrize(
    "flags, section",
    [
        (
            ["--runtime-bin", "/path/to/sprout-runtime"],
            {"from_image": True, "runtime_bin": "/path/to/sprout-runtime"},
        ),
        ([], {"from_image": True}),
    ],
    ids=["with a runtime binary", "without one"],
)
def test_a_microvm_run_starts_from_the_image_and_builds_its_template_with_the_runtime_binary_given(
    runner, flags, section
):
    """Without the binary, the run cold-starts the plain image, which answers each tool call with a 502."""
    specs = runner(COMPLETED)

    cli.main([*RUN, "--backend", "microvm", *flags])

    assert [spec.backend for spec in specs] == [{"backend": "microvm", "microvm": section}]


@pytest.mark.parametrize(
    "flags, refusal",
    [
        (["--store", "/path/to/store"], "^--backend podman needs --runtime-bin"),
        (
            [
                "--runtime-bin",
                "/path/to/sprout-runtime",
                "--store",
                "/path/to/store",
                "--final-memory-snapshot",
            ],
            "^--final-memory-snapshot needs a backend with memory snapshots; podman snapshots are disk-only$",
        ),
        (
            ["--runtime-bin", "/path/to/sprout-runtime", "--journal", "/tmp/path/to/run.jsonl"],
            r"^refusing: /tmp/path/to/run\.jsonl is under /tmp, .* \(pass --volatile-ok to override\)$",
        ),
    ],
    ids=["podman without the runtime binary", "podman with a memory snapshot", "journal under /tmp"],
)
def test_a_run_that_would_not_do_what_its_flags_say_is_refused_before_it_starts(runner, flags, refusal):
    """Without a refusal, a podman memory snapshot is skipped in silence, and a journal is lost at reboot."""
    specs = runner(COMPLETED)

    with pytest.raises(SystemExit, match=refusal):
        cli.main([*RUN, *flags])

    assert specs == []


def test_volatile_ok_lets_a_run_journal_under_tmp(runner):
    specs = runner(COMPLETED)
    flags = [
        "--runtime-bin",
        "/path/to/sprout-runtime",
        "--journal",
        "/tmp/path/to/run.jsonl",
        "--volatile-ok",
    ]

    cli.main([*RUN, *flags])

    assert [spec.journal_path for spec in specs] == ["/tmp/path/to/run.jsonl"]


def test_a_run_that_does_not_complete_exits_1_and_reports_its_error_on_stderr(runner, capsys):
    runner(RunOutcome("run-1", Path("/path/to/runs/run-1.jsonl"), "error", error="the sandbox did not start"))

    code = cli.main([*RUN, "--runtime-bin", "/path/to/sprout-runtime", "--store", "/path/to/store"])

    assert (code, capsys.readouterr().err) == (1, "error       the sandbox did not start\n")

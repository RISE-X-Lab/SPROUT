"""A Harbor task's Dockerfile as an overlay on its FROM image, and the replay of it in a sandbox.

A separate verifier's sandbox, and an evaluation from a task's ``environment/Dockerfile``, start
from the FROM image and replay the COPY and RUN steps. An instruction that a replay cannot
reproduce exactly must be refused, so that a replayed environment never differs from the built one.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout.worker.jobs.harbor.overlay import ImageOverlay, overlay_for, parse_overlay, replay
from sprout.worker.sandbox.commands import shell_outcome
from tests.worker.jobs.sandboxes import DirectorySandboxes


def dockerfile(directory: Path, text: str) -> Path:
    """``directory/Dockerfile`` with ``text``, beside a file ``check.sh`` and a directory ``data``."""
    (directory / "data").mkdir(parents=True)
    (directory / "data" / "case.json").write_text("{}\n")
    (directory / "check.sh").write_text("exit 0\n")
    (directory / "Dockerfile").write_text(text)
    return directory / "Dockerfile"


def test_an_overlay_holds_the_from_image_each_copy_and_each_run_in_order(tmp_path):
    """Comments and blank lines are not instructions, and a backslash continues one onto the next line."""
    path = dockerfile(
        tmp_path,
        "# the verifier\nFROM base:1 AS tests\n\nCOPY check.sh /tests/check.sh\n"
        "RUN apt-get update && \\\n    apt-get install -y jq\n"
        "copy data /tests/data\nRUN chmod +x /tests/check.sh\n",
    )

    overlay = parse_overlay(path)

    assert overlay == ImageOverlay(
        base="base:1",
        copies=((tmp_path / "check.sh", "/tests/check.sh"), (tmp_path / "data", "/tests/data")),
        runs=("apt-get update &&  apt-get install -y jq", "chmod +x /tests/check.sh"),
    )


@pytest.mark.parametrize(
    "text, message",
    [
        ("FROM base\nENV A=1\n", "ENV is not replayed in a sandbox"),
        ("FROM base\nWORKDIR /tests\n", "WORKDIR is not replayed in a sandbox"),
        ("FROM base\nUSER tester\n", "USER is not replayed in a sandbox"),
        ("FROM base\nFROM other\n", "multi-stage Dockerfiles are not replayed"),
        ("FROM base\nCOPY --link check.sh\n", "only `COPY <src> <dst>` is replayed"),
        ("FROM base\nCOPY check.sh data /tests/\n", "only `COPY <src> <dst>` is replayed"),
        ("FROM base\nCOPY absent.sh /tests/\n", "COPY source absent.sh is missing"),
        ("RUN true\n", "no FROM"),
    ],
)
def test_an_instruction_that_a_sandbox_cannot_replay_is_refused(tmp_path, text, message):
    path = dockerfile(tmp_path, text)

    with pytest.raises(ValueError, match=message):
        parse_overlay(path)


PARSABLE = "FROM task-image\nRUN true\n"
UNPARSABLE = "FROM task-image\nENV A=1\n"
REPLAYED = ImageOverlay(base="task-image", copies=(), runs=("true",))


def environment(directory: Path, text: str | None) -> Path:
    """A task's ``environment/`` directory, with a Dockerfile of ``text`` unless it is None."""
    directory.mkdir()
    if text is not None:
        (directory / "Dockerfile").write_text(text)
    return directory


@pytest.mark.parametrize(
    "text, docker_image, overlay",
    [
        pytest.param(None, "task-image", None, id="no-dockerfile"),
        pytest.param(UNPARSABLE, "another-image", None, id="dockerfile-that-built-the-image"),
        pytest.param(PARSABLE, "task-image", REPLAYED, id="from-the-image"),
        pytest.param(PARSABLE, None, REPLAYED, id="no-image"),
        pytest.param(UNPARSABLE, None, None, id="no-image-and-unparsable"),
    ],
)
def test_a_task_environment_has_an_overlay_only_when_a_sandbox_can_replay_its_dockerfile(
    tmp_path, text, docker_image, overlay
):
    """A Dockerfile FROM another image built the task's image, so nothing of it is left to replay."""
    directory = environment(tmp_path / "environment", text)

    assert overlay_for(directory, docker_image) == overlay


def test_a_dockerfile_from_the_task_image_that_a_sandbox_cannot_replay_is_refused(tmp_path):
    directory = environment(tmp_path / "environment", UNPARSABLE)

    with pytest.raises(ValueError, match="ENV is not replayed in a sandbox"):
        overlay_for(directory, "task-image")


def test_a_replay_copies_the_sources_and_runs_each_step_in_the_directory_with_the_env(tmp_path):
    (tmp_path / "image" / "tmp").mkdir(parents=True)
    (tmp_path / "image" / "app").mkdir()
    sandboxes = DirectorySandboxes(tmp_path / "sandboxes", images={"base": tmp_path / "image"})
    session = sandboxes(backend={})
    session.create("base")
    path = dockerfile(
        tmp_path / "task",
        "FROM base\nCOPY check.sh /tests/check.sh\nCOPY data /tests/data\n"
        'RUN echo "$MODE in $(pwd)" > /tests/run.txt\n',
    )

    replay(session, parse_overlay(path), cwd="/app", env={"MODE": "verify"}, timeout_s=60)

    listing = shell_outcome(session, "cd /tests && find . -type f | sort && cat run.txt")[0]
    assert listing == "./check.sh\n./data/case.json\n./run.txt\nverify in /app\n"


def test_a_run_step_that_exits_non_zero_stops_the_replay_with_its_output(tmp_path):
    (tmp_path / "image" / "tmp").mkdir(parents=True)
    sandboxes = DirectorySandboxes(tmp_path / "sandboxes", images={"base": tmp_path / "image"})
    session = sandboxes(backend={})
    session.create("base")
    overlay = ImageOverlay(
        base="base",
        copies=(),
        runs=("echo first > /tmp/first", "echo bad >&2; exit 4", "echo never > /tmp/never"),
    )

    with pytest.raises(RuntimeError, match=r"^overlay RUN failed \(echo bad >&2; exit 4\): bad\n$"):
        replay(session, overlay, cwd="/", env={}, timeout_s=60)

    assert shell_outcome(session, "ls /tmp")[0] == "first\n"

"""The ``harbor`` grade: a Harbor task's verifier, run by the worker the way Harbor runs it.

The sandboxes are the directory sandboxes of ``sandboxes.py``. ``snap-1`` is the agent's final
snapshot: ``/app/answer.txt`` says ready, and ``/app/notes/n.txt`` holds a note. ``pristine-image``
is the FROM image of a separate verifier's ``tests/Dockerfile``. A task's ``test.sh`` names the
sandbox by ``$SANDBOX_ROOT``, because the fake does not rewrite the paths in a file.
``tests/runstore/test_harbor_grade_live.py`` grades on podman beside Harbor's own verifier.
"""

from __future__ import annotations

import os
import re
import shlex
from pathlib import Path

import pytest

from sprout.worker.jobs.grade import grade
from tests.worker.jobs.sandboxes import DirectorySandboxes

PODMAN = {"backend": "podman", "podman": {"store": "/path/to/store"}}
#: SWE-bench Pro's shape: the agent offline, the verifier on the environment's public baseline.
SHARED = """\
[agent]
network_mode = "no-network"
[verifier]
timeout_sec = 30
[environment]
docker_image = "python:3.11-slim"
cpus = 1
memory_mb = 512
"""
SEPARATE = """\
artifacts = ["/app/notes", "/logs/artifacts/answer.txt", "/app/"]
[agent]
network_mode = "no-network"
[verifier]
environment_mode = "separate"
network_mode = "public"
timeout_sec = 30
[verifier.environment]
cpus = 2
memory_mb = 4096
[[verifier.collect]]
command = "mkdir -p /logs/artifacts && cp /app/answer.txt /logs/artifacts/answer.txt"
[environment]
docker_image = "task-image"
cpus = 1
memory_mb = 512
"""
#: A separate verifier's image: the FROM image, then the COPY and RUN steps that the grade replays.
TESTS_DOCKERFILE = "FROM pristine-image\nCOPY test.sh /tests/test.sh\nRUN touch /tests/replayed\n"
#: Rewards 1 when /app/answer.txt says ready.
ANSWER_TEST = """#!/bin/bash
if [ "$(cat "$SANDBOX_ROOT/app/answer.txt" 2>/dev/null)" = ready ]; then echo 1; else echo 0; fi \\
    > "$SANDBOX_ROOT/logs/verifier/reward.txt"
"""
#: Rewards 1 in a sandbox that has the artifacts and the replayed RUN step, and not the agent's /app.
ARTIFACTS_TEST = """#!/bin/bash
root="$SANDBOX_ROOT"
if [ "$(cat "$root/logs/artifacts/answer.txt")" = ready ] && [ -f "$root/app/notes/n.txt" ] \\
    && [ ! -e "$root/app/answer.txt" ] && [ -f "$root/tests/replayed" ]; then echo 1; else echo 0; fi \\
    > "$root/logs/verifier/reward.txt"
"""


def image(root: Path, files: dict[str, str]) -> Path:
    """A directory with the files of an image, and its ``/tmp``."""
    (root / "tmp").mkdir(parents=True)
    for name, text in files.items():
        (root / name).parent.mkdir(parents=True, exist_ok=True)
        (root / name).write_text(text)
    return root


def harbor_task(root: Path, *, toml: str, test_sh: str, tests_dockerfile: str | None = None) -> Path:
    (root / "tests").mkdir(parents=True)
    (root / "environment").mkdir()
    (root / "task.toml").write_text(toml)
    (root / "instruction.md").write_text("Write ready to /app/answer.txt.\n")
    (root / "tests" / "test.sh").write_text(test_sh)
    if tests_dockerfile is not None:
        (root / "tests" / "Dockerfile").write_text(tests_dockerfile)
    return root


def shapes(sandboxes: DirectorySandboxes) -> list[tuple[str | None, dict | None, bool | None]]:
    """The image, the resources and the egress of each sandbox that a session started, in order."""
    return [(session.image, session.resources, session.allow_internet) for session in sandboxes.sessions]


def files_under(directory: Path) -> list[str]:
    """The path of each file under ``directory``, relative to it, sorted."""
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*") if path.is_file())


def writes(files: dict[str, str]) -> str:
    """A test.sh that writes each of ``files`` into /logs/verifier, byte for byte."""
    lines = [
        'printf %%s %s > "$SANDBOX_ROOT/logs/verifier/%s"' % (shlex.quote(text), name)
        for name, text in files.items()
    ]
    return "#!/bin/bash\n" + "\n".join(lines) + "\n"


@pytest.fixture
def sandboxes(tmp_path):
    snapshot = image(
        tmp_path / "images" / "snap-1", {"app/answer.txt": "ready\n", "app/notes/n.txt": "note\n"}
    )
    pristine = image(tmp_path / "images" / "pristine", {"app/.keep": ""})
    return DirectorySandboxes(
        tmp_path / "sandboxes",
        images={"snap-1": snapshot, "pristine-image": pristine},
        configs={"pristine-image": {"WorkingDir": "/app"}},
    )


@pytest.fixture
def attempt(tmp_path):
    (tmp_path / "attempt").mkdir()
    return tmp_path / "attempt"


def harbor_grade(task_dir: Path, attempt: Path, sandboxes: DirectorySandboxes, **changes: object) -> dict:
    """The grade of ``task_dir`` from the snapshot ``snap-1``, on podman unless ``changes`` say otherwise."""
    body = {
        "benchmark": "harbor",
        "task_dir": str(task_dir),
        "snapshot_id": "snap-1",
        "backend": PODMAN,
        **changes,
    }
    return grade(body, attempt, session_factory=sandboxes)


def test_a_shared_verifier_grades_the_snapshot_in_one_sandbox_and_records_its_run(
    tmp_path, sandboxes, attempt
):
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh=ANSWER_TEST)

    result = harbor_grade(task_dir, attempt, sandboxes)

    assert result == {
        "status": "completed",
        "reward": 1.0,
        "reason": None,
        "verifier": {
            "in_place": False,
            "collect": [],
            "artifacts": [],
            "shared": True,
            "exit_code": 0,
            "timed_out": False,
            "rewards": {"reward": 1.0},
            "logs": str(attempt / "verifier"),
        },
    }
    assert sandboxes.lifecycle == [("create", "snap-1"), ("destroy", "sandbox-1")]
    assert (attempt / "verifier" / "reward.txt").read_text() == "1\n"


@pytest.mark.parametrize(
    "toml, tests_dockerfile, sandbox_shapes",
    [
        pytest.param(SHARED, None, [("snap-1", {"cpu": 1, "memory_mb": 512}, True)], id="shared"),
        pytest.param(
            SEPARATE,
            TESTS_DOCKERFILE,
            [
                ("snap-1", {"cpu": 1, "memory_mb": 512}, False),
                ("pristine-image", {"cpu": 2, "memory_mb": 4096}, True),
            ],
            id="separate",
        ),
    ],
)
def test_each_sandbox_has_the_shape_and_the_network_of_the_phase_that_runs_in_it(
    tmp_path, sandboxes, attempt, toml, tests_dockerfile, sandbox_shapes
):
    """The task's network policies decide egress. The sandbox that runs the verifier gets the verifier's."""
    task_dir = harbor_task(
        tmp_path / "task", toml=toml, test_sh=ANSWER_TEST, tests_dockerfile=tests_dockerfile
    )

    harbor_grade(task_dir, attempt, sandboxes)

    assert shapes(sandboxes) == sandbox_shapes


@pytest.mark.parametrize(
    "files, reward, reason, rewards",
    [
        ({"reward.txt": "1"}, 1.0, None, {"reward": 1.0}),
        ({"reward.txt": "0.25\n"}, 0.25, None, {"reward": 0.25}),
        ({"reward.json": '{"reward": 0.5, "tests": 2}'}, 0.5, None, {"reward": 0.5, "tests": 2}),
        ({"reward.json": '{"reward": 1}', "reward.txt": "0"}, 1.0, None, {"reward": 1}),
        ({}, 0.0, "reward_missing", None),
        ({"reward.txt": ""}, 0.0, "reward_unreadable", None),
        ({"reward.txt": "passed"}, 0.0, "reward_unreadable", None),
        ({"reward.txt": "nan"}, 0.0, "reward_unreadable", None),
        ({"reward.json": '{"reward": NaN}'}, 0.0, "reward_unreadable", None),
        ({"reward.json": '{"reward": Infinity}'}, 0.0, "reward_unreadable", None),
        ({"reward.json": '{"reward": 1, "loss": NaN}'}, 0.0, "reward_unreadable", None),
        ({"reward.json": '{"reward": true}'}, 0.0, "reward_unreadable", None),
        ({"reward.json": '{"score": 1}'}, 0.0, "reward_unreadable", None),
        ({"reward.json": "[1]"}, 0.0, "reward_unreadable", None),
        ({"reward.json": "not json", "reward.txt": "1"}, 0.0, "reward_unreadable", None),
    ],
)
def test_the_reward_is_read_the_way_harbor_reads_it(
    tmp_path, sandboxes, attempt, files, reward, reason, rewards
):
    """reward.json wins over reward.txt. A reward that is not one finite number scores 0 with the reason.

    A job result is JSON, which holds no NaN, so a reward.json with a NaN anywhere is unreadable."""
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh=writes(files))

    result = harbor_grade(task_dir, attempt, sandboxes)

    assert (result["reward"], result["reason"], result["verifier"]["rewards"]) == (reward, reason, rewards)


def test_a_verifier_that_outruns_its_timeout_scores_0(tmp_path, sandboxes, attempt):
    toml = SHARED.replace("timeout_sec = 30", "timeout_sec = 1")
    task_dir = harbor_task(tmp_path / "task", toml=toml, test_sh=writes({"reward.txt": "1"}) + "sleep 30\n")

    result = harbor_grade(task_dir, attempt, sandboxes)

    assert (result["reward"], result["reason"]) == (0.0, "verifier_timed_out")
    assert (result["verifier"]["exit_code"], result["verifier"]["timed_out"]) == (137, True)


def test_a_verifier_that_timed_out_scores_0_also_when_its_logs_do_not_move(tmp_path, sandboxes, attempt):
    """After a timeout the logs are only for a reader."""
    toml = SHARED.replace("timeout_sec = 30", "timeout_sec = 1")
    task_dir = harbor_task(tmp_path / "task", toml=toml, test_sh="#!/bin/bash\nsleep 30\n")
    sandboxes.refuse("download", "sprout-harbor-*.tar.gz")

    result = harbor_grade(task_dir, attempt, sandboxes)

    assert (result["reward"], result["reason"]) == (0.0, "verifier_timed_out")


def test_verifier_logs_that_do_not_move_fail_the_grade(tmp_path, sandboxes, attempt):
    """Without a timeout the reward is in the logs, so a grade without them is no grade, and not a 0."""
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh=ANSWER_TEST)
    sandboxes.refuse("download", "sprout-harbor-*.tar.gz")

    with pytest.raises(RuntimeError, match="^could not download /tmp/sprout-harbor-.*: PodmanError: refused"):
        harbor_grade(task_dir, attempt, sandboxes)

    assert sandboxes.live() == []


def test_a_reward_file_that_the_snapshot_holds_does_not_count(tmp_path, sandboxes, attempt):
    """The agent can write /logs/verifier. The grade empties it before test.sh runs."""
    (sandboxes.images["snap-1"] / "logs" / "verifier").mkdir(parents=True)
    (sandboxes.images["snap-1"] / "logs" / "verifier" / "reward.txt").write_text("1")
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh="#!/bin/bash\ntrue\n")

    result = harbor_grade(task_dir, attempt, sandboxes)

    assert (result["reward"], result["reason"]) == (0.0, "reward_missing")


def test_a_separate_verifier_grades_the_artifacts_in_its_own_sandbox_from_tests_dockerfile(
    tmp_path, sandboxes, attempt
):
    """Harbor's separate mode: collect in the agent's sandbox, then verify with the artifacts alone."""
    task_dir = harbor_task(
        tmp_path / "task", toml=SEPARATE, test_sh=ARTIFACTS_TEST, tests_dockerfile=TESTS_DOCKERFILE
    )

    result = harbor_grade(task_dir, attempt, sandboxes)

    assert result == {
        "status": "completed",
        "reward": 1.0,
        "reason": None,
        "verifier": {
            "in_place": False,
            "collect": [
                {
                    "command": "mkdir -p /logs/artifacts && cp /app/answer.txt /logs/artifacts/answer.txt",
                    "exit_code": 0,
                    "timed_out": False,
                }
            ],
            "artifacts": ["/logs/artifacts", "/app/notes"],
            "shared": False,
            "exit_code": 0,
            "timed_out": False,
            "rewards": {"reward": 1.0},
            "logs": str(attempt / "verifier"),
        },
    }
    assert sandboxes.lifecycle == [
        ("create", "snap-1"),
        ("destroy", "sandbox-1"),
        ("create", "pristine-image"),
        ("destroy", "sandbox-2"),
    ]


def test_the_attempt_directory_keeps_the_collected_artifacts_and_the_verifier_logs(
    tmp_path, sandboxes, attempt
):
    task_dir = harbor_task(
        tmp_path / "task", toml=SEPARATE, test_sh=ARTIFACTS_TEST, tests_dockerfile=TESTS_DOCKERFILE
    )

    harbor_grade(task_dir, attempt, sandboxes)

    assert files_under(attempt) == [
        "artifacts/app/notes/n.txt",
        "artifacts/logs/artifacts/answer.txt",
        "verifier/reward.txt",
        "verifier/test-stdout.txt",
    ]


def test_a_collect_command_that_fails_and_an_artifact_that_is_missing_do_not_fail_the_grade(
    tmp_path, sandboxes, attempt
):
    """Collection is best effort, as in Harbor. The verdict is the verifier's."""
    toml = SEPARATE.replace('"/app/notes"', '"/app/absent"').replace("mkdir -p", "exit 3; mkdir -p")
    task_dir = harbor_task(
        tmp_path / "task", toml=toml, test_sh=ANSWER_TEST, tests_dockerfile=TESTS_DOCKERFILE
    )

    result = harbor_grade(task_dir, attempt, sandboxes)

    assert (result["reward"], result["reason"]) == (0.0, None)
    assert result["verifier"]["collect"] == [
        {
            "command": "exit 3; mkdir -p /logs/artifacts && cp /app/answer.txt /logs/artifacts/answer.txt",
            "exit_code": 3,
            "timed_out": False,
        }
    ]
    assert result["verifier"]["artifacts"] == []


@pytest.mark.parametrize(
    "toml, tests_dockerfile, absent, message",
    [
        pytest.param(
            SHARED,
            None,
            "snap-1",
            "Could not restore the agent's snapshot: PodmanError: image snap-1 not known",
            id="agent",
        ),
        pytest.param(
            SEPARATE,
            TESTS_DOCKERFILE,
            "pristine-image",
            "Could not start the verifier sandbox from pristine-image: "
            "PodmanError: image pristine-image not known",
            id="verifier",
        ),
    ],
)
def test_a_sandbox_that_does_not_start_fails_the_grade_with_the_reason(
    tmp_path, sandboxes, attempt, toml, tests_dockerfile, absent, message
):
    del sandboxes.images[absent]
    task_dir = harbor_task(
        tmp_path / "task", toml=toml, test_sh=ANSWER_TEST, tests_dockerfile=tests_dockerfile
    )

    with pytest.raises(RuntimeError, match="^%s$" % re.escape(message)):
        harbor_grade(task_dir, attempt, sandboxes)

    assert sandboxes.live() == []


def test_a_replayed_run_step_that_fails_fails_the_grade_and_removes_the_verifier_sandbox(
    tmp_path, sandboxes, attempt
):
    dockerfile = "FROM pristine-image\nRUN echo broken >&2; exit 2\n"
    task_dir = harbor_task(tmp_path / "task", toml=SEPARATE, test_sh=ANSWER_TEST, tests_dockerfile=dockerfile)

    with pytest.raises(RuntimeError, match=r"^overlay RUN failed \(echo broken >&2; exit 2\): broken\n$"):
        harbor_grade(task_dir, attempt, sandboxes)

    assert sandboxes.live() == []


#: ``runuser`` and ``getent`` as a test that is not root sees them: every user exists, and
#: ``runuser -u NAME -- COMMAND`` runs COMMAND with ``RUNUSER=NAME``.
FAKE_RUNUSER = """#!/bin/sh
while [ "$1" != "--" ]; do
    [ "$1" = "-u" ] && user="$2"
    shift
done
shift
RUNUSER="$user" exec "$@"
"""
FAKE_GETENT = """#!/bin/sh
echo "$2:x:1000:1000::/home/$2:/bin/sh"
"""
#: Records the user that runs it and the FOO that it sees, and rewards 1.
IDENTITY_TEST = """#!/bin/bash
printf '%s|%s' "$RUNUSER" "$FOO" > "$SANDBOX_ROOT/logs/verifier/identity.txt"
echo 1 > "$SANDBOX_ROOT/logs/verifier/reward.txt"
"""


@pytest.fixture
def image_users(tmp_path, monkeypatch):
    """The fake runuser and getent, first on PATH."""
    tools = tmp_path / "user-tools"
    tools.mkdir()
    (tools / "runuser").write_text(FAKE_RUNUSER)
    (tools / "getent").write_text(FAKE_GETENT)
    (tools / "runuser").chmod(0o755)
    (tools / "getent").chmod(0o755)
    monkeypatch.setenv("PATH", "%s%s%s" % (tools, os.pathsep, os.environ["PATH"]))


def microvm(**options: bool) -> dict:
    """A microVM backend, which reads the config of the image that the grade names."""
    return {"backend": "microvm", "microvm": {"server_url": "http://127.0.0.1:1", **options}}


@pytest.mark.parametrize(
    "options, identity",
    [
        ({"image_user": True, "image_env": True}, "app|from the task image"),
        ({"image_user": True}, "app|"),
        ({"image_env": True}, "|"),
    ],
)
def test_a_shared_verifier_runs_test_sh_as_the_user_of_the_task_image_when_the_backend_asks(
    tmp_path, sandboxes, attempt, image_users, options, identity
):
    """As Harbor runs it. Without ``image_user``, test.sh runs as root, the runtime's user."""
    sandboxes.configs["python:3.11-slim"] = {
        "User": "app",
        "Env": ["FOO=from the task image"],
        "WorkingDir": "/app",
    }
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh=IDENTITY_TEST)

    result = harbor_grade(task_dir, attempt, sandboxes, backend=microvm(**options))

    assert (result["reward"], (attempt / "verifier" / "identity.txt").read_text()) == (1.0, identity)


def test_a_separate_verifier_runs_test_sh_as_the_user_of_its_own_image(
    tmp_path, sandboxes, attempt, image_users
):
    """The verifier's sandbox starts from the FROM image of tests/Dockerfile. Its USER runs test.sh."""
    sandboxes.configs["task-image"] = {"User": "agent", "Env": ["FOO=agent"], "WorkingDir": "/app"}
    sandboxes.configs["pristine-image"] = {"User": "checker", "Env": ["FOO=verifier"], "WorkingDir": "/app"}
    dockerfile = "FROM pristine-image\nCOPY test.sh /tests/test.sh\n"
    task_dir = harbor_task(
        tmp_path / "task", toml=SEPARATE, test_sh=IDENTITY_TEST, tests_dockerfile=dockerfile
    )

    result = harbor_grade(task_dir, attempt, sandboxes, backend=microvm(image_user=True, image_env=True))

    assert (result["reward"], (attempt / "verifier" / "identity.txt").read_text()) == (
        1.0,
        "checker|verifier",
    )


def grade_in_place(
    task_dir: Path, attempt: Path, sandboxes: DirectorySandboxes, kept_sandbox: dict | None, **changes
):
    """The grade of ``task_dir`` in the sandbox that the rollout ``rollout-job`` kept, ``kept_sandbox``."""
    body = {
        "benchmark": "harbor",
        "task_dir": str(task_dir),
        "in_place_job": "rollout-job",
        "backend": PODMAN,
        **changes,
    }
    return grade(body, attempt, session_factory=sandboxes, kept_sandbox=kept_sandbox)


FUTURE = "2999-01-01T00:00:00+00:00"


def test_a_grade_in_place_verifies_in_the_kept_sandbox_on_the_verifiers_network(tmp_path, sandboxes, attempt):
    """SWE-bench Pro's shape: the agent ran offline, and its verifier runs online, in the same sandbox."""
    kept = sandboxes.kept("snap-1")
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh=ANSWER_TEST)

    result = grade_in_place(task_dir, attempt, sandboxes, {"sandbox_id": kept, "keep_until": FUTURE})

    [session] = sandboxes.sessions
    assert (result["reward"], result["reason"], result["verifier"]["in_place"]) == (1.0, None, True)
    assert (sandboxes.lifecycle, session.networks) == ([("attach", kept), ("destroy", kept)], [True])


@pytest.mark.parametrize(
    "kept_sandbox, error, lifecycle",
    [
        (None, "this worker keeps no such sandbox, or its keep_until has passed", []),
        (
            {"sandbox_id": "gone-1", "keep_until": FUTURE},
            "sandbox gone-1 does not answer",
            [("attach", "gone-1")],
        ),
    ],
)
def test_a_grade_in_place_whose_sandbox_is_gone_scores_0_as_sandbox_lost(
    tmp_path, sandboxes, attempt, kept_sandbox, error, lifecycle
):
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh=ANSWER_TEST)

    result = grade_in_place(task_dir, attempt, sandboxes, kept_sandbox)

    assert result == {
        "status": "completed",
        "reward": 0.0,
        "reason": "sandbox_lost",
        "verifier": {"in_place": True, "error": error},
    }
    assert sandboxes.lifecycle == lifecycle


def test_a_grade_in_place_on_a_backend_that_cannot_switch_networks_fails_and_removes_the_sandbox(
    tmp_path, sandboxes, attempt
):
    """A microVM cannot change a running sandbox's egress. The attempt fails, and a retry finds it gone."""
    kept = sandboxes.kept("snap-1")
    task_dir = harbor_task(tmp_path / "task", toml=SHARED, test_sh=ANSWER_TEST)

    with pytest.raises(NotImplementedError):
        grade_in_place(
            task_dir, attempt, sandboxes, {"sandbox_id": kept, "keep_until": FUTURE}, backend=microvm()
        )

    assert (sandboxes.lifecycle, sandboxes.live()) == ([("attach", kept), ("destroy", kept)], [])

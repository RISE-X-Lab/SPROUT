"""``setup``: the guided setup, driven through ``main`` with scripted answers on stdin.

The interpreter in services.yaml is a stub that answers as ``python -m sprout.worker doctor --json``,
except on a new host, where the real doctor runs on the fake podman and btrfs. A fixture removes
each variable that a question reads its default from.
"""

from __future__ import annotations

import io
import json
import re
import sys
from pathlib import Path

import pytest

from sprout.services import __main__ as cli
from sprout.services.spec import template_path
from tests.services.helpers import StubProgram, doctor_report, make_stub, service_command, write_spec
from tests.support.http import chat_completion, chat_server
from tests.support.paths import RUNTIME_BIN

TEMPLATE = template_path("services.yaml").read_text(encoding="utf-8")
#: The flags that skip the model step and answer no to the start.
NO_MODEL = ["--model-endpoint", "", "--no-start"]
#: The value of the key variable. No output may hold it.
KEY = "secret-value-123"
#: The profiles whose worker_env names the model key.
KEY_PROFILES = ["mini", "mini-microvm", "review"]


@pytest.fixture(autouse=True)
def shell(monkeypatch, tmp_path):
    """A shell without the variables that a question reads its default from, and with its own store."""
    for name in ("QWEN_BASE_URL", "OPENAI_BASE_URL", "SPROUT_MODEL_API_KEY", "LUNA_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(tmp_path / "store"))


@pytest.fixture
def python(tmp_path) -> StubProgram:
    """The interpreter that services.yaml names. It reports a host that passes each check."""
    stub = make_stub(tmp_path / "venv" / "python")
    stub.answer(out=doctor_report())
    return stub


def answers(monkeypatch, *lines: str) -> None:
    """Make ``lines`` the input of the setup, one answer on each line."""
    monkeypatch.setattr(sys, "stdin", io.StringIO("".join(line + "\n" for line in lines)))


def on_stub(services: str, python: StubProgram) -> str:
    """The services.yaml text ``services`` with ``python`` as its interpreter."""
    return services.replace("python: null\n", "python: %s\n" % python.path, 1)


def write_host(paths, python: StubProgram, *, services: str = TEMPLATE) -> None:
    """Write ``services`` on ``python`` as services.yaml, then each other file as ``init`` writes it."""
    paths.config.mkdir(parents=True, exist_ok=True)
    paths.spec.write_text(on_stub(services, python))
    cli.main(["init"])


def runstore(paths) -> dict:
    return json.loads((paths.config / "runstore.json").read_text())


def write_runstore(paths, config: dict) -> str:
    text = json.dumps(config, indent=2) + "\n"
    (paths.config / "runstore.json").write_text(text)
    return text


def worker_envs(paths) -> dict[str, dict]:
    """The worker_env of each profile that names the model key, by profile."""
    profiles = runstore(paths)["profiles"]
    return {name: profiles[name].get("worker_env") for name in KEY_PROFILES}


def step(out: str, number: int) -> list[str]:
    """The lines of step ``number`` of a setup transcript, from its heading to the next blank line."""
    return next(part for part in out.split("\n\n") if part.startswith("%d/4 " % number)).splitlines()


def backups(paths) -> list[Path]:
    return sorted(paths.config.glob("backup-*"))


# --- the host checks ----------------------------------------------------------


def test_a_failed_host_check_shows_its_fix_and_the_answer_no_stops_before_any_config_file(
    paths, python, monkeypatch, capsys
):
    python.answer(out=doctor_report(checks={"store": ("fail", "store does not exist: /path/to/store")}))
    write_spec(paths, {"web": {"kind": "process", "command": ["true"]}}, python=str(python.path))
    answers(monkeypatch, "n")

    code = cli.main(["setup"])

    assert (code, step(capsys.readouterr().out, 1), sorted(path.name for path in paths.config.iterdir())) == (
        1,
        [
            "1/4 Host checks (python -m sprout.worker doctor)",
            "ok    podman           podman 9.9.9 (/path/to/podman)",
            "fail  store            store does not exist: /path/to/store",
            "      fix: ops/setup_podman.sh",
            "ok    runtime          /path/to/sprout-runtime",
            "ok    keyring          kernel.keys.maxkeys=20000",
            "A host check failed. Go on with the config files? [Y/n] n",
            "stopped: fix the host, then run `python -m sprout.services setup` again",
        ],
        ["services.yaml"],
    )


def test_setup_on_a_new_host_writes_the_files_that_init_writes_with_the_store_of_the_doctor(
    paths, fake_podman, fake_btrfs_store, monkeypatch, tmp_path
):
    """The real doctor runs on the fake podman and btrfs. A missing runtime binary only asks to go on."""
    monkeypatch.setenv("SPROUT_PODMAN_STORE", str(fake_btrfs_store))
    answers(monkeypatch)

    code = cli.main(["setup", "--yes", "--continue-after-failure", *NO_MODEL])
    written = {path.name: path.read_text() for path in paths.config.iterdir()}
    monkeypatch.setenv("SPROUT_SERVICES_CONFIG_DIR", str(tmp_path / "init-config"))
    cli.main(["init"])
    by_init = {path.name: path.read_text() for path in (tmp_path / "init-config").iterdir()}

    assert (code, written, json.loads(written["runstore.json"])["profiles"]["grade"]["grade_defaults"]) == (
        0,
        by_init,
        {
            "backend": {
                "backend": "podman",
                "podman": {"store": str(fake_btrfs_store), "runtime_bin": str(RUNTIME_BIN)},
            }
        },
    )


# --- the config files ---------------------------------------------------------


def test_setup_prints_each_problem_in_runstore_json_with_its_fix_and_changes_no_file(
    paths, python, monkeypatch, capsys
):
    monkeypatch.setenv("SPROUT_MODEL_API_KEY", "set")
    write_host(paths, python)
    config = runstore(paths)
    config["artifact_root"] = "/path/to/data"
    old = write_runstore(paths, config)
    answers(monkeypatch)

    code = cli.main(["setup", *NO_MODEL])

    assert (code, step(capsys.readouterr().out, 2)[-2:], (paths.config / "runstore.json").read_text()) == (
        0,
        [
            "fail  runstore.json    `artifact_root` is the example path /path/to/data of the template, at "
            "the top level",
            "      fix: set it to a path on this host",
        ],
        old,
    )


def test_with_no_problem_setup_says_so_and_makes_no_backup(paths, python, monkeypatch, capsys):
    monkeypatch.setenv("SPROUT_MODEL_API_KEY", "set")
    write_host(paths, python)
    answers(monkeypatch)

    cli.main(["setup", *NO_MODEL])

    assert (step(capsys.readouterr().out, 2)[-1], backups(paths)) == (
        "ok    no problem found in runstore.json",
        [],
    )


# --- the model endpoint -------------------------------------------------------


def test_the_model_step_references_the_key_in_the_actor_and_review_profiles_and_removes_an_unset_one(
    paths, python, monkeypatch, capsys
):
    """The worker passes a job only the variables that its profile references."""
    monkeypatch.setenv("LUNA_KEY", KEY)
    write_host(paths, python)
    answers(monkeypatch, "http://127.0.0.1:4000/v1", "luna", "LUNA_KEY", "y", "y", "n")

    code = cli.main(["setup", "--no-start"])

    out = capsys.readouterr().out
    assert (code, step(out, 3)[:7], worker_envs(paths), KEY in out) == (
        0,
        [
            "3/4 Model endpoint",
            "Model endpoint URL, empty to skip this step []: http://127.0.0.1:4000/v1",
            "Model name [qwen3.8-27b]: luna",
            "Environment variable that holds the key [SPROUT_MODEL_API_KEY]: LUNA_KEY",
            "      LUNA_KEY is set in this shell.",
            'Add {"LUNA_KEY": {"$env": "LUNA_KEY"}} to the worker_env of mini, mini-microvm, review? [Y/n] y',
            "SPROUT_MODEL_API_KEY is not set in this shell. Remove its reference from the worker_env of "
            "mini, mini-microvm, review? [y/N] y",
        ],
        {name: {"LUNA_KEY": {"$env": "LUNA_KEY"}} for name in KEY_PROFILES},
        False,
    )


def test_before_it_rewrites_runstore_json_setup_copies_it_into_a_timestamped_backup_directory(
    paths, python, monkeypatch, capsys
):
    """A change that the user regrets can go back: the directory holds the file as it was."""
    monkeypatch.setenv("LUNA_KEY", KEY)
    write_host(paths, python)
    old = (paths.config / "runstore.json").read_text()
    answers(monkeypatch, "http://127.0.0.1:4000", "luna", "LUNA_KEY", "y", "n", "n")

    cli.main(["setup", "--no-start"])

    (backup,) = backups(paths)
    assert (
        re.fullmatch(r"backup-\d{8}-\d{6}", backup.name) is not None,
        {path.name: path.read_text() for path in backup.iterdir()},
        "backup %s" % backup in capsys.readouterr().out.splitlines(),
    ) == (True, {"runstore.json": old}, True)


def test_by_default_setup_keeps_each_reference_to_an_unset_variable(paths, python, monkeypatch):
    """A reference is a choice of the host. The worker's shell can set the variable later."""
    write_host(paths, python)
    answers(monkeypatch)

    cli.main(["setup", "--yes", "--model-endpoint", "http://127.0.0.1:4000", "--model-key-env", "LUNA_KEY"])

    assert worker_envs(paths) == {
        name: {"SPROUT_MODEL_API_KEY": {"$env": "SPROUT_MODEL_API_KEY"}, "LUNA_KEY": {"$env": "LUNA_KEY"}}
        for name in KEY_PROFILES
    }


def test_an_endpoint_that_a_rollout_request_refuses_is_asked_again(paths, python, monkeypatch, capsys):
    write_host(paths, python)
    answers(monkeypatch, "ftp://127.0.0.1/v1", "http://127.0.0.1:4000", "luna", "LUNA_KEY", "n", "n", "n")

    code = cli.main(["setup", "--no-start"])

    assert (code, step(capsys.readouterr().out, 3)[1:4]) == (
        0,
        [
            "Model endpoint URL, empty to skip this step []: ftp://127.0.0.1/v1",
            "Rollout endpoint must be HTTP(S), without credentials/query/fragment",
            "Model endpoint URL, empty to skip this step []: http://127.0.0.1:4000",
        ],
    )


def test_an_endpoint_flag_that_a_rollout_request_refuses_stops_the_setup(paths, python, monkeypatch, capsys):
    write_host(paths, python)
    answers(monkeypatch)

    code = cli.main(["setup", "--yes", "--model-endpoint", "ftp://127.0.0.1/v1"])

    assert (code, capsys.readouterr().out.splitlines()[-1]) == (
        1,
        "stopped: --model-endpoint ftp://127.0.0.1/v1: Rollout endpoint must be HTTP(S), without "
        "credentials/query/fragment",
    )


def test_the_test_request_is_one_short_chat_completion_and_its_line_never_holds_the_key(
    paths, python, monkeypatch, capsys
):
    """An endpoint can echo the key in an error. The transcript of the setup must not."""
    monkeypatch.setenv("LUNA_KEY", KEY)
    write_host(paths, python)
    answers(monkeypatch)
    reply = (401, {"error": {"message": "bad key %s" % KEY}})

    with chat_server([reply]) as server:
        cli.main(
            [
                "setup",
                "--yes",
                "--model",
                "luna",
                "--model-endpoint",
                server.url,
                "--model-key-env",
                "LUNA_KEY",
                "--check-endpoint",
            ]
        )

    out = capsys.readouterr().out
    assert (server.requests, step(out, 3)[-1], KEY in out) == (
        [
            {
                "model": "luna",
                "messages": [{"role": "user", "content": "Reply with the word ok."}],
                "max_tokens": 16,
            }
        ],
        '      fail  HTTP 401: {"error": {"message": "bad key ***"}}',
        False,
    )


def test_an_endpoint_that_answers_the_test_request_reads_ok(paths, python, monkeypatch, capsys):
    monkeypatch.setenv("LUNA_KEY", KEY)
    write_host(paths, python)
    answers(monkeypatch)

    with chat_server([chat_completion()]) as server:
        cli.main(
            [
                "setup",
                "--yes",
                "--model-endpoint",
                server.url,
                "--model-key-env",
                "LUNA_KEY",
                "--check-endpoint",
            ]
        )

    assert step(capsys.readouterr().out, 3)[-1] == "      ok    HTTP 200: the model answered"


def test_by_default_setup_sends_no_test_request(paths, python, monkeypatch):
    """The request costs money, and it can count against the rate limit of a free tier."""
    monkeypatch.setenv("LUNA_KEY", KEY)
    write_host(paths, python)
    answers(monkeypatch)

    with chat_server([]) as server:
        cli.main(["setup", "--yes", "--model-endpoint", server.url, "--model-key-env", "LUNA_KEY"])

    assert server.requests == []


# --- the start and the next commands ------------------------------------------


def test_the_last_step_prints_the_env_line_and_a_plan_run_on_the_model_and_the_run_store_of_the_host(
    paths, python, monkeypatch, capsys
):
    """The Run Store URL is the port of runstore-api in services.yaml, which a host can change."""
    write_host(paths, python, services=TEMPLATE.replace("    port: 18110\n", "    port: 18111\n"))
    answers(monkeypatch)

    cli.main(
        [
            "setup",
            "--yes",
            "--model",
            "luna",
            "--model-endpoint",
            "http://127.0.0.1:4000/v1",
            "--model-key-env",
            "LUNA_KEY",
        ]
    )

    assert capsys.readouterr().out.splitlines()[-3:] == [
        "Next, in the shell of the experiment:",
        '  eval "$(python -m sprout.services env)"',
        "  python -m examples.qwen.run --dataset deepswe --method baseline --tasks-dir <tasks directory> "
        "--task <task> --output <output directory> --model luna --model-endpoint http://127.0.0.1:4000 "
        "--model-key-env LUNA_KEY --runstore-url http://127.0.0.1:18111 --plan",
    ]


def test_the_answer_yes_to_the_start_starts_the_run_store_and_the_worker(
    paths, python, monkeypatch, tmp_path
):
    process = {"kind": "process", "command": service_command(tmp_path), "health": {"kind": "none"}}
    write_spec(
        paths,
        {"runstore-api": process, "runstore-worker": {**process, "depends_on": ["runstore-api"]}},
        python=str(python.path),
    )
    answers(monkeypatch)

    code = cli.main(["setup", "--yes", "--model-endpoint", "", "--start"])

    assert (code, paths.pid_file("runstore-api").exists(), paths.pid_file("runstore-worker").exists()) == (
        0,
        True,
        True,
    )


def test_a_start_that_fails_makes_the_setup_exit_1(paths, python, monkeypatch, tmp_path):
    """A script that runs ``setup --yes --start`` must see that the Run Store did not come up."""
    failing = {"kind": "process", "command": service_command(tmp_path, "--exit", "3")}
    write_spec(paths, {"runstore-api": failing, "runstore-worker": failing}, python=str(python.path))
    answers(monkeypatch)

    code = cli.main(["setup", "--yes", "--model-endpoint", "", "--start"])

    assert code == 1


# --- without a terminal -------------------------------------------------------


def test_with_yes_and_flags_setup_needs_no_input(paths, python, monkeypatch):
    write_host(paths, python)
    answers(monkeypatch)

    code = cli.main(["setup", "--yes", "--model-endpoint", "http://127.0.0.1:4000", "--no-start"])

    assert code == 0


def test_without_yes_an_input_that_ends_before_an_answer_stops_the_setup_and_names_yes(
    paths, python, monkeypatch, capsys
):
    write_host(paths, python)
    answers(monkeypatch)

    code = cli.main(["setup"])

    assert (code, capsys.readouterr().out.splitlines()[-1]) == (
        1,
        "stopped: the input ended before an answer: pass --yes and the flags to set up without a terminal",
    )

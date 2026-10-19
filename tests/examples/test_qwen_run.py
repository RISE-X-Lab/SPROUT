"""``examples.qwen.run``: the experiment runner, which runs each stage as a ``sprout.eval.evaluate`` process.

``FakeEvaluate`` takes the place of that process where ``run.stage`` starts it (``subprocess.call``). It
writes the summary of its stage as the evaluator does, and its parents are rollouts of three turns that
the fake worker writes (``tests.eval.worker.write_rollout``), so they have branch points. ``FakeGit``
answers the runner's two git calls. The tasks are copies of the Harbor fixture task named ``org/<name>``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from examples.qwen import run
from sprout.eval import evaluate
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH
from sprout.worker.run.journal import JournalWriter
from tests.eval.worker import snapshot_id, write_rollout
from tests.support.paths import HARBOR_SHELL_TASK


def tasks_dir(tmp_path: Path, *names: str) -> Path:
    """``tmp_path/tasks`` with a copy of the fixture task for each of ``names``, each named ``org/<name>``."""
    for name in names:
        task = Path(shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "tasks" / name))
        (task / "task.toml").write_text((task / "task.toml").read_text() + f'\n[task]\nname = "org/{name}"\n')
    return tmp_path / "tasks"


def argv(tmp_path: Path, *extra: str) -> list[str]:
    """The command line of a baseline experiment on the task ``fixture`` of ``tmp_path/tasks``, and ``extra``.

    A ``--method`` in ``extra`` replaces the baseline."""
    return [
        "--dataset",
        "deepswe",
        "--method",
        "baseline",
        "--tasks-dir",
        str(tmp_path / "tasks"),
        "--task",
        "fixture",
        "--output",
        str(tmp_path / "out"),
        "--model-endpoint",
        "http://model.test/v1",
        *extra,
    ]


def runner_args(tmp_path: Path, *extra: str) -> argparse.Namespace:
    """The options of an experiment, completed as ``main`` completes them before the first stage."""
    args = run.parse_args(argv(tmp_path, *extra))
    args.digest, args.source_commit, args.preparation = (
        "sha256:fixture",
        "0" * 40,
        args.output / "preparation.json",
    )
    return args


def option(command: list[str], flag: str) -> str | None:
    """The value of ``flag`` in ``command``, or None without it."""
    return command[command.index(flag) + 1] if flag in command else None


class FakeEvaluate:
    """``python -m sprout.eval.evaluate`` as ``run.stage`` starts it: it logs its stage, writes its summary.

    ``resolved`` maps the name of a stage's output (``attempt-1``, ``sprout``) to the verdict of each attempt
    after the parent, and ``exit_codes`` to the exit status of a stage that fails. A baseline stage is one
    rollout. A branch stage starts its summary with the row of its parent, as the evaluator does."""

    def __init__(self, artifacts: Path) -> None:
        self.artifacts = artifacts
        self.resolved: dict[str, list[bool]] = {}
        self.exit_codes: dict[str, int] = {}
        #: The names of the stages whose rollouts took no snapshot after a turn, so they have no branch point.
        self.pointless: set[str] = set()
        #: Each command line, in call order.
        self.commands: list[list[str]] = []

    def __call__(self, command: list[str], **settings: Any) -> int:
        self.commands.append(command)
        output = Path(command[command.index("--output") + 1])
        settings["stdout"].write(f"evaluate {output.name}\n".encode())
        if output.name in self.exit_codes:
            return self.exit_codes[output.name]
        verdicts = self.resolved.get(output.name, [False])
        parent = option(command, "--parent-output")
        rows = [*json.loads((Path(parent) / "summary.json").read_text())["attempts"][:1]] if parent else []
        for index, resolved in enumerate(verdicts):
            directory = self.artifacts / "-".join(output.parts[-3:]) / str(index)
            directory.mkdir(parents=True)
            journal = (pointless if output.name in self.pointless else write_rollout)(
                directory, job_id=f"{output.name}-{index}"
            )
            rows.append(
                {
                    "name": f"{output.name}-{index}",
                    "status": "completed",
                    "resolved": resolved,
                    "journal": str(journal),
                }
            )
        run.write(output / "summary.json", {"attempts": rows, "complete": True})
        return 0

    def stages(self) -> list[str]:
        """The output of each stage that ran, as ``<task>/<stage>``, in call order."""
        names = [Path(command[command.index("--output") + 1]) for command in self.commands]
        return [
            f"{path.parts[-3]}/{path.name}"
            if path.parent.name == "baseline"
            else f"{path.parent.name}/{path.name}"
            for path in names
        ]


def pointless(directory: Path, *, job_id: str) -> Path:
    """Write the journal of a mini run that took no snapshot, into ``directory``. Returns the journal."""
    with JournalWriter(directory / "trajectory.jsonl", run_id=job_id) as writer:
        writer.emit("run.started", slot="mini-swe-agent")
        writer.emit("run.finished", status="completed")
    return directory / "trajectory.jsonl"


class FakeGit:
    """The two git calls of ``run.main``. ``rev-parse HEAD`` gives ``commit``.

    ``status`` lists a change when ``dirty``."""

    def __init__(self) -> None:
        self.commit = "a" * 40
        self.dirty = False

    def __call__(self, command: list[str], **settings: Any) -> str:
        if command[1] == "rev-parse":
            return self.commit + "\n"
        return " M src/sprout/eval/gate.py\n" if self.dirty else ""


@pytest.fixture
def evaluator(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FakeEvaluate:
    """The fake evaluator in the place of the evaluator process."""
    fake = FakeEvaluate(tmp_path / "artifacts")
    monkeypatch.setattr(
        run, "subprocess", SimpleNamespace(call=fake, check_output=FakeGit(), STDOUT=subprocess.STDOUT)
    )
    return fake


@pytest.fixture
def git(evaluator: FakeEvaluate) -> FakeGit:
    """The fake git of the runner."""
    git: FakeGit = run.subprocess.check_output
    return git


@pytest.fixture
def credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    """The Run Store token and the selector key that ``main`` checks before its first stage."""
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "fixture-token")
    monkeypatch.setenv("QWEN_API_KEY", "fixture-key")


# --- the stage command ----------------------------------------------------------------------


#: What ``sprout.eval.evaluate`` reads from a stage command: each setting that the runner passes.
STAGE_SETTINGS = (
    "task_dir",
    "dataset_digest",
    "preparation_manifest",
    "output",
    "method",
    "model",
    "model_endpoint",
    "model_key_env",
    "runstore_url",
    "runstore_token_env",
    "profile",
    "grade_profile",
    "review_profile",
    "reasoning_effort",
    "max_turns",
    "max_output_tokens",
    "review_output_tokens",
    "meta_max_tokens",
    "prompt_budget_chars",
    "command_timeout",
    "review_timeout",
    "max_rollouts",
    "branch_count_mode",
    "rounds",
    "branches",
    "checkpoint_mode",
    "parent_output",
    "structured_review",
    "leak_judge",
    "no_image_user",
    "no_image_env",
)


def stage_settings(command: list[str]) -> dict[str, Any]:
    """What ``sprout.eval.evaluate`` reads from the stage command ``command``."""
    args = evaluate.parse_args([*command[4:], "--volatile-ok"])
    return {name: getattr(args, name) for name in STAGE_SETTINGS}


#: A value other than the default for each option that the runner passes to every stage.
SETTINGS = (
    "--model luna --model-key-env LUNA_KEY --runstore-url http://127.0.0.1:7 --runstore-token-env RS_TOKEN "
    "--profile p --grade-profile g --review-profile r --max-turns 77 --max-output-tokens 1000 "
    "--review-output-tokens 900 --prompt-budget-chars 60000 --command-timeout 61 --review-timeout 62 "
    "--max-rollouts 6 --branch-count-mode fixed"
).split()
BASELINE_ROOT = ["--baseline-root", "/path/to/base"]


@pytest.mark.parametrize(
    "method, extra, parent, rounds",
    [
        ("baseline", [], None, 0),
        ("sprout", [], Path("/path/to/parent"), 2),
        ("bpo", BASELINE_ROOT, Path("/path/to/parent"), 0),
        ("shepherd", BASELINE_ROOT, Path("/path/to/parent"), 0),
    ],
)
def test_each_stage_command_is_an_evaluate_command_line_with_the_settings_of_the_run(
    tmp_path: Path, method: str, extra: list[str], parent: Path | None, rounds: int
) -> None:
    """A flag that eval.evaluate renamed, or that the runner forgot, makes a stage run on another setting."""
    args = runner_args(tmp_path, "--method", method, *SETTINGS, *extra)

    command = run.command(args, "fixture", tmp_path / "stage", method, parent)

    assert command[:4] == [sys.executable, "-u", "-m", "sprout.eval.evaluate"]
    assert stage_settings(command) == {
        "task_dir": tmp_path / "tasks" / "fixture",
        "dataset_digest": "sha256:fixture",
        "preparation_manifest": tmp_path / "out" / "preparation.json",
        "output": tmp_path / "stage",
        "method": method,
        "model": "luna",
        "model_endpoint": "http://model.test/v1",
        "model_key_env": "LUNA_KEY",
        "runstore_url": "http://127.0.0.1:7",
        "runstore_token_env": "RS_TOKEN",
        "profile": "p",
        "grade_profile": "g",
        "review_profile": "r",
        "reasoning_effort": "high",
        "max_turns": 77,
        "max_output_tokens": 1000,
        "review_output_tokens": 900,
        "meta_max_tokens": 1000,
        "prompt_budget_chars": 60000,
        "command_timeout": 61,
        "review_timeout": 62,
        "max_rollouts": 6,
        "branch_count_mode": "fixed",
        "rounds": rounds,
        "branches": [4, 1],
        "checkpoint_mode": "disk_only",
        "parent_output": parent,
        "structured_review": False,
        "leak_judge": False,
        "no_image_user": False,
        "no_image_env": False,
    }


@pytest.mark.parametrize(
    "method, reviewer",
    [
        ("sprout", ("critic", "http://critic.test/v1", "CRITIC_KEY", "medium", 777, True, 1, 0.5)),
        ("baseline", ("qwen3.8-27b", "http://model.test/v1", "QWEN_API_KEY", "high", 777, False, 2, 0.75)),
    ],
)
def test_only_a_sprout_stage_gets_its_own_reviewer_and_the_leak_judge(
    tmp_path: Path, method: str, reviewer: tuple[Any, ...]
) -> None:
    """A baseline has no reviewer to configure or to judge, so a shared baseline stays one setting."""
    review = [
        "--review-model",
        "critic",
        "--review-endpoint",
        "http://critic.test/v1",
        "--review-key-env",
        "CRITIC_KEY",
    ]
    args = runner_args(
        tmp_path,
        "--method",
        "sprout",
        *review,
        "--review-effort",
        "medium",
        "--review-output-tokens",
        "777",
        "--leak-judge",
        "--leak-judge-resamples",
        "1",
        "--leak-judge-threshold",
        "0.5",
    )

    parsed = evaluate.parse_args(
        [*run.command(args, "fixture", tmp_path / method, method)[4:], "--volatile-ok"]
    )

    assert (
        parsed.review_model,
        parsed.review_endpoint,
        parsed.review_key_env,
        parsed.review_effort,
        parsed.review_output_tokens,
        parsed.leak_judge,
        parsed.leak_judge_resamples,
        parsed.leak_judge_threshold,
    ) == reviewer


def edited_prompts(path: Path) -> Path:
    """A copy of the packaged prompts file at ``path``, with one prompt changed."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["reviewer"]["user_hint"] += "Edited for this test.\n"
    path.write_text(yaml.safe_dump(data))
    return path


@pytest.mark.parametrize("method", run.METHODS)
def test_a_prompts_file_reaches_every_stage_and_the_manifest_records_its_hash(
    tmp_path: Path, method: str
) -> None:
    """Shepherd's system prompt comes from the file too, so each method's stage gets the file."""
    prompts = edited_prompts(tmp_path / "prompts.yaml")
    args = runner_args(tmp_path, "--method", "sprout", "--prompts", str(prompts))

    parsed = evaluate.parse_args(
        [
            *run.command(args, "fixture", tmp_path / method, method, Path("/path/to/parent"))[4:],
            "--volatile-ok",
        ]
    )

    digest = hashlib.sha256(prompts.read_bytes()).hexdigest()
    assert (parsed.prompts.sha256, run.manifest_for(args)["prompts_sha256"]) == (digest, digest)


def test_without_a_prompts_file_every_stage_reads_the_packaged_one(tmp_path: Path) -> None:
    args = runner_args(tmp_path, "--method", "sprout")

    command = run.command(args, "fixture", tmp_path / "sprout", "sprout", Path("/path/to/parent"))

    assert (option(command, "--prompts"), run.manifest_for(args)["prompts_sha256"]) == (
        None,
        DEFAULT_PROMPTS.sha256,
    )


@pytest.mark.parametrize("method", run.METHODS)
def test_structured_review_reaches_every_stage(tmp_path: Path, method: str) -> None:
    args = runner_args(tmp_path, "--method", "sprout", "--structured-review")

    assert (
        stage_settings(run.command(args, "fixture", tmp_path / method, method, Path("/path/to/parent")))[
            "structured_review"
        ]
        is True
    )


#: Each flag that turns off a part of the Harbor protocol, with the two settings that it leaves.
PROTOCOL_FLAGS = [
    pytest.param("--no-image-user", (False, True), id="image-user"),
    pytest.param("--no-image-env", (True, False), id="image-env"),
]


@pytest.mark.parametrize("flag, kept", PROTOCOL_FLAGS)
@pytest.mark.parametrize("method", run.METHODS)
def test_each_flag_that_turns_off_a_part_of_the_harbor_protocol_reaches_every_stage(
    tmp_path: Path, method: str, flag: str, kept: tuple[bool, bool]
) -> None:
    """A branch stage gets the flag too: ADR-0005 has the runner pass the settings to each stage."""
    args = runner_args(tmp_path, *SPROUT, flag)

    command = run.command(args, "fixture", tmp_path / method, method, Path("/path/to/parent"))

    settings = stage_settings(command)
    assert (not settings["no_image_user"], not settings["no_image_env"]) == kept


def test_a_stage_without_checkpoints_asks_for_none(tmp_path: Path) -> None:
    """The baseline rollouts after the first are leaves: nothing branches from them."""
    command = run.command(
        runner_args(tmp_path), "fixture", tmp_path / "attempt-2", "baseline", checkpoint=False
    )

    assert stage_settings(command)["checkpoint_mode"] == "none"


@pytest.mark.parametrize(
    "extra, widths, max_rollouts",
    [
        ([], [4, 3], 8),
        (["--max-rollouts", "2"], [1], 2),
        (["--max-rollouts", "5"], [4], 5),
        (["--max-rollouts", "6"], [4, 1], 6),
        (["--sprout-rounds", "3", "--sprout-branches", "2"], [2, 2, 2], 7),
        (["--sprout-rounds", "8"], [1, 1, 1, 1, 1, 1, 1, 1], 9),
        (["--max-rollouts", "9", "--branches", "5", "3"], [5, 3], 9),
    ],
    ids=["protocol", "two", "five", "six", "rounds-of-two", "rounds-of-one", "explicit"],
)
def test_the_sprout_schedule_follows_the_budget_unless_a_schedule_is_given(
    tmp_path: Path, extra: list[str], widths: list[int], max_rollouts: int
) -> None:
    """The protocol's schedule is 4 branches, then the rest of the budget.

    ``--sprout-rounds`` sets the budget."""
    args = runner_args(tmp_path, "--method", "sprout", *extra)

    assert (run.branch_widths(args), args.max_rollouts) == (widths, max_rollouts)


# --- the command line -------------------------------------------------------------------------


def usage_error(capsys: pytest.CaptureFixture[str]) -> str:
    """The message of the error that argparse printed last, without the program name."""
    return capsys.readouterr().err.splitlines()[-1].split(": error: ", 1)[1]


SPROUT = ["--method", "sprout"]
BUDGET = "Select a dataset, positive workers/turns, and 2..128 total method rollouts"
ROUNDS = "--sprout-rounds must be 1..16 and --sprout-branches 1..4"
JUDGE = "--leak-judge-resamples must be >= 0 and --leak-judge-threshold in (0, 1]"
KEY_NAMES = "--model-key-env and --runstore-token-env must be environment variable names"


@pytest.mark.parametrize(
    "extra, message",
    [
        (
            ["--method", "bpo", "--baseline-root", "/path/to/base", "--sprout-rounds", "2"],
            "--sprout-rounds sets the sprout schedule; use it with --method sprout",
        ),
        (
            [*SPROUT, "--sprout-rounds", "2", "--branches", "1"],
            "--sprout-rounds and --branches each give the whole schedule; pass one of them",
        ),
        ([*SPROUT, "--sprout-rounds", "0"], ROUNDS),
        ([*SPROUT, "--sprout-rounds", "17"], ROUNDS),
        ([*SPROUT, "--sprout-rounds", "1", "--sprout-branches", "0"], ROUNDS),
        ([*SPROUT, "--sprout-rounds", "1", "--sprout-branches", "5"], ROUNDS),
        (
            [*SPROUT, "--sprout-rounds", "5", "--sprout-branches", "4"],
            "a sprout schedule may use at most 17 rollouts, the parent included; "
            "a wider one is spelled out with --branches",
        ),
        (
            [*SPROUT, "--max-rollouts", "9"],
            "Select 2..8 total method rollouts: the protocol's schedule derives from the budget, "
            "and a larger budget takes an explicit --branches or --sprout-rounds schedule",
        ),
        (
            [*SPROUT, "--max-rollouts", "1"],
            "Select 2..8 total method rollouts: the protocol's schedule derives from the budget, "
            "and a larger budget takes an explicit --branches or --sprout-rounds schedule",
        ),
        ([*SPROUT, "--max-rollouts", "129", "--branches", "1"], BUDGET),
        (["--workers", "0"], BUDGET),
        (["--max-turns", "0"], BUDGET),
        (
            [*SPROUT, "--branches", "4", "4"],
            "--branches must be positive counts whose sum plus the parent fits --max-rollouts",
        ),
        (
            [*SPROUT, "--branches", "0"],
            "--branches must be positive counts whose sum plus the parent fits --max-rollouts",
        ),
        (
            ["--baseline-rollouts", "0"],
            "--baseline-rollouts must be at least 1: the first rollout is every branch method's parent",
        ),
        (
            ["--prompt-budget-chars", "49999"],
            "--prompt-budget-chars below 50000 leaves no room for the task and a transcript",
        ),
        (["--command-timeout", "0"], "--command-timeout must be a positive number of seconds"),
        (["--review-timeout", "0"], "--review-timeout must be a positive number of seconds"),
        (["--review-key-env", "CRITIC-KEY"], "--review-key-env must be an environment variable name"),
        (["--review-output-tokens", "0"], "--review-output-tokens must be positive"),
        (
            ["--method", "baseline", "--leak-judge"],
            "--leak-judge checks reviewer-written turns; use it with --method sprout",
        ),
        ([*SPROUT, "--leak-judge", "--leak-judge-resamples", "-1"], JUDGE),
        ([*SPROUT, "--leak-judge", "--leak-judge-threshold", "0"], JUDGE),
        (["--method", "bpo"], "bpo and shepherd branch from a shared baseline: pass --baseline-root"),
        (["--method", "shepherd"], "bpo and shepherd branch from a shared baseline: pass --baseline-root"),
        (["--model-key-env", "QWEN KEY"], KEY_NAMES),
        (["--runstore-token-env", "1TOKEN"], KEY_NAMES),
        (["--task", "fixture"], "Task IDs must be unique directory names"),
        (["--task", "../fixture"], "Task IDs must be unique directory names"),
        (["--task", ".."], "Task IDs must be unique directory names"),
        (
            ["--prompts", "/path/to/absent.yaml"],
            "--prompts: [Errno 2] No such file or directory: '/path/to/absent.yaml'",
        ),
    ],
)
def test_an_experiment_option_that_breaks_the_protocol_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit):
        run.parse_args(argv(tmp_path, *extra))

    assert usage_error(capsys) == message


def test_the_command_line_refuses_a_prompts_file_that_does_not_load(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "prompts.yaml").write_text("analyst: [1]\n")

    with pytest.raises(SystemExit):
        run.parse_args(argv(tmp_path, "--prompts", str(tmp_path / "prompts.yaml")))

    assert usage_error(capsys).startswith("--prompts: %s: " % (tmp_path / "prompts.yaml"))


@pytest.mark.parametrize(
    "extra",
    [
        ["--max-rollouts", "2"],
        ["--max-rollouts", "8"],
        ["--sprout-rounds", "16"],
        ["--sprout-rounds", "4", "--sprout-branches", "4"],
        ["--branches", "7"],
        ["--max-rollouts", "128", "--branches", "127"],
        ["--baseline-rollouts", "1"],
        ["--prompt-budget-chars", "50000"],
        ["--leak-judge", "--leak-judge-resamples", "0", "--leak-judge-threshold", "1"],
    ],
)
def test_the_limits_of_each_experiment_option_are_accepted(tmp_path: Path, extra: list[str]) -> None:
    run.parse_args(argv(tmp_path, *SPROUT, *extra))


def test_an_experiment_without_a_model_endpoint_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("QWEN_BASE_URL", raising=False)
    endpointless = argv(tmp_path)[:-2]

    with pytest.raises(SystemExit):
        run.parse_args(endpointless)

    assert usage_error(capsys) == "Set QWEN_BASE_URL or --model-endpoint"


def test_an_experiment_without_an_explicit_task_list_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A cohort is never expanded to the whole tasks directory without a word."""
    selectionless = [word for word in argv(tmp_path) if word not in ("--task", "fixture")]

    with pytest.raises(SystemExit):
        run.parse_args(selectionless)

    assert (
        usage_error(capsys)
        == "Explicit --task or --selection is required; subsets are never silently expanded"
    )


@pytest.mark.parametrize("selection", [["b", "c"], {"hard-51": ["b", "c"]}], ids=["a-list", "a-named-list"])
def test_a_selection_file_adds_its_tasks_after_the_named_ones(tmp_path: Path, selection: object) -> None:
    (tmp_path / "selection.json").write_text(json.dumps(selection))

    args = run.parse_args(argv(tmp_path, "--selection", str(tmp_path / "selection.json")))

    assert args.tasks == ["fixture", "b", "c"]


# --- the manifest -------------------------------------------------------------------------------


def test_the_manifest_records_every_setting_that_the_results_depend_on(tmp_path: Path) -> None:
    """It names the reviewer's own endpoint only as a flag, and never a key."""
    args = runner_args(
        tmp_path,
        "--method",
        "sprout",
        "--review-model",
        "critic",
        "--review-endpoint",
        "http://critic.test/v1",
        "--sprout-rounds",
        "2",
        "--sprout-branches",
        "3",
    )

    assert run.manifest_for(args) == {
        "schema_version": 1,
        "dataset": "deepswe",
        "dataset_digest": "sha256:fixture",
        "tasks": ["fixture"],
        "model": "qwen3.8-27b",
        "reasoning_effort": "high",
        "agent": "mini-swe-agent==2.4.6",
        "branching": "assistant-turn",
        "baseline_rollouts": 4,
        "max_total_method_rollouts": 7,
        "branch_caps": [3, 3],
        "branch_count_mode": "adaptive",
        "prompts_sha256": DEFAULT_PROMPTS.sha256,
        "structured_review": False,
        "max_turns": 300,
        "max_output_tokens": 64000,
        "prompt_budget_chars": 400000,
        "command_timeout": 600,
        "image_env": True,
        "image_user": True,
        "review_timeout": 3600,
        "review": {"model": "critic", "own_endpoint": True, "effort": "high", "output_tokens": 64000},
        "runstore_url": "http://127.0.0.1:18110",
        "profile": "mini",
        "grade_profile": "grade",
        "review_profile": "review",
        "workers": 16,
        "source_commit": "0" * 40,
        "method": "sprout",
        "baseline_root": None,
    }


@pytest.mark.parametrize("flag, kept", PROTOCOL_FLAGS)
def test_the_manifest_records_each_part_of_the_harbor_protocol_that_a_flag_turns_off(
    tmp_path: Path, flag: str, kept: tuple[bool, bool]
) -> None:
    manifest = run.manifest_for(runner_args(tmp_path, flag))

    assert (manifest["image_user"], manifest["image_env"]) == kept


def test_only_a_judged_run_records_its_leak_judge(tmp_path: Path) -> None:
    judge = ["--leak-judge", "--leak-judge-resamples", "3", "--leak-judge-threshold", "0.5"]
    judged = run.manifest_for(runner_args(tmp_path, *SPROUT, *judge))
    unjudged = run.manifest_for(runner_args(tmp_path, *SPROUT))

    assert (judged["review"]["leak_judge"], "leak_judge" in unjudged["review"]) == (
        {"resamples": 3, "threshold": 0.5},
        False,
    )


@pytest.mark.parametrize(
    "key, value",
    [
        ("dataset", "terminalbench21"),
        ("dataset_digest", "sha256:e"),
        ("tasks", ["beta"]),
        ("model", "kimi"),
        ("reasoning_effort", "low"),
        ("max_turns", 299),
        ("max_output_tokens", 1),
        ("command_timeout", 60),
        ("runstore_url", "http://127.0.0.1:1"),
        ("profile", "mini-microvm"),
        ("grade_profile", "grade-microvm"),
        ("baseline_rollouts", 1),
        ("image_user", False),
        ("image_env", False),
    ],
)
def test_a_branch_method_must_share_each_setting_of_its_baseline(
    tmp_path: Path, key: str, value: object
) -> None:
    """A key that drops out of SHARED_BASELINE_KEYS lets a branch method continue an incomparable baseline."""
    manifest = run.manifest_for(runner_args(tmp_path))

    assert run.shared_baseline_mismatch({**manifest, key: value}, manifest) == key


@pytest.mark.parametrize(
    "key, value",
    [
        ("prompts_sha256", "0" * 64),
        ("review", {"model": "kimi"}),
        ("structured_review", True),
        ("workers", 1),
        ("source_commit", "1" * 40),
        ("max_total_method_rollouts", 33),
    ],
)
def test_a_branch_method_may_differ_from_its_baseline_in_a_setting_of_its_own(
    tmp_path: Path, key: str, value: object
) -> None:
    manifest = run.manifest_for(runner_args(tmp_path))

    assert run.shared_baseline_mismatch({**manifest, key: value}, manifest) is None


SHARED = {
    "dataset": "deepswe",
    "dataset_digest": "sha256:x",
    "tasks": ["a", "b"],
    "model": "m",
    "reasoning_effort": "high",
    "max_turns": 300,
    "max_output_tokens": 64000,
    "command_timeout": 60,
    "runstore_url": "http://runstore.test:18110",
    "profile": "mini",
    "grade_profile": "grade",
    "baseline_rollouts": 4,
    "image_user": True,
    "image_env": True,
}
#: A baseline manifest written before the image settings were recorded.
WITHOUT_IMAGE = {key: value for key, value in SHARED.items() if key not in ("image_user", "image_env")}


@pytest.mark.parametrize(
    "prior, manifest, mismatch",
    [
        (SHARED, {**SHARED, "command_timeout": 120, "profile": "mini-microvm"}, "command_timeout"),
        ({key: value for key, value in SHARED.items() if key != "command_timeout"}, SHARED, None),
        (
            {key: value for key, value in SHARED.items() if key != "command_timeout"},
            {**SHARED, "command_timeout": 300},
            "command_timeout",
        ),
        ({key: value for key, value in SHARED.items() if key != "baseline_rollouts"}, SHARED, None),
        (
            {key: value for key, value in SHARED.items() if key != "baseline_rollouts"},
            {**SHARED, "baseline_rollouts": 1},
            "baseline_rollouts",
        ),
        (WITHOUT_IMAGE, SHARED, None),
        (WITHOUT_IMAGE, {**SHARED, "image_user": False}, "image_user"),
        (WITHOUT_IMAGE, {**SHARED, "image_env": False}, "image_env"),
    ],
    ids=[
        "the-first-key-in-order",
        "old-baseline-60-s-commands",
        "old-baseline-other-commands",
        "old-baseline-four-rollouts",
        "old-baseline-one-rollout",
        "old-baseline-image-user-and-env",
        "old-baseline-method-as-root",
        "old-baseline-method-without-image-env",
    ],
)
def test_a_branch_method_continues_only_a_baseline_of_the_same_conditions(
    prior: dict[str, Any], manifest: dict[str, Any], mismatch: str | None
) -> None:
    """A baseline manifest written before a key existed ran with 60 s commands and four rollouts.

    Its commands ran as the image's USER with the image's ENV, as Harbor trials run them."""
    assert run.shared_baseline_mismatch(prior, manifest) == mismatch


# --- main ---------------------------------------------------------------------------------------


def test_an_experiment_without_a_run_store_token_fails_before_it_writes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluator: FakeEvaluate
) -> None:
    tasks_dir(tmp_path, "fixture")
    monkeypatch.delenv("SPROUT_RUNSTORE_TOKEN", raising=False)

    with pytest.raises(ValueError, match=r"^\$SPROUT_RUNSTORE_TOKEN holds no Run Store token$"):
        run.main(argv(tmp_path, "--method", "baseline"))

    assert (evaluator.commands, (tmp_path / "out").exists()) == ([], False)


@pytest.mark.parametrize("method", ["all", "bpo", "shepherd"])
def test_an_experiment_with_a_selector_fails_before_it_writes_anything_without_the_model_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, evaluator: FakeEvaluate, method: str
) -> None:
    """The bpo and shepherd selectors call the model from this host.

    The rollouts read the key on the workers."""
    tasks_dir(tmp_path, "fixture")
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "fixture-token")
    monkeypatch.delenv("UNSET_FIXTURE_KEY", raising=False)
    options = [
        "--method",
        method,
        "--model-key-env",
        "UNSET_FIXTURE_KEY",
        "--baseline-root",
        str(tmp_path / "base"),
    ]

    with pytest.raises(ValueError, match="^Model key environment variable UNSET_FIXTURE_KEY is unset$"):
        run.main(argv(tmp_path, *options))

    assert (evaluator.commands, (tmp_path / "out").exists()) == ([], False)


def test_a_plan_needs_no_credential_and_runs_nothing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    evaluator: FakeEvaluate,
) -> None:
    tasks_dir(tmp_path, "fixture")
    monkeypatch.delenv("SPROUT_RUNSTORE_TOKEN", raising=False)

    code = run.main(argv(tmp_path, "--method", "all", "--plan"))

    plan = json.loads(capsys.readouterr().out)
    assert (code, sorted(plan["example_commands"]), plan["manifest"]["method"]) == (
        0,
        ["baseline", "bpo", "shepherd", "sprout"],
        "all",
    )
    assert (evaluator.commands, (tmp_path / "out").exists()) == ([], False)


def test_an_experiment_with_uncommitted_source_changes_is_refused(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate, git: FakeGit
) -> None:
    """The manifest records the commit, so the source must be that commit."""
    tasks_dir(tmp_path, "fixture")
    git.dirty = True

    with pytest.raises(ValueError, match="^Commit source changes before running$"):
        run.main(argv(tmp_path, "--method", "baseline"))

    assert evaluator.commands == []


def test_main_runs_every_task_and_writes_the_manifest_the_preparation_and_the_metrics(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate
) -> None:
    tasks_dir(tmp_path, "fixture")

    code = run.main(argv(tmp_path))

    out = tmp_path / "out"
    assert (
        code,
        run.read(out / "preparation.json")["tasks"][0]["name"],
        run.read(out / "manifest-baseline.json")["source_commit"],
        run.read(out / "metrics-baseline.json")["schema_version"],
        (out / ".baseline.lock").exists(),
    ) == (
        0,
        "org/fixture",
        "a" * 40,
        2,
        False,
    )


def test_main_returns_2_when_a_task_did_not_complete(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate
) -> None:
    tasks_dir(tmp_path, "fixture")
    evaluator.exit_codes = {"attempt-1": 1}

    assert run.main(argv(tmp_path)) == 2


def test_a_root_resumes_under_the_settings_and_the_commit_that_wrote_it(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate
) -> None:
    """Each stage that completed is read again, not run again."""
    tasks_dir(tmp_path, "fixture")
    run.main(argv(tmp_path))

    code = run.main(argv(tmp_path))

    assert (code, len(evaluator.commands)) == (0, 4)


def test_a_root_refuses_to_resume_under_another_commit_and_rewrites_nothing(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate, git: FakeGit
) -> None:
    tasks_dir(tmp_path, "fixture")
    run.main(argv(tmp_path))
    git.commit = "b" * 40

    with pytest.raises(ValueError, match="^Experiment configuration changed; use a fresh output$"):
        run.main(argv(tmp_path))

    assert run.read(tmp_path / "out" / "manifest-baseline.json")["source_commit"] == "a" * 40


@pytest.mark.parametrize(
    "changed",
    [["--max-turns", "100"], ["--no-image-user"], ["--no-image-env"]],
    ids=["max-turns", "image-user", "image-env"],
)
def test_main_refuses_to_resume_a_root_under_a_changed_configuration(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate, changed: list[str]
) -> None:
    tasks_dir(tmp_path, "fixture")
    run.main(argv(tmp_path))

    with pytest.raises(ValueError, match="^Experiment configuration changed; use a fresh output$"):
        run.main(argv(tmp_path, *changed))


def test_a_branch_method_refuses_a_shared_baseline_of_other_conditions(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate
) -> None:
    tasks_dir(tmp_path, "fixture")
    run.main([*argv(tmp_path, "--method", "baseline"), "--output", str(tmp_path / "base")])
    options = ["--method", "bpo", "--baseline-root", str(tmp_path / "base"), "--command-timeout", "300"]

    with pytest.raises(ValueError, match="^Shared baseline configuration mismatch: command_timeout$"):
        run.main(argv(tmp_path, *options))


def test_a_sprout_run_with_review_settings_of_its_own_continues_a_baseline_run_without_them(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate
) -> None:
    """A baseline has no reviewer. The reviewer's prompts, JSON mode and judge are no shared setting."""
    tasks_dir(tmp_path, "fixture")
    run.main(argv(tmp_path, "--output", str(tmp_path / "base")))
    prompts = edited_prompts(tmp_path / "prompts.yaml")
    review = ["--prompts", str(prompts), "--structured-review", "--leak-judge"]

    code = run.main(argv(tmp_path, *SPROUT, "--baseline-root", str(tmp_path / "base"), *review))

    manifest = run.read(tmp_path / "out" / "manifest-sprout.json")
    assert (code, evaluator.stages()[-1], manifest["structured_review"], manifest["prompts_sha256"]) == (
        0,
        "fixture/sprout",
        True,
        hashlib.sha256(prompts.read_bytes()).hexdigest(),
    )


def test_the_preparation_manifest_pins_each_task_as_the_evaluator_checks_it(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate
) -> None:
    tasks = tasks_dir(tmp_path, "fixture")

    run.main(argv(tmp_path, "--method", "baseline"))

    assert run.read(tmp_path / "out" / "preparation.json") == {
        "dataset": "deepswe",
        "dataset_digest": run.task_digest(tasks, ["fixture"]),
        "tasks": [
            {
                "name": "org/fixture",
                "path": str(tasks / "fixture"),
                "task_toml_sha256": hashlib.sha256(
                    (tasks / "fixture" / "task.toml").read_bytes()
                ).hexdigest(),
            }
        ],
    }


def test_a_task_without_a_task_table_is_pinned_under_the_name_that_load_task_gives_it(
    tmp_path: Path, credentials: None, evaluator: FakeEvaluate
) -> None:
    """``load_task`` names such a task after its directory, and the evaluator checks the pin by that name."""
    shutil.copytree(HARBOR_SHELL_TASK, tmp_path / "tasks" / "fixture")

    run.main(argv(tmp_path, "--method", "baseline"))

    assert run.read(tmp_path / "out" / "preparation.json")["tasks"][0]["name"] == "fixture"


# --- the stages of a task -----------------------------------------------------------------------


def test_a_task_whose_first_rollout_resolves_runs_all_baseline_rollouts_and_no_method(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    """Only the first rollout takes a snapshot after each turn: it is the one that methods branch from."""
    evaluator.resolved = {"attempt-1": [True]}

    state = run.run_task(runner_args(tmp_path, "--method", "all"), "fixture")

    assert evaluator.stages() == [f"fixture/attempt-{index}" for index in (1, 2, 3, 4)]
    assert [option(command, "--checkpoint-mode") for command in evaluator.commands] == [
        "disk_only",
        "none",
        "none",
        "none",
    ]
    assert (state["complete"], state["initial_resolved"], state["methods"]) == (
        True,
        True,
        {
            method: {"status": "skipped_initial_success", "attempts": []}
            for method in ("sprout", "bpo", "shepherd")
        },
    )


def test_an_initial_failure_is_retained_and_each_method_continues_it(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    """Each method's attempts leave out its first row: the shared parent, which the baseline counts."""
    evaluator.resolved = {
        "attempt-1": [False],
        "attempt-2": [True],
        "sprout": [False, True],
        "bpo": [True],
        "shepherd": [False],
    }
    parent = tmp_path / "out" / "tasks" / "fixture" / "baseline" / "attempt-1"

    state = run.run_task(runner_args(tmp_path, "--method", "all"), "fixture")

    assert evaluator.stages()[4:] == ["fixture/sprout", "fixture/bpo", "fixture/shepherd"]
    assert [option(command, "--parent-output") for command in evaluator.commands[4:]] == [str(parent)] * 3
    assert [len(state["methods"][method]["attempts"]) for method in ("sprout", "bpo", "shepherd")] == [
        2,
        1,
        1,
    ]
    assert (
        state["complete"],
        state["initial_resolved"],
        (parent / "retained-checkpoints.json").is_file(),
    ) == (
        True,
        False,
        True,
    )


def test_a_stage_that_fails_leaves_the_task_incomplete_and_never_drops_the_retained_parent(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    evaluator.exit_codes = {"sprout": 3}
    parent = tmp_path / "out" / "tasks" / "fixture" / "baseline" / "attempt-1"

    state = run.run_task(runner_args(tmp_path, "--method", "sprout"), "fixture")

    assert (state["complete"], state.get("status"), state.get("error")) == (
        False,
        "infrastructure_error",
        "Stage exited 3; see sprout.log",
    )
    assert (parent / "retained-checkpoints.json").is_file()


def test_a_failed_parent_without_a_branch_point_is_recorded_and_the_baseline_still_runs_four_rollouts(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    """The baseline keeps its fixed count of rollouts."""
    evaluator.pointless = {"attempt-1"}

    state = run.run_task(runner_args(tmp_path), "fixture")

    assert (evaluator.stages(), state.get("retention_error"), state["complete"]) == (
        [f"fixture/attempt-{index}" for index in (1, 2, 3, 4)],
        "Initial failure has no restorable assistant-turn checkpoint",
        True,
    )


def test_a_branch_method_without_a_retained_parent_fails_the_task_without_raising(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    """The state on disk is the state that ``run_task`` returns."""
    evaluator.pointless = {"attempt-1"}

    state = run.run_task(runner_args(tmp_path, "--method", "sprout"), "fixture")

    on_disk = run.read(tmp_path / "out" / "tasks" / "fixture" / "sprout.task.json")
    assert (state["complete"], state.get("status"), state.get("error"), on_disk) == (
        False,
        "infrastructure_error",
        "Shared failure lacks retained-checkpoints.json",
        state,
    )


def test_one_baseline_rollout_is_the_parent_that_a_branch_method_continues(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    evaluator.resolved = {"attempt-1": [False], "sprout": [True]}

    state = run.run_task(runner_args(tmp_path, "--method", "sprout", "--baseline-rollouts", "1"), "fixture")

    assert evaluator.stages() == ["fixture/attempt-1", "fixture/sprout"]
    assert (
        state["complete"],
        len(state["baseline"]["attempts"]),
        state["methods"]["sprout"]["attempts"][0]["resolved"],
    ) == (
        True,
        1,
        True,
    )


def shared_baseline(tmp_path: Path, evaluator: FakeEvaluate) -> Path:
    """The root of a finished baseline experiment on the task ``fixture``, without its stage receipts.

    A branch method reads the stages' summaries, so it does not need their receipts."""
    run.run_task(runner_args(tmp_path, "--method", "baseline", "--output", str(tmp_path / "base")), "fixture")
    for receipt in (tmp_path / "base" / "tasks" / "fixture" / "baseline").glob("*.stage.json"):
        receipt.unlink()
    evaluator.commands.clear()
    return tmp_path / "base"


def test_a_shared_baseline_is_read_and_not_run_again(tmp_path: Path, evaluator: FakeEvaluate) -> None:
    shared = ["--method", "bpo", "--baseline-root", str(shared_baseline(tmp_path, evaluator))]

    state = run.run_task(runner_args(tmp_path, *shared), "fixture")

    assert (state.get("error"), evaluator.stages(), state["complete"]) == (None, ["fixture/bpo"], True)
    assert len(state["baseline"]["attempts"]) == 4


def test_a_shared_baseline_stage_of_more_than_one_rollout_is_refused(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    """Each baseline stage of the protocol is one independent rollout."""
    base = shared_baseline(tmp_path, evaluator)
    summary = base / "tasks" / "fixture" / "baseline" / "attempt-2" / "summary.json"
    rows = run.read(summary)["attempts"]
    run.write(summary, {"attempts": [*rows, *rows], "complete": True})

    state = run.run_task(runner_args(tmp_path, "--method", "bpo", "--baseline-root", str(base)), "fixture")

    assert (evaluator.stages(), state["complete"], state.get("error")) == (
        [],
        False,
        "Shared baseline must contain 4 independent single-rollout stages",
    )


# --- one stage ----------------------------------------------------------------------------------


def test_a_completed_stage_is_reused_without_running_again(tmp_path: Path, evaluator: FakeEvaluate) -> None:
    args = runner_args(tmp_path)
    first = run.stage(args, "fixture", tmp_path / "out" / "attempt-1", "baseline")

    again = run.stage(args, "fixture", tmp_path / "out" / "attempt-1", "baseline")

    assert (again, len(evaluator.commands)) == (first, 1)


def test_the_receipt_of_a_stage_records_its_run(tmp_path: Path, evaluator: FakeEvaluate) -> None:
    run.stage(runner_args(tmp_path), "fixture", tmp_path / "out" / "attempt-1", "baseline")

    receipt = run.read(tmp_path / "out" / "attempt-1.stage.json")
    assert {
        key: receipt[key] for key in ("task", "method", "complete", "output", "source_commit", "exit_code")
    } == {
        "task": "fixture",
        "method": "baseline",
        "complete": True,
        "output": str(tmp_path / "out" / "attempt-1"),
        "source_commit": "0" * 40,
        "exit_code": 0,
    }
    assert receipt["seconds"] == receipt["finished_at"] - receipt["started_at"]


def test_a_stage_writes_the_output_of_its_evaluator_to_a_log_beside_its_output(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    run.stage(runner_args(tmp_path), "fixture", tmp_path / "out" / "attempt-1", "baseline")

    assert (tmp_path / "out" / "attempt-1.log").read_text() == "evaluate attempt-1\n"


def test_a_stage_that_an_earlier_run_left_incomplete_needs_a_fresh_root(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    """Retrying it would erase the history of the failed execution."""
    evaluator.exit_codes = {"attempt-1": 2}
    args = runner_args(tmp_path)
    with pytest.raises(RuntimeError, match="^Stage exited 2; see attempt-1.log$"):
        run.stage(args, "fixture", tmp_path / "out" / "attempt-1", "baseline")
    evaluator.exit_codes = {}

    with pytest.raises(RuntimeError, match="choose a fresh experiment root to retry$"):
        run.stage(args, "fixture", tmp_path / "out" / "attempt-1", "baseline")

    receipt = run.read(tmp_path / "out" / "attempt-1.stage.json")
    assert (receipt["complete"], receipt.get("exit_code"), receipt.get("error_type")) == (
        False,
        2,
        "RuntimeError",
    )


def test_a_stage_whose_output_holds_files_of_another_run_is_refused(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    (tmp_path / "out" / "attempt-1").mkdir(parents=True)
    (tmp_path / "out" / "attempt-1" / "summary.json").write_text("{}")

    with pytest.raises(FileExistsError):
        run.stage(runner_args(tmp_path), "fixture", tmp_path / "out" / "attempt-1", "baseline")

    assert evaluator.commands == []


@pytest.mark.parametrize(
    "method, resolved, message",
    [
        ("baseline", [False, False], "^Baseline stage must grade exactly one independent rollout$"),
        ("sprout", [], "^sprout produced no graded continuation for a failed parent$"),
        ("bpo", [False] * 8, "^Method exceeded total rollout allowance$"),
    ],
    ids=["a-baseline-of-two", "a-method-without-continuation", "over-budget"],
)
def test_a_stage_whose_attempts_break_the_protocol_fails(
    tmp_path: Path, evaluator: FakeEvaluate, method: str, resolved: list[bool], message: str
) -> None:
    """The parent row of a branch stage counts in its budget of 8."""
    run.run_task(runner_args(tmp_path, "--method", "baseline", "--baseline-rollouts", "1"), "fixture")
    evaluator.resolved = {method: resolved}
    parent = tmp_path / "out" / "tasks" / "fixture" / "baseline" / "attempt-1"

    with pytest.raises(ValueError, match=message):
        run.stage(
            runner_args(tmp_path, "--baseline-rollouts", "1"),
            "fixture",
            tmp_path / "out" / method,
            method,
            parent,
        )


@pytest.mark.parametrize(
    "summary, message",
    [
        ({"attempts": [{"resolved": False}], "complete": False}, "^Incomplete evaluation$"),
        ({"attempts": [{"resolved": False}], "branch_schedule_complete": False}, "^Incomplete evaluation$"),
        ({"attempts": []}, "^Incomplete evaluation$"),
        (
            {"attempts": [{"status": "error", "resolved": False}]},
            "^Actor or verifier infrastructure failure$",
        ),
        (
            {"attempts": [{"resolved": False, "grading_error": "no reward"}]},
            "^Actor or verifier infrastructure failure$",
        ),
        (
            {"attempts": [{"resolved": False, "grade": {"error": "lost"}}]},
            "^Actor or verifier infrastructure failure$",
        ),
        (
            {"attempts": [{"resolved": False, "grade": {"verifier_artifact_error": "x"}}]},
            "^Actor or verifier infrastructure failure$",
        ),
        ({"attempts": [{"resolved": None}]}, "^Missing binary verdict$"),
        ({"attempts": [{"status": "completed", "resolved": 1}]}, "^Missing binary verdict$"),
    ],
    ids=[
        "incomplete",
        "schedule-cut-short",
        "no-attempt",
        "actor-error",
        "grading-error",
        "grade-error",
        "verifier-artifact-error",
        "no-verdict",
        "a-number-is-no-verdict",
    ],
)
def test_a_stage_whose_attempts_are_not_all_graded_verdicts_is_no_result(
    tmp_path: Path, summary: dict[str, Any], message: str
) -> None:
    """An infrastructure failure is incomplete, not an ordinary negative reward."""
    run.write(tmp_path / "summary.json", summary)

    with pytest.raises(ValueError, match=message):
        run.attempts(tmp_path)


def test_a_rollout_that_ran_out_of_time_is_still_a_graded_verdict(tmp_path: Path) -> None:
    run.write(
        tmp_path / "summary.json", {"attempts": [{"status": "timeout", "resolved": False}], "complete": True}
    )

    assert run.attempts(tmp_path) == [{"status": "timeout", "resolved": False}]


# --- the retained parent ------------------------------------------------------------------------


def test_a_retained_parent_records_its_branch_points_and_its_snapshots(
    tmp_path: Path, evaluator: FakeEvaluate
) -> None:
    """A future strategy restores these points, so the record names each native prefix and each snapshot."""
    output = tmp_path / "out" / "attempt-1"
    run.stage(runner_args(tmp_path), "fixture", output, "baseline")
    journal = Path(run.read(output / "summary.json")["attempts"][0]["journal"])

    run.retain_parent(output)

    record = run.read(output / "retained-checkpoints.json")
    assert {
        key: record[key]
        for key in ("schema_version", "mode", "journal", "journal_sha256", "backend", "store")
    } == {
        "schema_version": 1,
        "mode": "disk_only",
        "journal": str(journal),
        "journal_sha256": hashlib.sha256(journal.read_bytes()).hexdigest(),
        "backend": "podman",
        "store": "/path/to/store",
    }
    assert [(point["checkpoint"]["step"], point["native"]["cut"]) for point in record["points"]] == [
        (1, "turn-1"),
        (2, "turn-2"),
        (3, "turn-3"),
    ]
    assert record["snapshot_ids"] == sorted(snapshot_id("attempt-1-0", step) for step in (1, 2, 3))


def test_snapshot_ids_are_read_from_the_journals_in_the_output_and_the_ones_its_summary_names(
    tmp_path: Path,
) -> None:
    """A tool output keeps U+2028 and U+0085 verbatim. They must not split a journal record.

    A capture without a snapshot id names no snapshot."""
    text = "before 1 after\u00852\u0085tail"
    inside = tmp_path / "out" / "trial" / "trajectory.jsonl"
    inside.parent.mkdir(parents=True)
    inside.write_text(
        json.dumps(
            {"type": "checkpoint.captured", "snapshot_id": "snap-inside", "note": text}, ensure_ascii=False
        )
        + "\n"
    )
    (tmp_path / "out" / "numbers.jsonl").write_text("1\n2\nnot json\n")
    named = tmp_path / "artifacts" / "trajectory.jsonl"
    named.parent.mkdir()
    named.write_text(
        json.dumps({"type": "checkpoint.captured", "snapshot_id": "snap-named"})
        + "\n"
        + json.dumps({"type": "checkpoint.captured"})
        + "\n"
    )
    run.write(tmp_path / "out" / "summary.json", {"attempts": [{"journal": str(named)}]})

    assert run.snapshot_ids(tmp_path / "out") == {"snap-inside", "snap-named"}


# --- the task digest ----------------------------------------------------------------------------


def test_the_task_digest_changes_with_any_task_file_but_not_with_git_or_bytecode(tmp_path: Path) -> None:
    tasks = tasks_dir(tmp_path, "a", "b")
    before = run.task_digest(tasks, ["a", "b"])
    (tasks / "a" / ".git").mkdir()
    (tasks / "a" / ".git" / "HEAD").write_text("ref: refs/heads/main\n")
    (tasks / "b" / "__pycache__").mkdir()
    (tasks / "b" / "__pycache__" / "x.pyc").write_bytes(b"\0")
    unchanged = run.task_digest(tasks, ["b", "a"])
    (tasks / "b" / "tests" / "test.sh").write_text("exit 1\n")

    changed = run.task_digest(tasks, ["a", "b"])

    assert (unchanged == before, changed == before) == (True, False)


def write_task_files(tasks: Path, name: str, files: dict[str, str]) -> None:
    """Write each of ``files``, a text by its path, into the task ``name`` of ``tasks``."""
    for relative, text in files.items():
        path = tasks / name / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text)


def test_the_task_digest_ignores_git_caches_and_tasks_outside_the_selection(tmp_path: Path) -> None:
    write_task_files(tmp_path, "alpha", {"task.toml": "[task]\n"})
    before = run.task_digest(tmp_path, ["alpha"])
    write_task_files(tmp_path, "alpha", {".git/HEAD": "ref\n", "__pycache__/x.pyc": "x"})
    write_task_files(tmp_path, "beta", {"task.toml": "[task]\n"})

    assert run.task_digest(tmp_path, ["alpha"]) == before


def test_the_task_digest_of_a_task_without_a_task_toml_fails(tmp_path: Path) -> None:
    (tmp_path / "tasks" / "a").mkdir(parents=True)

    with pytest.raises(FileNotFoundError):
        run.task_digest(tmp_path / "tasks", ["a"])

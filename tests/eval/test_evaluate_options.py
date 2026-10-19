"""The options of ``python -m sprout.eval.evaluate``: what it refuses, its defaults, the summary's settings.

Each refusal happens before any job exists. None of these tests needs a Run Store.
"""

import hashlib
from pathlib import Path

import pytest
import yaml

from sprout.contract.harbor_task import load_task
from sprout.eval import evaluate
from sprout.worker.jobs.review.prompts import DEFAULT_PROMPTS, DEFAULT_PROMPTS_PATH, load_prompts
from tests.support.paths import HARBOR_SHELL_TASK

#: The options that every evaluation needs. None of the paths is read before a job would exist.
REQUIRED = [
    "--task-dir",
    "/path/to/task",
    "--dataset-digest",
    "sha256:pinned",
    "--preparation-manifest",
    "/path/to/prep.json",
    "--output",
    "/path/to/out",
    "--model",
    "policy",
    "--model-endpoint",
    "http://model.test/v1",
    "--model-key-env",
    "MODEL_KEY",
]
BASELINE = ["--method", "bpo", "--parent-output", "/path/to/baseline"]
SCHEDULE = "Invalid branch schedule, rollout budget or reviewer attempts"
JUDGE = ["--leak-judge"]


def usage_error(capsys: pytest.CaptureFixture[str]) -> str:
    """The message of the error that argparse printed last, without the program name."""
    return capsys.readouterr().err.splitlines()[-1].split(": error: ", 1)[1]


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--rounds", "-1"], SCHEDULE),
        (["--branches", "4", "0"], SCHEDULE),
        (["--reviewer-max-attempts", "0"], SCHEDULE),
        (["--max-rollouts", "0"], SCHEDULE),
        (["--max-rollouts", "129"], SCHEDULE),
        (["--bpo-top-k", "0"], SCHEDULE),
        (["--bpo-top-k", "21"], SCHEDULE),
        (
            [*BASELINE, "--max-rollouts", "9"],
            "bpo and shepherd allow at most 8 rollouts, the shared parent included",
        ),
        (
            ["--method", "shepherd", "--parent-output", "/path/to/baseline", "--max-rollouts", "9"],
            "bpo and shepherd allow at most 8 rollouts, the shared parent included",
        ),
        (
            ["--prompt-budget-chars", "49999"],
            "--prompt-budget-chars below 50000 leaves no room for the task and a transcript",
        ),
        (["--model-key-env", "MODEL-KEY"], "--model-key-env must name one environment variable"),
        (["--review-key-env", "1KEY"], "--review-key-env must name one environment variable"),
        (["--runstore-token-env", "RUN STORE"], "--runstore-token-env must name one environment variable"),
        (["--leak-judge-key-env", ""], "--leak-judge-key-env must name one environment variable"),
        (["--method", "bpo"], "bpo and shepherd branch from a recorded baseline: pass --parent-output"),
        (["--method", "shepherd"], "bpo and shepherd branch from a recorded baseline: pass --parent-output"),
        (
            ["--command-timeout", "0"],
            "--command-timeout, --review-timeout and --poll-interval must be positive",
        ),
        (
            ["--review-timeout", "0"],
            "--command-timeout, --review-timeout and --poll-interval must be positive",
        ),
        (
            ["--poll-interval", "0"],
            "--command-timeout, --review-timeout and --poll-interval must be positive",
        ),
        (
            [*JUDGE, "--method", "baseline"],
            "--leak-judge checks reviewer-written turns: --method sprout with assistant-turn guidance",
        ),
        (
            [*JUDGE, "--branch-guidance", "user-hint"],
            "--leak-judge checks reviewer-written turns: --method sprout with assistant-turn guidance",
        ),
        (
            [*JUDGE, "--leak-judge-resamples", "-1"],
            "--leak-judge-resamples must be >= 0 and --leak-judge-threshold in (0, 1]",
        ),
        (
            [*JUDGE, "--leak-judge-threshold", "0"],
            "--leak-judge-resamples must be >= 0 and --leak-judge-threshold in (0, 1]",
        ),
        (
            [*JUDGE, "--leak-judge-threshold", "1.01"],
            "--leak-judge-resamples must be >= 0 and --leak-judge-threshold in (0, 1]",
        ),
    ],
)
def test_an_option_that_no_evaluation_can_run_is_refused(
    capsys: pytest.CaptureFixture[str], extra: list[str], message: str
) -> None:
    with pytest.raises(SystemExit) as exit_info:
        evaluate.parse_args([*REQUIRED, *extra])

    assert (exit_info.value.code, usage_error(capsys)) == (2, message)


@pytest.mark.parametrize(
    "extra",
    [
        ["--rounds", "0"],
        ["--branches", "1"],
        ["--reviewer-max-attempts", "1"],
        ["--max-rollouts", "1"],
        ["--max-rollouts", "128"],
        ["--bpo-top-k", "1"],
        ["--bpo-top-k", "20"],
        [*BASELINE, "--max-rollouts", "8"],
        ["--prompt-budget-chars", "50000"],
        [*JUDGE, "--leak-judge-resamples", "0", "--leak-judge-threshold", "1"],
    ],
)
def test_the_limits_of_each_option_are_accepted(extra: list[str]) -> None:
    evaluate.parse_args([*REQUIRED, *extra])


def test_an_output_that_the_os_empties_on_reboot_needs_volatile_ok(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """The summary and the group documents under --output are the evaluation's only record of its jobs."""
    volatile = [*REQUIRED, "--output", "/tmp/sprout-evaluation"]

    with pytest.raises(SystemExit):
        evaluate.parse_args(volatile)
    refusal = usage_error(capsys)
    accepted = evaluate.parse_args([*volatile, "--volatile-ok"])

    assert refusal == (
        "refusing: /tmp/sprout-evaluation is under /tmp, which the OS empties on reboot -- a journal "
        "is the only record a run leaves (pass --volatile-ok to override)"
    )
    assert accepted.output == Path("/tmp/sprout-evaluation")


@pytest.mark.parametrize(
    "extra, reviewer",
    [
        (["--reasoning-effort", "high"], ("policy", "http://model.test/v1", "MODEL_KEY", "high")),
        (
            [
                "--reasoning-effort",
                "high",
                "--review-model",
                "critic",
                "--review-endpoint",
                "http://critic.test/v1",
            ],
            ("critic", "http://critic.test/v1", "MODEL_KEY", "high"),
        ),
        (
            ["--review-key-env", "CRITIC_KEY", "--review-effort", "medium"],
            ("policy", "http://model.test/v1", "CRITIC_KEY", "medium"),
        ),
    ],
    ids=["all-from-the-actor", "own-model-and-endpoint", "own-key-and-effort"],
)
def test_the_reviewer_takes_each_actor_setting_that_it_is_not_given(
    extra: list[str], reviewer: tuple[str, str, str, str]
) -> None:
    args = evaluate.parse_args([*REQUIRED, *extra])

    assert (args.review_model, args.review_endpoint, args.review_key_env, args.review_effort) == reviewer


def edited_prompts(path: Path) -> Path:
    """A copy of the packaged prompts file at ``path``, with one prompt changed."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    data["shepherd"]["system"] += "\nEdited for this test."
    path.write_text(yaml.safe_dump(data))
    return path


def test_the_prompts_file_is_loaded_with_the_options_and_the_summary_records_its_hash(tmp_path: Path) -> None:
    prompts = edited_prompts(tmp_path / "prompts.yaml")

    args = evaluate.parse_args([*REQUIRED, "--prompts", str(prompts)])

    assert args.prompts == load_prompts(prompts)
    assert evaluate.initial_state(args, load_task(HARBOR_SHELL_TASK))["prompts"] == {
        "path": str(prompts),
        "sha256": hashlib.sha256(prompts.read_bytes()).hexdigest(),
    }


def test_without_a_prompts_file_the_summary_records_the_packaged_one() -> None:
    args = evaluate.parse_args(REQUIRED)

    assert evaluate.initial_state(args, load_task(HARBOR_SHELL_TASK))["prompts"] == {
        "path": str(DEFAULT_PROMPTS_PATH),
        "sha256": DEFAULT_PROMPTS.sha256,
    }


def test_a_prompts_file_without_a_required_prompt_is_refused(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Each key is required, so a file without the analyst's case prompt fails before any job exists."""
    data = yaml.safe_load(DEFAULT_PROMPTS_PATH.read_text())
    del data["analyst"]["case"]
    (tmp_path / "prompts.yaml").write_text(yaml.safe_dump(data))

    with pytest.raises(SystemExit):
        evaluate.parse_args([*REQUIRED, "--prompts", str(tmp_path / "prompts.yaml")])

    assert usage_error(capsys).startswith("--prompts: ")


def test_a_prompts_file_that_does_not_exist_is_refused(capsys: pytest.CaptureFixture[str]) -> None:
    with pytest.raises(SystemExit):
        evaluate.parse_args([*REQUIRED, "--prompts", "/path/to/absent.yaml"])

    assert usage_error(capsys).startswith("--prompts: ")


def summary_without_id(argv: list[str]) -> dict[str, object]:
    """The summary of an evaluation of the fixture task before its first attempt, without its random id."""
    state = evaluate.initial_state(evaluate.parse_args(argv), load_task(HARBOR_SHELL_TASK))
    del state["evaluation_id"]
    return state


def test_the_summary_records_each_setting_that_the_attempts_depend_on() -> None:
    state = summary_without_id(
        [
            *REQUIRED,
            "--method",
            "baseline",
            "--runstore-url",
            "http://runstore.test:18110",
            "--profile",
            "mini-microvm",
            "--grade-profile",
            "grade-microvm",
            "--command-timeout",
            "60",
            "--no-image-user",
        ]
    )

    assert state == {
        "source_commit": None,
        "affinity": None,
        "task": "shell-task",
        "task_toml_sha256": hashlib.sha256((HARBOR_SHELL_TASK / "task.toml").read_bytes()).hexdigest(),
        "dataset_digest": "sha256:pinned",
        "model": "policy",
        "method": "baseline",
        "runstore": {
            "url": "http://runstore.test:18110",
            "profiles": {"rollout": "mini-microvm", "grade": "grade-microvm", "review": "review"},
        },
        "parent_output": None,
        "branch_guidance": "none",
        "branch_count_mode": None,
        "structured_review": False,
        "prompts": {"path": str(DEFAULT_PROMPTS_PATH), "sha256": DEFAULT_PROMPTS.sha256},
        "branch_caps": [4, 3],
        "rounds": 1,
        "command_timeout": 60,
        "image_env": True,
        "image_user": False,
        "max_rollouts": 8,
        "attempts": [],
        "earlier_runs": [],
        "complete": False,
    }


@pytest.mark.parametrize(
    "extra, review",
    [
        (
            ["--review-model", "critic", "--review-effort", "high"],
            {"model": "critic", "endpoint": "http://model.test/v1", "effort": "high"},
        ),
        (
            [*JUDGE, "--leak-judge-resamples", "3", "--leak-judge-key-env", "TS_KEY"],
            {
                "model": "policy",
                "endpoint": "http://model.test/v1",
                "effort": None,
                "leak_judge": {"threshold": 0.75, "model": "jev-1.13.0", "resamples": 3},
            },
        ),
    ],
    ids=["reviewer", "leak-judge"],
)
def test_a_sprout_summary_records_its_schedule_and_where_its_reviews_ran(
    extra: list[str], review: dict[str, object]
) -> None:
    """The summary names the review's endpoint and the judge's variable, never a key."""
    state = summary_without_id(
        [
            *REQUIRED,
            "--rounds",
            "3",
            "--branches",
            "2",
            "--branch-count-mode",
            "fixed",
            "--structured-review",
            *extra,
        ]
    )

    settings = (
        "branch_guidance",
        "branch_count_mode",
        "structured_review",
        "rounds",
        "branch_caps",
        "review",
    )
    assert [state[key] for key in settings] == ["assistant-turn", "fixed", True, 3, [2], review]


def test_an_evaluation_without_a_run_store_token_fails_before_it_writes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("SPROUT_RUNSTORE_TOKEN", raising=False)

    with pytest.raises(ValueError, match=r"^\$SPROUT_RUNSTORE_TOKEN holds no Run Store token$"):
        evaluate.main([*REQUIRED, "--output", str(tmp_path / "out"), "--volatile-ok"])

    assert not (tmp_path / "out").exists()


def test_a_blank_run_store_token_is_no_token(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "  ")

    with pytest.raises(ValueError, match=r"^\$SPROUT_RUNSTORE_TOKEN holds no Run Store token$"):
        evaluate.main([*REQUIRED, "--output", str(tmp_path / "out"), "--volatile-ok"])


@pytest.mark.parametrize("method", ["bpo", "shepherd"])
def test_a_selector_without_its_model_key_here_fails_before_it_writes_anything(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, method: str
) -> None:
    """The bpo and shepherd selectors call the model from this host, not from the worker."""
    monkeypatch.setenv("SPROUT_RUNSTORE_TOKEN", "fixture-token")
    monkeypatch.delenv("MODEL_KEY", raising=False)
    argv = [*REQUIRED, "--method", method, "--parent-output", "/path/to/baseline"]

    with pytest.raises(ValueError, match="^Model key environment variable MODEL_KEY is unset$"):
        evaluate.main([*argv, "--output", str(tmp_path / "out"), "--volatile-ok"])

    assert not (tmp_path / "out").exists()

"""The podman that every entry point runs, and the errors of a podman or btrfs command."""

from __future__ import annotations

from pathlib import Path

import pytest

from sprout_sandbox.podman_cli import PodmanError, default_podman_bin, run_command

pytestmark = pytest.mark.anyio

#: Where ``ops/setup_podman.sh`` installs podman 6, below the home directory.
INSTALLED = ".local/opt/podman/bin/podman"


def install_podman(home: Path, mode: int | None) -> None:
    """Put a file with ``mode`` where ``ops/setup_podman.sh`` installs podman, unless ``mode`` is None."""
    if mode is None:
        return
    installed = home / INSTALLED
    installed.parent.mkdir(parents=True)
    installed.write_text("#!/bin/sh\n")
    installed.chmod(mode)


def name_podman(monkeypatch: pytest.MonkeyPatch, value: str | None) -> None:
    """Set ``SPROUT_PODMAN_BIN`` to ``value``, or unset it for None."""
    if value is None:
        monkeypatch.delenv("SPROUT_PODMAN_BIN", raising=False)
    else:
        monkeypatch.setenv("SPROUT_PODMAN_BIN", value)


@pytest.mark.parametrize(
    "variable, installed_mode, expected",
    [
        pytest.param("~/elsewhere/podman", None, "{home}/elsewhere/podman", id="variable-with-tilde"),
        pytest.param("/path/to/podman6", 0o755, "/path/to/podman6", id="variable-over-installed"),
        pytest.param("podman", 0o755, "podman", id="variable-names-path-lookup"),
        pytest.param("", 0o755, "{home}/" + INSTALLED, id="empty-variable"),
        pytest.param(None, 0o755, "{home}/" + INSTALLED, id="installed"),
        pytest.param(None, 0o644, "podman", id="installed-not-executable"),
        pytest.param(None, None, "podman", id="nothing"),
    ],
)
def test_the_default_podman_is_the_variable_else_the_installed_one_else_podman_on_path(
    tmp_path, monkeypatch, variable, installed_mode, expected
):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    install_podman(home, installed_mode)
    name_podman(monkeypatch, variable)

    chosen = default_podman_bin()

    assert chosen == expected.format(home=home)


@pytest.mark.parametrize(
    "argv, failing, expected",
    [
        pytest.param(
            ["podman", "frobnicate"],
            (),
            "podman frobnicate failed (exit 2): fake podman: unknown command ['frobnicate']",
            id="one-command-word",
        ),
        pytest.param(
            ["podman", "rm", "-f", "sprout-000000000001"],
            ("rm",),
            "podman rm failed (exit 125): Error: simulated rm failure",
            id="words-end-at-a-flag",
        ),
        pytest.param(
            ["podman", "network", "rm", "sprout-000000000001"],
            (),
            "podman network rm failed (exit 1): Error: unable to find network with name or ID "
            "sprout-000000000001: network not found",
            id="two-command-words",
        ),
        pytest.param(
            ["podman", "container", "exists", "sprout-000000000001"],
            (),
            "podman container exists failed (exit 1): (no stderr)",
            id="no-stderr",
        ),
    ],
)
async def test_a_failed_command_names_the_tool_its_command_words_its_exit_code_and_its_stderr(
    fake_podman, tmp_path, argv, failing, expected
):
    """The pool and the reaper tell one podman refusal from another by the stderr in the message."""
    fake_podman.fail(*failing)

    with pytest.raises(PodmanError) as raised:
        await run_command(argv, timeout=60, cwd=tmp_path)

    assert str(raised.value) == expected


async def test_a_command_that_outlives_its_timeout_fails_and_names_the_command(fake_podman, tmp_path):
    fake_podman.hang("logs")

    with pytest.raises(PodmanError, match=r"^podman logs sprout-000000000001 did not finish within 1 s$"):
        await run_command(["podman", "logs", "sprout-000000000001"], timeout=1, cwd=tmp_path)

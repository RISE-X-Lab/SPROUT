"""The runtime's tool declaration, ``runtime/schema/tools.json``: its content, its drift and where it is read.

``ToolAccess`` refuses each tool and argument that the declaration does not list. So a declaration
that is older than the runtime refuses what the runtime takes, or passes what the runtime ignores.
The drift test asks the runtime source for its declaration with ``go run . --dump-schema``, as
``gate/checks/runtime_tools.py`` does. It skips without ``go``.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from sprout.worker.tools import schema
from sprout.worker.tools.schema import PACKAGED_SCHEMA, RUNTIME_SCHEMA, runtime_tools
from tests.support.paths import REPO

CHECKED_IN = REPO / "runtime" / "schema" / "tools.json"


@pytest.fixture
def dumped_declaration() -> object:
    """What the runtime source declares, from ``go run . --dump-schema`` in ``runtime/``."""
    if shutil.which("go") is None:
        pytest.skip(
            "needs go on PATH, to ask the runtime source what it declares (https://go.dev/doc/install)"
        )
    done = subprocess.run(
        ["go", "run", ".", "--dump-schema"], cwd=REPO / "runtime", capture_output=True, text=True, timeout=600
    )
    assert done.returncode == 0, (
        "`go run . --dump-schema` failed (`go generate ./...` fetches ripgrep):\n%s" % (done.stderr)
    )
    return json.loads(done.stdout)


def test_the_checked_in_declaration_matches_what_the_runtime_declares(dumped_declaration):
    assert json.loads(CHECKED_IN.read_text()) == dumped_declaration, (
        "runtime/schema/tools.json is stale: cd runtime && go run . --dump-schema > schema/tools.json"
    )


def test_the_runtime_serves_its_five_tools_in_order():
    """The tool set is the agent's action space, so a removed or added tool is a protocol change."""
    assert [tool["name"] for tool in runtime_tools()] == [
        "shell",
        "text_editor",
        "grep_files",
        "web_fetch",
        "web_search",
    ]


def declared(name: str) -> dict[str, Any]:
    """The declaration of the tool ``name``."""
    return {tool["name"]: tool for tool in runtime_tools()}[name]


def test_the_shell_tool_takes_the_arguments_that_its_callers_send():
    """mini sends ``command``, ``working_dir`` and ``timeout``. Harbor's exec path bounds its output."""
    shell = declared("shell")

    assert (sorted(shell["inputSchema"]["properties"]), shell["inputSchema"]["required"]) == (
        ["command", "max_output_bytes", "timeout", "working_dir"],
        ["command"],
    )


def test_a_source_checkout_reads_the_declaration_that_the_drift_test_checks():
    assert RUNTIME_SCHEMA == CHECKED_IN


def test_the_wheel_carries_the_declaration_under_the_name_that_an_installed_package_reads():
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))

    included = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]

    assert included["runtime/schema/tools.json"] == "sprout/worker/tools/" + PACKAGED_SCHEMA


def installed_layout(site: Path) -> Path:
    """The wheel's layout under ``site``: ``schema.py``, the declaration beside it, and no ``runtime/``."""
    package = site / "sprout" / "worker" / "tools"
    package.mkdir(parents=True)
    for directory in (site / "sprout", site / "sprout" / "worker", package):
        (directory / "__init__.py").write_text("")
    shutil.copy(Path(schema.__file__), package / "schema.py")
    shutil.copy(CHECKED_IN, package / PACKAGED_SCHEMA)
    return site


def installed_directory(tmp_path: Path) -> Path:
    return installed_layout(tmp_path / "site")


def installed_zip(tmp_path: Path) -> Path:
    """The installed layout as one archive, which Python imports through ``zipimport``."""
    site, archive = installed_layout(tmp_path / "site"), tmp_path / "sprout.whl"
    with zipfile.ZipFile(archive, "w") as handle:
        for path in sorted(site.rglob("*")):
            handle.write(path, path.relative_to(site).as_posix())
    return archive


#: Prints where ``schema`` reads the declaration, and the tools that it reads, for an import root in argv.
PROBE = (
    "import json, sys; sys.path.insert(0, sys.argv[1]); from sprout.worker.tools import schema; "
    "print(json.dumps([str(schema.RUNTIME_SCHEMA), schema.runtime_tools()]))"
)


@pytest.mark.parametrize("install", [installed_directory, installed_zip], ids=["directory", "zip"])
def test_an_install_without_a_source_tree_reads_the_packaged_copy(tmp_path, install):
    """``-I -S`` keeps this checkout and its site-packages off the import path."""
    root = install(tmp_path)

    done = subprocess.run(
        [sys.executable, "-I", "-S", "-c", PROBE, str(root)],
        capture_output=True,
        text=True,
        check=True,
        timeout=60,
    )

    assert json.loads(done.stdout) == [
        str(root / "sprout" / "worker" / "tools" / PACKAGED_SCHEMA),
        runtime_tools(),
    ]

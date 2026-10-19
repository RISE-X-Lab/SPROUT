"""An installed wheel holds the packages under ``src/`` and the data files that they read.

``pyproject.toml`` builds the wheel with hatchling. Its ``packages`` key names the packages, and its
``force-include`` key copies ``resources/`` and the runtime's tool declaration into them. Each test
runs on a copy that follows those two keys, as a medium test, and on the wheel that pip builds, as a
large test marked ``slow``. pip gets the hatchling build backend from PyPI.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path
from typing import Any

import pytest

from tests.support.paths import REPO

SRC = REPO / "src"
RESOURCES = REPO / "resources"
TOOLS_JSON = REPO / "runtime" / "schema" / "tools.json"
#: Puts the installed directory, argv[1], first on sys.path.
PRELUDE = "import json, sys; sys.path.insert(0, sys.argv[1]); "
READ_RESOURCES = (
    "from sprout.resource import resource; "
    "print(json.dumps({name: resource(name).read_text() for name in sys.argv[2:]}))"
)
READ_TOOLS = "from sprout.worker.tools.schema import runtime_tools; print(json.dumps(runtime_tools()))"


def lay_out_as_pyproject_says(site: Path) -> None:
    """Copy into ``site`` what the ``packages`` and ``force-include`` keys put into the wheel."""
    wheel = tomllib.loads((REPO / "pyproject.toml").read_text())["tool"]["hatch"]["build"]["targets"]["wheel"]
    for package in wheel["packages"]:
        shutil.copytree(
            REPO / package, site / Path(package).name, ignore=shutil.ignore_patterns("__pycache__")
        )
    for source, target in wheel["force-include"].items():
        (site / target).parent.mkdir(parents=True, exist_ok=True)
        if (REPO / source).is_dir():
            shutil.copytree(REPO / source, site / target, dirs_exist_ok=True)
        else:
            shutil.copyfile(REPO / source, site / target)


def unpack_the_built_wheel(site: Path, *, dist: Path) -> None:
    """Build the wheel of the checkout with pip, and unpack it into ``site``, as pip installs it."""
    try:
        socket.create_connection(("pypi.org", 443), timeout=10).close()
    except OSError as error:
        pytest.skip("needs PyPI, where pip gets the hatchling build backend: %s" % error)
    argv = [sys.executable, "-m", "pip", "wheel", "--no-deps", "--wheel-dir", str(dist), str(REPO)]
    subprocess.run(argv, check=True, capture_output=True, timeout=600)
    (wheel,) = dist.glob("*.whl")
    zipfile.ZipFile(wheel).extractall(site)


@pytest.fixture(params=["pyproject-layout", pytest.param("built-wheel", marks=pytest.mark.slow)])
def installed(request: pytest.FixtureRequest, tmp_path: Path) -> Path:
    """A directory that holds the installed package, and no source checkout around it."""
    site = tmp_path / "site"
    if request.param == "pyproject-layout":
        lay_out_as_pyproject_says(site)
    else:
        unpack_the_built_wheel(site, dist=tmp_path / "dist")
    return site


def run_installed(site: Path, code: str, *arguments: str) -> Any:
    """What ``code`` prints as JSON in an interpreter that sees the standard library and ``site`` only."""
    argv = [sys.executable, "-I", "-S", "-c", PRELUDE + code, str(site), *arguments]
    run = subprocess.run(argv, cwd=site, capture_output=True, text=True, timeout=60)
    assert run.returncode == 0, run.stderr
    return json.loads(run.stdout)


def checkout_resources() -> dict[str, str]:
    """The text of each file under ``resources/``, by the name that ``resource()`` takes."""
    return {
        str(path.relative_to(RESOURCES)): path.read_text() for path in RESOURCES.rglob("*") if path.is_file()
    }


def python_files(root: Path) -> list[str]:
    return sorted(
        str(path.relative_to(root)) for path in root.rglob("*.py") if "__pycache__" not in path.parts
    )


def test_the_installed_package_reads_each_resource_from_its_own_copy(installed):
    expected = checkout_resources()

    texts = run_installed(installed, READ_RESOURCES, *expected)

    assert texts == expected


def test_the_installed_package_reads_the_runtime_tool_declaration_from_its_own_copy(installed):
    tools = run_installed(installed, READ_TOOLS)

    assert tools == json.loads(TOOLS_JSON.read_text())["tools"]


def test_the_installed_package_holds_each_python_file_under_src(installed):
    """A package under ``src/`` that the ``packages`` key does not name is missing from the wheel."""
    assert python_files(installed) == python_files(SRC)

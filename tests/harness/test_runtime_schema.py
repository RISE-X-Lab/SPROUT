"""The checked-in runtime declaration must match the runtime that produced it.

``runtime/schema/tools.json`` is what ``ToolAccess`` validates every call against
and what tells mini's executor which arguments ``shell`` takes. A hand-written
copy once drifted from the runtime on four of seven tools; the worst case was a
parameter the runtime had dropped, so the model could ask for a bound and not
get one. A stale copy has to fail here, not at a model's first call. CI checks
that this test ran rather than skipped.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tomllib
import zipfile
from pathlib import Path

import pytest

from sprout.worker.tools import schema
from sprout.worker.tools.schema import PACKAGED_SCHEMA, RUNTIME_SCHEMA, runtime_tools

REPO = Path(__file__).resolve().parents[2]


def _runtime_binary() -> "str | None":
    found = shutil.which("sprout-runtime")
    if found:
        return found
    root = Path(__file__).resolve().parents[2]
    candidate = root / "runtime" / "sprout-runtime"
    return str(candidate) if candidate.is_file() else None


def test_the_checked_in_declaration_matches_the_runtime():
    binary = _runtime_binary()
    if binary is None:
        pytest.skip("no sprout-runtime binary; run "
                    "`cd runtime && go generate ./... && go build -o sprout-runtime .`")
    fresh = json.loads(subprocess.run([binary, "--dump-schema"], capture_output=True,
                                      text=True, timeout=60).stdout)
    stored = json.loads(RUNTIME_SCHEMA.read_text())
    assert fresh == stored, (
        "runtime/schema/tools.json is stale; regenerate with "
        "`./sprout-runtime --dump-schema > runtime/schema/tools.json`")


def test_a_source_checkout_reads_the_checked_in_declaration():
    """The check above compares the runtime with what the harness loads, so in
    a checkout that has to be runtime/schema/tools.json itself, not a copy."""
    assert RUNTIME_SCHEMA == REPO / "runtime" / "schema" / "tools.json" == schema.SOURCE_SCHEMA


def test_the_wheel_carries_the_declaration_where_an_install_reads_it():
    """An installed wheel has no runtime/ directory: the build copies the
    checked-in file in beside schema.py, under the name it reads."""
    config = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
    included = config["tool"]["hatch"]["build"]["targets"]["wheel"]["force-include"]
    assert included["runtime/schema/tools.json"] == "sprout/worker/tools/" + PACKAGED_SCHEMA


@pytest.mark.parametrize("layout", ["directory", "zip"])
def test_an_install_without_a_source_tree_reads_the_packaged_copy(tmp_path, layout):
    """The wheel's layout -- schema.py with the declaration beside it and no
    runtime/ above -- installed as files or imported from the archive."""
    site = tmp_path / "site"
    package = site / "sprout" / "worker" / "tools"
    package.mkdir(parents=True)
    for directory in (site / "sprout", site / "sprout" / "worker", package):
        (directory / "__init__.py").write_text("")
    shutil.copy(Path(schema.__file__), package / "schema.py")
    shutil.copy(REPO / "runtime" / "schema" / "tools.json", package / PACKAGED_SCHEMA)
    root = site
    if layout == "zip":
        root = tmp_path / "sprout.whl"
        with zipfile.ZipFile(root, "w") as archive:
            for path in sorted(site.rglob("*")):
                archive.write(path, path.relative_to(site).as_posix())
    probe = ("import json, sys; sys.path.insert(0, sys.argv[1]); "
             "from sprout.worker.tools import schema; "
             "print(json.dumps([schema.SOURCE_SCHEMA.is_file(), str(schema.RUNTIME_SCHEMA), "
             "schema.runtime_tools()]))")
    # -I -S: no PYTHONPATH, no site-packages, so nothing reaches this checkout.
    found, loaded, tools = json.loads(subprocess.run(
        [sys.executable, "-I", "-S", "-c", probe, str(root)],
        capture_output=True, text=True, check=True, timeout=60).stdout)
    assert not found and loaded == str(root / "sprout" / "worker" / "tools" / PACKAGED_SCHEMA)
    assert tools == runtime_tools()


def test_the_runtime_serves_its_five_tools():
    """The tool set is the protocol: adding or removing a tool is a breaking
    change, and this is where it shows up first."""
    assert [t["name"] for t in runtime_tools()] == [
        "shell", "text_editor", "grep_files", "web_fetch", "web_search"]


@pytest.mark.parametrize("parameters", [
    # mini's ToolEnvironment (agents/mini_swe/runtime.py) refuses a `shell` without these.
    pytest.param({"command", "working_dir", "timeout"}, id="minis-executor"),
    # SandboxEnvironment.exec (harbor_plugin/environment.py, inherited by the podman
    # and AgentENV environments) sends this on every verifier and setup command.
    pytest.param({"max_output_bytes"}, id="harbors-exec-path"),
])
def test_the_declaration_is_the_documented_surface(parameters):
    """shell(command, working_dir, timeout, max_output_bytes) is the whole surface
    AGENTS.md documents, each parameter with a consumer; checked where the
    declaration lives, so a runtime change is caught before a run. The exec path
    is why the bound needs pinning: it goes through SandboxSession.execute, not
    ToolAccess's schema check, so the runtime dropping `max_output_bytes` would
    silently unbound every verifier command instead of being refused."""
    shell = {t["name"]: t for t in runtime_tools()}["shell"]
    assert parameters <= set(shell["inputSchema"]["properties"])
    assert set(shell["inputSchema"]["properties"]) == {"command", "working_dir", "timeout", "max_output_bytes"}
    assert shell["inputSchema"]["required"] == ["command"]

"""The README and the docs name files, packages, commands and tools that the repository has.

A test cannot check the prose of a document, but it can look up each name in it. The documents are
the README files and ``docs/*.md``. The decisions in ``docs/decisions/`` record a past state, and
nobody edits them, so no test reads them.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tests.support.paths import REPO

RUNTIME_README = REPO / "runtime" / "README.md"
TOOLS_JSON = REPO / "runtime" / "schema" / "tools.json"
DOCUMENTS = [
    "README.md",
    "gate/README.md",
    "runtime/README.md",
    "examples/qwen/PROTOCOL.md",
    "docs/ARCHITECTURE.md",
    "docs/EVALUATION.md",
    "docs/EXPERIMENTS.md",
    "docs/HARBOR_PLUGIN.md",
    "docs/MODEL_ENDPOINTS.md",
    "docs/RL_DRIVER.md",
    "docs/RUNSTORE.md",
    "docs/SERVICES.md",
]
#: The target of a Markdown link or image, or of an HTML src, srcset or href.
LINK = re.compile(r"\]\(([^)\s]+)\)|(?:src|srcset|href)=\"([^\"]+)\"")
#: A URL scheme, such as https: or mailto:.
SCHEME = re.compile(r"[a-z][a-z0-9+.-]*:")
#: A Python file of this repository, named from the repository root.
PYTHON_PATH = re.compile(r"(?<![\w/.~-])((?:src|ops|gate|examples|tests)/[\w/.-]*\.py)\b")
#: The arguments of a pytest command, to the end of the command.
PYTEST_ARGUMENTS = re.compile(r"-m pytest ([^\n`]*)")
#: A ``python -m`` command of this repository, and its first word when that word is not a path.
COMMAND = re.compile(r"python[\d.]* -m ((?:sprout|ops|examples)[\w.]*)(?: ([a-z][\w-]*)(?=[\s`)]|$))?", re.M)


def text_of(document: str) -> str:
    return (REPO / document).read_text(encoding="utf-8")


def missing_link_targets(document: str) -> list[str]:
    """Each local link target in ``document`` that names nothing, from the directory of the document."""
    targets = [markdown or html for markdown, html in LINK.findall(text_of(document))]
    local = {target.partition("#")[0] for target in targets if not SCHEME.match(target)} - {""}
    return sorted(target for target in local if not (REPO / document).parent.joinpath(target).exists())


def missing_named_paths(document: str) -> list[str]:
    """Each Python file and each pytest test path that ``document`` names and that does not exist."""
    text = text_of(document)
    arguments = [argument for command in PYTEST_ARGUMENTS.findall(text) for argument in command.split()]
    tests = {argument.partition("::")[0] for argument in arguments if argument.startswith("tests/")}
    return sorted(path for path in set(PYTHON_PATH.findall(text)) | tests if not (REPO / path).exists())


def documented_toolset() -> dict[str, dict[str, list[str]]]:
    """Each tool in the runtime/README.md section "Runtime toolset support", with the arguments of its row."""
    text = RUNTIME_README.read_text(encoding="utf-8")
    section = text.split("## Runtime toolset support", 1)[1].split("\n## ", 1)[0]
    rows = re.findall(r"^\| `(\w+)` \| ([^|]*) \|", section, re.M)
    return {name: listed_arguments(cell) for name, cell in rows}


def listed_arguments(cell: str) -> dict[str, list[str]]:
    """The arguments in ``cell``: the required ones before the ``;``, the optional ones after it.

    The values in parentheses, such as the commands of ``text_editor``, are not arguments.
    """
    required, _, optional = re.sub(r"\([^)]*\)", "", cell).partition(";")
    return {
        "required": sorted(re.findall(r"`(\w+)`", required)),
        "optional": sorted(re.findall(r"`(\w+)`", optional)),
    }


def declared_toolset() -> dict[str, dict[str, list[str]]]:
    """Each tool in ``runtime/schema/tools.json``, with its required and its optional arguments."""
    tools = json.loads(TOOLS_JSON.read_text(encoding="utf-8"))["tools"]
    return {tool["name"]: declared_arguments(tool["inputSchema"]) for tool in tools}


def declared_arguments(schema: dict) -> dict[str, list[str]]:
    required = schema.get("required", [])
    return {"required": sorted(required), "optional": sorted(set(schema["properties"]) - set(required))}


def documented_commands() -> list[list[str]]:
    """Each ``python -m`` command of this repository in the documents, as its module and its first word."""
    found = {match for document in DOCUMENTS for match in COMMAND.findall(text_of(document))}
    return sorted([module, word] if word else [module] for module, word in found)


def refusals_of_help(commands: list[list[str]], *, home: Path) -> dict[str, str]:
    """Each command that exits non-zero when ``--help`` follows it, with the last line of its stderr.

    Each command runs from the repository root, with the home directory ``home``.
    """
    environment = {**os.environ, "HOME": str(home)}
    refused = {}
    for command in commands:
        argv = [sys.executable, "-m", *command, "--help"]
        run = subprocess.run(argv, cwd=REPO, env=environment, capture_output=True, text=True, timeout=60)
        if run.returncode != 0:
            refused[" ".join(command)] = (run.stderr.strip().splitlines() or ["exit %d" % run.returncode])[-1]
    return refused


@pytest.mark.parametrize("document", DOCUMENTS)
def test_each_link_in_a_document_names_a_file_that_exists(document):
    assert missing_link_targets(document) == []


@pytest.mark.parametrize("document", DOCUMENTS)
def test_each_python_file_and_test_path_that_a_document_names_exists(document):
    """The docs name test files to run and modules to read. A move or a deletion leaves the name behind."""
    assert missing_named_paths(document) == []


def test_each_test_path_that_the_podman_setup_runs_exists():
    """The setup ends with the live podman tests. A test path that is gone fails it after a good install."""
    assert missing_named_paths("ops/setup_podman.sh") == []


def test_the_runtime_toolset_table_lists_each_runtime_tool_with_its_arguments():
    """A new runtime tool goes into the runtime/README.md section "Runtime toolset support" (CLAUDE.md)."""
    declared = declared_toolset()

    assert documented_toolset() == declared


def test_each_documented_command_is_one_that_its_module_accepts(tmp_path):
    """ADR-0001 moved ``python -m sprout.runstore worker`` to ``python -m sprout.worker start``. A
    document that names a module or a subcommand that is gone sends its reader to an error."""
    commands = documented_commands()

    assert refusals_of_help(commands, home=tmp_path) == {}

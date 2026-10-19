"""The runtime's tool declaration: what an agent may call, and with which arguments.

``runtime/schema/tools.json`` is the output of `sprout-runtime --dump-schema`. It is checked in, so
the worker needs no sandbox to know the tools. A runtime change that is not regenerated here fails a
test (tests/worker/tools/test_schema.py) before it reaches a model. The Go tools read their
arguments by name and ignore the rest. So ``ToolAccess`` checks each call against this declaration,
and refuses an argument that the runtime does not take. A dropped argument would let a model believe
that it set an option that it did not set.

A source checkout reads that file in place, the copy that the drift test compares with the runtime.
An installed wheel has no ``runtime/`` directory. Its build copies the file in beside this module
(``force-include`` in pyproject.toml), and it is read as package data.
"""

from __future__ import annotations

import json
from importlib import resources
from pathlib import Path
from typing import Any

#: The checked-in declaration, when this module runs from a source checkout
#: (src/sprout/worker/tools -> repository root).
SOURCE_SCHEMA = Path(__file__).resolve().parents[4] / "runtime" / "schema" / "tools.json"
#: Its name as package data of this package, where a wheel carries it.
PACKAGED_SCHEMA = "tools.json"

#: The declaration that this process reads: the source checkout's copy if it exists, else the package data.
RUNTIME_SCHEMA = (
    SOURCE_SCHEMA if SOURCE_SCHEMA.is_file() else resources.files(__package__).joinpath(PACKAGED_SCHEMA)
)


def runtime_tools() -> list[dict[str, Any]]:
    """``[{name, description, inputSchema}, ...]``, in the runtime's order."""
    # Not validated here: the drift test keeps the checked-in file in the runtime's shape.
    tools: list[dict[str, Any]] = json.loads(RUNTIME_SCHEMA.read_text(encoding="utf-8"))["tools"]
    return tools

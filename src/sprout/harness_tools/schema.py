"""The runtime's tool declaration: what an agent may call, and with which arguments.

``runtime/schema/tools.json`` is `sprout-runtime --dump-schema`, checked in so
the harness needs no sandbox to know the tool surface. A runtime change that is
not regenerated here fails a test (tests/harness/test_runtime_schema.py) rather
than reaching a model. The Go tools read their arguments by name and ignore the
rest, so ``ToolAccess`` checks every call against this declaration: an argument
the runtime does not take is refused instead of silently dropped, which once let
a model believe it had set an output bound it did not have.

A source checkout reads that file in place, the copy the drift test compares
with the runtime. An installed wheel has no ``runtime/`` directory: its build
copies the file in beside this module (``force-include`` in pyproject.toml),
and it is read as package data.
"""

from __future__ import annotations

from importlib import resources
import json
from pathlib import Path

#: The checked-in declaration, when this module runs from a source checkout
#: (src/sprout/harness_tools -> repository root).
SOURCE_SCHEMA = Path(__file__).resolve().parents[3] / "runtime" / "schema" / "tools.json"
#: Its name as package data of this package, where a wheel carries it.
PACKAGED_SCHEMA = "tools.json"

RUNTIME_SCHEMA = (
    SOURCE_SCHEMA if SOURCE_SCHEMA.is_file() else resources.files(__package__).joinpath(PACKAGED_SCHEMA)
)


def runtime_tools() -> list[dict]:
    """``[{name, description, inputSchema}, ...]``, in the runtime's order."""
    return json.loads(RUNTIME_SCHEMA.read_text(encoding="utf-8"))["tools"]

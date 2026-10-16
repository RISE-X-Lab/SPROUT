"""The data files the package reads at run time: SQL, YAML and JSON kept outside ``src/``.

Each file lives at ``resources/<package>/<name>`` in the repository. A source checkout
reads it there. An installed wheel has no ``resources/`` directory: its build copies the
directory into the package as ``sprout/resources`` (``force-include`` in pyproject.toml),
and :func:`resource` returns that copy. The copy must be installed as files, not a zip.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path

CHECKOUT = Path(__file__).resolve().parents[2] / "resources"


def resource(name: str) -> Path:
    """The data file ``name`` (``"runstore/schema.sql"``): the checkout's copy, else the wheel's.

    Returns the path whether or not the file exists. Reading a missing one raises ``OSError``.
    """
    checkout = CHECKOUT / name
    if checkout.is_file():
        return checkout
    return Path(str(resources.files("sprout").joinpath("resources", name)))

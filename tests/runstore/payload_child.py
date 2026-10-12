"""Real stdin bootstrap with a deterministic executor instead of a VM or model."""

import os
from pathlib import Path
import sys
import time

from sprout.runstore import child
from sprout.runstore.files import write_json


def execute(payload: dict, directory: Path) -> dict:
    write_json(directory / "executed.json", {"payload": payload, "env_names": sorted(os.environ)})
    if payload["effective_spec"].get("prompt") == "hold-fixture":
        time.sleep(120)
    if payload["effective_spec"].get("prompt") == "gate-fixture":
        while not (directory / "release").exists():
            time.sleep(0.02)
    return {"status": "completed", "fixture_kind": payload["kind"]}


if __name__ == "__main__":
    child.execute = execute
    (Path(sys.argv[1]) / "bootstrap-ready").touch()
    raise SystemExit(child.main())

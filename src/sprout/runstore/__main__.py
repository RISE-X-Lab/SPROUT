"""The Run Store service: create its schema (`init`) or serve its HTTP API (`serve`)."""

import argparse
import os
from pathlib import Path

from sprout.contract.storage import volatile_reason
from sprout.runstore.config import read_config, snapshot_validator
from sprout.runstore.index import Index
from sprout.runstore.store import Store


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["init", "serve"])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18110)
    args = parser.parse_args()
    config = read_config(args.config)
    reason = volatile_reason(config["artifact_root"])
    if reason:
        parser.error(reason)
    store = Store(os.environ["SPROUT_RUNSTORE_DSN"])
    if args.command == "init":
        store.initialize()
    else:
        import uvicorn

        from sprout.runstore.api import create_app

        index = Index(store, snapshot_validator(store, config))
        uvicorn.run(
            create_app(store, os.environ["SPROUT_RUNSTORE_TOKEN"], index=index, profiles=config["profiles"]),
            host=args.host,
            port=args.port,
        )


if __name__ == "__main__":
    main()

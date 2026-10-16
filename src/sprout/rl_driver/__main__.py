"""Start only the RL driver; Run Store and its workers are separate services."""

import argparse
import json
import math
import os
from pathlib import Path

import httpx
import uvicorn

from sprout.rl_driver.backend import RunStoreClient
from sprout.rl_driver.driver import Driver
from sprout.rl_driver.ledger import Ledger
from sprout.rl_driver.messages import MessageAdapter
from sprout.rl_driver.server import DEFAULT_PORT, create_app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    args = parser.parse_args()
    try:
        config = json.loads(args.config.read_text())
        allowed = {
            "runstore_url",
            "runstore_token_env",
            "driver_token_env",
            "ledger",
            "poll_interval_s",
            "miles",
        }
        if not isinstance(config, dict) or set(config) - allowed:
            raise ValueError(f"Config fields must be drawn from {sorted(allowed)}")
        url = config.get("runstore_url", "http://127.0.0.1:18110").rstrip("/")
        parsed = httpx.URL(url)
        if (
            parsed.scheme not in {"http", "https"}
            or not parsed.host
            or parsed.userinfo
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError(
                "runstore_url must be an HTTP(S) URL without embedded credentials/query/fragment"
            )
        token = os.environ[config.get("runstore_token_env", "SPROUT_RUNSTORE_TOKEN")]
        driver_token_env = config.get("driver_token_env", "SPROUT_RL_DRIVER_TOKEN")
        driver_token = os.environ[driver_token_env] if driver_token_env is not None else None
        if not token or driver_token == "":
            raise ValueError("Configured tokens must be nonempty")
        path = Path(config["ledger"]).expanduser()
        if not path.is_absolute():
            path = args.config.resolve().parent / path
        interval = config.get("poll_interval_s", 1)
        if type(interval) not in (int, float) or not math.isfinite(interval) or interval <= 0:
            raise ValueError("poll_interval_s must be finite and positive")
        ledger = Ledger(path)
        ledger.bind(url)
    except (KeyError, ValueError, TypeError, OSError) as error:
        parser.error(str(error))
    client = RunStoreClient(url, token)
    client.http.timeout = httpx.Timeout(10)
    try:
        driver = Driver(client, ledger)
        miles = MessageAdapter(driver, config["miles"]) if config.get("miles") is not None else None
        uvicorn.run(
            create_app(driver, driver_token, poll_interval_s=interval, miles=miles),
            host=args.host,
            port=args.port,
        )
    finally:
        client.close()


if __name__ == "__main__":
    main()

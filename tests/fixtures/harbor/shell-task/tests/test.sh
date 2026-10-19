#!/bin/bash
set -eu
mkdir -p /logs/verifier
python3 - <<'PY'
from pathlib import Path

answer = Path("/app/answer.txt")
reward = int(answer.exists() and answer.read_text() == "ready\n")
Path("/logs/verifier/reward.txt").write_text(str(reward))
PY

#!/usr/bin/env bash
# Run every gate: the repository checks that each commit must pass.
#
# Each gate is an executable file in gate/checks/. This script runs each one from the
# repository root. It exits 1 if a gate fails or is not executable, and 0 otherwise.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2

failed=0
for check in gate/checks/*; do
  [ -f "$check" ] || continue  # Skip directories, such as __pycache__.
  # A check that cannot run must not read as one that passed.
  if [ ! -x "$check" ]; then
    echo "gate: $check is not executable" >&2
    failed=1
    continue
  fi
  if "$check"; then
    echo "gate: pass  $check"
  else
    echo "gate: FAIL  $check" >&2
    failed=1
  fi
done
exit "$failed"

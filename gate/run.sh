#!/usr/bin/env bash
# Run every gate: the repo-specific checks a commit has to pass.
#
# Each gate is an executable in gate/checks/, run from the repo root; a non-zero
# exit fails the commit. The pre-commit hook in gate/hooks/ runs this script.
set -u
cd "$(git rev-parse --show-toplevel)" || exit 2

failed=0
for check in gate/checks/*; do
  [ -f "$check" ] || continue  # __pycache__ and the like
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

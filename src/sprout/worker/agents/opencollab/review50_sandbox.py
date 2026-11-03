"""Review50 file operations executed inside the sandbox, using only stdlib."""

from __future__ import annotations

import difflib
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def operate(name: str, args: dict[str, Any]) -> dict[str, Any]:
    root = Path.cwd().resolve()
    if name in {"submit_patch", "get_status"}:
        marker = Path("/opt/review50-baseline")
        baseline = marker.read_text().strip() if marker.is_file() else "HEAD"

        def git(*parts: Any) -> Any:
            return subprocess.run(["git", *parts], check=True, capture_output=True, text=True).stdout

        if name == "get_status":
            status = git("status", "--porcelain=v1", "--untracked-files=all")
            diff = git("diff", "--stat", baseline)
            return {
                "status": "ok",
                "patch": {
                    "baseline": baseline,
                    "working_tree_status": status[:10000],
                    "tracked_diff_stat": diff[:10000],
                    "is_truncated": len(status) > 10000 or len(diff) > 10000,
                },
            }
        # Include newly created files by the same mechanism as verifier.collect.
        git("add", "-N", "--", ".")
        patch = git("diff", "--binary", baseline)
        return {
            "status": "submitted",
            "patch_sha256": hashlib.sha256(patch.encode()).hexdigest(),
            "patch_chars": len(patch),
            "baseline": baseline,
            "grading": "pending official verifier",
        }
    path = (root / args["filepath"]).resolve()
    try:
        path.relative_to(root)
    except ValueError:
        raise ValueError("File paths must remain within the repository workspace") from None
    result = {"status": "ok", "filepath": args["filepath"]}
    if name == "read_file":
        lines = path.read_text().splitlines(keepends=True)
        start = args.get("start_line", 1)
        end = args.get("end_line", start + 149)
        if start < 1 or end < start:
            raise ValueError("Line bounds must be positive, ordered and inclusive")
        stop = min(end, start + 149, len(lines))
        text = "".join(lines[start - 1 : stop])
        result.update(
            content=text[:10000],
            start_line=start,
            end_line=stop,
            total_lines=len(lines),
            is_truncated=stop < len(lines) or len(text) > 10000,
        )
    elif name == "write_file":
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(args["content"])
        result["size"] = len(args["content"].encode())
    elif name == "edit_file":
        before = path.read_text()
        old, new = args["old_string"], args["new_string"]
        if not old:
            raise ValueError("old_string must not be empty")
        occurrences = before.count(old)
        if occurrences == 0 or occurrences > 1 and not args.get("allow_multiple", False):
            raise ValueError(f"old_string matches {occurrences} times; require one or allow_multiple")
        after = before.replace(old, new)
        path.write_text(after)
        diff = "".join(
            difflib.unified_diff(
                before.splitlines(True),
                after.splitlines(True),
                fromfile="a/" + args["filepath"],
                tofile="b/" + args["filepath"],
            )
        )
        result.update(
            occurrences=occurrences, strategy="exact", diff=diff[:10000], is_truncated=len(diff) > 10000
        )
    else:
        raise ValueError(f"Unknown review50 operation: {name}")
    return result


if __name__ == "__main__":
    try:
        answer = operate(sys.argv[1], json.loads(sys.argv[2]))
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        answer = {"status": "error", "error_type": type(error).__name__, "error_message": str(error)}
    print(json.dumps(answer, ensure_ascii=False))

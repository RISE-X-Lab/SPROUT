"""Combine the five requested method rows from separate run directories."""

import argparse
import csv
import json
from pathlib import Path

from examples.qwen.report import build_report
from examples.qwen.run import SHARED_BASELINE_HISTORY, SHARED_BASELINE_KEYS


def compare(roots: dict[str, Path]) -> dict:
    reports = {method: build_report(root, method) for method, root in roots.items()}
    baseline = reports["baseline"]["manifest"]
    for method, report in reports.items():
        for key in SHARED_BASELINE_KEYS:
            # Both manifests come from disk: a key written before it existed takes the same
            # historical value on either side (run.py applies it to the baseline alone).
            default = SHARED_BASELINE_HISTORY.get(key)
            if report["manifest"].get(key, default) != baseline.get(key, default):
                raise ValueError(f"Incomparable {method} input: {key}")
        if method != "baseline":
            reference = report["manifest"].get("baseline_root")
            if reference is None or Path(reference).resolve() != roots["baseline"].resolve():
                raise ValueError(f"{method} did not use the selected shared baseline")
    rows = [
        row for method in ("baseline", "bpo", "shepherd", "sprout") for row in reports[method]["comparison"]
    ]
    return {
        "dataset": baseline["dataset"],
        "tasks": len(baseline["tasks"]),
        "complete": all(r["complete"] for r in rows),
        "comparison": rows,
    }


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "bpo", "shepherd", "sprout"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    result = compare({name: getattr(args, name) for name in ("baseline", "bpo", "shepherd", "sprout")})
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "comparison.json").write_text(json.dumps(result, indent=2) + "\n")
    columns = (
        "method",
        "Resolve Rate",
        "Recovery",
        "Steps",
        "Tokens (M)",
        "complete",
        "completed_tasks",
        "total_tasks",
    )
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(result["comparison"])
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

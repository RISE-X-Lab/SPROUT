#!/usr/bin/env python3
"""Delete the snapshot layers that no record references from an AgentENV posixfs store.

AgentENV's DELETE /snapshots removes a snapshot's record, not its bytes. The bytes stay in
the shared, content-addressed `managed-layers/` store, and AgentENV, local or upstream, never
reclaims a layer that no record references (checked on 2026-09-03). Retire this script when
AgentENV reclaims such layers itself.

The script keeps a layer unless it is sure that the layer is dead:

- It marks each sha256 digest that occurs anywhere in a catalog record, whatever the state or
  kind of the record. Keeping too much wastes disk. Keeping too little destroys live snapshots.
- It deletes a `sha256_*` layer file only when its digest is unmarked and the file is older
  than --min-age-hours (default 2). A new layer is on disk before its record commits.
- It deletes nothing without --yes. It prints the counts and the reclaimable size first.

    sudo python3 ops/gc_layers.py --store /opt/aenv-bench/home/snapshot-store
    sudo python3 ops/gc_layers.py --store ... --yes
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

DIGEST = re.compile(r"sha256:([0-9a-f]{64})")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--store", required=True, help="the snapshot-store root (contains repository/)")
    parser.add_argument(
        "--min-age-hours",
        type=float,
        default=2.0,
        help="never touch layers younger than this (capture window guard)",
    )
    parser.add_argument("--yes", action="store_true", help="actually delete (default: report only)")
    args = parser.parse_args()

    repo = Path(args.store) / "repository"
    records_dir = repo / "catalog" / "records"
    layers_dir = repo / "managed-layers"
    for required in (records_dir, layers_dir):
        if not required.is_dir():
            print("not a posixfs snapshot store: missing %s" % required, file=sys.stderr)
            return 2

    referenced: set[str] = set()
    records = 0
    for record in records_dir.iterdir():
        if record.suffix != ".json":
            continue
        records += 1
        referenced.update(DIGEST.findall(record.read_text(errors="replace")))
    print("records scanned: %d, distinct digests referenced: %d" % (records, len(referenced)))

    now = time.time()
    min_age = args.min_age_hours * 3600
    keep = orphan = young = 0
    orphan_bytes = 0
    victims: list[Path] = []
    for layer in layers_dir.iterdir():
        name = layer.name
        if not name.startswith("sha256_"):
            keep += 1  # unknown naming: never touch what we do not understand
            continue
        digest = name[len("sha256_") :].split(".")[0]
        if digest in referenced:
            keep += 1
            continue
        stat = layer.stat()
        if now - stat.st_mtime < min_age:
            young += 1
            continue
        orphan += 1
        orphan_bytes += stat.st_size
        victims.append(layer)

    print("layers: %d referenced/kept, %d too young to judge, %d orphaned" % (keep, young, orphan))
    print("reclaimable: %.1f GiB" % (orphan_bytes / 2**30))

    if not args.yes:
        print("\nDRY RUN -- nothing deleted. Pass --yes to reclaim.")
        return 0

    freed = 0
    for layer in victims:
        try:
            size = layer.stat().st_size
            layer.unlink()
            freed += size
        except OSError as exc:
            print("failed: %s (%s)" % (layer.name, exc), file=sys.stderr)
    print("deleted %d layers, freed %.1f GiB" % (len(victims), freed / 2**30))
    return 0


if __name__ == "__main__":
    sys.exit(main())

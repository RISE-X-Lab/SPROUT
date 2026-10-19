"""A fake ``btrfs`` executable. It keeps each subvolume as a plain directory and records each call.

It answers the commands that ``sprout_sandbox.btrfs_store.BtrfsStore`` runs, with the output of
btrfs-progs 6.6: `subvolume create`, `subvolume snapshot [-r]`, `subvolume delete` and
`property set`. A snapshot is a full copy of its source. Nothing is read-only: `property set`
changes nothing, and a write into a read-only snapshot succeeds. It appends each subvolume that it
makes to ``btrfs.subvolumes`` in the records directory ``$FAKE_TOOLS_DIR``, where the fake podman's
`unshare find` finds nested subvolumes. ``tests/support/test_fakes.py`` runs the same calls against
the real btrfs. ``tests.support.tools.FakeBtrfs`` sets the knob:

    FAKE_BTRFS_FAIL   subcommands (create, snapshot, delete, set) that exit 1 with an error
"""

import json
import os
import shutil
import sys
from pathlib import Path


def main(argv: list[str]) -> int:
    records = Path(os.environ["FAKE_TOOLS_DIR"])
    _record(records, argv)
    subcommand = argv[1] if len(argv) > 1 else ""
    if subcommand in [item for item in os.environ.get("FAKE_BTRFS_FAIL", "").split(",") if item]:
        sys.stderr.write("ERROR: simulated %s failure\n" % subcommand)
        return 1
    if argv[:2] == ["subvolume", "create"]:
        os.mkdir(argv[2])
        _made(records, argv[2])
        print("Create subvolume '%s'" % argv[2])
    elif argv[:2] == ["subvolume", "snapshot"]:
        source, destination = [arg for arg in argv[2:] if arg != "-r"]
        shutil.copytree(source, destination, symlinks=True)
        _made(records, destination)
        readonly = "readonly " if "-r" in argv else ""
        print("Create a %ssnapshot of '%s' in '%s'" % (readonly, source, destination))
    elif argv[:2] == ["subvolume", "delete"]:
        shutil.rmtree(argv[2])
        sys.stderr.write("WARNING: cannot read default subvolume id: Operation not permitted\n")
        print("Delete subvolume 999 (no-commit): '%s'" % argv[2])
    elif argv[:2] != ["property", "set"]:
        sys.stderr.write("fake btrfs: unknown command %r\n" % (argv,))
        return 2
    return 0


def _record(records: Path, argv: list[str]) -> None:
    for name, entry in (("btrfs.log", argv), ("tools.log", ["btrfs", *argv])):
        with open(records / name, "a") as log:
            log.write(json.dumps(entry) + "\n")


def _made(records: Path, subvolume: str) -> None:
    with open(records / "btrfs.subvolumes", "a") as made:
        made.write(subvolume + "\n")


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

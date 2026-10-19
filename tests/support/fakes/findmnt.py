"""A fake ``findmnt`` executable. It answers the mount check of a store and records each call.

It answers only `findmnt -no FSTYPE,OPTIONS --target <path>`, the command that
``sprout_sandbox.btrfs_store.mount_problem`` runs, the way util-linux 2.39 does: one line with the
filesystem type and the mount options. By default it reports the mount that ``ops/btrfs_store.sh``
makes, so every path is on a store. ``tests/support/test_fakes.py`` runs the same check against the
real findmnt. ``tests.support.tools.FakeFindmnt`` sets the knobs:

    FAKE_FINDMNT_FSTYPE    the filesystem type that it reports (default btrfs)
    FAKE_FINDMNT_OPTIONS   the mount options that it reports (default STORE_MOUNT_OPTIONS)
"""

import json
import os
import sys
from pathlib import Path

#: The options of the mount that ``ops/btrfs_store.sh`` makes, as findmnt reports them.
STORE_MOUNT_OPTIONS = "rw,noatime,ssd,discard=async,space_cache=v2,user_subvol_rm_allowed,subvolid=5,subvol=/"


def main(argv: list[str]) -> int:
    records = Path(os.environ["FAKE_TOOLS_DIR"])
    for name, entry in (("findmnt.log", argv), ("tools.log", ["findmnt", *argv])):
        with open(records / name, "a") as log:
            log.write(json.dumps(entry) + "\n")
    if argv[:2] != ["-no", "FSTYPE,OPTIONS"] or "--target" not in argv:
        sys.stderr.write("fake findmnt: unknown command %r\n" % (argv,))
        return 2
    fstype = os.environ.get("FAKE_FINDMNT_FSTYPE", "btrfs")
    # findmnt pads a column to the width of its header, FSTYPE, even with -n.
    print("%-6s %s" % (fstype, os.environ.get("FAKE_FINDMNT_OPTIONS", STORE_MOUNT_OPTIONS)))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

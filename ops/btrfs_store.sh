#!/usr/bin/env bash
# Create and mount the btrfs store for the podman backend's rootfs subvolumes. This is the
# only step of the backend that needs root. The unprivileged owner of the mount does the rest:
# it extracts each base rootfs, snapshots it, starts containers on it and deletes snapshots.
#
#   sudo ops/btrfs_store.sh MOUNT_POINT IMAGE_FILE SIZE [OWNER]
#   sudo ops/btrfs_store.sh /path/to/sprout-btrfs /var/lib/sprout-btrfs-rootless.img 200G
#
# IMAGE_FILE is a sparse file of SIZE (truncate(1) syntax) that holds a btrfs filesystem. The
# script mounts it at MOUNT_POINT with noatime and user_subvol_rm_allowed, the option that lets
# the owner delete the subvolumes that it created. It gives the mount point to OWNER (default:
# the user who ran sudo) and adds a nofail fstab line that mounts the store at boot.
# A second run only mounts the store if it is not mounted. The script reuses an image that
# already holds btrfs and keeps an fstab line that is already there. It refuses an image that
# holds anything else, and a mount without user_subvol_rm_allowed.
set -euo pipefail

usage() {
    echo "usage: sudo $0 MOUNT_POINT IMAGE_FILE SIZE [OWNER]" >&2
    exit 2
}
if [ $# -lt 3 ] || [ $# -gt 4 ]; then usage; fi
mount_point=$1
image=$2
size=$3
owner=${4:-${SUDO_USER:-}}

[ "$(id -u)" -eq 0 ] || { echo "$0: must run as root (sudo)" >&2; exit 1; }
[ -n "$owner" ] || { echo "$0: no OWNER given and SUDO_USER is unset" >&2; exit 1; }
[ "$owner" != root ] || { echo "$0: the store must belong to the unprivileged user that runs podman, not root" >&2; exit 1; }
id "$owner" >/dev/null
for tool in truncate mkfs.btrfs blkid mount mountpoint findmnt; do
    command -v "$tool" >/dev/null || { echo "$0: $tool not found" >&2; exit 1; }
done
grep -qw btrfs /proc/filesystems || modprobe btrfs

opts=loop,noatime,user_subvol_rm_allowed

if [ -e "$image" ]; then
    type=$(blkid -p -o value -s TYPE "$image" || true)
    if [ "$type" != btrfs ]; then
        echo "$0: $image exists and holds no btrfs filesystem; refusing to format it" >&2
        exit 1
    fi
    echo "reusing the btrfs filesystem in $image"
else
    truncate -s "$size" "$image"
    mkfs.btrfs -q -L sprout-store "$image"
    echo "created $image: $size sparse, btrfs"
fi

mkdir -p "$mount_point"
if mountpoint -q "$mount_point"; then
    echo "$mount_point is already mounted"
else
    mount -o "$opts" "$image" "$mount_point"
    echo "mounted $image at $mount_point ($opts)"
fi
if ! findmnt -no OPTIONS "$mount_point" | tr , '\n' | grep -qx user_subvol_rm_allowed; then
    echo "$0: $mount_point is mounted without user_subvol_rm_allowed; unmount it and run again" >&2
    exit 1
fi

chown "$owner:$(id -gn "$owner")" "$mount_point"

if awk -v mp="$mount_point" '$1 !~ /^#/ && $2 == mp { found = 1 } END { exit !found }' /etc/fstab; then
    echo "fstab already lists $mount_point"
else
    printf '%s %s btrfs %s,nofail 0 0\n' "$image" "$mount_point" "$opts" >>/etc/fstab
    command -v systemctl >/dev/null && systemctl daemon-reload
    echo "added $mount_point to /etc/fstab (nofail)"
fi

echo "store ready: $mount_point owned by $owner"

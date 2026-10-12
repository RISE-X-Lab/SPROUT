#!/usr/bin/env bash
# Set up this host for Sprout's podman backend: podman 6 (a static build installed
# for this user), a btrfs store, the Python package and the runtime binary, then
# prove it works.
#
#   ops/setup_podman.sh [--store DIR] [--image FILE] [--size SIZE]
#                       [--podman-prefix DIR] [--podman-version TAG] [--share-images DIR|none]
#                       [--skip-python] [--skip-verify] [--check]
#
# Defaults: --store ~/sprout-btrfs --image /var/lib/sprout-btrfs-rootless.img
# --size 200G (the image is sparse; size it for every base rootfs you will
# extract, about the sum of the uncompressed images) --podman-prefix
# ~/.local/opt/podman --podman-version v6.1.3. Every step is idempotent and
# skipped when already done, so re-running is cheap. Root is used for four
# things only, through sudo: installing packages, the subordinate id range, the
# cgroup delegation drop-in and making the store (ops/btrfs_store.sh).
# Everything the backend does afterwards is rootless. --check verifies without
# changing anything and exits non-zero on a gap.
#
# Podman is the static build published by mgoltzsche/podman-static (podman,
# crun, runc, conmon, netavark, aardvark-dns, pasta, fuse-overlayfs), verified
# against a pinned sha256 (v6.1.3 amd64) or, for another version, the
# maintainer's GPG signature when gpg is installed, and unpacked under the
# prefix with a configuration, image storage, run root and file locks of its
# own. A distribution podman on the same host is left alone: two podmans of one
# user must not share state (the distribution's keeps its container locks in a
# shared-memory segment named after the uid); the one piece podman insists on
# sharing is the per-user rootless pause process. Nothing goes on PATH: every
# Sprout entry point runs the podman $SPROUT_PODMAN_BIN names, else this one
# when it is installed, else `podman` on PATH (sprout_sandbox.podman.
# default_podman_bin); `python -m sprout.services env` exports the resolved
# path. --share-images DIR adds another podman's image storage as a read-only
# additional store, so images it pulled need no second pull; the choice is
# remembered in <prefix>/etc/share-images (--share-images none forgets it).
# Container cgroups go through the user's systemd (cgroup_manager systemd, the
# bundled runc as the OCI runtime), so --cpus/--memory apply from any login
# shell, as they did with the distribution's podman.
#
# What the backend needs from the host: (1) a subordinate uid/gid range for
# this user, (2) cgroup v2 with the cpu, memory and pids controllers delegated
# to the user session (or --cpus/--memory on a rootless container fail), (3) a
# btrfs filesystem mounted with user_subvol_rm_allowed, (4) Go for the runtime
# and Python 3.12 for the rest, (5) nftables for netavark's firewall rules.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STORE=$HOME/sprout-btrfs
IMAGE=/var/lib/sprout-btrfs-rootless.img
SIZE=200G
PREFIX=$HOME/.local/opt/podman
VERSION=v6.1.3
SHARE=unset
CHECK=0
SKIP_PYTHON=0
SKIP_VERIFY=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --store) STORE=$2; shift 2;;
        --image) IMAGE=$2; shift 2;;
        --size) SIZE=$2; shift 2;;
        --podman-prefix) PREFIX=$2; shift 2;;
        --podman-version) VERSION=$2; shift 2;;
        --share-images) SHARE=$2; shift 2;;
        --check) CHECK=1; shift;;
        --skip-python) SKIP_PYTHON=1; shift;;
        --skip-verify) SKIP_VERIFY=1; shift;;
        -h|--help) awk 'NR > 1 && !/^#/ { exit } NR > 1 { print }' "$0"; exit 0;;
        *) echo "unknown argument: $1" >&2; exit 2;;
    esac
done
[[ $EUID -ne 0 ]] || { echo "run as the user who will own the sandboxes, not root" >&2; exit 2; }
# One spelling of the prefix, whatever was typed: the run root is derived from it.
PREFIX=$(realpath -m -- "$PREFIX")
DEFAULT_PREFIX=$(realpath -m -- "$HOME/.local/opt/podman")

ok()   { printf '  ok    %s\n' "$*"; }
todo() { printf '  TODO  %s\n' "$*"; }
note() { printf '  note  %s\n' "$*"; }
GAPS=0
gap()  { GAPS=$((GAPS + 1)); todo "$@"; }

# --- 1. packages ------------------------------------------------------------
echo "[1/7] packages"
PACKAGES=(btrfs-progs uidmap nftables golang-go curl ca-certificates gnupg)
MISSING=()
for pkg in "${PACKAGES[@]}"; do
    dpkg -s "$pkg" >/dev/null 2>&1 || MISSING+=("$pkg")
done
# The venv module ships under a versioned package name (python3.12-venv); test
# the capability rather than a name.
[[ $SKIP_PYTHON -eq 1 ]] || python3 -c 'import venv, ensurepip' >/dev/null 2>&1 || MISSING+=(python3-venv)
if [[ ${#MISSING[@]} -eq 0 ]]; then
    ok "all installed: ${PACKAGES[*]}"
elif [[ $CHECK -eq 1 ]]; then
    gap "missing packages: ${MISSING[*]}"
else
    echo "  installing: ${MISSING[*]}"
    sudo apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq --no-install-recommends "${MISSING[@]}"
    ok "installed: ${MISSING[*]}"
fi

# --- 2. podman --------------------------------------------------------------
echo "[2/7] podman $VERSION under $PREFIX"
case "$(uname -m)" in
    x86_64) ARCH=amd64;;
    aarch64) ARCH=arm64;;
    *) echo "no podman-static build for $(uname -m)" >&2; exit 2;;
esac
BUNDLE=$PREFIX/root/podman-linux-$ARCH
PODMAN=$PREFIX/bin/podman
# The run root (tmp dir, run root, locks) is remembered per prefix: libpod refuses
# to start when they move under an existing database.
if [[ -f $PREFIX/etc/rundir ]]; then
    RUNDIR=$(<"$PREFIX/etc/rundir")
else
    RUNDIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/sprout-podman
    [[ $PREFIX == "$DEFAULT_PREFIX" ]] || RUNDIR=$RUNDIR-$(printf %s "$PREFIX" | sha256sum | cut -c1-8)
fi
# --share-images is remembered too: a plain re-run must not silently drop it.
if [[ $SHARE == unset ]]; then
    SHARE=$( [[ -f $PREFIX/etc/share-images ]] && cat "$PREFIX/etc/share-images" || true )
elif [[ $SHARE == none ]]; then
    SHARE=
else
    SHARE=$(realpath -m -- "$SHARE")
fi
# sha256 of the release tarballs this script has installed; another version or
# architecture is verified with the maintainer's GPG signature instead.
declare -A SHA256=(
    [v6.1.3-amd64]=dcfff14eae60569653c21eed75c96e5fd4140f23448637bc6e0ee75a12e1267a
)
SIGNING_KEY=0CCF102C4F95D89E583FF1D4F8B5AF50344BB503   # Max Goltzsche, from the project's README

have_podman() {
    [[ -x $PODMAN && -x $BUNDLE/usr/local/bin/podman ]] && \
        "$BUNDLE/usr/local/bin/podman" --version 2>/dev/null | grep -q " ${VERSION#v}\$"
}

running_containers() {
    [[ -x $PODMAN ]] && [[ -n "$("$PODMAN" ps -q 2>/dev/null | head -1)" ]]
}

render_configs() {
    # The three generated files, rendered into directory $1 (never edited by hand:
    # a change of prefix, version or --share-images lands here on the next run).
    local into=$1 shared=""
    [[ -n $SHARE ]] && shared=$(printf 'additionalimagestores = ["%s"]\n' "$SHARE")
    mkdir -p "$into/etc" "$into/bin"
    cat > "$into/etc/containers.conf" <<EOF
# podman $VERSION for Sprout's sandboxes, written by ops/setup_podman.sh: its own
# configuration, storage ($PREFIX/storage), run root and locks, apart from any
# other podman of this user. Selected through $PODMAN, which sets
# CONTAINERS_CONF to this file.

[containers]

[network]
network_backend = "netavark"
firewall_driver = "nftables"
network_config_dir = "$PREFIX/storage/networks"
default_rootless_network_cmd = "pasta"

[engine]
# container cgroups through the user's systemd, so --cpus/--memory apply from any login
# shell; the bundled runc speaks to systemd without libsystemd, the bundled crun cannot
cgroup_manager = "systemd"
runtime = "runc"
events_logger = "file"
# file locks under tmp_dir: the shared-memory segment belongs to the distribution's podman
lock_type = "file"
conmon_path = ["$BUNDLE/usr/local/lib/podman/conmon"]
helper_binaries_dir = ["$BUNDLE/usr/local/lib/podman", "$BUNDLE/usr/local/bin"]
network_cmd_path = "$BUNDLE/usr/local/bin/pasta"
tmp_dir = "$RUNDIR/libpod"
static_dir = "$PREFIX/storage/libpod"
volume_path = "$PREFIX/storage/volumes"

[engine.runtimes]
runc = ["$BUNDLE/usr/local/bin/runc"]
crun = ["$BUNDLE/usr/local/bin/crun"]
EOF
    cat > "$into/etc/storage.conf" <<EOF
# Image storage for podman $VERSION, written by ops/setup_podman.sh. Native rootless
# overlay; fuse-overlayfs from the bundle is the fallback if the kernel refuses.
[storage]
driver = "overlay"
graphroot = "$PREFIX/storage/graph"
runroot = "$RUNDIR/run"

[storage.options]
${shared}
[storage.options.overlay]
mountopt = "nodev"
EOF
    cat > "$into/bin/podman" <<EOF
#!/bin/sh
# podman $VERSION (static) with its own configuration and storage; written by ops/setup_podman.sh.
# Sprout finds it through SPROUT_PODMAN_BIN=$PODMAN (python -m sprout.services env).
export CONTAINERS_CONF="$PREFIX/etc/containers.conf"
export CONTAINERS_STORAGE_CONF="$PREFIX/etc/storage.conf"
export CONTAINERS_REGISTRIES_CONF="$BUNDLE/etc/containers/registries.conf"
exec "$BUNDLE/usr/local/bin/podman" "\$@"
EOF
    chmod +x "$into/bin/podman"
}

configs_differ() {
    # 0 when the rendered configs differ from the installed ones (or are missing).
    local staged=$1 f
    for f in etc/containers.conf etc/storage.conf bin/podman; do
        cmp -s "$staged/$f" "$PREFIX/$f" || return 0
    done
    return 1
}

install_configs() {
    # Rendered beside the live files and moved into place one by one, so a podman
    # starting meanwhile reads a whole file, never a truncated one.
    local staged f
    mkdir -p "$PREFIX"
    staged=$(mktemp -d "$PREFIX/.render.XXXXXX")
    render_configs "$staged"
    if configs_differ "$staged"; then
        if running_containers; then
            rm -rf "$staged"
            echo "  $PODMAN has running containers; the configuration cannot change under them (stop them first)" >&2
            exit 1
        fi
        mkdir -p "$PREFIX/etc" "$PREFIX/bin" "$PREFIX/storage"
        for f in etc/containers.conf etc/storage.conf bin/podman; do
            cmp -s "$staged/$f" "$PREFIX/$f" 2>/dev/null || { mv -f "$staged/$f" "$PREFIX/$f"; note "wrote $PREFIX/$f"; }
        done
    fi
    rm -rf "$staged"
    printf '%s\n' "$RUNDIR" > "$PREFIX/etc/rundir"
    printf '%s\n' "$SHARE" > "$PREFIX/etc/share-images"
    # c/image reads $HOME/.config/containers/policy.json, then /etc/containers/policy.json;
    # a host without any podman package has neither.
    if [[ ! -f /etc/containers/policy.json && ! -f $HOME/.config/containers/policy.json ]]; then
        mkdir -p "$HOME/.config/containers"
        cp "$BUNDLE/etc/containers/policy.json" "$HOME/.config/containers/policy.json"
        note "wrote $HOME/.config/containers/policy.json (the bundle's default: accept any image)"
    fi
}

if have_podman && [[ $CHECK -eq 1 ]]; then
    staged=$(mktemp -d)
    render_configs "$staged"
    if configs_differ "$staged"; then
        gap "$PREFIX/{etc/containers.conf,etc/storage.conf,bin/podman} differ from what this script would write (run without --check)"
    else
        ok "$("$PODMAN" --version) at $PODMAN$( [[ -n $SHARE ]] && printf ', reading images from %s' "$SHARE")"
    fi
    rm -rf "$staged"
    [[ -f /etc/containers/policy.json || -f $HOME/.config/containers/policy.json ]] || \
        gap "no policy.json for the image library (run without --check; it copies the bundle's)"
elif have_podman; then
    install_configs
    ok "$("$PODMAN" --version) at $PODMAN$( [[ -n $SHARE ]] && printf ', reading images from %s' "$SHARE")"
elif [[ $CHECK -eq 1 ]]; then
    gap "podman $VERSION is not installed at $PODMAN (run without --check)"
else
    if running_containers; then
        echo "  $PODMAN has running containers; another version cannot replace it under them (stop them first)" >&2
        exit 1
    fi
    TARBALL=podman-linux-$ARCH.tar.gz
    URL=https://github.com/mgoltzsche/podman-static/releases/download/$VERSION/$TARBALL
    mkdir -p "$PREFIX/dl"
    echo "  downloading $URL"
    curl -fsSL -o "$PREFIX/dl/$TARBALL" "$URL"
    PIN=${SHA256[$VERSION-$ARCH]:-}
    if [[ -n $PIN ]]; then
        echo "$PIN  $PREFIX/dl/$TARBALL" | sha256sum -c --quiet -
        ok "sha256 matches the pinned digest for $VERSION $ARCH"
    elif command -v gpg >/dev/null; then
        curl -fsSL -o "$PREFIX/dl/$TARBALL.asc" "$URL.asc"
        gpg --batch --keyserver hkps://keyserver.ubuntu.com --recv-keys "$SIGNING_KEY" >/dev/null 2>&1 || true
        if gpg --batch --verify "$PREFIX/dl/$TARBALL.asc" "$PREFIX/dl/$TARBALL" >/dev/null 2>&1; then
            ok "GPG signature by $SIGNING_KEY verifies"
        else
            echo "  the GPG signature of $TARBALL does not verify (key $SIGNING_KEY); not installing" >&2
            exit 1
        fi
    else
        note "no pinned sha256 for $VERSION $ARCH and no gpg to check the signature: installing unverified"
    fi
    rm -rf "$PREFIX/root"
    mkdir -p "$PREFIX/root"
    tar -xzf "$PREFIX/dl/$TARBALL" -C "$PREFIX/root"
    [[ -x $BUNDLE/usr/local/bin/podman ]] || { echo "the tarball did not unpack to $BUNDLE" >&2; exit 1; }
    install_configs
    have_podman || { echo "$PODMAN does not report version ${VERSION#v}" >&2; exit 1; }
    ok "installed $("$PODMAN" --version) at $PODMAN"
fi

# --- 3. rootless prerequisites ---------------------------------------------
echo "[3/7] rootless podman prerequisites"
# A rootless container can hold only the ids the user's subordinate ranges map,
# and an image with a file owned by any other id fails to pull ("insufficient
# UIDs or GIDs"). npm keeps a package publisher's uid, which reaches the
# billions (two SWE-bench Pro images: 120012366, 1516583083), so the user gets
# nearly the whole 32-bit range, not the usual 65536.
SUB_FIRST=100000 SUB_LAST=4294000000 SUB_ENOUGH=4000000000
read -r SUB_COUNT SUB_END < <(awk -F: -v u="$(id -un)" '$1 == u { n += $3; if ($2 + $3 - 1 > e) e = $2 + $3 - 1 }
    END { printf "%d %d\n", n, e }' /etc/subuid)
read -r GID_COUNT GID_END < <(awk -F: -v u="$(id -un)" '$1 == u { n += $3; if ($2 + $3 - 1 > e) e = $2 + $3 - 1 }
    END { printf "%d %d\n", n, e }' /etc/subgid)
SUB_START=$(( (SUB_END > GID_END ? SUB_END : GID_END) + 1 )); (( SUB_START < SUB_FIRST )) && SUB_START=$SUB_FIRST
OTHERS=$(awk -F: -v u="$(id -un)" -v a="$SUB_START" -v b="$SUB_LAST" '$1 != u && $2 <= b && $2 + $3 - 1 >= a { print $1 }' \
    /etc/subuid /etc/subgid | sort -u | tr '\n' ' ')
if (( SUB_COUNT >= SUB_ENOUGH && GID_COUNT >= SUB_ENOUGH )); then
    ok "subordinate uid/gid ranges for $(id -un): $SUB_COUNT ids"
elif [[ -n $OTHERS ]]; then
    gap "$(id -un) has $SUB_COUNT subordinate ids and $SUB_START-$SUB_LAST overlaps the ranges of: $OTHERS-- pick a free range"
elif [[ $CHECK -eq 1 ]]; then
    gap "$(id -un) has $SUB_COUNT subordinate ids: sudo usermod --add-subuids $SUB_START-$SUB_LAST --add-subgids $SUB_START-$SUB_LAST $(id -un) && $PODMAN system migrate"
else
    sudo usermod --add-subuids "$SUB_START-$SUB_LAST" --add-subgids "$SUB_START-$SUB_LAST" "$(id -un)"
    # the per-user pause process must pick the new mapping up; it is shared with any other podman
    [[ -x $PODMAN ]] && "$PODMAN" system migrate >/dev/null 2>&1 || true
    ok "added subuid/subgid $SUB_START-$SUB_LAST for $(id -un)"
fi
CONTROLLERS_FILE=/sys/fs/cgroup/user.slice/user-$(id -u).slice/user@$(id -u).service/cgroup.controllers
CONTROLLERS=$(cat "$CONTROLLERS_FILE" 2>/dev/null || true)
NEED=(cpu memory pids)
LACK=()
for c in "${NEED[@]}"; do
    [[ " $CONTROLLERS " == *" $c "* ]] || LACK+=("$c")
done
if [[ ${#LACK[@]} -eq 0 ]]; then
    ok "cgroup v2 controllers delegated to the user session: $CONTROLLERS"
else
    DROPIN=/etc/systemd/system/user@.service.d/sprout-delegate.conf
    if [[ $CHECK -eq 1 ]]; then
        gap "cgroup controllers not delegated (${LACK[*]}); needs $DROPIN with Delegate=cpu cpuset io memory pids and a re-login"
    else
        sudo mkdir -p "$(dirname "$DROPIN")"
        printf '[Service]\nDelegate=cpu cpuset io memory pids\n' | sudo tee "$DROPIN" >/dev/null
        sudo systemctl daemon-reload
        gap "wrote $DROPIN; log out and back in (or reboot) so the user session gets ${LACK[*]}, then re-run"
    fi
fi
# The user's systemd is what places container cgroups (cgroup_manager systemd).
if systemctl --user show-environment >/dev/null 2>&1; then
    ok "the user's systemd answers (container cgroups are created through it)"
else
    gap "systemctl --user does not answer: podman's systemd cgroup manager needs the user session bus (loginctl enable-linger $(id -un), then log in again)"
fi
if [[ -x $PODMAN ]]; then
    INFO=$("$PODMAN" info --format '{{.Host.Security.Rootless}} {{.Host.CgroupsVersion}} {{.Host.CgroupManager}} {{.Host.NetworkBackend}}' 2>/dev/null || true)
    if [[ $INFO == "true v2 systemd netavark" ]]; then
        ok "$("$PODMAN" version --format 'podman {{.Client.Version}}'): rootless, cgroups v2 through systemd, netavark"
    else
        gap "$PODMAN info does not report rootless + cgroups v2 + systemd + netavark: ${INFO:-(no answer)}"
    fi
fi

# --- 4. the btrfs store -----------------------------------------------------
echo "[4/7] btrfs store at $STORE"
store_ready() {
    local opts
    opts=$(findmnt -no FSTYPE,OPTIONS --target "$STORE" 2>/dev/null || true)
    [[ $opts == btrfs* && $opts == *user_subvol_rm_allowed* ]] && [[ -O $STORE ]]
}
if store_ready; then
    ok "mounted, btrfs, user_subvol_rm_allowed, owned by $(id -un)"
elif [[ $CHECK -eq 1 ]]; then
    gap "no store: sudo $ROOT/ops/btrfs_store.sh $STORE $IMAGE $SIZE"
else
    sudo "$ROOT/ops/btrfs_store.sh" "$STORE" "$IMAGE" "$SIZE"
    store_ready && ok "created and mounted"
fi

# --- 5. python --------------------------------------------------------------
echo "[5/7] python package"
PY=$ROOT/.venv/bin/python
if [[ $SKIP_PYTHON -eq 1 ]]; then
    ok "skipped (--skip-python)"
elif [[ -x $PY ]] && "$PY" -c 'import sprout, sprout_sandbox, minisweagent, harbor, pytest' >/dev/null 2>&1; then
    ok ".venv has sprout with the mini, harbor and dev extras"
elif [[ $CHECK -eq 1 ]]; then
    gap "python: python3 -m venv .venv && .venv/bin/pip install -e '.[mini,harbor,dev]'"
else
    [[ -x $PY ]] || python3 -m venv "$ROOT/.venv"
    "$ROOT/.venv/bin/pip" install -q -e "${ROOT}[mini,harbor,dev]"
    ok "installed into .venv"
fi

# --- 6. the runtime ---------------------------------------------------------
echo "[6/7] runtime binary"
RUNTIME=$ROOT/runtime/sprout-runtime
if [[ -x $RUNTIME && $RUNTIME -nt $ROOT/runtime/main.go ]] && \
   ! find "$ROOT/runtime" -name '*.go' -newer "$RUNTIME" | grep -q .; then
    ok "$RUNTIME is newer than every Go source"
elif [[ $CHECK -eq 1 ]]; then
    gap "runtime: (cd runtime && go build -o sprout-runtime .)"
else
    # go.mod's toolchain line is newer than the distribution's Go; Go 1.21+
    # downloads the toolchain it names (GOTOOLCHAIN=auto, the default).
    (cd "$ROOT/runtime" && go build -o sprout-runtime .)
    ok "built $RUNTIME"
fi

# --- 7. prove it ------------------------------------------------------------
echo "[7/7] verification"
HOW="SPROUT_PODMAN_BIN=$PODMAN SPROUT_PODMAN_STORE=$STORE $ROOT/.venv/bin/python -m pytest -m slow tests/sdk/test_podman_live.py -s"
if [[ $GAPS -gt 0 ]]; then
    echo "  skipped: $GAPS gap(s) above"
    exit 1
fi
if [[ $SKIP_VERIFY -eq 1 || $CHECK -eq 1 ]]; then
    echo "  skipped (--check or --skip-verify); a memory limit must reach a container, and the live test is:"
    echo "    $PODMAN run --rm --memory 64m docker.io/library/alpine:3.20 cat /sys/fs/cgroup/memory.max   # 67108864"
    echo "    $HOW"
else
    # A limit that podman accepts but the kernel never saw is the failure mode of a
    # cgroup manager without a place to make cgroups: ask the container.
    LIMIT=$("$PODMAN" run --rm --memory 64m docker.io/library/alpine:3.20 cat /sys/fs/cgroup/memory.max 2>/dev/null || true)
    if [[ $LIMIT == 67108864 ]]; then
        ok "--memory reaches the container's cgroup (memory.max 67108864 for --memory 64m)"
    else
        echo "  a 64m memory limit did not reach the container (memory.max: ${LIMIT:-no answer}); the user's" >&2
        echo "  systemd must be able to place container cgroups -- see step 3 -- before sandboxes are shaped" >&2
        exit 1
    fi
    if [[ $SKIP_PYTHON -eq 1 ]]; then
        echo "  the live test is skipped (--skip-python):"
        echo "    $HOW"
    # The live tests remove every container and snapshot that appears while they
    # run, so they must have the store to themselves.
    elif [[ -n "$(find "$STORE/sandboxes" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)" ]] || \
         [[ -n "$("$PODMAN" ps -a -q --filter label=sprout.managed=1 2>/dev/null | head -1)" ]]; then
        echo "  skipped: the store is in use by running sandboxes; run the live test when it is idle:"
        echo "    $HOW"
    else
        # The SDK's live test: base extraction of python:3.11-slim, spawn, a command,
        # snapshot, branch from it, network toggle, teardown; it prints the timings.
        mkdir -p "$ROOT/runs"
        LOG=$ROOT/runs/setup-podman-live-test.log
        if (cd "$ROOT" && SPROUT_PODMAN_BIN=$PODMAN SPROUT_PODMAN_STORE=$STORE "$PY" -m pytest -q -p no:cacheprovider -m slow tests/sdk/test_podman_live.py -s > "$LOG" 2>&1); then
            grep -E "spawn|snapshot|network|passed" "$LOG" | sed 's/^/  /'
            ok "the podman backend works on this host"
        else
            echo "  the live test failed; its last lines ($LOG):" >&2
            tail -n 15 "$LOG" | sed 's/^/    /' >&2
            exit 1
        fi
    fi
fi
echo
echo "  podman:  $PODMAN  ($("$PODMAN" --version 2>/dev/null))"
echo "  store:   $STORE"
echo "  Every Sprout entry point runs that podman: it is the default when it is installed here,"
echo "  SPROUT_PODMAN_BIN names another, and \`python -m sprout.services env\` exports the choice:"
echo "    export SPROUT_PODMAN_BIN=$PODMAN"
echo "    eval \"\$(python -m sprout.services env)\"   # after \`python -m sprout.services init\`; SPROUT_PODMAN_STORE too"

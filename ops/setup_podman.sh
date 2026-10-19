#!/usr/bin/env bash
# Set up this host for Sprout's podman backend, then prove that it works: podman 6 for this
# user, a btrfs store, the Python package and the runtime binary.
#
#   ops/setup_podman.sh [--store DIR] [--image FILE] [--size SIZE]
#                       [--podman-prefix DIR] [--podman-version TAG] [--share-images DIR|none]
#                       [--skip-python] [--skip-verify] [--check]
#
# Defaults: --store ~/sprout-btrfs --image /var/lib/sprout-btrfs-rootless.img --size 200G
# --podman-prefix ~/.local/opt/podman --podman-version v6.1.3. The image file is sparse. Size
# it for every base rootfs that you will extract, about the sum of the uncompressed images.
# --share-images DIR adds the image storage of another podman as a read-only store, so its
# images need no second pull. --share-images none drops it. A re-run keeps the last choice.
# --check changes nothing and exits non-zero on a gap.
#
# The script skips each step that is already done. It uses sudo only to install packages, add
# the subordinate id range, write the cgroup delegation drop-in and make the store
# (ops/btrfs_store.sh). Everything after that is rootless. The podman goes under the prefix,
# not on PATH, and a distribution podman stays as it is. Sprout runs $SPROUT_PODMAN_BIN, else
# the podman under the default prefix, else `podman` on PATH.
set -euo pipefail

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
STORE=$HOME/sprout-btrfs
IMAGE=/var/lib/sprout-btrfs-rootless.img
SIZE=200G
PREFIX=$HOME/.local/opt/podman
VERSION=v6.1.3
SHARE='unset'
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
# The venv module ships in a package with a versioned name (python3.12-venv). So test the
# capability, not a package name.
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
# This podman has its own configuration, storage, run root and locks. Two podmans of one user
# must not share state. They share only the per-user rootless pause process.
case "$(uname -m)" in
    x86_64) ARCH=amd64;;
    aarch64) ARCH=arm64;;
    *) echo "no podman-static build for $(uname -m)" >&2; exit 2;;
esac
BUNDLE=$PREFIX/root/podman-linux-$ARCH
PODMAN=$PREFIX/bin/podman
# The prefix keeps its run directory, which holds tmp_dir, runroot and the locks. libpod
# refuses to start when they move under an existing database.
if [[ -f $PREFIX/etc/rundir ]]; then
    RUNDIR=$(<"$PREFIX/etc/rundir")
else
    RUNDIR=${XDG_RUNTIME_DIR:-/run/user/$(id -u)}/sprout-podman
    [[ $PREFIX == "$DEFAULT_PREFIX" ]] || RUNDIR=$RUNDIR-$(printf %s "$PREFIX" | sha256sum | cut -c1-8)
fi
# The prefix keeps --share-images too, so that a plain re-run does not silently drop it.
if [[ $SHARE == unset ]]; then
    SHARE=
    if [[ -f $PREFIX/etc/share-images ]]; then SHARE=$(cat "$PREFIX/etc/share-images" || true); fi
elif [[ $SHARE == none ]]; then
    SHARE=
else
    SHARE=$(realpath -m -- "$SHARE")
fi
# The sha256 of each release tarball that this script has installed. The script verifies
# another version or architecture with the GPG signature of the maintainer instead.
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
    # Render the three generated files into directory $1. Nobody edits them by hand: the next
    # run writes a change of prefix, version or --share-images.
    local into=$1 shared=""
    [[ -n $SHARE ]] && shared=$(printf 'additionalimagestores = ["%s"]\n' "$SHARE")
    mkdir -p "$into/etc" "$into/bin"
    cat > "$into/etc/containers.conf" <<EOF
# podman $VERSION for Sprout's sandboxes, written by ops/setup_podman.sh. $PODMAN selects this
# file through CONTAINERS_CONF. Its storage, run root and locks are apart from any other podman.

[containers]

[network]
network_backend = "netavark"
firewall_driver = "nftables"
network_config_dir = "$PREFIX/storage/networks"
default_rootless_network_cmd = "pasta"

[engine]
# The user's systemd makes container cgroups, so --cpus and --memory apply from any login shell.
# The bundled runc talks to systemd without libsystemd. The bundled crun cannot.
cgroup_manager = "systemd"
runtime = "runc"
events_logger = "file"
# File locks under tmp_dir. The shared-memory lock segment belongs to the distribution's podman.
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
# Image storage for podman $VERSION, written by ops/setup_podman.sh. Native rootless overlay.
# fuse-overlayfs from the bundle is the fallback when the kernel refuses native overlay.
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
# podman $VERSION (static) with its own configuration and storage. Written by ops/setup_podman.sh.
export CONTAINERS_CONF="$PREFIX/etc/containers.conf"
export CONTAINERS_STORAGE_CONF="$PREFIX/etc/storage.conf"
export CONTAINERS_REGISTRIES_CONF="$BUNDLE/etc/containers/registries.conf"
exec "$BUNDLE/usr/local/bin/podman" "\$@"
EOF
    chmod +x "$into/bin/podman"
}

configs_differ() {
    # Return 0 when a file rendered in $1 differs from the installed file, or none is installed.
    local staged=$1 f
    for f in etc/containers.conf etc/storage.conf bin/podman; do
        cmp -s "$staged/$f" "$PREFIX/$f" || return 0
    done
    return 1
}

install_configs() {
    # Render beside the live files and move each into place, so that a podman that starts
    # meanwhile reads a whole file, never a truncated one.
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
    # c/image reads $HOME/.config/containers/policy.json, then /etc/containers/policy.json.
    # A host without any podman package has neither.
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
# A file owned by an id outside the subordinate ranges fails the pull ("insufficient UIDs or
# GIDs"). npm keeps publishers' uids (1516583083 in a SWE-bench Pro image). So take nearly 2^32 ids.
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
    # The per-user pause process must pick up the new mapping. Any other podman of the user shares it.
    if [[ -x $PODMAN ]]; then "$PODMAN" system migrate >/dev/null 2>&1 || true; fi
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
# The user's systemd makes the container cgroups (cgroup_manager systemd).
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
    gap "runtime: (cd runtime && go generate ./... && go build -o sprout-runtime .)"
else
    # Go 1.21+ downloads the newer toolchain that go.mod names (GOTOOLCHAIN=auto). go generate
    # downloads the ripgrep release that the binary embeds.
    (cd "$ROOT/runtime" && go generate ./... && go build -o sprout-runtime .)
    ok "built $RUNTIME"
fi

# --- 7. prove it ------------------------------------------------------------
echo "[7/7] verification"
HOW="SPROUT_PODMAN_BIN=$PODMAN SPROUT_PODMAN_STORE=$STORE $ROOT/.venv/bin/python -m pytest -m slow tests/sprout_sandbox/test_podman_live.py -s"
if [[ $GAPS -gt 0 ]]; then
    echo "  skipped: $GAPS gap(s) above"
    exit 1
fi
if [[ $SKIP_VERIFY -eq 1 || $CHECK -eq 1 ]]; then
    echo "  skipped (--check or --skip-verify); a memory limit must reach a container, and the live test is:"
    echo "    $PODMAN run --rm --memory 64m docker.io/library/alpine:3.20 cat /sys/fs/cgroup/memory.max   # 67108864"
    echo "    $HOW"
else
    # A cgroup manager with no place to make cgroups accepts a limit that never reaches the
    # kernel. So read the limit inside the container.
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
    # The live tests remove each container and snapshot that appears while they run. So they
    # must have the store to themselves.
    elif [[ -n "$(find "$STORE/sandboxes" -mindepth 1 -maxdepth 1 -type d 2>/dev/null | head -1)" ]] || \
         [[ -n "$("$PODMAN" ps -a -q --filter label=sprout.managed=1 2>/dev/null | head -1)" ]]; then
        echo "  skipped: the store is in use by running sandboxes; run the live test when it is idle:"
        echo "    $HOW"
    else
        # The live test prints its spawn, snapshot and network timings. The grep keeps them.
        mkdir -p "$ROOT/runs"
        LOG=$ROOT/runs/setup-podman-live-test.log
        if (cd "$ROOT" && SPROUT_PODMAN_BIN=$PODMAN SPROUT_PODMAN_STORE=$STORE "$PY" -m pytest -q -p no:cacheprovider -m slow tests/sprout_sandbox/test_podman_live.py -s > "$LOG" 2>&1); then
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

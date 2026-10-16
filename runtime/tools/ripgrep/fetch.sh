#!/bin/sh
# Download the ripgrep release grep_files embeds (go:generate in ../ripgrep.go).
# SHA256SUMS names the archive and pins its hash; the archive is not in git.
set -eu
cd "$(dirname "$0")"
read -r _ name < SHA256SUMS
version=${name#ripgrep-}
version=${version%%-*}
for stale in *.tar.gz; do
    # go:embed takes every archive here; only the pinned one belongs.
    [ "$stale" = "$name" ] || rm -f "$stale"
done
if ! sha256sum --check --status SHA256SUMS 2>/dev/null; then
    curl -fsSL --retry 3 -o "$name.part" \
        "https://github.com/BurntSushi/ripgrep/releases/download/$version/$name"
    mv "$name.part" "$name"
    sha256sum --check --quiet SHA256SUMS || { rm -f "$name"; exit 1; }
fi

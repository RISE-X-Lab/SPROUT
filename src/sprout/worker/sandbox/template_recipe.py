"""What a per-image microVM template is: the files baked into it, the command
that starts the runtime (under the image's OCI ENV when asked), and the
content-addressed name that identifies that recipe."""

from __future__ import annotations

import hashlib
from pathlib import Path

import httpx

#: Where the runtime lands in the template, and how it is started.
RUNTIME_PATH = "/usr/local/bin/sprout-runtime"
DEFAULT_RUNTIME_PORT = 3000

#: ripgrep baked in beside the runtime, so `rg` is on every sandbox shell's PATH
#: (grep_files runs the copy embedded in the runtime, not this one).
RIPGREP_PATH = "/usr/local/bin/rg"
RIPGREP_VERSION = "14.1.1"
RIPGREP_URL = (
    "https://github.com/BurntSushi/ripgrep/releases/download/"
    f"{RIPGREP_VERSION}/ripgrep-{RIPGREP_VERSION}-x86_64-unknown-linux-musl.tar.gz"
)


class TemplateError(RuntimeError):
    """A template could not be built, so the instance cannot run."""


def template_name(
    image: str, runtime_fingerprint: str, port: int, resources: dict | None = None, salt: str = ""
) -> str:
    """A stable, legal template name for one (image, runtime, port, shape).

    Content-addressed rather than derived from the image name: template names
    are length-limited and the alias grammar is narrow, while image names
    carry slashes, colons and dots. Hashing the runtime's fingerprint in too
    means a rebuilt runtime binary cannot land on a template built with the
    old one.

    The shape is part of the identity because a microVM's CPU and memory are
    fixed by its template: a 16 GB task must not be handed the 1 GB template
    an earlier run built from the same image.

    ``salt`` is anything else that changes what the template *does* -- the
    start command's environment, say. Empty by default so every name computed
    before it existed is unchanged.
    """
    shape = ""
    if resources:
        shape = f"{resources.get('cpu', '')}x{resources.get('memory_mb', '')}"
        if "disk_size_mb" in resources:
            shape += f"x{resources['disk_size_mb']}"
    parts = [image, runtime_fingerprint, str(port), shape]
    if salt:
        parts.append(salt)
    digest = hashlib.sha256("\0".join(parts).encode()).hexdigest()[:24]
    return f"sprout-swebench-{digest}"


# --- the image's own environment ---------------------------------------------
#
# The guest agent that launches startCmd hands the runtime most of the image's
# ENV but rebuilds PATH from its own defaults. Measured on a DeepSWE image:
# image PATH `/opt/venv/bin:/root/.cargo/bin:/root/.local/bin:/root/.rye/shims:
# ...`, runtime PATH `/root/.bun/bin:/root/.cargo/bin:/usr/local/sbin:...` --
# /opt/venv/bin gone, so `pytest` was "not found" and `python` was the system
# interpreter, not the project's. Across all 113 DeepSWE images, every one had
# PATH entries the runtime lacked (/root/go/bin x34, /app/node_modules/.bin,
# /opt/venv/bin, ...). Faithful grading needs the image's environment, so a
# builder can be asked to read the OCI config and launch the runtime under it.
# Opt-in: the SWE-bench templates that produced the measured results keep
# their names and their behaviour.

REGCTL_DEPS_GLOB = "/var/lib/aenv/deps/regctl/*/regctl"


def find_regctl() -> Path | None:
    """regctl on this host: ``$SPROUT_REGCTL``, PATH, or AgentENV's own deps copy.

    AgentENV downloads regctl to resolve images, so a host that can run
    sandboxes already has one; it also carries whatever registry logins the
    server uses (Docker Hub, public ECR and ghcr all work anonymously here).
    """
    import glob
    import os
    import shutil

    candidate = os.environ.get("SPROUT_REGCTL")
    if candidate and Path(candidate).is_file():
        return Path(candidate)
    found = shutil.which("regctl")
    if found:
        return Path(found)
    for path in sorted(glob.glob(REGCTL_DEPS_GLOB), reverse=True):
        if Path(path).is_file():
            return Path(path)
    return None


def image_config_env(image: str, regctl: Path, timeout: float = 120.0) -> list[str]:
    """``KEY=VALUE`` entries of the image's OCI config ``Env``, in order."""
    import json
    import subprocess

    try:
        proc = subprocess.run(
            [str(regctl), "image", "config", image, "--format", "{{json .Config.Env}}"],
            capture_output=True,
            text=True,
            timeout=timeout,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise TemplateError(f"could not read {image}'s config with regctl: {exc}") from exc
    if proc.returncode != 0:
        raise TemplateError(f"regctl could not read {image}'s config: {proc.stderr.strip()[:300]}")
    try:
        env = json.loads(proc.stdout.strip() or "null") or []
    except ValueError as exc:
        raise TemplateError(f"regctl returned no JSON for {image}: {proc.stdout[:200]}") from exc
    return [str(e) for e in env if "=" in str(e)]


def start_command(port: int, env: list[str] | None = None) -> str:
    """The template's startCmd: the runtime, under the image's ENV when given.

    ``env K=V ... runtime`` rather than exporting in a shell profile: it is
    exactly the image's list, applied to exactly the process whose children
    are every tool call, and nothing the guest agent adds (its E2B_* variables,
    HOME) is lost -- only the listed keys are overridden.
    """
    import shlex

    base = f"{RUNTIME_PATH} --port {port}"
    if not env:
        return base
    return "env " + " ".join(shlex.quote(item) for item in env) + " " + base


def env_salt(env: list[str]) -> str:
    return "env:" + hashlib.sha256("\0".join(env).encode()).hexdigest()[:12]


def runtime_fingerprint(runtime_bin: Path) -> str:
    """Content hash of the runtime binary, so a rebuild forces a new template."""
    digest = hashlib.sha256()
    with open(runtime_bin, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()[:16]


def ensure_ripgrep(cache_dir: Path | None = None) -> Path | None:
    """A ripgrep binary to bake into templates, downloaded once per host.

    Returns ``None`` when it cannot be fetched -- templates still build, but
    without rg on the sandbox PATH, so the environment differs from one built
    with it (hence the warning).
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "sprout-swebench"
    binary = cache_dir / f"rg-{RIPGREP_VERSION}"
    if binary.is_file():
        return binary
    import io
    import tarfile

    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
        with httpx.Client(follow_redirects=True, timeout=120.0) as client:
            payload = client.get(RIPGREP_URL)
            payload.raise_for_status()
        with tarfile.open(fileobj=io.BytesIO(payload.content), mode="r:gz") as tar:
            member = next(m for m in tar.getmembers() if m.name.endswith("/rg") and m.isfile())
            extracted = tar.extractfile(member)
            assert extracted is not None
            tmp = binary.with_suffix(".tmp")
            tmp.write_bytes(extracted.read())
            tmp.chmod(0o755)
            tmp.rename(binary)
        return binary
    except Exception as exc:  # noqa: BLE001 -- optional asset, never fatal
        print(
            f"note: could not fetch ripgrep to bake into templates ({exc}); "
            "sandboxes built from these templates have no rg on PATH"
        )
        return None

"""The persistent resources an experiment host needs, started once and kept up.

``python -m sprout.services`` owns what used to be hand-started before every
batch: the Run Store's PostgreSQL (a rootless podman container on a local
port), the LiteLLM proxy in front of the providers Sprout cannot talk to
directly, the Run Store API and its worker, the RL driver, plus three things it
only checks -- the podman the sandboxes run on, its btrfs store and, optionally,
the AgentENV microVM server (root; started by hand). Eight commands:

    init              write ~/.config/sprout/services.yaml and the per-service
                      config files from the packaged templates, make the secrets
    up [NAME...]      start what is not already healthy, dependencies first
    down [NAME...]    stop, dependents first
    restart NAME...   down then up for the named services (and their dependents)
    status            one line per service: name | kind | pid | health | port
    logs NAME [-f]    the service's log (a container's: `podman logs`)
    env               shell export lines an experiment shell evaluates
    doctor            every check and every health probe, starting nothing;
                      exit 1 when something is wrong

Everything the CLI runs is logged under ~/.local/state/sprout/services/<name>/log,
the generated secrets live in secrets/ there with mode 0600, and the two roots
move with SPROUT_SERVICES_CONFIG_DIR / SPROUT_SERVICES_STATE_DIR. The package
imports nothing from the rest of Sprout (stdlib and PyYAML only), so it works
before the optional extras are installed and never inherits a bug from the
layers it starts. docs/SERVICES.md is the page about it.
"""

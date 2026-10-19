"""The persistent resources of an experiment host, started once and kept up.

``python -m sprout.services`` runs the Run Store's PostgreSQL (a rootless podman container),
the LiteLLM proxy, the Run Store API and its worker, and the RL driver. It only checks the
sandboxes' podman, its btrfs store and, optionally, the AgentENV microVM server. Eight commands:

    init              write ~/.config/sprout/services.yaml, its config files and the secrets
    up [NAME...]      start what is not healthy yet, dependencies first
    down [NAME...]    stop, dependents first
    restart NAME...   down, then up, for the named services and their dependents
    status            one line per service: name | kind | pid | health | port
    logs NAME [-f]    the service's log (for a container, `podman logs`)
    env               shell export lines for an experiment shell to evaluate
    doctor            run every check and health probe, start nothing, exit 1 on a failure

The CLI logs what it runs to ~/.local/state/sprout/services/<name>/log, and keeps the secrets
in secrets/ there, mode 0600. SPROUT_SERVICES_CONFIG_DIR and SPROUT_SERVICES_STATE_DIR move the
two roots. The package imports no Sprout layer, so it works before the optional extras are
installed. Details: docs/SERVICES.md.
"""

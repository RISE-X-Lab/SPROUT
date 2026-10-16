# gate

Repo-specific checks every commit has to pass. Each is an executable in `checks/`;
`run.sh` runs them all from the repo root and fails if any exits non-zero. Commits
run it through `hooks/pre-commit`, enabled once per clone:

    git config core.hooksPath gate/hooks

CI runs `gate/run.sh` too, so a commit made with `--no-verify` still meets it.

| Check | Fails when |
| --- | --- |
| `checks/runtime_tools.py` | the runtime serves, or `runtime/schema/tools.json` declares, a tool other than `shell`, `text_editor`, `grep_files`, `web_fetch`, `web_search` |
| `checks/no_host_paths.py` | a tracked file names a path in a home directory (`/home/<user>/...`, `/Users/<user>/...`); examples and fixtures use placeholders under `/path/to/` |
| `checks/docs_layout.py` | an image sits outside `docs/figures/`, `docs/` holds anything but markdown the top README links to, or a figure is referenced by neither README.md nor `docs/*.md` |

A new gate is a new executable in `checks/`; one that is not executable fails the run
rather than being skipped.

`pending/` holds gates the repository does not pass yet; `run.sh` does not run them.
Each moves into `checks/` once the code meets it.

| Pending | Fails when |
| --- | --- |
| `pending/src_layout.py` | a package under `src/` nests another (except `worker/agents/<agent>/`, `worker/tools/interceptors/`), holds more than 10 files, holds anything but `.py`, or a file is over 500 lines |
| `pending/type_hints.py` | `mypy` (strict, warn_unreachable; `[tool.mypy]` in pyproject.toml) reports anything in `src/` |
| `pending/lint.py` | `ruff check` or `ruff format --check` (pyproject.toml), `gofmt -l`, `go vet`, `staticcheck` (runtime/) or `shellcheck` reports anything |
| `pending/comment_length.py` | a file docstring is over 20 lines, a function or class docstring over 5, or another run of whole-line comments over 2 (Go and shell: the block opening the file, and the block above a declaration) |

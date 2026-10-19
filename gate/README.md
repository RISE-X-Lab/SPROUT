# gate

The gates are the repository checks that each commit must pass. Each gate is an executable file in
`gate/checks/`. `gate/run.sh` runs all of them, and the pre-commit hook and CI run `gate/run.sh`.
Read this page when a gate rejects your commit, or when you add a gate. For each gate,
[The gates](#the-gates) says what it rejects and how to fix a failure.

## Set up the gates

Do these steps one time in each clone, from the repository root:

1. Install the package with its development tools: `pip install -e '.[dev]'`. This installs
   `ruff` and `mypy`.
2. Put these tools on `PATH`:
   - `go` and `gofmt`, from the Go toolchain
   - `shellcheck`
3. Download the ripgrep release that the runtime embeds: run `go generate ./...` in `runtime/`.
   If you do not, `lint.py` and `runtime_tools.py` fail with
   `pattern ripgrep/*.tar.gz: no matching files found`.
4. Enable the hook: `git config core.hooksPath gate/hooks`.

After step 4, git runs `gate/hooks/pre-commit` before each commit. The hook runs `gate/run.sh`. If
a gate fails, git does not make the commit.

Each Python gate runs with the `python3` on `PATH`. So activate the environment that has the `dev`
extra before you commit or run a gate.

## Run the gates

To run all gates, run `./gate/run.sh`. To run one gate, run its file, for example
`gate/checks/lint.py`.

`run.sh` does these steps:

1. It changes to the repository root. If `git rev-parse --show-toplevel` fails, it exits 2.
2. It takes each entry of `gate/checks/` in the order of the shell glob. It skips an entry that is
   not a file, such as `__pycache__`.
3. If the file is not executable, it prints `gate: <file> is not executable` and counts a failure.
   So a gate that cannot run never reads as a pass.
4. Otherwise, it runs the file. It prints `gate: pass  <file>` when the file exits 0, and
   `gate: FAIL  <file>` for any other exit code.
5. After the last gate, it exits 1 if a gate failed, and 0 if not.

`run.sh` runs every gate, also after a failure. So one run shows all failures.

The CI workflow `.github/workflows/ci.yml` also runs `./gate/run.sh`, in its `python` job. The job
runs the gates after it installs the package and builds the runtime. The workflow runs on these
events:

| Event | Branches |
|---|---|
| push | `main`, `dev`, `feat/**` |
| pull request | `main`, `dev` |

So a commit made with `--no-verify` still meets the gates on those branches.

## The gates

The repository has these gates:

| Gate | Rejects | Exit codes |
|---|---|---|
| `comment_length.py` | A docstring or a comment over its length limit. | 0 pass, 1 finding |
| `docs_layout.py` | An image outside `docs/figures/`, a file in `docs/` that nothing links to, or a file of the wrong type in `docs/`. | 0 pass, 1 finding |
| `lint.py` | A finding of `ruff`, `gofmt`, `go vet`, `staticcheck` or `shellcheck`. | 0 pass, 1 finding, 2 tool missing |
| `no_host_paths.py` | A path in a home directory, in a tracked file. | 0 pass, 1 finding |
| `runtime_tools.py` | A runtime tool outside the allowed set. | 0 pass, 1 finding, 2 setup error |
| `src_layout.py` | A package under `src/` that is too deep or too large, a file that is not Python, or a file that is too long. | 0 pass, 1 finding |
| `type_hints.py` | Any error that `mypy` reports. | 0 pass, the exit code of `mypy`, 2 `mypy` missing |

### `comment_length.py`

The gate rejects a comment over its limit, in each tracked Python, Go or shell file:

| Comment | Limit |
|---|---|
| a file docstring | 20 lines |
| a function or class docstring | 5 lines |
| any other run of consecutive whole-line comments | 2 lines |

Each language has its own docstrings:

- Python: the string literal that opens the module, the function or the class.
- Go: the comment block that opens the file is the file docstring. A comment block directly above
  a declaration is the docstring of that declaration. A declaration is a line that starts with one
  of these keywords:
  - `func`
  - `type`
  - `var`
  - `const`

  Comment markers in strings and runes do not count.
- Shell: the comment block that opens the file, after the shebang, is the file docstring. A comment
  block directly above a shell function is its docstring. The gate does not parse the script, so
  a `#` line in a heredoc counts as a comment too.

A file is a shell file if its name ends in `.sh`. A file without a suffix is a shell file if its
shebang ends in `sh` or `bash`.

To run the gate alone, run `gate/checks/comment_length.py`. It prints one line for each finding,
for example `<file>:<line>: docstring is 7 lines (max 5)`.

To fix a failure, shorten the comment to its limit. Put design rationale and measurements in
`docs/*.md`, and the reason for a change in the commit message.

### `docs_layout.py`

The gate checks the place of each image and of each file in `docs/`, and that a document links to
each one. It checks each tracked file of the repository against these rules:

| Rule | Message | Fix |
|---|---|---|
| An image is in `docs/figures/`. | `<path>: an image outside docs/figures/` | Move the image to `docs/figures/`. |
| `docs/figures/` holds images only. | `<path>: docs/figures/ holds images only` | Move the file out of `docs/figures/`. |
| `README.md` or a markdown file under `docs/` names each image in `docs/figures/`. | `<path>: referenced by neither README.md nor docs/*.md` | Refer to the figure from `README.md` as `docs/figures/<name>`, or from a file in `docs/` as `figures/<name>`. Or delete the figure. |
| `docs/` outside `docs/figures/` holds markdown only. | `<path>: docs/ outside docs/figures/ holds markdown only` | Move the file out of `docs/`. |
| `docs/decisions/README.md` names each ADR in `docs/decisions/`. | `<path>: docs/decisions/README.md does not list it` | Add the ADR to `docs/decisions/README.md`. |
| `README.md` names each other markdown file under `docs/`. | `<path>: README.md does not link to it` | Link the document from `README.md`. |

An image is a file with one of these suffixes, in upper or lower case:

- `.png`
- `.svg`
- `.jpg` and `.jpeg`
- `.gif`
- `.webp`
- `.bmp`
- `.ico`
- `.tif` and `.tiff`

A file names a path when the path occurs in its text. The gate accepts the path from the
repository root or the path relative to the file. A link is enough, and any other mention of the
path is enough too.

To run the gate alone, run `gate/checks/docs_layout.py`.

### `lint.py`

The gate rejects any finding of these tools:

| Tool | Checks | Fix |
|---|---|---|
| `ruff check` | The Python files, as `[tool.ruff]` in `pyproject.toml` configures it. | Run `python -m ruff check --fix .`, then fix the rest by hand. |
| `ruff format --check` | The format of the Python files. | Run `python -m ruff format .`. |
| `go vet ./...` | The Go code in `runtime/`. | Fix each finding. |
| `staticcheck ./...` | The Go code in `runtime/`. The gate runs `honnef.co/go/tools/cmd/staticcheck@2025.1.1` through `go run`. | Fix each finding. |
| `gofmt -l .` | The format of the Go files in `runtime/`. The gate prints `runtime/<file>: not gofmt-formatted`. | Run `gofmt -w <file>` in `runtime/`. |
| `shellcheck` | Each tracked `.sh` file, and `gate/hooks/pre-commit`. | Fix each finding. |

The gate needs these tools:

- `ruff`, as a Python module of the `python3` that runs the gate
- `go`
- `gofmt`
- `shellcheck`

If one is missing, the gate prints `lint: not installed: <tools> (pip install -e '.[dev]')` and
exits 2. The `dev` extra installs only `ruff`. Install the other tools as
[Set up the gates](#set-up-the-gates) says.

The first `go run` of `staticcheck` downloads it, so it needs the network or a module cache.

To run the gate alone, run `gate/checks/lint.py`.

### `no_host_paths.py`

The gate rejects a path in a home directory, in each tracked text file. Such a path is the layout of
one machine. Copied out of an example, it points nowhere. A fixture that holds one is correct only
on the host that recorded it.

The gate matches `/home/<user>` and `/Users/<user>` at the start of a path only. So `/opt/x/home/y`
does not match, and a path that starts with `~` does not match. The gate skips a file that holds a
NUL byte, because that file is binary. It prints one line for each match:
`<file>:<line>: <path> -- use a placeholder under /path/to/`.

To run the gate alone, run `gate/checks/no_host_paths.py`.

To fix a failure, replace the path with a placeholder under `/path/to/`, such as
`/path/to/store`.

### `runtime_tools.py`

The gate rejects a runtime tool outside the allowed set. The tools of the runtime are the whole
action space of the agent, so a new tool changes what every run can do. The gate reads two
declarations:

- the tools that the runtime source serves, from `go run . --dump-schema` in `runtime/`, on the
  working tree
- `runtime/schema/tools.json`, which the worker checks each tool call against

The allowed set is `ALLOWED` in `gate/checks/runtime_tools.py`:

- `shell`
- `text_editor`
- `grep_files`
- `web_fetch`
- `web_search`

The gate does not compare the two declarations with each other. A test in
`tests/worker/tools/test_schema.py` does that.

To run the gate alone, run `gate/checks/runtime_tools.py`.

To fix a failure, do one of these:

- If you added the tool on purpose, add it to `ALLOWED` in the same commit. Also add a row for it
  to the `runtime/README.md` section "Runtime toolset support". `tests/repo/test_docs.py` checks
  that the table matches `runtime/schema/tools.json`.
- If you did not intend the tool, remove it from the runtime.

If the gate exits 2, it could not ask the runtime. Put `go` on `PATH`, and run `go generate ./...`
in `runtime/`. The gate prints the error of `go run . --dump-schema`.

### `src_layout.py`

The gate keeps `src/` as shallow packages of a few short Python files. It checks each tracked file
under `src/` against these rules:

| Rule | Limit | Message | Fix |
|---|---|---|---|
| The depth of a directory, with `src/sprout` and `src/sprout_sandbox` as level 1. | 5 | `src/<dir>/: <n> levels deep (max 5)` | Move the package up. |
| The files directly in one directory, `__init__.py` included. | 10 | `src/<dir>/: <n> files (max 10)` | Split the package into subpackages. |
| The type of each file. | `.py` only | `<file>: not Python source; data lives in resources/` | Move the data file to `resources/` at the repository root. |
| The lines of a Python file. | 1000 | `<file>: <n> lines (max 1000)` | Split the file. |

To run the gate alone, run `gate/checks/src_layout.py`.

### `type_hints.py`

The gate runs `python -m mypy` from the repository root. It rejects any error that `mypy` reports.
`[tool.mypy]` in `pyproject.toml` sets these options:

- `files`: `src/sprout` and `src/sprout_sandbox`
- `strict = true`: each parameter and each return value needs an annotation.
- `warn_unreachable = true`: `mypy` reports a `None` check on a value whose type cannot be `None`
  as unreachable code.

If `mypy` is not installed for the `python3` that runs the gate, the gate exits 2. It prints
`type_hints: mypy is not installed for <python>; pip install -e '.[dev]'`. Otherwise, it exits with
the exit code of `mypy`.

To run the gate alone, run `gate/checks/type_hints.py`.

To fix a failure, fix each error that `mypy` reports.

## Add a gate

Do these steps to add a gate:

1. Write the check as a file in `gate/checks/`.
2. Make the file executable. If it is not executable, `run.sh` counts a failure.
3. Make the check exit 0 when the repository passes, and another code when it fails. The other
   gates exit 1 for a finding and 2 for a missing tool.
4. Make the check print each finding on stderr, as the other Python gates do.
5. Run `./gate/run.sh`. The output shows `gate: pass  gate/checks/<name>`.
6. Add the gate to the table and to a section of this page.

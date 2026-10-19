# sprout-runtime

`runtime/` holds the source of `sprout-runtime`, the Go binary that runs inside every sandbox.

## Runtime toolset support

The runtime serves the tools of the table over JSON-RPC, with the methods `tools/list` and
`tools/call`. `runtime/schema/tools.json` is the checked-in declaration of these tools.
`ToolAccess` rejects a call of a tool that the declaration does not list, and a call with an
argument that the tool does not declare. A rejected call does not run, and `ToolAccess` records
it in the journal as `tool.rejected`. The model of mini-swe-agent sees one `bash` tool, which
runs as `shell`.

| Tool | Arguments (first ones required) | Does | Counts as a change |
| --- | --- | --- | --- |
| `shell` | `command`; `working_dir`, `timeout`, `max_output_bytes` | runs `sh -c`, and returns the exit code, stdout and stderr | always |
| `text_editor` | `command` (`view`, `str_replace`, `insert`, `write`), `path`; `view_range`, `old_str`, `new_str`, `insert_line`, `insert_text`, `file_text`, `max_output_bytes` | views or edits one file | except `view` |
| `grep_files` | `pattern`; `path`, `include`, `limit`, `max_output_bytes` | searches files with the ripgrep that the runtime embeds | no |
| `web_fetch` | `url`; `format` (`markdown`, `text`, `html`), `headers`, `timeout`, `max_output_bytes` | fetches a page | no |
| `web_search` | `query`; `backend`, `max_results` | searches the web through one or more engines of ddgs-go | no |

These rules apply to the tools:

- "Counts as a change" decides whether the turn gets a snapshot of its own
  ([Checkpoint every turn](../docs/ARCHITECTURE.md#checkpoint-every-turn)). Each tool that
  `READ_ONLY_TOOLS` in `src/sprout/worker/tools/interceptors/mutation.py` does not list counts as
  a change. The one exception is the `view` command of `text_editor`.
- If the output of a call is longer than `max_output_bytes`, the runtime keeps the first 40% and
  the last 60%, and the output says so. A call without `max_output_bytes` gets 15 MiB.
- `text_editor` and `grep_files` run as the runtime's own process. So a relative path resolves
  against the working directory of the runtime, not of the task.

`runtime/tools/ripgrep/SHA256SUMS` pins the ripgrep release that `grep_files` embeds. Git does
not track the release itself. Before a build, `go generate` downloads it:

```bash
cd runtime && go generate ./... && go build -o sprout-runtime .
```

When you add a tool to the runtime, do these steps in the same commit:

1. Add the tool to `ALLOWED` in `gate/checks/runtime_tools.py`.
2. In `runtime/`, write the new declaration with `go run . --dump-schema > schema/tools.json`.
3. Add a row for the tool to the table above.

The gate fails a commit if the runtime serves, or `runtime/schema/tools.json` declares, a tool
outside `ALLOWED`. `tests/repo/test_docs.py` fails if the table differs from
`runtime/schema/tools.json`.

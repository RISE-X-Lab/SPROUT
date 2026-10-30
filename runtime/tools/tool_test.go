// Tests of what every tool shares, and the helpers of the tests in package tools_test.

package tools_test

import (
	"os"
	"path/filepath"
	"syscall"
	"testing"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

func assertResult(t *testing.T, got, want tools.Result) {
	t.Helper()
	if got != want {
		t.Errorf("result = Success %v, Output %q, Error %q\nwant     Success %v, Output %q, Error %q",
			got.Success, got.Output, got.Error, want.Success, want.Output, want.Error)
	}
}

func readFile(t *testing.T, path string) string {
	t.Helper()
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatalf("reading %s: %v", path, err)
	}
	return string(data)
}

// writeFile writes content to name in a new temporary directory, and returns its path.
func writeFile(t *testing.T, name, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), name)
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("writing %s: %v", path, err)
	}
	return path
}

func TestACallWithoutItsRequiredArgumentIsRefused(t *testing.T) {
	cases := []struct {
		name string
		tool tools.Tool
		args map[string]any
		want string
	}{
		{"shell without command", &tools.ShellTool{}, map[string]any{}, "command is required"},
		{"text_editor without path", &tools.EditTool{}, map[string]any{"command": "view"}, "path is required"},
		{"grep_files without pattern", &tools.GrepTool{}, map[string]any{}, "pattern is required"},
		{"web_fetch without url", &tools.WebFetchTool{}, map[string]any{}, "url is required"},
		{"web_search without query", &tools.WebSearchTool{}, map[string]any{}, "query is required"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := tc.tool.Execute(tc.args)

			assertResult(t, got, tools.Result{Success: false, Error: tc.want})
		})
	}
}

// ownerOf returns the uid and gid that own path.
func ownerOf(t *testing.T, path string) [2]uint32 {
	t.Helper()
	info, err := os.Stat(path)
	if err != nil {
		t.Fatalf("stat %s: %v", path, err)
	}
	stat := info.Sys().(*syscall.Stat_t)
	return [2]uint32{stat.Uid, stat.Gid}
}

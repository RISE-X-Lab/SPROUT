package tools

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func TestEditCommandsChangeTheFileTheyName(t *testing.T) {
	tool := &EditTool{}
	path := filepath.Join(t.TempDir(), "sub", "a.txt")

	if r := tool.Execute(map[string]any{"command": "write", "path": path, "file_text": "one\ntwo\nthree"}); !r.Success {
		t.Fatalf("write failed: %s", r.Error)
	}
	if r := tool.Execute(map[string]any{"command": "str_replace", "path": path, "old_str": "two", "new_str": "TWO"}); !r.Success {
		t.Fatalf("str_replace failed: %s", r.Error)
	}
	if r := tool.Execute(map[string]any{"command": "insert", "path": path, "insert_line": float64(0), "insert_text": "zero"}); !r.Success {
		t.Fatalf("insert failed: %s", r.Error)
	}
	data, _ := os.ReadFile(path)
	if string(data) != "zero\none\nTWO\nthree" {
		t.Fatalf("unexpected file content:\n%s", data)
	}

	view := tool.Execute(map[string]any{"command": "view", "path": path, "view_range": []any{float64(2), float64(3)}})
	if !view.Success || view.Output != "       2 | one\n       3 | TWO\n" {
		t.Fatalf("unexpected view: %#v", view)
	}
}

func TestEditRefusesWhatItCannotDoExactly(t *testing.T) {
	tool := &EditTool{}
	path := filepath.Join(t.TempDir(), "a.txt")
	os.WriteFile(path, []byte("x\nx\n"), 0644)

	for _, c := range []struct {
		args map[string]any
		want string
	}{
		{map[string]any{"command": "view"}, "path is required"},
		{map[string]any{"command": "delete", "path": path}, "unknown command: delete"},
		{map[string]any{"command": "str_replace", "path": path, "old_str": "x", "new_str": "y"}, "Multiple matches (2)"},
		{map[string]any{"command": "str_replace", "path": path, "old_str": "z", "new_str": "y"}, "No match found"},
		{map[string]any{"command": "insert", "path": path, "insert_line": float64(9), "insert_text": "y"}, "insert_line out of range"},
	} {
		r := tool.Execute(c.args)
		if r.Success || !strings.Contains(r.Error, c.want) {
			t.Errorf("%v: want an error containing %q, got %#v", c.args, c.want, r)
		}
	}
	if data, _ := os.ReadFile(path); string(data) != "x\nx\n" {
		t.Fatalf("a refused edit changed the file:\n%s", data)
	}
}

func TestEditViewIsBoundedAndSaysSo(t *testing.T) {
	path := filepath.Join(t.TempDir(), "big.txt")
	os.WriteFile(path, []byte(strings.Repeat("line\n", 2000)), 0644)

	r := (&EditTool{}).Execute(map[string]any{"command": "view", "path": path, "max_output_bytes": float64(1024)})
	if !r.Success || !strings.Contains(r.Output, "file larger than 1024 bytes") ||
		!strings.Contains(r.Output, "output truncated") {
		t.Fatalf("expected a bounded view that says so, got:\n%.400s", r.Output)
	}
}

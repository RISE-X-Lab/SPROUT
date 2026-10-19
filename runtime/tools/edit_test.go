// Tests of the text_editor tool. An edit either changes the file as asked and answers "OK", or
// is refused with the reason and leaves the file as it was.

package tools_test

import (
	"path/filepath"
	"strings"
	"testing"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

func TestWriteLeavesExactlyFileTextInTheFile(t *testing.T) {
	cases := []struct {
		name string
		path string
	}{
		{"a new file in missing directories", filepath.Join(t.TempDir(), "sub", "deeper", "a.txt")},
		{"an existing longer file", writeFile(t, "a.txt", "a much longer old content\n")},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := (&tools.EditTool{}).Execute(map[string]any{"command": "write", "path": tc.path, "file_text": "new\n"})

			assertResult(t, got, tools.Result{Success: true, Output: "OK"})
			if content := readFile(t, tc.path); content != "new\n" {
				t.Errorf("file = %q, want %q", content, "new\n")
			}
		})
	}
}

func TestStrReplaceReplacesTheOneOccurrenceOfOldStr(t *testing.T) {
	cases := []struct {
		name   string
		oldStr string
		newStr string
		want   string
	}{
		{"with new text", "two", "TWO", "one\nTWO\nthree\n"},
		{"with an empty new_str, which deletes old_str", "two\n", "", "one\nthree\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, "a.txt", "one\ntwo\nthree\n")

			got := (&tools.EditTool{}).Execute(map[string]any{
				"command": "str_replace", "path": path, "old_str": tc.oldStr, "new_str": tc.newStr,
			})

			assertResult(t, got, tools.Result{Success: true, Output: "OK"})
			if content := readFile(t, path); content != tc.want {
				t.Errorf("file = %q, want %q", content, tc.want)
			}
		})
	}
}

func TestInsertPutsTheTextAfterTheGivenLine(t *testing.T) {
	cases := []struct {
		name       string
		insertLine float64
		want       string
	}{
		{"0, the start", 0, "new\none\ntwo\nthree"},
		{"3, the last line", 3, "one\ntwo\nthree\nnew"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, "a.txt", "one\ntwo\nthree")

			got := (&tools.EditTool{}).Execute(map[string]any{
				"command": "insert", "path": path, "insert_line": tc.insertLine, "insert_text": "new",
			})

			assertResult(t, got, tools.Result{Success: true, Output: "OK"})
			if content := readFile(t, path); content != tc.want {
				t.Errorf("file = %q, want %q", content, tc.want)
			}
		})
	}
}

func TestARefusedEditSaysWhyAndLeavesTheFileUnchanged(t *testing.T) {
	cases := []struct {
		name string
		args map[string]any
		want string
	}{
		{"str_replace without old_str", map[string]any{"command": "str_replace", "new_str": "y"},
			"old_str is required for str_replace"},
		{"str_replace with an empty old_str", map[string]any{"command": "str_replace", "old_str": "", "new_str": "y"},
			"old_str is required for str_replace"},
		{"str_replace without new_str", map[string]any{"command": "str_replace", "old_str": "one"},
			"new_str is required for str_replace"},
		{"str_replace of text that is not in the file", map[string]any{"command": "str_replace", "old_str": "four", "new_str": "y"},
			"No match found"},
		{"str_replace of text that is in the file twice", map[string]any{"command": "str_replace", "old_str": "o", "new_str": "y"},
			"Multiple matches (2). old_str must be unique."},
		{"insert without insert_line", map[string]any{"command": "insert", "insert_text": "y"},
			"insert_line is required for insert"},
		{"insert without insert_text", map[string]any{"command": "insert", "insert_line": 1.0},
			"insert_text is required for insert"},
		{"insert before the start", map[string]any{"command": "insert", "insert_line": -1.0, "insert_text": "y"},
			"insert_line out of range: got -1, want 0..3"},
		{"insert after the end", map[string]any{"command": "insert", "insert_line": 4.0, "insert_text": "y"},
			"insert_line out of range: got 4, want 0..3"},
		{"write without file_text", map[string]any{"command": "write"},
			"file_text is required for write"},
		{"an unknown command", map[string]any{"command": "delete"},
			"unknown command: delete"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, "a.txt", "one\ntwo\nthree")
			tc.args["path"] = path

			got := (&tools.EditTool{}).Execute(tc.args)

			assertResult(t, got, tools.Result{Success: false, Error: tc.want})
			if content := readFile(t, path); content != "one\ntwo\nthree" {
				t.Errorf("file = %q, want it unchanged", content)
			}
		})
	}
}

func TestViewNumbersEachLineOfViewRangeClampedToTheFile(t *testing.T) {
	cases := []struct {
		name string
		args map[string]any
		want string
	}{
		{"no view_range", map[string]any{},
			"       1 | one\n       2 | two\n       3 | three\n"},
		{"a range inside the file", map[string]any{"view_range": []any{2.0, 3.0}},
			"       2 | two\n       3 | three\n"},
		{"a range past both ends", map[string]any{"view_range": []any{0.0, 99.0}},
			"       1 | one\n       2 | two\n       3 | three\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.args["command"] = "view"
			tc.args["path"] = writeFile(t, "a.txt", "one\ntwo\nthree")

			got := (&tools.EditTool{}).Execute(tc.args)

			assertResult(t, got, tools.Result{Success: true, Output: tc.want})
		})
	}
}

func TestViewReadsOnlyMaxOutputBytesOfALargerFileAndSaysSo(t *testing.T) {
	// The view of the first 1024 bytes is 1036 bytes, with the line number. So both views are
	// over the budget too, and keep their first 409 and last 615 bytes.
	boundedView := "       1 | " + strings.Repeat("x", 398) +
		"\n... [output truncated: 1036 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
		strings.Repeat("x", 614) + "\n"
	cases := []struct {
		name     string
		fileSize int
		want     string
	}{
		{"a file of exactly the budget", 1024, boundedView},
		{"a file one byte over the budget", 1025, boundedView +
			"\n... [file larger than 1024 bytes: only the first 1024 bytes were read; " +
			"use view_range, or raise max_output_bytes] ...\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, "big.txt", strings.Repeat("x", tc.fileSize))

			got := (&tools.EditTool{}).Execute(map[string]any{"command": "view", "path": path, "max_output_bytes": 1024.0})

			assertResult(t, got, tools.Result{Success: true, Output: tc.want})
		})
	}
}

func TestViewOfAMissingFileIsRefused(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent.txt")

	got := (&tools.EditTool{}).Execute(map[string]any{"command": "view", "path": path})

	assertResult(t, got, tools.Result{Success: false, Error: "open " + path + ": no such file or directory"})
}

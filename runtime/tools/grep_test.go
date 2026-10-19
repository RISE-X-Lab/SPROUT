// Tests of the grep_files tool. A search of a directory reports each match as path:line:text,
// and a search of one file as line:text, as ripgrep does.

package tools_test

import (
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

// directoryWith makes a temporary directory that holds each file of files, by name and content.
func directoryWith(t *testing.T, files map[string]string) string {
	t.Helper()
	dir := t.TempDir()
	for name, content := range files {
		if err := os.WriteFile(filepath.Join(dir, name), []byte(content), 0o644); err != nil {
			t.Fatalf("writing %s: %v", name, err)
		}
	}
	return dir
}

func TestEachMatchIsReportedAsPathLineAndText(t *testing.T) {
	dir := directoryWith(t, map[string]string{"a.txt": "needle 1\nhay\nneedle 2\n"})

	got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "needle", "path": dir})

	assertResult(t, got, tools.Result{Success: true, Output: dir + "/a.txt:1:needle 1\n" + dir + "/a.txt:3:needle 2\n"})
}

func TestIncludeLimitsTheSearchToTheFilesThatMatchItsGlob(t *testing.T) {
	dir := directoryWith(t, map[string]string{"a.py": "needle\n", "b.txt": "needle\n"})

	got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "needle", "path": dir, "include": "*.py"})

	assertResult(t, got, tools.Result{Success: true, Output: dir + "/a.py:1:needle\n"})
}

func TestLimitBoundsTheNumberOfMatchesAndSaysWhenItCutThem(t *testing.T) {
	// A limit of 0 or less means the default, 100. The file has 3 matches. So 0 and 1 are on each
	// side of the default, and 2 and 3 on each side of the number of matches.
	cases := []struct {
		name  string
		limit float64
		want  string
	}{
		{"0", 0, "1:needle 1\n2:needle 2\n3:needle 3\n"},
		{"1", 1, "1:needle 1\n... (truncated at 1 matches)\n"},
		{"2", 2, "1:needle 1\n2:needle 2\n... (truncated at 2 matches)\n"},
		{"3", 3, "1:needle 1\n2:needle 2\n3:needle 3\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			path := writeFile(t, "a.txt", "needle 1\nneedle 2\nneedle 3\n")

			got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "needle", "path": path, "limit": tc.limit})

			assertResult(t, got, tools.Result{Success: true, Output: tc.want})
		})
	}
}

func TestASearchWithoutMatchesSaysSo(t *testing.T) {
	path := writeFile(t, "a.txt", "hay\n")

	got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "needle", "path": path})

	assertResult(t, got, tools.Result{Success: true, Output: "No matches found."})
}

func TestASearchOfAMissingPathFailsWithTheMessageOfRipgrep(t *testing.T) {
	path := filepath.Join(t.TempDir(), "absent")

	got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "needle", "path": path})

	assertResult(t, got, tools.Result{Success: false,
		Error: "rg: " + path + ": IO error for operation on " + path + ": No such file or directory (os error 2)\n"})
}

func TestAPatternThatStartsWithADashIsSearchedForAndNotTakenAsAFlag(t *testing.T) {
	path := writeFile(t, "a.txt", "plain\n--files\n")

	got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "--files", "path": path})

	assertResult(t, got, tools.Result{Success: true, Output: "2:--files\n"})
}

func TestGrepRunsTheRipgrepEmbeddedInTheRuntimeAndNotOneOnPath(t *testing.T) {
	path := writeFile(t, "a.txt", "needle\n")
	t.Setenv("PATH", "")

	got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "needle", "path": path})

	assertResult(t, got, tools.Result{Success: true, Output: "1:needle\n"})
}

func TestAMatchLongerThanMaxOutputBytesIsCutToItsFirst40AndLast60Percent(t *testing.T) {
	// The output is "1:", the line of 2997 bytes and a newline: 3000 bytes.
	line := "needle" + strings.Repeat("h", 401) + strings.Repeat("m", 1976) + strings.Repeat("t", 614)
	path := writeFile(t, "a.txt", line+"\n")

	got := (&tools.GrepTool{}).Execute(map[string]any{"pattern": "needle", "path": path, "max_output_bytes": 1024.0})

	assertResult(t, got, tools.Result{Success: true, Output: "1:needle" + strings.Repeat("h", 401) +
		"\n... [output truncated: 3000 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
		strings.Repeat("t", 614) + "\n"})
}

package tools

import (
	"os"
	"os/exec"
	"path/filepath"
	goruntime "runtime"
	"strings"
	"testing"
)

func TestTheEmbeddedRipgrepIsThePinnedReleaseAndRuns(t *testing.T) {
	if goruntime.GOARCH != "amd64" {
		t.Skip("the embedded ripgrep is x86_64")
	}
	_, name, err := pinnedRipgrepArchive()
	if err != nil {
		t.Fatal(err)
	}
	rg, err := ripgrep()
	if err != nil {
		t.Fatal(err)
	}
	out, err := exec.Command(rg, "--version").Output()
	version := strings.SplitN(strings.TrimPrefix(name, "ripgrep-"), "-", 2)[0]
	if err != nil || !strings.HasPrefix(string(out), "ripgrep "+version+" ") {
		t.Fatalf("%s --version = %q, %v; want ripgrep %s", rg, out, err, version)
	}
	if _, err := ripgrepBinary.Write([]byte("x")); err == nil {
		t.Fatal("the sealed binary accepted a write")
	}
}

func TestGrepFindsMatchesAndHonoursTheLimit(t *testing.T) {
	dir := t.TempDir()
	os.WriteFile(filepath.Join(dir, "a.py"), []byte("needle 1\nhay\nneedle 2\n"), 0644)
	os.WriteFile(filepath.Join(dir, "b.txt"), []byte("needle 3\n"), 0644)
	tool := &GrepTool{}

	r := tool.Execute(map[string]any{"pattern": "needle", "path": dir, "include": "*.py"})
	if !r.Success || strings.Count(r.Output, "needle") != 2 || strings.Contains(r.Output, "b.txt") {
		t.Fatalf("unexpected matches: %#v", r)
	}
	limited := tool.Execute(map[string]any{"pattern": "needle", "path": dir, "limit": float64(1)})
	if !limited.Success || !strings.Contains(limited.Output, "truncated at 1 matches") {
		t.Fatalf("expected the limit to be reported: %#v", limited)
	}
	none := tool.Execute(map[string]any{"pattern": "absent", "path": dir})
	if !none.Success || none.Output != "No matches found." {
		t.Fatalf("unexpected no-match result: %#v", none)
	}
}

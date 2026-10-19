// Tests of the shell tool. The harness parses its Output in two ways: bare stdout for a plain
// success, and the CommandOutcome JSON for any other outcome (src/sprout_sandbox/result.py).

package tools_test

import (
	"encoding/json"
	"errors"
	"fmt"
	"io/fs"
	"os"
	"path/filepath"
	"reflect"
	"strconv"
	"strings"
	"syscall"
	"testing"
	"time"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

// assertOutcome checks that a shell result has the success wantSuccess, no Error, and an Output
// that is the JSON value wantJSON.
func assertOutcome(t *testing.T, got tools.Result, wantSuccess bool, wantJSON string) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal([]byte(got.Output), &gotValue); err != nil {
		t.Fatalf("Output %q is not JSON: %v", got.Output, err)
	}
	if err := json.Unmarshal([]byte(wantJSON), &wantValue); err != nil {
		t.Fatalf("the expected outcome is not JSON: %v", err)
	}
	if got.Success != wantSuccess || got.Error != "" || !reflect.DeepEqual(gotValue, wantValue) {
		t.Errorf("result = Success %v, Error %q, Output\n%s\nwant Success %v, no Error, Output\n%s",
			got.Success, got.Error, got.Output, wantSuccess, wantJSON)
	}
}

// decodeOutcome is the CommandOutcome that a shell result carries in its Output.
func decodeOutcome(t *testing.T, got tools.Result) tools.CommandOutcome {
	t.Helper()
	var outcome tools.CommandOutcome
	if err := json.Unmarshal([]byte(got.Output), &outcome); err != nil {
		t.Fatalf("Output %q is not a CommandOutcome: %v", got.Output, err)
	}
	return outcome
}

// emit is a shell command that writes n bytes of char to stdout.
func emit(char byte, n int) string {
	return fmt.Sprintf("head -c %d /dev/zero | tr '\\0' '%c'", n, char)
}

// processEnds waits up to 5 s for the process pid to end, and reports whether it did. A zombie
// counts as ended, because it runs nothing and only waits for its parent to reap it.
func processEnds(pid int) bool {
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		stat, err := os.ReadFile("/proc/" + strconv.Itoa(pid) + "/stat")
		if errors.Is(err, fs.ErrNotExist) || strings.Contains(string(stat), ") Z ") {
			return true
		}
		time.Sleep(10 * time.Millisecond)
	}
	return false
}

func intPointer(n int) *int { return &n }

func TestAPlainSuccessIsReportedAsBareStdout(t *testing.T) {
	got := (&tools.ShellTool{}).Execute(map[string]any{"command": "echo hello"})

	assertResult(t, got, tools.Result{Success: true, Output: "hello\n"})
}

func TestAnyOtherOutcomeIsReportedAsTheCommandOutcomeJSON(t *testing.T) {
	cases := []struct {
		name        string
		args        map[string]any
		wantSuccess bool
		wantJSON    string
	}{
		{"a non-zero exit", map[string]any{"command": "echo partial; exit 7"}, false,
			`{"stdout": "partial\n", "stderr": "", "exit_code": 7, "running": false, "timed_out": false,
			  "stdout_bytes": 8, "stderr_bytes": 0, "stdout_truncated": false, "stderr_truncated": false,
			  "max_per_stream": 15728640}`},
		{"output on stderr", map[string]any{"command": "echo warn >&2"}, true,
			`{"stdout": "", "stderr": "warn\n", "exit_code": 0, "running": false, "timed_out": false,
			  "stdout_bytes": 0, "stderr_bytes": 5, "stdout_truncated": false, "stderr_truncated": false,
			  "max_per_stream": 15728640}`},
		// SIGKILL ends a timed-out command, and a death by signal N is exit code 128+N.
		{"a timeout", map[string]any{"command": "sleep 30", "timeout": 1.0}, false,
			`{"stdout": "", "stderr": "", "exit_code": 137, "running": false, "timed_out": true,
			  "stdout_bytes": 0, "stderr_bytes": 0, "stdout_truncated": false, "stderr_truncated": false,
			  "max_per_stream": 15728640}`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			got := (&tools.ShellTool{}).Execute(tc.args)

			assertOutcome(t, got, tc.wantSuccess, tc.wantJSON)
		})
	}
}

func TestATimeoutAlsoKillsTheBackgroundProcessesOfTheCommand(t *testing.T) {
	t.Parallel()
	pidFile := filepath.Join(t.TempDir(), "pid")

	(&tools.ShellTool{}).Execute(map[string]any{
		"command": "sleep 300 & echo $! > " + pidFile + "; wait",
		"timeout": 1.0,
	})

	pid, err := strconv.Atoi(strings.TrimSpace(readFile(t, pidFile)))
	if err != nil {
		t.Fatalf("the command wrote no pid: %v", err)
	}
	t.Cleanup(func() { syscall.Kill(pid, syscall.SIGKILL) })
	if !processEnds(pid) {
		t.Errorf("the background sleep (pid %d) still runs after the timeout of its command", pid)
	}
}

func TestACommandRunsInItsWorkingDir(t *testing.T) {
	dir := t.TempDir()

	(&tools.ShellTool{}).Execute(map[string]any{"command": "echo here > marker", "working_dir": dir})

	if got := readFile(t, filepath.Join(dir, "marker")); got != "here\n" {
		t.Errorf("marker in working_dir = %q, want %q", got, "here\n")
	}
}

func TestEachStreamIsBoundedOnItsOwnAndTheOutcomeSaysWhichWasCut(t *testing.T) {
	command := emit('h', 409) + "; " + emit('m', 1976) + "; " + emit('t', 615) + "; " + emit('e', 1000) + " >&2"

	got := (&tools.ShellTool{}).Execute(map[string]any{"command": command, "max_output_bytes": 1024.0})

	want := tools.CommandOutcome{
		Stdout: strings.Repeat("h", 409) +
			"\n... [output truncated: 3000 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
			strings.Repeat("t", 615),
		Stderr:          strings.Repeat("e", 1000),
		ExitCode:        intPointer(0),
		StdoutBytes:     3000,
		StderrBytes:     1000,
		StdoutTruncated: true,
		StderrTruncated: false,
		MaxPerStream:    1024,
	}
	if outcome := decodeOutcome(t, got); !got.Success || !reflect.DeepEqual(outcome, want) {
		t.Errorf("result = Success %v, outcome\n%+v\nwant Success true, outcome\n%+v", got.Success, outcome, want)
	}
}

func TestTheBudgetOfAStreamIsMaxOutputBytesClampedTo1KiBThrough64MiB(t *testing.T) {
	// "exit 1" makes each result an outcome, which reports the budget. The values are those on
	// each side of 0 (below it, the default of 15 MiB applies), 1024 and 64 MiB.
	cases := []struct {
		name string
		args map[string]any
		want int
	}{
		{"absent", map[string]any{"command": "exit 1"}, 15728640},
		{"0", map[string]any{"command": "exit 1", "max_output_bytes": 0.0}, 15728640},
		{"1", map[string]any{"command": "exit 1", "max_output_bytes": 1.0}, 1024},
		{"1023", map[string]any{"command": "exit 1", "max_output_bytes": 1023.0}, 1024},
		{"1024", map[string]any{"command": "exit 1", "max_output_bytes": 1024.0}, 1024},
		{"64 MiB", map[string]any{"command": "exit 1", "max_output_bytes": 67108864.0}, 67108864},
		{"64 MiB + 1", map[string]any{"command": "exit 1", "max_output_bytes": 67108865.0}, 67108864},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := (&tools.ShellTool{}).Execute(tc.args)

			if budget := decodeOutcome(t, got).MaxPerStream; budget != tc.want {
				t.Errorf("max_per_stream = %d, want %d", budget, tc.want)
			}
		})
	}
}

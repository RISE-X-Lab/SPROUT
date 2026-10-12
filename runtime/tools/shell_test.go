package tools

import (
	"encoding/json"
	"fmt"
	"os"
	"strconv"
	"strings"
	"testing"
	"time"
)

// countProcessesMatching counts live processes whose cmdline contains needle,
// skipping the scanning process itself.
func countProcessesMatching(t *testing.T, needle string) int {
	t.Helper()
	entries, err := os.ReadDir("/proc")
	if err != nil {
		t.Skipf("no /proc on this platform: %v", err)
	}
	self := strconv.Itoa(os.Getpid())
	count := 0
	for _, entry := range entries {
		if !entry.IsDir() || entry.Name() == self {
			continue
		}
		if _, err := strconv.Atoi(entry.Name()); err != nil {
			continue
		}
		raw, err := os.ReadFile("/proc/" + entry.Name() + "/cmdline")
		if err != nil {
			continue // exited between listing and reading
		}
		if strings.Contains(strings.ReplaceAll(string(raw), "\x00", " "), needle) {
			count++
		}
	}
	return count
}

func TestForegroundTimeoutKillsDescendantsAndReturns(t *testing.T) {
	// CommandContext's default cancellation kills only the direct `sh`. A
	// backgrounded descendant then keeps stdout/stderr open, so cmd.Run waits
	// forever even though the requested timeout elapsed.
	marker := fmt.Sprintf("SPROUT_TIMEOUT_PROBE_%d_%d", os.Getpid(), time.Now().UnixNano())
	start := time.Now()
	result := (&ShellTool{}).Execute(map[string]any{
		"command": "sh -c 'sleep 7117 # " + marker + "' & wait",
		"timeout": float64(1),
	})
	elapsed := time.Since(start)

	if result.Success {
		t.Fatal("a timed-out command must fail")
	}
	if elapsed > 4*time.Second {
		t.Fatalf("timeout returned after %v, want no more than 4s", elapsed)
	}
	var outcome struct {
		ExitCode *int `json:"exit_code"`
		TimedOut bool `json:"timed_out"`
		Running  bool `json:"running"`
	}
	if err := json.Unmarshal([]byte(result.Output), &outcome); err != nil {
		t.Fatalf("decode timeout outcome %q: %v", result.Output, err)
	}
	if !outcome.TimedOut || outcome.Running {
		t.Fatalf("timeout outcome = %+v, want timed_out and not running", outcome)
	}
	// 128+signal, the convention every shell reports for a signalled death.
	if outcome.ExitCode == nil || *outcome.ExitCode != 128+9 {
		t.Fatalf("exit_code = %v, want %d", outcome.ExitCode, 128+9)
	}
	if n := countProcessesMatching(t, marker); n != 0 {
		t.Fatalf("%d descendant(s) survived the foreground timeout", n)
	}

	// A leaked pipe or process must not poison the runtime after the timeout.
	if followUp := (&ShellTool{}).Execute(map[string]any{
		"command": "printf follow-up-ok",
		"timeout": float64(2),
	}); !followUp.Success || followUp.Output != "follow-up-ok" {
		t.Fatalf("follow-up shell failed after timeout: %#v", followUp)
	}
}

// TestForegroundFailureStillFails: a command that ran and came out non-zero is a
// failed tool call, because for `shell` the command's success IS the tool's
// success -- and the exit code travels in the outcome, not in Error.
func TestForegroundFailureStillFails(t *testing.T) {
	r := (&ShellTool{}).Execute(map[string]any{"command": "exit 7"})
	if r.Success {
		t.Fatal("a command exiting 7 must report failure")
	}
	if r.Error != "" {
		t.Fatalf("a failed command is an observation, not a tool error: %q", r.Error)
	}
	var snap struct {
		ExitCode *int `json:"exit_code"`
	}
	if err := json.Unmarshal([]byte(r.Output), &snap); err != nil {
		t.Fatalf("decode: %v", err)
	}
	if snap.ExitCode == nil || *snap.ExitCode != 7 {
		t.Fatalf("exit_code = %v, want 7", snap.ExitCode)
	}
}

// TestOutputBeyondTheBudgetIsClippedAndSaysSo pins the one output knob the
// harness drives (`max_output_bytes`, from the Harbor exec path) and the flags
// it reads back: a clipped success is not bare stdout, it carries the outcome
// with `stdout_truncated` set, and both ends of the output survive.
func TestOutputBeyondTheBudgetIsClippedAndSaysSo(t *testing.T) {
	r := (&ShellTool{}).Execute(map[string]any{
		"command":          "seq 1 200000",
		"max_output_bytes": float64(4096),
	})
	if !r.Success {
		t.Fatalf("command failed: %#v", r)
	}
	var outcome struct {
		Stdout          string `json:"stdout"`
		StdoutBytes     int64  `json:"stdout_bytes"`
		StdoutTruncated bool   `json:"stdout_truncated"`
		StderrTruncated bool   `json:"stderr_truncated"`
		MaxPerStream    int    `json:"max_per_stream"`
	}
	if err := json.Unmarshal([]byte(r.Output), &outcome); err != nil {
		t.Fatalf("a clipped success must carry the outcome schema, got %.80q: %v", r.Output, err)
	}
	if !outcome.StdoutTruncated || outcome.StderrTruncated {
		t.Fatalf("outcome = %+v, want stdout clipped and stderr not", outcome)
	}
	if outcome.MaxPerStream != 4096 || outcome.StdoutBytes <= 4096 {
		t.Fatalf("outcome = %+v, want a 4096-byte budget the output exceeded", outcome)
	}
	if !strings.HasPrefix(outcome.Stdout, "1\n") || !strings.HasSuffix(strings.TrimRight(outcome.Stdout, "\n"), "200000") {
		t.Fatalf("clipped output should keep the first line and the last, got %.60q ... %.60q",
			outcome.Stdout, outcome.Stdout[len(outcome.Stdout)-60:])
	}
	if !strings.Contains(outcome.Stdout, "truncated") {
		t.Fatalf("clipped output must say so in the text too: %.200q", outcome.Stdout)
	}
}

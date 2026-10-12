package tools

import (
	"encoding/json"
	"fmt"
	"sync"
)

const (
	// defaultMaxOutputBytes is the budget for each output stream when a call
	// names none: generous, because deciding how much belongs in an LLM
	// context is a consumer-side concern. The runtime's job is only to stay
	// alive.
	defaultMaxOutputBytes = 15 << 20 // 15 MiB
	minMaxOutputBytes     = 1024
	// maxMaxOutputBytes caps what a caller may ask for, so a single call
	// cannot OOM the runtime (and with it the sandbox).
	maxMaxOutputBytes = 64 << 20 // 64 MiB
	// Over-long output keeps its head and its tail in this proportion (40/60):
	// a test run's verdict is at the end, the command that produced it at the
	// start, and the middle is what a reader scrolls past.
	headWeight = 2
	tailWeight = 3
)

// BoundedLog stores command output with a fixed memory ceiling. Once the
// output exceeds the limit, Render keeps the head and tail sections in the
// headWeight:tailWeight proportion and says so.
type BoundedLog struct {
	mu    sync.Mutex
	max   int
	head  []byte
	tail  []byte
	total int64
}

// NewBoundedLog creates a log bounded to max bytes, clamped to the runtime's
// floor and ceiling.
func NewBoundedLog(max int) *BoundedLog {
	return &BoundedLog{max: normalizeMaxOutputBytes(max)}
}

func normalizeMaxOutputBytes(max int) int {
	if max <= 0 {
		return defaultMaxOutputBytes
	}
	if max < minMaxOutputBytes {
		return minMaxOutputBytes
	}
	if max > maxMaxOutputBytes {
		return maxMaxOutputBytes
	}
	return max
}

func outputBytesArg(args map[string]any) int {
	max := defaultMaxOutputBytes
	if v, ok := args["max_output_bytes"].(float64); ok {
		max = int(v)
	}
	return normalizeMaxOutputBytes(max)
}

func (b *BoundedLog) Write(p []byte) (int, error) {
	b.mu.Lock()
	defer b.mu.Unlock()

	n := len(p)
	b.total += int64(n)

	headLimit := b.headLimit()
	if len(b.head) < headLimit {
		remaining := headLimit - len(b.head)
		if remaining > n {
			remaining = n
		}
		b.head = append(b.head, p[:remaining]...)
	}

	tailLimit := b.tailLimit()
	b.tail = append(b.tail, p...)
	if len(b.tail) > tailLimit {
		copy(b.tail, b.tail[len(b.tail)-tailLimit:])
		b.tail = b.tail[:tailLimit]
	}

	return n, nil
}

func (b *BoundedLog) Render() (string, bool) {
	b.mu.Lock()
	defer b.mu.Unlock()

	if b.total <= int64(b.max) {
		total := int(b.total)
		headLimit := b.headLimit()
		tailLimit := b.tailLimit()
		if total <= headLimit {
			return string(b.head), false
		}
		if total <= tailLimit {
			return string(b.tail), false
		}
		overlap := b.max - total
		if overlap < 0 {
			overlap = 0
		}
		if overlap > len(b.tail) {
			overlap = len(b.tail)
		}
		return string(b.head) + string(b.tail[overlap:]), false
	}

	headLimit := b.headLimit()
	tailLimit := b.tailLimit()
	head := b.head
	if len(head) > headLimit {
		head = head[:headLimit]
	}
	tail := b.tail
	if len(tail) > tailLimit {
		tail = tail[len(tail)-tailLimit:]
	}

	marker := fmt.Sprintf(
		"\n... [output truncated: %d bytes total, showing first %d bytes and last %d bytes] ...\n",
		b.total, len(head), len(tail),
	)
	return string(head) + marker + string(tail), true
}

func (b *BoundedLog) TotalBytes() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.total
}

// headLimit is the byte budget for the head section; tailLimit is whatever the
// head does not claim.
func (b *BoundedLog) headLimit() int {
	return b.max * headWeight / (headWeight + tailWeight)
}

func (b *BoundedLog) tailLimit() int {
	return b.max - b.headLimit()
}

// CommandOutcome is what running a command produced.
//
// Streams stay separate. Merging them into one blob was lossy in three ways: the
// "[stderr]" divider it inserted is forgeable by any command that prints one,
// the exit code had nowhere to go (`exit 5` prints nothing and arrived as an
// empty string), and truncation was described in prose a consumer had to grep.
//
// Every field is always present. `running: false` and `timed_out: false` are
// answers, not absences, and omitempty would make a consumer unable to tell "it
// finished" from "this runtime never said". A foreground command has always
// finished by the time it is reported, so `running` is false and `exit_code`
// is set; both stay in the schema because the harness parses an outcome by its
// field set.
type CommandOutcome struct {
	Stdout          string `json:"stdout"`
	Stderr          string `json:"stderr"`
	ExitCode        *int   `json:"exit_code"`
	Running         bool   `json:"running"`
	TimedOut        bool   `json:"timed_out"`
	StdoutBytes     int64  `json:"stdout_bytes"`
	StderrBytes     int64  `json:"stderr_bytes"`
	StdoutTruncated bool   `json:"stdout_truncated"`
	StderrTruncated bool   `json:"stderr_truncated"`
	MaxPerStream    int    `json:"max_per_stream"`
}

// commandOutcome reads both logs into the shared schema.
func commandOutcome(stdout, stderr *BoundedLog) CommandOutcome {
	stdoutText, stdoutTruncated := stdout.Render()
	stderrText, stderrTruncated := stderr.Render()
	return CommandOutcome{
		Stdout:          stdoutText,
		Stderr:          stderrText,
		StdoutBytes:     stdout.TotalBytes(),
		StderrBytes:     stderr.TotalBytes(),
		StdoutTruncated: stdoutTruncated,
		StderrTruncated: stderrTruncated,
		MaxPerStream:    stdout.max,
	}
}

// Result renders an outcome for the wire: `shell`'s own success is the
// command's success.
//
// A plain success -- exit 0, nothing on stderr, nothing clipped -- is returned
// as bare stdout: there is nothing to say beyond it, and wrapping `echo hello`
// in JSON would make every consumer parse a payload to find one word. Anything
// else carries the schema, because something about the run cannot be read off
// the text.
func (o CommandOutcome) Result() Result {
	success := o.commandSucceeded()
	if success && o.plain() {
		return Ok(o.Stdout)
	}
	return Result{Success: success, Output: o.encode()}
}

func (o CommandOutcome) encode() string {
	payload, err := json.Marshal(o)
	if err != nil {
		// Unreachable for this struct; degrade to the merged text rather than
		// losing the output entirely.
		return o.Stdout + o.Stderr
	}
	return string(payload)
}

// commandSucceeded reports whether the command itself came out clean.
func (o CommandOutcome) commandSucceeded() bool {
	return !o.TimedOut && (o.ExitCode == nil || *o.ExitCode == 0)
}

// plain reports whether the outcome holds nothing a bare stdout would omit.
func (o CommandOutcome) plain() bool {
	return !o.Running && o.Stderr == "" &&
		!o.StdoutTruncated && !o.StderrTruncated
}

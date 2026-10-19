package tools

import (
	"encoding/json"
	"fmt"
	"sync"
)

const (
	// The budget of a call that names none. It is large because the consumer
	// decides what fits a model's context. The runtime only bounds its memory.
	defaultMaxOutputBytes = 15 << 20 // 15 MiB
	minMaxOutputBytes     = 1024
	// maxMaxOutputBytes caps what a caller may ask for, so a single call
	// cannot OOM the runtime (and with it the sandbox).
	maxMaxOutputBytes = 64 << 20 // 64 MiB
	// Over-long output keeps its head and its tail at 40/60. A test run prints
	// its command first and its verdict last, and a reader skips the middle.
	headWeight = 2
	tailWeight = 3
)

// BoundedLog is an io.Writer that keeps command output within a fixed memory
// ceiling. When the output exceeds the ceiling, Render keeps the head and the
// tail in the headWeight:tailWeight proportion and says so. Write never fails,
// and all methods are safe for concurrent use.
type BoundedLog struct {
	mu    sync.Mutex
	max   int
	head  []byte
	tail  []byte
	total int64
}

// NewBoundedLog creates a log bounded to max bytes, clamped to the runtime's
// floor and ceiling. A max of zero or less means defaultMaxOutputBytes.
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

// outputBytesSchema is the max_output_bytes schema of a tool that returns one
// block of text. Each such tool shares it, so the bound reads alike on each.
func outputBytesSchema() map[string]any {
	return map[string]any{
		"type":        "integer",
		"default":     defaultMaxOutputBytes,
		"description": "Total bytes of output to return. Larger output is truncated to the first 40% and last 60% of the budget, and the output says so.",
	}
}

// boundText bounds text that a tool already holds in memory to max bytes, with
// the same head and tail rule as command output. So every tool truncates alike.
func boundText(text string, max int) string {
	log := NewBoundedLog(max)
	log.Write([]byte(text))
	out, _ := log.Render()
	return out
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

// Render returns the output and whether it was truncated. Truncated output is
// its head and its tail around a marker line that gives the byte counts.
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

// TotalBytes is the number of bytes written, the dropped ones included.
func (b *BoundedLog) TotalBytes() int64 {
	b.mu.Lock()
	defer b.mu.Unlock()
	return b.total
}

// headLimit is the byte budget of the head section. tailLimit, below, gets the
// rest of max.
func (b *BoundedLog) headLimit() int {
	return b.max * headWeight / (headWeight + tailWeight)
}

func (b *BoundedLog) tailLimit() int {
	return b.max - b.headLimit()
}

// CommandOutcome is what running a command produced, as the JSON that the
// harness parses. Every field is always encoded: a false `running` or
// `timed_out` is an answer, not an absence. A foreground command has finished
// when it is reported, so `running` is false and `exit_code` is set.
type CommandOutcome struct {
	// A new field also goes into _OUTCOME_FIELDS (src/sprout_sandbox/result.py).
	// The harness reads JSON with any other field as plain output, not as an outcome.
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

// commandOutcome reads both logs into an outcome that is not running. The
// caller sets ExitCode and TimedOut.
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

// Result renders an outcome for the wire. Its Success is the command's success.
// A plain success (exit 0, empty stderr, nothing truncated, not running) is bare
// stdout, so `echo hello` reaches the consumer as one word. Any other outcome is
// its JSON encoding, because the text alone cannot say what happened.
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
		// json.Marshal cannot fail on this struct. If it ever does, return the
		// merged text rather than nothing.
		return o.Stdout + o.Stderr
	}
	return string(payload)
}

// commandSucceeded reports whether the command did not time out and exited 0.
// An outcome without an exit code counts as a success.
func (o CommandOutcome) commandSucceeded() bool {
	return !o.TimedOut && (o.ExitCode == nil || *o.ExitCode == 0)
}

// plain reports whether the outcome holds nothing a bare stdout would omit.
func (o CommandOutcome) plain() bool {
	return !o.Running && o.Stderr == "" &&
		!o.StdoutTruncated && !o.StderrTruncated
}

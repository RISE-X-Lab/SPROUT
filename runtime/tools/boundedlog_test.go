// Tests of BoundedLog, the memory bound on what a tool returns. A budget of 1024 bytes, the
// floor, has a head of 409 bytes and a tail of 615. Each case writes its text in one piece and
// in 100-byte pieces, as a pipe delivers output, because the result must not depend on that.

package tools_test

import (
	"strings"
	"testing"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

// renderBounded writes text to a log with a budget of max bytes, in pieces of chunk bytes, and returns
// what the log renders.
func renderBounded(max int, text string, chunk int) (string, bool) {
	log := tools.NewBoundedLog(max)
	for start := 0; start < len(text); start += chunk {
		log.Write([]byte(text[start:min(start+chunk, len(text))]))
	}
	return log.Render()
}

// sections is head bytes of "h", then middle bytes of "m", then tail bytes of "t".
func sections(head, middle, tail int) string {
	return strings.Repeat("h", head) + strings.Repeat("m", middle) + strings.Repeat("t", tail)
}

func TestOutputWithinTheBudgetIsReturnedWhole(t *testing.T) {
	// The sizes are the values on each side of the head limit (409) and the tail limit (615),
	// and the budget itself.
	cases := []struct {
		name  string
		text  string
		chunk int
	}{
		{"409 bytes in one write", sections(409, 0, 0), 409},
		{"409 bytes in 100-byte writes", sections(409, 0, 0), 100},
		{"410 bytes in one write", sections(409, 0, 1), 410},
		{"410 bytes in 100-byte writes", sections(409, 0, 1), 100},
		{"615 bytes in one write", sections(409, 0, 206), 615},
		{"615 bytes in 100-byte writes", sections(409, 0, 206), 100},
		{"616 bytes in one write", sections(409, 0, 207), 616},
		{"616 bytes in 100-byte writes", sections(409, 0, 207), 100},
		{"1024 bytes in one write", sections(409, 0, 615), 1024},
		{"1024 bytes in 100-byte writes", sections(409, 0, 615), 100},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, truncated := renderBounded(1024, tc.text, tc.chunk)

			if out != tc.text || truncated {
				t.Errorf("render = %q, truncated %v; want the input whole, not truncated", out, truncated)
			}
		})
	}
}

func TestOutputOverTheBudgetKeepsItsFirst40AndLast60PercentAroundAMarker(t *testing.T) {
	cases := []struct {
		name  string
		text  string
		chunk int
		want  string
	}{
		{"1025 bytes in one write", sections(409, 1, 615), 1025, sections(409, 0, 0) +
			"\n... [output truncated: 1025 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
			sections(0, 0, 615)},
		{"1025 bytes in 100-byte writes", sections(409, 1, 615), 100, sections(409, 0, 0) +
			"\n... [output truncated: 1025 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
			sections(0, 0, 615)},
		{"3000 bytes in one write", sections(409, 1976, 615), 3000, sections(409, 0, 0) +
			"\n... [output truncated: 3000 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
			sections(0, 0, 615)},
		{"3000 bytes in 100-byte writes", sections(409, 1976, 615), 100, sections(409, 0, 0) +
			"\n... [output truncated: 3000 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
			sections(0, 0, 615)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			out, truncated := renderBounded(1024, tc.text, tc.chunk)

			if out != tc.want || !truncated {
				t.Errorf("render = %q, truncated %v;\nwant %q, truncated", out, truncated, tc.want)
			}
		})
	}
}

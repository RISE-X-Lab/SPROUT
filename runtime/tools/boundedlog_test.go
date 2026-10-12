package tools

// Tests for the output bounding in boundedlog.go (run by `go test ./tools`;
// shell_test.go covers command execution). The budget is per stream and the
// split is fixed: over-long output keeps its first 40% and last 60%.

import (
	"strings"
	"testing"
)

// renderWith writes n bytes through a BoundedLog bounded to max and returns the
// rendered output plus the truncated flag. The first half is "A", the second
// "Z", so a test can tell which side survived.
func renderWith(max, n int) (string, bool) {
	log := NewBoundedLog(max)
	body := strings.Repeat("A", n/2) + strings.Repeat("Z", n-n/2)
	log.Write([]byte(body))
	return log.Render()
}

func TestTruncationKeepsBothEnds(t *testing.T) {
	out, truncated := renderWith(1000, 4000)
	if !truncated {
		t.Fatal("expected truncation")
	}
	if !strings.Contains(out, "A") || !strings.Contains(out, "Z") {
		t.Errorf("both the head and the tail should survive, got %d bytes", len(out))
	}
	if !strings.HasPrefix(out, "A") || !strings.HasSuffix(out, "Z") {
		t.Errorf("the head comes first and the tail last, got %.20q ... %.20q", out, out[len(out)-20:])
	}
}

func TestTheSplitIsFortySixty(t *testing.T) {
	// 4000 is above minMaxOutputBytes, so the budget is used verbatim.
	log := NewBoundedLog(4000)
	if got := log.headLimit(); got != 1600 {
		t.Errorf("head limit = %d, want 1600", got)
	}
	if got := log.tailLimit(); got != 2400 {
		t.Errorf("tail limit = %d, want 2400", got)
	}
}

func TestTruncationRespectsTheTotalBudget(t *testing.T) {
	// Rendered output = kept bytes + a truncation marker line.
	out, _ := renderWith(2000, 10000)
	if len(out) > 2000+300 {
		t.Errorf("exceeded the total budget: %d bytes", len(out))
	}
}

func TestTruncationIsSelfDescribing(t *testing.T) {
	out, truncated := renderWith(1000, 5000)
	if !truncated {
		t.Fatal("expected truncation")
	}
	if !strings.Contains(out, "truncated") {
		t.Errorf("truncated output must say so; got: %.120s", out)
	}
}

func TestShortOutputIsNotTruncated(t *testing.T) {
	out, truncated := renderWith(1000, 100)
	if truncated {
		t.Error("output within budget should not be marked truncated")
	}
	if strings.Contains(out, "truncated") {
		t.Error("output within budget should carry no truncation marker")
	}
	if len(out) != 100 {
		t.Errorf("output within budget must be returned whole, got %d bytes", len(out))
	}
}

func TestTheBudgetIsClampedToTheFloorAndCeiling(t *testing.T) {
	// The constants replaced per-deployment environment knobs, so what the
	// schema advertises as the default is what a call without the argument
	// gets -- nothing can change it underneath the declaration.
	cases := []struct{ in, want int }{
		{0, defaultMaxOutputBytes},
		{-1, defaultMaxOutputBytes},
		{10, minMaxOutputBytes},
		{4096, 4096},
		{1 << 30, maxMaxOutputBytes},
	}
	for _, c := range cases {
		if got := normalizeMaxOutputBytes(c.in); got != c.want {
			t.Errorf("normalizeMaxOutputBytes(%d) = %d, want %d", c.in, got, c.want)
		}
	}
	if got := outputBytesArg(map[string]any{}); got != defaultMaxOutputBytes {
		t.Errorf("absent argument = %d, want the default %d", got, defaultMaxOutputBytes)
	}
	if got := outputBytesArg(map[string]any{"max_output_bytes": float64(8192)}); got != 8192 {
		t.Errorf("explicit argument = %d, want 8192", got)
	}
}

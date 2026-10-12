package main

import (
	"bufio"
	"encoding/json"
	"fmt"
	"net"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// A tool call answers only when its command finishes, so nothing reaches the
// client until then and a proxy in between reads the silence as a dead upstream
// (AgentENV cuts at 30 s). These tests pin the two halves of the fix: headers go
// out before the tool runs, and the heartbeat that follows stays invisible to an
// ordinary client.
//
// Asserted over a real connection rather than an httptest.ResponseRecorder: a
// recorder reports Code 200 and a populated Header by default, so it cannot tell
// "headers were sent" from "nothing happened at all" -- which is exactly the
// distinction being made here.

// readUntilHeadersEnd returns how long the server took to commit its headers.
func headerLatency(t *testing.T, handler http.HandlerFunc) (time.Duration, string) {
	t.Helper()
	srv := httptest.NewServer(handler)
	defer srv.Close()

	addr := strings.TrimPrefix(srv.URL, "http://")
	conn, err := net.Dial("tcp", addr)
	if err != nil {
		t.Fatalf("dial: %v", err)
	}
	defer conn.Close()

	start := time.Now()
	fmt.Fprintf(conn, "GET / HTTP/1.1\r\nHost: x\r\nConnection: close\r\n\r\n")

	reader := bufio.NewReader(conn)
	var headers strings.Builder
	for {
		line, err := reader.ReadString('\n')
		if err != nil {
			t.Fatalf("reading headers: %v", err)
		}
		headers.WriteString(line)
		if line == "\r\n" {
			break // end of headers
		}
	}
	elapsed := time.Since(start)

	rest, _ := reader.ReadString(0) // read the body until EOF
	return elapsed, headers.String() + rest
}

func TestHeadersArriveWhileTheToolIsStillRunning(t *testing.T) {
	const work = 600 * time.Millisecond

	slow := func(w http.ResponseWriter, r *http.Request) {
		stop, started := beginResponse(w)
		if !started {
			t.Error("a real connection should be flushable")
		}
		time.Sleep(work) // stands in for a command
		stop()
		writeRPC(w, json.RawMessage("1"),
			map[string]any{"content": []map[string]any{{"type": "text", "text": "done"}}}, nil)
	}

	latency, wire := headerLatency(t, slow)
	if latency >= work {
		t.Fatalf("headers took %v, i.e. they waited for the tool (%v) -- a proxy "+
			"would have given up by now", latency, work)
	}
	if !strings.Contains(wire, "200 OK") || !strings.Contains(wire, `"done"`) {
		t.Fatalf("response incomplete: %q", wire)
	}
}

func TestWithoutBeginResponseHeadersWaitForTheTool(t *testing.T) {
	// The control: this is what every handler did before, and why the proxy cut
	// the connection. Keeps the test above honest about what it is measuring.
	const work = 600 * time.Millisecond

	old := func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(work)
		writeRPC(w, json.RawMessage("1"),
			map[string]any{"content": []map[string]any{{"type": "text", "text": "done"}}}, nil)
	}

	latency, _ := headerLatency(t, old)
	if latency < work {
		t.Fatalf("headers took %v, expected to wait for the tool (%v); if this "+
			"fails, the premise of beginResponse no longer holds", latency, work)
	}
}

func TestTheHeartbeatIsInvisibleToAJSONClient(t *testing.T) {
	// JSON ignores leading whitespace, which is what lets the heartbeat exist
	// without a protocol change, a new content type, or version negotiation.
	handler := func(w http.ResponseWriter, r *http.Request) {
		stop, _ := beginResponse(w)
		// Write heartbeats directly rather than waiting out the real interval.
		flusher := w.(http.Flusher)
		for i := 0; i < 5; i++ {
			w.Write([]byte("\n"))
			flusher.Flush()
		}
		stop()
		writeRPC(w, json.RawMessage("1"), map[string]any{
			"content": []map[string]any{{"type": "text", "text": "ok"}},
			"isError": false,
		}, nil)
	}

	srv := httptest.NewServer(http.HandlerFunc(handler))
	defer srv.Close()

	resp, err := http.Get(srv.URL)
	if err != nil {
		t.Fatalf("get: %v", err)
	}
	defer resp.Body.Close()

	// An ordinary client decodes it with no knowledge of the heartbeat.
	var parsed struct {
		Result struct {
			Content []struct{ Text string } `json:"content"`
		} `json:"result"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		t.Fatalf("a plain JSON client could not parse the response: %v", err)
	}
	if got := parsed.Result.Content[0].Text; got != "ok" {
		t.Fatalf("text = %q, want ok", got)
	}
}

func TestStopIsIdempotentAndSilencesTheHeartbeat(t *testing.T) {
	// A ResponseWriter is not safe for concurrent use and the caller writes the
	// body right after stop(), so stop() waits for the heartbeat to exit rather
	// than merely signalling it. Calling it twice must also not panic on a
	// closed channel.
	//
	// Run with -race to catch the interleaving this guards against: the interval
	// is shortened so the heartbeat is mid-flight when stop() is called.
	restore := keepaliveInterval
	keepaliveInterval = time.Millisecond
	defer func() { keepaliveInterval = restore }()

	for i := 0; i < 50; i++ {
		rec := httptest.NewRecorder()
		stop, _ := beginResponse(rec)
		time.Sleep(3 * time.Millisecond) // let some heartbeats land
		stop()
		stop() // idempotent

		// After stop() returns, writing is exclusively the caller's.
		before := rec.Body.Len()
		writeRPC(rec, json.RawMessage("1"),
			map[string]any{"content": []map[string]any{{"type": "text", "text": "ok"}}}, nil)
		body := rec.Body.String()
		if len(body) <= before {
			t.Fatal("the body was not written after stop()")
		}
		// The trailing JSON must be intact -- a heartbeat landing mid-write
		// would have split it.
		if !strings.HasSuffix(strings.TrimRight(body, "\n"), "}") {
			t.Fatalf("response tail looks interleaved: %q", body[max(0, len(body)-60):])
		}
	}
}

func TestNonFlushableWriterIsLeftUntouched(t *testing.T) {
	// Not every ResponseWriter can flush (a wrapped writer, some middleware).
	// Such a writer must be left completely alone, so the response is still
	// written in one piece at the end.
	nf := &nonFlushingWriter{header: http.Header{}}
	stop, started := beginResponse(nf)
	stop()

	if started {
		t.Fatal("started should be false for a writer that cannot flush")
	}
	if nf.wroteHeader {
		t.Fatal("headers must not be committed when they cannot be flushed")
	}
	if nf.written.Len() != 0 {
		t.Fatalf("nothing should have been written, got %q", nf.written.String())
	}
}

func TestKeepaliveIntervalLeavesRoomForALostTick(t *testing.T) {
	// The interval has to sit well under the window a proxy allows for response
	// headers -- 30 s for AgentENV -- with room for one tick to be missed.
	const proxyWindow = 30 * time.Second
	if keepaliveInterval*2 >= proxyWindow {
		t.Fatalf("keepaliveInterval %v leaves no margin for a lost tick within "+
			"a %v proxy window", keepaliveInterval, proxyWindow)
	}
}

type nonFlushingWriter struct {
	header      http.Header
	written     strings.Builder
	wroteHeader bool
}

func (w *nonFlushingWriter) Header() http.Header { return w.header }
func (w *nonFlushingWriter) Write(p []byte) (int, error) {
	return w.written.Write(p)
}
func (w *nonFlushingWriter) WriteHeader(int) { w.wroteHeader = true }

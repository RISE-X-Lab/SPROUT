// Tests of a tools/call whose command runs for a while. A proxy in front of the runtime, such as
// the one of AgentENV, drops a call that sends no response headers for 30 s.

package main

import (
	"context"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

// blockedCommand is a shell command that runs until release is called, and then prints
// "released". A cleanup releases it at the end of the test.
func blockedCommand(t *testing.T) (command string, release func()) {
	t.Helper()
	gate := filepath.Join(t.TempDir(), "release")
	release = func() { os.WriteFile(gate, nil, 0o600) }
	t.Cleanup(release)
	return "while [ ! -e '" + gate + "' ]; do sleep 0.01; done; printf released", release
}

// callShell sends a tools/call of shell with command, and returns once the response headers
// arrive. It fails the test if they do not arrive within 10 s.
func callShell(t *testing.T, runtime *runtimeProcess, command string) *http.Response {
	t.Helper()
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	t.Cleanup(cancel)
	request, err := http.NewRequestWithContext(ctx, http.MethodPost, "http://sprout-runtime/", strings.NewReader(
		`{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "shell", "arguments": {"command": "`+command+`"}}}`))
	if err != nil {
		t.Fatalf("making the request: %v", err)
	}
	response, err := runtime.client.Do(request)
	if err != nil {
		t.Fatalf("no response headers within 10 s while the command ran: %v", err)
	}
	t.Cleanup(func() { response.Body.Close() })
	return response
}

const releasedReply = `{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "released"}], "isError": false}}`

func TestTheHeadersArriveWhileTheCommandRunsAndTheResultAfterIt(t *testing.T) {
	runtime := startRuntime(t, nil, "--listen", "unix:"+socketPath(t))
	command, release := blockedCommand(t)

	response := callShell(t, runtime, command)
	release()
	body, err := io.ReadAll(response.Body)

	if err != nil {
		t.Fatalf("reading the body: %v", err)
	}
	assertReply(t, reply{status: response.StatusCode, body: string(body)}, releasedReply)
}

func TestTheHeartbeatsWhileTheCommandRunsAreWhitespaceThatAJSONClientSkips(t *testing.T) {
	runtime := startRuntime(t, []string{keepaliveOverride + "=1ms"}, "--listen", "unix:"+socketPath(t))
	command, release := blockedCommand(t)
	response := callShell(t, runtime, command)

	heartbeat := make([]byte, 1)
	_, heartbeatErr := io.ReadFull(response.Body, heartbeat)
	release()
	rest, err := io.ReadAll(response.Body)

	if heartbeatErr != nil || string(heartbeat) != "\n" {
		t.Errorf("the body began with %q (%v) while the command ran, want a newline", heartbeat, heartbeatErr)
	}
	if err != nil {
		t.Fatalf("reading the body: %v", err)
	}
	assertJSONEqual(t, "the body", string(heartbeat)+string(rest), releasedReply)
}

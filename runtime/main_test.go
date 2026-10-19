// The callers of the runtime know it as a process: they start it with flags, send JSON-RPC to
// the address that it listens on, and stop it with SIGTERM. The tests do the same. TestMain makes
// the test binary run main() when asRuntime is set, and the helpers here start that process.

package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/json"
	"io"
	"net"
	"net/http"
	"os"
	"os/exec"
	"path/filepath"
	"reflect"
	"strings"
	"syscall"
	"testing"
	"time"
)

const (
	// asRuntime makes the test binary run main() with its arguments, instead of the tests.
	asRuntime = "SPROUT_RUNTIME_TEST_AS_MAIN"
	// keepaliveOverride sets keepaliveInterval in that process, so that a test of the heartbeat
	// does not wait the real 10 s.
	keepaliveOverride = "SPROUT_RUNTIME_TEST_KEEPALIVE"
)

func TestMain(m *testing.M) {
	if os.Getenv(asRuntime) == "" {
		os.Exit(m.Run())
	}
	if interval, err := time.ParseDuration(os.Getenv(keepaliveOverride)); err == nil {
		keepaliveInterval = interval
	}
	main()
	os.Exit(0)
}

// runtimeCommand is the command that runs the runtime with args, and env added to the environment.
func runtimeCommand(t *testing.T, env []string, args ...string) *exec.Cmd {
	t.Helper()
	self, err := os.Executable()
	if err != nil {
		t.Fatalf("finding the test binary: %v", err)
	}
	cmd := exec.Command(self, args...)
	cmd.Env = append(append(os.Environ(), asRuntime+"=1"), env...)
	return cmd
}

// exited is how a runtime process ended, and what it printed.
type exited struct {
	code   int
	stdout string
	stderr string
}

// runRuntime runs the runtime with args until it exits, for at most 10 s.
func runRuntime(t *testing.T, args ...string) exited {
	t.Helper()
	cmd := runtimeCommand(t, nil, args...)
	var stdout, stderr bytes.Buffer
	cmd.Stdout, cmd.Stderr = &stdout, &stderr
	if err := cmd.Start(); err != nil {
		t.Fatalf("starting the runtime: %v", err)
	}
	timer := time.AfterFunc(10*time.Second, func() { cmd.Process.Kill() })
	defer timer.Stop()
	cmd.Wait()
	return exited{code: cmd.ProcessState.ExitCode(), stdout: stdout.String(), stderr: stderr.String()}
}

// runtimeProcess is a runtime that startRuntime started, and a client that reaches it.
type runtimeProcess struct {
	// address is where the runtime listens, in the --listen grammar, as its banner says.
	address string
	client  *http.Client
	cmd     *exec.Cmd
	stderr  bytes.Buffer
	done    chan struct{}
}

// startRuntime starts the runtime with args, and env added to its environment. It returns when
// the runtime says where it listens. A cleanup stops the runtime at the end of the test.
func startRuntime(t *testing.T, env []string, args ...string) *runtimeProcess {
	t.Helper()
	p := &runtimeProcess{cmd: runtimeCommand(t, env, args...), done: make(chan struct{})}
	p.cmd.Stderr = &p.stderr
	stdout, err := p.cmd.StdoutPipe()
	if err != nil {
		t.Fatalf("piping the stdout of the runtime: %v", err)
	}
	if err := p.cmd.Start(); err != nil {
		t.Fatalf("starting the runtime: %v", err)
	}
	t.Cleanup(func() { p.terminate(t) })
	listening := make(chan string, 1)
	go p.readBanner(stdout, listening)
	select {
	case p.address = <-listening:
	case <-p.done:
		t.Fatalf("the runtime exited before it listened, with code %d:\n%s", p.cmd.ProcessState.ExitCode(), p.stderr.String())
	case <-time.After(10 * time.Second):
		t.Fatalf("the runtime did not say where it listens within 10 s")
	}
	network, address, _ := strings.Cut(p.address, ":")
	p.client = &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, address)
		},
	}}
	return p
}

// readBanner sends the address of the "listening on" line of stdout to listening. At the end of
// stdout, it waits for the process and closes p.done.
func (p *runtimeProcess) readBanner(stdout io.Reader, listening chan<- string) {
	lines := bufio.NewScanner(stdout)
	for lines.Scan() {
		if address, found := strings.CutPrefix(lines.Text(), "  listening on "); found {
			listening <- address
		}
	}
	p.cmd.Wait()
	close(p.done)
}

// terminate sends SIGTERM to the runtime, and returns its exit code. After 10 s it kills it.
func (p *runtimeProcess) terminate(t *testing.T) int {
	t.Helper()
	p.cmd.Process.Signal(syscall.SIGTERM)
	select {
	case <-p.done:
	case <-time.After(10 * time.Second):
		t.Errorf("the runtime did not exit within 10 s of SIGTERM")
		p.cmd.Process.Kill()
		<-p.done
	}
	return p.cmd.ProcessState.ExitCode()
}

// kill ends the runtime with SIGKILL, as the kernel ends a runtime that runs out of memory.
func (p *runtimeProcess) kill() {
	p.cmd.Process.Kill()
	<-p.done
}

// reply is the HTTP status and the body of a response.
type reply struct {
	status int
	body   string
}

func (p *runtimeProcess) post(t *testing.T, body string) reply {
	t.Helper()
	return p.send(t, http.MethodPost, body)
}

func (p *runtimeProcess) get(t *testing.T) reply {
	t.Helper()
	return p.send(t, http.MethodGet, "")
}

// send makes one request to the runtime. The host in the URL does not matter: the client dials
// the address of the runtime.
func (p *runtimeProcess) send(t *testing.T, method, body string) reply {
	t.Helper()
	request, err := http.NewRequest(method, "http://sprout-runtime/", strings.NewReader(body))
	if err != nil {
		t.Fatalf("making the request: %v", err)
	}
	response, err := p.client.Do(request)
	if err != nil {
		t.Fatalf("%s to the runtime: %v", method, err)
	}
	defer response.Body.Close()
	data, err := io.ReadAll(response.Body)
	if err != nil {
		t.Fatalf("reading the response: %v", err)
	}
	return reply{status: response.StatusCode, body: string(data)}
}

// assertReply checks that a reply has status 200 and a body that is the JSON value wantJSON.
func assertReply(t *testing.T, got reply, wantJSON string) {
	t.Helper()
	if got.status != http.StatusOK {
		t.Errorf("status = %d, want 200", got.status)
	}
	assertJSONEqual(t, "the body", got.body, wantJSON)
}

// assertJSONEqual checks that got and want are the same JSON value, whatever their layout.
func assertJSONEqual(t *testing.T, what, got, want string) {
	t.Helper()
	var gotValue, wantValue any
	if err := json.Unmarshal([]byte(got), &gotValue); err != nil {
		t.Fatalf("%s is not JSON: %v\n%s", what, err, got)
	}
	if err := json.Unmarshal([]byte(want), &wantValue); err != nil {
		t.Fatalf("the expected value of %s is not JSON: %v", what, err)
	}
	if !reflect.DeepEqual(gotValue, wantValue) {
		t.Errorf("%s =\n%s\nwant\n%s", what, got, want)
	}
}

// checkedInDeclaration is runtime/schema/tools.json, the tools that the worker checks each call
// against.
func checkedInDeclaration(t *testing.T) []byte {
	t.Helper()
	data, err := os.ReadFile("schema/tools.json")
	if err != nil {
		t.Fatalf("reading the declaration: %v", err)
	}
	return data
}

// socketPath is a path for a socket in a new directory under the system temporary directory.
// A path under t.TempDir() can exceed the 108 bytes of sun_path.
func socketPath(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "sprout-sock-")
	if err != nil {
		t.Fatalf("making a directory for the socket: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return filepath.Join(dir, "runtime.sock")
}

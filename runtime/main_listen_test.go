package main

import (
	"bytes"
	"context"
	"encoding/json"
	"io/fs"
	"net"
	"net/http"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

// The runtime is reached two ways: over TCP (--port, the microVM template's
// startCmd, behind AgentENV's proxy) and over a Unix socket (--listen unix:...,
// which the podman backend bind-mounts out of the container, so nothing has to
// be published on the host). These tests pin the socket's contract -- it exists
// with the stated mode, a stale one is replaced, anything else at the path is
// left alone -- and that the same JSON-RPC handler answers on both.

// serve runs the runtime's handler on ln and returns a client that dials it,
// whatever network ln speaks.
func serve(t *testing.T, ln net.Listener) *http.Client {
	t.Helper()
	srv := &http.Server{Handler: newHandler(tools.All())}
	go srv.Serve(ln)
	t.Cleanup(func() { srv.Close() })

	network, address := ln.Addr().Network(), ln.Addr().String()
	return &http.Client{Transport: &http.Transport{
		DialContext: func(ctx context.Context, _, _ string) (net.Conn, error) {
			return (&net.Dialer{}).DialContext(ctx, network, address)
		},
	}}
}

// rpc posts one JSON-RPC request and returns the decoded result, failing the
// test on a transport or protocol error.
func rpc(t *testing.T, client *http.Client, method string, params any) json.RawMessage {
	t.Helper()
	body, err := json.Marshal(map[string]any{
		"jsonrpc": "2.0", "id": 1, "method": method, "params": params,
	})
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// The host in the URL is irrelevant: the transport dials the listener.
	resp, err := client.Post("http://sprout-runtime/", "application/json", bytes.NewReader(body))
	if err != nil {
		t.Fatalf("%s: %v", method, err)
	}
	defer resp.Body.Close()
	var parsed struct {
		Result json.RawMessage `json:"result"`
		Error  *RPCError       `json:"error"`
	}
	if err := json.NewDecoder(resp.Body).Decode(&parsed); err != nil {
		t.Fatalf("%s: decoding the response: %v", method, err)
	}
	if parsed.Error != nil {
		t.Fatalf("%s: rpc error %d %s", method, parsed.Error.Code, parsed.Error.Message)
	}
	return parsed.Result
}

func toolNames(t *testing.T, client *http.Client) []string {
	t.Helper()
	var list []struct {
		Name string `json:"name"`
	}
	if err := json.Unmarshal(rpc(t, client, "tools/list", map[string]any{}), &list); err != nil {
		t.Fatalf("tools/list result: %v", err)
	}
	names := make([]string, 0, len(list))
	for _, item := range list {
		names = append(names, item.Name)
	}
	return names
}

func shellText(t *testing.T, client *http.Client, command string) string {
	t.Helper()
	var result struct {
		Content []struct {
			Text string `json:"text"`
		} `json:"content"`
		IsError bool `json:"isError"`
	}
	raw := rpc(t, client, "tools/call", map[string]any{
		"name": "shell", "arguments": map[string]any{"command": command},
	})
	if err := json.Unmarshal(raw, &result); err != nil {
		t.Fatalf("tools/call result: %v", err)
	}
	if result.IsError || len(result.Content) != 1 {
		t.Fatalf("tools/call failed: %s", raw)
	}
	return result.Content[0].Text
}

// socketPath is a short path under a fresh temp dir: sun_path is 108 bytes,
// and t.TempDir() under a long TMPDIR can exceed it.
func socketPath(t *testing.T) string {
	t.Helper()
	dir, err := os.MkdirTemp("", "sprout-sock-")
	if err != nil {
		t.Fatalf("temp dir: %v", err)
	}
	t.Cleanup(func() { os.RemoveAll(dir) })
	return filepath.Join(dir, "runtime.sock")
}

func TestTheUnixSocketServesTheProtocol(t *testing.T) {
	path := socketPath(t)
	ln, err := listenOn("unix:" + path)
	if err != nil {
		t.Fatalf("listenOn: %v", err)
	}
	defer ln.Close()

	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("the socket file was not created: %v", err)
	}
	if info.Mode()&fs.ModeSocket == 0 {
		t.Fatalf("%s is not a socket: %v", path, info.Mode())
	}
	if perm := info.Mode().Perm(); perm != 0o660 {
		t.Fatalf("socket mode = %o, want 660", perm)
	}
	if got := describeListener(ln); got != "unix:"+path {
		t.Fatalf("describeListener = %q, want %q", got, "unix:"+path)
	}

	client := serve(t, ln)
	if names := toolNames(t, client); len(names) != 1 || names[0] != "shell" {
		t.Fatalf("tools/list over the socket = %v, want [shell]", names)
	}
	if got := strings.TrimSpace(shellText(t, client, "echo over-unix")); got != "over-unix" {
		t.Fatalf("shell over the socket returned %q", got)
	}
}

func TestClosingTheListenerRemovesTheSocketFile(t *testing.T) {
	// A clean shutdown leaves nothing behind; the stale-file path below is for
	// the runtime that was killed instead.
	path := socketPath(t)
	ln, err := listenOn("unix:" + path)
	if err != nil {
		t.Fatalf("listenOn: %v", err)
	}
	ln.Close()
	if _, err := os.Lstat(path); !os.IsNotExist(err) {
		t.Fatalf("socket file still there after Close: %v", err)
	}
}

func TestAStaleSocketFileIsReplaced(t *testing.T) {
	// What a killed runtime leaves: the socket file with nobody listening.
	// Binding over it would fail with "address already in use".
	path := socketPath(t)
	stale, err := net.Listen("unix", path)
	if err != nil {
		t.Fatalf("making the stale socket: %v", err)
	}
	stale.(*net.UnixListener).SetUnlinkOnClose(false)
	stale.Close()
	if _, err := os.Lstat(path); err != nil {
		t.Fatalf("the stale socket should still be on disk: %v", err)
	}

	ln, err := listenOn("unix:" + path)
	if err != nil {
		t.Fatalf("listenOn over a stale socket: %v", err)
	}
	defer ln.Close()
	client := serve(t, ln)
	if names := toolNames(t, client); len(names) != 1 || names[0] != "shell" {
		t.Fatalf("tools/list after replacing the stale socket = %v", names)
	}
}

func TestANonSocketAtThePathIsLeftAlone(t *testing.T) {
	// Removing "whatever is there" would make a typo in --listen delete a file.
	path := socketPath(t)
	if err := os.WriteFile(path, []byte("not a socket"), 0o600); err != nil {
		t.Fatalf("write: %v", err)
	}
	ln, err := listenOn("unix:" + path)
	if err == nil {
		ln.Close()
		t.Fatal("listenOn replaced a regular file; it must refuse")
	}
	if !strings.Contains(err.Error(), "not a socket") {
		t.Fatalf("error should say why: %v", err)
	}
	content, readErr := os.ReadFile(path)
	if readErr != nil || string(content) != "not a socket" {
		t.Fatalf("the file was touched: %q %v", content, readErr)
	}
}

func TestTCPListensAndServesTheSameHandler(t *testing.T) {
	ln, err := listenOn("tcp:127.0.0.1:0")
	if err != nil {
		t.Fatalf("listenOn tcp: %v", err)
	}
	defer ln.Close()
	if !strings.HasPrefix(describeListener(ln), "tcp:127.0.0.1:") {
		t.Fatalf("describeListener = %q", describeListener(ln))
	}
	client := serve(t, ln)
	if got := strings.TrimSpace(shellText(t, client, "echo over-tcp")); got != "over-tcp" {
		t.Fatalf("shell over tcp returned %q", got)
	}
}

func TestListenSpecGrammar(t *testing.T) {
	for _, spec := range []string{"", "unix:", "tcp:", "udp:127.0.0.1:0", "127.0.0.1:3000", "/run/sprout/runtime.sock"} {
		ln, err := listenOn(spec)
		if err == nil {
			ln.Close()
			t.Errorf("listenOn(%q) accepted; want unix:<path> or tcp:<host:port>", spec)
		}
	}
}

func TestPortAndListenAreAlternatives(t *testing.T) {
	// --port alone is the TCP default on every interface, the address the
	// microVM template's startCmd and readyCmd assume.
	spec, err := listenSpec("", false, 3000)
	if err != nil || spec != "tcp:0.0.0.0:3000" {
		t.Fatalf("default = %q, %v; want tcp:0.0.0.0:3000", spec, err)
	}
	// --listen alone is taken as written.
	spec, err = listenSpec("unix:/run/sprout/runtime.sock", false, 3000)
	if err != nil || spec != "unix:/run/sprout/runtime.sock" {
		t.Fatalf("listen = %q, %v", spec, err)
	}
	// Both given: one would be silently ignored, so it is an error instead.
	if _, err := listenSpec("unix:/run/sprout/runtime.sock", true, 4000); err == nil {
		t.Fatal("--port with --listen must be refused")
	}
}

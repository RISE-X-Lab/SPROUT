// Tests of --listen unix:<path>. The podman backend bind-mounts the directory of the socket out
// of each container, so the socket file is the whole interface between the two.

package main

import (
	"errors"
	"io/fs"
	"os"
	"testing"
)

func TestTheSocketFileOfTheRuntimeHasMode0660(t *testing.T) {
	path := socketPath(t)

	startRuntime(t, nil, "--listen", "unix:"+path)

	info, err := os.Lstat(path)
	if err != nil {
		t.Fatalf("the socket file: %v", err)
	}
	if want := fs.ModeSocket | 0o660; info.Mode() != want {
		t.Errorf("mode of the socket file = %v, want %v", info.Mode(), want)
	}
}

func TestARuntimeReplacesTheSocketFileThatAKilledRuntimeLeft(t *testing.T) {
	path := socketPath(t)
	startRuntime(t, nil, "--listen", "unix:"+path).kill()

	runtime := startRuntime(t, nil, "--listen", "unix:"+path)

	assertReply(t, runtime.get(t), `{"status": "ok"}`)
}

func TestAFileThatIsNotASocketStopsTheRuntimeAndIsLeftUnchanged(t *testing.T) {
	path := socketPath(t)
	if err := os.WriteFile(path, []byte("not a socket"), 0o600); err != nil {
		t.Fatalf("writing the file: %v", err)
	}

	got := runRuntime(t, "--listen", "unix:"+path)

	want := exited{code: 1, stderr: "sprout-runtime: listening on unix socket " + path + ": the path exists and is not a socket\n"}
	if got != want {
		t.Errorf("the runtime exited %+v, want %+v", got, want)
	}
	if content, err := os.ReadFile(path); err != nil || string(content) != "not a socket" {
		t.Errorf("the file holds %q (%v), want it unchanged", content, err)
	}
}

func TestSIGTERMStopsTheRuntimeAndRemovesItsSocketFile(t *testing.T) {
	path := socketPath(t)
	runtime := startRuntime(t, nil, "--listen", "unix:"+path)
	// The banner comes before the SIGTERM handler, so a SIGTERM at once can kill the runtime with
	// the default action. A caller stops a runtime that has served it, as here.
	runtime.get(t)

	code := runtime.terminate(t)

	if code != 0 {
		t.Errorf("exit code = %d, want 0; stderr:\n%s", code, runtime.stderr.String())
	}
	if _, err := os.Lstat(path); !errors.Is(err, fs.ErrNotExist) {
		t.Errorf("the socket file is still there after SIGTERM: %v", err)
	}
}

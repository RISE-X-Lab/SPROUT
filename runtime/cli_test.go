// Tests of the command line of the runtime: --dump-schema, and the choice between --port and
// --listen.

package main

import (
	"regexp"
	"testing"
)

func TestDumpSchemaPrintsTheCheckedInDeclaration(t *testing.T) {
	declaration := checkedInDeclaration(t)

	got := runRuntime(t, "--dump-schema")

	if got.code != 0 {
		t.Errorf("exit code = %d, want 0; stderr:\n%s", got.code, got.stderr)
	}
	assertJSONEqual(t, "--dump-schema (if the change is intended, run `go run . --dump-schema > schema/tools.json`)",
		got.stdout, string(declaration))
}

func TestPortAndListenTogetherAreRefused(t *testing.T) {
	// 3000 is the default of --port, so only the presence of the flag can tell that it was given.
	got := runRuntime(t, "--port", "3000", "--listen", "unix:/path/to/runtime.sock")

	want := exited{code: 2, stderr: "sprout-runtime: --port and --listen are alternatives; give one\n"}
	if got != want {
		t.Errorf("the runtime exited %+v, want %+v", got, want)
	}
}

func TestAListenAddressOutsideTheGrammarIsRefused(t *testing.T) {
	cases := []struct {
		name   string
		listen string
		want   string
	}{
		{"no network", "/path/to/runtime.sock",
			`sprout-runtime: listen address "/path/to/runtime.sock": want unix:<path> or tcp:<host:port>` + "\n"},
		{"no address", "unix:",
			`sprout-runtime: listen address "unix:": want unix:<path> or tcp:<host:port>` + "\n"},
		{"an unknown network", "udp:127.0.0.1:0",
			`sprout-runtime: listen address "udp:127.0.0.1:0": unknown network "udp" (unix or tcp)` + "\n"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := runRuntime(t, "--listen", tc.listen)

			want := exited{code: 1, stderr: tc.want}
			if got != want {
				t.Errorf("the runtime exited %+v, want %+v", got, want)
			}
		})
	}
}

func TestPortAloneListensOnTCPOnEveryInterface(t *testing.T) {
	runtime := startRuntime(t, nil, "--port", "0")

	got := runtime.get(t)

	// Go names a listener on 0.0.0.0 by its dual-stack address, [::]. Both mean every interface.
	if !regexp.MustCompile(`^tcp:(0\.0\.0\.0|\[::\]):[0-9]+$`).MatchString(runtime.address) {
		t.Errorf("the runtime listens on %q, want tcp:0.0.0.0:<port> or tcp:[::]:<port>", runtime.address)
	}
	assertReply(t, got, `{"status": "ok"}`)
}

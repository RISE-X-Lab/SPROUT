// Tests of the JSON-RPC 2.0 endpoint. The harness (src/sprout_sandbox/backends.py) treats a status
// other than 200 as a transport failure, so each answer here, an error too, has status 200.

package main

import (
	"encoding/json"
	"testing"
)

func TestGetAnswersTheHealthCheck(t *testing.T) {
	runtime := startRuntime(t, nil, "--listen", "unix:"+socketPath(t))

	got := runtime.get(t)

	assertReply(t, got, `{"status": "ok"}`)
}

func TestToolsListAnswersWithTheToolsOfTheCheckedInDeclaration(t *testing.T) {
	var declaration struct {
		Tools json.RawMessage `json:"tools"`
	}
	if err := json.Unmarshal(checkedInDeclaration(t), &declaration); err != nil {
		t.Fatalf("schema/tools.json is not JSON: %v", err)
	}
	runtime := startRuntime(t, nil, "--listen", "unix:"+socketPath(t))

	got := runtime.post(t, `{"jsonrpc": "2.0", "id": 1, "method": "tools/list"}`)

	assertReply(t, got, `{"jsonrpc": "2.0", "id": 1, "result": `+string(declaration.Tools)+`}`)
}

func TestToolsCallCarriesTheErrorOfTheToolOnlyForARefusal(t *testing.T) {
	cases := []struct {
		name      string
		arguments string
		want      string
	}{
		{"a success carries its output", `{"command": "printf hello"}`,
			`{"content": [{"type": "text", "text": "hello"}], "isError": false}`},
		{"a failed command carries its output", `{"command": "exit 3"}`,
			`{"content": [{"type": "text", "text": "{\"stdout\":\"\",\"stderr\":\"\",\"exit_code\":3,` +
				`\"running\":false,\"timed_out\":false,\"stdout_bytes\":0,\"stderr_bytes\":0,` +
				`\"stdout_truncated\":false,\"stderr_truncated\":false,\"max_per_stream\":15728640}"}], "isError": true}`},
		{"a refusal carries its error", `{}`,
			`{"content": [{"type": "text", "text": "command is required"}], "isError": true}`},
	}
	runtime := startRuntime(t, nil, "--listen", "unix:"+socketPath(t))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := runtime.post(t, `{"jsonrpc": "2.0", "id": 1, "method": "tools/call", `+
				`"params": {"name": "shell", "arguments": `+tc.arguments+`}}`)

			assertReply(t, got, `{"jsonrpc": "2.0", "id": 1, "result": `+tc.want+`}`)
		})
	}
}

func TestAToolsCallThatNamesItsAgentRunsTheTool(t *testing.T) {
	// The harness sends agent_id beside the arguments (call_params in src/sprout_sandbox/backends.py).
	runtime := startRuntime(t, nil, "--listen", "unix:"+socketPath(t))

	got := runtime.post(t, `{"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": `+
		`{"name": "shell", "arguments": {"command": "printf hello"}, "agent_id": "root"}}`)

	assertReply(t, got, `{"jsonrpc": "2.0", "id": 1, "result": {"content": [{"type": "text", "text": "hello"}], "isError": false}}`)
}

func TestAMalformedRequestGetsItsJSONRPCErrorWithTheIDOfTheRequest(t *testing.T) {
	// The codes are those of the JSON-RPC 2.0 specification. A request that does not parse has no
	// ID to return, so its answer has the ID null.
	cases := []struct {
		name    string
		request string
		want    string
	}{
		{"a body that is not JSON", `{"jsonrpc": "2.0", "id": 7,`,
			`{"jsonrpc": "2.0", "id": null, "error": {"code": -32700, "message": "parse error"}}`},
		{"a version other than 2.0", `{"jsonrpc": "1.0", "id": 7, "method": "tools/list"}`,
			`{"jsonrpc": "2.0", "id": 7, "error": {"code": -32600, "message": "invalid request"}}`},
		{"an unknown method", `{"jsonrpc": "2.0", "id": 7, "method": "tools/delete"}`,
			`{"jsonrpc": "2.0", "id": 7, "error": {"code": -32601, "message": "method not found"}}`},
		{"arguments that are not an object",
			`{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "shell", "arguments": "echo hi"}}`,
			`{"jsonrpc": "2.0", "id": 7, "error": {"code": -32602, "message": "invalid params"}}`},
		{"an unknown tool",
			`{"jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": {"name": "nope", "arguments": {}}}`,
			`{"jsonrpc": "2.0", "id": 7, "error": {"code": -32602, "message": "unknown tool: nope"}}`},
	}
	runtime := startRuntime(t, nil, "--listen", "unix:"+socketPath(t))
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := runtime.post(t, tc.request)

			assertReply(t, got, tc.want)
		})
	}
}

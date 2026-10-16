package main

import (
	"context"
	"encoding/json"
	"errors"
	"flag"
	"fmt"
	"io/fs"
	"log"
	"net"
	"net/http"
	"os"
	"os/signal"
	"strings"
	"sync"
	"syscall"
	"time"

	// Embedded CA roots, used only when the image has none, so web_fetch and
	// web_search reach HTTPS from images without ca-certificates.
	_ "golang.org/x/crypto/x509roots/fallback"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

type RPCRequest struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Method  string          `json:"method"`
	Params  json.RawMessage `json:"params,omitempty"`
}

type RPCResponse struct {
	JSONRPC string          `json:"jsonrpc"`
	ID      json.RawMessage `json:"id"`
	Result  any             `json:"result,omitempty"`
	Error   *RPCError       `json:"error,omitempty"`
}

type RPCError struct {
	Code    int    `json:"code"`
	Message string `json:"message"`
}

// runtimeVersion is reported by --dump-schema and the HTTP banner. The tool set
// is the protocol: 0.2.0 served shell alone, 0.3.0 adds the four others back.
const runtimeVersion = "0.3.0"

// toolDescriptors is the runtime's declaration of what it serves: every tool's
// name, description and input schema. One function because there are two
// consumers -- tools/list over HTTP and --dump-schema -- and a hand-rolled copy
// of the same loop is how they would drift apart.
func toolDescriptors(allTools []tools.Tool) []map[string]any {
	list := make([]map[string]any, 0, len(allTools))
	for _, t := range allTools {
		list = append(list, map[string]any{
			"name":        t.Name(),
			"description": t.Description(),
			"inputSchema": t.Schema(),
		})
	}
	return list
}

func main() {
	port := flag.Int("port", 3000, "TCP port to listen on, on every interface")
	listen := flag.String("listen", "",
		"where to listen instead of --port: unix:<path> (a Unix socket, mode 0660, "+
			"a stale socket file replaced) or tcp:<host:port>")
	dumpSchema := flag.Bool("dump-schema", false,
		"print the tool declarations as JSON and exit (no sandbox needed)")
	flag.Parse()

	allTools := tools.All()

	// The harness validates every agent call against this declaration
	// (runtime/schema/tools.json, checked in), and that has to work with no
	// sandbox running -- in tests, and when validating a config before a run.
	// So the runtime can state what it serves without serving anything.
	if *dumpSchema {
		out, err := json.MarshalIndent(map[string]any{
			"version": runtimeVersion,
			"tools":   toolDescriptors(allTools),
		}, "", "  ")
		if err != nil {
			fmt.Fprintf(os.Stderr, "dump-schema: %v\n", err)
			os.Exit(1)
		}
		fmt.Println(string(out))
		return
	}

	portGiven := false
	flag.Visit(func(f *flag.Flag) {
		if f.Name == "port" {
			portGiven = true
		}
	})
	spec, err := listenSpec(*listen, portGiven, *port)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sprout-runtime: %v\n", err)
		os.Exit(2)
	}
	ln, err := listenOn(spec)
	if err != nil {
		fmt.Fprintf(os.Stderr, "sprout-runtime: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("sprout-runtime v%s\n  tools: %d loaded\n  listening on %s\n",
		runtimeVersion, len(allTools), describeListener(ln))

	srv := &http.Server{
		Handler:           newHandler(allTools),
		ReadHeaderTimeout: 10 * time.Second,
		ReadTimeout:       30 * time.Second,
		// No WriteTimeout: it is a deadline on the whole response, and a tool
		// call's response is only written once its command finishes, so any
		// value here is really a cap on how long a command may run -- one that
		// silently disagrees with the caller's own `timeout` argument. Killing
		// the command is the shell tool's job, where the caller's timeout is
		// known; the write deadline is set per request in beginResponse instead.
		IdleTimeout: 120 * time.Second,
	}
	go func() {
		if err := srv.Serve(ln); err != nil && !errors.Is(err, http.ErrServerClosed) {
			log.Fatalf("server error: %v", err)
		}
	}()

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)
	<-quit
	log.Println("shutting down...")
	ctx, cancel := context.WithTimeout(context.Background(), 5*time.Second)
	defer cancel()
	// Closing the listener unlinks a Unix socket file, so a clean stop leaves
	// nothing behind; a runtime that was killed does, and listenUnix clears it.
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("forced shutdown: %v", err)
	}
	log.Println("server stopped")
}

// listenSpec resolves the two listening flags into one address of the form
// listenOn takes. --port is the TCP default (every interface, as the microVM
// template's startCmd relies on); --listen names a Unix socket or a TCP address
// outright. Giving both would leave one silently ignored, so it is refused.
func listenSpec(listen string, portGiven bool, port int) (string, error) {
	if listen == "" {
		return fmt.Sprintf("tcp:0.0.0.0:%d", port), nil
	}
	if portGiven {
		return "", errors.New("--port and --listen are alternatives; give one")
	}
	return listen, nil
}

// listenOn opens the listener a --listen value names. The grammar is
// <network>:<address>: "tcp:0.0.0.0:3000" or "unix:/run/sprout/runtime.sock".
func listenOn(spec string) (net.Listener, error) {
	network, address, ok := strings.Cut(spec, ":")
	if !ok || address == "" {
		return nil, fmt.Errorf("listen address %q: want unix:<path> or tcp:<host:port>", spec)
	}
	switch network {
	case "tcp":
		ln, err := net.Listen("tcp", address)
		if err != nil {
			return nil, fmt.Errorf("listening on %s: %w", spec, err)
		}
		return ln, nil
	case "unix":
		return listenUnix(address)
	default:
		return nil, fmt.Errorf("listen address %q: unknown network %q (unix or tcp)", spec, network)
	}
}

// listenUnix binds a Unix socket at path.
//
// A socket file already there was left by a runtime that did not exit cleanly
// (a killed container, a sandbox restarted on the same socket directory);
// binding over it fails with "address already in use", so it is removed first.
// Only a socket is removed: anything else at the path is someone's file and a
// misconfiguration, not a stale runtime.
//
// The socket is 0660 -- the owner and its group reach the runtime, nobody else
// on the host. Set by chmod after the bind rather than through the umask, which
// is process-wide; the directory the socket lives in is the boundary during
// that instant, as it is in general (the podman backend bind-mounts a directory
// per sandbox).
func listenUnix(path string) (net.Listener, error) {
	if info, err := os.Lstat(path); err == nil {
		if info.Mode()&fs.ModeSocket == 0 {
			return nil, fmt.Errorf("listening on unix socket %s: the path exists and is not a socket", path)
		}
		if err := os.Remove(path); err != nil {
			return nil, fmt.Errorf("removing stale socket %s: %w", path, err)
		}
	} else if !errors.Is(err, fs.ErrNotExist) {
		return nil, fmt.Errorf("listening on unix socket %s: %w", path, err)
	}
	ln, err := net.Listen("unix", path)
	if err != nil {
		return nil, fmt.Errorf("listening on unix socket %s: %w", path, err)
	}
	if err := os.Chmod(path, 0o660); err != nil {
		ln.Close()
		return nil, fmt.Errorf("setting the mode of %s: %w", path, err)
	}
	return ln, nil
}

// describeListener renders a listener's address in the --listen grammar.
func describeListener(ln net.Listener) string {
	return ln.Addr().Network() + ":" + ln.Addr().String()
}

// newHandler is the JSON-RPC endpoint, the same whatever it listens on: a
// health check on GET, tools/list and tools/call on POST.
func newHandler(allTools []tools.Tool) http.Handler {
	mux := http.NewServeMux()

	mux.HandleFunc("/", func(w http.ResponseWriter, r *http.Request) {
		if r.Method == http.MethodGet {
			w.Header().Set("Content-Type", "application/json")
			w.Write([]byte(`{"status":"ok"}`))
			return
		}
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}

		var req RPCRequest
		if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
			writeRPC(w, nil, nil, &RPCError{Code: -32700, Message: "parse error"})
			return
		}
		if req.JSONRPC != "2.0" {
			writeRPC(w, req.ID, nil, &RPCError{Code: -32600, Message: "invalid request"})
			return
		}

		switch req.Method {
		case "tools/list":
			list := toolDescriptors(allTools)
			writeRPC(w, req.ID, list, nil)

		case "tools/call":
			var params struct {
				Name      string         `json:"name"`
				Arguments map[string]any `json:"arguments"`
			}
			if err := json.Unmarshal(req.Params, &params); err != nil {
				writeRPC(w, req.ID, nil, &RPCError{Code: -32602, Message: "invalid params"})
				return
			}

			var target tools.Tool
			for _, t := range allTools {
				if t.Name() == params.Name {
					target = t
					break
				}
			}
			if target == nil {
				writeRPC(w, req.ID, nil, &RPCError{Code: -32602, Message: "unknown tool: " + params.Name})
				return
			}

			// Headers first: the tool is about to take as long as its command
			// does, and a proxy watching for a response gives up long before.
			stop, _ := beginResponse(w)
			result := executeTool(target, params.Arguments)
			stop()

			text := result.Output
			if !result.Success && result.Error != "" {
				text = result.Error
			}
			writeRPC(w, req.ID, map[string]any{
				"content": []map[string]any{{"type": "text", "text": text}},
				"isError": !result.Success,
			}, nil)

		default:
			writeRPC(w, req.ID, nil, &RPCError{Code: -32601, Message: "method not found"})
		}
	})

	return mux
}

// executeTool is the one dispatch path from a tools/call request to a tool. A
// request may omit "arguments" altogether; the tool then reads from an empty
// map rather than a nil one.
func executeTool(target tools.Tool, args map[string]any) tools.Result {
	if args == nil {
		args = map[string]any{}
	}
	return target.Execute(args)
}

func writeRPC(w http.ResponseWriter, id json.RawMessage, result any, rpcErr *RPCError) {
	resp := RPCResponse{JSONRPC: "2.0", ID: id, Result: result, Error: rpcErr}
	w.Header().Set("Content-Type", "application/json")
	json.NewEncoder(w).Encode(resp)
}

// keepaliveInterval is how often a long-running call writes a byte to prove the
// connection is alive. Comfortably under the 30 s that reverse proxies commonly
// allow for response headers, with room for one lost tick.
var keepaliveInterval = 10 * time.Second

// beginResponse sends the response headers before the tool runs, then keeps the
// connection warm until stop() is called.
//
// A tool call takes as long as its command does -- minutes, for a test suite or
// a compile. The runtime answers in one shot, so nothing at all reaches the
// client until the command finishes, and a proxy in between reads that silence
// as a dead upstream: AgentENV cuts the connection at 30 s
// (response_header_timeout_ms=30000), and any gateway-style backend may do the
// same. The command keeps running in the sandbox, so the caller loses the result
// of work that actually happened.
//
// Sending the headers early answers the only question a proxy is asking. The
// heartbeat is a newline, which JSON ignores as leading whitespace, so a client
// that knows nothing about any of this parses the response exactly as before --
// no protocol change, no client change, no version negotiation.
//
// Returns a stop function; the caller must invoke it before writing the body.
// Also reports whether headers were actually sent, since a ResponseWriter that
// cannot flush (a test recorder, a wrapped writer) must fall back to writing the
// whole response at the end.
func beginResponse(w http.ResponseWriter) (stop func(), started bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return func() {}, false
	}

	// Clear the write deadline for this response. A tool call is not done until
	// its command is, and the caller states how long that may be (`timeout`); a
	// transport-level deadline would cut the connection while the command still
	// runs, losing the result of work that happened. The shell tool remains the
	// thing that stops a command.
	if rc := http.NewResponseController(w); rc != nil {
		_ = rc.SetWriteDeadline(time.Time{})
	}

	w.Header().Set("Content-Type", "application/json")
	// A tool call answers 200 even when the tool fails: the failure travels in
	// the JSON-RPC body (or isError), not the HTTP status. That is what makes it
	// safe to commit to a status code before knowing the outcome.
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	done := make(chan struct{})
	exited := make(chan struct{})
	go func() {
		defer close(exited)
		ticker := time.NewTicker(keepaliveInterval)
		defer ticker.Stop()
		for {
			select {
			case <-done:
				return
			case <-ticker.C:
				// Writing after the handler returns panics, so the write must
				// happen only while the handler is still running -- guaranteed
				// because stop() runs before the handler writes the body.
				if _, err := w.Write([]byte("\n")); err != nil {
					return // client hung up; the body write will fail too
				}
				flusher.Flush()
			}
		}
	}()

	var once sync.Once
	return func() {
		// Wait for the heartbeat to exit, not just signal it: a ResponseWriter
		// is not safe for concurrent use, and the caller writes the body next.
		once.Do(func() {
			close(done)
			<-exited
		})
	}, true
}

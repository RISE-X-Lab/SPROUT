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

// runtimeVersion is the version that --dump-schema and the startup banner report. It changes
// with the tool set, because the tool set is the protocol.
const runtimeVersion = "0.3.0"

// toolDescriptors is the runtime's declaration of what it serves: each tool's name,
// description and input schema. tools/list and --dump-schema both answer with it, so the two
// cannot drift apart.
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

	// ToolAccess checks each agent call against this declaration (runtime/schema/tools.json) with
	// no sandbox running, so the runtime states what it serves without serving anything.
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
		// No WriteTimeout: a tool call's response ends when its command does, so a write deadline
		// would cap the command's run time. The shell tool stops a command at the caller's `timeout`.
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
	// Shutdown closes the listener, which unlinks a Unix socket file. A killed runtime leaves the
	// file behind, and listenUnix removes it at the next start.
	if err := srv.Shutdown(ctx); err != nil {
		log.Fatalf("forced shutdown: %v", err)
	}
	log.Println("server stopped")
}

// listenSpec resolves the two listening flags into one address in the form that listenOn takes.
// Without --listen it is TCP on every interface at port, which the microVM template's startCmd
// relies on. --listen is taken as written. If --port was also given (portGiven), it returns an
// error, because one of the two flags would be silently ignored.
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

// listenUnix binds a Unix socket at path with mode 0660: only the owner and its group reach the
// runtime. It removes a stale socket file at path first. It returns an error, and leaves the
// file alone, if anything other than a socket is at path.
func listenUnix(path string) (net.Listener, error) {
	// A runtime that did not exit cleanly leaves its socket file, and a bind over it fails with
	// "address already in use". Anything else at the path is someone's file and a misconfiguration.
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
	// Chmod after the bind, not the umask, which is process-wide. Until then the directory is the
	// boundary, as it is in general: the podman backend bind-mounts one directory per sandbox.
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

// executeTool is the one dispatch path from a tools/call request to a tool. If the request
// omits "arguments", the tool gets an empty map, not nil.
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

// keepaliveInterval is how often a long-running call writes a byte to show that the connection
// is alive. It is well under the 30 s that reverse proxies commonly allow for response headers,
// with room for one lost tick.
var keepaliveInterval = 10 * time.Second

// beginResponse sends the response headers before the tool runs, and then writes a newline
// every keepaliveInterval until stop is called. The caller must call stop before it writes the
// body. started is false if w cannot flush, for example a test recorder or a wrapped writer.
// Then nothing was sent, stop does nothing, and the whole response is written at the end.
func beginResponse(w http.ResponseWriter) (stop func(), started bool) {
	flusher, ok := w.(http.Flusher)
	if !ok {
		return func() {}, false
	}

	// Clear the write deadline: a transport deadline would cut the connection while the command
	// runs, and its result would be lost. The shell tool stops a command at the caller's `timeout`.
	if rc := http.NewResponseController(w); rc != nil {
		_ = rc.SetWriteDeadline(time.Time{})
	}

	// A proxy reads a silent upstream as dead. AgentENV cuts a connection that has sent no
	// response headers after 30 s (response_header_timeout_ms=30000), while the command still runs.
	w.Header().Set("Content-Type", "application/json")
	// Status 200 is safe before the outcome is known: a tool call reports a failure in the
	// JSON-RPC body (error or isError), never in the HTTP status.
	w.WriteHeader(http.StatusOK)
	flusher.Flush()

	// Each heartbeat is a newline. JSON ignores it as leading whitespace, so a client that does
	// not know about the heartbeat parses the response as before.
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
				// A write after the handler returns panics. stop() runs before the handler writes
				// the body, so each heartbeat write happens while the handler still runs.
				if _, err := w.Write([]byte("\n")); err != nil {
					return // the client hung up, and the body write fails too
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

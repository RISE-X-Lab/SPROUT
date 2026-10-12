package tools

// Tool is the interface a sandbox tool implements. The runtime serves one, and
// the interface stays because the HTTP handler and --dump-schema are written
// against a list of declarations rather than against ShellTool itself.
type Tool interface {
	Name() string
	Description() string
	Schema() map[string]any
	Execute(args map[string]any) Result
}

// Result of a tool execution.
//
// Success answers "did this work". The two text fields say where to read why:
// Output is what the tool produced, Error is the tool refusing to produce
// anything. They are alternatives, never two copies of one message -- the wire
// format carries a single text slot, so a duplicate would reach the model twice.
type Result struct {
	Success bool
	Output  string
	Error   string
}

func Ok(output string) Result {
	return Result{Success: true, Output: output}
}

// Err reports that the tool could not run the request at all: a missing or
// invalid argument, an unknown action. There is no output in this case, only a
// reason.
func Err(msg string) Result {
	return Result{Success: false, Error: msg}
}

// All returns the tools the runtime serves: shell alone. Every action an agent
// takes in the sandbox is a command; editing, searching and fetching are
// programs it runs, not tools the runtime knows about.
func All() []Tool {
	return []Tool{&ShellTool{}}
}

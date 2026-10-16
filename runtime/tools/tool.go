package tools

// Tool is the interface all sandbox tools implement.
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

// Err reports that the tool could not run the request at all (a bad argument,
// an unknown action, an unreachable URL): no output, only a reason.
func Err(msg string) Result {
	return Result{Success: false, Error: msg}
}

// All returns the tools the runtime serves, in the order tools/list reports
// them.
func All() []Tool {
	return []Tool{
		&ShellTool{},
		&EditTool{},
		&GrepTool{},
		&WebFetchTool{},
		&WebSearchTool{},
	}
}

package tools

// Tool is one sandbox tool that the runtime serves. tools/list reports its Name,
// Description and Schema, the JSON Schema of its arguments. tools/call passes the
// decoded arguments to Execute, as an empty map when the request has none.
type Tool interface {
	Name() string
	Description() string
	Schema() map[string]any
	Execute(args map[string]any) Result
}

// Result is the outcome of one tool call. Success says whether the call worked.
// Output is what the tool produced. Error means that the tool refused to produce
// anything. Set only one of them. tools/call carries one text: Error if the call
// failed and Error is set, else Output.
type Result struct {
	Success bool
	Output  string
	Error   string
}

// Ok reports a call that worked, with what it produced.
func Ok(output string) Result {
	return Result{Success: true, Output: output}
}

// Err reports that the tool could not run the request at all, for example a bad
// argument, an unknown action or an unreachable URL. It carries only the reason.
func Err(msg string) Result {
	return Result{Success: false, Error: msg}
}

// All returns the tools that the runtime serves, in the order that tools/list
// reports them.
func All() []Tool {
	return []Tool{
		&ShellTool{},
		&EditTool{},
		&GrepTool{},
		&WebFetchTool{},
		&WebSearchTool{},
	}
}

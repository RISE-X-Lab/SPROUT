package tools

import (
	"context"
	"os"
	"os/exec"
	"syscall"
	"time"
)

// defaultTimeoutSeconds bounds a command whose caller named no timeout.
const defaultTimeoutSeconds = 300

// ShellTool runs one command in the foreground and reports what it produced.
type ShellTool struct{}

func (s *ShellTool) Name() string { return "shell" }

func (s *ShellTool) Description() string {
	return "Execute a shell command and return its output"
}

func (s *ShellTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"command": map[string]any{"type": "string", "description": "Shell command to execute"},
			"timeout": map[string]any{"type": "integer", "default": defaultTimeoutSeconds, "description": "Timeout in seconds"},
			"max_output_bytes": map[string]any{
				"type":        "integer",
				"default":     defaultMaxOutputBytes,
				"description": "Total captured bytes per output stream. Larger output is truncated to the first 40% and last 60% of the budget, and the outcome says so.",
			},
			"working_dir": map[string]any{"type": "string", "description": "Working directory"},
		},
		"required": []string{"command"},
	}
}

func (s *ShellTool) Execute(args map[string]any) Result {
	command, _ := args["command"].(string)
	if command == "" {
		return Err("command is required")
	}

	timeout := defaultTimeoutSeconds
	if t, ok := args["timeout"].(float64); ok {
		timeout = int(t)
	}
	workingDir, _ := args["working_dir"].(string)
	maxOutputBytes := outputBytesArg(args)

	ctx, cancel := context.WithTimeout(context.Background(), time.Duration(timeout)*time.Second)
	defer cancel()

	cmd := exec.CommandContext(ctx, "sh", "-c", command)
	setProcessGroup(cmd)
	// CommandContext's default Cancel kills only cmd.Process. A shell pipeline
	// or backgrounded job leaves descendants alive, and they can keep the
	// stdout/stderr pipes open after the shell dies so Run never returns.
	// Cancel the whole process group, then bound pipe cleanup in case a child
	// escaped the group or retained an inherited descriptor.
	cmd.Cancel = func() error { return cancelProcessGroup(cmd) }
	cmd.WaitDelay = 2 * time.Second
	if workingDir != "" {
		cmd.Dir = workingDir
	}

	stdout := NewBoundedLog(maxOutputBytes)
	stderr := NewBoundedLog(maxOutputBytes)
	cmd.Stdout = stdout
	cmd.Stderr = stderr

	err := cmd.Run()

	// A command that ran and failed is not a tool error. Success=false says it
	// failed and the exit code says how; Error stays reserved for the tool
	// refusing the request (bad arguments) -- the meaning it carries at every
	// other Err() call site.
	outcome := commandOutcome(stdout, stderr)
	// exitCodeFromErr: Run() has already waited, and a second Wait() fails with
	// "already called" -- not an ExitError, so every command would report a
	// generic 1, successes included.
	code := exitCodeFromErr(err)
	outcome.ExitCode = &code
	outcome.TimedOut = ctx.Err() == context.DeadlineExceeded
	return outcome.Result()
}

// setProcessGroup puts the command in its own process group, so the whole job
// can be signalled later. Killing only the direct child left grandchildren
// running -- and a `cmd &` or a pipeline is the usual shape of a long job --
// which also held the output pipes open and stalled cmd.Wait() indefinitely.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// cancelProcessGroup signals the entire job, falling back to the direct child
// if signalling the group fails. Returning the error lets exec.CommandContext
// distinguish a successful cancellation from a process that already exited.
func cancelProcessGroup(cmd *exec.Cmd) error {
	if cmd == nil || cmd.Process == nil {
		return os.ErrProcessDone
	}
	// Negative pid means "the group with this leader". Setpgid made the child
	// its own leader, so this reaches its descendants too.
	if err := syscall.Kill(-cmd.Process.Pid, syscall.SIGKILL); err == nil {
		return nil
	}
	return cmd.Process.Kill()
}

// exitCodeFromErr turns the error from Run() into an exit code, using the shell
// convention a caller already knows: 128+signal for a signalled death. Go's
// ExitCode() returns -1 there, which says only "not a normal exit".
func exitCodeFromErr(err error) int {
	if err == nil {
		return 0
	}
	exitErr, ok := err.(*exec.ExitError)
	if !ok {
		return 1
	}
	if status, ok := exitErr.Sys().(syscall.WaitStatus); ok && status.Signaled() {
		return 128 + int(status.Signal())
	}
	return exitErr.ExitCode()
}

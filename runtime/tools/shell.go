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
	// The default Cancel kills only `sh`, and its descendants keep the pipes open.
	// So kill the whole group. WaitDelay ends the wait on a pipe held outside it.
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

	// A command that ran and failed is not a tool error: Success is false and the
	// exit code says how. Error means only that the tool refused the request.
	outcome := commandOutcome(stdout, stderr)
	// Take the code from Run's error. Run has already waited, and a second Wait
	// returns an error that is not an ExitError, so every command would get 1.
	code := exitCodeFromErr(err)
	outcome.ExitCode = &code
	outcome.TimedOut = ctx.Err() == context.DeadlineExceeded
	return outcome.Result()
}

// setProcessGroup puts the command in its own process group, so that
// cancelProcessGroup can signal the whole job. A pipeline or a `cmd &` has
// descendants, and they hold the output pipes open after `sh` dies.
func setProcessGroup(cmd *exec.Cmd) {
	if cmd.SysProcAttr == nil {
		cmd.SysProcAttr = &syscall.SysProcAttr{}
	}
	cmd.SysProcAttr.Setpgid = true
}

// cancelProcessGroup sends SIGKILL to the whole job, or to the direct child if
// the group signal fails. It returns the error of that kill, so that exec.Cmd
// can tell a cancellation from a process that had already exited.
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

// exitCodeFromErr turns the error from Run() into an exit code. A signalled
// death is 128+signal, the shell convention, where Go's ExitCode() says only -1.
// An error that is not an ExitError, such as a failure to start, is 1.
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

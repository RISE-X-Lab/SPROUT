package tools

import (
	"bufio"
	"bytes"
	"fmt"
	"os/exec"
	"strings"
)

// GrepTool searches files with the ripgrep embedded in the runtime (ripgrep.go).
// It returns at most `limit` matching lines and says when it stopped there. A
// relative path resolves against the runtime's working directory, not a task's.
type GrepTool struct{}

func (g *GrepTool) Name() string { return "grep_files" }

func (g *GrepTool) Description() string {
	return "Search files using ripgrep with a regex pattern"
}

func (g *GrepTool) Schema() map[string]any {
	props := map[string]any{
		"pattern": map[string]any{"type": "string", "description": "Regex pattern"},
		"path":    map[string]any{"type": "string", "default": ".", "description": "Search path"},
		"include": map[string]any{"type": "string", "description": "File glob (e.g. *.py)"},
		"limit":   map[string]any{"type": "integer", "default": 100, "description": "Maximum number of matching lines to return globally"},
		// Matches are bounded by `limit` (count) and by the shared byte budget.
		"max_output_bytes": outputBytesSchema(),
	}
	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   []string{"pattern"},
	}
}

func (g *GrepTool) Execute(args map[string]any) Result {
	rg, err := ripgrep()
	if err != nil {
		return Err("ripgrep unavailable: " + err.Error())
	}

	pattern, _ := args["pattern"].(string)
	if pattern == "" {
		return Err("pattern is required")
	}

	path := "."
	if p, ok := args["path"].(string); ok && p != "" {
		path = p
	}

	limit := 100
	if l, ok := args["limit"].(float64); ok && int(l) > 0 {
		limit = int(l)
	}

	cmdArgs := []string{
		"--line-number", "--no-heading", "--color=never",
		"--regexp", pattern,
	}
	if include, ok := args["include"].(string); ok && include != "" {
		cmdArgs = append(cmdArgs, "--glob", include)
	}
	cmdArgs = append(cmdArgs, "--", path)

	cmd := exec.Command(rg, cmdArgs...)
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		return Err("failed to create stdout pipe: " + err.Error())
	}
	var stderr bytes.Buffer
	cmd.Stderr = &stderr

	if err := cmd.Start(); err != nil {
		return Err("failed to start ripgrep: " + err.Error())
	}

	var lines []string
	truncated := false
	scanner := bufio.NewScanner(stdout)
	scanner.Buffer(make([]byte, 64*1024), 1024*1024)
	for scanner.Scan() {
		if len(lines) >= limit {
			truncated = true
			_ = cmd.Process.Kill()
			break
		}
		lines = append(lines, scanner.Text())
	}
	scanErr := scanner.Err()
	err = cmd.Wait()

	if scanErr != nil {
		return Err("read ripgrep output: " + scanErr.Error())
	}
	if len(lines) > 0 {
		output := strings.Join(lines, "\n")
		if truncated {
			output += fmt.Sprintf("\n... (truncated at %d matches)", limit)
		}
		// `limit` bounds the number of matches, not their size. A few very long
		// lines still need the shared byte bound.
		return Ok(boundText(output+"\n", outputBytesArg(args)))
	}
	if exitErr, ok := err.(*exec.ExitError); ok && exitErr.ExitCode() == 1 {
		return Ok("No matches found.")
	}
	if truncated {
		return Ok(fmt.Sprintf("... (truncated at %d matches)\n", limit))
	}
	if err == nil {
		return Ok("")
	}
	return Err(stderr.String())
}

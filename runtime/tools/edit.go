package tools

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
)

// EditTool views and edits files as the runtime's own process: relative paths
// resolve against its working directory, not a task's, so pass absolute ones.
type EditTool struct{}

func (e *EditTool) Name() string { return "text_editor" }

func (e *EditTool) Description() string {
	return "View or edit files. command=view reads a file or range; command=str_replace requires old_str and new_str; command=insert requires insert_line and insert_text; command=write requires file_text and creates or overwrites the full file."
}

func (e *EditTool) Schema() map[string]any {
	props := map[string]any{
		"command":     map[string]any{"type": "string", "enum": []string{"view", "str_replace", "insert", "write"}, "description": "Command to execute: view, str_replace, insert, or write"},
		"path":        map[string]any{"type": "string", "description": "File path"},
		"view_range":  map[string]any{"type": "array", "items": map[string]any{"type": "integer"}, "description": "For view only: [start, end] inclusive line range."},
		"old_str":     map[string]any{"type": "string", "description": "Required for str_replace. Text to find; must match exactly once."},
		"new_str":     map[string]any{"type": "string", "description": "Required for str_replace. Replacement text."},
		"insert_line": map[string]any{"type": "integer", "description": "Required for insert. Line number to insert after; use 0 to insert at the start."},
		"insert_text": map[string]any{"type": "string", "description": "Required for insert. Text to insert."},
		"file_text":   map[string]any{"type": "string", "description": "Required for write. Full file content; creates or overwrites the file."},
		// view produces byte output, so it takes the shared output bound.
		"max_output_bytes": outputBytesSchema(),
	}
	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   []string{"command", "path"},
	}
}

func (e *EditTool) Execute(args map[string]any) Result {
	command, _ := args["command"].(string)
	path, _ := args["path"].(string)
	if path == "" {
		return Err("path is required")
	}

	switch command {
	case "view":
		return e.view(path, args)
	case "str_replace":
		return e.strReplace(path, args)
	case "insert":
		return e.insert(path, args)
	case "write":
		return e.write(path, args)
	default:
		return Err("unknown command: " + command)
	}
}

func (e *EditTool) view(path string, args map[string]any) Result {
	maxBytes := outputBytesArg(args)

	// Read one budget plus a probe byte: viewing a multi-gigabyte log must not
	// pull it all into memory to throw most of it away.
	f, err := os.Open(path)
	if err != nil {
		return Err(err.Error())
	}
	defer f.Close()
	data, err := io.ReadAll(io.LimitReader(f, int64(maxBytes)+1))
	if err != nil {
		return Err(err.Error())
	}
	fileTruncated := len(data) > maxBytes
	if fileTruncated {
		data = data[:maxBytes]
	}

	lines := strings.Split(string(data), "\n")
	start, end := 1, len(lines)

	if vr, ok := args["view_range"].([]any); ok && len(vr) == 2 {
		if s, ok := vr[0].(float64); ok {
			start = int(s)
		}
		if e, ok := vr[1].(float64); ok {
			end = int(e)
		}
	}

	if start < 1 {
		start = 1
	}
	if end > len(lines) {
		end = len(lines)
	}

	var b strings.Builder
	for i := start - 1; i < end; i++ {
		fmt.Fprintf(&b, "  %6d | %s\n", i+1, lines[i])
	}

	// Bound the rendered view with the shared mechanism, then say so if the
	// file itself was cut short at read time.
	out := boundText(b.String(), maxBytes)
	if fileTruncated {
		out += fmt.Sprintf(
			"\n... [file larger than %d bytes: only the first %d bytes were read; "+
				"use view_range, or raise max_output_bytes] ...\n", maxBytes, maxBytes)
	}
	return Ok(out)
}

func (e *EditTool) strReplace(path string, args map[string]any) Result {
	oldStr, ok := args["old_str"].(string)
	if !ok || oldStr == "" {
		return Err("old_str is required for str_replace")
	}
	newStr, ok := args["new_str"].(string)
	if !ok {
		return Err("new_str is required for str_replace")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return Err(err.Error())
	}

	content := string(data)
	count := strings.Count(content, oldStr)

	if count == 0 {
		return Err("No match found")
	}
	if count > 1 {
		return Err(fmt.Sprintf("Multiple matches (%d). old_str must be unique.", count))
	}

	content = strings.Replace(content, oldStr, newStr, 1)
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return Err(err.Error())
	}

	return Ok("OK")
}

func (e *EditTool) insert(path string, args map[string]any) Result {
	rawInsertLine, ok := args["insert_line"].(float64)
	if !ok {
		return Err("insert_line is required for insert")
	}
	insertLine := int(rawInsertLine)
	insertText, ok := args["insert_text"].(string)
	if !ok {
		return Err("insert_text is required for insert")
	}

	data, err := os.ReadFile(path)
	if err != nil {
		return Err(err.Error())
	}

	lines := strings.Split(string(data), "\n")
	if insertLine < 0 || insertLine > len(lines) {
		return Err(fmt.Sprintf("insert_line out of range: got %d, want 0..%d", insertLine, len(lines)))
	}
	newLines := strings.Split(insertText, "\n")

	result := make([]string, 0, len(lines)+len(newLines))
	result = append(result, lines[:insertLine]...)
	result = append(result, newLines...)
	result = append(result, lines[insertLine:]...)

	if err := os.WriteFile(path, []byte(strings.Join(result, "\n")), 0644); err != nil {
		return Err(err.Error())
	}

	return Ok("OK")
}

func (e *EditTool) write(path string, args map[string]any) Result {
	fileText, ok := args["file_text"].(string)
	if !ok {
		return Err("file_text is required for write")
	}

	if err := os.MkdirAll(filepath.Dir(path), 0755); err != nil {
		return Err(err.Error())
	}
	if err := os.WriteFile(path, []byte(fileText), 0644); err != nil {
		return Err(err.Error())
	}

	return Ok("OK")
}

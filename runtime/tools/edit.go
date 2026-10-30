package tools

import (
	"fmt"
	"io"
	"os"
	"path/filepath"
	"strings"
	"syscall"
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
	if oldStr == newStr {
		return Err("old_str and new_str are identical; nothing to change")
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
		return Err(fmt.Sprintf("Multiple matches (%d), starting at lines %s. old_str must be unique.",
			count, matchLines(content, oldStr)))
	}

	at := strings.Index(content, oldStr)
	content = content[:at] + newStr + content[at+len(oldStr):]
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		return Err(err.Error())
	}

	start := strings.Count(content[:at], "\n") + 1
	oldLines, newLines := blockLines(oldStr), blockLines(newStr)
	replaced := fmt.Sprintf("line %d", start)
	if oldLines > 1 {
		replaced = fmt.Sprintf("lines %d-%d", start, start+oldLines-1)
	}
	return Ok(fmt.Sprintf("Edited %s: replaced %s (%s) with %s:\n%s", path, replaced,
		plural(oldLines, "line"), plural(newLines, "line"), snippet(content, start, newLines)))
}

// snippetContext is how many unchanged lines a str_replace result shows on each side of the edit.
const snippetContext = 3

// blockLines is the number of lines that text spans when it replaces part of a file: a final
// newline ends its last line rather than starting another, and empty text spans none.
func blockLines(text string) int {
	if text == "" {
		return 0
	}
	lines := strings.Count(text, "\n")
	if !strings.HasSuffix(text, "\n") {
		lines++
	}
	return lines
}

// snippet numbers the lines of content around the edited block that starts at line start and
// spans edited lines, as view numbers them.
func snippet(content string, start, edited int) string {
	lines := strings.Split(strings.TrimSuffix(content, "\n"), "\n")
	first := max(1, start-snippetContext)
	last := min(len(lines), start+max(edited, 1)-1+snippetContext)
	var b strings.Builder
	for i := first; i <= last; i++ {
		fmt.Fprintf(&b, "  %6d | %s\n", i, lines[i-1])
	}
	return b.String()
}

// matchLines lists the distinct lines where the occurrences of sub in content start, as "1, 4 and
// 9", at most maxListedMatches of them.
func matchLines(content, sub string) string {
	var lines []string
	line, counted := 1, 0
	for offset := 0; ; {
		at := strings.Index(content[offset:], sub)
		if at < 0 {
			break
		}
		line += strings.Count(content[counted:offset+at], "\n")
		counted = offset + at
		if n := fmt.Sprint(line); len(lines) == 0 || lines[len(lines)-1] != n {
			lines = append(lines, n)
		}
		offset += at + len(sub)
	}
	if len(lines) > maxListedMatches {
		more := len(lines) - maxListedMatches
		return strings.Join(lines[:maxListedMatches], ", ") + fmt.Sprintf(" and %d more lines", more)
	}
	if len(lines) == 1 {
		return lines[0]
	}
	return strings.Join(lines[:len(lines)-1], ", ") + " and " + lines[len(lines)-1]
}

// maxListedMatches is how many lines of a non-unique old_str a refusal names.
const maxListedMatches = 10

// plural writes n with unit, adding an s unless n is 1.
func plural(n int, unit string) string {
	if n == 1 {
		return "1 " + unit
	}
	return fmt.Sprintf("%d %ss", n, unit)
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

	before, statErr := os.Stat(path)
	existed := statErr == nil
	if existed && !before.Mode().IsRegular() {
		return Err(path + " is not a regular file")
	}
	var old []byte
	if existed && before.Size() <= maxCountedBytes {
		old, _ = os.ReadFile(path) // Only for its line count: a failed read counts none.
	}
	created, err := makeParents(filepath.Dir(path))
	if err != nil {
		return Err(err.Error())
	}
	if err := os.WriteFile(path, []byte(fileText), 0644); err != nil {
		return Err(err.Error())
	}
	if !existed {
		created = append(created, path)
	}
	inheritOwner(created)

	size := fmt.Sprintf("%s, %s", plural(textLines(fileText), "line"), plural(len(fileText), "byte"))
	if !existed {
		return Ok(fmt.Sprintf("Created %s: %s", path, size))
	}
	was := plural(int(before.Size()), "byte")
	if before.Size() <= maxCountedBytes {
		was = plural(textLines(string(old)), "line") + ", " + was
	}
	return Ok(fmt.Sprintf("Overwrote %s: %s (was %s)", path, size, was))
}

// maxCountedBytes is the largest old file whose lines an overwrite counts.
const maxCountedBytes = 16 << 20

// textLines is the number of lines of a file's text: a final newline ends the last line.
func textLines(text string) int {
	lines := strings.Count(text, "\n")
	if text != "" && !strings.HasSuffix(text, "\n") {
		lines++
	}
	return lines
}

// makeParents creates dir and its missing parents, and returns the directories it created,
// outermost first.
func makeParents(dir string) ([]string, error) {
	var missing []string
	for d := dir; ; d = filepath.Dir(d) {
		if _, err := os.Stat(d); err == nil || filepath.Dir(d) == d {
			break
		}
		missing = append([]string{d}, missing...)
	}
	return missing, os.MkdirAll(dir, 0755)
}

// inheritOwner gives each new path the owner of the directory that held the first one. The
// runtime runs as root, and the agent's own commands may run as the image's user, who must be
// able to change what the agent wrote. A failed chown leaves the path root's, as before.
func inheritOwner(paths []string) {
	if os.Geteuid() != 0 || len(paths) == 0 {
		return
	}
	info, err := os.Stat(filepath.Dir(paths[0]))
	if err != nil {
		return
	}
	stat, ok := info.Sys().(*syscall.Stat_t)
	if !ok {
		return
	}
	for _, path := range paths {
		_ = os.Chown(path, int(stat.Uid), int(stat.Gid))
	}
}

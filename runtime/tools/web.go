package tools

import (
	"context"
	"fmt"
	"io"
	"net/http"
	"slices"
	"strings"
	"sync"
	"time"

	htmltomarkdown "github.com/JohannesKaufmann/html-to-markdown/v2"
	"github.com/PuerkitoBio/goquery"
	ddgs "github.com/dreamyang-liu/ddgs-go"
)

const defaultUA = "SproutRuntime/1.0 (+https://github.com/dreamyang-liu/sprout)"

const (
	defaultWebTimeoutSeconds = 15
	maxWebTimeoutSeconds     = 60
	// How much of a document is read before conversion, only to bound memory.
	// The output budget applies to the converted text, often a tenth the size.
	maxFetchBodyBytes = 8 << 20
)

// ---- WebFetchTool ----

// WebFetchTool fetches one URL with GET. It returns the status, the content type
// if the server sent one, and the page as HTML, text or Markdown.
type WebFetchTool struct{}

func (w *WebFetchTool) Name() string { return "web_fetch" }

func (w *WebFetchTool) Description() string {
	return "Fetch a URL and return its content in the specified format"
}

func (w *WebFetchTool) Schema() map[string]any {
	props := map[string]any{
		"url":     map[string]any{"type": "string", "description": "URL to fetch"},
		"format":  map[string]any{"type": "string", "enum": []string{"html", "text", "markdown"}, "default": "markdown"},
		"headers": map[string]any{"type": "object", "description": "Additional HTTP headers"},
		"timeout": map[string]any{"type": "integer", "default": defaultWebTimeoutSeconds, "description": "Request timeout in seconds. Values are clamped to the runtime maximum."},
		// Page content is byte output, so it takes the shared output bound.
		"max_output_bytes": outputBytesSchema(),
	}
	return map[string]any{
		"type":       "object",
		"properties": props,
		"required":   []string{"url"},
	}
}

func (w *WebFetchTool) Execute(args map[string]any) Result {
	rawURL, _ := args["url"].(string)
	if rawURL == "" {
		return Err("url is required")
	}
	format := "markdown"
	if f, ok := args["format"].(string); ok && f != "" {
		format = f
	}
	maxLen := outputBytesArg(args)
	timeoutSeconds := defaultWebTimeoutSeconds
	if t, ok := args["timeout"].(float64); ok && int(t) > 0 {
		timeoutSeconds = clampInt(int(t), 1, maxWebTimeoutSeconds)
	}

	client := &http.Client{Timeout: time.Duration(timeoutSeconds) * time.Second}
	req, err := http.NewRequest("GET", rawURL, nil)
	if err != nil {
		return Err("invalid url: " + err.Error())
	}
	req.Header.Set("User-Agent", defaultUA)
	if hdrs, ok := args["headers"].(map[string]any); ok {
		for k, v := range hdrs {
			if s, ok := v.(string); ok {
				req.Header.Set(k, s)
			}
		}
	}

	resp, err := client.Do(req)
	if err != nil {
		return Err("fetch failed: " + err.Error())
	}
	defer resp.Body.Close()

	// Do not cut the body at maxLen. Conversion shrinks it, so an early cut drops
	// text that fits the budget, and boundText then sees no loss to report.
	body, err := io.ReadAll(io.LimitReader(resp.Body, maxFetchBodyBytes+1))
	if err != nil {
		return Err("read body: " + err.Error())
	}
	bodyTruncated := len(body) > maxFetchBodyBytes
	if bodyTruncated {
		body = body[:maxFetchBodyBytes]
	}
	html := string(body)

	var output string
	switch format {
	case "html":
		output = html
	case "text":
		output = stripHTML(html)
	case "markdown":
		output = toMarkdown(html)
	default:
		return Err("invalid format: " + format)
	}

	prefix := fmt.Sprintf("[status] %s\n", resp.Status)
	if contentType := resp.Header.Get("Content-Type"); contentType != "" {
		prefix += fmt.Sprintf("[content_type] %s\n", contentType)
	}
	if bodyTruncated {
		// The document itself is incomplete, whatever the budget. Report it apart
		// from the truncation notice of the output budget.
		prefix += fmt.Sprintf("[source_truncated] read the first %d bytes of the document\n",
			maxFetchBodyBytes)
	}
	return Ok(prefix + boundText(output, maxLen))
}

func stripHTML(s string) string {
	doc, err := goquery.NewDocumentFromReader(strings.NewReader(s))
	if err != nil {
		// fallback: naive tag stripping
		var b strings.Builder
		inTag := false
		for _, r := range s {
			if r == '<' {
				inTag = true
			} else if r == '>' {
				inTag = false
			} else if !inTag {
				b.WriteRune(r)
			}
		}
		return strings.Join(strings.Fields(b.String()), " ")
	}
	doc.Find("script, style, noscript, nav, footer, header").Remove()
	// Insert newlines before block elements so text doesn't run together
	doc.Find("p, h1, h2, h3, h4, h5, h6, li, br, div, tr, blockquote, pre").Each(func(_ int, s *goquery.Selection) {
		s.PrependHtml("\n")
	})
	text := doc.Find("body").Text()
	lines := strings.Split(text, "\n")
	var out []string
	for _, line := range lines {
		trimmed := strings.TrimSpace(line)
		if trimmed != "" {
			out = append(out, trimmed)
		}
	}
	return strings.Join(out, "\n")
}

func toMarkdown(html string) string {
	md, err := htmltomarkdown.ConvertString(html)
	if err != nil {
		return stripHTML(html)
	}
	return strings.TrimSpace(md)
}

func clampInt(v, min, max int) int {
	if v < min {
		return min
	}
	if v > max {
		return max
	}
	return v
}

// ---- WebSearchTool ----

// WebSearchTool searches the web through one or more engines of ddgs-go. It
// returns up to max_results results (20 at most): a title, a URL and a snippet
// if the engine gives one. A search that finds no result fails.
type WebSearchTool struct{}

func (w *WebSearchTool) Name() string { return "web_search" }

func (w *WebSearchTool) Description() string {
	return "Search the web and return results from multiple engines"
}

// webSearchBackends are the accepted backend values, mirroring the ddgs
// text-engine registry plus the "auto" aggregate mode.
var webSearchBackends = []string{
	"auto", "brave", "duckduckgo", "google", "grokipedia",
	"mojeek", "startpage", "wikipedia", "yahoo", "yandex",
}

func (w *WebSearchTool) Schema() map[string]any {
	return map[string]any{
		"type": "object",
		"properties": map[string]any{
			"query":       map[string]any{"type": "string", "description": "Search query"},
			"backend":     map[string]any{"type": "string", "enum": webSearchBackends, "default": "auto"},
			"max_results": map[string]any{"type": "integer", "default": 5, "description": "Maximum results. Values are clamped to the runtime maximum."},
		},
		"required": []string{"query"},
	}
}

type searchResult struct {
	Title string
	URL   string
	Body  string
}

// ddgsClient is shared so that the engine-failure cooldown of ddgs holds across
// calls. In auto mode, ddgs then skips a blocked engine for a few minutes.
var ddgsClient = sync.OnceValues(func() (*ddgs.DDGS, error) {
	return ddgs.New(&ddgs.Config{Timeout: 15 * time.Second})
})

// ddgsTextSearch runs one ddgs text search. It is a variable so that tests can
// replace the network access.
var ddgsTextSearch = func(query, backend string, maxResults int) ([]searchResult, error) {
	client, err := ddgsClient()
	if err != nil {
		return nil, err
	}
	ctx, cancel := context.WithTimeout(context.Background(), 30*time.Second)
	defer cancel()
	raw, err := client.Text(ctx, query, &ddgs.SearchOptions{
		Backend:    backend,
		MaxResults: maxResults,
	})
	if err != nil {
		return nil, err
	}
	results := make([]searchResult, 0, len(raw))
	for _, r := range raw {
		title, _ := r["title"].(string)
		href, _ := r["href"].(string)
		body, _ := r["body"].(string)
		results = append(results, searchResult{Title: title, URL: href, Body: body})
	}
	return results, nil
}

func (w *WebSearchTool) Execute(args map[string]any) Result {
	query, _ := args["query"].(string)
	if query == "" {
		return Err("query is required")
	}
	backend := "auto"
	if b, ok := args["backend"].(string); ok && b != "" {
		backend = b
	}
	if !slices.Contains(webSearchBackends, backend) {
		return Err("invalid backend: " + backend)
	}
	maxResults := 5
	if m, ok := args["max_results"].(float64); ok && int(m) > 0 {
		maxResults = clampInt(int(m), 1, 20)
	}

	results, err := ddgsTextSearch(query, backend, maxResults)
	if err != nil || len(results) == 0 {
		msg := "all search engines failed"
		if err != nil {
			msg += ": " + err.Error()
		}
		return Err(msg)
	}

	if len(results) > maxResults {
		results = results[:maxResults]
	}

	var b strings.Builder
	for i, r := range results {
		fmt.Fprintf(&b, "%d. %s\n   %s\n", i+1, r.Title, r.URL)
		if r.Body != "" {
			fmt.Fprintf(&b, "   %s\n", r.Body)
		}
		b.WriteString("\n")
	}
	return Ok(strings.TrimSpace(b.String()))
}

// Tests of the web_fetch tool against a local HTTP server. Its output is a [status] line, a
// [content_type] line if the server sent one, and then the page in the requested format.

package tools_test

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/dreamyang-liu/sprout/runtime/tools"
)

// serve starts a local HTTP server that answers each request with status, contentType and body,
// and returns its URL. An empty contentType sends no Content-Type header.
func serve(t *testing.T, status int, contentType, body string) string {
	t.Helper()
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, _ *http.Request) {
		// A nil value stops net/http from sniffing a Content-Type of its own.
		w.Header()["Content-Type"] = nil
		if contentType != "" {
			w.Header().Set("Content-Type", contentType)
		}
		w.WriteHeader(status)
		w.Write([]byte(body))
	}))
	t.Cleanup(server.Close)
	return server.URL
}

func TestFetchReturnsThePageInTheRequestedFormat(t *testing.T) {
	page := `<html><head><title>Page</title></head><body><nav>Menu</nav>` +
		`<h1>Hello</h1><p>Alpha <strong>Beta</strong></p><script>hidden()</script></body></html>`
	prefix := "[status] 200 OK\n[content_type] text/html; charset=utf-8\n"
	cases := []struct {
		name string
		args map[string]any
		want string
	}{
		{"html", map[string]any{"format": "html"}, prefix + page},
		{"text", map[string]any{"format": "text"}, prefix + "Hello\nAlpha Beta"},
		{"markdown", map[string]any{"format": "markdown"}, prefix + "Menu\n\n# Hello\n\nAlpha **Beta**"},
		{"absent, which means markdown", map[string]any{}, prefix + "Menu\n\n# Hello\n\nAlpha **Beta**"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			tc.args["url"] = serve(t, http.StatusOK, "text/html; charset=utf-8", page)

			got := (&tools.WebFetchTool{}).Execute(tc.args)

			assertResult(t, got, tools.Result{Success: true, Output: tc.want})
		})
	}
}

func TestFetchSendsTheHeadersOfTheCallerWithItsOwnUserAgent(t *testing.T) {
	received := make(chan http.Header, 1)
	server := httptest.NewServer(http.HandlerFunc(func(_ http.ResponseWriter, r *http.Request) {
		received <- r.Header.Clone()
	}))
	t.Cleanup(server.Close)

	(&tools.WebFetchTool{}).Execute(map[string]any{"url": server.URL, "headers": map[string]any{"X-Test": "sprout"}})

	headers := <-received
	got := [2]string{headers.Get("User-Agent"), headers.Get("X-Test")}
	want := [2]string{"SproutRuntime/1.0 (+https://github.com/dreamyang-liu/sprout)", "sprout"}
	if got != want {
		t.Errorf("User-Agent and X-Test = %q, want %q", got, want)
	}
}

func TestTheFirstLinesGiveTheStatusAndTheContentTypeOfTheResponse(t *testing.T) {
	cases := []struct {
		name        string
		status      int
		contentType string
		want        string
	}{
		{"200 with a content type", http.StatusOK, "text/plain", "[status] 200 OK\n[content_type] text/plain\nbody"},
		{"404 without a content type", http.StatusNotFound, "", "[status] 404 Not Found\nbody"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			url := serve(t, tc.status, tc.contentType, "body")

			got := (&tools.WebFetchTool{}).Execute(map[string]any{"url": url, "format": "html"})

			assertResult(t, got, tools.Result{Success: true, Output: tc.want})
		})
	}
}

func TestMaxOutputBytesBoundsTheConvertedTextAndNotTheMarkup(t *testing.T) {
	cases := []struct {
		name string
		page string
		want string
	}{
		{"markup over the budget whose text is under it",
			`<p data-padding="` + strings.Repeat("x", 2000) + `">short text</p>`,
			"short text"},
		{"text over the budget",
			"<p>" + strings.Repeat("h", 409) + strings.Repeat("m", 1976) + strings.Repeat("t", 615) + "</p>",
			strings.Repeat("h", 409) +
				"\n... [output truncated: 3000 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
				strings.Repeat("t", 615)},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			url := serve(t, http.StatusOK, "text/html", tc.page)

			got := (&tools.WebFetchTool{}).Execute(map[string]any{"url": url, "format": "text", "max_output_bytes": 1024.0})

			assertResult(t, got, tools.Result{Success: true, Output: "[status] 200 OK\n[content_type] text/html\n" + tc.want})
		})
	}
}

func TestFetchReadsAtMost8MiBOfADocumentAndSaysWhenItStopped(t *testing.T) {
	boundedBody := strings.Repeat("x", 409) +
		"\n... [output truncated: 8388608 bytes total, showing first 409 bytes and last 615 bytes] ...\n" +
		strings.Repeat("x", 615)
	cases := []struct {
		name     string
		bodySize int
		want     string
	}{
		{"a document of exactly 8 MiB", 8388608,
			"[status] 200 OK\n[content_type] text/plain\n" + boundedBody},
		{"a document one byte over 8 MiB", 8388609,
			"[status] 200 OK\n[content_type] text/plain\n" +
				"[source_truncated] read the first 8388608 bytes of the document\n" + boundedBody},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			url := serve(t, http.StatusOK, "text/plain", strings.Repeat("x", tc.bodySize))

			got := (&tools.WebFetchTool{}).Execute(map[string]any{"url": url, "format": "html", "max_output_bytes": 1024.0})

			assertResult(t, got, tools.Result{Success: true, Output: tc.want})
		})
	}
}

func TestAFetchGivesUpOnASilentServerAtTheTimeoutOfTheCaller(t *testing.T) {
	t.Parallel()
	unblock := make(chan struct{})
	server := httptest.NewServer(http.HandlerFunc(func(http.ResponseWriter, *http.Request) { <-unblock }))
	t.Cleanup(server.Close)
	t.Cleanup(func() { close(unblock) })
	start := time.Now()

	got := (&tools.WebFetchTool{}).Execute(map[string]any{"url": server.URL, "timeout": 1.0})

	elapsed := time.Since(start)
	assertResult(t, got, tools.Result{Success: false, Error: `fetch failed: Get "` + server.URL +
		`": context deadline exceeded (Client.Timeout exceeded while awaiting headers)`})
	// 5 s is well over the timeout of 1 s, and well under the default of 15 s.
	if elapsed >= 5*time.Second {
		t.Errorf("the fetch gave up after %v, want about 1 s", elapsed)
	}
}

func TestAnUnknownFormatIsRefused(t *testing.T) {
	url := serve(t, http.StatusOK, "text/html", "<p>page</p>")

	got := (&tools.WebFetchTool{}).Execute(map[string]any{"url": url, "format": "pdf"})

	assertResult(t, got, tools.Result{Success: false, Error: "invalid format: pdf"})
}

func TestAFetchThatCannotBeMadeIsRefusedWithTheReason(t *testing.T) {
	closed := httptest.NewServer(http.NotFoundHandler())
	closed.Close()
	address := strings.TrimPrefix(closed.URL, "http://")
	cases := []struct {
		name string
		url  string
		want string
	}{
		{"a URL that does not parse", "://no-scheme",
			`invalid url: parse "://no-scheme": missing protocol scheme`},
		{"a server that does not answer", closed.URL,
			`fetch failed: Get "` + closed.URL + `": dial tcp ` + address + `: connect: connection refused`},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := (&tools.WebFetchTool{}).Execute(map[string]any{"url": tc.url})

			assertResult(t, got, tools.Result{Success: false, Error: tc.want})
		})
	}
}

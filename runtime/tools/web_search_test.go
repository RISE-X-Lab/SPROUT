// Tests of the web_search tool. They are in package tools because they replace ddgsTextSearch,
// the call that reaches the search engines over the network, with a spy that answers for them.

package tools

import (
	"errors"
	"testing"
)

// engineCall is what web_search asked the search engines for.
type engineCall struct {
	query      string
	backend    string
	maxResults int
}

// answerSearches makes every search for the rest of the test return results and err, and
// returns the calls that it receives.
func answerSearches(t *testing.T, results []searchResult, err error) *[]engineCall {
	t.Helper()
	calls := &[]engineCall{}
	original := ddgsTextSearch
	ddgsTextSearch = func(query, backend string, maxResults int) ([]searchResult, error) {
		*calls = append(*calls, engineCall{query: query, backend: backend, maxResults: maxResults})
		return results, err
	}
	t.Cleanup(func() { ddgsTextSearch = original })
	return calls
}

func assertSearchResult(t *testing.T, got, want Result) {
	t.Helper()
	if got != want {
		t.Errorf("result = Success %v, Output %q, Error %q\nwant     Success %v, Output %q, Error %q",
			got.Success, got.Output, got.Error, want.Success, want.Output, want.Error)
	}
}

func TestResultsAreNumberedWithTheirURLAndSnippet(t *testing.T) {
	answerSearches(t, []searchResult{
		{Title: "No snippet", URL: "https://example.com/bare"},
		{Title: "Sprout", URL: "https://example.com/sprout", Body: "Checkpointed rollouts."},
	}, nil)

	got := (&WebSearchTool{}).Execute(map[string]any{"query": "sprout"})

	assertSearchResult(t, got, Result{Success: true, Output: "1. No snippet\n   https://example.com/bare\n\n" +
		"2. Sprout\n   https://example.com/sprout\n   Checkpointed rollouts."})
}

func TestTheEnginesGetTheQueryAndTheBackendWhichIsAutoByDefault(t *testing.T) {
	cases := []struct {
		name string
		args map[string]any
		want engineCall
	}{
		{"no backend", map[string]any{"query": "sprout"}, engineCall{"sprout", "auto", 5}},
		{"a named backend", map[string]any{"query": "sprout", "backend": "wikipedia"}, engineCall{"sprout", "wikipedia", 5}},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls := answerSearches(t, []searchResult{{Title: "T", URL: "U"}}, nil)

			(&WebSearchTool{}).Execute(tc.args)

			if len(*calls) != 1 || (*calls)[0] != tc.want {
				t.Errorf("engine calls = %+v, want one: %+v", *calls, tc.want)
			}
		})
	}
}

func TestMaxResultsIsClampedTo1Through20AndIs5ByDefault(t *testing.T) {
	// The values on each side of 0 (the default applies to 0) and of 20.
	cases := []struct {
		name string
		args map[string]any
		want int
	}{
		{"absent", map[string]any{"query": "sprout"}, 5},
		{"0", map[string]any{"query": "sprout", "max_results": 0.0}, 5},
		{"1", map[string]any{"query": "sprout", "max_results": 1.0}, 1},
		{"20", map[string]any{"query": "sprout", "max_results": 20.0}, 20},
		{"21", map[string]any{"query": "sprout", "max_results": 21.0}, 20},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			calls := answerSearches(t, []searchResult{{Title: "T", URL: "U"}}, nil)

			(&WebSearchTool{}).Execute(tc.args)

			if len(*calls) != 1 || (*calls)[0].maxResults != tc.want {
				t.Errorf("engine calls = %+v, want one for %d results", *calls, tc.want)
			}
		})
	}
}

func TestOnlyMaxResultsOfTheResultsOfTheEnginesAreShown(t *testing.T) {
	answerSearches(t, []searchResult{{Title: "A", URL: "a"}, {Title: "B", URL: "b"}, {Title: "C", URL: "c"}}, nil)

	got := (&WebSearchTool{}).Execute(map[string]any{"query": "sprout", "max_results": 2.0})

	assertSearchResult(t, got, Result{Success: true, Output: "1. A\n   a\n\n2. B\n   b"})
}

func TestAnUnknownBackendIsRefusedBeforeAnySearch(t *testing.T) {
	calls := answerSearches(t, []searchResult{{Title: "T", URL: "U"}}, nil)

	got := (&WebSearchTool{}).Execute(map[string]any{"query": "sprout", "backend": "bing"})

	assertSearchResult(t, got, Result{Success: false, Error: "invalid backend: bing"})
	if len(*calls) != 0 {
		t.Errorf("engine calls = %+v, want none", *calls)
	}
}

func TestASearchThatFindsNothingFails(t *testing.T) {
	cases := []struct {
		name    string
		results []searchResult
		err     error
		want    string
	}{
		{"the engines fail", nil, errors.New("boom"), "all search engines failed: boom"},
		{"the engines find no result", []searchResult{}, nil, "all search engines failed"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			answerSearches(t, tc.results, tc.err)

			got := (&WebSearchTool{}).Execute(map[string]any{"query": "sprout"})

			assertSearchResult(t, got, Result{Success: false, Error: tc.want})
		})
	}
}

package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/distributed/orchestrator"
	"github.com/ollama/ollama/distributed/state"
)

type stubSource struct{ snap orchestrator.Snapshot }

func (s *stubSource) Snapshot() orchestrator.Snapshot { return s.snap }

func sampleSnap() orchestrator.Snapshot {
	return orchestrator.Snapshot{
		Nodes: []orchestrator.NodeSnapshot{
			{ID: "n1", Hostname: "h1", Collective: "c1", State: state.Available, CurrentLPU: 2.5},
			{ID: "n2", Hostname: "h2", Collective: "c1", State: state.Syncing, CurrentLPU: 1.0},
		},
		Collectives: []orchestrator.CollectiveSnapshot{
			{Name: "c1", MaxNodes: 4, NodeCount: 2, AvailCount: 1, AvgLPU: 1.75, NodeIDs: []string{"n1", "n2"}},
		},
		Jobs:            []orchestrator.JobSnapshot{{ID: "j1", Collective: "c1", Segments: 3, Completed: 1, Nodes: []string{"n1"}}},
		StarvationIndex: 0.8,
		MaxNodes:        4,
		CapturedAt:      time.Now(),
	}
}

func TestNewValidation(t *testing.T) {
	if _, err := New(Options{}); err == nil {
		t.Fatal("missing source should fail")
	}
	d, err := New(Options{Source: &stubSource{}})
	if err != nil {
		t.Fatal(err)
	}
	if d.opts.Title == "" {
		t.Fatal("default title not applied")
	}
}

func TestSnapshotEndpoint(t *testing.T) {
	d, _ := New(Options{Source: &stubSource{snap: sampleSnap()}})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/api/distributed/snapshot")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	if ct := resp.Header.Get("Content-Type"); !strings.Contains(ct, "application/json") {
		t.Fatalf("content-type=%q", ct)
	}
	if cc := resp.Header.Get("Cache-Control"); !strings.Contains(cc, "no-store") {
		t.Fatalf("cache-control=%q", cc)
	}
	var got orchestrator.Snapshot
	if err := json.NewDecoder(resp.Body).Decode(&got); err != nil {
		t.Fatal(err)
	}
	if len(got.Nodes) != 2 || got.Nodes[0].ID != "n1" {
		t.Fatalf("nodes=%+v", got.Nodes)
	}
	if got.StarvationIndex != 0.8 {
		t.Fatalf("starvation=%v", got.StarvationIndex)
	}
}

func TestIndexHTML(t *testing.T) {
	d, _ := New(Options{Source: &stubSource{snap: sampleSnap()}, Title: "My <Dashboard>"})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	buf := new(bytes.Buffer)
	_, _ = buf.ReadFrom(resp.Body)
	body := buf.String()
	// Title is escaped.
	if !strings.Contains(body, "My &lt;Dashboard&gt;") {
		t.Fatalf("title not escaped: %s", body[:200])
	}
	// Page includes the polling endpoint.
	if !strings.Contains(body, "/api/distributed/snapshot") {
		t.Fatal("page missing snapshot URL")
	}
	// Should not leave any raw %s / %!s placeholders.
	if strings.Contains(body, "%!") {
		t.Fatalf("template left fmt error: %s", body)
	}
}

func TestIndexNotFound(t *testing.T) {
	d, _ := New(Options{Source: &stubSource{snap: sampleSnap()}})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()
	resp, err := http.Get(srv.URL + "/does-not-exist")
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status=%d", resp.StatusCode)
	}
}

func TestPersonaNotConfigured(t *testing.T) {
	d, _ := New(Options{Source: &stubSource{snap: sampleSnap()}})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()
	body := strings.NewReader(`{"collective":"c1","persona":"p"}`)
	resp, err := http.Post(srv.URL+"/api/distributed/persona", "application/json", body)
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotImplemented {
		t.Fatalf("status=%d", resp.StatusCode)
	}
}

func TestPersonaBadJSON(t *testing.T) {
	d, _ := New(Options{
		Source:         &stubSource{snap: sampleSnap()},
		PersonaApplier: PersonaApplierFunc(func(ctx context.Context, id, p string) error { return nil }),
	})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/api/distributed/persona", "application/json", strings.NewReader("{"))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status=%d", resp.StatusCode)
	}
}

func TestPersonaMissingCollective(t *testing.T) {
	d, _ := New(Options{
		Source:         &stubSource{snap: sampleSnap()},
		PersonaApplier: PersonaApplierFunc(func(ctx context.Context, id, p string) error { return nil }),
	})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/api/distributed/persona", "application/json", strings.NewReader(`{"persona":"p"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusBadRequest {
		t.Fatalf("status=%d", resp.StatusCode)
	}
}

func TestPersonaUnknownCollective(t *testing.T) {
	d, _ := New(Options{
		Source:         &stubSource{snap: sampleSnap()},
		PersonaApplier: PersonaApplierFunc(func(ctx context.Context, id, p string) error { return nil }),
	})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()
	resp, err := http.Post(srv.URL+"/api/distributed/persona", "application/json", strings.NewReader(`{"collective":"ghost"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusNotFound {
		t.Fatalf("status=%d", resp.StatusCode)
	}
}

func TestPersonaHappyPathAndPartialFailure(t *testing.T) {
	var calls atomic.Int64
	applier := PersonaApplierFunc(func(ctx context.Context, nodeID, persona string) error {
		calls.Add(1)
		if nodeID == "n2" {
			return errors.New("n2 broken")
		}
		return nil
	})
	d, _ := New(Options{Source: &stubSource{snap: sampleSnap()}, PersonaApplier: applier})
	srv := httptest.NewServer(d.Handler())
	defer srv.Close()

	resp, err := http.Post(srv.URL+"/api/distributed/persona", "application/json", strings.NewReader(`{"collective":"c1","persona":"coder"}`))
	if err != nil {
		t.Fatal(err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		t.Fatalf("status=%d", resp.StatusCode)
	}
	var out PersonaResponse
	if err := json.NewDecoder(resp.Body).Decode(&out); err != nil {
		t.Fatal(err)
	}
	if out.Succeeded != 1 || out.Failed != 1 {
		t.Fatalf("%+v", out)
	}
	if calls.Load() != 2 {
		t.Fatalf("calls=%d want 2", calls.Load())
	}
	// Result order matches NodeIDs order; partial failure is carried
	// through and does not roll back the peer.
	var foundOK, foundErr bool
	for _, r := range out.Results {
		if r.OK && r.NodeID == "n1" {
			foundOK = true
		}
		if !r.OK && r.NodeID == "n2" && r.Error != "" {
			foundErr = true
		}
	}
	if !foundOK || !foundErr {
		t.Fatalf("results=%+v", out.Results)
	}
}

// Compile-time check that *orchestrator.Orchestrator satisfies Source.
func TestOrchestratorIsASource(t *testing.T) {
	// The only way this compiles is if the interface is satisfied.
	var _ Source = (*orchestrator.Orchestrator)(nil)
}

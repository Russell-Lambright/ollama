package modelsync

import (
	"context"
	"errors"
	"reflect"
	"sync/atomic"
	"testing"

	"github.com/ollama/ollama/distributed/secondary"
)

func TestEntryValidate(t *testing.T) {
	if err := (Entry{}).Validate(); err == nil {
		t.Fatal("empty entry should fail")
	}
	if err := (Entry{Name: "m"}).Validate(); err == nil {
		t.Fatal("missing digest should fail")
	}
	if err := (Entry{Name: "m", Digest: "d"}).Validate(); err != nil {
		t.Fatal(err)
	}
}

func TestManifestNamesSorted(t *testing.T) {
	m := NewManifest(
		Entry{Name: "z", Digest: "1"},
		Entry{Name: "a", Digest: "2"},
		Entry{Name: "m", Digest: "3"},
	)
	got := m.Names()
	want := []string{"a", "m", "z"}
	if !reflect.DeepEqual(got, want) {
		t.Fatalf("got %v want %v", got, want)
	}
}

func TestNewManifestDuplicateLastWins(t *testing.T) {
	m := NewManifest(
		Entry{Name: "x", Digest: "1"},
		Entry{Name: "x", Digest: "2"},
	)
	if m.Entries["x"].Digest != "2" {
		t.Fatalf("last-wins failed: %+v", m.Entries["x"])
	}
}

func TestDiffPullWhenMissing(t *testing.T) {
	want := NewManifest(Entry{Name: "m", Digest: "d"})
	have := NewManifest()
	plan := Diff(want, have)
	if len(plan.Items) != 1 || plan.Items[0].Op != OpPull {
		t.Fatalf("bad plan: %+v", plan)
	}
	if plan.IsEmpty() {
		t.Fatal("non-empty plan reports empty")
	}
}

func TestDiffRedownloadOnDigestDrift(t *testing.T) {
	want := NewManifest(Entry{Name: "m", Digest: "new"})
	have := NewManifest(Entry{Name: "m", Digest: "old"})
	plan := Diff(want, have)
	if len(plan.Items) != 1 || plan.Items[0].Op != OpRedownload {
		t.Fatalf("bad plan: %+v", plan)
	}
	if plan.Items[0].Have.Digest != "old" || plan.Items[0].Want.Digest != "new" {
		t.Fatalf("fields wrong: %+v", plan.Items[0])
	}
}

func TestDiffNoopWhenMatching(t *testing.T) {
	want := NewManifest(Entry{Name: "m", Digest: "d"})
	have := NewManifest(Entry{Name: "m", Digest: "d"})
	plan := Diff(want, have)
	if len(plan.Items) != 1 || plan.Items[0].Op != OpNoop {
		t.Fatalf("bad plan: %+v", plan)
	}
	if !plan.IsEmpty() {
		t.Fatal("noop-only plan should be empty")
	}
}

func TestDiffIgnoresExtraObserved(t *testing.T) {
	// Node has more than expected → ignored, not deleted.
	want := NewManifest(Entry{Name: "a", Digest: "1"})
	have := NewManifest(
		Entry{Name: "a", Digest: "1"},
		Entry{Name: "b", Digest: "2"},
		Entry{Name: "c", Digest: "3"},
	)
	plan := Diff(want, have)
	if len(plan.Items) != 1 {
		t.Fatalf("expected 1 item (for 'a'); got %d", len(plan.Items))
	}
}

func TestDiffDeterministicOrder(t *testing.T) {
	want := NewManifest(
		Entry{Name: "z", Digest: "1"},
		Entry{Name: "a", Digest: "1"},
		Entry{Name: "m", Digest: "1"},
	)
	have := NewManifest()
	p1 := Diff(want, have)
	p2 := Diff(want, have)
	if !reflect.DeepEqual(p1.Items, p2.Items) {
		t.Fatal("Diff is non-deterministic")
	}
	// First item should be 'a' (alpha order).
	if p1.Items[0].Want.Name != "a" {
		t.Fatalf("not sorted: %+v", p1.Items)
	}
}

func TestNewValidation(t *testing.T) {
	if _, err := New(Options{}); err == nil {
		t.Fatal("missing provider should fail")
	}
	if _, err := New(Options{Provider: ManifestProviderFunc{}}); err == nil {
		t.Fatal("missing puller should fail")
	}
	s, err := New(Options{
		Provider: ManifestProviderFunc{},
		Puller:   ModelPullerFunc(func(ctx context.Context, e Entry) error { return nil }),
	})
	if err != nil {
		t.Fatal(err)
	}
	if s.opts.MaxAttempts != 1 {
		t.Fatalf("MaxAttempts default: %d", s.opts.MaxAttempts)
	}
}

func TestSyncAlreadyInSync(t *testing.T) {
	m := NewManifest(Entry{Name: "m", Digest: "d"})
	var pulls atomic.Int64
	s, _ := New(Options{
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return m, nil },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return m, nil },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error {
			pulls.Add(1)
			return nil
		}),
	})
	if err := s.Sync(context.Background()); err != nil {
		t.Fatal(err)
	}
	if pulls.Load() != 0 {
		t.Fatalf("unexpected pulls: %d", pulls.Load())
	}
}

func TestSyncPullsMissingAndDriftedOnly(t *testing.T) {
	want := NewManifest(
		Entry{Name: "a", Digest: "1"},
		Entry{Name: "b", Digest: "1"},
		Entry{Name: "c", Digest: "1"},
	)
	have := NewManifest(
		Entry{Name: "a", Digest: "1"}, // noop
		Entry{Name: "b", Digest: "0"}, // redownload
		// c missing → pull
	)
	var pulled []string
	s, _ := New(Options{
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return want, nil },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return have, nil },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error {
			pulled = append(pulled, e.Name)
			return nil
		}),
	})
	if err := s.Sync(context.Background()); err != nil {
		t.Fatal(err)
	}
	if !reflect.DeepEqual(pulled, []string{"b", "c"}) {
		t.Fatalf("pulled=%v want [b c]", pulled)
	}
}

func TestSyncExpectedError(t *testing.T) {
	sentinel := errors.New("catalog down")
	s, _ := New(Options{
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return Manifest{}, sentinel },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return NewManifest(), nil },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error { return nil }),
	})
	if err := s.Sync(context.Background()); !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want sentinel", err)
	}
}

func TestSyncObservedError(t *testing.T) {
	sentinel := errors.New("disk read failed")
	s, _ := New(Options{
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return NewManifest(), nil },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return Manifest{}, sentinel },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error { return nil }),
	})
	if err := s.Sync(context.Background()); !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want sentinel", err)
	}
}

func TestSyncRetriesThenSucceeds(t *testing.T) {
	want := NewManifest(Entry{Name: "m", Digest: "d"})
	have := NewManifest()
	var attempts atomic.Int64
	s, _ := New(Options{
		MaxAttempts: 3,
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return want, nil },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return have, nil },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error {
			n := attempts.Add(1)
			if n < 3 {
				return errors.New("transient")
			}
			return nil
		}),
	})
	if err := s.Sync(context.Background()); err != nil {
		t.Fatal(err)
	}
	if attempts.Load() != 3 {
		t.Fatalf("attempts=%d want 3", attempts.Load())
	}
}

func TestSyncRetriesExhausted(t *testing.T) {
	want := NewManifest(Entry{Name: "m", Digest: "d"})
	have := NewManifest()
	sentinel := errors.New("always bad")
	s, _ := New(Options{
		MaxAttempts: 2,
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return want, nil },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return have, nil },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error { return sentinel }),
	})
	err := s.Sync(context.Background())
	if err == nil {
		t.Fatal("expected error")
	}
	if !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want sentinel", err)
	}
}

func TestSyncCtxCancellation(t *testing.T) {
	want := NewManifest(
		Entry{Name: "a", Digest: "1"},
		Entry{Name: "b", Digest: "1"},
	)
	have := NewManifest()
	ctx, cancelFn := context.WithCancel(context.Background())
	pulls := atomic.Int64{}
	s, _ := New(Options{
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return want, nil },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return have, nil },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error {
			pulls.Add(1)
			cancelFn() // cancel during first pull
			return nil
		}),
	})
	err := s.Sync(ctx)
	if err == nil {
		t.Fatal("expected ctx cancellation")
	}
	if pulls.Load() != 1 {
		t.Fatalf("should have stopped after first pull; pulls=%d", pulls.Load())
	}
}

// Confirm Syncer satisfies secondary.ModelSyncer.
func TestSyncerImplementsModelSyncer(t *testing.T) {
	s, _ := New(Options{
		Provider: ManifestProviderFunc{
			ExpectedFn: func(ctx context.Context) (Manifest, error) { return NewManifest(), nil },
			ObservedFn: func(ctx context.Context) (Manifest, error) { return NewManifest(), nil },
		},
		Puller: ModelPullerFunc(func(ctx context.Context, e Entry) error { return nil }),
	})
	var _ secondary.ModelSyncer = s
}

// --- fanout ---

func TestFanoutAllSucceed(t *testing.T) {
	ids := []string{"a", "b", "c"}
	var calls atomic.Int64
	got := Fanout(context.Background(), ids, func(ctx context.Context, id string) error {
		calls.Add(1)
		return nil
	})
	if int(calls.Load()) != len(ids) {
		t.Fatalf("calls=%d want %d", calls.Load(), len(ids))
	}
	if len(got) != len(ids) {
		t.Fatalf("results=%d want %d", len(got), len(ids))
	}
	for i, r := range got {
		if r.NodeID != ids[i] || r.Err != nil {
			t.Fatalf("index %d: %+v", i, r)
		}
	}
}

func TestFanoutIndependentFailures(t *testing.T) {
	ids := []string{"a", "b", "c"}
	got := Fanout(context.Background(), ids, func(ctx context.Context, id string) error {
		if id == "b" {
			return errors.New("broken")
		}
		return nil
	})
	if len(got) != 3 {
		t.Fatalf("results=%d", len(got))
	}
	if got[0].Err != nil || got[2].Err != nil {
		t.Fatalf("unexpected neighbor failures: %+v", got)
	}
	if got[1].Err == nil {
		t.Fatalf("expected b to fail")
	}
}

func TestFanoutEmptyNoop(t *testing.T) {
	got := Fanout(context.Background(), nil, func(ctx context.Context, id string) error {
		t.Fatal("should not be called")
		return nil
	})
	if len(got) != 0 {
		t.Fatalf("empty fanout: %+v", got)
	}
}

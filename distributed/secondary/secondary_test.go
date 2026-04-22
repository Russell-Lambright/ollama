package secondary

import (
	"context"
	"errors"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/distributed/cancel"
	"github.com/ollama/ollama/distributed/node"
	"github.com/ollama/ollama/distributed/state"
	"github.com/ollama/ollama/distributed/transport"
)

func newIdentity(t *testing.T) node.Identity {
	t.Helper()
	id, err := node.NewID()
	if err != nil {
		t.Fatal(err)
	}
	return node.Identity{ID: id, Hostname: "h", Collective: "c", AdvertisedLPU: 1}
}

// simpleExec emits one tokens event then done.
func simpleExec() ExecutorFunc {
	return func(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error) {
		ch := make(chan transport.SegmentEvent, 2)
		ch <- transport.SegmentEvent{Kind: transport.SegmentTokens, Tokens: "hello"}
		ch <- transport.SegmentEvent{Kind: transport.SegmentDone}
		close(ch)
		return ch, nil
	}
}

func TestNewValidates(t *testing.T) {
	lb := transport.NewLoopback(0)
	defer lb.Close()
	if _, err := New(Options{}); err == nil {
		t.Fatal("expected error for missing Primary")
	}
	if _, err := New(Options{Primary: lb}); err == nil {
		t.Fatal("expected error for missing Executor")
	}
	if _, err := New(Options{Primary: lb, Executor: simpleExec()}); err == nil {
		t.Fatal("expected error for invalid identity")
	}
}

func TestRunHappyPath(t *testing.T) {
	lb := transport.NewLoopback(10 * time.Millisecond)
	defer lb.Close()

	id := newIdentity(t)
	var syncCalled atomic.Bool
	s, err := New(Options{
		Identity:          id,
		Primary:           lb,
		Executor:          simpleExec(),
		Syncer:            ModelSyncerFunc(func(ctx context.Context) error { syncCalled.Store(true); return nil }),
		HeartbeatOverride: 10 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := lb.RegisterSecondary(id.ID, s); err != nil {
		t.Fatal(err)
	}

	ctx, cancelFn := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()

	// Wait until the node is Available.
	waitForState(t, s, state.Available, time.Second)
	if !syncCalled.Load() {
		t.Fatalf("Syncer not invoked")
	}

	// Assign a segment and confirm a tokens+done pair shows up.
	if err := lb.Assign(id.ID, transport.AssignSegment{JobID: "j1", SegmentID: "s1", Prompt: "p", Model: "m"}); err != nil {
		t.Fatal(err)
	}
	// Events should arrive and node returns to Available.
	if !waitFor(func() bool {
		for _, e := range lb.Events() {
			if e.Kind == transport.SegmentDone && e.SegmentID == "s1" {
				return true
			}
		}
		return false
	}, time.Second) {
		t.Fatalf("SegmentDone not observed; events=%+v", lb.Events())
	}
	waitForState(t, s, state.Available, time.Second)

	// Heartbeats should be recorded.
	time.Sleep(30 * time.Millisecond)
	if len(lb.Heartbeats()) == 0 {
		t.Fatalf("expected at least one heartbeat")
	}

	// State updates should include Syncing→Available and Available→Processing→Available.
	updates := lb.StateUpdates()
	if !containsTransition(updates, state.Syncing, state.Available) {
		t.Fatalf("missing Syncing→Available in %+v", updates)
	}
	if !containsTransition(updates, state.Available, state.Processing) {
		t.Fatalf("missing Available→Processing in %+v", updates)
	}

	// Clean shutdown.
	cancelFn()
	if err := <-done; err != nil && !errors.Is(err, context.Canceled) {
		t.Fatalf("Run returned %v", err)
	}
}

func TestSyncFailureGoesToFailed(t *testing.T) {
	lb := transport.NewLoopback(0)
	defer lb.Close()
	id := newIdentity(t)
	s, err := New(Options{
		Identity: id,
		Primary:  lb,
		Executor: simpleExec(),
		Syncer:   ModelSyncerFunc(func(ctx context.Context) error { return errors.New("boom") }),
	})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Run(context.Background()); err == nil {
		t.Fatal("expected sync failure")
	}
	if s.State() != state.Failed {
		t.Fatalf("state after sync failure: got %s want Failed", s.State())
	}
}

func TestCancelAbortsSegment(t *testing.T) {
	lb := transport.NewLoopback(50 * time.Millisecond)
	defer lb.Close()
	id := newIdentity(t)

	blocking := ExecutorFunc(func(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error) {
		ch := make(chan transport.SegmentEvent, 1)
		go func() {
			defer close(ch)
			select {
			case <-ctx.Done():
				ch <- transport.SegmentEvent{Kind: transport.SegmentError, Detail: "cancelled"}
			case <-time.After(5 * time.Second):
				ch <- transport.SegmentEvent{Kind: transport.SegmentDone}
			}
		}()
		return ch, nil
	})

	s, err := New(Options{Identity: id, Primary: lb, Executor: blocking})
	if err != nil {
		t.Fatal(err)
	}
	if err := lb.RegisterSecondary(id.ID, s); err != nil {
		t.Fatal(err)
	}

	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()
	waitForState(t, s, state.Available, time.Second)

	if err := lb.Assign(id.ID, transport.AssignSegment{JobID: "j", SegmentID: "s"}); err != nil {
		t.Fatal(err)
	}
	waitForState(t, s, state.Processing, time.Second)

	if err := lb.Cancel(context.Background(), id.ID, transport.CancelRequest{
		JobID: "j", SegmentID: "s", Reason: cancel.ReasonCallerCancelled,
	}); err != nil {
		t.Fatal(err)
	}
	// Back to Available after cancellation.
	waitForState(t, s, state.Available, 2*time.Second)

	// Unknown segment cancel is a no-op (idempotent).
	if err := s.Cancel(context.Background(), transport.CancelRequest{SegmentID: "nope"}); err != nil {
		t.Fatalf("idempotent cancel: %v", err)
	}
}

func TestApplyPersona(t *testing.T) {
	lb := transport.NewLoopback(0)
	defer lb.Close()
	id := newIdentity(t)
	var applied atomic.Value
	s, err := New(Options{
		Identity: id,
		Primary:  lb,
		Executor: simpleExec(),
		PersonaApplier: PersonaApplierFunc(func(ctx context.Context, p string) error {
			applied.Store(p)
			return nil
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()
	waitForState(t, s, state.Available, time.Second)

	if err := s.ApplyPersona("code-reviewer"); err != nil {
		t.Fatal(err)
	}
	// Observe Training then back to Available.
	if !waitFor(func() bool {
		for _, u := range lb.StateUpdates() {
			if u.To == state.Training {
				return true
			}
		}
		return false
	}, time.Second) {
		t.Fatal("never entered Training")
	}
	waitForState(t, s, state.Available, time.Second)
	if applied.Load() != "code-reviewer" {
		t.Fatalf("persona not applied: %v", applied.Load())
	}
}

func TestPersonaApplierFailureGoesToFailed(t *testing.T) {
	lb := transport.NewLoopback(0)
	defer lb.Close()
	id := newIdentity(t)
	s, err := New(Options{
		Identity: id,
		Primary:  lb,
		Executor: simpleExec(),
		PersonaApplier: PersonaApplierFunc(func(ctx context.Context, p string) error {
			return errors.New("boom")
		}),
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()
	done := make(chan error, 1)
	go func() { done <- s.Run(ctx) }()
	waitForState(t, s, state.Available, time.Second)
	_ = s.ApplyPersona("x")
	err = <-done
	if err == nil {
		t.Fatal("expected Run to return error")
	}
	if s.State() != state.Failed {
		t.Fatalf("state: %s want Failed", s.State())
	}
}

func TestRegisterRejection(t *testing.T) {
	lb := transport.NewLoopback(0)
	defer lb.Close()
	id := newIdentity(t)
	lb.Close() // loopback reports ErrClosed → register returns error
	s, err := New(Options{Identity: id, Primary: lb, Executor: simpleExec()})
	if err != nil {
		t.Fatal(err)
	}
	if err := s.Run(context.Background()); err == nil {
		t.Fatal("expected register error")
	}
	if s.State() != state.Failed {
		t.Fatalf("state: %s want Failed", s.State())
	}
}

func TestDoubleRunRejected(t *testing.T) {
	lb := transport.NewLoopback(20 * time.Millisecond)
	defer lb.Close()
	id := newIdentity(t)
	s, err := New(Options{Identity: id, Primary: lb, Executor: simpleExec(), HeartbeatOverride: 20 * time.Millisecond})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()
	go func() { _ = s.Run(ctx) }()
	waitForState(t, s, state.Available, time.Second)
	if err := s.Run(ctx); err == nil {
		t.Fatal("expected double-Run to be rejected")
	}
}

// --- helpers ---

func waitForState(t *testing.T, s *Secondary, want state.State, timeout time.Duration) {
	t.Helper()
	if !waitFor(func() bool { return s.State() == want }, timeout) {
		t.Fatalf("state never reached %s (last=%s)", want, s.State())
	}
}

func waitFor(pred func() bool, timeout time.Duration) bool {
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if pred() {
			return true
		}
		time.Sleep(5 * time.Millisecond)
	}
	return pred()
}

func containsTransition(us []transport.StateUpdate, from, to state.State) bool {
	for _, u := range us {
		if u.From == from && u.To == to {
			return true
		}
	}
	return false
}

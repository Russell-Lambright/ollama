package transport

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/distributed/cancel"
	"github.com/ollama/ollama/distributed/node"
	"github.com/ollama/ollama/distributed/state"
)

func mustID(t *testing.T) node.ID {
	t.Helper()
	id, err := node.NewID()
	if err != nil {
		t.Fatal(err)
	}
	return id
}

func newIdentity(t *testing.T, coll string) node.Identity {
	return node.Identity{
		ID:            mustID(t),
		Hostname:      "host-" + coll,
		Collective:    coll,
		AdvertisedLPU: 1.0,
	}
}

func TestLoopbackRegisterAndHeartbeat(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()

	id := newIdentity(t, "default")
	resp, err := lb.Register(context.Background(), RegisterRequest{Identity: id})
	if err != nil {
		t.Fatalf("Register: %v", err)
	}
	if !resp.Accepted {
		t.Fatalf("Register not accepted: %s", resp.Reason)
	}
	if resp.HeartbeatInterval != DefaultHeartbeatInterval {
		t.Fatalf("heartbeat interval: got %v want %v", resp.HeartbeatInterval, DefaultHeartbeatInterval)
	}

	if err := lb.Heartbeat(context.Background(), Heartbeat{NodeID: id.ID, State: state.Available, CurrentLPU: 1.0}); err != nil {
		t.Fatalf("Heartbeat: %v", err)
	}
	if got := lb.Heartbeats(); len(got) != 1 {
		t.Fatalf("heartbeats: got %d want 1", len(got))
	}

	// Heartbeat for unknown node is rejected.
	other := mustID(t)
	err = lb.Heartbeat(context.Background(), Heartbeat{NodeID: other, State: state.Available})
	if !errors.Is(err, ErrUnknownNode) {
		t.Fatalf("unknown heartbeat: err=%v", err)
	}
}

func TestLoopbackRegisterRejectsBadIdentity(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()
	resp, err := lb.Register(context.Background(), RegisterRequest{Identity: node.Identity{}})
	if err == nil {
		t.Fatalf("expected error for invalid identity")
	}
	if resp.Accepted {
		t.Fatalf("bad identity should not be accepted")
	}
}

func TestLoopbackAssignAndEvents(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()

	id := newIdentity(t, "c1")
	if _, err := lb.Register(context.Background(), RegisterRequest{Identity: id}); err != nil {
		t.Fatal(err)
	}
	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()
	ch, err := lb.SubscribeAssignments(ctx, id.ID)
	if err != nil {
		t.Fatal(err)
	}

	// Second subscription rejected.
	if _, err := lb.SubscribeAssignments(ctx, id.ID); err == nil {
		t.Fatalf("expected double-subscription to be rejected")
	}

	if err := lb.Assign(id.ID, AssignSegment{JobID: "j1", SegmentID: "s1", Prompt: "hi", Model: "m"}); err != nil {
		t.Fatal(err)
	}
	select {
	case a := <-ch:
		if a.JobID != "j1" || a.SegmentID != "s1" {
			t.Fatalf("unexpected assignment: %+v", a)
		}
	case <-time.After(time.Second):
		t.Fatal("timed out waiting for assignment")
	}

	// Stream tokens then done.
	if err := lb.SendSegmentEvent(ctx, SegmentEvent{JobID: "j1", SegmentID: "s1", Kind: SegmentTokens, Tokens: "hello"}); err != nil {
		t.Fatal(err)
	}
	if err := lb.SendSegmentEvent(ctx, SegmentEvent{JobID: "j1", SegmentID: "s1", Kind: SegmentDone}); err != nil {
		t.Fatal(err)
	}
	evs := lb.Events()
	if len(evs) != 2 || evs[0].Kind != SegmentTokens || evs[1].Kind != SegmentDone {
		t.Fatalf("events: %+v", evs)
	}

	// Cancel the subscription and confirm re-subscription works.
	cancelFn()
	// Give the watcher goroutine a moment.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		ctx2, cancel2 := context.WithCancel(context.Background())
		if _, err := lb.SubscribeAssignments(ctx2, id.ID); err == nil {
			cancel2()
			return
		} else {
			cancel2()
			time.Sleep(10 * time.Millisecond)
		}
	}
	t.Fatal("subscription slot not released after ctx cancel")
}

func TestLoopbackAssignUnknownNode(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()
	err := lb.Assign(mustID(t), AssignSegment{JobID: "j"})
	if !errors.Is(err, ErrUnknownNode) {
		t.Fatalf("err=%v want ErrUnknownNode", err)
	}
}

// fakeSecondary captures cancels for assertion.
type fakeSecondary struct {
	mu      sync.Mutex
	calls   []CancelRequest
	returns error
}

func (f *fakeSecondary) Cancel(ctx context.Context, req CancelRequest) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, req)
	return f.returns
}

func TestLoopbackCancelForwarding(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()

	id := newIdentity(t, "c1")
	if _, err := lb.Register(context.Background(), RegisterRequest{Identity: id}); err != nil {
		t.Fatal(err)
	}
	fs := &fakeSecondary{}
	if err := lb.RegisterSecondary(id.ID, fs); err != nil {
		t.Fatal(err)
	}

	req := CancelRequest{JobID: "j1", SegmentID: "s1", Reason: cancel.ReasonCallerCancelled}
	if err := lb.Cancel(context.Background(), id.ID, req); err != nil {
		t.Fatalf("Cancel: %v", err)
	}
	fs.mu.Lock()
	defer fs.mu.Unlock()
	if len(fs.calls) != 1 || fs.calls[0] != req {
		t.Fatalf("cancel not forwarded: %+v", fs.calls)
	}

	// Unknown node.
	err := lb.Cancel(context.Background(), mustID(t), req)
	if !errors.Is(err, ErrUnknownNode) {
		t.Fatalf("unknown cancel: err=%v", err)
	}
}

func TestLoopbackStateUpdates(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()
	id := newIdentity(t, "c1")
	if _, err := lb.Register(context.Background(), RegisterRequest{Identity: id}); err != nil {
		t.Fatal(err)
	}
	if err := lb.ReportStateUpdate(context.Background(), StateUpdate{NodeID: id.ID, From: state.Available, To: state.Syncing}); err != nil {
		t.Fatal(err)
	}
	if got := lb.StateUpdates(); len(got) != 1 || got[0].To != state.Syncing {
		t.Fatalf("state updates: %+v", got)
	}
}

func TestLoopbackCloseStopsOperations(t *testing.T) {
	lb := NewLoopback(0)
	id := newIdentity(t, "c1")
	if _, err := lb.Register(context.Background(), RegisterRequest{Identity: id}); err != nil {
		t.Fatal(err)
	}
	lb.Close()
	// Double-close is a no-op.
	lb.Close()

	if _, err := lb.Register(context.Background(), RegisterRequest{Identity: newIdentity(t, "c2")}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Register after close: err=%v", err)
	}
	if err := lb.Heartbeat(context.Background(), Heartbeat{NodeID: id.ID}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Heartbeat after close: err=%v", err)
	}
	if err := lb.Assign(id.ID, AssignSegment{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("Assign after close: err=%v", err)
	}
	if err := lb.SendSegmentEvent(context.Background(), SegmentEvent{}); !errors.Is(err, ErrClosed) {
		t.Fatalf("SendSegmentEvent after close: err=%v", err)
	}
	if err := lb.ReportStateUpdate(context.Background(), StateUpdate{NodeID: id.ID}); !errors.Is(err, ErrClosed) {
		t.Fatalf("ReportStateUpdate after close: err=%v", err)
	}
	if _, err := lb.SubscribeAssignments(context.Background(), id.ID); !errors.Is(err, ErrClosed) {
		t.Fatalf("SubscribeAssignments after close: err=%v", err)
	}
}

func TestLoopbackAssignmentQueueFull(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()
	id := newIdentity(t, "c1")
	if _, err := lb.Register(context.Background(), RegisterRequest{Identity: id}); err != nil {
		t.Fatal(err)
	}
	// Default buffer = 16. Fill it then confirm the next Assign errors.
	for i := 0; i < 16; i++ {
		if err := lb.Assign(id.ID, AssignSegment{JobID: "j", SegmentID: "s"}); err != nil {
			t.Fatalf("fill %d: %v", i, err)
		}
	}
	if err := lb.Assign(id.ID, AssignSegment{JobID: "j", SegmentID: "overflow"}); err == nil {
		t.Fatalf("expected queue-full error")
	}
}

func TestLoopbackConcurrentRegister(t *testing.T) {
	lb := NewLoopback(0)
	defer lb.Close()

	var wg sync.WaitGroup
	var accepted int32
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			id := newIdentity(t, "c1")
			resp, err := lb.Register(context.Background(), RegisterRequest{Identity: id})
			if err == nil && resp.Accepted {
				atomic.AddInt32(&accepted, 1)
			}
		}()
	}
	wg.Wait()
	if atomic.LoadInt32(&accepted) != 50 {
		t.Fatalf("expected 50 accepted, got %d", accepted)
	}
	if got := len(lb.Nodes()); got != 50 {
		t.Fatalf("registered nodes: got %d want 50", got)
	}
}

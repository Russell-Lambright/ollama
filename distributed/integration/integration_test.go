// Package integration houses end-to-end tests for the distributed
// MPI-style framework. These tests wire the real Orchestrator, real
// Secondary runtimes, and a real ModelSync Syncer together through the
// in-memory Loopback transport — no fakes at the component boundary.
//
// They live in a separate package (distributed/integration) to keep the
// unit tests per package narrow and fast while still letting us assert
// the cross-module contracts spelled out in DISTRIBUTED_ARCHITECTURE.md.
package integration

import (
	"context"
	"errors"
	"fmt"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/distributed/cancel"
	"github.com/ollama/ollama/distributed/config"
	"github.com/ollama/ollama/distributed/modelsync"
	dcnode "github.com/ollama/ollama/distributed/node"
	"github.com/ollama/ollama/distributed/orchestrator"
	"github.com/ollama/ollama/distributed/secondary"
	"github.com/ollama/ollama/distributed/state"
	"github.com/ollama/ollama/distributed/transport"
	"github.com/ollama/ollama/sppr"
)

// ---------------------------------------------------------------------
// Harness: composes Orchestrator + Loopback into a single Primary and
// Dispatcher so Secondaries and Orchestrator share a consistent view.
// ---------------------------------------------------------------------

// primaryShim routes RPCs to either the Orchestrator (registration,
// heartbeat, state, events) or the Loopback (assignment subscription).
// Real gRPC / HTTP2-SSE adapters will perform the same split.
type primaryShim struct {
	orch *orchestrator.Orchestrator
	loop *transport.Loopback
}

func (p *primaryShim) Register(ctx context.Context, req transport.RegisterRequest) (transport.RegisterResponse, error) {
	// Register with BOTH so the loopback knows about the node for
	// assignment delivery AND the orchestrator has it in its registry.
	if _, err := p.loop.Register(ctx, req); err != nil {
		return transport.RegisterResponse{}, err
	}
	return p.orch.Register(ctx, req)
}

func (p *primaryShim) Heartbeat(ctx context.Context, hb transport.Heartbeat) error {
	return p.orch.Heartbeat(ctx, hb)
}

func (p *primaryShim) ReportStateUpdate(ctx context.Context, su transport.StateUpdate) error {
	return p.orch.ReportStateUpdate(ctx, su)
}

func (p *primaryShim) SubscribeAssignments(ctx context.Context, id dcnode.ID) (<-chan transport.AssignSegment, error) {
	return p.loop.SubscribeAssignments(ctx, id)
}

func (p *primaryShim) SendSegmentEvent(ctx context.Context, ev transport.SegmentEvent) error {
	return p.orch.SendSegmentEvent(ctx, ev)
}

// harnessDispatcher adapts the Loopback to the orchestrator.Dispatcher
// contract.
type harnessDispatcher struct{ l *transport.Loopback }

func (d *harnessDispatcher) Assign(id dcnode.ID, a transport.AssignSegment) error {
	return d.l.Assign(id, a)
}

func (d *harnessDispatcher) Cancel(ctx context.Context, id dcnode.ID, req transport.CancelRequest) error {
	return d.l.Cancel(ctx, id, req)
}

// harness wires one Orchestrator and N Secondaries together.
type harness struct {
	t          *testing.T
	cfg        *config.DistributedConfig
	orch       *orchestrator.Orchestrator
	loop       *transport.Loopback
	primary    transport.Primary
	secs       []*secondary.Secondary
	secCancels []context.CancelFunc
	secDone    []chan struct{}
}

// newHarness creates an orchestrator + Loopback and registers n
// secondaries using the given executor factory. The secondaries are
// fully registered and Available by the time newHarness returns.
func newHarness(t *testing.T, n int, collective string, executor secondary.Executor) *harness {
	t.Helper()
	cfg := config.Default()
	cfg.DefaultCollective = collective
	cfg.MaxNodesPerCollective = 8
	loop := transport.NewLoopback(50 * time.Millisecond)
	disp := &harnessDispatcher{l: loop}
	orch, err := orchestrator.New(orchestrator.Options{
		Cfg:               &cfg,
		Dispatcher:        disp,
		HeartbeatInterval: 50 * time.Millisecond,
	})
	if err != nil {
		t.Fatal(err)
	}
	h := &harness{
		t:       t,
		cfg:     &cfg,
		orch:    orch,
		loop:    loop,
		primary: &primaryShim{orch: orch, loop: loop},
	}
	for i := 0; i < n; i++ {
		h.addSecondary(collective, executor)
	}
	// Wait for every secondary to become Available.
	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) {
		snap := orch.Snapshot()
		avail := 0
		for _, nd := range snap.Nodes {
			if nd.State == state.Available {
				avail++
			}
		}
		if avail == n {
			return h
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("only %d/%d secondaries became Available in time", len(h.secs), n)
	return h
}

func (h *harness) addSecondary(collective string, executor secondary.Executor) *secondary.Secondary {
	id, err := dcnode.NewID()
	if err != nil {
		h.t.Fatal(err)
	}
	identity := dcnode.Identity{
		ID:            id,
		Hostname:      fmt.Sprintf("node-%d", len(h.secs)),
		Collective:    collective,
		AdvertisedLPU: 1.0 + float64(len(h.secs)),
	}
	sec, err := secondary.New(secondary.Options{
		Identity:          identity,
		Primary:           h.primary,
		Executor:          executor,
		HeartbeatOverride: 50 * time.Millisecond,
	})
	if err != nil {
		h.t.Fatal(err)
	}
	// Register this secondary's Cancel handler with the loopback.
	if err := h.loop.RegisterSecondary(identity.ID, sec); err != nil {
		h.t.Fatal(err)
	}
	ctx, cancelFn := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		defer close(done)
		_ = sec.Run(ctx)
	}()
	h.secs = append(h.secs, sec)
	h.secCancels = append(h.secCancels, cancelFn)
	h.secDone = append(h.secDone, done)
	return sec
}

func (h *harness) shutdown() {
	for _, c := range h.secCancels {
		c()
	}
	for _, d := range h.secDone {
		select {
		case <-d:
		case <-time.After(2 * time.Second):
			h.t.Log("secondary did not shut down within 2s")
		}
	}
	h.loop.Close()
}

// echoExecutor emits `Prompt` uppercased as a single tokens event + done.
func echoExecutor() secondary.Executor {
	return secondary.ExecutorFunc(func(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error) {
		ch := make(chan transport.SegmentEvent, 2)
		ch <- transport.SegmentEvent{JobID: a.JobID, SegmentID: a.SegmentID, Kind: transport.SegmentTokens, Tokens: strings.ToUpper(a.Prompt)}
		ch <- transport.SegmentEvent{JobID: a.JobID, SegmentID: a.SegmentID, Kind: transport.SegmentDone}
		close(ch)
		return ch, nil
	})
}

// ---------------------------------------------------------------------
// End-to-end tests
// ---------------------------------------------------------------------

func TestEndToEnd_SPPRToCoalesce(t *testing.T) {
	h := newHarness(t, 3, "c1", echoExecutor())
	defer h.shutdown()

	// Synthesize SPPR segments (bypass the real model; this test
	// exercises the orchestration path, not linguistic parsing).
	segs := []sppr.Segment{
		{ID: "s0", Order: 0, Text: "first part"},
		{ID: "s1", Order: 1, Text: "second part"},
		{ID: "s2", Order: 2, Text: "third part"},
	}

	res, err := h.orch.Dispatch(context.Background(), orchestrator.Job{
		ID:              "job-e2e",
		Collective:      "c1",
		Model:           "test-model",
		Segments:        segs,
		ConcurrencyHint: 1,
	})
	if err != nil {
		t.Fatalf("dispatch: %v", err)
	}
	if len(res.Outputs) != 3 {
		t.Fatalf("outputs=%+v", res.Outputs)
	}
	want := []string{"FIRST PART", "SECOND PART", "THIRD PART"}
	for i, got := range res.Outputs {
		if got != want[i] {
			t.Fatalf("output[%d]=%q want %q", i, got, want[i])
		}
	}
	if !strings.Contains(res.Coalesced, "FIRST PART") || !strings.Contains(res.Coalesced, "THIRD PART") {
		t.Fatalf("coalesced missing parts: %q", res.Coalesced)
	}
	// All three secondaries should have been used because
	// ceil(3/1)=3 ≤ 3 available.
	if len(res.NodeIDs) != 3 {
		t.Fatalf("nodes=%v", res.NodeIDs)
	}
}

func TestEndToEnd_NoAvailableNodes(t *testing.T) {
	// Zero secondaries → fixed rejection message.
	h := newHarness(t, 0, "c1", echoExecutor())
	defer h.shutdown()
	_, err := h.orch.Dispatch(context.Background(), orchestrator.Job{
		ID:       "j",
		Segments: []sppr.Segment{{ID: "s", Text: "x"}},
	})
	if !errors.Is(err, orchestrator.ErrNoAvailableNodes) {
		t.Fatalf("err=%v want ErrNoAvailableNodes", err)
	}
	if err.Error() != orchestrator.NoAvailableNodesMessage {
		t.Fatalf("wrong message: %q", err.Error())
	}
}

func TestEndToEnd_CallerCancel(t *testing.T) {
	// Executor blocks on ctx.Done so we can test cancellation cleanly.
	blockingExec := secondary.ExecutorFunc(func(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error) {
		ch := make(chan transport.SegmentEvent, 1)
		go func() {
			defer close(ch)
			<-ctx.Done()
			ch <- transport.SegmentEvent{JobID: a.JobID, SegmentID: a.SegmentID, Kind: transport.SegmentError, Detail: "cancelled"}
		}()
		return ch, nil
	})
	h := newHarness(t, 2, "c1", blockingExec)
	defer h.shutdown()

	ctx, cancelFn := context.WithCancel(context.Background())
	done := make(chan orchestrator.Result, 1)
	errCh := make(chan error, 1)
	go func() {
		r, err := h.orch.Dispatch(ctx, orchestrator.Job{
			ID:       "cancel-me",
			Model:    "m",
			Segments: []sppr.Segment{{ID: "s0", Order: 0, Text: "a"}, {ID: "s1", Order: 1, Text: "b"}},
		})
		done <- r
		errCh <- err
	}()
	// Give the assignments time to propagate.
	time.Sleep(100 * time.Millisecond)
	cancelFn()

	select {
	case r := <-done:
		if r.CancelReason != cancel.ReasonCallerCancelled {
			t.Fatalf("reason=%s want caller_cancelled", r.CancelReason)
		}
		if err := <-errCh; err == nil {
			t.Fatal("expected cancel error")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("dispatch did not return after caller cancel")
	}
}

func TestEndToEnd_NodeFailureCancelsJob(t *testing.T) {
	var started atomic.Int64
	blocker := secondary.ExecutorFunc(func(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error) {
		started.Add(1)
		ch := make(chan transport.SegmentEvent, 1)
		go func() {
			defer close(ch)
			<-ctx.Done()
			ch <- transport.SegmentEvent{JobID: a.JobID, SegmentID: a.SegmentID, Kind: transport.SegmentError, Detail: "cancelled"}
		}()
		return ch, nil
	})
	h := newHarness(t, 2, "c1", blocker)
	defer h.shutdown()

	done := make(chan orchestrator.Result, 1)
	errCh := make(chan error, 1)
	go func() {
		r, err := h.orch.Dispatch(context.Background(), orchestrator.Job{
			ID:       "fault",
			Model:    "m",
			Segments: []sppr.Segment{{ID: "s0", Order: 0, Text: "a"}, {ID: "s1", Order: 1, Text: "b"}},
		})
		done <- r
		errCh <- err
	}()
	// Wait until at least one executor has started, then simulate a
	// node failure by reporting Failed via the orchestrator's
	// state-update RPC.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && started.Load() == 0 {
		time.Sleep(5 * time.Millisecond)
	}
	if started.Load() == 0 {
		t.Fatal("executor never started")
	}
	// Pick any assigned node from the job and mark it Failed.
	snap := h.orch.Snapshot()
	var failID string
	for _, j := range snap.Jobs {
		if j.ID == "fault" && len(j.Nodes) > 0 {
			failID = j.Nodes[0]
			break
		}
	}
	if failID == "" {
		t.Fatalf("could not find assigned node: %+v", snap.Jobs)
	}
	if err := h.orch.ReportStateUpdate(context.Background(), transport.StateUpdate{NodeID: dcnode.ID(failID), From: state.Processing, To: state.Failed}); err != nil {
		t.Fatal(err)
	}
	select {
	case r := <-done:
		if r.CancelReason != cancel.ReasonNodeFailed {
			t.Fatalf("reason=%s want node_failed", r.CancelReason)
		}
		if err := <-errCh; err == nil {
			t.Fatal("expected error")
		}
	case <-time.After(3 * time.Second):
		t.Fatal("dispatch did not return after node failure")
	}
}

func TestEndToEnd_StarvationIndexTightens(t *testing.T) {
	// Use an executor that reports errors to force the starvation
	// monitor's failure counter up. After a burst of failed jobs the
	// live StarvationIndex should drop below the configured default.
	failingExec := secondary.ExecutorFunc(func(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error) {
		ch := make(chan transport.SegmentEvent, 1)
		ch <- transport.SegmentEvent{JobID: a.JobID, SegmentID: a.SegmentID, Kind: transport.SegmentError, Detail: "boom"}
		close(ch)
		return ch, nil
	})
	h := newHarness(t, 2, "c1", failingExec)
	defer h.shutdown()

	initial := h.orch.StarvationIndex()
	for i := 0; i < 10; i++ {
		_, _ = h.orch.Dispatch(context.Background(), orchestrator.Job{
			ID:       fmt.Sprintf("j%d", i),
			Model:    "m",
			Segments: []sppr.Segment{{ID: "s", Text: "x"}},
		})
	}
	after := h.orch.StarvationIndex()
	if after >= initial {
		t.Fatalf("starvation did not tighten: before=%v after=%v", initial, after)
	}
	if after < config.MinStarvationIndex {
		t.Fatalf("starvation below floor: %v", after)
	}
}

// ---------------------------------------------------------------------
// Model sync end-to-end
// ---------------------------------------------------------------------

func TestEndToEnd_ModelSyncPullsMissing(t *testing.T) {
	var pulls atomic.Int64
	prov := modelsync.ManifestProviderFunc{
		ExpectedFn: func(ctx context.Context) (modelsync.Manifest, error) {
			return modelsync.NewManifest(modelsync.Entry{Name: "m", Digest: "d"}), nil
		},
		ObservedFn: func(ctx context.Context) (modelsync.Manifest, error) {
			return modelsync.NewManifest(), nil
		},
	}
	puller := modelsync.ModelPullerFunc(func(ctx context.Context, e modelsync.Entry) error {
		pulls.Add(1)
		return nil
	})
	s, err := modelsync.New(modelsync.Options{Provider: prov, Puller: puller})
	if err != nil {
		t.Fatal(err)
	}
	// Plug this Syncer into a fresh secondary to exercise the
	// Syncing→Available transition end-to-end.
	loop := transport.NewLoopback(50 * time.Millisecond)
	defer loop.Close()
	id := dcnode.Identity{ID: dcnode.MustNewID(), Hostname: "h", Collective: "c1", AdvertisedLPU: 1}
	sec, err := secondary.New(secondary.Options{
		Identity: id,
		Primary:  loop,
		Executor: echoExecutor(),
		Syncer:   s,
	})
	if err != nil {
		t.Fatal(err)
	}
	ctx, cancelFn := context.WithCancel(context.Background())
	defer cancelFn()
	go func() { _ = sec.Run(ctx) }()

	// Wait for Available.
	deadline := time.Now().Add(2 * time.Second)
	for time.Now().Before(deadline) && sec.State() != state.Available {
		time.Sleep(5 * time.Millisecond)
	}
	if sec.State() != state.Available {
		t.Fatalf("state=%s want Available", sec.State())
	}
	if pulls.Load() != 1 {
		t.Fatalf("pulls=%d want 1", pulls.Load())
	}
}

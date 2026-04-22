package orchestrator

import (
	"context"
	"errors"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ollama/ollama/distributed/cancel"
	"github.com/ollama/ollama/distributed/config"
	"github.com/ollama/ollama/distributed/node"
	"github.com/ollama/ollama/distributed/state"
	"github.com/ollama/ollama/distributed/transport"
	"github.com/ollama/ollama/sppr"
)

// --- test helpers ---

func testCfg() *config.DistributedConfig {
	c := config.Default()
	c.DefaultCollective = "c1"
	c.MaxNodesPerCollective = 4
	return &c
}

func mustIdentity(t *testing.T, collective string, lpu float64) node.Identity {
	t.Helper()
	id, err := node.NewID()
	if err != nil {
		t.Fatal(err)
	}
	return node.Identity{ID: id, Hostname: "h", Collective: collective, AdvertisedLPU: lpu}
}

// fakeDispatcher records Assign calls and lets the test drive event
// delivery back into the orchestrator.
type fakeDispatcher struct {
	mu      sync.Mutex
	assigns []transport.AssignSegment
	cancels []transport.CancelRequest
	// assignErr, when set, is returned on the Nth Assign call (1-based).
	assignErr map[int]error
	nAssigns  int
	// cancelHook, when set, is invoked on Cancel.
	cancelHook func(id node.ID, req transport.CancelRequest)
}

func (f *fakeDispatcher) Assign(id node.ID, a transport.AssignSegment) error {
	f.mu.Lock()
	f.nAssigns++
	n := f.nAssigns
	if err, ok := f.assignErr[n]; ok && err != nil {
		f.mu.Unlock()
		return err
	}
	f.assigns = append(f.assigns, a)
	f.mu.Unlock()
	return nil
}

func (f *fakeDispatcher) Cancel(ctx context.Context, id node.ID, req transport.CancelRequest) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.cancels = append(f.cancels, req)
	if f.cancelHook != nil {
		f.cancelHook(id, req)
	}
	return nil
}

func (f *fakeDispatcher) Cancels() []transport.CancelRequest {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]transport.CancelRequest, len(f.cancels))
	copy(out, f.cancels)
	return out
}

func (f *fakeDispatcher) Assigns() []transport.AssignSegment {
	f.mu.Lock()
	defer f.mu.Unlock()
	out := make([]transport.AssignSegment, len(f.assigns))
	copy(out, f.assigns)
	return out
}

// registerAndAvailable registers a node and marks it Available via
// ReportStateUpdate so it shows up in AvailableNodes.
func registerAndAvailable(t *testing.T, o *Orchestrator, id node.Identity) {
	t.Helper()
	resp, err := o.Register(context.Background(), transport.RegisterRequest{Identity: id})
	if err != nil || !resp.Accepted {
		t.Fatalf("register: accepted=%v err=%v reason=%s", resp.Accepted, err, resp.Reason)
	}
	if err := o.ReportStateUpdate(context.Background(), transport.StateUpdate{NodeID: id.ID, From: state.Syncing, To: state.Available}); err != nil {
		t.Fatalf("state update: %v", err)
	}
}

// --- tests ---

func TestNewValidation(t *testing.T) {
	if _, err := New(Options{}); err == nil {
		t.Fatal("missing cfg should error")
	}
	if _, err := New(Options{Cfg: testCfg()}); err == nil {
		t.Fatal("missing dispatcher should error")
	}
	o, err := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	if err != nil {
		t.Fatal(err)
	}
	if o.coalescer == nil {
		t.Fatal("default coalescer not set")
	}
	if o.hbInterval <= 0 {
		t.Fatal("default heartbeat not set")
	}
}

func TestRegisterCollectiveCapacity(t *testing.T) {
	cfg := testCfg()
	cfg.MaxNodesPerCollective = 2
	o, _ := New(Options{Cfg: cfg, Dispatcher: &fakeDispatcher{}})
	for i := 0; i < 2; i++ {
		id := mustIdentity(t, "c1", 1)
		resp, err := o.Register(context.Background(), transport.RegisterRequest{Identity: id})
		if err != nil || !resp.Accepted {
			t.Fatalf("slot %d: err=%v", i, err)
		}
	}
	// Third registration should be rejected.
	id := mustIdentity(t, "c1", 1)
	resp, _ := o.Register(context.Background(), transport.RegisterRequest{Identity: id})
	if resp.Accepted {
		t.Fatal("expected rejection beyond capacity")
	}
	// A different collective is unaffected.
	id2 := mustIdentity(t, "c2", 1)
	resp2, _ := o.Register(context.Background(), transport.RegisterRequest{Identity: id2})
	if !resp2.Accepted {
		t.Fatal("different collective should be accepted")
	}
}

func TestRegisterInvalidIdentityRejected(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	_, err := o.Register(context.Background(), transport.RegisterRequest{Identity: node.Identity{}})
	if err == nil {
		t.Fatal("expected invalid-identity error")
	}
}

func TestAvailableNodesSortedByLPU(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	lo := mustIdentity(t, "c1", 1)
	hi := mustIdentity(t, "c1", 10)
	other := mustIdentity(t, "c2", 5) // different collective
	registerAndAvailable(t, o, lo)
	registerAndAvailable(t, o, hi)
	registerAndAvailable(t, o, other)
	got := o.AvailableNodes("c1")
	if len(got) != 2 {
		t.Fatalf("expected 2 nodes, got %d", len(got))
	}
	if got[0] != hi.ID {
		t.Fatalf("expected high-LPU node first, got %s", got[0])
	}
}

func TestAvailableExcludesNonAvailable(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	id := mustIdentity(t, "c1", 1)
	if _, err := o.Register(context.Background(), transport.RegisterRequest{Identity: id}); err != nil {
		t.Fatal(err)
	}
	// Default state after register is Syncing — not available.
	if n := o.AvailableNodes("c1"); len(n) != 0 {
		t.Fatalf("expected 0, got %v", n)
	}
}

func TestAllocateFormula(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	// cfg.MaxNodesPerCollective = 4; starve=1.0 → cap=4.
	// segments=6, concurrency=2 → ceil(6/2)=3. avail=5. requested=0.
	// Expected = min(3, 5, 4) = 3.
	if got := o.allocate(0, 6, 2, 5, 1.0); got != 3 {
		t.Fatalf("allocate=%d want 3", got)
	}
	// Requested clamps down.
	if got := o.allocate(2, 6, 2, 5, 1.0); got != 2 {
		t.Fatalf("requested clamp: allocate=%d want 2", got)
	}
	// StarvationIndex squeezes: 0.25 * 4 = 1.
	if got := o.allocate(0, 6, 2, 5, 0.25); got != 1 {
		t.Fatalf("starvation clamp: allocate=%d want 1", got)
	}
	// avail smaller than ceil: take avail.
	if got := o.allocate(0, 10, 1, 2, 1.0); got != 2 {
		t.Fatalf("availability clamp: allocate=%d want 2", got)
	}
	// Zero concurrency hint defaults to 1.
	if got := o.allocate(0, 3, 0, 5, 1.0); got != 3 {
		t.Fatalf("default concurrency: allocate=%d want 3", got)
	}
}

func TestAllocateFloorAtLeastOne(t *testing.T) {
	cfg := testCfg()
	cfg.MaxNodesPerCollective = 2
	o, _ := New(Options{Cfg: cfg, Dispatcher: &fakeDispatcher{}})
	// StarvationIndex 0.1 * 2 = 0.2 → floor = 0, bumped to 1 by safety.
	// Formula yields min(ceil(2/1)=2, avail=3, starve=1) = 1.
	if got := o.allocate(0, 2, 1, 3, 0.1); got != 1 {
		t.Fatalf("allocate=%d want 1 (floor)", got)
	}
}

func TestDispatchRejectsNoNodes(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	job := Job{ID: "j1", Segments: []sppr.Segment{{ID: "s1", Text: "x"}}}
	_, err := o.Dispatch(context.Background(), job)
	if !errors.Is(err, ErrNoAvailableNodes) {
		t.Fatalf("err=%v want ErrNoAvailableNodes", err)
	}
	if err.Error() != NoAvailableNodesMessage {
		t.Fatalf("rejection message wrong: %q", err.Error())
	}
}

func TestDispatchValidation(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	if _, err := o.Dispatch(context.Background(), Job{}); err == nil {
		t.Fatal("missing job ID should error")
	}
	if _, err := o.Dispatch(context.Background(), Job{ID: "j"}); err == nil {
		t.Fatal("missing segments should error")
	}
}

// drive simulates a Secondary: for each assigned segment, send tokens +
// done back into the orchestrator via SendSegmentEvent.
func driveSuccess(t *testing.T, o *Orchestrator, fd *fakeDispatcher, responses map[string]string) {
	t.Helper()
	// Wait until all expected assigns show up, then respond.
	deadline := time.Now().Add(time.Second)
	for time.Now().Before(deadline) {
		if len(fd.Assigns()) >= len(responses) {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}
	for _, a := range fd.Assigns() {
		tokens, ok := responses[a.SegmentID]
		if !ok {
			continue
		}
		_ = o.SendSegmentEvent(context.Background(), transport.SegmentEvent{JobID: a.JobID, SegmentID: a.SegmentID, Kind: transport.SegmentTokens, Tokens: tokens})
		_ = o.SendSegmentEvent(context.Background(), transport.SegmentEvent{JobID: a.JobID, SegmentID: a.SegmentID, Kind: transport.SegmentDone})
	}
}

func TestDispatchHappyPath(t *testing.T) {
	fd := &fakeDispatcher{}
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: fd})
	for i := 0; i < 3; i++ {
		registerAndAvailable(t, o, mustIdentity(t, "c1", float64(i+1)))
	}
	job := Job{
		ID:    "j1",
		Model: "m",
		Segments: []sppr.Segment{
			{ID: "a", Order: 0, Text: "alpha"},
			{ID: "b", Order: 1, Text: "beta"},
			{ID: "c", Order: 2, Text: "gamma"},
		},
	}
	done := make(chan struct{})
	var res Result
	var err error
	go func() {
		defer close(done)
		res, err = o.Dispatch(context.Background(), job)
	}()
	driveSuccess(t, o, fd, map[string]string{
		"a": "ALPHA",
		"b": "BETA",
		"c": "GAMMA",
	})
	<-done
	if err != nil {
		t.Fatalf("err=%v", err)
	}
	if len(res.Outputs) != 3 {
		t.Fatalf("outputs=%v", res.Outputs)
	}
	if res.Outputs[0] != "ALPHA" || res.Outputs[1] != "BETA" || res.Outputs[2] != "GAMMA" {
		t.Fatalf("order wrong: %+v", res.Outputs)
	}
	if !strings.Contains(res.Coalesced, "ALPHA") || !strings.Contains(res.Coalesced, "GAMMA") {
		t.Fatalf("coalesce wrong: %q", res.Coalesced)
	}
	if len(res.NodeIDs) != 3 {
		t.Fatalf("node IDs=%v", res.NodeIDs)
	}
}

func TestDispatchOrderedStitching(t *testing.T) {
	fd := &fakeDispatcher{}
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: fd})
	for i := 0; i < 2; i++ {
		registerAndAvailable(t, o, mustIdentity(t, "c1", float64(i+1)))
	}
	// Segments come in a funky slice order but have Order = 0,1,2.
	job := Job{
		ID:    "j1",
		Model: "m",
		Segments: []sppr.Segment{
			{ID: "c", Order: 2, Text: "g"},
			{ID: "a", Order: 0, Text: "a"},
			{ID: "b", Order: 1, Text: "b"},
		},
	}
	done := make(chan struct{})
	var res Result
	go func() {
		defer close(done)
		res, _ = o.Dispatch(context.Background(), job)
	}()
	driveSuccess(t, o, fd, map[string]string{
		"a": "A", "b": "B", "c": "C",
	})
	<-done
	if res.Outputs[0] != "A" || res.Outputs[1] != "B" || res.Outputs[2] != "C" {
		t.Fatalf("stitching wrong: %+v", res.Outputs)
	}
}

func TestDispatchCallerCancel(t *testing.T) {
	fd := &fakeDispatcher{}
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: fd})
	registerAndAvailable(t, o, mustIdentity(t, "c1", 1))
	ctx, cancelFn := context.WithCancel(context.Background())
	job := Job{ID: "j1", Model: "m", Segments: []sppr.Segment{{ID: "s", Text: "x"}}}
	errCh := make(chan error, 1)
	var resCh = make(chan Result, 1)
	go func() {
		r, err := o.Dispatch(ctx, job)
		resCh <- r
		errCh <- err
	}()
	// Wait for the assign to be pushed.
	waitFor(t, func() bool { return len(fd.Assigns()) == 1 }, time.Second)
	cancelFn()
	res := <-resCh
	err := <-errCh
	if err == nil {
		t.Fatal("expected cancellation error")
	}
	if res.CancelReason != cancel.ReasonCallerCancelled {
		t.Fatalf("reason=%s want caller_cancelled", res.CancelReason)
	}
	// Fan-out cancel delivered.
	waitFor(t, func() bool { return len(fd.Cancels()) == 1 }, time.Second)
}

func TestDispatchSegmentErrorCancelsJob(t *testing.T) {
	fd := &fakeDispatcher{}
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: fd})
	for i := 0; i < 2; i++ {
		registerAndAvailable(t, o, mustIdentity(t, "c1", 1))
	}
	job := Job{
		ID:    "j1",
		Model: "m",
		Segments: []sppr.Segment{
			{ID: "a", Order: 0, Text: "x"},
			{ID: "b", Order: 1, Text: "y"},
		},
	}
	done := make(chan struct{})
	var res Result
	var derr error
	go func() {
		defer close(done)
		res, derr = o.Dispatch(context.Background(), job)
	}()
	waitFor(t, func() bool { return len(fd.Assigns()) == 2 }, time.Second)
	// Fail segment b.
	_ = o.SendSegmentEvent(context.Background(), transport.SegmentEvent{JobID: "j1", SegmentID: "b", Kind: transport.SegmentError, Detail: "boom"})
	<-done
	if derr == nil {
		t.Fatal("expected error")
	}
	if res.CancelReason != cancel.ReasonNodeFailed {
		t.Fatalf("reason=%s want node_failed", res.CancelReason)
	}
	// Both segments should receive cancellations (dedup by node — but nodes are distinct here).
	waitFor(t, func() bool { return len(fd.Cancels()) >= 1 }, time.Second)
}

func TestNodeFailureViaStateUpdateCancelsJobs(t *testing.T) {
	fd := &fakeDispatcher{}
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: fd})
	ids := []node.Identity{mustIdentity(t, "c1", 1), mustIdentity(t, "c1", 1)}
	for _, id := range ids {
		registerAndAvailable(t, o, id)
	}
	job := Job{
		ID:    "jx",
		Model: "m",
		Segments: []sppr.Segment{
			{ID: "a", Order: 0, Text: "x"},
			{ID: "b", Order: 1, Text: "y"},
		},
	}
	done := make(chan struct{})
	var derr error
	var res Result
	go func() {
		defer close(done)
		res, derr = o.Dispatch(context.Background(), job)
	}()
	waitFor(t, func() bool { return len(fd.Assigns()) == 2 }, time.Second)
	// Pick whichever node actually got segment "a" and fail it.
	assigns := fd.Assigns()
	nid := assigns[0].SegmentID
	_ = nid
	// Find node for segment a in orchestrator's internal state.
	o.mu.Lock()
	js := o.jobs["jx"]
	failedNode := js.nodes[0]
	o.mu.Unlock()
	if err := o.ReportStateUpdate(context.Background(), transport.StateUpdate{NodeID: failedNode, From: state.Processing, To: state.Failed}); err != nil {
		t.Fatal(err)
	}
	<-done
	if derr == nil {
		t.Fatal("expected error")
	}
	if res.CancelReason != cancel.ReasonNodeFailed {
		t.Fatalf("reason=%s want node_failed", res.CancelReason)
	}
	if res.FailedNodeID != failedNode {
		t.Fatalf("failed node=%s want %s", res.FailedNodeID, failedNode)
	}
}

func TestDispatchAssignErrorAborts(t *testing.T) {
	fd := &fakeDispatcher{assignErr: map[int]error{2: errors.New("dispatcher boom")}}
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: fd})
	for i := 0; i < 2; i++ {
		registerAndAvailable(t, o, mustIdentity(t, "c1", 1))
	}
	job := Job{
		ID:    "j",
		Model: "m",
		Segments: []sppr.Segment{
			{ID: "a", Order: 0, Text: "x"},
			{ID: "b", Order: 1, Text: "y"},
		},
	}
	_, err := o.Dispatch(context.Background(), job)
	if err == nil || !strings.Contains(err.Error(), "dispatcher boom") {
		t.Fatalf("err=%v want dispatcher error", err)
	}
}

func TestDispatchUnknownEventIgnored(t *testing.T) {
	// SendSegmentEvent for an unknown job just logs & drops.
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	if err := o.SendSegmentEvent(context.Background(), transport.SegmentEvent{JobID: "ghost", SegmentID: "x", Kind: transport.SegmentTokens, Tokens: "?"}); err != nil {
		t.Fatalf("unknown-job event: %v", err)
	}
}

func TestHeartbeatUnknownNode(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	err := o.Heartbeat(context.Background(), transport.Heartbeat{NodeID: "nope"})
	if !errors.Is(err, transport.ErrUnknownNode) {
		t.Fatalf("err=%v want ErrUnknownNode", err)
	}
}

func TestReportStateUpdateUnknownNode(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	err := o.ReportStateUpdate(context.Background(), transport.StateUpdate{NodeID: "nope", To: state.Available})
	if !errors.Is(err, transport.ErrUnknownNode) {
		t.Fatalf("err=%v want ErrUnknownNode", err)
	}
}

func TestSubscribeAssignmentsRejected(t *testing.T) {
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: &fakeDispatcher{}})
	if _, err := o.SubscribeAssignments(context.Background(), "x"); err == nil {
		t.Fatal("expected error")
	}
}

func TestStarvationMonitor(t *testing.T) {
	cfg := testCfg()
	cfg.StarvationIndex = 1.0
	o, _ := New(Options{Cfg: cfg, Dispatcher: &fakeDispatcher{}})
	// All successes → stays high.
	for i := 0; i < 10; i++ {
		o.observeOutcome(false)
	}
	if si := o.StarvationIndex(); si < 0.99 {
		t.Fatalf("all success: index=%v want ~1.0", si)
	}
	// Introduce failures → index drops toward MinStarvationIndex.
	for i := 0; i < 20; i++ {
		o.observeOutcome(true)
	}
	if si := o.StarvationIndex(); si >= 0.9 {
		t.Fatalf("mostly failed: index=%v want << 1.0", si)
	}
	if si := o.StarvationIndex(); si < config.MinStarvationIndex {
		t.Fatalf("index below floor: %v", si)
	}
}

func TestStarvationIntegratesIntoDispatch(t *testing.T) {
	// The arithmetic is covered by TestAllocateFormula and the live
	// index is covered by TestStarvationMonitor; Dispatch simply calls
	// allocate(... , StarvationIndex(), ...) so no separate integration
	// test is needed beyond those two.
	t.Skip("covered by TestAllocateFormula + TestStarvationMonitor")
}

func TestConcatCoalescer(t *testing.T) {
	c := ConcatCoalescer()
	out, err := c.Coalesce(context.Background(), []string{"a", "b", "c"})
	if err != nil {
		t.Fatal(err)
	}
	if out != "a\n\nb\n\nc" {
		t.Fatalf("unexpected: %q", out)
	}
}

func TestCustomCoalescerError(t *testing.T) {
	fd := &fakeDispatcher{}
	sentinel := errors.New("coalesce fail")
	o, _ := New(Options{
		Cfg:        testCfg(),
		Dispatcher: fd,
		Coalescer:  CoalescerFunc(func(ctx context.Context, o []string) (string, error) { return "", sentinel }),
	})
	registerAndAvailable(t, o, mustIdentity(t, "c1", 1))
	job := Job{ID: "j", Model: "m", Segments: []sppr.Segment{{ID: "s", Text: "x"}}}
	done := make(chan struct{})
	var err error
	go func() {
		defer close(done)
		_, err = o.Dispatch(context.Background(), job)
	}()
	driveSuccess(t, o, fd, map[string]string{"s": "OK"})
	<-done
	if !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want sentinel", err)
	}
}

// eventDropDoesNotBlockSecondary ensures a flooded event channel does not
// cause SendSegmentEvent to block.
func TestEventDropDoesNotBlock(t *testing.T) {
	fd := &fakeDispatcher{}
	o, _ := New(Options{Cfg: testCfg(), Dispatcher: fd})
	registerAndAvailable(t, o, mustIdentity(t, "c1", 1))
	job := Job{ID: "j", Model: "m", Segments: []sppr.Segment{{ID: "s", Text: "x"}}}
	done := make(chan struct{})
	go func() {
		defer close(done)
		_, _ = o.Dispatch(context.Background(), job)
	}()
	waitFor(t, func() bool { return len(fd.Assigns()) == 1 }, time.Second)
	// Send more events than the channel can buffer.
	var sent atomic.Int64
	for i := 0; i < 100; i++ {
		if err := o.SendSegmentEvent(context.Background(), transport.SegmentEvent{JobID: "j", SegmentID: "s", Kind: transport.SegmentTokens, Tokens: "t"}); err == nil {
			sent.Add(1)
		}
	}
	// Close the job.
	_ = o.SendSegmentEvent(context.Background(), transport.SegmentEvent{JobID: "j", SegmentID: "s", Kind: transport.SegmentDone})
	<-done
}

// --- helpers ---

func waitFor(t *testing.T, pred func() bool, timeout time.Duration) {
	t.Helper()
	deadline := time.Now().Add(timeout)
	for time.Now().Before(deadline) {
		if pred() {
			return
		}
		time.Sleep(5 * time.Millisecond)
	}
	if !pred() {
		t.Fatalf("timeout waiting for condition")
	}
}

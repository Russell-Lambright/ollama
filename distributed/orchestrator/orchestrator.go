// Package orchestrator implements the Primary-side job scheduler for
// the distributed MPI-style framework.
//
// Scope (Phase 5):
//
//   - Node registry: tracks which Secondaries are registered, their
//     collectives, and their current state.
//   - Scheduler: implements the spec §5 allocation formula with
//     StarvationIndex throttling.
//   - Starvation monitor: adjusts StarvationIndex in [0.1, 1.0] based on
//     queue depth and failure rate.
//   - Execution driver: dispatches segments to selected nodes, correlates
//     SegmentEvents back, stitches them in-order.
//   - Shared per-job cancel context: caller-cancel or node-failure cancels
//     every assigned Secondary.
//   - Optional QA "coherence" pass via a pluggable Coalescer.
//   - Fixed rejection message when no nodes are available.
//
// The orchestrator is transport-agnostic: it drives any transport.Primary
// implementation (loopback for tests, gRPC/HTTP2+SSE for production).
// It is also configuration-driven: every knob is read from a
// DistributedConfig value. Collective membership is tracked per node
// identity; there is no requirement that all nodes belong to the same
// collective.
package orchestrator

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ollama/ollama/distributed/cancel"
	"github.com/ollama/ollama/distributed/config"
	"github.com/ollama/ollama/distributed/node"
	"github.com/ollama/ollama/distributed/state"
	"github.com/ollama/ollama/distributed/transport"
	"github.com/ollama/ollama/sppr"
)

// NoAvailableNodesMessage is the fixed rejection message the
// orchestrator returns to callers when no node in the targeted
// collective is Available. Spec §5.
const NoAvailableNodesMessage = "Sorry but we are not able to take your call at this time. Please try again later."

// ErrNoAvailableNodes is returned by Dispatch when the target collective
// has zero Available nodes. The error message is NoAvailableNodesMessage
// so callers can surface it verbatim.
var ErrNoAvailableNodes = errors.New(NoAvailableNodesMessage)

// Dispatcher is the transport surface the orchestrator uses to push
// assignments and cancels to Secondaries. It is a narrow subset of the
// Primary-side helpers on transport.Loopback; production adapters
// (gRPC, HTTP2+SSE) will implement the same two methods.
type Dispatcher interface {
	// Assign pushes an assignment to the given node's subscription.
	Assign(id node.ID, a transport.AssignSegment) error
	// Cancel tells the given node's Secondary-side handler to abort the
	// specified segment. Idempotent per transport.Secondary.Cancel.
	Cancel(ctx context.Context, id node.ID, req transport.CancelRequest) error
}

// Coalescer is an optional post-correlation step that takes the ordered
// list of segment outputs and returns a single coherent response. The
// simplest implementation is string concatenation (see ConcatCoalescer),
// but production deployments can plug in a QA model.
type Coalescer interface {
	Coalesce(ctx context.Context, outputs []string) (string, error)
}

// CoalescerFunc adapts a plain function to the Coalescer interface.
type CoalescerFunc func(ctx context.Context, outputs []string) (string, error)

// Coalesce implements Coalescer.
func (f CoalescerFunc) Coalesce(ctx context.Context, outputs []string) (string, error) {
	return f(ctx, outputs)
}

// ConcatCoalescer joins segment outputs with a single blank line.
// Useful as a default when no QA model is configured.
func ConcatCoalescer() Coalescer {
	return CoalescerFunc(func(ctx context.Context, outputs []string) (string, error) {
		return strings.Join(outputs, "\n\n"), nil
	})
}

// Options configures an Orchestrator.
type Options struct {
	// Cfg is the distributed config; required.
	Cfg *config.DistributedConfig
	// Dispatcher pushes assignments / cancels; required.
	Dispatcher Dispatcher
	// Coalescer is optional; defaults to ConcatCoalescer.
	Coalescer Coalescer
	// HeartbeatInterval is returned to registering Secondaries. Zero
	// falls back to transport.DefaultHeartbeatInterval.
	HeartbeatInterval time.Duration
	// StarvationWindow is how much historical queue/failure data the
	// starvation monitor considers. Zero disables the monitor (fixed
	// StarvationIndex = Cfg.StarvationIndex).
	StarvationWindow time.Duration
}

// Job describes a single distributed request.
type Job struct {
	// ID is a caller-supplied (or auto-generated) job identifier.
	ID string
	// Collective is the collective to dispatch within. Empty means
	// Cfg.DefaultCollective.
	Collective string
	// Model is the model name segments will run under.
	Model string
	// RequestedNodes, when > 0, caps the number of nodes used (before
	// clamping by availability and StarvationIndex).
	RequestedNodes int
	// ConcurrencyHint influences the allocation formula:
	// ceil(segment_count / concurrency_hint). Defaults to 1 when ≤ 0.
	ConcurrencyHint int
	// Segments are the SPPR segments to fan out.
	Segments []sppr.Segment
	// Options are sampling parameters forwarded to each Secondary.
	Options map[string]any
}

// Result is the outcome of a completed Job.
type Result struct {
	JobID string
	// Outputs are the per-segment outputs in Segment.Order order.
	Outputs []string
	// Coalesced is the Coalescer's output over Outputs.
	Coalesced string
	// NodeIDs are the nodes that participated, in assignment order.
	NodeIDs []node.ID
	// CancelReason is populated when the job was cancelled.
	CancelReason cancel.Reason
	// FailedNodeID is populated when CancelReason == ReasonNodeFailed.
	FailedNodeID node.ID
}

// ---------------------------------------------------------------------
// Node registry
// ---------------------------------------------------------------------

type nodeEntry struct {
	identity node.Identity
	// state is the last known state (from Register / StateUpdate /
	// Heartbeat). Machine transitions are not enforced server-side;
	// the orchestrator treats the reported state as authoritative.
	state state.State
	// lastHeartbeat is used by a future liveness monitor (not part of
	// Phase 5 scope but tracked so the monitor can land without
	// reshaping this struct).
	lastHeartbeat time.Time
	// currentLPU is the latest reported LPU (may differ from advertised
	// due to thermal throttling).
	currentLPU float64
}

// Orchestrator is the Primary's job scheduler. It implements
// transport.Primary (so Secondaries can call it for register /
// heartbeat / state-update / event) AND exposes Dispatch for callers.
// Construct with New. Safe for concurrent use.
type Orchestrator struct {
	cfg        *config.DistributedConfig
	dispatcher Dispatcher
	coalescer  Coalescer
	hbInterval time.Duration
	log        *slog.Logger

	// Starvation monitor state. Protected by starveMu.
	starveMu       sync.Mutex
	starveIndex    float64 // live value in [0.1, 1.0]
	starveWindow   time.Duration
	recentFailures int
	recentSuccess  int
	queueDepth     int

	// Node + job registry. Protected by mu.
	mu   sync.Mutex
	seq  uint64
	reg  map[node.ID]*nodeEntry
	jobs map[string]*jobState // live jobs (by job ID)
	// perNodeJobs tracks the job IDs active per node for fast lookup
	// during node-failure fan-out.
	perNodeJobs map[node.ID]map[string]struct{}
	// assignSubs simulates a subscription — maintained by loopback but
	// unneeded here; the dispatcher performs the actual assign push.
}

// jobState holds the orchestrator's live view of a dispatched job. It is
// immutable after Dispatch returns, except for the event fan-in channel
// and per-segment output accumulator.
type jobState struct {
	id         string
	collective string
	segments   []sppr.Segment
	nodes      []node.ID // parallel to segments — segments[i] assigned to nodes[i]
	outputs    []strings.Builder
	done       []bool
	events     chan transport.SegmentEvent
	cancel     context.CancelFunc
	cancelOnce sync.Once
	reason     cancel.Reason
	failedNode node.ID
}

// New constructs an Orchestrator.
func New(opts Options) (*Orchestrator, error) {
	if opts.Cfg == nil {
		return nil, errors.New("orchestrator: Cfg is required")
	}
	if opts.Dispatcher == nil {
		return nil, errors.New("orchestrator: Dispatcher is required")
	}
	if opts.Coalescer == nil {
		opts.Coalescer = ConcatCoalescer()
	}
	hb := opts.HeartbeatInterval
	if hb <= 0 {
		hb = transport.DefaultHeartbeatInterval
	}
	si := opts.Cfg.StarvationIndex
	if si <= 0 {
		si = config.DefaultStarvationIndex
	}
	o := &Orchestrator{
		cfg:          opts.Cfg,
		dispatcher:   opts.Dispatcher,
		coalescer:    opts.Coalescer,
		hbInterval:   hb,
		log:          slog.With("component", "distributed/orchestrator"),
		starveIndex:  si,
		starveWindow: opts.StarvationWindow,
		reg:          make(map[node.ID]*nodeEntry),
		jobs:         make(map[string]*jobState),
		perNodeJobs:  make(map[node.ID]map[string]struct{}),
	}
	o.log.Info("orchestrator: initialized",
		"max_nodes_per_collective", opts.Cfg.MaxNodesPerCollective,
		"default_collective", opts.Cfg.DefaultCollective,
		"starvation_index", si,
		"heartbeat_interval", hb,
	)
	return o, nil
}

// StarvationIndex returns the live starvation-index value.
func (o *Orchestrator) StarvationIndex() float64 {
	o.starveMu.Lock()
	defer o.starveMu.Unlock()
	return o.starveIndex
}

// ---------------------------------------------------------------------
// transport.Primary implementation
// ---------------------------------------------------------------------

// Register implements transport.Primary.
func (o *Orchestrator) Register(ctx context.Context, req transport.RegisterRequest) (transport.RegisterResponse, error) {
	if err := req.Identity.Validate(); err != nil {
		o.log.Warn("orchestrator: register rejected (invalid identity)", "err", err)
		return transport.RegisterResponse{Accepted: false, Reason: err.Error()}, err
	}
	o.mu.Lock()
	// Enforce MaxNodesPerCollective at registration time (spec §5).
	if o.cfg.MaxNodesPerCollective > 0 && req.Identity.Collective != "" {
		same := 0
		for _, n := range o.reg {
			if n.identity.Collective == req.Identity.Collective && n.identity.ID != req.Identity.ID {
				same++
			}
		}
		if same >= o.cfg.MaxNodesPerCollective {
			o.mu.Unlock()
			reason := fmt.Sprintf("collective %q at capacity (%d)", req.Identity.Collective, o.cfg.MaxNodesPerCollective)
			o.log.Warn("orchestrator: register rejected (collective full)", "node", string(req.Identity.ID), "collective", req.Identity.Collective)
			return transport.RegisterResponse{Accepted: false, Reason: reason}, nil
		}
	}
	existing, ok := o.reg[req.Identity.ID]
	if !ok {
		o.reg[req.Identity.ID] = &nodeEntry{
			identity:      req.Identity,
			state:         state.Syncing,
			lastHeartbeat: time.Now(),
			currentLPU:    req.Identity.AdvertisedLPU,
		}
	} else {
		existing.identity = req.Identity
		existing.lastHeartbeat = time.Now()
	}
	o.mu.Unlock()
	o.log.Info("orchestrator: node registered",
		"node", string(req.Identity.ID),
		"hostname", req.Identity.Hostname,
		"collective", req.Identity.Collective,
		"advertised_lpu", req.Identity.AdvertisedLPU,
	)
	return transport.RegisterResponse{Accepted: true, HeartbeatInterval: o.hbInterval}, nil
}

// Heartbeat implements transport.Primary.
func (o *Orchestrator) Heartbeat(ctx context.Context, hb transport.Heartbeat) error {
	o.mu.Lock()
	n, ok := o.reg[hb.NodeID]
	if ok {
		n.lastHeartbeat = time.Now()
		n.state = hb.State
		n.currentLPU = hb.CurrentLPU
	}
	o.mu.Unlock()
	if !ok {
		o.log.Warn("orchestrator: heartbeat from unknown node", "node", string(hb.NodeID))
		return fmt.Errorf("%w: %s", transport.ErrUnknownNode, hb.NodeID)
	}
	o.log.Debug("orchestrator: heartbeat", "node", string(hb.NodeID), "state", hb.State, "lpu", hb.CurrentLPU, "jobs", len(hb.JobIDs))
	return nil
}

// ReportStateUpdate implements transport.Primary.
func (o *Orchestrator) ReportStateUpdate(ctx context.Context, su transport.StateUpdate) error {
	o.mu.Lock()
	n, ok := o.reg[su.NodeID]
	if ok {
		n.state = su.To
	}
	// Gather job IDs the node is participating in so we can fan
	// cancellations out below without holding the lock across I/O.
	var affectedJobs []string
	if ok && su.To == state.Failed {
		for jid := range o.perNodeJobs[su.NodeID] {
			affectedJobs = append(affectedJobs, jid)
		}
	}
	o.mu.Unlock()
	if !ok {
		o.log.Warn("orchestrator: state update from unknown node", "node", string(su.NodeID))
		return fmt.Errorf("%w: %s", transport.ErrUnknownNode, su.NodeID)
	}
	o.log.Info("orchestrator: node state updated", "node", string(su.NodeID), "from", su.From, "to", su.To)
	// Node-failure → cancel every job this node was part of.
	for _, jid := range affectedJobs {
		o.cancelJob(jid, cancel.ReasonNodeFailed, su.NodeID, fmt.Sprintf("node %s entered Failed", su.NodeID))
	}
	return nil
}

// SubscribeAssignments implements transport.Primary by delegating to the
// underlying transport. The orchestrator does not maintain its own
// subscription book; it dispatches through opts.Dispatcher.
func (o *Orchestrator) SubscribeAssignments(ctx context.Context, id node.ID) (<-chan transport.AssignSegment, error) {
	// The Orchestrator side does not host subscriptions; Secondaries
	// talk to the transport directly. Return an error so callers that
	// accidentally plug the orchestrator in as a Secondary-side
	// transport see a clear failure.
	return nil, errors.New("orchestrator: SubscribeAssignments is served by the transport, not the orchestrator")
}

// SendSegmentEvent implements transport.Primary. It is the ingress point
// for Secondary-produced events during job execution.
//
// For non-terminal `tokens` events we prefer a non-blocking send: an
// over-eager Secondary must never be able to wedge the orchestrator.
// For terminal events (`done`, `error`) we fall back to a ctx-bounded
// blocking send so the dispatch loop always sees completion.
func (o *Orchestrator) SendSegmentEvent(ctx context.Context, ev transport.SegmentEvent) error {
	o.mu.Lock()
	j, ok := o.jobs[ev.JobID]
	o.mu.Unlock()
	if !ok {
		// Late event for an already-finished/cancelled job; drop
		// silently beyond a debug log.
		o.log.Debug("orchestrator: segment event for unknown/finished job", "job", ev.JobID, "segment", ev.SegmentID, "kind", ev.Kind)
		return nil
	}
	// Fast path: non-blocking.
	select {
	case j.events <- ev:
		return nil
	default:
	}
	if ev.Kind == transport.SegmentTokens {
		// Drop non-terminal events under back-pressure to avoid wedging
		// the transport layer.
		o.log.Warn("orchestrator: event channel full; dropping non-terminal event", "job", ev.JobID, "segment", ev.SegmentID, "kind", ev.Kind)
		return nil
	}
	// Terminal events block (ctx-bounded) so the dispatch loop always
	// observes completion.
	select {
	case j.events <- ev:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// ---------------------------------------------------------------------
// Scheduling
// ---------------------------------------------------------------------

// AvailableNodes returns the node IDs in the target collective whose
// last-reported state is Available, sorted by currentLPU descending
// (higher LPU → preferred). The returned slice is a snapshot.
func (o *Orchestrator) AvailableNodes(collective string) []node.ID {
	o.mu.Lock()
	defer o.mu.Unlock()
	type candidate struct {
		id  node.ID
		lpu float64
	}
	var cands []candidate
	for id, n := range o.reg {
		if collective != "" && n.identity.Collective != collective {
			continue
		}
		if n.state != state.Available {
			continue
		}
		cands = append(cands, candidate{id: id, lpu: n.currentLPU})
	}
	sort.Slice(cands, func(i, j int) bool { return cands[i].lpu > cands[j].lpu })
	out := make([]node.ID, len(cands))
	for i, c := range cands {
		out[i] = c.id
	}
	return out
}

// allocate implements the spec §5 allocation formula:
//
//	n = min(
//	  requested_nodes (if supplied),
//	  ceil(segment_count / concurrency_hint),
//	  available_nodes_in_collective,
//	  floor(MaxNodesPerCollective * STARVATION_INDEX),
//	)
//
// Returns the node-count to actually use. The caller then takes the
// top-N nodes from AvailableNodes.
func (o *Orchestrator) allocate(requested, segmentCount, concurrencyHint, availableCount int, starvationIndex float64) int {
	ch := concurrencyHint
	if ch <= 0 {
		ch = 1
	}
	ceilSegments := int(math.Ceil(float64(segmentCount) / float64(ch)))
	max := o.cfg.MaxNodesPerCollective
	if max <= 0 {
		max = segmentCount
	}
	starveCap := int(math.Floor(float64(max) * starvationIndex))
	if starveCap < 1 {
		starveCap = 1
	}
	candidates := []int{ceilSegments, availableCount, starveCap}
	if requested > 0 {
		candidates = append(candidates, requested)
	}
	n := candidates[0]
	for _, c := range candidates[1:] {
		if c < n {
			n = c
		}
	}
	if n < 0 {
		n = 0
	}
	return n
}

// ---------------------------------------------------------------------
// Starvation monitor
// ---------------------------------------------------------------------

// observeOutcome feeds the starvation monitor with job outcomes.
// Positive results ease throttling; failures and queue buildup tighten it.
func (o *Orchestrator) observeOutcome(failed bool) {
	o.starveMu.Lock()
	defer o.starveMu.Unlock()
	if failed {
		o.recentFailures++
	} else {
		o.recentSuccess++
	}
	o.recalcLocked()
}

// observeQueueDepth updates the latest queue-depth reading used by the
// starvation monitor.
func (o *Orchestrator) observeQueueDepth(depth int) {
	o.starveMu.Lock()
	defer o.starveMu.Unlock()
	o.queueDepth = depth
	o.recalcLocked()
}

// recalcLocked applies a simple proportional heuristic:
//
//   - Base index tracks config value.
//   - Failure rate > 0 lowers the index; rate == 0 restores it.
//   - Queue depth is ignored in Phase 5a — the scheduler's natural
//     availability clamp already handles back-pressure; we leave the
//     hook here so later phases can extend it without changing the
//     public surface.
//
// Clamped to [MinStarvationIndex, MaxStarvationIndex].
func (o *Orchestrator) recalcLocked() {
	total := o.recentFailures + o.recentSuccess
	if total == 0 {
		return
	}
	failRate := float64(o.recentFailures) / float64(total)
	target := o.cfg.StarvationIndex
	if target <= 0 {
		target = config.DefaultStarvationIndex
	}
	// Linear scale: fail_rate=0 → target; fail_rate=1 → MinStarvationIndex.
	new := target*(1.0-failRate) + config.MinStarvationIndex*failRate
	if new < config.MinStarvationIndex {
		new = config.MinStarvationIndex
	}
	if new > config.MaxStarvationIndex {
		new = config.MaxStarvationIndex
	}
	if math.Abs(new-o.starveIndex) > 0.001 {
		o.log.Info("orchestrator: starvation index updated", "from", o.starveIndex, "to", new, "failures", o.recentFailures, "successes", o.recentSuccess, "queue_depth", o.queueDepth)
	}
	o.starveIndex = new
}

// ---------------------------------------------------------------------
// Dispatch
// ---------------------------------------------------------------------

// Dispatch schedules and executes a job synchronously: it returns only
// after all segments have completed, failed, or been cancelled. The
// caller's ctx is the parent of the job's cancel context — cancelling
// ctx cancels the whole job (ReasonCallerCancelled).
//
// Single-segment jobs are still valid: the orchestrator executes them on
// exactly one node. The "small-job fallback" decision is owned by the
// caller (typically via sppr.Plan.SingleNode) — if SingleNode is true,
// the caller should NOT call Dispatch.
func (o *Orchestrator) Dispatch(ctx context.Context, job Job) (Result, error) {
	if job.ID == "" {
		return Result{}, errors.New("orchestrator: Job.ID is required")
	}
	if len(job.Segments) == 0 {
		return Result{}, errors.New("orchestrator: Job.Segments must be non-empty")
	}
	collective := job.Collective
	if collective == "" {
		collective = o.cfg.DefaultCollective
	}
	o.log.Info("orchestrator: dispatching",
		"job", job.ID,
		"collective", collective,
		"model", job.Model,
		"segments", len(job.Segments),
		"requested_nodes", job.RequestedNodes,
		"concurrency_hint", job.ConcurrencyHint,
	)

	avail := o.AvailableNodes(collective)
	if len(avail) == 0 {
		o.log.Warn("orchestrator: no available nodes", "job", job.ID, "collective", collective)
		return Result{JobID: job.ID}, ErrNoAvailableNodes
	}

	starve := o.StarvationIndex()
	nodeCount := o.allocate(job.RequestedNodes, len(job.Segments), job.ConcurrencyHint, len(avail), starve)
	if nodeCount <= 0 {
		o.log.Warn("orchestrator: allocation yielded zero nodes", "job", job.ID, "starve", starve)
		return Result{JobID: job.ID}, ErrNoAvailableNodes
	}
	if nodeCount > len(avail) {
		nodeCount = len(avail)
	}
	selected := avail[:nodeCount]
	o.log.Info("orchestrator: allocation complete",
		"job", job.ID,
		"available", len(avail),
		"selected", nodeCount,
		"starvation_index", starve,
	)

	// Round-robin segments across selected nodes — maintains order
	// determinism and is simpler than a multi-knapsack partition.
	nodes := make([]node.ID, len(job.Segments))
	for i := range job.Segments {
		nodes[i] = selected[i%len(selected)]
	}

	jobCtx, cancelFn := context.WithCancel(ctx)
	defer cancelFn()

	js := &jobState{
		id:         job.ID,
		collective: collective,
		segments:   job.Segments,
		nodes:      nodes,
		outputs:    make([]strings.Builder, len(job.Segments)),
		done:       make([]bool, len(job.Segments)),
		events:     make(chan transport.SegmentEvent, len(job.Segments)*8),
		cancel:     cancelFn,
	}
	o.mu.Lock()
	o.jobs[job.ID] = js
	for _, id := range selected {
		if o.perNodeJobs[id] == nil {
			o.perNodeJobs[id] = make(map[string]struct{})
		}
		o.perNodeJobs[id][job.ID] = struct{}{}
	}
	o.mu.Unlock()

	// Ensure full cleanup on exit.
	defer o.removeJob(job.ID)

	// Watch for caller-cancel to propagate to ReasonCallerCancelled.
	go func() {
		select {
		case <-ctx.Done():
			o.cancelJob(job.ID, cancel.ReasonCallerCancelled, "", "caller context cancelled")
		case <-jobCtx.Done():
		}
	}()

	// Push assignments.
	for i, seg := range job.Segments {
		prompt := seg.Text
		as := transport.AssignSegment{
			JobID:     job.ID,
			SegmentID: seg.ID,
			Prompt:    prompt,
			Model:     job.Model,
			Options:   job.Options,
		}
		o.log.Info("orchestrator: assigning segment",
			"job", job.ID,
			"segment", seg.ID,
			"order", i,
			"node", string(nodes[i]),
			"prompt_len", len(prompt),
		)
		if err := o.dispatcher.Assign(nodes[i], as); err != nil {
			o.log.Error("orchestrator: assign failed", "job", job.ID, "segment", seg.ID, "node", string(nodes[i]), "err", err)
			js.cancelOnce.Do(func() {
				js.reason = cancel.ReasonOrchestratorShutdown
				cancelFn()
			})
			return Result{JobID: job.ID}, fmt.Errorf("orchestrator: assign segment %s to node %s: %w", seg.ID, nodes[i], err)
		}
	}

	// Fan-in events until all segments finish or job is cancelled.
	pendingDone := len(job.Segments)
loop:
	for pendingDone > 0 {
		select {
		case <-jobCtx.Done():
			break loop
		case ev := <-js.events:
			idx := findSegmentIndex(job.Segments, ev.SegmentID)
			if idx < 0 {
				o.log.Warn("orchestrator: unknown segment event", "job", job.ID, "segment", ev.SegmentID)
				continue
			}
			switch ev.Kind {
			case transport.SegmentTokens:
				js.outputs[idx].WriteString(ev.Tokens)
			case transport.SegmentDone:
				if !js.done[idx] {
					js.done[idx] = true
					pendingDone--
					o.log.Info("orchestrator: segment done", "job", job.ID, "segment", ev.SegmentID, "remaining", pendingDone, "bytes", js.outputs[idx].Len())
				}
			case transport.SegmentError:
				o.log.Error("orchestrator: segment error", "job", job.ID, "segment", ev.SegmentID, "detail", ev.Detail)
				// A segment error fails the whole job. Record the node
				// (if we can) and cancel.
				failedNode := nodes[idx]
				o.cancelJob(job.ID, cancel.ReasonNodeFailed, failedNode, ev.Detail)
				break loop
			}
		}
	}

	// Build the result.
	res := Result{JobID: job.ID, NodeIDs: append([]node.ID(nil), nodes...)}
	// If the caller's ctx was cancelled, record that authoritatively —
	// Go's context propagation cancels jobCtx automatically, which means
	// the dispatch-loop exit may race the watcher goroutine that sets
	// js.reason. Reading ctx.Err() here removes the race.
	if ctx.Err() != nil {
		o.cancelJob(job.ID, cancel.ReasonCallerCancelled, "", "caller context cancelled")
	}
	// Capture reason if cancelled.
	o.mu.Lock()
	res.CancelReason = js.reason
	res.FailedNodeID = js.failedNode
	o.mu.Unlock()

	// Fan cancels out to all assigned nodes if the job was cancelled.
	if res.CancelReason != "" {
		o.fanOutCancel(ctx, js, res.CancelReason)
		o.observeOutcome(true)
		return res, fmt.Errorf("orchestrator: job %s cancelled: %s", job.ID, res.CancelReason)
	}

	// Collect outputs in Segment.Order.
	outputs := collectOutputs(job.Segments, js.outputs)
	res.Outputs = outputs

	coalesced, err := o.coalescer.Coalesce(ctx, outputs)
	if err != nil {
		o.log.Error("orchestrator: coalescer failed", "job", job.ID, "err", err)
		o.observeOutcome(true)
		return res, fmt.Errorf("orchestrator: coalesce: %w", err)
	}
	res.Coalesced = coalesced
	o.observeOutcome(false)
	o.log.Info("orchestrator: job complete", "job", job.ID, "segments", len(job.Segments), "bytes", len(coalesced))
	return res, nil
}

// cancelJob marks a live job for cancellation. Idempotent — only the
// first call records the reason and cancels the job ctx.
func (o *Orchestrator) cancelJob(jobID string, reason cancel.Reason, failedNode node.ID, detail string) {
	o.mu.Lock()
	j, ok := o.jobs[jobID]
	o.mu.Unlock()
	if !ok {
		return
	}
	j.cancelOnce.Do(func() {
		o.mu.Lock()
		j.reason = reason
		j.failedNode = failedNode
		o.mu.Unlock()
		o.log.Warn("orchestrator: cancelling job", "job", jobID, "reason", string(reason), "failed_node", string(failedNode), "detail", detail)
		j.cancel()
	})
}

// fanOutCancel tells every assigned node to abort its segment(s) for
// this job. Errors are logged but do not prevent fan-out.
func (o *Orchestrator) fanOutCancel(ctx context.Context, j *jobState, reason cancel.Reason) {
	seen := make(map[node.ID]struct{}, len(j.nodes))
	for i, id := range j.nodes {
		if _, dup := seen[id]; dup {
			continue
		}
		seen[id] = struct{}{}
		req := transport.CancelRequest{JobID: j.id, SegmentID: j.segments[i].ID, Reason: reason}
		if err := o.dispatcher.Cancel(ctx, id, req); err != nil {
			o.log.Warn("orchestrator: cancel delivery failed", "job", j.id, "node", string(id), "err", err)
		}
	}
}

// removeJob unregisters a job from the active set and releases its slots
// in perNodeJobs. Safe to call multiple times.
func (o *Orchestrator) removeJob(jobID string) {
	o.mu.Lock()
	defer o.mu.Unlock()
	j, ok := o.jobs[jobID]
	if !ok {
		return
	}
	for _, id := range j.nodes {
		if m, ok := o.perNodeJobs[id]; ok {
			delete(m, jobID)
			if len(m) == 0 {
				delete(o.perNodeJobs, id)
			}
		}
	}
	delete(o.jobs, jobID)
}

// collectOutputs returns a copy of the accumulated per-segment outputs
// in Segment.Order order.
func collectOutputs(segs []sppr.Segment, builders []strings.Builder) []string {
	type pair struct {
		order int
		text  string
	}
	pairs := make([]pair, len(segs))
	for i, s := range segs {
		pairs[i] = pair{order: s.Order, text: builders[i].String()}
	}
	sort.Slice(pairs, func(i, j int) bool { return pairs[i].order < pairs[j].order })
	out := make([]string, len(pairs))
	for i, p := range pairs {
		out[i] = p.text
	}
	return out
}

func findSegmentIndex(segs []sppr.Segment, segID string) int {
	for i, s := range segs {
		if s.ID == segID {
			return i
		}
	}
	return -1
}

// Compile-time assertions.
var _ transport.Primary = (*Orchestrator)(nil)

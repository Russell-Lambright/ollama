// Package transport defines the wire-agnostic contract between the
// Primary orchestrator and Secondary nodes in the distributed MPI-style
// framework, plus an in-memory loopback implementation suitable for tests
// and single-process development.
//
// Actual over-the-network implementations (gRPC and HTTP/2 + SSE, per the
// spec) are delivered as adapters that satisfy the interfaces declared
// here. They are NOT part of Phase 2 — the contract comes first so
// Phase 3 (Secondary runtime) and Phase 5 (Orchestrator) can be wired
// against a stable, testable surface.
//
// Design notes:
//
//   - Every RPC is request/response (Register, Heartbeat) or
//     long-running-stream-of-events (Assign produces a stream of segment
//     events; the Primary's Cancel RPC terminates it). This maps cleanly
//     onto both gRPC bidi streams and HTTP/2 + SSE.
//
//   - State updates flow from Secondary → Primary via Heartbeat (periodic
//     snapshot) and StateUpdate (push on transition). Both are kept even
//     though they carry overlapping information because heartbeats double
//     as liveness probes: a missed heartbeat window transitions the node
//     to Offline even if no StateUpdate was sent.
//
//   - Cancellation mirrors distributed/cancel.Reason — the transport does
//     not invent new cancellation taxonomies.
package transport

import (
	"context"
	"errors"
	"fmt"
	"sync"
	"time"

	"github.com/ollama/ollama/distributed/cancel"
	"github.com/ollama/ollama/distributed/node"
	"github.com/ollama/ollama/distributed/state"
)

// RegisterRequest is sent by a Secondary to announce its presence.
type RegisterRequest struct {
	Identity node.Identity
	// Capabilities is a free-form bag for forward compatibility (e.g.
	// "supports-persona-X", model availability hints). Empty is valid.
	Capabilities map[string]string
}

// RegisterResponse is returned by the Primary after accepting a Secondary.
type RegisterResponse struct {
	// Accepted is false if the Primary rejected the registration (e.g.
	// collective at MaxNodesPerCollective). Reason is populated in that case.
	Accepted bool
	Reason   string
	// HeartbeatInterval is the cadence at which the Secondary must send
	// heartbeats. A Secondary that fails to meet this cadence will be
	// marked Offline by the Primary.
	HeartbeatInterval time.Duration
}

// Heartbeat is a periodic liveness + state snapshot from a Secondary.
type Heartbeat struct {
	NodeID node.ID
	// State is the node's current lifecycle state.
	State state.State
	// CurrentLPU is the node's live capacity signal, which may differ from
	// the value advertised at registration (e.g. thermal throttling).
	CurrentLPU float64
	// JobIDs are the job IDs currently executing segments on this node.
	// Included so the Primary can cross-check its assignment book.
	JobIDs []string
}

// StateUpdate is a push notification sent on every state transition so the
// Primary's view of the node is never stale between heartbeats.
type StateUpdate struct {
	NodeID node.ID
	From   state.State
	To     state.State
	// JobID is the job associated with this transition, if any (e.g. when
	// entering a Processing* state). Empty when the transition is not
	// job-scoped (e.g. Available → Syncing for a model sync).
	JobID string
}

// AssignSegment instructs a Secondary to execute one segment of a
// distributed job. The Primary expects a stream of SegmentEvent values in
// response, terminated by either a final event or a cancellation.
type AssignSegment struct {
	JobID     string
	SegmentID string
	// Prompt is the already-SPPR-rendered text to execute. The Primary
	// performs all segmentation; Secondaries do not re-render.
	Prompt string
	// Model is the model name to use (the node must have it synced).
	Model string
	// Options are sampling parameters from DistributedConfig.FineTuning.
	// Kept as an opaque map so adding a parameter does not require a
	// transport revision.
	Options map[string]any
}

// SegmentEventKind classifies the events emitted during segment execution.
type SegmentEventKind string

const (
	// SegmentTokens: a chunk of generated tokens (streaming output).
	SegmentTokens SegmentEventKind = "tokens"
	// SegmentDone: terminal success event; no further events will arrive.
	SegmentDone SegmentEventKind = "done"
	// SegmentError: terminal error event; Detail holds the error string.
	SegmentError SegmentEventKind = "error"
)

// SegmentEvent is a single streamed event from a running segment.
type SegmentEvent struct {
	JobID     string
	SegmentID string
	Kind      SegmentEventKind
	// Tokens is populated for SegmentTokens events.
	Tokens string
	// Detail is populated for SegmentError events.
	Detail string
}

// CancelRequest aborts an in-flight segment. The Secondary MUST stop work,
// release resources, and (if the failure is not the node's fault)
// transition back to Available.
type CancelRequest struct {
	JobID     string
	SegmentID string
	Reason    cancel.Reason
}

// Primary is the server-side contract — what a Secondary sees when it
// talks to the orchestrator. Implementations live behind gRPC/HTTP2
// adapters (later phase) or the in-memory loopback in this package.
//
// Register is a single RPC. Heartbeat/StateUpdate are periodic one-ways.
// AssignSegment is driven by the Primary (via Stream returned from
// SubscribeAssignments) — the Secondary pulls assignments from that
// stream and replies with SegmentEvent channels.
type Primary interface {
	Register(ctx context.Context, req RegisterRequest) (RegisterResponse, error)
	Heartbeat(ctx context.Context, hb Heartbeat) error
	ReportStateUpdate(ctx context.Context, su StateUpdate) error
	// SubscribeAssignments returns a channel of assignments directed to
	// the given node. Cancel ctx to unsubscribe. The channel is closed
	// when the context is cancelled or the Primary tears down.
	SubscribeAssignments(ctx context.Context, nodeID node.ID) (<-chan AssignSegment, error)
	// SendSegmentEvent is how a Secondary streams output back to the
	// Primary. Calling with Kind=SegmentDone or SegmentError closes out
	// the segment on the Primary's side.
	SendSegmentEvent(ctx context.Context, ev SegmentEvent) error
}

// Secondary is the client-side contract — what the Primary sees when it
// talks to a worker node. The Primary uses it to push cancels.
type Secondary interface {
	// Cancel aborts a specific segment running on this Secondary. Must be
	// idempotent: cancelling an unknown or already-finished segment is
	// not an error.
	Cancel(ctx context.Context, req CancelRequest) error
}

// ErrUnknownNode is returned by the Primary when an operation targets a
// node that is not registered (or has been evicted).
var ErrUnknownNode = errors.New("transport: unknown node")

// ErrClosed is returned by operations on a transport that has been shut down.
var ErrClosed = errors.New("transport: closed")

// DefaultHeartbeatInterval is the cadence returned to Secondaries when the
// Primary does not override it.
const DefaultHeartbeatInterval = 5 * time.Second

// ---------------------------------------------------------------------
// In-memory loopback
// ---------------------------------------------------------------------

// Loopback is an in-process implementation of Primary + Secondary registry
// backed by Go channels. It is intended for unit tests and for Phase 3 /
// Phase 5 development before the wire transports are available. It is
// NOT safe to expose to untrusted code and does no authentication.
type Loopback struct {
	mu                sync.Mutex
	closed            bool
	heartbeatInterval time.Duration
	nodes             map[node.ID]*loopbackNode
	events            []SegmentEvent // append-only log for test inspection
	heartbeats        []Heartbeat
	stateUpdates      []StateUpdate
	// cancelHandlers maps nodeID → a Secondary implementation capable of
	// receiving Cancel calls. Populated by RegisterSecondary.
	cancelHandlers map[node.ID]Secondary
}

type loopbackNode struct {
	identity node.Identity
	// assignCh is per-node; the Primary writes AssignSegments, the
	// Secondary consumes them via SubscribeAssignments.
	assignCh chan AssignSegment
	// subscribed is true while a subscription is active; only one
	// subscriber per node is allowed (matches the real world: one live
	// gRPC stream per Secondary).
	subscribed bool
}

// NewLoopback constructs a new Loopback primary. heartbeatInterval is what
// it reports to Secondaries during registration; pass 0 for the default.
func NewLoopback(heartbeatInterval time.Duration) *Loopback {
	if heartbeatInterval <= 0 {
		heartbeatInterval = DefaultHeartbeatInterval
	}
	return &Loopback{
		heartbeatInterval: heartbeatInterval,
		nodes:             make(map[node.ID]*loopbackNode),
		cancelHandlers:    make(map[node.ID]Secondary),
	}
}

// Close tears down the loopback: closes every pending assignment channel
// and marks the Primary as unusable for further RPCs.
func (l *Loopback) Close() {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return
	}
	l.closed = true
	for _, n := range l.nodes {
		close(n.assignCh)
	}
}

// RegisterSecondary associates a client-side Secondary (usually the
// worker's local cancel handler) with a node ID so the loopback can
// deliver Cancel RPCs.
func (l *Loopback) RegisterSecondary(id node.ID, s Secondary) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return ErrClosed
	}
	l.cancelHandlers[id] = s
	return nil
}

// Primary contract.

func (l *Loopback) Register(ctx context.Context, req RegisterRequest) (RegisterResponse, error) {
	if err := req.Identity.Validate(); err != nil {
		return RegisterResponse{Accepted: false, Reason: err.Error()}, err
	}
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return RegisterResponse{}, ErrClosed
	}
	if _, exists := l.nodes[req.Identity.ID]; !exists {
		l.nodes[req.Identity.ID] = &loopbackNode{
			identity: req.Identity,
			assignCh: make(chan AssignSegment, 16),
		}
	} else {
		// Re-registration updates identity but preserves the channel so
		// any pending assignments are not lost.
		l.nodes[req.Identity.ID].identity = req.Identity
	}
	return RegisterResponse{Accepted: true, HeartbeatInterval: l.heartbeatInterval}, nil
}

func (l *Loopback) Heartbeat(ctx context.Context, hb Heartbeat) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return ErrClosed
	}
	if _, ok := l.nodes[hb.NodeID]; !ok {
		return fmt.Errorf("%w: %s", ErrUnknownNode, hb.NodeID)
	}
	l.heartbeats = append(l.heartbeats, hb)
	return nil
}

func (l *Loopback) ReportStateUpdate(ctx context.Context, su StateUpdate) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return ErrClosed
	}
	if _, ok := l.nodes[su.NodeID]; !ok {
		return fmt.Errorf("%w: %s", ErrUnknownNode, su.NodeID)
	}
	l.stateUpdates = append(l.stateUpdates, su)
	return nil
}

func (l *Loopback) SubscribeAssignments(ctx context.Context, id node.ID) (<-chan AssignSegment, error) {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return nil, ErrClosed
	}
	n, ok := l.nodes[id]
	if !ok {
		l.mu.Unlock()
		return nil, fmt.Errorf("%w: %s", ErrUnknownNode, id)
	}
	if n.subscribed {
		l.mu.Unlock()
		return nil, fmt.Errorf("transport: node %s already has an active subscriber", id)
	}
	n.subscribed = true
	ch := n.assignCh
	l.mu.Unlock()

	// Spawn a watcher to release the subscription slot when the caller's
	// ctx is cancelled. The channel itself outlives the subscription so
	// unconsumed assignments persist across reconnects (matches the
	// real-world behavior of a durable queue).
	go func() {
		<-ctx.Done()
		l.mu.Lock()
		if n2, ok := l.nodes[id]; ok {
			n2.subscribed = false
		}
		l.mu.Unlock()
	}()
	return ch, nil
}

func (l *Loopback) SendSegmentEvent(ctx context.Context, ev SegmentEvent) error {
	l.mu.Lock()
	defer l.mu.Unlock()
	if l.closed {
		return ErrClosed
	}
	l.events = append(l.events, ev)
	return nil
}

// Primary-side helpers (not part of the interface) used by the
// orchestrator to push work and cancels into the loopback.

// Assign enqueues an assignment for the given node. Returns ErrUnknownNode
// if the node has not registered.
func (l *Loopback) Assign(id node.ID, a AssignSegment) error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return ErrClosed
	}
	n, ok := l.nodes[id]
	if !ok {
		l.mu.Unlock()
		return fmt.Errorf("%w: %s", ErrUnknownNode, id)
	}
	ch := n.assignCh
	l.mu.Unlock()

	select {
	case ch <- a:
		return nil
	default:
		return fmt.Errorf("transport: assignment queue full for node %s", id)
	}
}

// Cancel forwards a cancellation to the Secondary registered for the
// given node, if any. The loopback resolves the registered Secondary from
// the node ID embedded in req.
func (l *Loopback) Cancel(ctx context.Context, id node.ID, req CancelRequest) error {
	l.mu.Lock()
	if l.closed {
		l.mu.Unlock()
		return ErrClosed
	}
	h, ok := l.cancelHandlers[id]
	l.mu.Unlock()
	if !ok {
		return fmt.Errorf("%w: %s", ErrUnknownNode, id)
	}
	return h.Cancel(ctx, req)
}

// Snapshot helpers for tests — return copies so callers can't mutate
// internal state.

func (l *Loopback) Heartbeats() []Heartbeat {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]Heartbeat, len(l.heartbeats))
	copy(out, l.heartbeats)
	return out
}

func (l *Loopback) StateUpdates() []StateUpdate {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]StateUpdate, len(l.stateUpdates))
	copy(out, l.stateUpdates)
	return out
}

func (l *Loopback) Events() []SegmentEvent {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]SegmentEvent, len(l.events))
	copy(out, l.events)
	return out
}

// Nodes returns a snapshot of registered node identities.
func (l *Loopback) Nodes() []node.Identity {
	l.mu.Lock()
	defer l.mu.Unlock()
	out := make([]node.Identity, 0, len(l.nodes))
	for _, n := range l.nodes {
		out = append(out, n.identity)
	}
	return out
}

// Compile-time assertion: *Loopback satisfies Primary.
var _ Primary = (*Loopback)(nil)

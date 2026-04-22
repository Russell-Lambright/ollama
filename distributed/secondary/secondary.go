// Package secondary implements the Secondary-mode runtime: the
// long-running loop a worker node executes to register with a Primary,
// report liveness, consume segment assignments, and stream results back.
//
// The package is intentionally transport-agnostic — it accepts any
// transport.Primary and any Executor / ModelSyncer / PersonaApplier. This
// keeps the runtime unit-testable against the in-memory loopback while
// the gRPC and HTTP/2+SSE wire adapters are still pending.
//
// Lifecycle (matches DISTRIBUTED_ARCHITECTURE.md §4):
//
//	Register → Syncing → Available ↻
//	                        ↓
//	                     Processing (per assignment) → Available
//	                        ↓ (on persona change)
//	                     Training → Available
//	                        ↓ (fatal error)
//	                     Failed
//	                        ↓ (connectivity loss)
//	                     Offline
//
// Only Available nodes accept new assignments. A sync failure surfaces as
// Failed and halts the consume loop. Cancellation of the node's root
// context tears down every goroutine cleanly.
package secondary

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/ollama/ollama/distributed/node"
	"github.com/ollama/ollama/distributed/state"
	"github.com/ollama/ollama/distributed/transport"
)

// Executor runs a single segment assigned to this node and emits events
// (tokens → done|error) on the returned channel. The channel MUST be
// closed by the executor before it returns. Cancellation is propagated
// via ctx; a cancelled context should produce at most one final
// SegmentError event (with Detail="cancelled" or similar) before close.
type Executor interface {
	Execute(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error)
}

// ExecutorFunc adapts a plain function to the Executor interface.
type ExecutorFunc func(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error)

// Execute implements Executor.
func (f ExecutorFunc) Execute(ctx context.Context, a transport.AssignSegment) (<-chan transport.SegmentEvent, error) {
	return f(ctx, a)
}

// ModelSyncer is invoked during the Syncing state to pull/verify models
// before the node advertises itself as Available. Return an error to
// force the node into Failed. Nil is a valid value meaning "no-op sync".
type ModelSyncer interface {
	Sync(ctx context.Context) error
}

// ModelSyncerFunc adapts a plain function to ModelSyncer.
type ModelSyncerFunc func(ctx context.Context) error

// Sync implements ModelSyncer.
func (f ModelSyncerFunc) Sync(ctx context.Context) error { return f(ctx) }

// PersonaApplier applies (or removes) a persona on this node. Called
// while the node is in the Training state. Return an error to force the
// node into Failed.
type PersonaApplier interface {
	Apply(ctx context.Context, persona string) error
}

// PersonaApplierFunc adapts a plain function to PersonaApplier.
type PersonaApplierFunc func(ctx context.Context, persona string) error

// Apply implements PersonaApplier.
func (f PersonaApplierFunc) Apply(ctx context.Context, persona string) error { return f(ctx, persona) }

// Options configures a Secondary.
type Options struct {
	// Identity is the self-description sent at registration. ID must be
	// valid; see distributed/node.Identity.Validate.
	Identity node.Identity
	// Primary is the transport to the orchestrator.
	Primary transport.Primary
	// Executor runs segment assignments. Required.
	Executor Executor
	// Syncer is invoked during the initial Syncing state. Optional; nil
	// means "no-op, transition straight to Available".
	Syncer ModelSyncer
	// PersonaApplier is invoked on ApplyPersona(). Optional.
	PersonaApplier PersonaApplier
	// MaxConcurrentSegments caps how many AssignSegment events may be
	// in flight simultaneously. Defaults to 1 (strict serialization,
	// matching the single-Processing-state constraint in the spec).
	MaxConcurrentSegments int
	// HeartbeatOverride, when > 0, overrides the interval returned by
	// the Primary in RegisterResponse. Tests use a small value.
	HeartbeatOverride time.Duration
}

// Secondary is the runtime. Construct with New and drive with Run.
type Secondary struct {
	opts    Options
	machine *state.Machine
	log     *slog.Logger

	mu       sync.Mutex
	running  bool
	jobs     map[string]context.CancelFunc // segmentID → cancel
	personaQ chan string                   // buffered persona-apply channel
	hbEvery  time.Duration
}

// New constructs a Secondary. Returns an error if required fields are
// missing or the identity is invalid.
func New(opts Options) (*Secondary, error) {
	if opts.Primary == nil {
		return nil, errors.New("secondary: Primary transport is required")
	}
	if opts.Executor == nil {
		return nil, errors.New("secondary: Executor is required")
	}
	if err := opts.Identity.Validate(); err != nil {
		return nil, err
	}
	if opts.MaxConcurrentSegments < 0 {
		opts.MaxConcurrentSegments = 0
	}
	if opts.MaxConcurrentSegments == 0 {
		opts.MaxConcurrentSegments = 1
	}
	// Start the machine in Syncing — the node is never "Available" from
	// the moment of construction, only after Run() has completed a sync.
	m, err := state.New(state.Syncing)
	if err != nil {
		return nil, err
	}
	return &Secondary{
		opts:     opts,
		machine:  m,
		log:      slog.With("component", "distributed/secondary", "node", string(opts.Identity.ID)),
		jobs:     make(map[string]context.CancelFunc),
		personaQ: make(chan string, 4),
	}, nil
}

// State returns the current lifecycle state.
func (s *Secondary) State() state.State { return s.machine.State() }

// ID returns the node ID.
func (s *Secondary) ID() node.ID { return s.opts.Identity.ID }

// Run drives the Secondary lifecycle. It registers with the Primary,
// performs initial model sync, then consumes assignments until ctx is
// cancelled or a fatal error occurs. Returns nil on clean shutdown
// (ctx.Err()) or the first fatal error.
func (s *Secondary) Run(ctx context.Context) error {
	s.mu.Lock()
	if s.running {
		s.mu.Unlock()
		return errors.New("secondary: already running")
	}
	s.running = true
	s.mu.Unlock()
	s.log.Info("secondary: starting", "hostname", s.opts.Identity.Hostname, "collective", s.opts.Identity.Collective, "advertised_lpu", s.opts.Identity.AdvertisedLPU)

	// Wire the state listener BEFORE registration so the very first
	// Syncing→Available transition is visible to the Primary.
	removeListener := s.machine.AddListener(func(from, to state.State) {
		s.log.Info("secondary: state transition", "from", from, "to", to)
		// Best-effort: a state-update delivery failure is not fatal; the
		// next heartbeat will reconcile.
		if err := s.opts.Primary.ReportStateUpdate(ctx, transport.StateUpdate{
			NodeID: s.opts.Identity.ID,
			From:   from,
			To:     to,
		}); err != nil {
			s.log.Debug("secondary: state-update delivery failed (will reconcile via heartbeat)", "from", from, "to", to, "err", err)
		}
	})
	defer removeListener()

	// 1) Register.
	s.log.Debug("secondary: registering with primary")
	resp, err := s.opts.Primary.Register(ctx, transport.RegisterRequest{
		Identity: s.opts.Identity,
	})
	if err != nil {
		_ = s.machine.Transition(state.Failed)
		s.log.Error("secondary: register failed", "err", err)
		return fmt.Errorf("secondary: register: %w", err)
	}
	if !resp.Accepted {
		_ = s.machine.Transition(state.Failed)
		s.log.Error("secondary: register rejected", "reason", resp.Reason)
		return fmt.Errorf("secondary: register rejected: %s", resp.Reason)
	}
	s.hbEvery = resp.HeartbeatInterval
	if s.opts.HeartbeatOverride > 0 {
		s.hbEvery = s.opts.HeartbeatOverride
	}
	if s.hbEvery <= 0 {
		s.hbEvery = transport.DefaultHeartbeatInterval
	}
	s.log.Info("secondary: registered", "heartbeat_interval", s.hbEvery)

	// 2) Sync models. Sync failure → Failed + return.
	if s.opts.Syncer != nil {
		s.log.Info("secondary: model sync starting")
		if err := s.opts.Syncer.Sync(ctx); err != nil {
			_ = s.machine.Transition(state.Failed)
			s.log.Error("secondary: model sync failed", "err", err)
			return fmt.Errorf("secondary: sync: %w", err)
		}
		s.log.Info("secondary: model sync complete")
	}
	if err := s.machine.Transition(state.Available); err != nil {
		s.log.Error("secondary: entering Available failed", "err", err)
		return fmt.Errorf("secondary: enter Available: %w", err)
	}

	// 3) Subscribe to assignments.
	assignCh, err := s.opts.Primary.SubscribeAssignments(ctx, s.opts.Identity.ID)
	if err != nil {
		_ = s.machine.Transition(state.Failed)
		s.log.Error("secondary: subscribe to assignments failed", "err", err)
		return fmt.Errorf("secondary: subscribe: %w", err)
	}
	s.log.Debug("secondary: subscribed to assignments", "max_concurrent", s.opts.MaxConcurrentSegments)

	// runCtx is cancelled when the consume loop exits so auxiliary
	// goroutines (heartbeat) tear down even if the caller's ctx is
	// still live (e.g. a fatal error from a persona applier).
	runCtx, runCancel := context.WithCancel(ctx)
	defer runCancel()

	// 4) Launch heartbeat goroutine.
	var wg sync.WaitGroup
	wg.Add(1)
	go func() {
		defer wg.Done()
		s.heartbeatLoop(runCtx)
	}()

	// 5) Main loop: consume assignments + persona requests until ctx done.
	sem := make(chan struct{}, s.opts.MaxConcurrentSegments)
	loopErr := s.consumeLoop(runCtx, assignCh, sem, &wg)

	runCancel()
	wg.Wait()
	// Best-effort terminal state. If we're exiting because ctx was
	// cancelled, mark Offline. If already Failed, keep Failed.
	if loopErr != nil && !errors.Is(loopErr, context.Canceled) && !errors.Is(loopErr, context.DeadlineExceeded) {
		if s.machine.State() != state.Failed {
			_ = s.machine.Transition(state.Failed)
		}
		s.log.Error("secondary: exiting with error", "err", loopErr)
		return loopErr
	}
	// Graceful shutdown path: only emit Offline if we're currently in a
	// non-terminal, non-processing state.
	if cur := s.machine.State(); cur == state.Available || cur == state.Syncing {
		_ = s.machine.Transition(state.Offline)
	}
	s.log.Info("secondary: stopped", "final_state", s.machine.State())
	return ctx.Err()
}

// ApplyPersona enqueues a persona-apply request. Processed by the main
// loop when the node is Available. Returns immediately; errors from the
// PersonaApplier surface as a Training→Failed transition.
func (s *Secondary) ApplyPersona(persona string) error {
	select {
	case s.personaQ <- persona:
		return nil
	default:
		return errors.New("secondary: persona queue full")
	}
}

// Cancel implements transport.Secondary. It is the handler the Primary
// calls when a segment (or whole job) must be aborted. Idempotent:
// cancelling an unknown segment is not an error.
func (s *Secondary) Cancel(ctx context.Context, req transport.CancelRequest) error {
	s.mu.Lock()
	cancelFn, ok := s.jobs[req.SegmentID]
	if ok {
		delete(s.jobs, req.SegmentID)
	}
	s.mu.Unlock()
	if ok {
		s.log.Info("secondary: cancel received", "job", req.JobID, "segment", req.SegmentID, "reason", string(req.Reason))
		cancelFn()
	} else {
		s.log.Debug("secondary: cancel for unknown segment (idempotent)", "segment", req.SegmentID)
	}
	return nil
}

// heartbeatLoop ticks at s.hbEvery, emitting a snapshot until ctx is done.
func (s *Secondary) heartbeatLoop(ctx context.Context) {
	t := time.NewTicker(s.hbEvery)
	defer t.Stop()
	for {
		select {
		case <-ctx.Done():
			s.log.Debug("secondary: heartbeat loop stopped")
			return
		case <-t.C:
			hb := transport.Heartbeat{
				NodeID:     s.opts.Identity.ID,
				State:      s.machine.State(),
				CurrentLPU: s.opts.Identity.AdvertisedLPU,
				JobIDs:     s.activeJobIDs(),
			}
			if err := s.opts.Primary.Heartbeat(ctx, hb); err != nil {
				s.log.Debug("secondary: heartbeat delivery failed", "err", err)
			}
		}
	}
}

func (s *Secondary) activeJobIDs() []string {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(s.jobs) == 0 {
		return nil
	}
	out := make([]string, 0, len(s.jobs))
	for id := range s.jobs {
		out = append(out, id)
	}
	return out
}

// consumeLoop processes assignments and persona requests. Returns when
// ctx is done or when the assignment channel closes unexpectedly.
func (s *Secondary) consumeLoop(ctx context.Context, assignCh <-chan transport.AssignSegment, sem chan struct{}, wg *sync.WaitGroup) error {
	for {
		select {
		case <-ctx.Done():
			return ctx.Err()

		case persona := <-s.personaQ:
			if err := s.applyPersona(ctx, persona); err != nil {
				return err
			}

		case a, ok := <-assignCh:
			if !ok {
				// Primary closed the stream.
				return errors.New("secondary: assignment stream closed")
			}
			// Only accept assignments while Available; otherwise surface
			// the orchestrator bug immediately.
			if !s.machine.State().IsAssignable() {
				_ = s.opts.Primary.SendSegmentEvent(ctx, transport.SegmentEvent{
					JobID:     a.JobID,
					SegmentID: a.SegmentID,
					Kind:      transport.SegmentError,
					Detail:    fmt.Sprintf("node in state %s cannot accept assignments", s.machine.State()),
				})
				continue
			}
			// Acquire semaphore slot.
			select {
			case sem <- struct{}{}:
			case <-ctx.Done():
				return ctx.Err()
			}
			wg.Add(1)
			go func(a transport.AssignSegment) {
				defer wg.Done()
				defer func() { <-sem }()
				s.runSegment(ctx, a)
			}(a)
		}
	}
}

// runSegment drives a single assignment through Processing and back to
// Available. Per-segment errors are reported as SegmentError events but
// do NOT take the node Failed (the spec reserves Failed for node-level
// faults such as sync failure).
func (s *Secondary) runSegment(parent context.Context, a transport.AssignSegment) {
	s.log.Info("secondary: segment starting", "job", a.JobID, "segment", a.SegmentID, "model", a.Model, "prompt_len", len(a.Prompt))
	start := time.Now()
	if err := s.machine.Transition(state.Processing); err != nil {
		s.log.Warn("secondary: cannot transition to Processing", "err", err)
		_ = s.opts.Primary.SendSegmentEvent(parent, transport.SegmentEvent{
			JobID:     a.JobID,
			SegmentID: a.SegmentID,
			Kind:      transport.SegmentError,
			Detail:    err.Error(),
		})
		return
	}
	var eventCount int
	defer func() {
		// Return to Available regardless of outcome.
		if cur := s.machine.State(); cur == state.Processing {
			_ = s.machine.Transition(state.Available)
		}
		s.log.Info("secondary: segment finished", "job", a.JobID, "segment", a.SegmentID, "events", eventCount, "duration", time.Since(start))
	}()

	segCtx, cancelFn := context.WithCancel(parent)
	s.mu.Lock()
	s.jobs[a.SegmentID] = cancelFn
	s.mu.Unlock()
	defer func() {
		s.mu.Lock()
		delete(s.jobs, a.SegmentID)
		s.mu.Unlock()
		cancelFn()
	}()

	events, err := s.opts.Executor.Execute(segCtx, a)
	if err != nil {
		s.log.Error("secondary: executor failed", "job", a.JobID, "segment", a.SegmentID, "err", err)
		_ = s.opts.Primary.SendSegmentEvent(parent, transport.SegmentEvent{
			JobID:     a.JobID,
			SegmentID: a.SegmentID,
			Kind:      transport.SegmentError,
			Detail:    err.Error(),
		})
		return
	}
	for ev := range events {
		// Ensure correlation IDs are populated even if the executor
		// forgot — downstream consumers rely on them.
		if ev.JobID == "" {
			ev.JobID = a.JobID
		}
		if ev.SegmentID == "" {
			ev.SegmentID = a.SegmentID
		}
		eventCount++
		if err := s.opts.Primary.SendSegmentEvent(parent, ev); err != nil {
			// Transport failure mid-stream: abandon the rest; the
			// orchestrator will notice via its own bookkeeping.
			s.log.Warn("secondary: event delivery failed; abandoning segment", "job", a.JobID, "segment", a.SegmentID, "err", err)
			return
		}
	}
}

func (s *Secondary) applyPersona(ctx context.Context, persona string) error {
	s.log.Info("secondary: applying persona", "persona", persona)
	if err := s.machine.Transition(state.Training); err != nil {
		// If we're not Available (e.g. mid-segment), drop the request;
		// the orchestrator may retry. Not a fatal error.
		s.log.Warn("secondary: cannot enter Training; persona request dropped", "err", err)
		return nil
	}
	if s.opts.PersonaApplier != nil {
		if err := s.opts.PersonaApplier.Apply(ctx, persona); err != nil {
			_ = s.machine.Transition(state.Failed)
			s.log.Error("secondary: persona applier failed", "persona", persona, "err", err)
			return fmt.Errorf("secondary: apply persona: %w", err)
		}
	}
	s.opts.Identity.Persona = persona
	if err := s.machine.Transition(state.Available); err != nil {
		return fmt.Errorf("secondary: leave Training: %w", err)
	}
	s.log.Info("secondary: persona applied", "persona", persona)
	return nil
}

// Compile-time assertion that *Secondary satisfies transport.Secondary.
var _ transport.Secondary = (*Secondary)(nil)

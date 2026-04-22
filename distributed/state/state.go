// Package state implements the formal state machine for Secondary node
// lifecycle in the distributed MPI-style framework.
//
// The state set and transition rules are derived directly from
// DISTRIBUTED_ARCHITECTURE.md §4. Only Available nodes are eligible for
// job assignment. The machine is safe for concurrent use by multiple
// goroutines and notifies registered listeners on every successful
// transition.
package state

import (
	"errors"
	"fmt"
	"log/slog"
	"sync"
)

// State is a Secondary node's lifecycle state.
type State string

const (
	// Available: ready to execute jobs. Only Available nodes receive
	// assignments.
	Available State = "Available"
	// Syncing: synchronizing models; not eligible for jobs.
	Syncing State = "Syncing"
	// Failed: sync failure or other error. Triggers job cancellation on
	// any job this node is assigned to (see distributed/cancel).
	Failed State = "Failed"
	// Offline: connectivity lost.
	Offline State = "Offline"
	// ProcessingThinking: in the thinking phase of a job.
	ProcessingThinking State = "Processing-Thinking"
	// ProcessingReasoning: in the reasoning phase of a job.
	ProcessingReasoning State = "Processing-Reasoning"
	// Processing: running a job (no thinking/reasoning).
	Processing State = "Processing"
	// Training: learning a new persona.
	Training State = "Training"
)

// All returns every defined state, in canonical order.
func All() []State {
	return []State{
		Available,
		Syncing,
		Failed,
		Offline,
		ProcessingThinking,
		ProcessingReasoning,
		Processing,
		Training,
	}
}

// String implements fmt.Stringer.
func (s State) String() string { return string(s) }

// Valid reports whether s is one of the recognized states.
func (s State) Valid() bool {
	for _, v := range All() {
		if s == v {
			return true
		}
	}
	return false
}

// IsProcessing reports whether the state is any of the three processing
// variants. Useful to callers that only care "is this node running a job?".
func (s State) IsProcessing() bool {
	switch s {
	case Processing, ProcessingThinking, ProcessingReasoning:
		return true
	default:
		return false
	}
}

// IsAssignable reports whether the orchestrator may schedule new segments
// onto a node currently in this state. Per the spec only Available is
// assignable.
func (s State) IsAssignable() bool { return s == Available }

// IsTerminal reports whether the state means "cannot participate in a
// collective right now without external intervention". Failed and Offline
// are terminal from the scheduler's perspective; recovery requires either
// a reconnect (Offline → Syncing → Available) or operator remediation
// (Failed → Syncing → Available).
func (s State) IsTerminal() bool {
	return s == Failed || s == Offline
}

// transitions is the declarative transition table. A state `from` may move
// to any of `transitions[from]`. Edges not in the table are rejected by
// Machine.Transition.
//
// Rationale:
//   - From Available: node can begin syncing (model update), start a job
//     (any of the three Processing variants), start training a persona,
//     fail outright, or drop offline.
//   - From any Processing* state: the node finishes the job (→ Available),
//     fails (→ Failed), or drops (→ Offline). Direct cross-processing
//     transitions are NOT allowed — a job completes before the next stage.
//   - From Syncing: completes (→ Available), fails (→ Failed), or drops
//     (→ Offline).
//   - From Training: completes (→ Available), fails, or drops.
//   - From Failed: only Syncing (operator remediates and restarts sync).
//   - From Offline: only Syncing (reconnect path must re-sync models).
var transitions = map[State][]State{
	Available: {
		Syncing, ProcessingThinking, ProcessingReasoning, Processing, Training, Failed, Offline,
	},
	Syncing: {
		Available, Failed, Offline,
	},
	Failed: {
		Syncing,
	},
	Offline: {
		Syncing,
	},
	ProcessingThinking: {
		Available, Failed, Offline,
	},
	ProcessingReasoning: {
		Available, Failed, Offline,
	},
	Processing: {
		Available, Failed, Offline,
	},
	Training: {
		Available, Failed, Offline,
	},
}

// CanTransition reports whether a direct edge from → to is allowed by the
// lifecycle spec. Idempotent transitions (from == to) are allowed for every
// valid state — they are no-ops but the machine reports them as valid to
// simplify callers that assert "the node is now in state X" without
// tracking whether X differs from the current state.
func CanTransition(from, to State) bool {
	if !from.Valid() || !to.Valid() {
		return false
	}
	if from == to {
		return true
	}
	for _, allowed := range transitions[from] {
		if allowed == to {
			return true
		}
	}
	return false
}

// ErrInvalidTransition is returned by Machine.Transition when the edge is
// rejected by the lifecycle spec.
var ErrInvalidTransition = errors.New("distributed/state: invalid transition")

// InvalidTransitionError carries the from/to states for diagnostics while
// still unwrapping to ErrInvalidTransition.
type InvalidTransitionError struct {
	From, To State
}

func (e *InvalidTransitionError) Error() string {
	return fmt.Sprintf("distributed/state: invalid transition %s → %s", e.From, e.To)
}

func (e *InvalidTransitionError) Unwrap() error { return ErrInvalidTransition }

// Listener is invoked on every successful (non-idempotent) transition. It
// runs synchronously under the machine's lock; listeners MUST NOT call
// back into the Machine or perform blocking I/O. Use a channel or
// goroutine if heavier work is needed.
type Listener func(from, to State)

// Machine is a thread-safe state holder with transition validation and
// listener fan-out. Zero value is NOT ready — use New.
type Machine struct {
	mu        sync.RWMutex
	current   State
	nextID    uint64
	listeners map[uint64]Listener
}

// New constructs a Machine starting in the given initial state. Returns an
// error if the initial state is not valid.
func New(initial State) (*Machine, error) {
	if !initial.Valid() {
		return nil, fmt.Errorf("distributed/state: invalid initial state %q", initial)
	}
	return &Machine{current: initial, listeners: make(map[uint64]Listener)}, nil
}

// State returns the current state. Snapshot semantics — the returned value
// may be stale by the time the caller uses it.
func (m *Machine) State() State {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.current
}

// Transition moves the machine from its current state to `to`. Returns
// InvalidTransitionError if the edge is rejected. Idempotent transitions
// succeed silently and do NOT notify listeners.
func (m *Machine) Transition(to State) error {
	m.mu.Lock()
	from := m.current
	if !CanTransition(from, to) {
		m.mu.Unlock()
		slog.Warn("distributed/state: invalid transition rejected", "from", from, "to", to)
		return &InvalidTransitionError{From: from, To: to}
	}
	if from == to {
		m.mu.Unlock()
		return nil
	}
	m.current = to
	snapshot := make([]Listener, 0, len(m.listeners))
	for _, l := range m.listeners {
		snapshot = append(snapshot, l)
	}
	m.mu.Unlock()
	slog.Debug("distributed/state: transition", "from", from, "to", to, "listeners", len(snapshot))
	for _, l := range snapshot {
		l(from, to)
	}
	return nil
}

// AddListener registers a Listener. Returns a function that removes the
// listener when invoked; safe to call from any goroutine and idempotent.
func (m *Machine) AddListener(l Listener) func() {
	if l == nil {
		return func() {}
	}
	m.mu.Lock()
	id := m.nextID
	m.nextID++
	m.listeners[id] = l
	m.mu.Unlock()

	var once sync.Once
	return func() {
		once.Do(func() {
			m.mu.Lock()
			delete(m.listeners, id)
			m.mu.Unlock()
		})
	}
}

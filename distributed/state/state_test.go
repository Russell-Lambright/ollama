package state

import (
	"errors"
	"sync"
	"sync/atomic"
	"testing"
)

func TestAllValid(t *testing.T) {
	for _, s := range All() {
		if !s.Valid() {
			t.Errorf("All() returned invalid state %q", s)
		}
	}
	if State("nope").Valid() {
		t.Errorf("unrecognized state should not be Valid()")
	}
}

func TestStateHelpers(t *testing.T) {
	assignable := []State{Available}
	processing := []State{Processing, ProcessingThinking, ProcessingReasoning}
	terminal := []State{Failed, Offline}

	for _, s := range All() {
		gotAssign := contains(assignable, s)
		if s.IsAssignable() != gotAssign {
			t.Errorf("IsAssignable(%s)=%v want %v", s, s.IsAssignable(), gotAssign)
		}
		gotProc := contains(processing, s)
		if s.IsProcessing() != gotProc {
			t.Errorf("IsProcessing(%s)=%v want %v", s, s.IsProcessing(), gotProc)
		}
		gotTerm := contains(terminal, s)
		if s.IsTerminal() != gotTerm {
			t.Errorf("IsTerminal(%s)=%v want %v", s, s.IsTerminal(), gotTerm)
		}
	}
}

func contains(ss []State, want State) bool {
	for _, s := range ss {
		if s == want {
			return true
		}
	}
	return false
}

func TestCanTransition(t *testing.T) {
	// Spot-check the key spec rules.
	good := [][2]State{
		{Available, Syncing},
		{Available, Processing},
		{Available, ProcessingThinking},
		{Available, ProcessingReasoning},
		{Available, Training},
		{Available, Failed},
		{Available, Offline},
		{Syncing, Available},
		{Syncing, Failed},
		{Processing, Available},
		{ProcessingThinking, Available},
		{ProcessingReasoning, Available},
		{Training, Available},
		{Failed, Syncing},
		{Offline, Syncing},
	}
	for _, e := range good {
		if !CanTransition(e[0], e[1]) {
			t.Errorf("CanTransition(%s,%s)=false want true", e[0], e[1])
		}
	}
	bad := [][2]State{
		{Available, Available}, // idempotent is allowed, check later
		{Processing, Training},
		{Processing, Syncing},
		{ProcessingThinking, ProcessingReasoning}, // cross-processing forbidden
		{Failed, Available},                       // must re-sync first
		{Offline, Available},                      // must re-sync first
		{Failed, Offline},
		{State("bogus"), Available},
		{Available, State("bogus")},
	}
	for _, e := range bad {
		if e[0] == Available && e[1] == Available {
			// idempotent must be allowed
			if !CanTransition(e[0], e[1]) {
				t.Errorf("idempotent transition unexpectedly rejected")
			}
			continue
		}
		if CanTransition(e[0], e[1]) {
			t.Errorf("CanTransition(%s,%s)=true want false", e[0], e[1])
		}
	}
}

func TestMachineBasic(t *testing.T) {
	if _, err := New(State("bogus")); err == nil {
		t.Fatalf("expected error for invalid initial state")
	}
	m, err := New(Syncing)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	if m.State() != Syncing {
		t.Fatalf("initial state: got %s want %s", m.State(), Syncing)
	}
	if err := m.Transition(Available); err != nil {
		t.Fatalf("Syncing→Available: %v", err)
	}
	if m.State() != Available {
		t.Fatalf("post-transition state: got %s", m.State())
	}
	// Idempotent.
	if err := m.Transition(Available); err != nil {
		t.Fatalf("idempotent: %v", err)
	}
	// Invalid edge.
	err = m.Transition(Failed) // allowed: Available→Failed
	if err != nil {
		t.Fatalf("Available→Failed should succeed: %v", err)
	}
	err = m.Transition(Available) // disallowed: Failed→Available
	if err == nil {
		t.Fatalf("Failed→Available should be rejected")
	}
	var ite *InvalidTransitionError
	if !errors.As(err, &ite) {
		t.Fatalf("error is not InvalidTransitionError: %T %v", err, err)
	}
	if ite.From != Failed || ite.To != Available {
		t.Fatalf("InvalidTransitionError fields: from=%s to=%s", ite.From, ite.To)
	}
	if !errors.Is(err, ErrInvalidTransition) {
		t.Fatalf("errors.Is(ErrInvalidTransition) failed")
	}
	// State unchanged after a rejected transition.
	if m.State() != Failed {
		t.Fatalf("state changed despite rejected transition: %s", m.State())
	}
}

func TestMachineListeners(t *testing.T) {
	m, err := New(Available)
	if err != nil {
		t.Fatal(err)
	}
	var hits int32
	var seenFrom, seenTo State
	remove := m.AddListener(func(from, to State) {
		atomic.AddInt32(&hits, 1)
		seenFrom, seenTo = from, to
	})
	// Idempotent: no notification.
	_ = m.Transition(Available)
	if got := atomic.LoadInt32(&hits); got != 0 {
		t.Fatalf("idempotent transition fired listener: hits=%d", got)
	}
	// Real transition: notified.
	_ = m.Transition(Syncing)
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("expected 1 hit, got %d", got)
	}
	if seenFrom != Available || seenTo != Syncing {
		t.Fatalf("listener saw %s→%s", seenFrom, seenTo)
	}
	// Remove and confirm no further hits.
	remove()
	_ = m.Transition(Available)
	if got := atomic.LoadInt32(&hits); got != 1 {
		t.Fatalf("removed listener fired: hits=%d", got)
	}
	// Double-remove is safe.
	remove()
}

func TestAddNilListener(t *testing.T) {
	m, _ := New(Available)
	// Must not panic and must not keep a slot.
	m.AddListener(nil)()
}

func TestMachineConcurrency(t *testing.T) {
	m, _ := New(Available)
	var hits int32
	m.AddListener(func(from, to State) { atomic.AddInt32(&hits, 1) })

	// Multiple listeners removed/added concurrently while transitions run.
	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			rm := m.AddListener(func(from, to State) {})
			rm()
		}()
	}
	for i := 0; i < 50; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			_ = m.Transition(Syncing)
			_ = m.Transition(Available)
		}()
	}
	wg.Wait()
	if atomic.LoadInt32(&hits) == 0 {
		t.Fatalf("expected at least one listener hit")
	}
}

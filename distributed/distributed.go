// Package distributed is the root of the MPI-style distributed Ollama
// framework. See distributed/README.md and ../DISTRIBUTED_ARCHITECTURE.md for
// the specification and phased roadmap.
//
// This package is feature-gated: unless the operator starts Ollama with
// --mode=primary or --mode=secondary (or sets OLLAMA_NODE_MODE), nothing in
// this tree should affect runtime behavior.
package distributed

// Mode describes the operational mode an Ollama process is running in.
//
// Standalone is the default and matches Ollama's historical behavior exactly;
// Primary and Secondary are reachable only through explicit opt-in
// (CLI flag or environment variable).
type Mode string

const (
	// ModeStandalone is the default. Distributed code paths are disabled.
	ModeStandalone Mode = "standalone"
	// ModePrimary starts the orchestrator. It does not execute inference
	// jobs in collective mode — it schedules, correlates, and QA-checks.
	ModePrimary Mode = "primary"
	// ModeSecondary starts a worker node that attaches to a Primary.
	ModeSecondary Mode = "secondary"
)

// Valid reports whether the mode is one of the recognized values.
func (m Mode) Valid() bool {
	switch m {
	case ModeStandalone, ModePrimary, ModeSecondary:
		return true
	default:
		return false
	}
}

// String implements fmt.Stringer.
func (m Mode) String() string { return string(m) }

// ParseMode parses a human-supplied mode string. Empty input resolves to
// ModeStandalone. Unknown input returns a non-nil error; callers must decide
// whether to reject or fall back to standalone.
func ParseMode(s string) (Mode, error) {
	switch s {
	case "", string(ModeStandalone):
		return ModeStandalone, nil
	case string(ModePrimary):
		return ModePrimary, nil
	case string(ModeSecondary):
		return ModeSecondary, nil
	default:
		return ModeStandalone, errInvalidMode(s)
	}
}

type errInvalidMode string

func (e errInvalidMode) Error() string {
	return "distributed: invalid mode " + string(e) + " (expected standalone, primary, or secondary)"
}

// Package cancel defines the cancellation contract for distributed jobs.
//
// Two cancellation sources are supported, per the spec:
//
//  1. CallerCancelled — the original caller disconnected or cancelled its
//     context. The orchestrator must propagate a cancel signal to every
//     Secondary currently executing a segment of this job so those nodes can
//     abandon their work and return to the Available pool.
//
//  2. NodeFailed — one of the Secondaries assigned to the job entered the
//     Failed state (sync failure, crash, transport error, etc.). The
//     orchestrator must cancel the whole job: peer segments running on other
//     Secondaries are aborted, those Secondaries are returned to the pool,
//     and the caller receives an error.
//
// This package intentionally holds no runtime code in Phase 1 — only the
// types and enums. The orchestrator (Phase 5) and transport (Phase 2) wire
// the actual propagation:
//
//   - Transport layer owns a per-segment cancel RPC (gRPC bidirectional
//     stream close, or SSE connection abort for HTTP/2).
//   - Orchestrator watches the caller's context AND every assigned node's
//     state transitions; on either trigger it cancels the shared job
//     context, which fans out to every in-flight segment.
package cancel

// Reason classifies why a distributed job was cancelled. It is recorded in
// logs and surfaced to callers so failures are distinguishable from
// voluntary cancels.
type Reason string

const (
	// ReasonCallerCancelled is set when the caller's context is cancelled
	// (client disconnect, explicit cancel, timeout). Secondaries assigned
	// to the job MUST be signalled and returned to the Available pool.
	ReasonCallerCancelled Reason = "caller_cancelled"

	// ReasonNodeFailed is set when any Secondary assigned to the job
	// transitions to the Failed state. The whole job is aborted and all
	// peer Secondaries are returned to the pool.
	ReasonNodeFailed Reason = "node_failed"

	// ReasonOrchestratorShutdown is set when the Primary is shutting down
	// and must tear down in-flight jobs.
	ReasonOrchestratorShutdown Reason = "orchestrator_shutdown"
)

// Event is the value published by the orchestrator when a job is cancelled.
// Downstream consumers (transport, UI, logs) subscribe to a stream of these.
type Event struct {
	// JobID identifies the cancelled job.
	JobID string
	// Reason is why the job was cancelled.
	Reason Reason
	// NodeID is populated when Reason == ReasonNodeFailed and names the
	// node whose failure triggered the cancellation.
	NodeID string
	// Detail is a short human-readable message suitable for logs or UI.
	Detail string
}

// String implements fmt.Stringer.
func (r Reason) String() string { return string(r) }

// Valid reports whether the reason is one of the recognized values.
func (r Reason) Valid() bool {
	switch r {
	case ReasonCallerCancelled, ReasonNodeFailed, ReasonOrchestratorShutdown:
		return true
	default:
		return false
	}
}

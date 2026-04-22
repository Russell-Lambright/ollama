// Package modelsync implements model fan-out and manifest-diff syncing
// for the distributed MPI-style framework.
//
// Scope (Phase 6):
//
//   - Manifest data model: a set of (Name, Digest, SizeBytes) tuples
//     describing what models a node has on disk.
//   - Diff: given an "expected" manifest (the Primary's authoritative
//     list) and an "observed" manifest (what the node currently has),
//     produce an ordered plan of pulls, redownloads, and no-ops.
//   - Syncer: a pluggable type that satisfies secondary.ModelSyncer; it
//     consults a ManifestProvider + ModelPuller and converges the
//     node's manifest toward the expected set.
//   - Fanout: a primary-side helper that drives a Syncer-equivalent
//     operation across every node in a collective (used by the
//     orchestrator when starting a fresh job that requires a model no
//     node has yet, and by the UI's collective-wide actions).
//
// This package is intentionally transport-agnostic: the actual model
// pull is delegated to a ModelPuller, which in production will wrap
// Ollama's existing pull/registry pipeline. Keeping that contract narrow
// lets us unit-test the sync logic against an in-memory fake.
package modelsync

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sort"
	"strings"
	"sync"
)

// Entry is one model in a manifest.
type Entry struct {
	// Name is the model reference (e.g. "qwen2.5" or "llama3:8b-q4").
	Name string
	// Digest is the content-addressed identifier (e.g. sha256 hash) of
	// the model blob. Two entries with the same Name but different
	// Digest are the same model at different versions.
	Digest string
	// SizeBytes is the on-disk size; used only for logging/metrics.
	SizeBytes int64
}

// Validate returns an error if the entry is malformed.
func (e Entry) Validate() error {
	if strings.TrimSpace(e.Name) == "" {
		return errors.New("modelsync: entry Name is required")
	}
	if strings.TrimSpace(e.Digest) == "" {
		return errors.New("modelsync: entry Digest is required")
	}
	return nil
}

// Manifest is a set of model entries keyed by Name. Two manifests are
// "equal" when every Name maps to the same Digest.
type Manifest struct {
	// Entries by Name. Digests may differ between two manifests with the
	// same set of names.
	Entries map[string]Entry
}

// NewManifest builds a Manifest from a slice of entries. Duplicates (by
// Name) are resolved by last-wins.
func NewManifest(entries ...Entry) Manifest {
	m := Manifest{Entries: make(map[string]Entry, len(entries))}
	for _, e := range entries {
		m.Entries[e.Name] = e
	}
	return m
}

// Names returns the sorted list of model names in the manifest.
func (m Manifest) Names() []string {
	out := make([]string, 0, len(m.Entries))
	for n := range m.Entries {
		out = append(out, n)
	}
	sort.Strings(out)
	return out
}

// Op is the kind of action a sync Plan prescribes for a single model.
type Op string

const (
	// OpNoop: the observed manifest already matches the expected entry.
	OpNoop Op = "noop"
	// OpPull: the model is absent from the observed manifest and must
	// be pulled fresh.
	OpPull Op = "pull"
	// OpRedownload: the model is present but at a different Digest
	// (version drift); pull to overwrite.
	OpRedownload Op = "redownload"
)

// PlanItem is one step in a sync plan.
type PlanItem struct {
	// Op is the action to take for Entry.Name.
	Op Op
	// Want is the target entry (populated for Pull, Redownload, Noop).
	Want Entry
	// Have is the observed entry (populated for Redownload, Noop; zero
	// for Pull).
	Have Entry
}

// Plan is an ordered list of sync operations. Ordering is deterministic
// (by Name) so repeated invocations produce identical plans, which
// simplifies logging and reasoning about retries.
type Plan struct {
	Items []PlanItem
}

// Actionable returns only the Plan items that require work (Pull,
// Redownload). Convenient for callers that only want to drive the
// ModelPuller.
func (p Plan) Actionable() []PlanItem {
	out := make([]PlanItem, 0, len(p.Items))
	for _, it := range p.Items {
		if it.Op != OpNoop {
			out = append(out, it)
		}
	}
	return out
}

// IsEmpty reports whether the plan requires no work.
func (p Plan) IsEmpty() bool { return len(p.Actionable()) == 0 }

// Diff computes the sync plan that converges `observed` toward
// `expected`. Entries present in observed but absent from expected are
// intentionally ignored — nodes may legitimately hold extra models, and
// the framework has no authority to delete them.
func Diff(expected, observed Manifest) Plan {
	items := make([]PlanItem, 0, len(expected.Entries))
	for _, name := range expected.Names() {
		want := expected.Entries[name]
		have, ok := observed.Entries[name]
		switch {
		case !ok:
			items = append(items, PlanItem{Op: OpPull, Want: want})
		case have.Digest != want.Digest:
			items = append(items, PlanItem{Op: OpRedownload, Want: want, Have: have})
		default:
			items = append(items, PlanItem{Op: OpNoop, Want: want, Have: have})
		}
	}
	return Plan{Items: items}
}

// ---------------------------------------------------------------------
// Syncer
// ---------------------------------------------------------------------

// ManifestProvider returns the expected and observed manifests at the
// moment the Syncer runs. Implementations plug in to whatever source
// of truth exists on the Primary (catalog, config, etc.) and whatever
// on-disk inspection the node performs.
type ManifestProvider interface {
	Expected(ctx context.Context) (Manifest, error)
	Observed(ctx context.Context) (Manifest, error)
}

// ManifestProviderFunc adapts a pair of plain functions to the
// ManifestProvider interface.
type ManifestProviderFunc struct {
	ExpectedFn func(ctx context.Context) (Manifest, error)
	ObservedFn func(ctx context.Context) (Manifest, error)
}

// Expected implements ManifestProvider.
func (f ManifestProviderFunc) Expected(ctx context.Context) (Manifest, error) {
	return f.ExpectedFn(ctx)
}

// Observed implements ManifestProvider.
func (f ManifestProviderFunc) Observed(ctx context.Context) (Manifest, error) {
	return f.ObservedFn(ctx)
}

// ModelPuller performs a single pull operation. In production this wraps
// Ollama's existing pull/registry pipeline; in tests it's a thin fake.
type ModelPuller interface {
	Pull(ctx context.Context, entry Entry) error
}

// ModelPullerFunc adapts a plain function to the ModelPuller interface.
type ModelPullerFunc func(ctx context.Context, entry Entry) error

// Pull implements ModelPuller.
func (f ModelPullerFunc) Pull(ctx context.Context, entry Entry) error { return f(ctx, entry) }

// Options configures a Syncer.
type Options struct {
	// Provider supplies expected + observed manifests. Required.
	Provider ManifestProvider
	// Puller executes pulls. Required.
	Puller ModelPuller
	// MaxAttempts caps retries per item. Zero → 1 (no retry).
	MaxAttempts int
}

// Syncer implements secondary.ModelSyncer. It is safe for concurrent
// callers; each Sync call computes a fresh plan.
type Syncer struct {
	opts Options
	log  *slog.Logger
}

// New constructs a Syncer.
func New(opts Options) (*Syncer, error) {
	if opts.Provider == nil {
		return nil, errors.New("modelsync: Provider is required")
	}
	if opts.Puller == nil {
		return nil, errors.New("modelsync: Puller is required")
	}
	if opts.MaxAttempts <= 0 {
		opts.MaxAttempts = 1
	}
	return &Syncer{opts: opts, log: slog.With("component", "distributed/modelsync")}, nil
}

// Sync implements secondary.ModelSyncer. It fetches expected/observed,
// diffs them, and invokes the puller for every actionable item. Returns
// the first unrecoverable error; individual items are retried up to
// MaxAttempts.
func (s *Syncer) Sync(ctx context.Context) error {
	want, err := s.opts.Provider.Expected(ctx)
	if err != nil {
		s.log.Error("modelsync: fetch expected manifest failed", "err", err)
		return fmt.Errorf("modelsync: expected: %w", err)
	}
	have, err := s.opts.Provider.Observed(ctx)
	if err != nil {
		s.log.Error("modelsync: fetch observed manifest failed", "err", err)
		return fmt.Errorf("modelsync: observed: %w", err)
	}
	plan := Diff(want, have)
	if plan.IsEmpty() {
		s.log.Info("modelsync: already in sync", "models", len(want.Entries))
		return nil
	}
	work := plan.Actionable()
	s.log.Info("modelsync: sync starting", "items", len(work), "expected", len(want.Entries), "observed", len(have.Entries))
	for _, item := range work {
		if err := s.pullWithRetry(ctx, item); err != nil {
			s.log.Error("modelsync: item failed", "name", item.Want.Name, "op", string(item.Op), "err", err)
			return err
		}
	}
	s.log.Info("modelsync: sync complete", "items", len(work))
	return nil
}

func (s *Syncer) pullWithRetry(ctx context.Context, item PlanItem) error {
	var lastErr error
	for attempt := 1; attempt <= s.opts.MaxAttempts; attempt++ {
		if err := ctx.Err(); err != nil {
			return err
		}
		s.log.Info("modelsync: pulling model",
			"name", item.Want.Name,
			"digest", item.Want.Digest,
			"op", string(item.Op),
			"attempt", attempt,
			"max_attempts", s.opts.MaxAttempts,
		)
		err := s.opts.Puller.Pull(ctx, item.Want)
		if err == nil {
			s.log.Info("modelsync: pull succeeded", "name", item.Want.Name, "digest", item.Want.Digest)
			return nil
		}
		lastErr = err
		s.log.Warn("modelsync: pull failed",
			"name", item.Want.Name,
			"attempt", attempt,
			"max_attempts", s.opts.MaxAttempts,
			"err", err,
		)
	}
	return fmt.Errorf("modelsync: pull %s after %d attempts: %w", item.Want.Name, s.opts.MaxAttempts, lastErr)
}

// ---------------------------------------------------------------------
// Fan-out across a collective
// ---------------------------------------------------------------------

// FanoutResult is the outcome of a single node's sync during a fan-out.
type FanoutResult struct {
	NodeID string
	Err    error
}

// Fanout drives `sync` concurrently across the provided node IDs. Each
// node gets its own copy of the ctx; a failure on one node does NOT
// cancel the others (matches the spec's collective-wide persona
// application: "individual nodes apply independently, so a failure on
// one node does not roll back the others").
//
// The returned slice is in the same order as `nodeIDs`. The `trigger`
// callback is invoked once per node and is expected to drive that
// node's sync path (typically by forwarding a command over the
// transport, or by invoking the node's local Syncer in-process tests).
func Fanout(ctx context.Context, nodeIDs []string, trigger func(ctx context.Context, nodeID string) error) []FanoutResult {
	log := slog.With("component", "distributed/modelsync")
	log.Info("modelsync: fanout starting", "nodes", len(nodeIDs))
	results := make([]FanoutResult, len(nodeIDs))
	var wg sync.WaitGroup
	for i, id := range nodeIDs {
		wg.Add(1)
		go func(i int, id string) {
			defer wg.Done()
			err := trigger(ctx, id)
			results[i] = FanoutResult{NodeID: id, Err: err}
			if err != nil {
				log.Warn("modelsync: fanout node failed", "node", id, "err", err)
			} else {
				log.Debug("modelsync: fanout node succeeded", "node", id)
			}
		}(i, id)
	}
	wg.Wait()
	failed := 0
	for _, r := range results {
		if r.Err != nil {
			failed++
		}
	}
	log.Info("modelsync: fanout complete", "nodes", len(nodeIDs), "failed", failed)
	return results
}

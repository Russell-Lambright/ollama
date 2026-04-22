package ml

import (
	"context"
	"errors"
	"sync"
)

// ErrNoWeightLoader is returned when a caller tries to resolve a weight
// loader that has not been registered.
var ErrNoWeightLoader = errors.New("ml: no weight loader registered")

// WeightLoader abstracts the source of tensor bytes for a single loaded
// model. Implementations may keep the bytes in RAM, stream them from a
// VHDX virtual disk, fetch them over a remote transport, or any
// combination of the above.
//
// The interface is deliberately minimal so that it can be implemented in
// pure Go and exposed to the C-side of the inference engine (ggml /
// llama.cpp) via a thin CGO bridge. ReadLayer fills dst with the bytes
// of the given layer starting at offset. Implementations must:
//
//   - return len(dst) on success;
//   - return (0, [ErrNoWeightLoader]) when the index is unknown;
//   - be safe for concurrent use by multiple inference threads;
//   - not retain dst beyond the call (it may be a backend-owned buffer).
//
// The interface obeys the Liskov Substitution Principle: any
// WeightLoader may replace any other without changing the observable
// behavior of the inference engine.
type WeightLoader interface {
	// ReadLayer copies size bytes of layer index starting at offset into
	// dst. The contract is equivalent to io.ReaderAt.ReadAt on a
	// per-layer virtual "file".
	ReadLayer(ctx context.Context, index int, offset int64, dst []byte) (int, error)

	// LayerSize returns the total size of the layer in bytes. A size of
	// zero means "unknown" and prevents prefetching/validation.
	LayerSize(index int) int64
}

// weightLoaderRegistry is a process-wide registry that maps a runner ID
// (typically the llama.cpp model handle or a Go-side UUID) to a
// WeightLoader. It lets the CGO bridge look up the loader for a given
// model without having to pass a Go pointer across the C boundary.
type weightLoaderRegistry struct {
	mu      sync.RWMutex
	loaders map[uint64]WeightLoader
	nextID  uint64
}

var globalWeightLoaders = &weightLoaderRegistry{
	loaders: make(map[uint64]WeightLoader),
}

// RegisterWeightLoader stores loader under a newly allocated handle and
// returns the handle. The handle is stable and may be safely passed
// across the CGO boundary as a uintptr/uint64 model-scoped token.
func RegisterWeightLoader(loader WeightLoader) uint64 {
	globalWeightLoaders.mu.Lock()
	defer globalWeightLoaders.mu.Unlock()
	globalWeightLoaders.nextID++
	id := globalWeightLoaders.nextID
	globalWeightLoaders.loaders[id] = loader
	return id
}

// UnregisterWeightLoader drops the loader under handle. Further ReadLayer
// lookups for handle return [ErrNoWeightLoader].
func UnregisterWeightLoader(handle uint64) {
	globalWeightLoaders.mu.Lock()
	defer globalWeightLoaders.mu.Unlock()
	delete(globalWeightLoaders.loaders, handle)
}

// LookupWeightLoader returns the loader registered under handle.
func LookupWeightLoader(handle uint64) (WeightLoader, error) {
	globalWeightLoaders.mu.RLock()
	defer globalWeightLoaders.mu.RUnlock()
	l, ok := globalWeightLoaders.loaders[handle]
	if !ok {
		return nil, ErrNoWeightLoader
	}
	return l, nil
}

// ResetWeightLoaders drops every registered loader. It is intended for
// test isolation; production code should call [UnregisterWeightLoader]
// instead.
func ResetWeightLoaders() {
	globalWeightLoaders.mu.Lock()
	defer globalWeightLoaders.mu.Unlock()
	globalWeightLoaders.loaders = make(map[uint64]WeightLoader)
	globalWeightLoaders.nextID = 0
}

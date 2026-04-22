// Package vhdx implements the Bllama VHDX Virtual Disk Memory Offloading
// Enhancement. It provides a storage abstraction that allows model layers
// to reside either in system RAM (hot layers) or on a VHDX virtual hard
// drive (cold layers) without the inference engine needing to distinguish
// between the two storage media.
//
// The design follows the Dependency Inversion Principle: higher-level
// inference code depends on the [StorageProvider] abstraction rather than
// on concrete RAM or disk implementations. Any StorageProvider must be
// substitutable for any other (Liskov Substitution) so that layer access
// logic remains transparent regardless of where a layer is stored.
package vhdx

import (
	"context"
	"errors"
	"fmt"
	"io"
	"sync"
)

// ErrLayerNotFound is returned when a requested layer index is not known
// to the storage provider.
var ErrLayerNotFound = errors.New("vhdx: layer not found")

// ErrStorageUnavailable is returned when a storage provider cannot satisfy
// a request because the underlying medium has failed or is not ready. It
// is used to signal callers that they should fall back to a secondary
// provider if one is available.
var ErrStorageUnavailable = errors.New("vhdx: storage unavailable")

// Layer is the minimal description of a model layer held by a
// [StorageProvider]. The inference engine does not care whether Data lives
// in RAM or on disk; it only cares that it can be read back identically.
type Layer struct {
	// Index is the zero-based layer index within the model.
	Index int
	// Size is the size of the layer in bytes.
	Size int64
	// Data is the raw bytes of the layer.
	Data []byte
}

// StorageProvider abstracts the medium that holds model layers. Concrete
// implementations may keep layers in RAM, on a VHDX virtual disk, or on a
// remote backend. Implementations must be safe for concurrent use.
//
// Implementations must obey the Liskov Substitution Principle: any
// StorageProvider can be swapped in for another without affecting the
// correctness of the inference engine. In particular, a Load call that
// succeeds must return bytes identical to those originally stored via
// Store for the same index.
type StorageProvider interface {
	// Name is a short human-readable identifier (e.g. "ram" or "vhdx")
	// that is surfaced by "ollama ps" and in logs.
	Name() string

	// Store persists a layer so that later calls to Load will return it.
	Store(ctx context.Context, layer Layer) error

	// Load returns the layer at the given index. It must return
	// [ErrLayerNotFound] when the index is unknown and
	// [ErrStorageUnavailable] when the medium has failed.
	Load(ctx context.Context, index int) (Layer, error)

	// Size returns the total number of logical bytes currently held by
	// the provider (not counting on-disk padding, if any).
	Size() int64

	// Has reports whether the provider currently holds the given layer.
	Has(index int) bool

	// Close releases any resources held by the provider.
	Close() error
}

// ReadWriteSeekCloser is the minimal interface used by file-backed
// providers such as [VHDXStorage]. It is factored out so tests (and
// future non-file backends) can provide in-memory substitutes without
// touching the disk.
type ReadWriteSeekCloser interface {
	io.ReaderAt
	io.WriterAt
	io.Closer
}

// RAMStorage is a [StorageProvider] backed by an in-memory map. It is
// used as the "hot" tier for layers that should not incur disk I/O.
type RAMStorage struct {
	mu     sync.RWMutex
	layers map[int]Layer
	bytes  int64
}

// NewRAMStorage constructs an empty [RAMStorage].
func NewRAMStorage() *RAMStorage {
	return &RAMStorage{layers: make(map[int]Layer)}
}

// Name returns "ram".
func (r *RAMStorage) Name() string { return "ram" }

// Store copies the layer bytes into the RAM map. The copy guards against
// callers mutating the underlying buffer after Store returns.
func (r *RAMStorage) Store(_ context.Context, layer Layer) error {
	if layer.Size < 0 {
		return fmt.Errorf("vhdx: negative layer size %d", layer.Size)
	}
	if int64(len(layer.Data)) != layer.Size {
		return fmt.Errorf("vhdx: layer %d declared size %d but has %d bytes", layer.Index, layer.Size, len(layer.Data))
	}
	buf := make([]byte, len(layer.Data))
	copy(buf, layer.Data)

	r.mu.Lock()
	defer r.mu.Unlock()
	if existing, ok := r.layers[layer.Index]; ok {
		r.bytes -= existing.Size
	}
	r.layers[layer.Index] = Layer{Index: layer.Index, Size: layer.Size, Data: buf}
	r.bytes += layer.Size
	return nil
}

// Load returns a defensive copy of the stored layer.
func (r *RAMStorage) Load(ctx context.Context, index int) (Layer, error) {
	if err := ctx.Err(); err != nil {
		return Layer{}, err
	}
	r.mu.RLock()
	layer, ok := r.layers[index]
	r.mu.RUnlock()
	if !ok {
		return Layer{}, ErrLayerNotFound
	}
	buf := make([]byte, len(layer.Data))
	copy(buf, layer.Data)
	return Layer{Index: layer.Index, Size: layer.Size, Data: buf}, nil
}

// Size returns the total bytes held in RAM.
func (r *RAMStorage) Size() int64 {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return r.bytes
}

// Has reports whether the given layer is present in RAM.
func (r *RAMStorage) Has(index int) bool {
	r.mu.RLock()
	defer r.mu.RUnlock()
	_, ok := r.layers[index]
	return ok
}

// Close drops all layers and releases memory.
func (r *RAMStorage) Close() error {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.layers = make(map[int]Layer)
	r.bytes = 0
	return nil
}

// Delete removes a single layer from RAM. It is used by multi-tier
// providers (see [Tiered]) to implement eviction policies.
func (r *RAMStorage) Delete(index int) {
	r.mu.Lock()
	defer r.mu.Unlock()
	if layer, ok := r.layers[index]; ok {
		r.bytes -= layer.Size
		delete(r.layers, index)
	}
}

// Ensure RAMStorage satisfies StorageProvider at compile time.
var _ StorageProvider = (*RAMStorage)(nil)

package vhdx

import (
	"context"
	"encoding/binary"
	"errors"
	"fmt"
	"io"
	"sync"
)

// AllocationUnit is the VHDX cluster size recommended by the design spec
// (64 KiB). Layers are always aligned to this boundary on disk to
// minimize fragmentation and maximize sequential throughput. Matching the
// host-volume allocation unit (NTFS/ReFS formatted with /A:64K) avoids
// double-I/O from misalignment.
const AllocationUnit int64 = 64 * 1024

// layerHeaderSize is the number of bytes prefixed to every on-disk slot
// to record the logical (un-padded) layer size.
const layerHeaderSize = 8

// alignUp rounds n up to the nearest multiple of unit.
func alignUp(n, unit int64) int64 {
	if unit <= 0 {
		return n
	}
	rem := n % unit
	if rem == 0 {
		return n
	}
	return n + (unit - rem)
}

// vhdxIndex describes where a layer lives inside the backing file.
type vhdxIndex struct {
	offset int64
	size   int64 // logical size in bytes
	slot   int64 // aligned slot size on disk, including header
}

// VHDXStorage is a [StorageProvider] backed by a random-access file such
// as a VHDX virtual disk. All writes are aligned to the 64 KiB allocation
// unit. The implementation is safe for concurrent use.
//
// Once a fatal I/O error is observed on the backing medium, all further
// operations return [ErrStorageUnavailable] and [Failed] returns true.
// Callers (typically [Tiered]) can then fall back to another provider.
type VHDXStorage struct {
	mu sync.RWMutex

	backing  ReadWriteSeekCloser
	index    map[int]vhdxIndex
	nextFree int64 // next aligned offset where a new layer may be written
	totalLog int64 // logical bytes stored
	failed   bool
}

// NewVHDXStorage constructs a [VHDXStorage] over the given backing file.
// The caller transfers ownership of backing; [VHDXStorage.Close] will
// close it.
func NewVHDXStorage(backing ReadWriteSeekCloser) *VHDXStorage {
	return &VHDXStorage{
		backing: backing,
		index:   make(map[int]vhdxIndex),
	}
}

// Name returns "vhdx".
func (v *VHDXStorage) Name() string { return "vhdx" }

// Store appends the layer to the backing file at the next aligned offset.
// The logical size is encoded as an 8-byte little-endian header followed
// by the raw layer bytes, padded to [AllocationUnit].
func (v *VHDXStorage) Store(ctx context.Context, layer Layer) error {
	if err := ctx.Err(); err != nil {
		return err
	}
	if layer.Size < 0 {
		return fmt.Errorf("vhdx: negative layer size %d", layer.Size)
	}
	if int64(len(layer.Data)) != layer.Size {
		return fmt.Errorf("vhdx: layer %d declared size %d but has %d bytes", layer.Index, layer.Size, len(layer.Data))
	}

	v.mu.Lock()
	defer v.mu.Unlock()
	if v.failed {
		return ErrStorageUnavailable
	}
	if v.backing == nil {
		return ErrStorageUnavailable
	}

	slot := alignUp(layerHeaderSize+layer.Size, AllocationUnit)
	buf := make([]byte, slot)
	binary.LittleEndian.PutUint64(buf[:layerHeaderSize], uint64(layer.Size))
	copy(buf[layerHeaderSize:], layer.Data)

	offset := v.nextFree
	if _, err := v.backing.WriteAt(buf, offset); err != nil {
		v.failed = true
		return fmt.Errorf("%w: %v", ErrStorageUnavailable, err)
	}

	if existing, ok := v.index[layer.Index]; ok {
		v.totalLog -= existing.size
	}
	v.index[layer.Index] = vhdxIndex{offset: offset, size: layer.Size, slot: slot}
	v.nextFree += slot
	v.totalLog += layer.Size
	return nil
}

// Load reads the layer at the given index from the backing file.
func (v *VHDXStorage) Load(ctx context.Context, index int) (Layer, error) {
	if err := ctx.Err(); err != nil {
		return Layer{}, err
	}

	v.mu.RLock()
	if v.failed || v.backing == nil {
		v.mu.RUnlock()
		return Layer{}, ErrStorageUnavailable
	}
	meta, ok := v.index[index]
	backing := v.backing
	v.mu.RUnlock()
	if !ok {
		return Layer{}, ErrLayerNotFound
	}

	buf := make([]byte, meta.slot)
	if _, err := backing.ReadAt(buf, meta.offset); err != nil && !errors.Is(err, io.EOF) {
		v.mu.Lock()
		v.failed = true
		v.mu.Unlock()
		return Layer{}, fmt.Errorf("%w: %v", ErrStorageUnavailable, err)
	}

	declared := int64(binary.LittleEndian.Uint64(buf[:layerHeaderSize]))
	if declared != meta.size {
		return Layer{}, fmt.Errorf("vhdx: layer %d header size %d mismatches index size %d", index, declared, meta.size)
	}
	data := make([]byte, meta.size)
	copy(data, buf[layerHeaderSize:layerHeaderSize+meta.size])
	return Layer{Index: index, Size: meta.size, Data: data}, nil
}

// Size returns the total logical bytes (excluding padding) currently
// stored on the virtual disk.
func (v *VHDXStorage) Size() int64 {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.totalLog
}

// AllocatedSize returns the total physical bytes (including padding)
// written to the backing file.
func (v *VHDXStorage) AllocatedSize() int64 {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.nextFree
}

// Has reports whether the given layer has been stored.
func (v *VHDXStorage) Has(index int) bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	_, ok := v.index[index]
	return ok
}

// Failed reports whether the backing medium has encountered a fatal I/O
// error.
func (v *VHDXStorage) Failed() bool {
	v.mu.RLock()
	defer v.mu.RUnlock()
	return v.failed
}

// Close closes the backing file.
func (v *VHDXStorage) Close() error {
	v.mu.Lock()
	defer v.mu.Unlock()
	if v.backing == nil {
		return nil
	}
	err := v.backing.Close()
	v.backing = nil
	v.index = nil
	return err
}

// indexes returns the set of layer indices currently known. It is used
// by [Tiered] for bookkeeping.
func (v *VHDXStorage) indexes() []int {
	v.mu.RLock()
	defer v.mu.RUnlock()
	out := make([]int, 0, len(v.index))
	for i := range v.index {
		out = append(out, i)
	}
	return out
}

// Ensure VHDXStorage satisfies StorageProvider at compile time.
var _ StorageProvider = (*VHDXStorage)(nil)

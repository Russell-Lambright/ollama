package vhdx

import (
	"context"
	"errors"
	"sync"
)

// Tiered composes two [StorageProvider]s: a fast "hot" tier (typically
// RAM) and a slower "cold" tier (typically a VHDX virtual disk).
//
// Load prefers the hot tier. On a miss it falls through to the cold tier
// and promotes the returned layer into the hot tier subject to the
// configured HotCapacity. When the cold tier reports
// [ErrStorageUnavailable] and the hot tier has a copy, Load returns the
// hot copy, satisfying the "robust error handling with fallback
// mechanisms" requirement.
//
// Tiered itself satisfies [StorageProvider], so callers can stack tiers
// or substitute a different multi-tier policy without changing the
// inference engine.
type Tiered struct {
	hot  StorageProvider
	cold StorageProvider

	// HotCapacity bounds the number of layers kept promoted in the hot
	// tier. Zero means "unlimited". The policy is least-recently-used.
	HotCapacity int

	mu      sync.Mutex
	lruSeq  uint64
	hotSeen map[int]uint64
}

// NewTiered constructs a [Tiered] provider over hot and cold. Either tier
// may be nil; if both are nil, NewTiered returns nil.
func NewTiered(hot, cold StorageProvider) *Tiered {
	if hot == nil && cold == nil {
		return nil
	}
	return &Tiered{hot: hot, cold: cold, hotSeen: make(map[int]uint64)}
}

// Name returns "tiered".
func (t *Tiered) Name() string { return "tiered" }

// Hot returns the hot tier (may be nil).
func (t *Tiered) Hot() StorageProvider { return t.hot }

// Cold returns the cold tier (may be nil).
func (t *Tiered) Cold() StorageProvider { return t.cold }

// Store writes the layer to the cold tier (system of record) when
// present, falling back to the hot tier if there is no cold tier.
func (t *Tiered) Store(ctx context.Context, layer Layer) error {
	if t.cold != nil {
		return t.cold.Store(ctx, layer)
	}
	if t.hot != nil {
		return t.hot.Store(ctx, layer)
	}
	return ErrStorageUnavailable
}

// Load retrieves a layer, preferring the hot tier.
func (t *Tiered) Load(ctx context.Context, index int) (Layer, error) {
	if t.hot != nil && t.hot.Has(index) {
		layer, err := t.hot.Load(ctx, index)
		if err == nil {
			t.touch(index)
			return layer, nil
		}
		// Hot-tier failure falls through to cold.
	}
	if t.cold != nil {
		layer, err := t.cold.Load(ctx, index)
		switch {
		case err == nil:
			t.promote(ctx, layer)
			return layer, nil
		case errors.Is(err, ErrStorageUnavailable):
			// Fall through: try the hot tier as a last resort.
		default:
			return Layer{}, err
		}
	}
	if t.hot != nil && t.hot.Has(index) {
		return t.hot.Load(ctx, index)
	}
	return Layer{}, ErrLayerNotFound
}

// Size reports the logical byte total. When a cold tier exists it is the
// system of record; otherwise the hot tier is used.
func (t *Tiered) Size() int64 {
	if t.cold != nil {
		return t.cold.Size()
	}
	if t.hot != nil {
		return t.hot.Size()
	}
	return 0
}

// Has reports whether either tier holds the layer.
func (t *Tiered) Has(index int) bool {
	if t.hot != nil && t.hot.Has(index) {
		return true
	}
	if t.cold != nil && t.cold.Has(index) {
		return true
	}
	return false
}

// Close closes both tiers.
func (t *Tiered) Close() error {
	var errs []error
	if t.hot != nil {
		if err := t.hot.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if t.cold != nil {
		if err := t.cold.Close(); err != nil {
			errs = append(errs, err)
		}
	}
	if len(errs) == 0 {
		return nil
	}
	return errors.Join(errs...)
}

func (t *Tiered) touch(index int) {
	t.mu.Lock()
	t.lruSeq++
	t.hotSeen[index] = t.lruSeq
	t.mu.Unlock()
}

func (t *Tiered) promote(ctx context.Context, layer Layer) {
	if t.hot == nil {
		return
	}
	if !t.hot.Has(layer.Index) {
		if err := t.hot.Store(ctx, layer); err != nil {
			return
		}
	}
	t.touch(layer.Index)
	t.evictIfNeeded()
}

// evictIfNeeded drops the least-recently-used hot entries until the hot
// tier is at or below HotCapacity. Providers that implement a
// Delete(int) method are asked to evict; others are ignored and the LRU
// map simply forgets them.
func (t *Tiered) evictIfNeeded() {
	if t.HotCapacity <= 0 || t.hot == nil {
		return
	}
	type deleter interface{ Delete(int) }

	t.mu.Lock()
	defer t.mu.Unlock()
	for len(t.hotSeen) > t.HotCapacity {
		var oldestIdx int
		var oldestSeq uint64
		first := true
		for idx, seq := range t.hotSeen {
			if first || seq < oldestSeq {
				oldestIdx = idx
				oldestSeq = seq
				first = false
			}
		}
		delete(t.hotSeen, oldestIdx)
		if d, ok := t.hot.(deleter); ok {
			d.Delete(oldestIdx)
		}
	}
}

// Ensure Tiered satisfies StorageProvider at compile time.
var _ StorageProvider = (*Tiered)(nil)

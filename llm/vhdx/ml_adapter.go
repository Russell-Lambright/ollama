package vhdx

import (
	"context"
	"errors"
	"fmt"

	"github.com/ollama/ollama/ml"
)

// ProviderWeightLoader adapts a [StorageProvider] to the
// [ml.WeightLoader] interface so that the inference engine (ggml /
// llama.cpp) can read model layers transparently from either RAM, VHDX,
// or any other provider that satisfies the abstraction.
//
// This is the concrete hand-off between the storage tier and the
// ml package: registering a ProviderWeightLoader with
// [ml.RegisterWeightLoader] returns a handle that can be passed to the
// C side of the inference engine.
type ProviderWeightLoader struct {
	provider StorageProvider
}

// NewProviderWeightLoader wraps provider so it satisfies
// [ml.WeightLoader]. Nil provider is treated as an empty loader that
// returns ErrNoWeightLoader for every index.
func NewProviderWeightLoader(provider StorageProvider) *ProviderWeightLoader {
	return &ProviderWeightLoader{provider: provider}
}

// ReadLayer implements [ml.WeightLoader]. It performs a single Load
// against the underlying provider and copies the requested range into
// dst. The provider already applies its own caching and prefetching
// policies, so ReadLayer is intentionally thin.
func (p *ProviderWeightLoader) ReadLayer(ctx context.Context, index int, offset int64, dst []byte) (int, error) {
	if p == nil || p.provider == nil {
		return 0, ml.ErrNoWeightLoader
	}
	layer, err := p.provider.Load(ctx, index)
	if err != nil {
		if errors.Is(err, ErrLayerNotFound) {
			return 0, ml.ErrNoWeightLoader
		}
		return 0, err
	}
	if offset < 0 || offset > int64(len(layer.Data)) {
		return 0, fmt.Errorf("vhdx: offset %d out of range for layer %d (size %d)", offset, index, len(layer.Data))
	}
	n := copy(dst, layer.Data[offset:])
	return n, nil
}

// LayerSize implements [ml.WeightLoader]. It attempts to load the layer
// to measure it; layers that are not present return zero.
func (p *ProviderWeightLoader) LayerSize(index int) int64 {
	if p == nil || p.provider == nil || !p.provider.Has(index) {
		return 0
	}
	// The Layer record carries Size directly; we avoid a full Load by
	// requiring providers to answer Has cheaply and trusting Size on a
	// subsequent Load. For now we take the simple path: Load and
	// discard. Providers that want to avoid this cost can implement a
	// LayerMeta(index) extension in a future iteration.
	layer, err := p.provider.Load(context.Background(), index)
	if err != nil {
		return 0
	}
	return layer.Size
}

// Register wraps the provider in a [ProviderWeightLoader] and registers
// it with [ml.RegisterWeightLoader], returning the resulting handle.
// The caller is responsible for calling [ml.UnregisterWeightLoader]
// when the model is unloaded.
func Register(provider StorageProvider) uint64 {
	return ml.RegisterWeightLoader(NewProviderWeightLoader(provider))
}

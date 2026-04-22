package vhdx

import (
	"context"
	"errors"
	"testing"

	"github.com/ollama/ollama/ml"
)

func TestProviderWeightLoaderRoundTrip(t *testing.T) {
	ml.ResetWeightLoaders()
	t.Cleanup(ml.ResetWeightLoaders)

	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 3, Size: 11, Data: []byte("hello world")})

	loader := NewProviderWeightLoader(r)
	if sz := loader.LayerSize(3); sz != 11 {
		t.Fatalf("LayerSize = %d, want 11", sz)
	}
	buf := make([]byte, 5)
	n, err := loader.ReadLayer(context.Background(), 3, 6, buf)
	if err != nil {
		t.Fatalf("ReadLayer: %v", err)
	}
	if n != 5 || string(buf) != "world" {
		t.Fatalf("ReadLayer = %q (n=%d)", buf[:n], n)
	}
}

func TestProviderWeightLoaderUnknownLayer(t *testing.T) {
	loader := NewProviderWeightLoader(NewRAMStorage())
	_, err := loader.ReadLayer(context.Background(), 42, 0, make([]byte, 4))
	if !errors.Is(err, ml.ErrNoWeightLoader) {
		t.Fatalf("ReadLayer unknown = %v, want ErrNoWeightLoader", err)
	}
	if sz := loader.LayerSize(42); sz != 0 {
		t.Fatalf("LayerSize unknown = %d", sz)
	}
}

func TestProviderWeightLoaderBadOffset(t *testing.T) {
	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 0, Size: 2, Data: []byte("ab")})
	loader := NewProviderWeightLoader(r)
	if _, err := loader.ReadLayer(context.Background(), 0, -1, make([]byte, 1)); err == nil {
		t.Fatal("expected error for negative offset")
	}
	if _, err := loader.ReadLayer(context.Background(), 0, 10, make([]byte, 1)); err == nil {
		t.Fatal("expected error for out-of-range offset")
	}
}

func TestProviderWeightLoaderNil(t *testing.T) {
	var loader *ProviderWeightLoader
	if _, err := loader.ReadLayer(context.Background(), 0, 0, make([]byte, 1)); !errors.Is(err, ml.ErrNoWeightLoader) {
		t.Fatalf("nil loader ReadLayer = %v", err)
	}
	if sz := loader.LayerSize(0); sz != 0 {
		t.Fatalf("nil loader LayerSize = %d", sz)
	}
}

func TestRegisterAndLookup(t *testing.T) {
	ml.ResetWeightLoaders()
	t.Cleanup(ml.ResetWeightLoaders)

	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 0, Size: 3, Data: []byte("abc")})

	h := Register(r)
	loader, err := ml.LookupWeightLoader(h)
	if err != nil {
		t.Fatalf("LookupWeightLoader: %v", err)
	}
	buf := make([]byte, 3)
	n, err := loader.ReadLayer(context.Background(), 0, 0, buf)
	if err != nil || n != 3 || string(buf) != "abc" {
		t.Fatalf("ReadLayer = %q,%d,%v", buf[:n], n, err)
	}
	ml.UnregisterWeightLoader(h)
	if _, err := ml.LookupWeightLoader(h); !errors.Is(err, ml.ErrNoWeightLoader) {
		t.Fatalf("after Unregister = %v", err)
	}
}

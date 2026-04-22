package vhdx

import (
	"context"
	"errors"
	"testing"
)

// failingProvider is a StorageProvider that always returns
// ErrStorageUnavailable, used to simulate a dead cold tier.
type failingProvider struct{}

func (failingProvider) Name() string                             { return "failing" }
func (failingProvider) Store(context.Context, Layer) error       { return ErrStorageUnavailable }
func (failingProvider) Load(context.Context, int) (Layer, error) { return Layer{}, ErrStorageUnavailable }
func (failingProvider) Size() int64                              { return 0 }
func (failingProvider) Has(int) bool                             { return false }
func (failingProvider) Close() error                             { return nil }

func TestTieredHotFirst(t *testing.T) {
	hot := NewRAMStorage()
	cold := NewVHDXStorage(&memFile{})
	tr := NewTiered(hot, cold)
	defer tr.Close()

	// Different bytes in each tier so we can detect which one was read.
	_ = hot.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("HOT")})
	_ = cold.Store(context.Background(), Layer{Index: 1, Size: 4, Data: []byte("COLD")})

	got, err := tr.Load(context.Background(), 1)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if string(got.Data) != "HOT" {
		t.Fatalf("expected hot tier to win, got %q", got.Data)
	}
}

func TestTieredColdPromotes(t *testing.T) {
	hot := NewRAMStorage()
	cold := NewVHDXStorage(&memFile{})
	tr := NewTiered(hot, cold)
	defer tr.Close()

	_ = cold.Store(context.Background(), Layer{Index: 7, Size: 5, Data: []byte("hello")})
	if hot.Has(7) {
		t.Fatal("hot should start empty")
	}

	if _, err := tr.Load(context.Background(), 7); err != nil {
		t.Fatalf("Load: %v", err)
	}
	if !hot.Has(7) {
		t.Fatal("cold-hit should promote into hot")
	}
}

func TestTieredLRUEviction(t *testing.T) {
	hot := NewRAMStorage()
	cold := NewVHDXStorage(&memFile{})
	tr := NewTiered(hot, cold)
	tr.HotCapacity = 2
	defer tr.Close()

	for i := 0; i < 3; i++ {
		_ = cold.Store(context.Background(), Layer{Index: i, Size: 1, Data: []byte{byte(i)}})
	}
	for i := 0; i < 3; i++ {
		if _, err := tr.Load(context.Background(), i); err != nil {
			t.Fatalf("Load %d: %v", i, err)
		}
	}
	// After three cold-hit promotions with HotCapacity=2, the oldest
	// (index 0) must have been evicted from hot.
	if hot.Has(0) {
		t.Fatal("LRU eviction failed: hot still contains index 0")
	}
	if !hot.Has(2) {
		t.Fatal("hot should still contain most-recent index")
	}
}

func TestTieredColdUnavailableFallsBackToHot(t *testing.T) {
	hot := NewRAMStorage()
	tr := NewTiered(hot, failingProvider{})
	defer tr.Close()

	_ = hot.Store(context.Background(), Layer{Index: 0, Size: 2, Data: []byte("OK")})
	got, err := tr.Load(context.Background(), 0)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if string(got.Data) != "OK" {
		t.Fatalf("fallback read wrong data: %q", got.Data)
	}
}

func TestTieredMissReturnsNotFound(t *testing.T) {
	tr := NewTiered(NewRAMStorage(), NewVHDXStorage(&memFile{}))
	defer tr.Close()
	if _, err := tr.Load(context.Background(), 99); !errors.Is(err, ErrLayerNotFound) {
		t.Fatalf("miss = %v, want ErrLayerNotFound", err)
	}
}

func TestTieredStoreGoesToColdTier(t *testing.T) {
	hot := NewRAMStorage()
	cold := NewVHDXStorage(&memFile{})
	tr := NewTiered(hot, cold)
	defer tr.Close()

	if err := tr.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("abc")}); err != nil {
		t.Fatalf("Store: %v", err)
	}
	if !cold.Has(1) {
		t.Fatal("Store did not reach cold tier")
	}
	if hot.Has(1) {
		t.Fatal("Store should not pre-populate hot tier")
	}
}

func TestTieredHotOnly(t *testing.T) {
	hot := NewRAMStorage()
	tr := NewTiered(hot, nil)
	defer tr.Close()

	if err := tr.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("abc")}); err != nil {
		t.Fatalf("Store: %v", err)
	}
	if !tr.Has(1) {
		t.Fatal("Has(1) should be true")
	}
	if tr.Size() != 3 {
		t.Fatalf("Size = %d, want 3", tr.Size())
	}
}

func TestNewTieredNilReturnsNil(t *testing.T) {
	if tr := NewTiered(nil, nil); tr != nil {
		t.Fatal("NewTiered(nil, nil) must return nil")
	}
}

func TestTieredName(t *testing.T) {
	tr := NewTiered(NewRAMStorage(), nil)
	defer tr.Close()
	if tr.Name() != "tiered" {
		t.Fatalf("Name = %q", tr.Name())
	}
}

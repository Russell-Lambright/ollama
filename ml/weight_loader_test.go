package ml

import (
	"context"
	"errors"
	"testing"
)

type fakeLoader struct {
	data map[int][]byte
}

func (f *fakeLoader) ReadLayer(_ context.Context, index int, offset int64, dst []byte) (int, error) {
	b, ok := f.data[index]
	if !ok {
		return 0, ErrNoWeightLoader
	}
	if offset >= int64(len(b)) {
		return 0, nil
	}
	n := copy(dst, b[offset:])
	return n, nil
}

func (f *fakeLoader) LayerSize(index int) int64 {
	if b, ok := f.data[index]; ok {
		return int64(len(b))
	}
	return 0
}

func TestWeightLoaderRegistry(t *testing.T) {
	ResetWeightLoaders()
	t.Cleanup(ResetWeightLoaders)

	f := &fakeLoader{data: map[int][]byte{0: []byte("hello"), 1: []byte("world!!")}}
	h := RegisterWeightLoader(f)
	if h == 0 {
		t.Fatal("RegisterWeightLoader returned zero handle")
	}

	got, err := LookupWeightLoader(h)
	if err != nil {
		t.Fatalf("LookupWeightLoader: %v", err)
	}
	if size := got.LayerSize(1); size != 7 {
		t.Fatalf("LayerSize = %d, want 7", size)
	}

	buf := make([]byte, 5)
	n, err := got.ReadLayer(context.Background(), 0, 0, buf)
	if err != nil || n != 5 || string(buf) != "hello" {
		t.Fatalf("ReadLayer = %q,%d,%v", buf[:n], n, err)
	}

	UnregisterWeightLoader(h)
	if _, err := LookupWeightLoader(h); !errors.Is(err, ErrNoWeightLoader) {
		t.Fatalf("Lookup after Unregister = %v, want ErrNoWeightLoader", err)
	}
}

func TestWeightLoaderRegistryIsolation(t *testing.T) {
	ResetWeightLoaders()
	t.Cleanup(ResetWeightLoaders)

	h1 := RegisterWeightLoader(&fakeLoader{})
	h2 := RegisterWeightLoader(&fakeLoader{})
	if h1 == h2 {
		t.Fatalf("handles must be unique; got %d for both", h1)
	}
}

func TestWeightLoaderLookupUnknown(t *testing.T) {
	ResetWeightLoaders()
	if _, err := LookupWeightLoader(99); !errors.Is(err, ErrNoWeightLoader) {
		t.Fatalf("lookup unknown = %v", err)
	}
}

package vhdx

import (
	"context"
	"errors"
	"sync"
	"testing"
)

func TestRAMStorageRoundTrip(t *testing.T) {
	r := NewRAMStorage()
	defer r.Close()

	data := []byte("hello world")
	if err := r.Store(context.Background(), Layer{Index: 1, Size: int64(len(data)), Data: data}); err != nil {
		t.Fatalf("Store: %v", err)
	}
	if !r.Has(1) {
		t.Fatalf("expected layer 1 to be present")
	}
	if got := r.Size(); got != int64(len(data)) {
		t.Fatalf("Size = %d, want %d", got, len(data))
	}

	got, err := r.Load(context.Background(), 1)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if string(got.Data) != string(data) {
		t.Fatalf("Load data = %q, want %q", got.Data, data)
	}
	if got.Size != int64(len(data)) || got.Index != 1 {
		t.Fatalf("Load meta = %+v", got)
	}
}

func TestRAMStorageDefensiveCopy(t *testing.T) {
	r := NewRAMStorage()
	defer r.Close()

	data := []byte{1, 2, 3, 4}
	if err := r.Store(context.Background(), Layer{Index: 0, Size: 4, Data: data}); err != nil {
		t.Fatalf("Store: %v", err)
	}
	// Mutate the caller's buffer; stored copy must not be affected.
	data[0] = 99

	got, err := r.Load(context.Background(), 0)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if got.Data[0] != 1 {
		t.Fatalf("Store did not make a defensive copy; got %v", got.Data)
	}

	// Mutate the returned buffer; a second Load must still be pristine.
	got.Data[0] = 77
	again, err := r.Load(context.Background(), 0)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if again.Data[0] != 1 {
		t.Fatalf("Load did not make a defensive copy; got %v", again.Data)
	}
}

func TestRAMStorageMissingLayer(t *testing.T) {
	r := NewRAMStorage()
	defer r.Close()

	_, err := r.Load(context.Background(), 42)
	if !errors.Is(err, ErrLayerNotFound) {
		t.Fatalf("Load missing = %v, want ErrLayerNotFound", err)
	}
}

func TestRAMStorageInvalidInputs(t *testing.T) {
	r := NewRAMStorage()
	defer r.Close()

	if err := r.Store(context.Background(), Layer{Index: 0, Size: -1}); err == nil {
		t.Fatal("expected error for negative size")
	}
	if err := r.Store(context.Background(), Layer{Index: 0, Size: 10, Data: []byte{1, 2}}); err == nil {
		t.Fatal("expected error for size/length mismatch")
	}
}

func TestRAMStorageOverwrite(t *testing.T) {
	r := NewRAMStorage()
	defer r.Close()

	_ = r.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("abc")})
	_ = r.Store(context.Background(), Layer{Index: 1, Size: 5, Data: []byte("hello")})

	if got := r.Size(); got != 5 {
		t.Fatalf("Size after overwrite = %d, want 5", got)
	}
	got, _ := r.Load(context.Background(), 1)
	if string(got.Data) != "hello" {
		t.Fatalf("overwrite lost; got %q", got.Data)
	}
}

func TestRAMStorageDelete(t *testing.T) {
	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("abc")})
	r.Delete(1)
	if r.Has(1) {
		t.Fatal("Delete did not remove layer")
	}
	if r.Size() != 0 {
		t.Fatalf("Size after Delete = %d, want 0", r.Size())
	}
	// Delete of unknown index is a no-op.
	r.Delete(9999)
}

func TestRAMStorageClose(t *testing.T) {
	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("abc")})
	if err := r.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if r.Has(1) || r.Size() != 0 {
		t.Fatal("Close did not drop state")
	}
}

func TestRAMStorageContextCancelled(t *testing.T) {
	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("abc")})

	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if _, err := r.Load(ctx, 1); !errors.Is(err, context.Canceled) {
		t.Fatalf("Load with cancelled ctx = %v, want context.Canceled", err)
	}
}

func TestRAMStorageConcurrent(t *testing.T) {
	r := NewRAMStorage()
	defer r.Close()

	const N = 64
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			buf := []byte{byte(i)}
			_ = r.Store(context.Background(), Layer{Index: i, Size: 1, Data: buf})
			_, _ = r.Load(context.Background(), i)
		}(i)
	}
	wg.Wait()

	for i := 0; i < N; i++ {
		got, err := r.Load(context.Background(), i)
		if err != nil {
			t.Fatalf("Load %d: %v", i, err)
		}
		if len(got.Data) != 1 || got.Data[0] != byte(i) {
			t.Fatalf("Load %d = %v", i, got.Data)
		}
	}
}

func TestRAMStorageName(t *testing.T) {
	if got := (&RAMStorage{}).Name(); got != "ram" {
		t.Fatalf("Name = %q, want ram", got)
	}
}

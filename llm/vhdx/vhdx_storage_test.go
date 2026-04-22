package vhdx

import (
	"context"
	"errors"
	"io"
	"sync"
	"testing"
)

// memFile is an in-memory ReadWriteSeekCloser used to exercise
// VHDXStorage without touching the disk. It grows on write and can be
// programmed to fail on the next WriteAt or ReadAt to simulate disk
// failures.
type memFile struct {
	mu       sync.Mutex
	data     []byte
	failNext bool
	closed   bool
}

func (m *memFile) WriteAt(p []byte, off int64) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return 0, errors.New("closed")
	}
	if m.failNext {
		m.failNext = false
		return 0, errors.New("simulated disk write failure")
	}
	end := int(off) + len(p)
	if end > len(m.data) {
		m.data = append(m.data, make([]byte, end-len(m.data))...)
	}
	copy(m.data[off:], p)
	return len(p), nil
}

func (m *memFile) ReadAt(p []byte, off int64) (int, error) {
	m.mu.Lock()
	defer m.mu.Unlock()
	if m.closed {
		return 0, errors.New("closed")
	}
	if m.failNext {
		m.failNext = false
		return 0, errors.New("simulated disk read failure")
	}
	if off >= int64(len(m.data)) {
		return 0, io.EOF
	}
	n := copy(p, m.data[off:])
	if n < len(p) {
		return n, io.EOF
	}
	return n, nil
}

func (m *memFile) Close() error {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.closed = true
	return nil
}

func TestAlignUp(t *testing.T) {
	cases := []struct{ in, unit, want int64 }{
		{0, 64, 0},
		{1, 64, 64},
		{64, 64, 64},
		{65, 64, 128},
		{100, 0, 100}, // guard against unit=0
	}
	for _, c := range cases {
		if got := alignUp(c.in, c.unit); got != c.want {
			t.Errorf("alignUp(%d,%d) = %d, want %d", c.in, c.unit, got, c.want)
		}
	}
}

func TestVHDXStorageRoundTripAndAlignment(t *testing.T) {
	m := &memFile{}
	v := NewVHDXStorage(m)
	defer v.Close()

	payload := make([]byte, 1234) // smaller than the allocation unit
	for i := range payload {
		payload[i] = byte(i)
	}

	if err := v.Store(context.Background(), Layer{Index: 0, Size: int64(len(payload)), Data: payload}); err != nil {
		t.Fatalf("Store: %v", err)
	}
	// A small layer should consume exactly one 64 KiB slot on disk.
	if got := v.AllocatedSize(); got != AllocationUnit {
		t.Fatalf("AllocatedSize = %d, want %d", got, AllocationUnit)
	}
	if got := v.Size(); got != int64(len(payload)) {
		t.Fatalf("Size = %d, want %d", got, len(payload))
	}
	if !v.Has(0) {
		t.Fatal("Has(0) = false")
	}

	got, err := v.Load(context.Background(), 0)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if string(got.Data) != string(payload) {
		t.Fatal("Load returned different bytes than Store")
	}
}

func TestVHDXStorageLargerThanAllocationUnit(t *testing.T) {
	m := &memFile{}
	v := NewVHDXStorage(m)
	defer v.Close()

	payload := make([]byte, int(AllocationUnit)+123)
	for i := range payload {
		payload[i] = byte(i)
	}
	if err := v.Store(context.Background(), Layer{Index: 1, Size: int64(len(payload)), Data: payload}); err != nil {
		t.Fatalf("Store: %v", err)
	}
	if got := v.AllocatedSize(); got != 2*AllocationUnit {
		t.Fatalf("AllocatedSize = %d, want %d", got, 2*AllocationUnit)
	}
	loaded, err := v.Load(context.Background(), 1)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if string(loaded.Data) != string(payload) {
		t.Fatal("round-trip mismatch on multi-slot layer")
	}
}

func TestVHDXStorageMissingLayer(t *testing.T) {
	v := NewVHDXStorage(&memFile{})
	defer v.Close()

	_, err := v.Load(context.Background(), 5)
	if !errors.Is(err, ErrLayerNotFound) {
		t.Fatalf("got %v, want ErrLayerNotFound", err)
	}
}

func TestVHDXStorageInvalidInputs(t *testing.T) {
	v := NewVHDXStorage(&memFile{})
	defer v.Close()

	if err := v.Store(context.Background(), Layer{Index: 0, Size: -1}); err == nil {
		t.Fatal("expected error for negative size")
	}
	if err := v.Store(context.Background(), Layer{Index: 0, Size: 10, Data: []byte{1, 2}}); err == nil {
		t.Fatal("expected error for size/length mismatch")
	}
}

func TestVHDXStorageWriteFailureLatches(t *testing.T) {
	m := &memFile{failNext: true}
	v := NewVHDXStorage(m)
	defer v.Close()

	err := v.Store(context.Background(), Layer{Index: 0, Size: 3, Data: []byte("abc")})
	if !errors.Is(err, ErrStorageUnavailable) {
		t.Fatalf("first Store err = %v, want ErrStorageUnavailable", err)
	}
	if !v.Failed() {
		t.Fatal("Failed() should latch after a write error")
	}
	// Subsequent operations must keep reporting ErrStorageUnavailable.
	if err := v.Store(context.Background(), Layer{Index: 1, Size: 1, Data: []byte("a")}); !errors.Is(err, ErrStorageUnavailable) {
		t.Fatalf("subsequent Store err = %v", err)
	}
	if _, err := v.Load(context.Background(), 0); !errors.Is(err, ErrStorageUnavailable) {
		t.Fatalf("Load after failure = %v", err)
	}
}

func TestVHDXStorageReadFailureLatches(t *testing.T) {
	m := &memFile{}
	v := NewVHDXStorage(m)
	defer v.Close()

	if err := v.Store(context.Background(), Layer{Index: 0, Size: 3, Data: []byte("abc")}); err != nil {
		t.Fatalf("Store: %v", err)
	}
	m.mu.Lock()
	m.failNext = true
	m.mu.Unlock()

	if _, err := v.Load(context.Background(), 0); !errors.Is(err, ErrStorageUnavailable) {
		t.Fatalf("Load after read failure = %v", err)
	}
	if !v.Failed() {
		t.Fatal("Failed() should latch after a read error")
	}
}

func TestVHDXStorageOverwrite(t *testing.T) {
	v := NewVHDXStorage(&memFile{})
	defer v.Close()

	_ = v.Store(context.Background(), Layer{Index: 1, Size: 3, Data: []byte("abc")})
	_ = v.Store(context.Background(), Layer{Index: 1, Size: 5, Data: []byte("hello")})

	if got := v.Size(); got != 5 {
		t.Fatalf("Size after overwrite = %d, want 5", got)
	}
	got, err := v.Load(context.Background(), 1)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if string(got.Data) != "hello" {
		t.Fatalf("overwrite lost; got %q", got.Data)
	}
}

func TestVHDXStorageClose(t *testing.T) {
	m := &memFile{}
	v := NewVHDXStorage(m)
	if err := v.Close(); err != nil {
		t.Fatalf("Close: %v", err)
	}
	if !m.closed {
		t.Fatal("backing file not closed")
	}
	// Second close is a no-op.
	if err := v.Close(); err != nil {
		t.Fatalf("second Close: %v", err)
	}
	// Operations after Close must report unavailability.
	if err := v.Store(context.Background(), Layer{Index: 0, Size: 1, Data: []byte{0}}); !errors.Is(err, ErrStorageUnavailable) {
		t.Fatalf("Store after Close = %v", err)
	}
	if _, err := v.Load(context.Background(), 0); !errors.Is(err, ErrStorageUnavailable) {
		t.Fatalf("Load after Close = %v", err)
	}
}

func TestVHDXStorageConcurrent(t *testing.T) {
	v := NewVHDXStorage(&memFile{})
	defer v.Close()

	const N = 32
	var wg sync.WaitGroup
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			buf := make([]byte, 256)
			for j := range buf {
				buf[j] = byte(i)
			}
			if err := v.Store(context.Background(), Layer{Index: i, Size: int64(len(buf)), Data: buf}); err != nil {
				t.Errorf("Store %d: %v", i, err)
			}
		}(i)
	}
	wg.Wait()

	for i := 0; i < N; i++ {
		got, err := v.Load(context.Background(), i)
		if err != nil {
			t.Fatalf("Load %d: %v", i, err)
		}
		for j := range got.Data {
			if got.Data[j] != byte(i) {
				t.Fatalf("layer %d byte %d = %d", i, j, got.Data[j])
				break
			}
		}
	}
}

func TestVHDXStorageContextCancelled(t *testing.T) {
	v := NewVHDXStorage(&memFile{})
	defer v.Close()
	ctx, cancel := context.WithCancel(context.Background())
	cancel()
	if err := v.Store(ctx, Layer{Index: 0, Size: 1, Data: []byte{0}}); !errors.Is(err, context.Canceled) {
		t.Fatalf("Store with cancelled ctx = %v", err)
	}
	if _, err := v.Load(ctx, 0); !errors.Is(err, context.Canceled) {
		t.Fatalf("Load with cancelled ctx = %v", err)
	}
}

func TestVHDXStorageName(t *testing.T) {
	if got := (&VHDXStorage{}).Name(); got != "vhdx" {
		t.Fatalf("Name = %q", got)
	}
}

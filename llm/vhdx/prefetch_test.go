package vhdx

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

// countingProvider wraps a provider and counts how many times Load was
// called. It is used to verify deduplication and worker-cap behavior.
type countingProvider struct {
	inner StorageProvider
	loads atomic.Int64
	block chan struct{} // if non-nil, Load blocks until closed
}

func (c *countingProvider) Name() string { return "counting" }
func (c *countingProvider) Store(ctx context.Context, l Layer) error {
	return c.inner.Store(ctx, l)
}

func (c *countingProvider) Load(ctx context.Context, i int) (Layer, error) {
	c.loads.Add(1)
	if c.block != nil {
		select {
		case <-c.block:
		case <-ctx.Done():
			return Layer{}, ctx.Err()
		}
	}
	return c.inner.Load(ctx, i)
}
func (c *countingProvider) Size() int64    { return c.inner.Size() }
func (c *countingProvider) Has(i int) bool { return c.inner.Has(i) }
func (c *countingProvider) Close() error   { return c.inner.Close() }

func TestPrefetcherWarms(t *testing.T) {
	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 0, Size: 3, Data: []byte("abc")})

	c := &countingProvider{inner: r}
	p := NewPrefetcher(c, 2)

	p.Prefetch(context.Background(), 0)
	if err := p.Wait(context.Background(), 0); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if c.loads.Load() != 1 {
		t.Fatalf("loads = %d, want 1", c.loads.Load())
	}
}

func TestPrefetcherDeduplicates(t *testing.T) {
	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 0, Size: 1, Data: []byte{0}})

	block := make(chan struct{})
	c := &countingProvider{inner: r, block: block}
	p := NewPrefetcher(c, 4)

	// Queue the same index several times before any load can finish.
	p.Prefetch(context.Background(), 0, 0, 0, 0, 0)
	// Small sleep to let the goroutine register itself in inFlight.
	time.Sleep(10 * time.Millisecond)
	if got := p.InFlight(); got != 1 {
		t.Fatalf("InFlight = %d, want 1", got)
	}
	close(block)
	if err := p.Wait(context.Background(), 0); err != nil {
		t.Fatalf("Wait: %v", err)
	}
	if got := c.loads.Load(); got != 1 {
		t.Fatalf("Load called %d times, want 1", got)
	}
}

func TestPrefetcherWorkerCap(t *testing.T) {
	r := NewRAMStorage()
	for i := 0; i < 10; i++ {
		_ = r.Store(context.Background(), Layer{Index: i, Size: 1, Data: []byte{byte(i)}})
	}

	block := make(chan struct{})
	c := &countingProvider{inner: r, block: block}
	p := NewPrefetcher(c, 2)

	p.Prefetch(context.Background(), 0, 1, 2, 3, 4)
	// Give goroutines a chance to attempt to acquire the semaphore.
	time.Sleep(20 * time.Millisecond)
	// Two should be in flight; the rest queued.
	if got := c.loads.Load(); got > 2 {
		t.Fatalf("more than worker cap in flight: %d", got)
	}
	close(block)

	for i := 0; i < 5; i++ {
		if err := p.Wait(context.Background(), i); err != nil {
			t.Fatalf("Wait %d: %v", i, err)
		}
	}
	if got := c.loads.Load(); got != 5 {
		t.Fatalf("total loads = %d, want 5", got)
	}
}

func TestPrefetcherWaitUnknownIndex(t *testing.T) {
	p := NewPrefetcher(NewRAMStorage(), 1)
	if err := p.Wait(context.Background(), 42); err != nil {
		t.Fatalf("Wait on unknown index = %v", err)
	}
}

func TestPrefetcherCancellation(t *testing.T) {
	r := NewRAMStorage()
	_ = r.Store(context.Background(), Layer{Index: 0, Size: 1, Data: []byte{0}})
	block := make(chan struct{})
	defer close(block)
	c := &countingProvider{inner: r, block: block}
	p := NewPrefetcher(c, 1)

	ctx, cancel := context.WithCancel(context.Background())
	p.Prefetch(ctx, 0)
	time.Sleep(5 * time.Millisecond)

	waitCtx, waitCancel := context.WithTimeout(context.Background(), 50*time.Millisecond)
	defer waitCancel()
	err := p.Wait(waitCtx, 0)
	if err == nil {
		t.Fatal("Wait should time out while prefetch is blocked")
	}
	cancel()
}

func TestPrefetcherWorkerCapNormalization(t *testing.T) {
	p := NewPrefetcher(NewRAMStorage(), 0)
	if p.workers != 1 {
		t.Fatalf("workers = %d, want 1", p.workers)
	}
}

package vhdx

import (
	"context"
	"sync"
)

// Prefetcher schedules asynchronous layer loads from a [StorageProvider]
// so that layers are already warm in the hot tier by the time the
// inference engine needs them. It exists to keep model inference off the
// critical path of disk I/O, satisfying the design requirement that
// "layer loading must not block the main inference process".
type Prefetcher struct {
	provider StorageProvider
	workers  int

	mu       sync.Mutex
	inFlight map[int]chan struct{}
	sem      chan struct{}
}

// NewPrefetcher returns a [Prefetcher] backed by the given provider.
// Workers controls the maximum number of concurrent prefetch operations;
// values less than 1 default to 1.
func NewPrefetcher(provider StorageProvider, workers int) *Prefetcher {
	if workers < 1 {
		workers = 1
	}
	return &Prefetcher{
		provider: provider,
		workers:  workers,
		inFlight: make(map[int]chan struct{}),
		sem:      make(chan struct{}, workers),
	}
}

// Prefetch asynchronously warms the provider for the given layer
// indices. It returns immediately. Concurrent requests for the same
// index are deduplicated. Use [Prefetcher.Wait] to block until a
// specific layer has been loaded.
func (p *Prefetcher) Prefetch(ctx context.Context, indices ...int) {
	for _, idx := range indices {
		idx := idx
		p.mu.Lock()
		if _, already := p.inFlight[idx]; already {
			p.mu.Unlock()
			continue
		}
		done := make(chan struct{})
		p.inFlight[idx] = done
		p.mu.Unlock()

		go func() {
			defer close(done)
			select {
			case p.sem <- struct{}{}:
				defer func() { <-p.sem }()
			case <-ctx.Done():
				p.mu.Lock()
				delete(p.inFlight, idx)
				p.mu.Unlock()
				return
			}
			// A successful Load through a Tiered provider also promotes
			// the layer into the hot tier, which is exactly the side
			// effect we want.
			_, _ = p.provider.Load(ctx, idx)

			p.mu.Lock()
			delete(p.inFlight, idx)
			p.mu.Unlock()
		}()
	}
}

// Wait blocks until any outstanding prefetch for index has completed or
// ctx is cancelled. It is safe to call Wait for an index that was never
// prefetched; it returns nil immediately in that case.
func (p *Prefetcher) Wait(ctx context.Context, index int) error {
	p.mu.Lock()
	done, ok := p.inFlight[index]
	p.mu.Unlock()
	if !ok {
		return nil
	}
	select {
	case <-done:
		return nil
	case <-ctx.Done():
		return ctx.Err()
	}
}

// InFlight reports the number of prefetches currently outstanding. It is
// primarily useful for tests and diagnostics.
func (p *Prefetcher) InFlight() int {
	p.mu.Lock()
	defer p.mu.Unlock()
	return len(p.inFlight)
}

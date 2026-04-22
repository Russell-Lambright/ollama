//go:build cgo && ollama_vhdx_bridge

// Package cgo_bridge exposes an `ollama_vhdx_read_layer` symbol to
// C/C++ callers (notably llama.cpp / ggml) so that the inference engine
// can fetch tensor bytes from a registered [ml.WeightLoader] without
// having to pass a Go pointer across the FFI boundary.
//
// This file is only built when the "ollama_vhdx_bridge" build tag is
// supplied, e.g.:
//
//	go build -tags ollama_vhdx_bridge ./...
//
// Default builds of Ollama are unaffected. Once the llama.cpp side of
// the integration lands upstream, this file can be promoted to an
// unconditional build.
package vhdx

/*
#include <stdint.h>
#include <stdlib.h>

// See vhdx_bridge.h for the C-facing declaration.
*/
import "C"
import (
	"context"
	"errors"
	"sync/atomic"
	"unsafe"

	"github.com/ollama/ollama/ml"
)

// bridgeContext is a package-scoped context used by the CGO callback.
// The C side cannot supply a Go context, so we keep a single cancellable
// context here and reset it between model loads.
var bridgeContext atomic.Pointer[context.Context]

// SetBridgeContext installs the context that the C bridge will use for
// subsequent ReadLayer calls. Callers should install a fresh context
// per model load and cancel it when the model is unloaded.
func SetBridgeContext(ctx context.Context) {
	bridgeContext.Store(&ctx)
}

func currentBridgeContext() context.Context {
	p := bridgeContext.Load()
	if p == nil {
		return context.Background()
	}
	return *p
}

// ollama_vhdx_read_layer reads size bytes starting at offset from the
// layer `index` of the WeightLoader registered under `handle` and
// copies them into the destination buffer `dst`.
//
// Return values:
//
//	> 0  number of bytes copied
//	= 0  index is unknown (fall back to default loader)
//	< 0  fatal error (abort)
//
//export ollama_vhdx_read_layer
func ollama_vhdx_read_layer(handle C.uint64_t, index C.int, offset C.int64_t, dst *C.char, size C.int64_t) C.int64_t {
	if size <= 0 || dst == nil {
		return 0
	}
	loader, err := ml.LookupWeightLoader(uint64(handle))
	if err != nil {
		if errors.Is(err, ml.ErrNoWeightLoader) {
			return 0
		}
		return -1
	}
	buf := unsafe.Slice((*byte)(unsafe.Pointer(dst)), int(size))
	n, err := loader.ReadLayer(currentBridgeContext(), int(index), int64(offset), buf)
	if err != nil {
		if errors.Is(err, ml.ErrNoWeightLoader) {
			return 0
		}
		return -1
	}
	return C.int64_t(n)
}

// ollama_vhdx_layer_size returns the size in bytes of the given layer,
// or zero if unknown.
//
//export ollama_vhdx_layer_size
func ollama_vhdx_layer_size(handle C.uint64_t, index C.int) C.int64_t {
	loader, err := ml.LookupWeightLoader(uint64(handle))
	if err != nil {
		return 0
	}
	return C.int64_t(loader.LayerSize(int(index)))
}

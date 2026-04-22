//go:build !cgo || !ollama_vhdx_bridge

package vhdx

import "context"

// SetBridgeContext is a no-op in builds that do not include the CGO
// bridge. Keeping the symbol available on all platforms means callers
// can install a context unconditionally without needing build tags of
// their own.
func SetBridgeContext(_ context.Context) {}

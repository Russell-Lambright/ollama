//go:build !windows

package vhdx

import "errors"

// ErrUnsupportedPlatform is returned by [Provision] on non-Windows
// platforms. The storage abstraction itself works everywhere, but VHDX
// provisioning requires Windows tooling.
var ErrUnsupportedPlatform = errors.New("vhdx: VHDX provisioning is only supported on Windows")

// ProvisionOptions is kept on all platforms so callers can share the
// same type. See the Windows build of this file for field semantics.
type ProvisionOptions struct {
	Path                string
	SizeBytes           int64
	AllocationUnitBytes int64
	FallbackToFile      bool
}

// ProvisionResult mirrors the Windows type. On non-Windows platforms
// Provision never returns a non-empty result.
type ProvisionResult struct {
	Path      string
	SizeBytes int64
	Format    string
}

// Provision is a no-op on non-Windows platforms and always returns
// [ErrUnsupportedPlatform]. Tests and cross-platform callers can detect
// this and skip VHDX-specific behavior.
func Provision(_ ProvisionOptions) (ProvisionResult, error) {
	return ProvisionResult{}, ErrUnsupportedPlatform
}

//go:build windows

package vhdx

import (
	"errors"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// ProvisionOptions configures [Provision].
type ProvisionOptions struct {
	// Path is the destination file. If it has no extension, ".vhdx" is
	// appended. Parent directories are created if missing.
	Path string
	// SizeBytes is the desired fixed size of the VHDX in bytes. Values
	// less than or equal to zero default to 1 GiB.
	SizeBytes int64
	// AllocationUnitBytes is the cluster size inside the VHDX. Defaults
	// to [AllocationUnit] (64 KiB).
	AllocationUnitBytes int64
	// FallbackToFile controls whether a failed VHDX creation degrades to
	// a plain preallocated file. This keeps developer setups working on
	// Windows editions without Hyper-V tooling. Default is true.
	FallbackToFile bool
}

// ProvisionResult describes what Provision actually created.
type ProvisionResult struct {
	// Path is the final path on disk.
	Path string
	// SizeBytes is the requested fixed size.
	SizeBytes int64
	// Format is either "vhdx" or "file" (the latter when VHDX tooling
	// was unavailable and FallbackToFile was true).
	Format string
}

// Provision creates a fixed-size VHDX at opts.Path using PowerShell's
// New-VHD cmdlet. On Windows editions that do not provide Hyper-V
// tooling, and when FallbackToFile is true, it creates a plain
// preallocated file of the requested size so that [VHDXStorage] still
// has a backing file to write to.
//
// This helper intentionally does not mount the VHDX. Mounting is left
// to the operator (or a future enhancement) because it typically
// requires administrator privileges and is out of scope for unit tests.
func Provision(opts ProvisionOptions) (ProvisionResult, error) {
	if opts.Path == "" {
		return ProvisionResult{}, errors.New("vhdx: Provision: Path is required")
	}
	if opts.SizeBytes <= 0 {
		opts.SizeBytes = 1 << 30 // 1 GiB default
	}
	if opts.AllocationUnitBytes <= 0 {
		opts.AllocationUnitBytes = AllocationUnit
	}
	if ext := strings.ToLower(filepath.Ext(opts.Path)); ext == "" {
		opts.Path += ".vhdx"
	}
	if err := os.MkdirAll(filepath.Dir(opts.Path), 0o755); err != nil {
		return ProvisionResult{}, fmt.Errorf("vhdx: Provision: %w", err)
	}

	// If the file already exists, treat provisioning as a no-op: callers
	// are responsible for re-using an existing VHDX.
	if _, err := os.Stat(opts.Path); err == nil {
		return ProvisionResult{Path: opts.Path, SizeBytes: opts.SizeBytes, Format: "existing"}, nil
	}

	if err := newVHD(opts); err == nil {
		return ProvisionResult{Path: opts.Path, SizeBytes: opts.SizeBytes, Format: "vhdx"}, nil
	} else if !opts.FallbackToFile {
		return ProvisionResult{}, fmt.Errorf("vhdx: New-VHD failed: %w", err)
	}

	if err := preallocateFile(opts.Path, opts.SizeBytes); err != nil {
		return ProvisionResult{}, fmt.Errorf("vhdx: fallback preallocation failed: %w", err)
	}
	return ProvisionResult{Path: opts.Path, SizeBytes: opts.SizeBytes, Format: "file"}, nil
}

// newVHD invokes PowerShell's New-VHD cmdlet to create a fixed-size
// VHDX. The call fails (and we fall back) if Hyper-V is not installed.
func newVHD(opts ProvisionOptions) error {
	sizeMB := (opts.SizeBytes + (1 << 20) - 1) >> 20
	blockSizeBytes := strconv.FormatInt(opts.AllocationUnitBytes, 10)

	// Use -ArgumentList style and quote the path to tolerate spaces.
	script := fmt.Sprintf(
		`New-VHD -Path '%s' -SizeBytes %dMB -Fixed -BlockSizeBytes %s | Out-Null`,
		strings.ReplaceAll(opts.Path, "'", "''"), sizeMB, blockSizeBytes,
	)

	cmd := exec.Command("powershell.exe", "-NoProfile", "-NonInteractive", "-Command", script)
	out, err := cmd.CombinedOutput()
	if err != nil {
		return fmt.Errorf("%w: %s", err, strings.TrimSpace(string(out)))
	}
	return nil
}

// preallocateFile creates a file of exactly size bytes, zero-filled. We
// use [os.File.Truncate] which is fast on NTFS and avoids holding the
// entire buffer in memory.
func preallocateFile(path string, size int64) error {
	f, err := os.Create(path)
	if err != nil {
		return err
	}
	defer f.Close()
	if err := f.Truncate(size); err != nil {
		return err
	}
	return nil
}

// Ensure io is referenced to simplify future extension points.
var _ = io.Copy

//go:build !windows

package vhdx

import (
	"errors"
	"testing"
)

func TestProvisionUnsupportedOnNonWindows(t *testing.T) {
	_, err := Provision(ProvisionOptions{Path: "/tmp/ignored.vhdx"})
	if !errors.Is(err, ErrUnsupportedPlatform) {
		t.Fatalf("Provision = %v, want ErrUnsupportedPlatform", err)
	}
}

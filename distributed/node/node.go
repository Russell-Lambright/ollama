// Package node defines stable identity for Secondary nodes in the
// distributed MPI-style framework: a UUID, the advertised hostname, the
// collective the node belongs to, and the advertised LPU (a weighted
// GPU+CPU score used by the orchestrator for allocation).
//
// Phase 2 scope: types, validation, and a deterministic LPU formula.
// Actual LPU measurement (probing GPU/CPU) lives in the secondary runtime
// (Phase 3); this package only defines the contract.
package node

import (
	"crypto/rand"
	"encoding/hex"
	"errors"
	"fmt"
	"os"
	"strings"
)

// ID is an opaque, stable identifier for a node. It is generated once at
// node startup and reused across reconnects so the Primary can recognize a
// returning Secondary.
type ID string

// NewID returns a freshly generated random node ID. The format is a
// 32-character lowercase hex string (128 bits of entropy); it is
// intentionally not a formatted UUID because the Primary treats this as an
// opaque token.
func NewID() (ID, error) {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		return "", fmt.Errorf("distributed/node: generate id: %w", err)
	}
	return ID(hex.EncodeToString(b[:])), nil
}

// MustNewID is like NewID but panics on entropy-source failure. Suitable
// for process startup where no sensible recovery exists.
func MustNewID() ID {
	id, err := NewID()
	if err != nil {
		panic(err)
	}
	return id
}

// String implements fmt.Stringer.
func (id ID) String() string { return string(id) }

// Valid reports whether the ID is a non-empty 32-char lowercase hex string.
func (id ID) Valid() bool {
	if len(id) != 32 {
		return false
	}
	for i := 0; i < len(id); i++ {
		c := id[i]
		switch {
		case c >= '0' && c <= '9':
		case c >= 'a' && c <= 'f':
		default:
			return false
		}
	}
	return true
}

// Identity is the self-description a Secondary sends to the Primary at
// registration time and re-asserts on reconnection.
type Identity struct {
	// ID is the stable node identifier.
	ID ID
	// Hostname is the node's advertised hostname (informational; used in UI).
	Hostname string
	// Collective is the name of the collective the node joins.
	Collective string
	// AdvertisedLPU is the node's self-reported capacity score, produced by
	// LPU. The Primary uses this as an allocation hint; the node's actual
	// latency is still the ultimate source of truth.
	AdvertisedLPU float64
	// Persona is the name of the currently applied persona, or empty.
	Persona string
}

// DefaultHostname returns os.Hostname with a sensible fallback so callers
// never have to branch on error for a non-critical identity field.
func DefaultHostname() string {
	h, err := os.Hostname()
	if err != nil || strings.TrimSpace(h) == "" {
		return "unknown-host"
	}
	return h
}

// Validate reports whether the identity is well-formed enough for the
// Primary to accept a registration.
func (i Identity) Validate() error {
	if !i.ID.Valid() {
		return errors.New("distributed/node: invalid node id")
	}
	if strings.TrimSpace(i.Hostname) == "" {
		return errors.New("distributed/node: hostname required")
	}
	if strings.TrimSpace(i.Collective) == "" {
		return errors.New("distributed/node: collective required")
	}
	if i.AdvertisedLPU < 0 {
		return fmt.Errorf("distributed/node: advertised_lpu must be >= 0 (got %v)", i.AdvertisedLPU)
	}
	return nil
}

// LPU computes a node's capacity score from a GPU score and a CPU score.
// Both inputs are expected to be normalized to a comparable scale (e.g.
// relative FLOPS or relative tokens/sec from a calibration pass). Negative
// inputs are clamped to zero.
//
// The default weighting favors GPU heavily because accelerator presence is
// the dominant factor in throughput for transformer inference.
//
// The formula is deliberately simple and stable — the orchestrator treats
// LPU as a relative sort key, not a precise performance model.
func LPU(gpuScore, cpuScore float64) float64 {
	return LPUWeighted(gpuScore, cpuScore, 0.75, 0.25)
}

// LPUWeighted is LPU with explicit weights. Weights are normalized so they
// always sum to 1 (guarding against operators passing e.g. 3/1 instead of
// 0.75/0.25). Weights ≤ 0 are treated as 0; if both are 0 the function
// returns 0.
func LPUWeighted(gpuScore, cpuScore, gpuWeight, cpuWeight float64) float64 {
	if gpuScore < 0 {
		gpuScore = 0
	}
	if cpuScore < 0 {
		cpuScore = 0
	}
	if gpuWeight < 0 {
		gpuWeight = 0
	}
	if cpuWeight < 0 {
		cpuWeight = 0
	}
	total := gpuWeight + cpuWeight
	if total == 0 {
		return 0
	}
	return (gpuScore*gpuWeight + cpuScore*cpuWeight) / total
}

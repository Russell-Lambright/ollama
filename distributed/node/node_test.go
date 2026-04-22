package node

import (
	"strings"
	"testing"
)

func TestNewIDIsValidAndUnique(t *testing.T) {
	seen := make(map[ID]struct{})
	for i := 0; i < 100; i++ {
		id, err := NewID()
		if err != nil {
			t.Fatalf("NewID: %v", err)
		}
		if !id.Valid() {
			t.Fatalf("id %q not valid", id)
		}
		if strings.ToLower(string(id)) != string(id) {
			t.Fatalf("id %q must be lowercase", id)
		}
		if _, dup := seen[id]; dup {
			t.Fatalf("duplicate id %q after %d iterations", id, i)
		}
		seen[id] = struct{}{}
	}
}

func TestIDValid(t *testing.T) {
	cases := map[string]struct {
		in   ID
		want bool
	}{
		"empty":         {"", false},
		"short":         {"abc", false},
		"upper":         {"ABCDEF0123456789ABCDEF0123456789", false},
		"non-hex":       {"zzzzzzzzzzzzzzzzzzzzzzzzzzzzzzzz", false},
		"valid":         {"0123456789abcdef0123456789abcdef", true},
		"valid-all-9s":  {"99999999999999999999999999999999", true},
		"too-long":      {"0123456789abcdef0123456789abcdef0", false},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			if got := tc.in.Valid(); got != tc.want {
				t.Errorf("Valid()=%v want %v", got, tc.want)
			}
		})
	}
}

func TestIdentityValidate(t *testing.T) {
	ok := Identity{ID: "0123456789abcdef0123456789abcdef", Hostname: "n1", Collective: "default", AdvertisedLPU: 1.0}
	if err := ok.Validate(); err != nil {
		t.Fatalf("valid identity rejected: %v", err)
	}

	tests := map[string]Identity{
		"bad id":         {ID: "bogus", Hostname: "n1", Collective: "c"},
		"no hostname":    {ID: ok.ID, Hostname: " ", Collective: "c"},
		"no collective":  {ID: ok.ID, Hostname: "n1", Collective: ""},
		"negative lpu":   {ID: ok.ID, Hostname: "n1", Collective: "c", AdvertisedLPU: -0.1},
	}
	for name, id := range tests {
		t.Run(name, func(t *testing.T) {
			if err := id.Validate(); err == nil {
				t.Errorf("expected error for %s", name)
			}
		})
	}
}

func TestDefaultHostname(t *testing.T) {
	// Whatever os.Hostname returns, the function must never return empty.
	if h := DefaultHostname(); strings.TrimSpace(h) == "" {
		t.Fatalf("DefaultHostname returned empty")
	}
}

func TestLPU(t *testing.T) {
	// Default weights: 0.75 gpu, 0.25 cpu.
	if got, want := LPU(100, 0), 75.0; got != want {
		t.Errorf("LPU(100,0)=%v want %v", got, want)
	}
	if got, want := LPU(0, 100), 25.0; got != want {
		t.Errorf("LPU(0,100)=%v want %v", got, want)
	}
	if got := LPU(100, 100); got != 100 {
		t.Errorf("LPU(100,100)=%v want 100", got)
	}
	// Negative scores clamp to zero.
	if got := LPU(-5, -5); got != 0 {
		t.Errorf("LPU(-5,-5)=%v want 0", got)
	}
}

func TestLPUWeighted(t *testing.T) {
	// Weights are normalized: 3:1 is equivalent to 0.75:0.25.
	if got, want := LPUWeighted(100, 0, 3, 1), 75.0; got != want {
		t.Errorf("weights normalize: got %v want %v", got, want)
	}
	// Zero total weight returns zero (no allocation signal).
	if got := LPUWeighted(100, 100, 0, 0); got != 0 {
		t.Errorf("zero weights: got %v want 0", got)
	}
	// Negative weights clamp to zero; the other weight still works.
	if got, want := LPUWeighted(100, 50, -1, 1), 50.0; got != want {
		t.Errorf("negative weight clamped: got %v want %v", got, want)
	}
}

package config

import (
	"os"
	"path/filepath"
	"testing"
)

// withEnv sets envs for the duration of the test and clears them after.
func withEnv(t *testing.T, kv map[string]string) {
	t.Helper()
	for k, v := range kv {
		t.Setenv(k, v)
	}
}

func TestDefault(t *testing.T) {
	c := Default()
	if c.MaxNodesPerCollective != DefaultMaxNodesPerCollective {
		t.Errorf("MaxNodesPerCollective = %d, want %d", c.MaxNodesPerCollective, DefaultMaxNodesPerCollective)
	}
	if c.StarvationIndex != DefaultStarvationIndex {
		t.Errorf("StarvationIndex = %v, want %v", c.StarvationIndex, DefaultStarvationIndex)
	}
	if c.FineTuning.Temperature != DefaultTemperature {
		t.Errorf("FineTuning.Temperature = %v, want %v", c.FineTuning.Temperature, DefaultTemperature)
	}
	if c.FineTuning.NumCtx != DefaultNumCtx {
		t.Errorf("FineTuning.NumCtx = %d, want %d", c.FineTuning.NumCtx, DefaultNumCtx)
	}
	if err := c.Validate(); err != nil {
		t.Fatalf("Default().Validate() = %v, want nil", err)
	}
}

func TestLoadMissingFile(t *testing.T) {
	// Isolate from any user env.
	for _, k := range []string{
		"OLLAMA_PRIMARY_HOST", "OLLAMA_COLLECTIVE", "OLLAMA_DEFAULT_COLLECTIVE",
		"OLLAMA_SPPR_MODEL", "OLLAMA_MAX_NODES_PER_COLLECTIVE", "OLLAMA_STARVATION_INDEX",
	} {
		t.Setenv(k, "")
	}
	cfg, err := Load(filepath.Join(t.TempDir(), "does-not-exist.yaml"))
	if err != nil {
		t.Fatalf("Load(missing) error = %v", err)
	}
	// Should match default.
	want := Default()
	_ = want.Normalize()
	if cfg.MaxNodesPerCollective != want.MaxNodesPerCollective {
		t.Errorf("MaxNodesPerCollective = %d, want %d", cfg.MaxNodesPerCollective, want.MaxNodesPerCollective)
	}
	if cfg.StarvationIndex != want.StarvationIndex {
		t.Errorf("StarvationIndex = %v, want %v", cfg.StarvationIndex, want.StarvationIndex)
	}
	if cfg.SPPRModel != DefaultSPPRModel {
		t.Errorf("SPPRModel = %q, want default %q", cfg.SPPRModel, DefaultSPPRModel)
	}
}

func TestLoadYAML(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "distributed.yaml")
	content := `
max_nodes_per_collective: 4
default_collective: engineering
primary_host: primary.local:11434
collective_membership:
  - engineering
  - research
  - engineering
fine_tuning:
  temperature: 0.35
  top_p: 0.8
  num_ctx: 16384
personas:
  - name: reviewer
    description: Code reviewer
    system_prompt: You are a meticulous reviewer.
    preferred_model: llama3
    tags: [engineering, review]
  - name: architect
    description: System architect
    system_prompt: You design systems.
sppr_model: linguistics-small
starvation_index: 0.7
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	if cfg.MaxNodesPerCollective != 4 {
		t.Errorf("MaxNodesPerCollective = %d, want 4", cfg.MaxNodesPerCollective)
	}
	if cfg.DefaultCollective != "engineering" {
		t.Errorf("DefaultCollective = %q", cfg.DefaultCollective)
	}
	// Duplicate "engineering" should be deduped by Normalize.
	if got := cfg.CollectiveMembership; len(got) != 2 || got[0] != "engineering" || got[1] != "research" {
		t.Errorf("CollectiveMembership = %v, want [engineering research]", got)
	}
	if cfg.FineTuning.Temperature != 0.35 {
		t.Errorf("FineTuning.Temperature = %v", cfg.FineTuning.Temperature)
	}
	if len(cfg.Personas) != 2 {
		t.Fatalf("len(Personas) = %d, want 2", len(cfg.Personas))
	}
	if p := cfg.FindPersona("reviewer"); p == nil || p.PreferredModel != "llama3" {
		t.Errorf("FindPersona(reviewer) = %+v", p)
	}
	if cfg.FindPersona("missing") != nil {
		t.Error("FindPersona(missing) should be nil")
	}
	if cfg.SPPRModel != "linguistics-small" {
		t.Errorf("SPPRModel = %q", cfg.SPPRModel)
	}
	if cfg.StarvationIndex != 0.7 {
		t.Errorf("StarvationIndex = %v", cfg.StarvationIndex)
	}
}

func TestLoadEnvOverrides(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "distributed.yaml")
	content := `
primary_host: file.local:1234
sppr_model: from-file
starvation_index: 0.5
max_nodes_per_collective: 3
default_collective: from-file
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatal(err)
	}

	withEnv(t, map[string]string{
		"OLLAMA_PRIMARY_HOST":             "env.local:9999",
		"OLLAMA_COLLECTIVE":               "alpha, beta , alpha",
		"OLLAMA_DEFAULT_COLLECTIVE":       "from-env",
		"OLLAMA_SPPR_MODEL":               "from-env-model",
		"OLLAMA_MAX_NODES_PER_COLLECTIVE": "12",
		"OLLAMA_STARVATION_INDEX":         "0.9",
	})

	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load error = %v", err)
	}
	if cfg.PrimaryHost != "env.local:9999" {
		t.Errorf("PrimaryHost = %q, want env override", cfg.PrimaryHost)
	}
	if cfg.DefaultCollective != "from-env" {
		t.Errorf("DefaultCollective = %q, want env override", cfg.DefaultCollective)
	}
	if cfg.SPPRModel != "from-env-model" {
		t.Errorf("SPPRModel = %q, want env override", cfg.SPPRModel)
	}
	if cfg.MaxNodesPerCollective != 12 {
		t.Errorf("MaxNodesPerCollective = %d, want 12", cfg.MaxNodesPerCollective)
	}
	if cfg.StarvationIndex != 0.9 {
		t.Errorf("StarvationIndex = %v, want 0.9", cfg.StarvationIndex)
	}
	// Env-supplied collective list should be split, trimmed, deduped.
	if got := cfg.CollectiveMembership; len(got) != 2 || got[0] != "alpha" || got[1] != "beta" {
		t.Errorf("CollectiveMembership = %v, want [alpha beta]", got)
	}
}

func TestNormalizeClampsStarvationIndex(t *testing.T) {
	cases := map[string]struct {
		in, want float64
	}{
		"below min": {0.01, MinStarvationIndex},
		"above max": {5.0, MaxStarvationIndex},
		"zero defaults": {0, DefaultStarvationIndex},
		"in range":   {0.5, 0.5},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			c := Default()
			c.StarvationIndex = tc.in
			if err := c.Normalize(); err != nil {
				t.Fatalf("Normalize err = %v", err)
			}
			if c.StarvationIndex != tc.want {
				t.Errorf("StarvationIndex = %v, want %v", c.StarvationIndex, tc.want)
			}
		})
	}
}

func TestValidateRejectsBadPersonas(t *testing.T) {
	c := Default()
	c.Personas = []Persona{{Name: "dup"}, {Name: "dup"}}
	if err := c.Validate(); err == nil {
		t.Error("expected error for duplicate personas")
	}
	c.Personas = []Persona{{Name: ""}}
	if err := c.Validate(); err == nil {
		t.Error("expected error for empty persona name")
	}
}

func TestValidateRejectsBadStarvation(t *testing.T) {
	c := Default()
	c.StarvationIndex = 2.0
	if err := c.Validate(); err == nil {
		t.Error("expected error for out-of-range starvation index")
	}
	c = Default()
	c.MaxNodesPerCollective = 0
	if err := c.Validate(); err == nil {
		t.Error("expected error for non-positive max nodes")
	}
}

func TestLoadInvalidYAML(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.yaml")
	if err := os.WriteFile(path, []byte("max_nodes_per_collective: [not-an-int\n"), 0o600); err != nil {
		t.Fatal(err)
	}
	if _, err := Load(path); err == nil {
		t.Error("expected error for invalid YAML")
	}
}

func TestDefaultPathRespectsEnv(t *testing.T) {
	t.Setenv("OLLAMA_DISTRIBUTED_CONFIG", "/custom/path.yaml")
	got, err := DefaultPath()
	if err != nil {
		t.Fatal(err)
	}
	if got != "/custom/path.yaml" {
		t.Errorf("DefaultPath() = %q", got)
	}
}

func TestParseTransport(t *testing.T) {
	cases := map[string]struct {
		in      string
		want    Transport
		wantErr bool
	}{
		"empty default": {"", DefaultTransport, false},
		"grpc":          {"grpc", TransportGRPC, false},
		"http2-sse":     {"http2-sse", TransportHTTP2SSE, false},
		"http2_sse":     {"http2_sse", TransportHTTP2SSE, false},
		"sse alias":     {"sse", TransportHTTP2SSE, false},
		"whitespace":    {"  grpc  ", TransportGRPC, false},
		"invalid":       {"websocket", "", true},
	}
	for name, tc := range cases {
		t.Run(name, func(t *testing.T) {
			got, err := ParseTransport(tc.in)
			if (err != nil) != tc.wantErr {
				t.Fatalf("ParseTransport(%q) err=%v wantErr=%v", tc.in, err, tc.wantErr)
			}
			if got != tc.want {
				t.Errorf("ParseTransport(%q) = %q, want %q", tc.in, got, tc.want)
			}
		})
	}
}

func TestTransportValid(t *testing.T) {
	for _, v := range []Transport{TransportGRPC, TransportHTTP2SSE} {
		if !v.Valid() {
			t.Errorf("Transport(%q).Valid() = false", v)
		}
	}
	if Transport("rest").Valid() {
		t.Error(`Transport("rest").Valid() = true`)
	}
}

func TestLoadTransportEnvOverride(t *testing.T) {
	t.Setenv("OLLAMA_TRANSPORT", "http2-sse")
	for _, k := range []string{
		"OLLAMA_PRIMARY_HOST", "OLLAMA_COLLECTIVE", "OLLAMA_DEFAULT_COLLECTIVE",
		"OLLAMA_SPPR_MODEL", "OLLAMA_MAX_NODES_PER_COLLECTIVE", "OLLAMA_STARVATION_INDEX",
	} {
		t.Setenv(k, "")
	}
	cfg, err := Load(filepath.Join(t.TempDir(), "missing.yaml"))
	if err != nil {
		t.Fatal(err)
	}
	if cfg.Transport != TransportHTTP2SSE {
		t.Errorf("Transport = %q, want %q", cfg.Transport, TransportHTTP2SSE)
	}
}

func TestValidateRejectsBadTransport(t *testing.T) {
	c := Default()
	c.Transport = "invalid"
	if err := c.Validate(); err == nil {
		t.Error("expected error for invalid transport")
	}
}

func TestDefaultPathFallsBackToHome(t *testing.T) {
	t.Setenv("OLLAMA_DISTRIBUTED_CONFIG", "")
	got, err := DefaultPath()
	if err != nil {
		t.Fatal(err)
	}
	if filepath.Base(got) != "distributed.yaml" {
		t.Errorf("DefaultPath() = %q, want ending in distributed.yaml", got)
	}
}

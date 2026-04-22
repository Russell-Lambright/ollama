package cmd

import (
	"testing"

	"github.com/spf13/cobra"

	"github.com/ollama/ollama/distributed"
	distconfig "github.com/ollama/ollama/distributed/config"
)

// newServeTestCmd returns a cobra.Command carrying the same flag set as
// the real `serve` command. It is used to exercise resolveDistributedMode
// without booting an HTTP server.
func newServeTestCmd() *cobra.Command {
	c := &cobra.Command{Use: "serve"}
	c.Flags().String("mode", "", "")
	c.Flags().String("primary", "", "")
	c.Flags().String("collective", "", "")
	c.Flags().String("persona", "", "")
	c.Flags().String("transport", "", "")
	return c
}

// clearDistEnv unsets every distributed-related environment variable so a
// test starts from a known-clean state regardless of the host environment.
func clearDistEnv(t *testing.T) {
	t.Helper()
	for _, k := range []string{
		"OLLAMA_NODE_MODE",
		"OLLAMA_PRIMARY_HOST",
		"OLLAMA_COLLECTIVE",
		"OLLAMA_DEFAULT_COLLECTIVE",
		"OLLAMA_SPPR_MODEL",
		"OLLAMA_MAX_NODES_PER_COLLECTIVE",
		"OLLAMA_STARVATION_INDEX",
		"OLLAMA_TRANSPORT",
		"OLLAMA_DISTRIBUTED_CONFIG",
	} {
		t.Setenv(k, "")
	}
	// Also redirect the config file to a non-existent path so user-level
	// config cannot leak into the tests.
	t.Setenv("OLLAMA_DISTRIBUTED_CONFIG", t.TempDir()+"/none.yaml")
}

func TestResolveDistributedMode_DefaultStandalone(t *testing.T) {
	clearDistEnv(t)
	c := newServeTestCmd()
	if err := resolveDistributedMode(c, nil); err != nil {
		t.Fatalf("resolveDistributedMode err = %v", err)
	}
	mode, cfg, persona := ResolvedDistributed()
	if mode != distributed.ModeStandalone {
		t.Errorf("mode = %q, want standalone", mode)
	}
	if cfg.SPPRModel != distconfig.DefaultSPPRModel {
		t.Errorf("SPPRModel = %q, want default %q", cfg.SPPRModel, distconfig.DefaultSPPRModel)
	}
	if cfg.Transport != distconfig.DefaultTransport {
		t.Errorf("Transport = %q, want default %q", cfg.Transport, distconfig.DefaultTransport)
	}
	if persona != "" {
		t.Errorf("persona = %q, want empty", persona)
	}
}

func TestResolveDistributedMode_FlagOverridesEnv(t *testing.T) {
	clearDistEnv(t)
	t.Setenv("OLLAMA_NODE_MODE", "primary")
	t.Setenv("OLLAMA_TRANSPORT", "grpc")
	c := newServeTestCmd()
	// CLI overrides env: --mode=secondary beats OLLAMA_NODE_MODE=primary.
	_ = c.Flags().Set("mode", "secondary")
	_ = c.Flags().Set("primary", "host.example:11434")
	_ = c.Flags().Set("collective", "engineering")
	_ = c.Flags().Set("transport", "http2-sse")
	_ = c.Flags().Set("persona", "reviewer")

	if err := resolveDistributedMode(c, nil); err != nil {
		t.Fatalf("resolveDistributedMode err = %v", err)
	}
	mode, cfg, persona := ResolvedDistributed()
	if mode != distributed.ModeSecondary {
		t.Errorf("mode = %q, want secondary", mode)
	}
	if cfg.PrimaryHost != "host.example:11434" {
		t.Errorf("PrimaryHost = %q", cfg.PrimaryHost)
	}
	if len(cfg.CollectiveMembership) != 1 || cfg.CollectiveMembership[0] != "engineering" {
		t.Errorf("CollectiveMembership = %v, want [engineering]", cfg.CollectiveMembership)
	}
	if cfg.Transport != distconfig.TransportHTTP2SSE {
		t.Errorf("Transport = %q, want http2-sse", cfg.Transport)
	}
	if persona != "reviewer" {
		t.Errorf("persona = %q, want reviewer", persona)
	}
}

func TestResolveDistributedMode_SecondaryRequiresPrimary(t *testing.T) {
	clearDistEnv(t)
	c := newServeTestCmd()
	_ = c.Flags().Set("mode", "secondary")
	// No --primary and no OLLAMA_PRIMARY_HOST and no config file: must error.
	if err := resolveDistributedMode(c, nil); err == nil {
		t.Fatal("expected error when secondary mode has no primary host")
	}
}

func TestResolveDistributedMode_InvalidMode(t *testing.T) {
	clearDistEnv(t)
	c := newServeTestCmd()
	_ = c.Flags().Set("mode", "worker")
	if err := resolveDistributedMode(c, nil); err == nil {
		t.Fatal("expected error for invalid mode")
	}
}

func TestResolveDistributedMode_InvalidTransport(t *testing.T) {
	clearDistEnv(t)
	c := newServeTestCmd()
	_ = c.Flags().Set("mode", "primary")
	_ = c.Flags().Set("transport", "websocket")
	if err := resolveDistributedMode(c, nil); err == nil {
		t.Fatal("expected error for invalid transport")
	}
}

func TestResolveDistributedMode_CollectiveFlagOnPrimaryBecomesDefault(t *testing.T) {
	clearDistEnv(t)
	c := newServeTestCmd()
	_ = c.Flags().Set("mode", "primary")
	_ = c.Flags().Set("collective", "alpha")
	if err := resolveDistributedMode(c, nil); err != nil {
		t.Fatalf("resolveDistributedMode err = %v", err)
	}
	_, cfg, _ := ResolvedDistributed()
	if cfg.DefaultCollective != "alpha" {
		t.Errorf("DefaultCollective = %q, want alpha", cfg.DefaultCollective)
	}
}

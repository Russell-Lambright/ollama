// Package config holds the typed configuration for the distributed
// MPI-style Ollama framework: personas, fine-tuning defaults, collective
// membership, and orchestrator knobs.
//
// Phase 1 scope: data model + loader only. Nothing in this package changes
// runtime behavior; it simply produces a validated DistributedConfig value
// that later phases will consume.
//
// Precedence (highest wins):
//  1. Explicit overrides applied by the caller (e.g. CLI flags).
//  2. Environment variables (OLLAMA_PRIMARY_HOST, OLLAMA_COLLECTIVE, …).
//  3. Values from ~/.ollama/distributed.yaml (or $OLLAMA_DISTRIBUTED_CONFIG).
//  4. Built-in defaults tuned for engineering / code-development tasks.
package config

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"

	"gopkg.in/yaml.v3"
)

// Default values. These are deliberately conservative and tuned for
// engineering / code-development workloads, per the spec.
const (
	DefaultMaxNodesPerCollective = 8
	// DefaultSPPRModel is the linguistics-trained model used by the Segment
	// Preprocessing Prompt Renderer unless explicitly overridden. Qwen2.5 is
	// used because its strong instruction-following and multilingual
	// coverage make it a good fit for prompt segmentation.
	DefaultSPPRModel       = "qwen2.5"
	DefaultStarvationIndex = 1.0
	MinStarvationIndex           = 0.1
	MaxStarvationIndex           = 1.0

	// Engineering-task fine-tuning defaults.
	DefaultTemperature   = 0.2
	DefaultTopP          = 0.9
	DefaultTopK          = 40
	DefaultNumCtx        = 8192
	DefaultRepeatPenalty = 1.1

	// Small-job fallback thresholds. A prompt below either of these is run
	// on a single node (standalone-equivalent path) instead of going
	// through the SPPR → orchestrator → fan-out pipeline. Both bounds are
	// configurable; zero disables that particular check.
	DefaultSmallJobPromptRuneThreshold = 400 // ~characters; rough proxy for "short prompt"
	DefaultSmallJobMinSegments         = 2   // < this many SPPR segments → single node
)

// Transport selects the wire protocol used between Primary and Secondary
// nodes. Both options are roughly equivalent for this framework; gRPC is
// idiomatic Go streaming, HTTP/2 + SSE aligns with Ollama's existing REST
// streaming surface.
type Transport string

const (
	// TransportGRPC uses bidirectional gRPC between Primary and Secondaries.
	TransportGRPC Transport = "grpc"
	// TransportHTTP2SSE uses HTTP/2 with Server-Sent Events, matching the
	// streaming style of Ollama's existing public API.
	TransportHTTP2SSE Transport = "http2-sse"
	// DefaultTransport is gRPC.
	DefaultTransport = TransportGRPC
)

// Valid reports whether the transport is one of the recognized values.
func (t Transport) Valid() bool {
	switch t {
	case TransportGRPC, TransportHTTP2SSE:
		return true
	default:
		return false
	}
}

// ParseTransport parses a human-supplied transport string. Empty input
// resolves to DefaultTransport. Unknown input returns an error.
func ParseTransport(s string) (Transport, error) {
	switch strings.TrimSpace(s) {
	case "":
		return DefaultTransport, nil
	case string(TransportGRPC):
		return TransportGRPC, nil
	case string(TransportHTTP2SSE), "http2_sse", "sse":
		return TransportHTTP2SSE, nil
	default:
		return "", fmt.Errorf("distributed/config: invalid transport %q (expected grpc or http2-sse)", s)
	}
}

// FineTuning bundles the model sampling parameters that can be passed through
// to every AI processing area in the distributed pipeline (SPPR, orchestrator
// sub-agents, QA pass, secondary execution). Zero values mean "use the
// engine-level default".
type FineTuning struct {
	Temperature   float64 `yaml:"temperature"`
	TopP          float64 `yaml:"top_p"`
	TopK          int     `yaml:"top_k"`
	NumCtx        int     `yaml:"num_ctx"`
	RepeatPenalty float64 `yaml:"repeat_penalty"`
	Seed          *int    `yaml:"seed,omitempty"`
}

// Persona is a predefined identity a Secondary node can adopt. Personas are
// stored in the configuration file and may be requested by a node at startup
// or applied dynamically by the Primary via the UI.
type Persona struct {
	// Name is the unique identifier referenced from the UI, CLI, and config.
	Name string `yaml:"name"`
	// Description is a short human-readable summary shown in the UI during
	// the Training state.
	Description string `yaml:"description"`
	// SystemPrompt is prepended to prompts assigned to a node wearing this
	// persona.
	SystemPrompt string `yaml:"system_prompt"`
	// PreferredModel is an optional model name hint; when unset the persona
	// uses whatever model is requested by the caller.
	PreferredModel string `yaml:"preferred_model,omitempty"`
	// Tags are free-form labels for grouping/filtering in the UI.
	Tags []string `yaml:"tags,omitempty"`
}

// DistributedConfig is the typed, validated configuration for the
// distributed framework.
type DistributedConfig struct {
	// MaxNodesPerCollective caps the number of Secondary nodes that may
	// register in a single collective.
	MaxNodesPerCollective int `yaml:"max_nodes_per_collective"`

	// DefaultCollective, when non-empty, forces every prompt to use this
	// collective unless the caller overrides it. Only meaningful on Primary.
	DefaultCollective string `yaml:"default_collective"`

	// PrimaryHost is the host:port a Secondary node attaches to. Overridden
	// by the --primary CLI flag.
	PrimaryHost string `yaml:"primary_host"`

	// CollectiveMembership lists the collectives a Secondary node wants to
	// join at startup (normally a single entry). Overridden by --collective.
	CollectiveMembership []string `yaml:"collective_membership"`

	// FineTuning is propagated to every AI processing area.
	FineTuning FineTuning `yaml:"fine_tuning"`

	// Personas are the predefined identities available to the Primary for
	// overlay on any Available node.
	Personas []Persona `yaml:"personas"`

	// SPPRModel is the linguistics-trained model used by the Segment
	// Preprocessing Prompt Renderer. Overridable per-request for tests.
	SPPRModel string `yaml:"sppr_model"`

	// StarvationIndex throttles orchestrator node allocation. Clamped to
	// [MinStarvationIndex, MaxStarvationIndex].
	StarvationIndex float64 `yaml:"starvation_index"`

	// Transport selects the Primary↔Secondary wire protocol. Valid values:
	// "grpc" (default), "http2-sse". Overridable via --transport and
	// OLLAMA_TRANSPORT.
	Transport Transport `yaml:"transport"`

	// PromptExpander configures the optional pre-SPPR stage that rewrites
	// an incoming prompt into a more verbose but semantically equivalent
	// form so SPPR has more surface area to segment on. Disabled by
	// default — enable only after measuring that it improves segmentation
	// quality for your workload.
	PromptExpander PromptExpanderConfig `yaml:"prompt_expander"`

	// SmallJob controls the fallback path: jobs whose prompt is short or
	// whose SPPR output is trivial bypass the distributed pipeline and run
	// on a single node (standalone-equivalent).
	SmallJob SmallJobConfig `yaml:"small_job"`
}

// PromptExpanderConfig configures the optional Prompt Expander stage that
// precedes SPPR. When Enabled is true, the framework runs the expander
// before SPPR; when false, prompts flow directly into SPPR unchanged.
//
// The expander's contract is strict: it may add verbosity, restate,
// enumerate, or rephrase, but it MUST NOT alter the user's meaning,
// introduce new facts, or remove information. A downstream guard (Phase 4)
// compares expander output against the original prompt and falls back to
// the original on any meaning-drift signal.
type PromptExpanderConfig struct {
	// Enabled turns the stage on. Default false.
	Enabled bool `yaml:"enabled"`
	// Model is the model used to perform the expansion. Empty means
	// "reuse the SPPR model" so operators don't have to configure two.
	Model string `yaml:"model"`
	// MaxExpansionRatio caps how much the expander may grow the prompt
	// (expanded_len / original_len). Anything above this ratio is
	// discarded and the original prompt is used instead. Default 3.0.
	MaxExpansionRatio float64 `yaml:"max_expansion_ratio"`
}

// SmallJobConfig controls the threshold below which a job skips the
// distributed pipeline and is executed on a single node. This is a
// graceful-degradation path, not an error.
type SmallJobConfig struct {
	// PromptRuneThreshold: if the raw prompt has fewer runes than this,
	// the job falls back to single-node mode before even invoking SPPR.
	// Set to 0 to disable this check. Default: DefaultSmallJobPromptRuneThreshold.
	PromptRuneThreshold int `yaml:"prompt_rune_threshold"`
	// MinSegments: if SPPR emits fewer segments than this, the job falls
	// back to single-node execution (the distribution overhead is not
	// worth it). Set to 0 to disable. Default: DefaultSmallJobMinSegments.
	MinSegments int `yaml:"min_segments"`
}

// IsSmallJob reports whether a raw prompt is small enough to bypass the
// distributed pipeline up-front, before SPPR has run. The SPPR-output
// segment count check is applied separately by the orchestrator (Phase 5).
func (c *DistributedConfig) IsSmallJob(prompt string) bool {
	if c == nil {
		return false
	}
	if c.SmallJob.PromptRuneThreshold <= 0 {
		return false
	}
	// Use rune count (not byte count) so multi-byte languages are handled
	// correctly.
	n := 0
	for range prompt {
		n++
		if n >= c.SmallJob.PromptRuneThreshold {
			return false
		}
	}
	return true
}

// SegmentsBelowFallback reports whether a SPPR output of the given segment
// count should trigger single-node fallback. Used by the orchestrator.
func (c *DistributedConfig) SegmentsBelowFallback(segmentCount int) bool {
	if c == nil || c.SmallJob.MinSegments <= 0 {
		return false
	}
	return segmentCount < c.SmallJob.MinSegments
}

// Default returns a DistributedConfig populated with safe defaults tuned for
// engineering / code-development workloads.
func Default() DistributedConfig {
	return DistributedConfig{
		MaxNodesPerCollective: DefaultMaxNodesPerCollective,
		FineTuning: FineTuning{
			Temperature:   DefaultTemperature,
			TopP:          DefaultTopP,
			TopK:          DefaultTopK,
			NumCtx:        DefaultNumCtx,
			RepeatPenalty: DefaultRepeatPenalty,
		},
		SPPRModel:       DefaultSPPRModel,
		StarvationIndex: DefaultStarvationIndex,
		Transport:       DefaultTransport,
		PromptExpander: PromptExpanderConfig{
			Enabled:           false,
			Model:             "", // empty means "reuse SPPRModel"
			MaxExpansionRatio: 3.0,
		},
		SmallJob: SmallJobConfig{
			PromptRuneThreshold: DefaultSmallJobPromptRuneThreshold,
			MinSegments:         DefaultSmallJobMinSegments,
		},
	}
}

// DefaultPath returns the default on-disk location for the distributed
// config file: $OLLAMA_DISTRIBUTED_CONFIG if set, else ~/.ollama/distributed.yaml.
func DefaultPath() (string, error) {
	if p := strings.TrimSpace(os.Getenv("OLLAMA_DISTRIBUTED_CONFIG")); p != "" {
		return p, nil
	}
	home, err := os.UserHomeDir()
	if err != nil {
		return "", fmt.Errorf("distributed/config: locate home: %w", err)
	}
	return filepath.Join(home, ".ollama", "distributed.yaml"), nil
}

// Load reads the config file at path and merges it with built-in defaults and
// environment overrides. A missing file is not an error — Load returns the
// default config with env overrides applied.
func Load(path string) (DistributedConfig, error) {
	cfg := Default()

	if path != "" {
		data, err := os.ReadFile(path)
		switch {
		case err == nil:
			if err := yaml.Unmarshal(data, &cfg); err != nil {
				return DistributedConfig{}, fmt.Errorf("distributed/config: parse %s: %w", path, err)
			}
		case errors.Is(err, os.ErrNotExist):
			// Missing file is fine; defaults are used.
		default:
			return DistributedConfig{}, fmt.Errorf("distributed/config: read %s: %w", path, err)
		}
	}

	applyEnvOverrides(&cfg)

	if err := cfg.Normalize(); err != nil {
		return DistributedConfig{}, err
	}
	if err := cfg.Validate(); err != nil {
		return DistributedConfig{}, err
	}
	return cfg, nil
}

// applyEnvOverrides mutates cfg with values from process environment
// variables. Env wins over the config file but loses to explicit CLI flags
// applied by the caller afterwards.
func applyEnvOverrides(cfg *DistributedConfig) {
	if v := strings.TrimSpace(os.Getenv("OLLAMA_PRIMARY_HOST")); v != "" {
		cfg.PrimaryHost = v
	}
	if v := strings.TrimSpace(os.Getenv("OLLAMA_COLLECTIVE")); v != "" {
		cfg.CollectiveMembership = splitAndTrim(v, ",")
	}
	if v := strings.TrimSpace(os.Getenv("OLLAMA_DEFAULT_COLLECTIVE")); v != "" {
		cfg.DefaultCollective = v
	}
	if v := strings.TrimSpace(os.Getenv("OLLAMA_SPPR_MODEL")); v != "" {
		cfg.SPPRModel = v
	}
	if v := strings.TrimSpace(os.Getenv("OLLAMA_MAX_NODES_PER_COLLECTIVE")); v != "" {
		if n, err := strconv.Atoi(v); err == nil && n > 0 {
			cfg.MaxNodesPerCollective = n
		}
	}
	if v := strings.TrimSpace(os.Getenv("OLLAMA_STARVATION_INDEX")); v != "" {
		if f, err := strconv.ParseFloat(v, 64); err == nil {
			cfg.StarvationIndex = f
		}
	}
	if v := strings.TrimSpace(os.Getenv("OLLAMA_TRANSPORT")); v != "" {
		if t, err := ParseTransport(v); err == nil {
			cfg.Transport = t
		}
	}
}

// Normalize applies fallbacks and clamps numeric fields. Called automatically
// by Load; exported so tests and CLI wiring can re-normalize after applying
// CLI overrides.
func (c *DistributedConfig) Normalize() error {
	if c.MaxNodesPerCollective <= 0 {
		c.MaxNodesPerCollective = DefaultMaxNodesPerCollective
	}
	if c.StarvationIndex == 0 {
		c.StarvationIndex = DefaultStarvationIndex
	}
	if c.StarvationIndex < MinStarvationIndex {
		c.StarvationIndex = MinStarvationIndex
	}
	if c.StarvationIndex > MaxStarvationIndex {
		c.StarvationIndex = MaxStarvationIndex
	}
	if c.Transport == "" {
		c.Transport = DefaultTransport
	}
	if c.PromptExpander.MaxExpansionRatio <= 0 {
		c.PromptExpander.MaxExpansionRatio = 3.0
	}
	if c.SmallJob.PromptRuneThreshold < 0 {
		c.SmallJob.PromptRuneThreshold = 0 // treat negatives as "disabled"
	}
	if c.SmallJob.MinSegments < 0 {
		c.SmallJob.MinSegments = 0
	}
	// Fine-tuning: zero means "engine default" — leave alone so downstream
	// consumers can distinguish unset from set-to-zero using IsZero-style
	// helpers in later phases. Here we only backfill sensible defaults when
	// the user has not specified any fine-tuning block at all.
	if c.FineTuning == (FineTuning{}) {
		c.FineTuning = Default().FineTuning
	}
	// Deduplicate collective membership while preserving order.
	c.CollectiveMembership = dedupeStrings(c.CollectiveMembership)
	return nil
}

// Validate checks the semantic correctness of the configuration (persona
// uniqueness, non-empty persona names, etc.). It is intentionally
// conservative — unknown or extra YAML fields are ignored by the loader.
func (c *DistributedConfig) Validate() error {
	seen := make(map[string]struct{}, len(c.Personas))
	for i, p := range c.Personas {
		if strings.TrimSpace(p.Name) == "" {
			return fmt.Errorf("distributed/config: persona[%d] has empty name", i)
		}
		if _, dup := seen[p.Name]; dup {
			return fmt.Errorf("distributed/config: duplicate persona name %q", p.Name)
		}
		seen[p.Name] = struct{}{}
	}
	if c.StarvationIndex < MinStarvationIndex || c.StarvationIndex > MaxStarvationIndex {
		return fmt.Errorf("distributed/config: starvation_index %v outside [%v,%v]",
			c.StarvationIndex, MinStarvationIndex, MaxStarvationIndex)
	}
	if c.MaxNodesPerCollective <= 0 {
		return fmt.Errorf("distributed/config: max_nodes_per_collective must be > 0")
	}
	if !c.Transport.Valid() {
		return fmt.Errorf("distributed/config: invalid transport %q (expected grpc or http2-sse)", c.Transport)
	}
	return nil
}

// FindPersona returns a pointer to the persona with the given name, or nil if
// no such persona is defined.
func (c *DistributedConfig) FindPersona(name string) *Persona {
	for i := range c.Personas {
		if c.Personas[i].Name == name {
			return &c.Personas[i]
		}
	}
	return nil
}

func splitAndTrim(s, sep string) []string {
	parts := strings.Split(s, sep)
	out := parts[:0]
	for _, p := range parts {
		p = strings.TrimSpace(p)
		if p != "" {
			out = append(out, p)
		}
	}
	return out
}

func dedupeStrings(in []string) []string {
	if len(in) == 0 {
		return in
	}
	seen := make(map[string]struct{}, len(in))
	out := make([]string, 0, len(in))
	for _, s := range in {
		if s == "" {
			continue
		}
		if _, ok := seen[s]; ok {
			continue
		}
		seen[s] = struct{}{}
		out = append(out, s)
	}
	return out
}

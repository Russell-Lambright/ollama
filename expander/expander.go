// Package expander implements the optional Prompt Expander stage that
// runs before SPPR in the distributed pipeline. Its job is to add
// verbosity, restatement, enumeration, or rephrasing to an incoming
// prompt so SPPR has more surface area to segment on — WITHOUT
// altering the user's meaning, introducing new facts, or removing
// information.
//
// This package ships two things:
//
//  1. The Expander interface (wire-agnostic; any implementation works).
//  2. A ModelExpander that drives a model via a ModelClient and
//     enforces the ratio + drift guard declared in
//     DISTRIBUTED_ARCHITECTURE.md §2a. On any guard violation the
//     original prompt is returned, matching the spec's fail-closed
//     posture.
//
// The expander is OPTIONAL — the pipeline (sppr.Plan) only invokes it
// when PromptExpanderConfig.Enabled is true.
package expander

import (
	"context"
	"errors"
	"strings"
	"unicode"
)

// ModelClient is the narrow contract this package depends on to drive a
// model. It is intentionally minimal — implementations plug in the host
// process's inference surface (ollama runtime, an HTTP client, a mock
// for tests, etc.).
type ModelClient interface {
	Generate(ctx context.Context, model, prompt string, options map[string]any) (string, error)
}

// ModelClientFunc adapts a plain function to the ModelClient interface.
type ModelClientFunc func(ctx context.Context, model, prompt string, options map[string]any) (string, error)

// Generate implements ModelClient.
func (f ModelClientFunc) Generate(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
	return f(ctx, model, prompt, options)
}

// Expander rewrites a prompt into a semantically equivalent but more
// verbose form. Implementations MUST be safe under repeated calls and
// MUST return the original prompt unchanged on any detected
// meaning-drift signal.
type Expander interface {
	Expand(ctx context.Context, prompt string) (string, error)
}

// ExpanderFunc adapts a plain function to the Expander interface.
type ExpanderFunc func(ctx context.Context, prompt string) (string, error)

// Expand implements Expander.
func (f ExpanderFunc) Expand(ctx context.Context, prompt string) (string, error) {
	return f(ctx, prompt)
}

// ModelExpander drives expansion through a ModelClient with a built-in
// guard. Zero value is NOT ready; use New.
type ModelExpander struct {
	client            ModelClient
	model             string
	maxExpansionRatio float64
	options           map[string]any
	// keywordFloor is the minimum fraction of the original's significant
	// tokens that must survive into the expanded output. A drop below
	// this floor is treated as meaning-drift and the expander returns
	// the original prompt. Defaults to 0.8 — conservative but lenient
	// enough to tolerate rephrasing.
	keywordFloor float64
}

// Options bundles configuration for New.
type Options struct {
	// Model is the model name to drive. Must be non-empty.
	Model string
	// MaxExpansionRatio caps how much larger the output may be than the
	// input (by rune count). Matches DistributedConfig.PromptExpander.
	// Must be > 1.0; values ≤ 0 fall back to 3.0.
	MaxExpansionRatio float64
	// KeywordFloor is the survival threshold for the drift guard.
	// 0 selects the default (0.8).
	KeywordFloor float64
	// GenerateOptions is forwarded verbatim to the model client on each
	// expansion request. Nil is fine.
	GenerateOptions map[string]any
}

// New constructs a ModelExpander. The client and opts.Model are required.
func New(client ModelClient, opts Options) (*ModelExpander, error) {
	if client == nil {
		return nil, errors.New("expander: ModelClient is required")
	}
	if strings.TrimSpace(opts.Model) == "" {
		return nil, errors.New("expander: Model is required")
	}
	if opts.MaxExpansionRatio <= 1.0 {
		if opts.MaxExpansionRatio <= 0 {
			opts.MaxExpansionRatio = 3.0
		} else {
			return nil, errors.New("expander: MaxExpansionRatio must be > 1.0")
		}
	}
	if opts.KeywordFloor <= 0 || opts.KeywordFloor > 1.0 {
		opts.KeywordFloor = 0.8
	}
	return &ModelExpander{
		client:            client,
		model:             opts.Model,
		maxExpansionRatio: opts.MaxExpansionRatio,
		options:           opts.GenerateOptions,
		keywordFloor:      opts.KeywordFloor,
	}, nil
}

// systemPreamble is prepended to every expansion request. Its sole job
// is to constrain the model to a meaning-preserving rewrite.
const systemPreamble = `You are a prompt-expansion assistant. Rewrite the user's text to be more verbose, enumerated, and explicit WITHOUT adding new facts, changing meaning, removing information, answering the request, or adding commentary. Output ONLY the rewritten text, no preface, no explanation.

Text to rewrite:
`

// Expand implements Expander. The returned string is either a validated
// expansion or — on any guard violation, model error, or empty output —
// the original prompt.
func (e *ModelExpander) Expand(ctx context.Context, prompt string) (string, error) {
	if strings.TrimSpace(prompt) == "" {
		// Nothing to expand. Return as-is; this is not an error.
		return prompt, nil
	}
	out, err := e.client.Generate(ctx, e.model, systemPreamble+prompt, e.options)
	if err != nil {
		// Transport/model error is surfaced so the pipeline can decide
		// whether to retry or skip the expander stage.
		return prompt, err
	}
	out = strings.TrimSpace(out)
	if out == "" {
		return prompt, nil
	}
	if ok, _ := Guard(prompt, out, e.maxExpansionRatio, e.keywordFloor); !ok {
		return prompt, nil
	}
	return out, nil
}

// Guard reports whether `expanded` is an acceptable rewrite of
// `original` given the ratio and keyword-floor thresholds. The second
// return value is a short human-readable reason when the result is
// false, useful for logging.
//
// The guard is deliberately simple and deterministic; semantic-level
// checking is out of scope for Phase 4. Two checks:
//
//  1. Length ratio: len(expanded)/len(original) must be in
//     [1.0, maxRatio]. A rewrite shorter than the original is rejected
//     because "expansion" implies additive rewriting.
//  2. Keyword survival: a configurable fraction of the original's
//     significant tokens (alphanumeric, length ≥ 3, case-folded) must
//     appear in the expanded output. This is a crude but effective
//     proxy for "didn't drop the user's intent".
func Guard(original, expanded string, maxRatio, keywordFloor float64) (bool, string) {
	if strings.TrimSpace(expanded) == "" {
		return false, "expanded is empty"
	}
	origLen := len([]rune(original))
	expLen := len([]rune(expanded))
	if origLen == 0 {
		return true, ""
	}
	ratio := float64(expLen) / float64(origLen)
	if ratio < 1.0 {
		return false, "expanded is shorter than original"
	}
	if ratio > maxRatio {
		return false, "expansion ratio exceeds cap"
	}
	origTokens := significantTokens(original)
	if len(origTokens) == 0 {
		// Nothing significant to check — accept any non-empty result
		// that survived the ratio test.
		return true, ""
	}
	expLower := strings.ToLower(expanded)
	survived := 0
	for tok := range origTokens {
		if strings.Contains(expLower, tok) {
			survived++
		}
	}
	frac := float64(survived) / float64(len(origTokens))
	if frac < keywordFloor {
		return false, "keyword survival below floor"
	}
	return true, ""
}

// significantTokens returns the case-folded set of alphanumeric tokens
// of length ≥ 3 in s. Used by Guard as a stand-in for "user's intent
// tokens".
func significantTokens(s string) map[string]struct{} {
	out := make(map[string]struct{})
	var b strings.Builder
	flush := func() {
		if b.Len() >= 3 {
			out[strings.ToLower(b.String())] = struct{}{}
		}
		b.Reset()
	}
	for _, r := range s {
		if unicode.IsLetter(r) || unicode.IsDigit(r) {
			b.WriteRune(r)
		} else {
			flush()
		}
	}
	flush()
	return out
}

// Compile-time assertion.
var _ Expander = (*ModelExpander)(nil)

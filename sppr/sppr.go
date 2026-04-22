// Package sppr implements the Segment Preprocessing Prompt Renderer —
// the pipeline stage that turns a user prompt into an ordered list of
// Segments suitable for fan-out across Secondary nodes.
//
// Scope (Phase 4):
//
//   - Segment data model and a wire-agnostic Renderer interface.
//   - ModelRenderer: drives a model via a ModelClient, parses its JSON
//     output, and falls back to a single-segment rendering on ANY
//     malformed or unusable output (the spec's fail-closed contract).
//   - Plan: the high-level pipeline entry point that wires the Phase 1
//     small-job gate, the optional Prompt Expander, and the Renderer
//     into one call producing either a distributed plan or a
//     single-node fallback.
//
// Out of scope here: the actual orchestrator, node allocation, and
// correlation (Phase 5). This package is deliberately pure: given a
// prompt and some pluggable dependencies, produce a deterministic Plan.
package sppr

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"strings"
	"unicode"

	"github.com/ollama/ollama/distributed/config"
	"github.com/ollama/ollama/expander"
)

// ModelClient is the narrow contract this package depends on to drive a
// model. It mirrors the same-named interface in package expander so the
// two stages can share a single client implementation at call sites.
type ModelClient interface {
	Generate(ctx context.Context, model, prompt string, options map[string]any) (string, error)
}

// ModelClientFunc adapts a plain function to the ModelClient interface.
type ModelClientFunc func(ctx context.Context, model, prompt string, options map[string]any) (string, error)

// Generate implements ModelClient.
func (f ModelClientFunc) Generate(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
	return f(ctx, model, prompt, options)
}

// Segment is one unit of a segmented prompt.
type Segment struct {
	// ID uniquely identifies this segment within its parent job.
	// Populated by the renderer; stable across retries of the same
	// input.
	ID string
	// Order is the 0-based position of this segment in the original
	// prompt. The orchestrator uses Order to stitch responses back
	// together.
	Order int
	// Text is the segment payload sent to the assigned Secondary.
	Text string
	// Hints is a free-form map the renderer can use to annotate a
	// segment (e.g. suggested model, complexity score). Consumers MUST
	// treat unknown hints as informational only.
	Hints map[string]string
}

// Renderer segments a prompt into one or more Segments.
type Renderer interface {
	Render(ctx context.Context, prompt string) ([]Segment, error)
}

// RendererFunc adapts a plain function to the Renderer interface.
type RendererFunc func(ctx context.Context, prompt string) ([]Segment, error)

// Render implements Renderer.
func (f RendererFunc) Render(ctx context.Context, prompt string) ([]Segment, error) {
	return f(ctx, prompt)
}

// ModelRenderer is the production Renderer implementation. It drives a
// model with a strict JSON-output instruction and parses the result.
// On malformed output it returns a single-segment rendering (the
// entire prompt as Segment[0]) — matching the spec's "fail closed"
// rule.
type ModelRenderer struct {
	client  ModelClient
	model   string
	options map[string]any
}

// ModelRendererOptions bundles configuration for NewModelRenderer.
type ModelRendererOptions struct {
	// Model is the SPPR model name (e.g. "qwen2.5"). Required.
	Model string
	// GenerateOptions is forwarded verbatim to the model client. Nil is
	// fine.
	GenerateOptions map[string]any
}

// NewModelRenderer constructs a ModelRenderer. Client and opts.Model
// are required.
func NewModelRenderer(client ModelClient, opts ModelRendererOptions) (*ModelRenderer, error) {
	if client == nil {
		return nil, errors.New("sppr: ModelClient is required")
	}
	if strings.TrimSpace(opts.Model) == "" {
		return nil, errors.New("sppr: Model is required")
	}
	return &ModelRenderer{client: client, model: opts.Model, options: opts.GenerateOptions}, nil
}

// systemPreamble instructs the model to emit a strict JSON array of
// segment objects. Any deviation → fallback path.
const systemPreamble = `You are a prompt-segmentation assistant. Split the user's text into semantically coherent, independently answerable segments. Output ONLY a JSON array of objects; each object has a "text" string and an optional "hints" object of string→string. Do not include any commentary, prose, markdown fences, or trailing text. If the text cannot be meaningfully segmented, output a single-element array containing the whole text.

Text to segment:
`

type rawSegment struct {
	Text  string            `json:"text"`
	Hints map[string]string `json:"hints,omitempty"`
}

// Render implements Renderer. On any parsing or validation failure it
// returns a single-segment rendering containing the full prompt.
func (r *ModelRenderer) Render(ctx context.Context, prompt string) ([]Segment, error) {
	if strings.TrimSpace(prompt) == "" {
		return nil, errors.New("sppr: empty prompt")
	}
	slog.Debug("sppr: rendering", "model", r.model, "prompt_len", len(prompt))
	out, err := r.client.Generate(ctx, r.model, systemPreamble+prompt, r.options)
	if err != nil {
		slog.Warn("sppr: renderer model error; falling back to single segment", "model", r.model, "err", err)
		// Transport/model error surfaces as a single-segment fallback
		// with the error attached as a hint, so the orchestrator can
		// decide whether the upstream is healthy enough to dispatch.
		return []Segment{fallbackSegment(prompt, "renderer_error", err.Error())}, nil
	}
	segs, ok := parseSegments(prompt, out)
	if !ok || len(segs) == 0 {
		slog.Info("sppr: unparseable output; falling back to single segment", "raw_len", len(out))
		return []Segment{fallbackSegment(prompt, "fallback_reason", "unparseable_sppr_output")}, nil
	}
	slog.Debug("sppr: rendered", "segments", len(segs))
	return segs, nil
}

// parseSegments trims common decorations (markdown fences, leading
// prose) and attempts to JSON-decode an array of rawSegment. Returns
// (segments, ok).
func parseSegments(prompt, raw string) ([]Segment, bool) {
	s := strings.TrimSpace(raw)
	// Strip ``` fences if present.
	s = stripCodeFence(s)
	// Locate the first '[' and last ']' — tolerant of leading/trailing
	// prose the model might sneak in despite instructions.
	start := strings.Index(s, "[")
	end := strings.LastIndex(s, "]")
	if start < 0 || end <= start {
		return nil, false
	}
	s = s[start : end+1]
	var raws []rawSegment
	if err := json.Unmarshal([]byte(s), &raws); err != nil {
		return nil, false
	}
	out := make([]Segment, 0, len(raws))
	for i, rs := range raws {
		text := strings.TrimSpace(rs.Text)
		if text == "" {
			continue
		}
		out = append(out, Segment{
			ID:    fmt.Sprintf("seg-%04d", i),
			Order: len(out),
			Text:  text,
			Hints: rs.Hints,
		})
	}
	if len(out) == 0 {
		return nil, false
	}
	// Renumber Order so skipped empty entries do not leave gaps.
	for i := range out {
		out[i].Order = i
	}
	return out, true
}

func stripCodeFence(s string) string {
	if !strings.HasPrefix(s, "```") {
		return s
	}
	// Drop the opening fence line.
	if i := strings.IndexByte(s, '\n'); i >= 0 {
		s = s[i+1:]
	} else {
		return s
	}
	// Drop the closing fence if present.
	if j := strings.LastIndex(s, "```"); j >= 0 {
		s = s[:j]
	}
	return strings.TrimSpace(s)
}

func fallbackSegment(prompt, hintKey, hintVal string) Segment {
	seg := Segment{ID: "seg-0000", Order: 0, Text: prompt}
	if hintKey != "" {
		seg.Hints = map[string]string{hintKey: hintVal}
	}
	return seg
}

// ---------------------------------------------------------------------
// Pipeline (small-job gate → Expander → Renderer → Plan)
// ---------------------------------------------------------------------

// Plan is the output of the pipeline: either a distributed fan-out
// plan (len(Segments) ≥ 2, SingleNode == false) or a single-node
// fallback (len(Segments) == 1, SingleNode == true).
type Plan struct {
	// Segments are ordered and non-empty on success.
	Segments []Segment
	// SingleNode is true when the caller should bypass the orchestrator
	// and run the job on a single node (standalone-equivalent path).
	SingleNode bool
	// FallbackReason, when non-empty, explains why SingleNode is true.
	FallbackReason string
	// ExpandedPrompt, when non-empty, is the post-Expander text that
	// was actually segmented. Exposed for observability.
	ExpandedPrompt string
}

// PlanOptions wires the pipeline's dependencies.
type PlanOptions struct {
	// Cfg governs the small-job gate and (when Enabled) the Expander.
	Cfg *config.DistributedConfig
	// Expander is optional. Pass a non-nil value AND Cfg.PromptExpander.Enabled
	// to actually run the Expander stage.
	Expander expander.Expander
	// Renderer is required unless the small-job gate fires first.
	Renderer Renderer
}

// Plan runs the full pipeline for `prompt` and returns a Plan.
//
// Ordering:
//  1. config.IsSmallJob(prompt) → single-node fallback (no renderer call).
//  2. opts.Expander.Expand (if enabled) → possibly replacing prompt.
//  3. opts.Renderer.Render → Segments.
//  4. config.SegmentsBelowFallback(len(segments)) → single-node fallback.
//
// Any error from the Renderer is surfaced to the caller; callers that
// want fail-closed behavior can treat an error as "degrade to single
// node" themselves. This is intentional so operators can tell
// "orchestrator-level failure" apart from "segmentation genuinely
// produced one segment".
func Run(ctx context.Context, prompt string, opts PlanOptions) (Plan, error) {
	if opts.Cfg == nil {
		return Plan{}, errors.New("sppr: Cfg is required")
	}
	if opts.Cfg.IsSmallJob(prompt) {
		slog.Info("sppr: small-job gate fired; single-node fallback", "prompt_len", len(prompt), "threshold", opts.Cfg.SmallJob.PromptRuneThreshold)
		return Plan{
			Segments:       []Segment{fallbackSegment(prompt, "fallback_reason", "small_job_prompt")},
			SingleNode:     true,
			FallbackReason: "small_job_prompt",
		}, nil
	}
	if opts.Renderer == nil {
		return Plan{}, errors.New("sppr: Renderer is required")
	}
	work := prompt
	expanded := ""
	if opts.Cfg.PromptExpander.Enabled && opts.Expander != nil {
		// An error from the expander is non-fatal: the spec's fail-closed
		// posture says to proceed with the original prompt. We swallow
		// the error here (observable via ExpandedPrompt being empty)
		// but surface a hint on the returned plan.
		out, err := opts.Expander.Expand(ctx, prompt)
		if err != nil {
			slog.Warn("sppr: expander error; continuing with original prompt", "err", err)
		} else if out != "" && out != prompt {
			work = out
			expanded = out
		}
	}
	segs, err := opts.Renderer.Render(ctx, work)
	if err != nil {
		slog.Error("sppr: renderer error propagating to caller", "err", err)
		return Plan{}, err
	}
	plan := Plan{Segments: segs, ExpandedPrompt: expanded}
	if opts.Cfg.SegmentsBelowFallback(len(segs)) {
		plan.SingleNode = true
		plan.FallbackReason = "segments_below_threshold"
		slog.Info("sppr: below-threshold fallback", "segments", len(segs), "min_segments", opts.Cfg.SmallJob.MinSegments)
	} else {
		slog.Debug("sppr: plan ready", "segments", len(segs), "expander_used", expanded != "")
	}
	return plan, nil
}

// IsWhitespaceOnly reports whether s contains no non-whitespace runes.
// Exposed for callers that want to validate prompts before invoking Run.
func IsWhitespaceOnly(s string) bool {
	for _, r := range s {
		if !unicode.IsSpace(r) {
			return false
		}
	}
	return true
}

// Compile-time assertion.
var _ Renderer = (*ModelRenderer)(nil)

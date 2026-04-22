package sppr

import (
	"context"
	"errors"
	"strings"
	"testing"

	"github.com/ollama/ollama/distributed/config"
	"github.com/ollama/ollama/expander"
)

func defaultCfg() *config.DistributedConfig {
	c := config.Default()
	return &c
}

func TestNewModelRendererValidation(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "", nil
	})
	if _, err := NewModelRenderer(nil, ModelRendererOptions{Model: "m"}); err == nil {
		t.Fatal("nil client should error")
	}
	if _, err := NewModelRenderer(client, ModelRendererOptions{Model: " "}); err == nil {
		t.Fatal("empty model should error")
	}
	r, err := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if r == nil {
		t.Fatal("nil renderer")
	}
}

func TestRenderParsesJSONArray(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return `[{"text":"alpha"},{"text":"beta","hints":{"k":"v"}}]`, nil
	})
	r, _ := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	segs, err := r.Render(context.Background(), "hello world")
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 2 {
		t.Fatalf("got %d segments", len(segs))
	}
	if segs[0].Text != "alpha" || segs[1].Text != "beta" {
		t.Fatalf("unexpected segments: %+v", segs)
	}
	if segs[0].Order != 0 || segs[1].Order != 1 {
		t.Fatalf("order wrong")
	}
	if segs[1].Hints["k"] != "v" {
		t.Fatalf("hints not preserved")
	}
}

func TestRenderStripsCodeFences(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "```json\n[{\"text\":\"only\"}]\n```", nil
	})
	r, _ := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	segs, err := r.Render(context.Background(), "x")
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 1 || segs[0].Text != "only" {
		t.Fatalf("fence strip failed: %+v", segs)
	}
}

func TestRenderToleratesLeadingProse(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "Sure! Here you go: [{\"text\":\"a\"},{\"text\":\"b\"}]  thanks!", nil
	})
	r, _ := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	segs, _ := r.Render(context.Background(), "p")
	if len(segs) != 2 {
		t.Fatalf("got %d", len(segs))
	}
}

func TestRenderFallsBackOnMalformed(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "not json at all", nil
	})
	r, _ := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	segs, err := r.Render(context.Background(), "the-prompt")
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 1 || segs[0].Text != "the-prompt" {
		t.Fatalf("fallback text wrong: %+v", segs)
	}
	if segs[0].Hints["fallback_reason"] == "" {
		t.Fatalf("fallback hint missing")
	}
}

func TestRenderFallsBackOnEmptyArray(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return `[{"text":"   "},{"text":""}]`, nil
	})
	r, _ := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	segs, _ := r.Render(context.Background(), "p")
	if len(segs) != 1 || segs[0].Text != "p" {
		t.Fatalf("empty-array fallback wrong: %+v", segs)
	}
}

func TestRenderModelErrorYieldsFallbackWithHint(t *testing.T) {
	sentinel := errors.New("backend down")
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "", sentinel
	})
	r, _ := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	segs, err := r.Render(context.Background(), "p")
	if err != nil {
		t.Fatal(err)
	}
	if len(segs) != 1 || segs[0].Hints["renderer_error"] == "" {
		t.Fatalf("error-hint missing: %+v", segs)
	}
}

func TestRenderRejectsEmptyPrompt(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		t.Fatal("should not be called")
		return "", nil
	})
	r, _ := NewModelRenderer(client, ModelRendererOptions{Model: "m"})
	if _, err := r.Render(context.Background(), "   "); err == nil {
		t.Fatal("empty prompt should error")
	}
}

// Plan pipeline tests.

func TestPlanSmallJobShortCircuits(t *testing.T) {
	cfg := defaultCfg() // default PromptRuneThreshold = 400
	calls := 0
	r := RendererFunc(func(ctx context.Context, prompt string) ([]Segment, error) {
		calls++
		return []Segment{{Text: "x"}}, nil
	})
	p, err := Run(context.Background(), "tiny prompt", PlanOptions{Cfg: cfg, Renderer: r})
	if err != nil {
		t.Fatal(err)
	}
	if !p.SingleNode || p.FallbackReason != "small_job_prompt" {
		t.Fatalf("expected small-job short-circuit, got %+v", p)
	}
	if calls != 0 {
		t.Fatalf("renderer called %d times despite short-circuit", calls)
	}
}

func TestPlanRunsRendererForBigPrompt(t *testing.T) {
	cfg := defaultCfg()
	cfg.SmallJob.PromptRuneThreshold = 10 // force big-job path
	r := RendererFunc(func(ctx context.Context, prompt string) ([]Segment, error) {
		return []Segment{{Text: "a"}, {Text: "b"}, {Text: "c"}}, nil
	})
	p, err := Run(context.Background(), "hello world this is long", PlanOptions{Cfg: cfg, Renderer: r})
	if err != nil {
		t.Fatal(err)
	}
	if p.SingleNode {
		t.Fatalf("unexpected fallback: %+v", p)
	}
	if len(p.Segments) != 3 {
		t.Fatalf("segments: %d", len(p.Segments))
	}
}

func TestPlanFallsBackOnFewSegments(t *testing.T) {
	cfg := defaultCfg()
	cfg.SmallJob.PromptRuneThreshold = 10
	cfg.SmallJob.MinSegments = 3
	r := RendererFunc(func(ctx context.Context, prompt string) ([]Segment, error) {
		return []Segment{{Text: "only one"}, {Text: "two"}}, nil
	})
	p, err := Run(context.Background(), "sufficiently long prompt", PlanOptions{Cfg: cfg, Renderer: r})
	if err != nil {
		t.Fatal(err)
	}
	if !p.SingleNode || p.FallbackReason != "segments_below_threshold" {
		t.Fatalf("expected below-threshold fallback: %+v", p)
	}
}

func TestPlanExpanderEnabled(t *testing.T) {
	cfg := defaultCfg()
	cfg.SmallJob.PromptRuneThreshold = 10
	cfg.PromptExpander.Enabled = true
	exp := expander.ExpanderFunc(func(ctx context.Context, p string) (string, error) {
		return p + " (expanded detail)", nil
	})
	var rendered string
	r := RendererFunc(func(ctx context.Context, prompt string) ([]Segment, error) {
		rendered = prompt
		return []Segment{{Text: "a"}, {Text: "b"}}, nil
	})
	p, err := Run(context.Background(), "this is the original prompt", PlanOptions{Cfg: cfg, Expander: exp, Renderer: r})
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(rendered, "expanded detail") {
		t.Fatalf("renderer did not receive expanded prompt: %q", rendered)
	}
	if p.ExpandedPrompt == "" {
		t.Fatal("ExpandedPrompt not populated")
	}
}

func TestPlanExpanderErrorIsSwallowed(t *testing.T) {
	cfg := defaultCfg()
	cfg.SmallJob.PromptRuneThreshold = 10
	cfg.PromptExpander.Enabled = true
	exp := expander.ExpanderFunc(func(ctx context.Context, p string) (string, error) {
		return p, errors.New("expander transport down")
	})
	var got string
	r := RendererFunc(func(ctx context.Context, prompt string) ([]Segment, error) {
		got = prompt
		return []Segment{{Text: "a"}, {Text: "b"}}, nil
	})
	p, err := Run(context.Background(), "the original prompt stays", PlanOptions{Cfg: cfg, Expander: exp, Renderer: r})
	if err != nil {
		t.Fatalf("expander error should not propagate: %v", err)
	}
	if got != "the original prompt stays" {
		t.Fatalf("renderer should have received original, got %q", got)
	}
	if p.ExpandedPrompt != "" {
		t.Fatalf("ExpandedPrompt should be empty on expander error")
	}
}

func TestPlanRequiresCfg(t *testing.T) {
	if _, err := Run(context.Background(), "p", PlanOptions{}); err == nil {
		t.Fatal("missing cfg should error")
	}
	cfg := defaultCfg()
	cfg.SmallJob.PromptRuneThreshold = 10 // force big-job path so renderer check kicks in
	if _, err := Run(context.Background(), "long enough prompt to avoid small-job gate", PlanOptions{Cfg: cfg}); err == nil {
		t.Fatal("missing renderer should error")
	}
}

func TestPlanRendererErrorSurfaces(t *testing.T) {
	cfg := defaultCfg()
	cfg.SmallJob.PromptRuneThreshold = 10
	sentinel := errors.New("renderer bad")
	r := RendererFunc(func(ctx context.Context, prompt string) ([]Segment, error) {
		return nil, sentinel
	})
	_, err := Run(context.Background(), "long prompt here", PlanOptions{Cfg: cfg, Renderer: r})
	if !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want sentinel", err)
	}
}

func TestIsWhitespaceOnly(t *testing.T) {
	if !IsWhitespaceOnly("   \t\n  ") {
		t.Fatal("expected true for whitespace-only")
	}
	if IsWhitespaceOnly("  x ") {
		t.Fatal("expected false for non-whitespace")
	}
}

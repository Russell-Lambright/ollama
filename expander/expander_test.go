package expander

import (
	"context"
	"errors"
	"strings"
	"testing"
)

func TestNewValidation(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "", nil
	})
	if _, err := New(nil, Options{Model: "m"}); err == nil {
		t.Fatal("nil client should error")
	}
	if _, err := New(client, Options{Model: ""}); err == nil {
		t.Fatal("empty model should error")
	}
	if _, err := New(client, Options{Model: "m", MaxExpansionRatio: 0.5}); err == nil {
		t.Fatal("ratio < 1 should error")
	}
	// Zero ratio → default 3.0.
	e, err := New(client, Options{Model: "m"})
	if err != nil {
		t.Fatal(err)
	}
	if e.maxExpansionRatio != 3.0 {
		t.Fatalf("default ratio = %v", e.maxExpansionRatio)
	}
	// Invalid floor → default 0.8.
	e, _ = New(client, Options{Model: "m", KeywordFloor: 1.5})
	if e.keywordFloor != 0.8 {
		t.Fatalf("invalid floor not defaulted: %v", e.keywordFloor)
	}
}

func TestExpandHappy(t *testing.T) {
	orig := "Write a function to parse a YAML file."
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		if !strings.Contains(prompt, orig) {
			t.Errorf("system preamble did not include original prompt")
		}
		return "Please write a function that will parse a YAML file. The function should read YAML input.", nil
	})
	e, _ := New(client, Options{Model: "m"})
	out, err := e.Expand(context.Background(), orig)
	if err != nil {
		t.Fatal(err)
	}
	if out == orig {
		t.Fatal("expected expanded output, got original")
	}
}

func TestExpandEmptyPromptPassthrough(t *testing.T) {
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		t.Fatal("should not be called for empty prompt")
		return "", nil
	})
	e, _ := New(client, Options{Model: "m"})
	out, err := e.Expand(context.Background(), "   ")
	if err != nil || out != "   " {
		t.Fatalf("empty passthrough: out=%q err=%v", out, err)
	}
}

func TestExpandModelErrorReturnsOriginal(t *testing.T) {
	orig := "original"
	sentinel := errors.New("transport down")
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "", sentinel
	})
	e, _ := New(client, Options{Model: "m"})
	out, err := e.Expand(context.Background(), orig)
	if !errors.Is(err, sentinel) {
		t.Fatalf("err=%v want sentinel", err)
	}
	if out != orig {
		t.Fatalf("out=%q want original on error", out)
	}
}

func TestExpandEmptyOutputReturnsOriginal(t *testing.T) {
	orig := "abc def ghi"
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "   ", nil
	})
	e, _ := New(client, Options{Model: "m"})
	out, err := e.Expand(context.Background(), orig)
	if err != nil {
		t.Fatal(err)
	}
	if out != orig {
		t.Fatalf("out=%q want original on empty", out)
	}
}

func TestExpandGuardRejectsShrinkage(t *testing.T) {
	orig := "A reasonably long original prompt that should not shrink"
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return "short", nil
	})
	e, _ := New(client, Options{Model: "m"})
	out, _ := e.Expand(context.Background(), orig)
	if out != orig {
		t.Fatalf("shrinkage not rejected; out=%q", out)
	}
}

func TestExpandGuardRejectsOverExpansion(t *testing.T) {
	orig := "Short prompt here."
	bloated := strings.Repeat("x y z ", 1000)
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return bloated, nil
	})
	e, _ := New(client, Options{Model: "m", MaxExpansionRatio: 3.0})
	out, _ := e.Expand(context.Background(), orig)
	if out != orig {
		t.Fatalf("over-expansion not rejected; out=%q", out[:40]+"…")
	}
}

func TestExpandGuardRejectsKeywordDrop(t *testing.T) {
	orig := "parse YAML configuration files into structured data"
	// Rewrite drops the salient tokens.
	drift := "please write some code that does things with input data"
	client := ModelClientFunc(func(ctx context.Context, model, prompt string, options map[string]any) (string, error) {
		return drift, nil
	})
	e, _ := New(client, Options{Model: "m"})
	out, _ := e.Expand(context.Background(), orig)
	if out != orig {
		t.Fatalf("keyword drift not rejected; out=%q", out)
	}
}

func TestGuardEdgeCases(t *testing.T) {
	// Empty expanded is always rejected.
	if ok, _ := Guard("abc", "", 3.0, 0.8); ok {
		t.Fatal("empty expanded should fail guard")
	}
	// Empty original accepts any non-empty expansion.
	if ok, _ := Guard("", "anything", 3.0, 0.8); !ok {
		t.Fatal("empty original should accept")
	}
	// Original with only short tokens (no significant set) accepts.
	if ok, _ := Guard("a b c", "a bb cc dd ee", 10.0, 0.8); !ok {
		t.Fatal("short-token original should accept")
	}
}

func TestExpanderFuncAdapter(t *testing.T) {
	var called bool
	f := ExpanderFunc(func(ctx context.Context, p string) (string, error) {
		called = true
		return p + "!", nil
	})
	out, _ := f.Expand(context.Background(), "hi")
	if !called || out != "hi!" {
		t.Fatalf("adapter not invoked: called=%v out=%q", called, out)
	}
}

# Examples

Worked examples showing how to drive the distributed framework from
both code and the command line. All code snippets compile against the
tree in this repository.

## 1. Dispatching a job programmatically

```go
package main

import (
    "context"
    "fmt"
    "log"

    "github.com/ollama/ollama/distributed/config"
    "github.com/ollama/ollama/distributed/orchestrator"
    "github.com/ollama/ollama/distributed/transport"
    "github.com/ollama/ollama/sppr"
)

// dispatcher adapts transport.Loopback (or any production transport)
// to the orchestrator.Dispatcher contract.
type dispatcher struct{ l *transport.Loopback }

func (d *dispatcher) Assign(id transport.NodeID, a transport.AssignSegment) error {
    return d.l.Assign(id, a)
}
func (d *dispatcher) Cancel(ctx context.Context, id transport.NodeID, r transport.CancelRequest) error {
    return d.l.Cancel(ctx, id, r)
}

func main() {
    cfg := config.Default()
    cfg.DefaultCollective = "c1"
    loop := transport.NewLoopback(0)
    o, err := orchestrator.New(orchestrator.Options{
        Cfg:        &cfg,
        Dispatcher: &dispatcher{l: loop},
    })
    if err != nil {
        log.Fatal(err)
    }
    // (… register Secondaries and move them to Available …)

    res, err := o.Dispatch(context.Background(), orchestrator.Job{
        ID:    "demo-42",
        Model: "qwen2.5:7b",
        Segments: []sppr.Segment{
            {ID: "s0", Order: 0, Text: "Summarize section 1."},
            {ID: "s1", Order: 1, Text: "Summarize section 2."},
        },
    })
    if err != nil {
        log.Fatal(err)
    }
    fmt.Println(res.Coalesced)
}
```

For a complete wiring (Orchestrator + Loopback + multiple Secondaries
driving actual executors) see
[`distributed/integration/integration_test.go`](../../distributed/integration/integration_test.go).

## 2. Running SPPR manually

```go
plan, err := sppr.Run(ctx, "Write a proposal then list risks.", sppr.PlanOptions{
    Cfg:      &cfg,
    Renderer: sppr.NewModelRenderer(modelClient, cfg.SPPRModel, nil, 0),
    Expander: nil, // or expander.NewModelExpander(…)
})
if err != nil {
    return err
}
if plan.SingleNode {
    // Fall back to a single-node, standalone-equivalent run.
}
```

The `plan.FallbackReason` tells you why (`small_job_prompt`,
`segments_below_threshold`, `unparseable_sppr_output`, or
`renderer_error`).

## 3. Custom Coalescer — QA coherence pass

```go
qaCoalescer := orchestrator.CoalescerFunc(func(ctx context.Context, outputs []string) (string, error) {
    joined := strings.Join(outputs, "\n\n")
    return qaModel.Generate(ctx, "qwen2.5:7b",
        "Smooth the following multi-segment answer into one coherent response:\n\n"+joined, nil)
})

o, _ := orchestrator.New(orchestrator.Options{
    Cfg:        &cfg,
    Dispatcher: disp,
    Coalescer:  qaCoalescer, // overrides the default ConcatCoalescer
})
```

## 4. Model sync plan

```go
syncer, _ := modelsync.New(modelsync.Options{
    Provider: modelsync.ManifestProviderFunc{
        ExpectedFn: func(ctx context.Context) (modelsync.Manifest, error) {
            return modelsync.NewManifest(
                modelsync.Entry{Name: "qwen2.5:7b",  Digest: "sha256:abc"},
                modelsync.Entry{Name: "gemma2:2b",  Digest: "sha256:def"},
            ), nil
        },
        ObservedFn: diskInspect, // walks $OLLAMA_MODELS on the node
    },
    Puller:      ollamaPuller,   // wraps existing pull pipeline
    MaxAttempts: 3,
})

if err := syncer.Sync(ctx); err != nil {
    // Node transitions to Failed; surfaced via state listener and
    // reconciled in the dashboard.
    return err
}
```

## 5. Fanout across a collective

```go
results := modelsync.Fanout(ctx, nodeIDs, func(ctx context.Context, nodeID string) error {
    return rpc.TriggerSyncOn(nodeID)
})
for _, r := range results {
    if r.Err != nil {
        log.Printf("node %s sync failed: %v", r.NodeID, r.Err)
    }
}
```

Per-node failures don't cancel peer operations.

## 6. Mounting the dashboard

```go
d, _ := dashboard.New(dashboard.Options{
    Source:         orch, // *orchestrator.Orchestrator implements Source
    PersonaApplier: applier, // optional
    Title:          "Ollama — Production Cluster",
})

mux := http.NewServeMux()
mux.Handle("/dashboard/", http.StripPrefix("/dashboard", d.Handler()))
mux.HandleFunc("/api/generate", /* existing Ollama handler */)
_ = http.ListenAndServe(":11434", mux)
```

## 7. curl — inspecting a live Primary

```bash
# Current cluster view
curl -s http://primary:11434/api/distributed/snapshot | jq

# Apply a persona collective-wide
curl -s -X POST http://primary:11434/api/distributed/persona \
  -H 'Content-Type: application/json' \
  -d '{"collective":"c1","persona":"coder"}' | jq
```

## 8. Observability — filtering slog output

Each package uses the `component` attribute, so you can filter on it
when piping through any structured-log tool:

```bash
OLLAMA_DEBUG=1 ollama serve --mode=primary 2>&1 \
  | jq 'select(.component | startswith("distributed/"))' \
  | jq 'select(.level == "INFO")'
```

Useful filters:

```bash
# Only state transitions
jq 'select(.msg == "secondary: state transition")'

# Only allocation decisions
jq 'select(.msg == "orchestrator: allocation complete")'

# Starvation-index changes (drifts from configured default)
jq 'select(.msg == "orchestrator: starvation index updated")'
```

## 9. Config overrides — priority order

```
CLI flag  >  env var  >  YAML config file  >  built-in default
```

Example — operator wants a one-off run with a different SPPR model:

```bash
ollama serve --mode=primary --sppr-model=qwen2.5:0.5b
# …even though distributed.yaml has sppr_model: gemma2:2b.
```

# `distributed/` — MPI-style orchestration for Ollama

This package houses all code for the distributed MPI-style Ollama framework.
It is **feature-gated**: unless the operator starts Ollama with
`--mode=primary` / `--mode=secondary` (or sets `OLLAMA_NODE_MODE`), nothing in
this tree affects runtime behavior.

The authoritative specification is [`../DISTRIBUTED_ARCHITECTURE.md`](../DISTRIBUTED_ARCHITECTURE.md).

## Module boundaries

| Subpackage                  | Purpose                                                         | Phase |
| --------------------------- | --------------------------------------------------------------- | ----- |
| `distributed/config`        | Typed config, persona definitions, YAML loader, env overrides   | 1     |
| `distributed/cancel`        | Typed cancellation contract (caller-cancel, node-failure)       | 1     |
| `distributed/node`          | Node identity (UUID, hostname, collective, advertised LPU)      | 2     |
| `distributed/state`         | Formal state machine for Secondary lifecycle                    | 2     |
| `distributed/transport`     | Primary ↔ Secondary RPC (register/heartbeat/assign/cancel/…)    | 2     |
| `distributed/secondary`     | Secondary mode runtime (sync → available → execute)             | 3     |
| `distributed/orchestrator`  | Primary scheduler, starvation monitor, execution/correlation    | 5     |
| `distributed/modelsync`     | Model fan-out and manifest-diff sync                            | 6     |

Two new packages live outside `distributed/`:

- `expander/` — Prompt Expander (optional pre-SPPR stage; adds verbosity
  without changing meaning).
- `sppr/` — Segment Preprocessing Prompt Renderer (pipeline stage).
- `integration/distributed_test.go` — Testcontainers-based integration tests.

## Phased roadmap

This initiative ships in eight sequential PRs. Every PR must:
- compile (`go build ./...`),
- keep all existing tests green,
- leave `main` usable in **standalone** mode (no behavior change when no
  `--mode` flag / `OLLAMA_NODE_MODE` is set).

| Phase | Scope                                                   | Status    |
| ----- | ------------------------------------------------------- | --------- |
| 0     | Foundations: package skeleton, feature gate, docs       | ✅ landed |
| 1     | Configuration: typed config, personas, YAML loader      | ✅ landed |
| 2     | Node identity, state machine, transport                 | ✅ landed |
| 3     | Secondary mode runtime                                  | 🚧 in progress |
| 4     | SPPR pipeline stage                                     | ⏳        |
| 5a    | Orchestrator scheduler + starvation monitor             | ⏳        |
| 5b    | Execution, correlation, QA coherence pass               | ⏳        |
| 6     | Model synchronization                                   | ⏳        |
| 7     | Distributed Dashboard UI                                | ⏳        |
| 8     | Integration tests & coverage gate (≥ 90 % on new code)  | ⏳        |

## Cross-cutting rules

1. **Reuse over rewrite.** Inference, model pull, templating, discovery, and
   event streaming already exist in this repository — call into them, do not
   reimplement.
2. **Fail closed.** Malformed SPPR output → single-segment standalone
   fallback. Zero available nodes → the spec's fixed rejection message. Any
   secondary error → `Failed` state surfaced in the UI.
3. **Argument over config.** Where both a CLI flag and a config-file field
   exist, the flag always wins.
4. **Standalone is the default.** If `--mode` is omitted, none of this code
   may be in the hot path.

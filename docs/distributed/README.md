# Distributed Ollama — Documentation

This directory documents the distributed MPI-style Ollama framework built
across phases 0–8.

| Document | Contents |
| -------- | -------- |
| [`overview.md`](overview.md) | System architecture, packages, component diagram |
| [`lifecycle.md`](lifecycle.md) | Node state machine + transition rules |
| [`request-flow.md`](request-flow.md) | End-to-end request sequence from HTTP → response |
| [`operations.md`](operations.md) | Running Primary/Secondary, configuration, dashboard |
| [`examples.md`](examples.md) | Practical examples: configs, curl invocations, code |

The authoritative specification is
[`../../DISTRIBUTED_ARCHITECTURE.md`](../../DISTRIBUTED_ARCHITECTURE.md).
The package-level roadmap and phase status live in
[`../../distributed/README.md`](../../distributed/README.md).

## TL;DR

```mermaid
flowchart LR
  User([User / API client]) -->|HTTP| Primary
  Primary -- SPPR segments --> Orchestrator
  Orchestrator -- Assign --> S1[Secondary A]
  Orchestrator -- Assign --> S2[Secondary B]
  Orchestrator -- Assign --> S3[Secondary C]
  S1 -- events --> Orchestrator
  S2 -- events --> Orchestrator
  S3 -- events --> Orchestrator
  Orchestrator -- coalesced --> Primary
  Primary -- response --> User
  Ops([Operator]) -->|browser| Dashboard
  Dashboard -->|snapshot| Orchestrator
```

If `--mode` is not set, none of this is in the hot path — Ollama runs
exactly as it did before.

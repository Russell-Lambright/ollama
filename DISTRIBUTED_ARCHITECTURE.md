# Distributed MPI-Style Ollama Framework — Architecture

This document is the spec of record for transforming Ollama into a distributed
computing framework inspired by the Message Passing Interface (MPI) used in
Beowulf high-performance computing clusters. It enables parallel prompt
processing across multiple nodes while maintaining centralized orchestration
and intelligent job distribution.

The full problem statement is tracked in the original feature issue and
summarized below. The authoritative implementation plan is under
[`distributed/README.md`](distributed/README.md).

---

## 1. Core architecture

### Primary node (Orchestrator)
- Central coordinator and job scheduler.
- **Does not execute inference jobs in collective mode** (orchestration only).
- Manages model distribution, synchronization, and quality assurance of
  correlated results.
- Hosts the Distributed Dashboard UI.

### Secondary nodes (Workers)
- Run in `secondary` mode and report to a Primary host (CLI argument
  overrides the configuration file).
- Execute assigned prompt segments.
- Support optional MCP (Model Context Protocol) integration.

### Collectives
- Secondary nodes are grouped into logical collectives.
- Nodes declare their collective membership at startup.
- Callers specify which collective to target when submitting a prompt.
- **Default behavior: if no collective is specified, Ollama operates in
  standalone mode** — byte-identical to `ollama serve` today.

## 2. Segment Preprocessing Prompt Renderer (SPPR)

A new pipeline stage inserted **before** the thinking/reasoning stage. It
rewrites an incoming prompt into a set of segments with explicit dependency
chains, emitted as a strict JSON schema that the orchestrator can schedule on.

- Uses a configurable linguistics-trained model (default: **Qwen2.5**;
  overridable per-request and via `OLLAMA_SPPR_MODEL` for tests).
- Degrades gracefully on malformed output (single-segment fallback).

### 2a. Prompt Expander (optional pre-SPPR stage)

An **optional** stage that runs *before* SPPR and rewrites the prompt into
a more verbose but semantically equivalent form, giving SPPR more surface
area to segment on.

- Disabled by default; enable via `prompt_expander.enabled: true` in
  `~/.ollama/distributed.yaml` after measuring that it improves
  segmentation quality for your workload.
- Strict contract: the expander may add verbosity, restate, enumerate, or
  rephrase, but **must not** alter meaning, add new facts, or remove
  information. A guard discards expansions above `max_expansion_ratio`
  (default 3×) and reverts to the original prompt.
- Reuses the SPPR model by default (`prompt_expander.model: ""`) so
  operators don't have to configure two.

### 2b. Small-job fallback

For jobs that are too small to benefit from distribution, the framework
**skips the distributed pipeline entirely** and executes the job on a
single node (standalone-equivalent). Two configurable thresholds, either
of which triggers fallback:

| Field                              | Default | Meaning                                                     |
| ---------------------------------- | ------- | ----------------------------------------------------------- |
| `small_job.prompt_rune_threshold`  | 400     | Raw prompts under this many runes skip SPPR entirely        |
| `small_job.min_segments`           | 2       | If SPPR emits fewer than N segments, run on a single node   |

Set either to `0` to disable that particular check. The fallback path is
normal success (not an error) — the caller still receives a response; it
just wasn't sharded.

## 3. Persona management

- Personas are predefined in configuration (`~/.ollama/distributed.yaml`).
- Secondary nodes can request a target collective and persona at startup.
- The Primary can force a persona on a node, or overlay one via the UI.
- **Collective-wide persona application:** the Primary can apply a
  persona to every node in a collective as a single operation (Node Tab,
  Collective View). The orchestrator fans the change out to all
  current members; each affected node transitions through `Training`
  (per §4) back to `Available`. The operation is atomic from the
  operator's point of view — the UI reports aggregate progress and
  per-node success/failure — but individual nodes apply independently,
  so a failure on one node does not roll back the others. Any nodes
  that join the collective after the operation also inherit the
  persona until it is explicitly changed or removed.
- Personas persist until explicitly removed or replaced.

## 4. Node states

Secondary nodes transition through:

| State                  | Meaning                                              |
| ---------------------- | ---------------------------------------------------- |
| `Available`            | Ready to execute jobs                                |
| `Syncing`              | Synchronizing models; not eligible for jobs          |
| `Failed`               | Sync failure or other error                          |
| `Offline`              | Connectivity lost                                    |
| `Processing-Thinking`  | In the thinking phase                                |
| `Processing-Reasoning` | In the reasoning phase                               |
| `Processing`           | Running a job (no thinking/reasoning)                |
| `Training`             | Learning a new persona                               |

Only `Available` nodes receive job assignments.

## 5. Job orchestration

- **Allocation formula** (clamped to `[0, MaxNodesPerCollective]`):
  ```
  n = min(
    requested_nodes (if supplied),
    ceil(segment_count / concurrency_hint),
    available_nodes_in_collective,
    floor(MaxNodesPerCollective * STARVATION_INDEX),
  )
  ```
- **Starvation monitor** adjusts `STARVATION_INDEX` (bounded `[0.1, 1.0]`)
  based on queue depth and failure rate.
- **No available nodes** → return the fixed rejection message:
  *"Sorry but we are not able to take your call at this time. Please try again later."*
- Reasoning/thinking outputs are suppressed in collective mode until a
  coherent coalescing mechanism is implemented.

### Cancellation

Every distributed job is governed by a shared cancellation context. Two
triggers cause the orchestrator to cancel the job and release every
assigned Secondary back to the Available pool:

1. **Caller cancels** — the originating HTTP request is cancelled (client
   disconnect, explicit abort, request timeout). The orchestrator must
   propagate the cancel to every Secondary currently running a segment of
   the job. Secondaries abort their in-flight work, clean up state, and
   transition back to `Available`.
2. **Node fails mid-job** — if any Secondary assigned to the job
   transitions to `Failed`, the whole job is cancelled. Peer Secondaries
   are signalled, released, and the caller receives an error identifying
   the failed node.

The transport layer (see §8) is responsible for the actual abort signal:
gRPC closes the bidirectional stream with a cancel status; HTTP/2 + SSE
closes the event-stream connection. The typed contract lives in
`distributed/cancel`.

## 6. UI — Distributed Dashboard

A new tab in the desktop/webview UI with:
- **Secondary Nodes** table: state badge, LPU (weighted GPU + CPU average),
  collective, persona, context menu (*Remove persona*, *Apply new persona*).
- **Collective View**: member count / max, live average LPU, default
  collective selector, and a context menu to **apply or remove a
  persona across the entire collective** in one action (see §3). The
  UI shows aggregate progress (e.g. *"Applying ‘code-reviewer’: 4 / 7
  nodes training…"*) and surfaces per-node outcomes on completion.

## 7. Testing

- ≥ 90 % coverage on new packages (`distributed/`, `sppr/`).
- Zero regression in standalone mode — all existing tests must pass unchanged.
- Integration: `testcontainers-go` spins up Primary + N Secondaries using a
  single tiny model.

## 8. Configuration & operational modes

| Mode                  | How to start                                                                     | Behavior                          |
| --------------------- | -------------------------------------------------------------------------------- | --------------------------------- |
| Standalone (default)  | `ollama serve`                                                                   | Today's Ollama, unchanged         |
| Primary (collective)  | `ollama serve --mode=primary [--collective=<default>] [--transport=<proto>]`     | Orchestrator + UI                 |
| Secondary (collective)| `ollama serve --mode=secondary --primary=<host:port> --collective=<name> [--transport=<proto>]` | Worker node      |

CLI arguments always override the configuration file.

### Transport (`--transport`)

The Primary↔Secondary wire protocol is operator-selectable:

| Value       | Description                                                                |
| ----------- | -------------------------------------------------------------------------- |
| `grpc`      | **Default.** Bidirectional gRPC streams, idiomatic Go.                     |
| `http2-sse` | HTTP/2 with Server-Sent Events — matches Ollama's existing REST streaming. |

Selectable via `--transport`, `OLLAMA_TRANSPORT`, or the `transport` field
in `~/.ollama/distributed.yaml` (precedence in that order). Both transports
must implement identical cancellation semantics (§5).

## 9. Design principles

1. **Separation of concerns** — Primary orchestrates; Secondaries execute.
2. **Intelligent allocation** — not every job needs every node.
3. **Persona persistence** — nodes keep personas until explicitly changed.
4. **Graceful degradation** — clear messaging when capacity is unavailable.
5. **Configurability** — every knob is overridable for tests and production.

---

## Implementation roadmap

See [`distributed/README.md`](distributed/README.md) for the phased plan.
Phases 0 and 1 (foundations + configuration) land first; each subsequent
phase is a separately reviewable PR that leaves `main` functional in
standalone mode.

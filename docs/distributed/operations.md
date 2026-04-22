# Operations Guide

How to run a distributed collective, configure it, and use the
dashboard.

## Modes

Ollama selects its mode from either the `--mode` CLI flag or the
`OLLAMA_NODE_MODE` environment variable (the flag wins).

| Mode | Behavior |
| ---- | -------- |
| _(unset)_ | Standalone. No distributed code is on the hot path. |
| `primary` | Orchestrator + dashboard + HTTP API for users |
| `secondary` | Worker node; registers with a Primary, executes segments |

## Primary startup

```bash
ollama serve \
  --mode=primary \
  --host=0.0.0.0:11434 \
  --distributed-config=/etc/ollama/distributed.yaml
```

Relevant flags (all have config-file equivalents — CLI wins):

| Flag | Purpose |
| ---- | ------- |
| `--mode=primary` | Enables Primary mode |
| `--distributed-config=PATH` | YAML config (see below) |
| `--transport={grpc,http2-sse}` | Secondary↔Primary wire protocol |
| `--dashboard-addr=HOST:PORT` | Address for the operator dashboard |

## Secondary startup

```bash
ollama serve \
  --mode=secondary \
  --primary=primary.cluster.local:11434 \
  --collective=c1 \
  --persona=coder
```

| Flag | Purpose |
| ---- | ------- |
| `--mode=secondary` | Enables Secondary mode |
| `--primary=HOST:PORT` | Primary to register with |
| `--collective=NAME` | Collective label to join |
| `--persona=NAME` | Optional initial persona |

## Configuration file

```yaml
# /etc/ollama/distributed.yaml
max_nodes_per_collective: 8
default_collective: c1
transport: grpc
starvation_index: 1.0
sppr_model: qwen2.5:0.5b

fine_tuning:
  temperature: 0.7
  top_p: 0.95
  repeat_penalty: 1.1

prompt_expander:
  enabled: false
  max_expansion_ratio: 3.0

small_job:
  prompt_rune_threshold: 400
  min_segments: 2

personas:
  - name: coder
    description: Careful senior engineer
    system_prompt: |
      You are a senior software engineer. Think through problems carefully
      and cite sources inline.
    preferred_model: qwen2.5:7b
    tags: [engineering]
  - name: summarizer
    description: Concise factual summaries
    system_prompt: |
      Summarize the given text in 3 bullet points.
```

## Environment variables

| Variable | Equivalent to |
| -------- | ------------- |
| `OLLAMA_NODE_MODE` | `--mode` |
| `OLLAMA_PRIMARY` | `--primary` |
| `OLLAMA_COLLECTIVE` | `--collective` |
| `OLLAMA_TRANSPORT` | `--transport` |
| `OLLAMA_DISTRIBUTED_CONFIG` | `--distributed-config` |

## Dashboard

The Primary exposes an operator dashboard that renders the current
registry, collectives, and active jobs. It polls a JSON snapshot
endpoint every two seconds.

### Routes

| Method | Path | Response |
| ------ | ---- | -------- |
| GET | `/` | HTML single-page dashboard |
| GET | `/api/distributed/snapshot` | JSON `orchestrator.Snapshot` |
| POST | `/api/distributed/persona` | Collective-wide persona apply |

### Snapshot response

```json
{
  "nodes": [
    {
      "id": "a51b64c22b77e46f3d3bb12d7366bfe6",
      "hostname": "gpu-01",
      "collective": "c1",
      "persona": "coder",
      "state": "Available",
      "advertised_lpu": 3.2,
      "current_lpu": 3.1,
      "last_heartbeat": "2026-04-22T06:15:03Z",
      "active_jobs": ["job-42"]
    }
  ],
  "collectives": [
    { "name": "c1", "max_nodes": 8, "node_count": 3, "available_count": 2, "avg_lpu": 2.4, "node_ids": ["…"] }
  ],
  "jobs": [
    { "id": "job-42", "collective": "c1", "segments": 3, "nodes": ["…"], "completed": 1 }
  ],
  "starvation_index": 0.82,
  "max_nodes_per_collective": 8,
  "captured_at": "2026-04-22T06:15:03Z"
}
```

### Collective-wide persona apply

```bash
curl -X POST http://primary:11434/api/distributed/persona \
  -H 'Content-Type: application/json' \
  -d '{"collective":"c1","persona":"coder"}'
```

Each node in the collective applies the persona independently. One
node's failure does **not** roll back the others — matching the spec's
"individual nodes apply independently" semantics. The response lists
per-node outcomes:

```json
{
  "collective": "c1",
  "persona": "coder",
  "succeeded": 2,
  "failed": 1,
  "results": [
    { "node_id": "…", "ok": true },
    { "node_id": "…", "ok": true },
    { "node_id": "…", "ok": false, "error": "sync failed: …" }
  ]
}
```

## Observability

Every distributed package emits structured `log/slog` events. Stable
keys you can grep on:

| Key | Produced by | Meaning |
| --- | ----------- | ------- |
| `component` | all | `distributed/<pkg>` — package emitting the log |
| `node` | secondary, orchestrator | Node ID (opaque 32-char hex) |
| `job` | orchestrator | Job ID |
| `segment` | orchestrator, secondary | Segment ID |
| `from`, `to` | state, secondary | State transition endpoints |
| `starvation_index` | orchestrator | Live scheduler throttle |
| `err` | all | Error context |

Enable verbose output with `OLLAMA_DEBUG=1` to see Debug-level events
(heartbeats, allocation details, back-pressure drops).

## Rejection message

When a Primary has **zero** Available nodes in the target collective,
the orchestrator returns the fixed rejection string spelled out in the
spec:

```
Sorry but we are not able to take your call at this time. Please try again later.
```

Callers receive this both as the error's message and as `ErrNoAvailableNodes`
for programmatic handling.

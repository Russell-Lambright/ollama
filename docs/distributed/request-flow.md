# Request Flow

This document traces a request from the moment the HTTP handler accepts
it through segmentation, distribution, coalescing, and response.

## Happy path — distributed job

```mermaid
sequenceDiagram
  autonumber
  participant User
  participant HTTP as Primary HTTP
  participant Exp as expander
  participant S as sppr.Run
  participant O as orchestrator
  participant T as transport
  participant A as Secondary A
  participant B as Secondary B
  participant C as orchestrator.Coalescer

  User->>HTTP: POST /api/chat { prompt, collective }
  HTTP->>Exp: Expand(prompt)
  Exp-->>HTTP: expanded prompt (or original)
  HTTP->>S: Run(expanded, cfg)
  S-->>HTTP: Plan{Segments[0..n], SingleNode?}
  alt plan.SingleNode
    HTTP->>HTTP: run on a single node (standalone-equivalent)
  else distributed
    HTTP->>O: Dispatch(Job{ID, Segments, Model})
    O->>O: AvailableNodes(collective) + allocate()
    O->>T: Assign(nodeA, seg0)
    O->>T: Assign(nodeB, seg1)
    T->>A: AssignSegment
    T->>B: AssignSegment
    par parallel execution
      A->>A: state: Available → Processing
      A->>T: SegmentEvent tokens
      A->>T: SegmentEvent done
      A->>A: state: Processing → Available
    and
      B->>B: state: Available → Processing
      B->>T: SegmentEvent tokens
      B->>T: SegmentEvent done
      B->>B: state: Processing → Available
    end
    T->>O: SendSegmentEvent(s)
    O->>O: stitch outputs by Segment.Order
    O->>C: Coalesce(outputs)
    C-->>O: coalesced
    O-->>HTTP: Result{Outputs, Coalesced, NodeIDs}
    HTTP-->>User: 200 { response }
  end
```

## Small-job fallback

Two thresholds (either triggers fallback) are configured in
`DistributedConfig.SmallJob`:

| Field | Default | Meaning |
| ----- | ------- | ------- |
| `prompt_rune_threshold` | 400 | Raw prompts under this many runes skip SPPR entirely |
| `min_segments` | 2 | If SPPR emits fewer than N segments, run on a single node |

```mermaid
flowchart TD
  Prompt([Incoming prompt]) --> CheckSize{runes < threshold?}
  CheckSize -- yes --> SingleNode[Run on a single node]
  CheckSize -- no --> RunSPPR[sppr.Run]
  RunSPPR --> CheckCount{segments < min?}
  CheckCount -- yes --> SingleNode
  CheckCount -- no --> Dispatch[orchestrator.Dispatch]
  SingleNode --> Response([Response to user])
  Dispatch --> Response
```

Fallback is a **normal success**, not an error — the caller still gets
a response; the request just wasn't sharded.

## Cancellation paths

### Caller cancels

```mermaid
sequenceDiagram
  participant User
  participant HTTP
  participant O as orchestrator
  participant T as transport
  participant A as Secondary A
  participant B as Secondary B

  User->>HTTP: close connection / abort
  HTTP-->>O: ctx cancelled
  O->>O: cancelJob(ReasonCallerCancelled)
  O->>T: Cancel(nodeA)
  O->>T: Cancel(nodeB)
  T->>A: CancelRequest
  T->>B: CancelRequest
  A->>A: abort executor, → Available
  B->>B: abort executor, → Available
  O-->>HTTP: Result{CancelReason=caller_cancelled}, error
```

### Node fails mid-job

```mermaid
sequenceDiagram
  participant A as Secondary A
  participant T as transport
  participant O as orchestrator
  participant B as Secondary B

  A->>T: StateUpdate{To: Failed}
  T->>O: ReportStateUpdate
  O->>O: cancelJob(ReasonNodeFailed, failedNodeID=A)
  O->>T: Cancel(B)
  T->>B: CancelRequest
  B->>B: abort, → Available
  O-->>Caller: Result{CancelReason=node_failed, FailedNodeID=A}, error
```

## Starvation index feedback loop

```mermaid
flowchart LR
  Job -- success --> Monitor[starvation monitor]
  Job -- failure --> Monitor
  Monitor -- recompute --> Index[StarvationIndex\n[0.1, 1.0]]
  Index --> Allocate[orchestrator.allocate]
  Allocate --> Dispatch[Next Dispatch]
  Dispatch --> Job
```

The monitor uses a proportional heuristic: `index = target × (1 − failure_rate) + MinStarvationIndex × failure_rate`,
clamped to `[MinStarvationIndex, MaxStarvationIndex]`.

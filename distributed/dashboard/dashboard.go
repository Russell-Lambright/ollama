// Package dashboard is the Primary-side operator surface for the
// distributed MPI-style framework.
//
// Scope (Phase 7):
//
//   - JSON snapshot API (`GET /api/distributed/snapshot`) backed by
//     `orchestrator.Orchestrator.Snapshot()`.
//   - A single-page HTML UI (`GET /`) that polls the snapshot endpoint
//     and renders the Node Tab and Collective View described in the
//     architecture spec §6.
//   - Persona action endpoint (`POST /api/distributed/persona`) that
//     drives a collective-wide persona apply via a pluggable
//     `PersonaApplier`. One node's failure does not cancel the others
//     (matches the spec's "individual nodes apply independently"
//     semantics).
//
// The dashboard is intentionally transport-light: it only needs a
// snapshot producer (any `Source`) and an optional `PersonaApplier`.
// Wiring it into a real webview or desktop UI is a separate concern —
// this package exposes an `http.Handler` that can be mounted by the
// Primary's HTTP server.
package dashboard

import (
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"log/slog"
	"net/http"
	"strings"
	"sync"

	"github.com/ollama/ollama/distributed/orchestrator"
)

// Source is anything that can produce an orchestrator snapshot. The
// concrete *orchestrator.Orchestrator satisfies this; tests can
// substitute a stub.
type Source interface {
	Snapshot() orchestrator.Snapshot
}

// PersonaApplier applies a persona to a single node. The Dashboard
// fans out across a collective by calling Apply once per node in
// parallel. Implementations must be safe for concurrent use.
type PersonaApplier interface {
	Apply(ctx context.Context, nodeID, persona string) error
}

// PersonaApplierFunc adapts a plain function to PersonaApplier.
type PersonaApplierFunc func(ctx context.Context, nodeID, persona string) error

// Apply implements PersonaApplier.
func (f PersonaApplierFunc) Apply(ctx context.Context, nodeID, persona string) error {
	return f(ctx, nodeID, persona)
}

// Options configures a Dashboard.
type Options struct {
	Source         Source         // required
	PersonaApplier PersonaApplier // optional; when nil the persona endpoint 501s
	// Title shown in the HTML page header. Empty → "Ollama — Distributed".
	Title string
}

// Dashboard exposes the operator HTTP surface. Zero value is NOT ready.
type Dashboard struct {
	opts Options
	log  *slog.Logger
}

// New constructs a Dashboard.
func New(opts Options) (*Dashboard, error) {
	if opts.Source == nil {
		return nil, errors.New("dashboard: Source is required")
	}
	if opts.Title == "" {
		opts.Title = "Ollama — Distributed"
	}
	return &Dashboard{
		opts: opts,
		log:  slog.With("component", "distributed/dashboard"),
	}, nil
}

// Handler returns an http.Handler that serves the dashboard on the
// following routes:
//
//	GET  /                            → HTML page
//	GET  /api/distributed/snapshot    → JSON snapshot
//	POST /api/distributed/persona     → collective-wide persona apply
func (d *Dashboard) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /api/distributed/snapshot", d.handleSnapshot)
	mux.HandleFunc("POST /api/distributed/persona", d.handlePersona)
	mux.HandleFunc("GET /", d.handleIndex)
	return mux
}

func (d *Dashboard) handleSnapshot(w http.ResponseWriter, r *http.Request) {
	snap := d.opts.Source.Snapshot()
	w.Header().Set("Content-Type", "application/json")
	// Do not cache — dashboards poll this endpoint continuously.
	w.Header().Set("Cache-Control", "no-store")
	if err := json.NewEncoder(w).Encode(snap); err != nil {
		d.log.Warn("dashboard: snapshot encode failed", "err", err)
	}
}

// PersonaRequest is the payload for POST /api/distributed/persona.
type PersonaRequest struct {
	// Collective scopes the apply. Required.
	Collective string `json:"collective"`
	// Persona is the persona name to apply (empty means "remove persona").
	Persona string `json:"persona"`
}

// PersonaResult is one row of the response — per-node outcome.
type PersonaResult struct {
	NodeID string `json:"node_id"`
	OK     bool   `json:"ok"`
	Error  string `json:"error,omitempty"`
}

// PersonaResponse is the body returned to the client.
type PersonaResponse struct {
	Collective string          `json:"collective"`
	Persona    string          `json:"persona"`
	Results    []PersonaResult `json:"results"`
	Succeeded  int             `json:"succeeded"`
	Failed     int             `json:"failed"`
}

func (d *Dashboard) handlePersona(w http.ResponseWriter, r *http.Request) {
	if d.opts.PersonaApplier == nil {
		http.Error(w, "persona applier not configured", http.StatusNotImplemented)
		return
	}
	var req PersonaRequest
	if err := json.NewDecoder(http.MaxBytesReader(w, r.Body, 1<<16)).Decode(&req); err != nil {
		http.Error(w, fmt.Sprintf("invalid JSON: %v", err), http.StatusBadRequest)
		return
	}
	if strings.TrimSpace(req.Collective) == "" {
		http.Error(w, "collective is required", http.StatusBadRequest)
		return
	}
	snap := d.opts.Source.Snapshot()
	var nodeIDs []string
	for _, c := range snap.Collectives {
		if c.Name == req.Collective {
			nodeIDs = append(nodeIDs, c.NodeIDs...)
			break
		}
	}
	if len(nodeIDs) == 0 {
		http.Error(w, fmt.Sprintf("collective %q has no registered nodes", req.Collective), http.StatusNotFound)
		return
	}
	d.log.Info("dashboard: collective persona apply starting",
		"collective", req.Collective,
		"persona", req.Persona,
		"nodes", len(nodeIDs),
	)

	// Fan out in parallel. Matches spec §3: individual nodes apply
	// independently; one failure does NOT roll back the others.
	results := make([]PersonaResult, len(nodeIDs))
	var wg sync.WaitGroup
	for i, id := range nodeIDs {
		wg.Add(1)
		go func(i int, id string) {
			defer wg.Done()
			err := d.opts.PersonaApplier.Apply(r.Context(), id, req.Persona)
			pr := PersonaResult{NodeID: id, OK: err == nil}
			if err != nil {
				pr.Error = err.Error()
				d.log.Warn("dashboard: persona apply failed", "node", id, "err", err)
			}
			results[i] = pr
		}(i, id)
	}
	wg.Wait()

	resp := PersonaResponse{
		Collective: req.Collective,
		Persona:    req.Persona,
		Results:    results,
	}
	for _, r := range results {
		if r.OK {
			resp.Succeeded++
		} else {
			resp.Failed++
		}
	}
	d.log.Info("dashboard: collective persona apply done",
		"collective", req.Collective,
		"persona", req.Persona,
		"succeeded", resp.Succeeded,
		"failed", resp.Failed,
	)
	w.Header().Set("Content-Type", "application/json")
	_ = json.NewEncoder(w).Encode(resp)
}

func (d *Dashboard) handleIndex(w http.ResponseWriter, r *http.Request) {
	// Any unmatched GET falls through to here; surface a clean 404 for
	// anything that is clearly not the index page.
	if r.URL.Path != "/" {
		http.NotFound(w, r)
		return
	}
	w.Header().Set("Content-Type", "text/html; charset=utf-8")
	w.Header().Set("Cache-Control", "no-store")
	title := htmlEscape(d.opts.Title)
	_, _ = fmt.Fprintf(w, indexHTML, title, title)
}

// htmlEscape is a tiny subset-replacement of html.EscapeString used
// for the page title only. Kept local to avoid pulling net/html on a
// path that only handles a single trusted string.
func htmlEscape(s string) string {
	r := strings.NewReplacer("&", "&amp;", "<", "&lt;", ">", "&gt;", "\"", "&quot;", "'", "&#39;")
	return r.Replace(s)
}

// indexHTML is a single-page dashboard that polls /api/distributed/snapshot.
// No build step, no npm dependency — intentionally tiny and inline so
// operators can curl or diff it. Any literal `%` in CSS/JS below must
// be doubled for fmt.Fprintf.
const indexHTML = `<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>%s</title>
<style>
  :root { --bg:#0f1115; --fg:#e7e9ee; --muted:#9aa3af; --accent:#4ea1ff; --ok:#3fb950; --warn:#d29922; --err:#f85149; --card:#161922; --border:#252a36; }
  * { box-sizing: border-box; }
  body { margin:0; font-family: ui-sans-serif, system-ui, -apple-system, "Segoe UI", Helvetica, Arial; background: var(--bg); color: var(--fg); }
  header { padding: 16px 24px; border-bottom: 1px solid var(--border); display:flex; justify-content:space-between; align-items:center; }
  header h1 { margin:0; font-size: 18px; font-weight: 600; }
  header .meta { color: var(--muted); font-size: 12px; }
  main { padding: 24px; display: grid; grid-template-columns: 1fr 1fr; gap: 24px; }
  section { background: var(--card); border:1px solid var(--border); border-radius: 10px; padding: 16px; }
  section h2 { margin: 0 0 12px 0; font-size: 14px; color: var(--muted); text-transform: uppercase; letter-spacing: 0.08em; }
  table { width:100%%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 6px 8px; border-bottom: 1px solid var(--border); text-align: left; vertical-align: top; }
  th { color: var(--muted); font-weight: 500; }
  .badge { display: inline-block; padding: 2px 8px; border-radius: 999px; font-size: 11px; }
  .s-Available { background: rgba(63,185,80,0.15); color: var(--ok); }
  .s-Syncing { background: rgba(78,161,255,0.15); color: var(--accent); }
  .s-Failed { background: rgba(248,81,73,0.15); color: var(--err); }
  .s-Offline { background: rgba(154,163,175,0.15); color: var(--muted); }
  .s-Training { background: rgba(210,153,34,0.15); color: var(--warn); }
  .s-Processing, .s-Processing-Thinking, .s-Processing-Reasoning { background: rgba(78,161,255,0.15); color: var(--accent); }
  .row-full { grid-column: 1 / -1; }
  .mono { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); }
  .err { color: var(--err); }
  .bar { height: 4px; background: var(--border); border-radius: 2px; overflow: hidden; margin-top: 4px; }
  .bar > span { display:block; height: 100%%; background: var(--accent); }
  form { display: flex; gap: 8px; flex-wrap: wrap; align-items: center; margin-top: 12px; }
  input, button, select { font: inherit; padding: 6px 10px; background: #1b1f2a; color: var(--fg); border: 1px solid var(--border); border-radius: 6px; }
  button { cursor: pointer; }
  button:hover { border-color: var(--accent); }
  #personaResult { margin-top: 8px; font-size: 12px; }
</style>
</head>
<body>
<header>
  <h1 id="title">%s</h1>
  <div class="meta" id="meta">connecting…</div>
</header>
<main>
  <section>
    <h2>Secondary Nodes</h2>
    <table>
      <thead><tr><th>ID</th><th>Host</th><th>Collective</th><th>State</th><th>LPU</th><th>Persona</th><th>Jobs</th></tr></thead>
      <tbody id="nodes"></tbody>
    </table>
  </section>
  <section>
    <h2>Collectives</h2>
    <table>
      <thead><tr><th>Name</th><th>Members</th><th>Available</th><th>Avg LPU</th></tr></thead>
      <tbody id="collectives"></tbody>
    </table>
    <form id="personaForm">
      <select id="personaCollective"></select>
      <input id="personaName" placeholder="persona name (empty = remove)">
      <button type="submit">Apply across collective</button>
    </form>
    <div id="personaResult"></div>
  </section>
  <section class="row-full">
    <h2>Active Jobs</h2>
    <table>
      <thead><tr><th>Job</th><th>Collective</th><th>Segments</th><th>Progress</th><th>Nodes</th></tr></thead>
      <tbody id="jobs"></tbody>
    </table>
  </section>
</main>
<script>
  const $ = (id) => document.getElementById(id);
  function shortID(id) { return id && id.length > 10 ? id.slice(0,8) + "…" : id || ""; }
  function render(snap) {
    $("meta").textContent = "starvation " + snap.starvation_index.toFixed(2) + " · max/collective " + snap.max_nodes_per_collective + " · " + new Date(snap.captured_at).toLocaleTimeString();
    const nodes = $("nodes");
    nodes.innerHTML = "";
    for (const n of snap.nodes || []) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td class=mono>" + shortID(n.id) + "</td>" +
        "<td>" + (n.hostname || "") + "</td>" +
        "<td>" + (n.collective || "") + "</td>" +
        "<td><span class='badge s-" + n.state + "'>" + n.state + "</span></td>" +
        "<td>" + n.current_lpu.toFixed(2) + "</td>" +
        "<td>" + (n.persona || "—") + "</td>" +
        "<td>" + ((n.active_jobs || []).length) + "</td>";
      nodes.appendChild(tr);
    }
    const coll = $("collectives");
    coll.innerHTML = "";
    const sel = $("personaCollective");
    const previousSel = sel.value;
    sel.innerHTML = "";
    for (const c of snap.collectives || []) {
      const tr = document.createElement("tr");
      tr.innerHTML = "<td>" + c.name + "</td>" +
        "<td>" + c.node_count + (c.max_nodes > 0 ? " / " + c.max_nodes : "") + "</td>" +
        "<td>" + c.available_count + "</td>" +
        "<td>" + c.avg_lpu.toFixed(2) + "</td>";
      coll.appendChild(tr);
      const opt = document.createElement("option");
      opt.value = c.name; opt.textContent = c.name;
      sel.appendChild(opt);
    }
    if (previousSel) sel.value = previousSel;
    const jobs = $("jobs");
    jobs.innerHTML = "";
    for (const j of snap.jobs || []) {
      const pct = j.segments > 0 ? Math.round(100 * j.completed / j.segments) : 0;
      const tr = document.createElement("tr");
      tr.innerHTML = "<td class=mono>" + j.id + "</td>" +
        "<td>" + (j.collective || "") + "</td>" +
        "<td>" + j.segments + "</td>" +
        "<td>" + j.completed + "/" + j.segments + " <div class=bar><span style='width:" + pct + "%%'></span></div></td>" +
        "<td class=mono>" + (j.nodes || []).map(shortID).join(", ") + "</td>";
      jobs.appendChild(tr);
    }
  }
  async function poll() {
    try {
      const r = await fetch("/api/distributed/snapshot");
      const snap = await r.json();
      render(snap);
    } catch (e) {
      $("meta").textContent = "disconnected — " + e.message;
    }
  }
  $("personaForm").addEventListener("submit", async (ev) => {
    ev.preventDefault();
    const body = { collective: $("personaCollective").value, persona: $("personaName").value };
    const r = await fetch("/api/distributed/persona", { method: "POST", headers: {"Content-Type":"application/json"}, body: JSON.stringify(body) });
    const out = $("personaResult");
    if (!r.ok) {
      out.innerHTML = "<span class=err>" + await r.text() + "</span>";
      return;
    }
    const data = await r.json();
    out.textContent = "Applied '" + (data.persona||"(removed)") + "' to " + (data.collective) + " — " + data.succeeded + " ok, " + data.failed + " failed";
  });
  poll();
  setInterval(poll, 2000);
</script>
</body>
</html>
`

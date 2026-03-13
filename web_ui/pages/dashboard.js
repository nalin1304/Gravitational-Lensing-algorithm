/* Dashboard Page */
const { api, fmtSci, toast } = window.LensPINN || {};
const L = () => window.LensPINN;

export function render() {
  return `
    <div class="grid-4 mb-20">
      <div class="stat-card">
        <div class="stat-label">System Status</div>
        <div class="stat-value" id="dHealthStatus">—</div>
        <div class="stat-meta" id="dHealthMeta">Checking...</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Test Suite</div>
        <div class="stat-value" id="dTestCount" style="color:var(--success)">—</div>
        <div class="stat-meta" id="dTestMeta">Loading...</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">GPU Available</div>
        <div class="stat-value" id="dGPU">—</div>
        <div class="stat-meta">Hardware backend</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">API Version</div>
        <div class="stat-value" id="dVersion">—</div>
        <div class="stat-meta">JAX/Equinox</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Database</div>
        <div class="stat-value" id="dbStatus">—</div>
        <div class="stat-meta">Connectivity</div>
      </div>
    </div>

    <div class="grid-2 gap-20">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Available Models</span>
        </div>
        <div id="dModels"><p class="section-desc">Loading...</p></div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">API Statistics</span>
        </div>
        <div id="dStats"><p class="section-desc">Loading...</p></div>
      </div>
    </div>

    <div class="card" style="margin-top:20px">
      <div class="card-header">
        <span class="card-title">Benchmark Scripts</span>
        <span class="card-subtitle" style="margin:0">Publication-ready outputs</span>
      </div>
      <table class="data-table">
        <thead><tr><th>Script</th><th>Purpose</th><th>Output</th></tr></thead>
        <tbody>
          <tr><td><code>ablation_study.py</code></td><td>Checkpoint-backed component study (5 configs)</td><td>results/ablation_table.tex</td></tr>
          <tr><td><code>validate_real_data.py</code></td><td>SLACS image-space diagnostic (--use-real)</td><td>results/real_data/</td></tr>
          <tr><td><code>sota_comparison.py</code></td><td>Checkpoint-backed neural vs analytic benchmark (4 methods)</td><td>results/sota_comparison_table.tex</td></tr>
          <tr><td><code>uncertainty_calibration.py</code></td><td>Reliability diagram, ECE, coverage on held-out synthetic analogs</td><td>results/uncertainty_calibration.png</td></tr>
          <tr><td><code>scalability_benchmark.py</code></td><td>Grid scaling (16→512)</td><td>results/scalability_analysis.png</td></tr>
          <tr><td><code>reproduce.sh</code></td><td>One-command full reproducibility</td><td>All results/</td></tr>
        </tbody>
      </table>
    </div>
  `;
}

export async function init() {
  const P = L();
  try {
    const h = await P.api("/health", { auth: false });
    document.getElementById("dHealthStatus").textContent = h.status === "healthy" ? "Online" : h.status;
    document.getElementById("dHealthStatus").style.color = h.status === "healthy" ? "var(--success)" : "var(--warning)";
    document.getElementById("dHealthMeta").textContent = h.timestamp?.slice(0, 19) || "";
    document.getElementById("dGPU").textContent = h.gpu_available ? "Yes" : "CPU Only";
    document.getElementById("dGPU").style.color = h.gpu_available ? "var(--success)" : "var(--text-secondary)";
    document.getElementById("dVersion").textContent = h.version || "2.0";
    const dbEl = document.getElementById('dbStatus');
    if (dbEl) {
      dbEl.textContent = h.database_connected ? '✓ Connected' : '✗ Disconnected';
      dbEl.className = h.database_connected ? 'text-success' : 'text-danger';
    }
  } catch (e) { console.error("Health check failed:", e); document.getElementById("dHealthStatus").textContent = "Offline"; }

  try {
    const models = await P.api("/api/v1/models", { auth: false });
    const el = document.getElementById("dModels");
    if (models.models && models.models.length) {
      el.innerHTML = models.models.map(m => `
        <div class="metric-row">
          <span class="metric-key">${P.esc(m.name || m.model_name || 'PINN')}</span>
          <span class="badge ${m.supports_inference ? 'badge-success' : 'badge-warning'}">${P.esc(m.status || 'unknown')}</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Inference readiness</span>
          <span class="metric-val">${m.supports_inference ? 'checkpoint + runtime ready' : 'checkpoint/runtime unavailable'}</span>
        </div>
      `).join("");
    } else {
      el.innerHTML = `<p class="section-desc" style="color:var(--text-muted)">No model metadata available</p>`;
    }
  } catch (e) { console.error("Models load failed:", e); document.getElementById("dModels").innerHTML = `<p class="section-desc" style="color:var(--text-muted)">Could not load models</p>`; }

  try {
    const stats = await P.api("/api/v1/stats", { auth: false });
    const el = document.getElementById("dStats");
    const rows = Object.entries(stats).filter(([, v]) => typeof v !== "object" || v === null).map(([k, v]) =>
      `<div class="metric-row"><span class="metric-key">${P.esc(k.replace(/_/g, ' '))}</span><span class="metric-val">${typeof v === 'number' ? P.fmtSci(v) : P.esc(String(v))}</span></div>`
    ).join("");
    el.innerHTML = rows || `<p class="section-desc">No stats available</p>`;
    const tcEl = document.getElementById("dTestCount");
    const tmEl = document.getElementById("dTestMeta");
    const regression = stats.regression_summary || {};
    if (tcEl) {
      if (regression.passed != null) {
        tcEl.textContent = String(regression.passed);
      } else if (regression.checks_passed != null && regression.checks_total != null) {
        tcEl.textContent = `${regression.checks_passed}/${regression.checks_total}`;
      } else {
        tcEl.textContent = "—";
      }
    }
    if (tmEl) {
      if (regression.passed != null) {
        tmEl.textContent = `${regression.skipped ?? '—'} skipped · ${regression.failed ?? '—'} failed · API jobs: ${stats.total_jobs ?? '—'}`;
      } else {
        tmEl.textContent = `${regression.publication_gate_passed ? 'Gate PASS' : 'Gate status unavailable'} · API jobs: ${stats.total_jobs ?? '—'}`;
      }
    }
  } catch (e) { console.error("Stats load failed:", e); document.getElementById("dStats").innerHTML = `<p class="section-desc" style="color:var(--text-muted)">Could not load stats</p>`; }
}

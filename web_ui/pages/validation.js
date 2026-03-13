/* Validation Page — SLACS, Calibration, Ablation */
const L = () => window.LensPINN;
const asBool = (value) => value === true || value === "true" || value === 1 || value === "1";

export function render() {
  return `
    <div class="mb-16" style="display:flex;gap:10px;align-items:center">
      <button id="vLoad" class="btn btn-primary">Load All Results</button>
      <span class="section-desc" style="margin:0">Benchmark results from publication pipeline</span>
    </div>

    <div class="grid-2 gap-20 mb-20">
      <div class="card">
        <div class="card-header">
          <span class="card-title">SLACS Survey Lenses</span>
          <span class="badge badge-info" id="vSlacsCount">—</span>
        </div>
        <div id="vSlacs"><p class="section-desc">Click "Load All Results"</p></div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Uncertainty Calibration</span>
          <span class="badge badge-purple" id="vCalBadge">—</span>
        </div>
        <div id="vCal"><p class="section-desc">Click "Load All Results"</p></div>
      </div>
    </div>

    <div class="grid-2 gap-20">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Ablation Study</span>
          <span class="card-subtitle">Component contribution analysis</span>
        </div>
        <div id="vAbl"><p class="section-desc">Click "Load All Results"</p></div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Regression Suite</span>
          <button id="vTestRefresh" class="btn btn-sm btn-secondary" title="Refresh test count from API">↻ Refresh</button>
        </div>
        <div id="vTestStats">
          <div class="metric-row"><span class="metric-key">Tests Passed</span><span class="metric-val" id="vTestPassed" style="color:var(--success)">—</span></div>
          <div class="metric-row"><span class="metric-key">Tests Skipped</span><span class="metric-val" id="vTestSkipped" style="color:var(--warning)">—</span></div>
          <div class="metric-row"><span class="metric-key">Failures</span><span class="metric-val" id="vTestFailed">—</span></div>
          <div class="metric-row"><span class="metric-key">Verification Mode</span><span class="metric-val" id="vTestMode">Loading...</span></div>
          <div class="metric-row"><span class="metric-key">Last Run</span><span class="metric-val" id="vTestLastRun">—</span></div>
        </div>
      </div>
    </div>
  `;
}

export async function init() {
  const P = L();
  const applyRegressionStats = async () => {
    try {
      const stats = await P.api("/api/v1/stats", { auth: false });
      const regression = stats.regression_summary || {};
      document.getElementById("vTestPassed").textContent = regression.passed != null ? String(regression.passed) : (regression.checks_passed != null ? String(regression.checks_passed) : "—");
      document.getElementById("vTestSkipped").textContent = regression.skipped != null ? String(regression.skipped) : "—";
      const failures = regression.failed != null ? regression.failed : null;
      const failEl = document.getElementById("vTestFailed");
      failEl.textContent = failures != null ? String(failures) : "—";
      failEl.style.color = (failures != null && failures > 0) ? 'var(--danger, #ff4444)' : 'var(--success)';
      document.getElementById("vTestMode").textContent = regression.status || (regression.publication_gate_passed ? "publication_gate" : "unavailable");
      document.getElementById("vTestLastRun").textContent = regression.generated_at_utc ? regression.generated_at_utc.replace("T", " ").slice(0, 19) : "—";
    } catch {
      P.toast("Could not reach API stats", "error");
    }
  };

  await applyRegressionStats();

  document.getElementById("vLoad").addEventListener("click", async () => {
    P.showLoading("Loading validation results...");

    // SLACS
    try {
      const data = await P.api("/api/v1/validation/slacs", { auth: false });
      const lenses = Array.isArray(data) ? data.filter(d => d.name) : [];
      const pass = lenses.filter(l => asBool(l.passed)).length;
      document.getElementById("vSlacsCount").textContent = `${pass}/${lenses.length} passed`;
      document.getElementById("vSlacsCount").className = `badge ${pass === lenses.length ? 'badge-success' : 'badge-warning'}`;
      document.getElementById("vSlacs").innerHTML = `
        <table class="data-table">
          <thead><tr><th>Lens</th><th>RMSE</th><th>SSIM</th><th>χ²ᵣ</th><th>Source</th><th>Scope</th><th>Status</th></tr></thead>
          <tbody>${lenses.map(l => `
            <tr>
              <td style="font-weight:500">${P.esc((l.name || '').replace('SDSS ', ''))}</td>
              <td>${(l.rmse || 0).toFixed(4)}</td>
              <td>${(l.ssim || 0).toFixed(3)}</td>
              <td>${(l.reduced_chi2 || 0).toFixed(1)}</td>
              <td>${P.esc(l.data_source || 'unknown')}</td>
              <td>${P.esc(l.validation_scope || l.prediction_mode || 'unknown')}</td>
              <td><span class="badge ${asBool(l.passed) ? 'badge-success' : 'badge-danger'}">${asBool(l.passed) ? 'Pass' : 'Fail'}</span></td>
            </tr>
          `).join("")}</tbody>
        </table>
        <p class="section-desc" style="margin-top:12px">
          SLACS rows are rendered with artifact provenance. Image-space diagnostics should not be interpreted as direct convergence-map reconstruction.
        </p>`;
    } catch { document.getElementById("vSlacs").innerHTML = `<p class="section-desc">No SLACS data — run <code>python scripts/validate_real_data.py</code></p>`; }

    // Calibration
    try {
      const data = await P.api("/api/v1/validation/calibration", { auth: false });
      const s = Array.isArray(data) ? data.find(d => d.summary)?.summary : data;
      if (s) {
        const meanEce = s.mean_ece || 0;
        const coverage90 = s.mean_coverage_90 || 0;
        const badgeClass = meanEce < 0.05 ? 'badge-success' : meanEce < 0.15 ? 'badge-warning' : 'badge-danger';
        const badgeLabel = meanEce < 0.05 ? "Calibrated" : `ECE: ${meanEce.toFixed(3)}`;
        document.getElementById("vCalBadge").textContent = badgeLabel;
        document.getElementById("vCalBadge").className = `badge ${badgeClass}`;
        document.getElementById("vCal").innerHTML = `
          <div class="metric-row"><span class="metric-key">Mean ECE</span><span class="metric-val">${meanEce.toFixed(4)}</span></div>
          <div class="metric-row"><span class="metric-key">Coverage @90%</span><span class="metric-val">${coverage90.toFixed(3)}</span></div>
          <div class="metric-row"><span class="metric-key">UQ-Error Correlation</span><span class="metric-val">${(s.mean_uq_error_correlation || 0).toFixed(3)}</span></div>
          <div class="metric-row"><span class="metric-key">Prediction Mode</span><span class="metric-val">${P.esc(s.prediction_mode || 'unknown')}</span></div>
          <div class="metric-row"><span class="metric-key">Evaluation Scope</span><span class="metric-val">${P.esc(s.publication_scope || s.evaluation_mode || 'unknown')}</span></div>
        `;
      }
    } catch { document.getElementById("vCal").innerHTML = `<p class="section-desc">No calibration data — run <code>python scripts/uncertainty_calibration.py</code></p>`; }

    // Ablation
    try {
      const data = await P.api("/api/v1/validation/ablation", { auth: false });
      if (Array.isArray(data) && data.length) {
        document.getElementById("vAbl").innerHTML = `
          <table class="data-table">
            <thead><tr><th>Configuration</th><th>RMSE</th><th>SSIM</th></tr></thead>
            <tbody>${data.map(r => `
              <tr><td>${P.esc(r.config || '')}</td><td>${(r.rmse_mean ?? r.mean_rmse ?? 0).toFixed(4)}</td><td>${(r.ssim_mean ?? r.mean_ssim ?? 0).toFixed(3)}</td></tr>
            `).join("")}</tbody>
          </table>`;
      }
    } catch { document.getElementById("vAbl").innerHTML = `<p class="section-desc">No ablation data — run <code>python scripts/ablation_study.py</code></p>`; }

    P.hideLoading();
    P.toast("Validation results loaded", "success");
  });

  // Live test count refresh — pulls API job stats as proxy
  document.getElementById("vTestRefresh")?.addEventListener("click", async () => {
    await applyRegressionStats();
    P.toast("Regression summary refreshed", "info");
  });
}

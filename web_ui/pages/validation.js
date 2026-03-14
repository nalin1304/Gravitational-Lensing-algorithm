/* Validation Page — SLACS, Calibration, Ablation */
const L = () => window.LensPINN;
const asBool = (value) => value === true || value === "true" || (typeof value === 'number' && value > 0);

export function render() {
  return `
    <div class="mb-24">
      <h1 class="page-title">Validation & Benchmarking</h1>
      <p class="section-desc" style="max-width:700px">
        Scientific validation results from SLACS survey data, uncertainty calibration metrics, ablation studies, and regression test suite.
      </p>
    </div>

    <div class="metric-grid mb-24">
      <div class="stat-card stat-card--success">
        <div class="stat-label">Tests Passed</div>
        <div class="stat-value" id="vTestPassed">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Tests Skipped</div>
        <div class="stat-value" id="vTestSkipped">—</div>
      </div>
      <div class="stat-card" id="vFailCard">
        <div class="stat-label">Tests Failed</div>
        <div class="stat-value" id="vTestFailed">—</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Test Mode</div>
        <div class="stat-value" id="vTestMode" style="font-size:1rem">Loading...</div>
      </div>
      <div class="stat-card">
        <div class="stat-label">Last Run</div>
        <div class="stat-value" id="vTestLastRun" style="font-size:0.9rem">—</div>
      </div>
    </div>

    <div class="mb-24">
      <button id="vLoad" class="btn btn-primary">Load All Results</button>
    </div>

    <div class="card mb-24">
      <div class="card-header">
        <span class="card-title">SLACS Survey Validation</span>
        <span class="badge badge-info" id="vSlacsCount">—</span>
      </div>
      <div id="vSlacs"><p class="section-desc">Click "Load All Results"</p></div>
    </div>

    <div class="grid-2 gap-20 mb-20">
      <div class="card">
        <div class="card-header">
          <span class="card-title">UQ Calibration</span>
          <span class="badge badge-purple" id="vCalBadge">—</span>
        </div>
        <div id="vCalib"><p class="section-desc">Click "Load All Results"</p></div>
      </div>

      <div class="card">
        <div class="card-header">
          <span class="card-title">Ablation Study</span>
          <span class="card-subtitle">Component contribution analysis</span>
        </div>
        <div id="vAblation"><p class="section-desc">Click "Load All Results"</p></div>
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
      const failCard = document.getElementById("vFailCard");
      failEl.textContent = failures != null ? String(failures) : "—";
      if (failures != null && failures > 0) {
        failCard.style.borderColor = 'var(--danger, #ff4444)';
        failEl.style.color = 'var(--danger, #ff4444)';
      } else {
        failCard.style.borderColor = '';
        failEl.style.color = '';
      }
      document.getElementById("vTestMode").textContent = regression.status || (regression.publication_gate_passed ? "publication_gate" : "unavailable");
      document.getElementById("vTestLastRun").textContent = regression.generated_at_utc ? regression.generated_at_utc.replace("T", " ").slice(0, 19) : "—";
    } catch {
      P.toast("Could not reach API stats", "error");
    }
  };

  await applyRegressionStats();

  document.getElementById("vLoad").addEventListener("click", async () => {
    P.showLoading("Loading validation results...");
    try {
    // SLACS
    try {
      const data = await P.api("/api/v1/validation/slacs", { auth: false });
      const lenses = Array.isArray(data) ? data.filter(d => d.name) : [];
      const pass = lenses.filter(l => asBool(l.passed)).length;
      document.getElementById("vSlacsCount").textContent = `${pass}/${lenses.length} passed`;
      document.getElementById("vSlacsCount").className = `badge ${pass === lenses.length ? 'badge-success' : 'badge-warning'}`;
      
      const colorMetric = (value, threshold, higherIsBetter = true) => {
        if (value == null) return '—';
        const passes = higherIsBetter ? value >= threshold : value <= threshold;
        const color = passes ? 'var(--success)' : 'var(--danger)';
        return `<span style="color:${color}">${value.toFixed(higherIsBetter ? 3 : 4)}</span>`;
      };

      document.getElementById("vSlacs").innerHTML = `
        <div style="display:flex;gap:12px;margin-bottom:16px;padding:8px 12px;background:var(--bg-secondary);border-radius:6px">
          <span class="badge badge-info" style="font-size:0.85rem">NRMSE ≤ 0.15</span>
          <span class="badge badge-info" style="font-size:0.85rem">SSIM ≥ 0.70</span>
          <span class="badge badge-info" style="font-size:0.85rem">ring_corr ≥ 0.60</span>
        </div>
        <table class="data-table">
          <thead><tr><th>System</th><th>z<sub>l</sub></th><th>z<sub>s</sub></th><th>θ<sub>E</sub></th><th>σ<sub>v</sub></th><th>NRMSE</th><th>SSIM</th><th>ring_corr</th><th>Status</th></tr></thead>
          <tbody>${lenses.map(l => `
            <tr>
              <td style="font-weight:500">${P.esc((l.name || '').replace('SDSS ', ''))}</td>
              <td>${l.z_lens != null ? l.z_lens.toFixed(3) : '—'}</td>
              <td>${l.z_source != null ? l.z_source.toFixed(3) : '—'}</td>
              <td>${l.einstein_radius != null ? l.einstein_radius.toFixed(2) : (l.theta_e != null ? l.theta_e.toFixed(2) : '—')}</td>
              <td>${l.sigma_v != null ? l.sigma_v.toFixed(0) : (l.velocity_dispersion != null ? l.velocity_dispersion.toFixed(0) : '—')} km/s</td>
              <td>${colorMetric(l.rmse ?? l.nrmse, 0.15, false)}</td>
              <td>${colorMetric(l.ssim, 0.70, true)}</td>
              <td>${colorMetric(l.ring_correlation, 0.60, true)}</td>
              <td><span class="badge ${asBool(l.passed) ? 'badge-success' : 'badge-danger'}">${asBool(l.passed) ? 'PASS' : 'FAIL'}</span></td>
            </tr>
          `).join("")}</tbody>
        </table>
        <p class="section-desc" style="margin-top:12px">
          SLACS rows are rendered with artifact provenance. Image-space diagnostics should not be interpreted as direct convergence-map reconstruction.
        </p>`;
    } catch { document.getElementById("vSlacs").innerHTML = `<p class="section-desc">No SLACS data — run <code>python scripts/validate_real_data.py</code></p>`; }

    // Calibration
    try {
      const raw = await P.api("/api/v1/validation/calibration", { auth: false });
      // Endpoint returns flat array of per-lens calibration results
      const items = Array.isArray(raw) ? raw : (raw.results || [raw]);
      if (items.length) {
        // Compute mean ECE and coverage across all lenses
        const meanEce = items.reduce((s, d) => s + (d.ece || 0), 0) / items.length;
        const coverage90 = items.reduce((s, d) => s + (d.coverage_90 || 0), 0) / items.length;
        const meanCorr = items.reduce((s, d) => s + (d.uq_error_correlation || 0), 0) / items.length;
        const badgeClass = meanEce < 0.1 ? 'badge-success' : meanEce < 0.15 ? 'badge-warning' : 'badge-danger';
        const badgeLabel = meanEce < 0.1 ? "Calibrated" : `ECE: ${meanEce.toFixed(3)}`;
        const calBadge = document.getElementById("vCalBadge");
        if (calBadge) { calBadge.textContent = badgeLabel; calBadge.className = `badge ${badgeClass}`; }
        const calEl = document.getElementById("vCalib");
        if (calEl) calEl.innerHTML = `
          <div style="display:grid;grid-template-columns:1fr 1fr;gap:20px;margin-bottom:20px">
            <div style="text-align:center;padding:16px;background:var(--bg-secondary);border-radius:8px">
              <div style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:4px">Mean ECE</div>
              <div style="font-size:2rem;font-weight:600;color:var(--text-primary)">${meanEce.toFixed(4)}</div>
            </div>
            <div style="text-align:center;padding:16px;background:var(--bg-secondary);border-radius:8px">
              <div style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:4px">Coverage @90%</div>
              <div style="font-size:2rem;font-weight:600;color:var(--text-primary)">${(coverage90 * 100).toFixed(1)}%</div>
            </div>
          </div>
          <div class="metric-row"><span class="metric-key">UQ-Error Correlation</span><span class="metric-val">${meanCorr.toFixed(3)}</span></div>
          <div class="metric-row"><span class="metric-key">Systems Evaluated</span><span class="metric-val">${items.length}</span></div>
          <div class="metric-row"><span class="metric-key">Evaluation Scope</span><span class="metric-val">Synthetic held-out NFW analogs</span></div>
        `;
      }
    } catch { const calEl = document.getElementById("vCalib"); if (calEl) calEl.innerHTML = `<p class="section-desc">No calibration data — run <code>python scripts/uncertainty_calibration.py</code></p>`; }

    // Ablation
    try {
      const data = await P.api("/api/v1/validation/ablation", { auth: false });
      if (Array.isArray(data) && data.length) {
        document.getElementById("vAblation").innerHTML = `
          <table class="data-table">
            <thead><tr><th>Configuration</th><th>RMSE</th><th>SSIM</th><th>Δ RMSE</th><th>Δ SSIM</th></tr></thead>
            <tbody>${data.map((r, idx) => {
              const baseline = data[0];
              const deltaRMSE = idx > 0 ? ((r.rmse_mean ?? r.mean_rmse ?? 0) - (baseline.rmse_mean ?? baseline.mean_rmse ?? 0)) : 0;
              const deltaSSIM = idx > 0 ? ((r.ssim_mean ?? r.mean_ssim ?? 0) - (baseline.ssim_mean ?? baseline.mean_ssim ?? 0)) : 0;
              const deltaRMSEColor = deltaRMSE > 0 ? 'var(--danger)' : (deltaRMSE < 0 ? 'var(--success)' : 'var(--text-secondary)');
              const deltaSSIMColor = deltaSSIM > 0 ? 'var(--success)' : (deltaSSIM < 0 ? 'var(--danger)' : 'var(--text-secondary)');
              return `
                <tr>
                  <td>${P.esc(r.config || '')}</td>
                  <td>${(r.rmse_mean ?? r.mean_rmse ?? 0).toFixed(4)}</td>
                  <td>${(r.ssim_mean ?? r.mean_ssim ?? 0).toFixed(3)}</td>
                  <td style="color:${deltaRMSEColor}">${idx > 0 ? (deltaRMSE > 0 ? '+' : '') + deltaRMSE.toFixed(4) : '—'}</td>
                  <td style="color:${deltaSSIMColor}">${idx > 0 ? (deltaSSIM > 0 ? '+' : '') + deltaSSIM.toFixed(3) : '—'}</td>
                </tr>`;
            }).join("")}</tbody>
          </table>`;
      }
    } catch { document.getElementById("vAblation").innerHTML = `<p class="section-desc">No ablation data — run <code>python scripts/ablation_study.py</code></p>`; }

    P.toast("Validation results loaded", "success");
    } finally {
      P.hideLoading();
    }
  });

}

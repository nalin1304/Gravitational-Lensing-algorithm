/* Workbench Page — Simulation + Inference */
const L = () => window.LensPINN;

const presets = {
  einstein_cross: { label: "Einstein Cross", profile_type: "Elliptical NFW", mass: 1.6e12, scale_radius: 160, ellipticity: 0.22, grid_size: 64 },
  twin_quasar: { label: "Twin Quasar", profile_type: "NFW", mass: 5e13, scale_radius: 500, ellipticity: 0.08, grid_size: 128 },
  jwst_cluster: { label: "JWST Cluster", profile_type: "Elliptical NFW", mass: 7.8e13, scale_radius: 410, ellipticity: 0.30, grid_size: 128 },
  generic_demo: { label: "Generic Demo", profile_type: "NFW", mass: 2e12, scale_radius: 200, ellipticity: 0.2, grid_size: 64 },
};

let _synReq = null, _synResp = null, _infResp = null;
let _modelStatus = null;

export function render() {
  const chipHtml = Object.entries(presets).map(([k, p]) =>
    `<button class="preset-chip" data-preset="${k}">${p.label}</button>`
  ).join("");

  return `
    <div class="workbench-layout" style="display:grid;grid-template-columns:1fr 1fr;gap:24px">
      <!-- LEFT PANEL: CONTROLS -->
      <div class="workbench-controls">
        <!-- Presets -->
        <div class="card mb-16">
          <div class="card-header"><span class="card-title">Quick Presets</span></div>
          <div style="display:flex;gap:8px;flex-wrap:wrap;padding:0 16px 16px">${chipHtml}</div>
        </div>

        <!-- Mass Profile Parameters -->
        <div class="card mb-16">
          <div class="card-header">
            <span style="font-size:12px;text-transform:uppercase;letter-spacing:0.05em;color:#22d3ee;font-weight:600">Mass Profile</span>
          </div>
          <div class="form-group">
            <label class="form-label">Profile Type</label>
            <select id="wProfile" class="form-select">
              <option value="NFW">NFW</option>
              <option value="Elliptical NFW" selected>Elliptical NFW</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Virial Mass M<sub>vir</sub> (M☉)</label>
            <input id="wMass" class="form-input" type="number" value="1600000000000" step="1e11" min="1e11" max="1e14" />
          </div>
          <div class="form-group">
            <label class="form-label">Scale Radius r<sub>s</sub> (kpc)</label>
            <input id="wRadius" class="form-input" type="number" value="160" step="10" />
          </div>
          <div class="form-group">
            <label class="form-label">Ellipticity ε</label>
            <input id="wEllip" class="form-input" type="number" value="0.22" min="0" max="0.5" step="0.05" />
          </div>
        </div>

        <!-- Cosmology -->
        <div class="card mb-16">
          <div class="card-header">
            <span style="font-size:12px;text-transform:uppercase;letter-spacing:0.05em;color:#22d3ee;font-weight:600">Cosmology</span>
          </div>
          <div class="form-group">
            <label class="form-label">Lens Redshift z<sub>l</sub></label>
            <input id="wZl" class="form-input" type="number" value="0.5" step="0.1" min="0.01" max="2" />
          </div>
          <div class="form-group">
            <label class="form-label">Source Redshift z<sub>s</sub></label>
            <input id="wZs" class="form-input" type="number" value="2.0" step="0.1" min="0.1" max="5" />
          </div>
          <div style="padding:8px 16px;font-size:11px;color:#94a3b8;border-top:1px solid #1e293b">
            H₀ = 67.4 km/s/Mpc, Ω<sub>m</sub> = 0.315 — Planck 2018 (arXiv:1807.06209)
          </div>
        </div>

        <!-- Grid Settings -->
        <div class="card mb-16">
          <div class="card-header">
            <span style="font-size:12px;text-transform:uppercase;letter-spacing:0.05em;color:#22d3ee;font-weight:600">Grid</span>
          </div>
          <div class="form-group">
            <label class="form-label">Resolution</label>
            <select id="wGrid" class="form-select">
              <option value="32">32×32</option>
              <option value="64" selected>64×64</option>
              <option value="128">128×128</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Field of View (arcsec)</label>
            <input id="wFov" class="form-input" type="number" value="4.0" step="0.5" min="1" max="10" />
          </div>
        </div>

        <!-- Action Buttons -->
        <div class="card">
          <button id="wGenerate" class="btn btn-primary btn-full" style="margin-bottom:8px">Generate Map</button>
          <button id="wInfer" class="btn btn-ghost btn-full" disabled>Run PINN Inference</button>
        </div>
      </div>

      <!-- RIGHT PANEL: VISUALIZATION -->
      <div class="workbench-viz">
        <!-- Stats Row -->
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:8px;margin-bottom:16px">
          <div class="stat-card"><div class="stat-label">Min κ</div><div class="stat-value" id="wMin">—</div></div>
          <div class="stat-card"><div class="stat-label">Mean κ</div><div class="stat-value" id="wMean">—</div></div>
          <div class="stat-card"><div class="stat-label">Max κ</div><div class="stat-value" id="wMax">—</div></div>
          <div class="stat-card"><div class="stat-label">Std κ</div><div class="stat-value" id="wStd">—</div></div>
        </div>

        <!-- Plots: 2×2 grid -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-bottom:16px">
          <div class="card">
            <div class="card-header"><span class="card-title">Convergence Map</span></div>
            <div id="wMapPlot" class="plot-container"></div>
          </div>
          <div class="card">
            <div class="card-header"><span class="card-title">Radial Profile</span></div>
            <div id="wRadialPlot" class="plot-container"></div>
          </div>
        </div>

        <div class="card mb-16">
          <div class="card-header"><span class="card-title">Deflection Field</span></div>
          <div id="wDeflPlot" class="plot-container"></div>
        </div>

        <!-- Inference + Pipeline Trace -->
        <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px">
          <div class="card">
            <div class="card-header"><span class="card-title">Inference Results</span></div>
            <pre id="wInfOutput" style="max-height:200px;overflow-y:auto">Generate a map and run inference to see results.</pre>
          </div>
          <div class="card">
            <div class="card-header"><span class="card-title">Pipeline Trace</span></div>
            <pre id="wMethodsOutput" style="max-height:200px;overflow-y:auto">Select a preset and generate a convergence map.</pre>
          </div>
        </div>
      </div>
    </div>

    <!-- BOTTOM: Full-width sections -->
    <div style="margin-top:24px">
      <div class="card">
        <div class="card-header">
          <span class="card-title">Batch Job Submission</span>
          <span class="card-subtitle" style="margin:0">Submit multiple job IDs for batch aggregation · <code>POST /api/v1/batch</code></span>
        </div>
        <div class="form-group">
          <label class="form-label">Job IDs (comma-separated)</label>
          <input id="wBatchIds" class="form-input" placeholder="job-uuid-1, job-uuid-2, ..." />
        </div>
        <div style="display:flex;gap:8px;margin-bottom:12px">
          <button id="wBatchSubmit" class="btn btn-primary">Submit Batch</button>
          <button id="wBatchStatus" class="btn btn-secondary">Poll Status</button>
        </div>
        <pre id="wBatchOutput" style="max-height:150px;overflow-y:auto">No batch job submitted yet.</pre>
      </div>

      <div class="card" style="margin-top:16px">
        <button id="wExport" class="btn btn-secondary btn-full">Export Run Package (JSON)</button>
      </div>
    </div>
  `;
}

function applyPreset(key) {
  const p = presets[key]; if (!p) return;
  document.getElementById("wProfile").value = p.profile_type;
  document.getElementById("wMass").value = p.mass;
  document.getElementById("wRadius").value = p.scale_radius;
  document.getElementById("wEllip").value = p.ellipticity;
  document.getElementById("wGrid").value = p.grid_size;
  // Optional fields with defaults
  if (document.getElementById("wZl")) document.getElementById("wZl").value = p.z_lens || 0.5;
  if (document.getElementById("wZs")) document.getElementById("wZs").value = p.z_source || 2.0;
  if (document.getElementById("wFov")) document.getElementById("wFov").value = p.fov_arcsec || 4.0;
  document.querySelectorAll(".preset-chip").forEach(c => c.classList.toggle("active", c.dataset.preset === key));
}

function readPayload() {
  return {
    profile_type: document.getElementById("wProfile").value,
    mass: Number(document.getElementById("wMass").value),
    scale_radius: Number(document.getElementById("wRadius").value),
    ellipticity: Number(document.getElementById("wEllip").value),
    grid_size: Number(document.getElementById("wGrid").value),
    z_lens: Number(document.getElementById("wZl")?.value || 0.5),
    z_source: Number(document.getElementById("wZs")?.value || 2.0),
    fov_arcsec: Number(document.getElementById("wFov")?.value || 4.0),
  };
}

function parseMap(mat) {
  const flat = mat.flat();
  const min = Math.min(...flat), max = Math.max(...flat);
  const mean = flat.reduce((a, b) => a + b, 0) / flat.length;
  const std = Math.sqrt(flat.reduce((a, b) => a + (b - mean) ** 2, 0) / flat.length);
  return { min, max, mean, std };
}

function radialProfile(mat) {
  const n = mat.length, cx = n / 2, cy = n / 2;
  const bins = {};
  for (let i = 0; i < n; i++) for (let j = 0; j < n; j++) {
    const r = Math.round(Math.sqrt((i - cx) ** 2 + (j - cy) ** 2));
    if (!bins[r]) bins[r] = [];
    bins[r].push(mat[i][j]);
  }
  const radii = Object.keys(bins).map(Number).sort((a, b) => a - b);
  return { r: radii, kappa: radii.map(r => bins[r].reduce((a, b) => a + b) / bins[r].length) };
}

function renderPlots(mat, coords) {
  const P = L();
  const plotBg = "#0c0d14", plotFont = { color: "#8b8fa8", family: "JetBrains Mono, monospace" };
  const layout = (t) => ({ title: { text: t, font: { size: 13, color: "#f0f1ff" } }, paper_bgcolor: plotBg, plot_bgcolor: "#050508", font: plotFont, margin: { l: 50, r: 20, t: 36, b: 40 } });

  // Arcsec axis arrays from coordinates (if available)
  const xArr = coords?.X?.[0] ?? null;
  const yArr = coords?.Y ? coords.Y.map(row => row[0]) : null;
  const hasArcsec = Boolean(xArr && yArr);

  // Heatmap
  if (typeof Plotly !== "undefined") {
    const heatData = { z: mat, type: "heatmap", colorscale: "Viridis", showscale: true,
      colorbar: { title: { text: "κ", side: "right" } } };
    if (hasArcsec) { heatData.x = xArr; heatData.y = yArr; }
    const heatLayout = { ...layout("Convergence κ(θ)"),
      xaxis: { title: hasArcsec ? "θ₁ (arcsec)" : "x (pixels)" },
      yaxis: { title: hasArcsec ? "θ₂ (arcsec)" : "y (pixels)", scaleanchor: "x" } };
    Plotly.newPlot("wMapPlot", [heatData], heatLayout, { responsive: true });

    // Radial
    const rp = radialProfile(mat);
    const pixScale = hasArcsec ? (xArr[xArr.length - 1] - xArr[0]) / mat.length : 1;
    const rLabel = hasArcsec ? "r (arcsec)" : "r (pixels)";
    const rValues = hasArcsec ? rp.r.map(r => r * pixScale) : rp.r;
    Plotly.newPlot("wRadialPlot", [{ x: rValues, y: rp.kappa, type: "scatter", mode: "lines", line: { color: "#38bdf8", width: 2 } }],
      { ...layout("Radial Profile κ(r)"), xaxis: { title: rLabel }, yaxis: { title: "κ" } }, { responsive: true });

    // Deflection
    const n = mat.length;
    const dx = [], dy = [], xp = [], yp = [];
    for (let i = 2; i < n - 2; i += 4) for (let j = 2; j < n - 2; j += 4) {
      const gx = (mat[i]?.[j + 1] ?? 0) - (mat[i]?.[j - 1] ?? 0);
      const gy = (mat[i + 1]?.[j] ?? 0) - (mat[i - 1]?.[j] ?? 0);
      const px = hasArcsec ? xArr[j] : j;
      const py = hasArcsec ? yArr[i] : i;
      xp.push(px); yp.push(py); dx.push(gx); dy.push(gy);
    }
    const maxd = Math.max(...dx.map(Math.abs), ...dy.map(Math.abs), 1e-10);
    const deflScale = 3 / maxd * (hasArcsec ? pixScale : 1);
    const xRange = hasArcsec ? [xArr[0], xArr[xArr.length - 1]] : [0, n];
    const yRange = hasArcsec ? [yArr[0], yArr[yArr.length - 1]] : [0, n];
    Plotly.newPlot("wDeflPlot", [{
      type: "scatter", mode: "lines", x: xp.flatMap((x, i) => [x, x + dx[i] * deflScale, null]),
      y: yp.flatMap((y, i) => [y, y + dy[i] * deflScale, null]),
      line: { color: "#a78bfa", width: 1 }
    }], { ...layout("Deflection α(θ)"),
      xaxis: { range: xRange, title: hasArcsec ? "θ₁ (arcsec)" : "x (pixels)" },
      yaxis: { range: yRange, scaleanchor: "x", title: hasArcsec ? "θ₂ (arcsec)" : "y (pixels)" }
    }, { responsive: true });
  }
}

async function syncInferenceAvailability() {
  const P = L();
  const inferBtn = document.getElementById("wInfer");
  const output = document.getElementById("wInfOutput");
  const trace = document.getElementById("wMethodsOutput");

  try {
    const models = await P.api("/api/v1/models", { auth: false });
    _modelStatus = models.models?.[0] || null;
  } catch {
    _modelStatus = null;
  }

  const ready = Boolean(_modelStatus?.supports_inference);
  if (inferBtn) inferBtn.disabled = !ready;
  if (!ready && output) {
    output.textContent = JSON.stringify({
      status: _modelStatus?.status || "unknown",
      detail: "Inference disabled until a trained PINN checkpoint and runtime dependencies are available.",
    }, null, 2);
  }
  if (trace && _modelStatus) {
    trace.textContent = JSON.stringify({
      checkpoint_status: _modelStatus.status,
      supports_inference: _modelStatus.supports_inference,
      checkpoint_path: _modelStatus.checkpoint_path,
    }, null, 2);
  }
}

export function init() {
  const P = L();

  document.querySelectorAll(".preset-chip").forEach(c =>
    c.addEventListener("click", () => applyPreset(c.dataset.preset))
  );

  applyPreset("einstein_cross");
  syncInferenceAvailability();

  document.getElementById("wGenerate").addEventListener("click", async () => {
    try {
      P.showLoading("Generating convergence map...");
      _synReq = readPayload();
      _synResp = await P.api("/api/v1/synthetic", { method: "POST", body: _synReq });
      const s = parseMap(_synResp.convergence_map);
      document.getElementById("wMin").textContent = P.fmtSci(s.min);
      document.getElementById("wMean").textContent = P.fmtSci(s.mean);
      document.getElementById("wMax").textContent = P.fmtSci(s.max);
      document.getElementById("wStd").textContent = P.fmtSci(s.std);
      renderPlots(_synResp.convergence_map, _synResp.coordinates);
      document.getElementById("wMethodsOutput").textContent = JSON.stringify({ request: _synReq, metadata: _synResp.metadata }, null, 2);
      // Store for rigor SBI consumption
      if (_synResp.convergence_map) {
        sessionStorage.setItem('lastConvergenceMap', JSON.stringify(_synResp.convergence_map));
      }
      P.toast("Convergence map generated", "success");
    } catch (e) { P.toast(`Generation failed: ${e.message}`, "error"); }
    finally { P.hideLoading(); }
  });

  document.getElementById("wInfer").addEventListener("click", async () => {
    if (!_synResp) { P.toast("Generate a map first", "error"); return; }
    if (_modelStatus && !_modelStatus.supports_inference) {
      document.getElementById("wInfOutput").textContent = JSON.stringify({
        status: _modelStatus.status,
        detail: "No checkpoint-backed PINN inference is currently available.",
      }, null, 2);
      P.toast("Inference unavailable: checkpoint missing", "error");
      return;
    }
    try {
      P.showLoading("Running PINN inference...");
      _infResp = await P.api("/api/v1/inference", {
        method: "POST", body: {
          convergence_map: _synResp.convergence_map, target_size: 64, mc_samples: 1 /* current PINN has no Dropout; >1 samples are identical (wasted compute) */
        }
      });
      document.getElementById("wInfOutput").textContent = JSON.stringify(_infResp, null, 2);
      P.toast(`Inference complete: ${_infResp.inference_mode || 'pinn'}`, "success");
    } catch (e) {
      document.getElementById("wInfOutput").textContent = JSON.stringify({
        error: e.message,
        model_status: _modelStatus?.status || "unknown",
      }, null, 2);
      P.toast(`Inference failed: ${e.message}`, "error");
    }
    finally { P.hideLoading(); }
  });

  document.getElementById("wExport").addEventListener("click", () => {
    if (!_synResp) { P.toast("Nothing to export", "error"); return; }
    const blob = new Blob([JSON.stringify({
      generated_at: new Date().toISOString(), synthetic_request: _synReq,
      synthetic_response: _synResp, inference_response: _infResp
    }, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob); a.download = "lensing_run_package.json"; a.click();
    P.toast("Run package downloaded", "success");
  });

  // ── Batch job submission
  let _batchId = null;
  document.getElementById("wBatchSubmit").addEventListener("click", async () => {
    const raw = document.getElementById("wBatchIds").value.trim();
    const ids = raw ? raw.split(/[,\s]+/).map(s => s.trim()).filter(Boolean) : [];
    if (!ids.length) {
      // Auto-populate with current job if available
      if (_infResp?.job_id) {
        document.getElementById("wBatchIds").value = _infResp.job_id;
        P.toast("Pre-filled with current inference job ID", "info"); return;
      }
      P.toast("Enter at least one job ID", "error"); return;
    }
    try {
      P.showLoading("Submitting batch...");
      const resp = await P.api("/api/v1/batch", { method: "POST", body: { job_ids: ids } });
      _batchId = resp.batch_id;
      document.getElementById("wBatchOutput").textContent = JSON.stringify(resp, null, 2);
      P.toast(`Batch submitted: ${_batchId}`, "success");
    } catch (e) { P.toast(`Batch failed: ${e.message}`, "error"); }
    finally { P.hideLoading(); }
  });

  document.getElementById("wBatchStatus").addEventListener("click", async () => {
    if (!_batchId) { P.toast("No batch job submitted", "error"); return; }
    try {
      const resp = await P.api(`/api/v1/batch/${_batchId}/status`, { auth: false });
      document.getElementById("wBatchOutput").textContent = JSON.stringify(resp, null, 2);
    } catch (e) { P.toast(`Status poll failed: ${e.message}`, "error"); }
  });
}


export function cleanup() {
  ["wMapPlot", "wRadialPlot", "wDeflPlot"].forEach(id => {
    const el = document.getElementById(id);
    if (el) try { Plotly.purge(el); } catch (_) {}
  });
}

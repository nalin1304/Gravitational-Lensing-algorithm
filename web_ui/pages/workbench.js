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
    `<button class="chip preset-chip" data-preset="${k}">${p.label}</button>`
  ).join("");

  return `
    <div class="workbench-layout">
      <div class="workbench-controls">
        <div class="card mb-16">
          <div class="card-header"><span class="card-title">Presets</span></div>
          <div class="chip-group">${chipHtml}</div>
        </div>

        <div class="card mb-16">
          <div class="card-header"><span class="card-title">Parameters</span></div>
          <div class="form-group">
            <label class="form-label">Profile Type</label>
            <select id="wProfile" class="form-select">
              <option value="NFW">NFW</option>
              <option value="Elliptical NFW" selected>Elliptical NFW</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">Virial Mass (M☉)</label>
            <input id="wMass" class="form-input" type="number" value="1600000000000" step="1e11" />
          </div>
          <div class="form-group">
            <label class="form-label">Scale Radius (kpc)</label>
            <input id="wRadius" class="form-input" type="number" value="160" step="10" />
          </div>
          <div class="form-group">
            <label class="form-label">Ellipticity</label>
            <input id="wEllip" class="form-input" type="number" value="0.22" min="0" max="0.5" step="0.05" />
          </div>
          <div class="form-group">
            <label class="form-label">Grid Size</label>
            <select id="wGrid" class="form-select">
              <option value="32">32×32</option>
              <option value="64" selected>64×64</option>
              <option value="128">128×128</option>
            </select>
          </div>
        </div>

        <div class="card">
          <button id="wGenerate" class="btn btn-primary" style="width:100%;margin-bottom:8px">Generate Map</button>
          <button id="wInfer" class="btn btn-secondary" style="width:100%;margin-bottom:8px" disabled>Run PINN Inference</button>
          <button id="wExport" class="btn btn-secondary" style="width:100%">Export JSON</button>
        </div>
      </div>

      <div>
        <div class="grid-4 mb-16">
          <div class="stat-card"><div class="stat-label">Min κ</div><div class="stat-value" id="wMin">—</div></div>
          <div class="stat-card"><div class="stat-label">Mean κ</div><div class="stat-value" id="wMean">—</div></div>
          <div class="stat-card"><div class="stat-label">Max κ</div><div class="stat-value" id="wMax">—</div></div>
          <div class="stat-card"><div class="stat-label">Std κ</div><div class="stat-value" id="wStd">—</div></div>
        </div>

        <div class="grid-2 mb-16">
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

        <div class="grid-2">
          <div class="card">
            <div class="card-header"><span class="card-title">Inference Results</span></div>
            <pre id="wInfOutput">Generate a map and run inference to see results.</pre>
          </div>
          <div class="card">
            <div class="card-header"><span class="card-title">Pipeline Trace</span></div>
            <pre id="wMethodsOutput">Select a preset and generate a convergence map.</pre>
          </div>
        </div>

        <div class="card" style="margin-top:20px">
          <div class="card-header">
            <span class="card-title">Batch Job Submission</span>
            <span class="card-subtitle" style="margin:0">Submit multiple job IDs for batch aggregation · <code>POST /api/v1/batch</code></span>
          </div>
          <div class="form-group">
            <label class="form-label">Job IDs (comma-separated)</label>
            <input id="wBatchIds" class="form-input" placeholder="job-uuid-1, job-uuid-2, ..." />
          </div>
          <div style="display:flex;gap:8px">
            <button id="wBatchSubmit" class="btn btn-primary">Submit Batch</button>
            <button id="wBatchStatus" class="btn btn-secondary">Poll Status</button>
          </div>
          <pre id="wBatchOutput" style="margin-top:12px">No batch job submitted yet.</pre>
        </div>
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
  document.querySelectorAll(".preset-chip").forEach(c => c.classList.toggle("active", c.dataset.preset === key));
}

function readPayload() {
  return {
    profile_type: document.getElementById("wProfile").value,
    mass: Number(document.getElementById("wMass").value),
    scale_radius: Number(document.getElementById("wRadius").value),
    ellipticity: Number(document.getElementById("wEllip").value),
    grid_size: Number(document.getElementById("wGrid").value),
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

function renderPlots(mat) {
  const P = L();
  const plotBg = "#111827", plotFont = { color: "#94a3b8", family: "Inter" };
  const layout = (t) => ({ title: { text: t, font: { size: 13, color: "#f1f5f9" } }, paper_bgcolor: plotBg, plot_bgcolor: plotBg, font: plotFont, margin: { l: 50, r: 20, t: 36, b: 40 } });

  // Heatmap
  if (typeof Plotly !== "undefined") {
    Plotly.newPlot("wMapPlot", [{ z: mat, type: "heatmap", colorscale: "Viridis", showscale: true }], layout("κ(x, y)"), { responsive: true });

    // Radial
    const rp = radialProfile(mat);
    Plotly.newPlot("wRadialPlot", [{ x: rp.r, y: rp.kappa, type: "scatter", mode: "lines", line: { color: "#38bdf8", width: 2 } }],
      { ...layout("κ(r)"), xaxis: { title: "r (pixels)" }, yaxis: { title: "κ" } }, { responsive: true });

    // Deflection
    const n = mat.length;
    const dx = [], dy = [], xp = [], yp = [];
    for (let i = 2; i < n - 2; i += 4) for (let j = 2; j < n - 2; j += 4) {
      const gx = (mat[i]?.[j + 1] ?? 0) - (mat[i]?.[j - 1] ?? 0);
      const gy = (mat[i + 1]?.[j] ?? 0) - (mat[i - 1]?.[j] ?? 0);
      xp.push(j); yp.push(i); dx.push(gx); dy.push(gy);
    }
    const maxd = Math.max(...dx.map(Math.abs), ...dy.map(Math.abs), 1e-10);
    const scale = 3 / maxd;
    Plotly.newPlot("wDeflPlot", [{
      type: "scatter", mode: "lines", x: xp.flatMap((x, i) => [x, x + dx[i] * scale, null]),
      y: yp.flatMap((y, i) => [y, y + dy[i] * scale, null]),
      line: { color: "#a78bfa", width: 1 }
    }], { ...layout("Deflection α(θ)"), xaxis: { range: [0, n] }, yaxis: { range: [0, n], scaleanchor: "x" } }, { responsive: true });
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
      renderPlots(_synResp.convergence_map);
      document.getElementById("wMethodsOutput").textContent = JSON.stringify({ request: _synReq, metadata: _synResp.metadata }, null, 2);
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
          convergence_map: _synResp.convergence_map, target_size: 64, mc_samples: 32
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

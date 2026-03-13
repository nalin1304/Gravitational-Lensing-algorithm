/**
 * Lensing Analysis Page
 *
 * Interactive computation and visualization of:
 *   - Critical curves & caustics
 *   - Magnification maps
 *   - Multiple image positions (lens equation solver)
 *
 * References:
 *   Schneider, Ehlers & Falco (1992) — §3.13–3.17, §5.3–5.4
 *   Birrer & Amara (2018) — §3.1
 */

const P = () => window.LensPINN;

const DEFAULTS = {
  M_vir: 1e14,
  concentration: 5.0,
  z_lens: 0.3,
  z_source: 1.5,
  grid_size: 200,
  grid_range: 30.0,
  source_x: 1.0,
  source_y: 0.0,
};

/* ── helpers ────────────────────────────────────────────────────────── */

function inp(id, label, val, attrs = "") {
  return `<div class="form-group">
    <label for="${id}">${label}</label>
    <input id="${id}" type="number" class="form-input" value="${val}" step="any" ${attrs} />
  </div>`;
}

function readParams() {
  return {
    M_vir:         parseFloat(document.getElementById("lns-Mvir").value),
    concentration: parseFloat(document.getElementById("lns-conc").value),
    z_lens:        parseFloat(document.getElementById("lns-zlens").value),
    z_source:      parseFloat(document.getElementById("lns-zsrc").value),
    grid_size:     parseInt(document.getElementById("lns-grid").value, 10),
    grid_range:    parseFloat(document.getElementById("lns-range").value),
  };
}

/* ── API calls (use shared api() helper) ────────────────────────────── */

async function fetchCriticalCurves(body) {
  return P().api("/api/v1/lensing/critical-curves", { method: "POST", body });
}

async function fetchImages(body) {
  return P().api("/api/v1/lensing/solve-images", { method: "POST", body });
}

/* ── Plotly rendering ───────────────────────────────────────────────── */

const DARK = { paper: "#0a0e27", plot: "#111635", font: "#e0e6ff" };

function plotMagnification(data) {
  const { magnification_map, grid_range, critical_curves } = data;
  const n = magnification_map.length;

  const traces = [{
    z: magnification_map,
    type: "heatmap",
    colorscale: "RdBu",
    zmin: -10, zmax: 10,
    x0: -grid_range, dx: (2 * grid_range) / n,
    y0: -grid_range, dy: (2 * grid_range) / n,
    colorbar: { title: "μ", titlefont: { color: DARK.font }, tickfont: { color: DARK.font } },
    hovertemplate: "θ_x: %{x:.2f}″<br>θ_y: %{y:.2f}″<br>μ: %{z:.2f}<extra></extra>",
  }];

  if (critical_curves.x.length > 0) {
    traces.push({
      x: critical_curves.x, y: critical_curves.y,
      mode: "markers", type: "scatter",
      marker: { color: "#00ff88", size: 2, opacity: 0.8 },
      name: "Critical curves (det A = 0)",
      hovertemplate: "θ_x: %{x:.3f}″<br>θ_y: %{y:.3f}″<extra>Critical curve</extra>",
    });
  }

  Plotly.newPlot("lns-mag-plot", traces, {
    title: { text: "Magnification Map & Critical Curves", font: { size: 14 } },
    xaxis: { title: "θ_x (arcsec)", scaleanchor: "y", gridcolor: "#1a2040", zerolinecolor: "#2a3060" },
    yaxis: { title: "θ_y (arcsec)", gridcolor: "#1a2040", zerolinecolor: "#2a3060" },
    paper_bgcolor: DARK.paper, plot_bgcolor: DARK.plot, font: { color: DARK.font },
    margin: { t: 45, b: 50, l: 55, r: 20 },
  }, { responsive: true });
}

function plotCaustics(data) {
  const { caustics } = data;
  const traces = [];

  if (caustics.x.length > 0) {
    traces.push({
      x: caustics.x, y: caustics.y,
      mode: "markers", type: "scatter",
      marker: { color: "#ff6644", size: 3, opacity: 0.8 },
      name: "Caustics",
      hovertemplate: "β_x: %{x:.3f}″<br>β_y: %{y:.3f}″<extra>Caustic</extra>",
    });
  }

  Plotly.newPlot("lns-caustic-plot", traces, {
    title: { text: "Caustic Structure (Source Plane)", font: { size: 14 } },
    xaxis: { title: "β_x (arcsec)", scaleanchor: "y", gridcolor: "#1a2040", zerolinecolor: "#2a3060" },
    yaxis: { title: "β_y (arcsec)", gridcolor: "#1a2040", zerolinecolor: "#2a3060" },
    paper_bgcolor: DARK.paper, plot_bgcolor: DARK.plot, font: { color: DARK.font },
    margin: { t: 45, b: 50, l: 55, r: 20 },
  }, { responsive: true });
}

function plotImages(imgData, ccData) {
  const { images, source_position } = imgData;
  const traces = [];

  if (ccData) {
    const n = ccData.magnification_map.length;
    const gr = ccData.grid_range;
    traces.push({
      z: ccData.magnification_map,
      type: "heatmap", colorscale: "RdBu",
      zmin: -10, zmax: 10,
      x0: -gr, dx: (2 * gr) / n,
      y0: -gr, dy: (2 * gr) / n,
      opacity: 0.35, showscale: false,
    });

    if (ccData.critical_curves.x.length > 0) {
      traces.push({
        x: ccData.critical_curves.x, y: ccData.critical_curves.y,
        mode: "markers", type: "scatter",
        marker: { color: "#00ff88", size: 1.5, opacity: 0.4 },
        name: "Critical curves",
        showlegend: false,
      });
    }
  }

  traces.push({
    x: [source_position.x], y: [source_position.y],
    mode: "markers", type: "scatter",
    marker: { color: "#ffcc00", size: 16, symbol: "star", line: { width: 1, color: "#fff" } },
    name: "Source position",
    hovertemplate: "Source<br>β_x: %{x:.3f}″<br>β_y: %{y:.3f}″<extra></extra>",
  });

  const COLORS = { minimum: "#00ccff", saddle: "#ff4466", maximum: "#88ff44" };
  const SYMBOLS = { minimum: "circle", saddle: "diamond", maximum: "square" };
  images.forEach((im, i) => {
    traces.push({
      x: [im.x], y: [im.y],
      mode: "markers+text", type: "scatter",
      marker: {
        color: COLORS[im.type] || "#ffffff",
        size: 12, symbol: SYMBOLS[im.type] || "circle",
        line: { width: 1, color: "#fff" },
      },
      text: [`${im.type} (μ=${im.magnification.toFixed(1)})`],
      textposition: "top center",
      textfont: { color: DARK.font, size: 10 },
      name: `Image ${i + 1}: ${im.type}`,
      hovertemplate: `Image ${i + 1}<br>θ_x: %{x:.4f}″<br>θ_y: %{y:.4f}″<br>μ: ${im.magnification.toFixed(3)}<br>Type: ${im.type}<br>Parity: ${im.parity > 0 ? '+' : '−'}<extra></extra>`,
    });
  });

  Plotly.newPlot("lns-images-plot", traces, {
    title: { text: `Image Positions — ${images.length} image${images.length !== 1 ? 's' : ''} found`, font: { size: 14 } },
    xaxis: { title: "θ_x (arcsec)", scaleanchor: "y", gridcolor: "#1a2040", zerolinecolor: "#2a3060" },
    yaxis: { title: "θ_y (arcsec)", gridcolor: "#1a2040", zerolinecolor: "#2a3060" },
    paper_bgcolor: DARK.paper, plot_bgcolor: DARK.plot, font: { color: DARK.font },
    showlegend: true,
    legend: { x: 1, xanchor: "right", y: 1, bgcolor: "rgba(10,14,39,0.85)", bordercolor: "#2a3060", borderwidth: 1 },
    margin: { t: 45, b: 50, l: 55, r: 20 },
  }, { responsive: true });
}

/* ── Image results table ────────────────────────────────────────────── */

function renderImageTable(images) {
  if (!images || images.length === 0)
    return `<p style="color:#8899bb;text-align:center;padding:20px">No images found for this source position.</p>`;

  const rows = images.map((im, i) => `
    <tr>
      <td>${i + 1}</td>
      <td style="font-family:monospace">${im.x.toFixed(4)}</td>
      <td style="font-family:monospace">${im.y.toFixed(4)}</td>
      <td style="font-family:monospace">${im.magnification.toFixed(3)}</td>
      <td><span class="badge" style="background:${
        im.type === 'minimum' ? '#00ccff22' : im.type === 'saddle' ? '#ff446622' : '#88ff4422'
      };color:${
        im.type === 'minimum' ? '#00ccff' : im.type === 'saddle' ? '#ff4466' : '#88ff44'
      };padding:2px 8px;border-radius:4px;font-size:0.85em">${im.type}</span></td>
      <td style="text-align:center;font-size:1.1em">${im.parity > 0 ? "+" : "−"}</td>
    </tr>`).join("");

  return `<table class="data-table" style="width:100%;border-collapse:collapse">
    <thead><tr style="border-bottom:1px solid #2a3060">
      <th style="padding:8px 12px">#</th>
      <th style="padding:8px 12px">θ_x (″)</th>
      <th style="padding:8px 12px">θ_y (″)</th>
      <th style="padding:8px 12px">μ</th>
      <th style="padding:8px 12px">Type</th>
      <th style="padding:8px 12px">Parity</th>
    </tr></thead>
    <tbody>${rows}</tbody>
  </table>
  <p style="color:#667799;font-size:0.8em;margin-top:8px">
    Image classification per Schneider (1992) §5.3: minimum (Type I), saddle (Type II), maximum (Type III)
  </p>`;
}

/* ── summary info panel ─────────────────────────────────────────────── */

function renderSummary(ccData) {
  const nCrit = ccData.critical_curves.x.length;
  const nCaus = ccData.caustics.x.length;
  const p = ccData.parameters;
  return `
    <div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(140px,1fr));gap:12px;margin-top:12px">
      <div class="stat-card">
        <div class="stat-value" style="color:#00ff88">${nCrit}</div>
        <div class="stat-label">Critical curve pts</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="color:#ff6644">${nCaus}</div>
        <div class="stat-label">Caustic pts</div>
      </div>
      <div class="stat-card">
        <div class="stat-value" style="font-size:0.9em">${P().fmtSci(p.M_vir)}</div>
        <div class="stat-label">M_vir (M☉)</div>
      </div>
      <div class="stat-card">
        <div class="stat-value">${p.concentration}</div>
        <div class="stat-label">Concentration</div>
      </div>
    </div>`;
}

/* ── main render ────────────────────────────────────────────────────── */

let _ccData = null;

export function render(container) {
  container.innerHTML = `
    <div style="margin-bottom:24px">
      <h1 style="margin:0 0 4px 0">🔭 Lensing Analysis</h1>
      <p class="subtitle" style="margin:0;color:#8899bb">
        Critical curves, caustics, magnification maps & image position solver
        <span style="opacity:0.6">— Schneider, Ehlers & Falco (1992)</span>
      </p>
    </div>

    <div class="card" style="border-left:3px solid #00ccff">
      <h2 style="margin-top:0">NFW Lens Parameters</h2>
      <p style="color:#8899bb;font-size:0.9em;margin-top:-4px">
        Configure the NFW dark-matter halo lens profile for analysis.
      </p>
      <div class="params-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px">
        ${inp("lns-Mvir",  "M_vir (M☉)",     DEFAULTS.M_vir,        'step="1e13" min="1e10"')}
        ${inp("lns-conc",  "Concentration c", DEFAULTS.concentration,'step="0.5" min="1"')}
        ${inp("lns-zlens", "z_lens",          DEFAULTS.z_lens,       'step="0.05" min="0.01"')}
        ${inp("lns-zsrc",  "z_source",        DEFAULTS.z_source,     'step="0.1" min="0.02"')}
        ${inp("lns-grid",  "Grid size",       DEFAULTS.grid_size,    'step="50" min="50" max="500"')}
        ${inp("lns-range", 'Grid range (″)',  DEFAULTS.grid_range,   'step="5" min="1"')}
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:16px">
        <button id="lns-run-cc" class="btn btn-primary">
          ⚡ Compute Critical Curves & Magnification
        </button>
        <span id="lns-cc-status" style="font-size:0.9em"></span>
      </div>
      <div id="lns-cc-summary"></div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
      <div class="card" style="padding:12px"><div id="lns-mag-plot" style="height:440px"></div></div>
      <div class="card" style="padding:12px"><div id="lns-caustic-plot" style="height:440px"></div></div>
    </div>

    <div class="card" style="margin-top:16px;border-left:3px solid #ffcc00">
      <h2 style="margin-top:0">Image Position Solver</h2>
      <p style="color:#8899bb;font-size:0.9em;margin-top:-4px">
        Find all multiple images of a background source lensed by the NFW halo above.
        Uses a two-phase algorithm: coarse grid search followed by Newton-Raphson refinement.
      </p>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:12px">
        ${inp("lns-srcx", 'Source x (″)', DEFAULTS.source_x, 'step="0.5"')}
        ${inp("lns-srcy", 'Source y (″)', DEFAULTS.source_y, 'step="0.5"')}
      </div>
      <div style="display:flex;align-items:center;gap:12px;margin-top:16px">
        <button id="lns-run-img" class="btn btn-primary">
          🎯 Solve Lens Equation
        </button>
        <span id="lns-img-status" style="font-size:0.9em"></span>
      </div>
    </div>

    <div class="card" style="margin-top:16px;padding:12px">
      <div id="lns-images-plot" style="height:440px"></div>
    </div>
    <div class="card" style="margin-top:16px">
      <h3 style="margin-top:0">Image Catalog</h3>
      <div id="lns-images-table">
        <p style="color:#667799;text-align:center;padding:20px">
          Run the solver above to find image positions.
        </p>
      </div>
    </div>
  `;

  /* ── Critical curves button ────────────────────────────────────── */
  document.getElementById("lns-run-cc").addEventListener("click", async () => {
    const btn = document.getElementById("lns-run-cc");
    const statusEl = document.getElementById("lns-cc-status");
    btn.disabled = true;
    statusEl.innerHTML = '<span style="color:#00ccff">⏳ Computing…</span>';
    try {
      const body = readParams();
      _ccData = await fetchCriticalCurves(body);
      plotMagnification(_ccData);
      plotCaustics(_ccData);
      document.getElementById("lns-cc-summary").innerHTML = renderSummary(_ccData);
      statusEl.innerHTML = '<span style="color:#00ff88">✓ Complete</span>';
    } catch (e) {
      statusEl.innerHTML = `<span style="color:#ff4466">✗ ${e.message}</span>`;
      document.getElementById("lns-cc-summary").innerHTML = "";
    } finally {
      btn.disabled = false;
    }
  });

  /* ── Image solver button ───────────────────────────────────────── */
  document.getElementById("lns-run-img").addEventListener("click", async () => {
    const btn = document.getElementById("lns-run-img");
    const statusEl = document.getElementById("lns-img-status");
    btn.disabled = true;
    statusEl.innerHTML = '<span style="color:#00ccff">⏳ Solving…</span>';
    try {
      const body = readParams();
      body.source_x = parseFloat(document.getElementById("lns-srcx").value);
      body.source_y = parseFloat(document.getElementById("lns-srcy").value);
      const imgData = await fetchImages(body);
      plotImages(imgData, _ccData);
      document.getElementById("lns-images-table").innerHTML = renderImageTable(imgData.images);
      statusEl.innerHTML = `<span style="color:#00ff88">✓ ${imgData.n_images} image${imgData.n_images !== 1 ? 's' : ''} found</span>`;
    } catch (e) {
      statusEl.innerHTML = `<span style="color:#ff4466">✗ ${e.message}</span>`;
    } finally {
      btn.disabled = false;
    }
  });
}

export function cleanup() {
  _ccData = null;
  ["lns-mag-plot", "lns-caustic-plot", "lns-images-plot"].forEach(id => {
    const el = document.getElementById(id);
    if (el) try { Plotly.purge(el); } catch (_) {}
  });
}

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

function input(id, label, val, attrs = "") {
  return `<div class="form-group">
    <label for="${id}">${label}</label>
    <input id="${id}" type="number" class="form-control" value="${val}" ${attrs} />
  </div>`;
}

function paramBlock() {
  return `
  <div class="params-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:12px">
    ${input("lns-Mvir",  "M_vir (M☉)",        DEFAULTS.M_vir,        'step="1e13"')}
    ${input("lns-conc",  "Concentration c",    DEFAULTS.concentration,'step="0.5"')}
    ${input("lns-zlens", "z_lens",             DEFAULTS.z_lens,       'step="0.05"')}
    ${input("lns-zsrc",  "z_source",           DEFAULTS.z_source,     'step="0.1"')}
    ${input("lns-grid",  "Grid size",          DEFAULTS.grid_size,    'step="50" min="50"')}
    ${input("lns-range", "Grid range (″)",     DEFAULTS.grid_range,   'step="5"')}
  </div>`;
}

function sourceBlock() {
  return `
  <div class="params-grid" style="display:grid;grid-template-columns:1fr 1fr;gap:12px;margin-top:8px">
    ${input("lns-srcx", "Source x (″)", DEFAULTS.source_x, 'step="0.5"')}
    ${input("lns-srcy", "Source y (″)", DEFAULTS.source_y, 'step="0.5"')}
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

/* ── API calls ──────────────────────────────────────────────────────── */

async function fetchCriticalCurves(params) {
  const q = new URLSearchParams(params).toString();
  const res = await fetch(`/api/v1/lensing/critical-curves?${q}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

async function fetchImages(params) {
  const q = new URLSearchParams(params).toString();
  const res = await fetch(`/api/v1/lensing/solve-images?${q}`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

/* ── Plotly rendering ───────────────────────────────────────────────── */

function plotMagnification(data) {
  const { magnification_map, grid_range, critical_curves, caustics } = data;
  const n = magnification_map.length;

  const traces = [
    {
      z: magnification_map,
      type: "heatmap",
      colorscale: "RdBu",
      zmin: -10, zmax: 10,
      x0: -grid_range, dx: (2 * grid_range) / n,
      y0: -grid_range, dy: (2 * grid_range) / n,
      colorbar: { title: "μ" },
      name: "Magnification",
    },
  ];

  if (critical_curves.x.length > 0) {
    traces.push({
      x: critical_curves.x,
      y: critical_curves.y,
      mode: "markers",
      marker: { color: "#00ff88", size: 2 },
      name: "Critical curves",
      type: "scatter",
    });
  }

  const layout = {
    title: "Magnification Map & Critical Curves",
    xaxis: { title: "θ_x (arcsec)", scaleanchor: "y" },
    yaxis: { title: "θ_y (arcsec)" },
    paper_bgcolor: "#0a0e27",
    plot_bgcolor: "#0a0e27",
    font: { color: "#e0e6ff" },
    margin: { t: 50, b: 50, l: 60, r: 30 },
  };

  Plotly.newPlot("lns-mag-plot", traces, layout, { responsive: true });
}

function plotCaustics(data) {
  const { caustics, critical_curves } = data;
  const traces = [];

  if (caustics.x.length > 0) {
    traces.push({
      x: caustics.x, y: caustics.y,
      mode: "markers",
      marker: { color: "#ff6644", size: 3 },
      name: "Caustics",
      type: "scatter",
    });
  }

  const layout = {
    title: "Caustic Structure (Source Plane)",
    xaxis: { title: "β_x (arcsec)", scaleanchor: "y" },
    yaxis: { title: "β_y (arcsec)" },
    paper_bgcolor: "#0a0e27",
    plot_bgcolor: "#0a0e27",
    font: { color: "#e0e6ff" },
    margin: { t: 50, b: 50, l: 60, r: 30 },
  };

  Plotly.newPlot("lns-caustic-plot", traces, layout, { responsive: true });
}

function plotImages(imgData, ccData) {
  const { images, source_position } = imgData;
  const traces = [];

  // magnification background
  if (ccData) {
    const n = ccData.magnification_map.length;
    const gr = ccData.grid_range;
    traces.push({
      z: ccData.magnification_map,
      type: "heatmap",
      colorscale: "RdBu",
      zmin: -10, zmax: 10,
      x0: -gr, dx: (2 * gr) / n,
      y0: -gr, dy: (2 * gr) / n,
      opacity: 0.4,
      showscale: false,
      name: "μ map",
    });
  }

  // source marker
  traces.push({
    x: [source_position.x], y: [source_position.y],
    mode: "markers", type: "scatter",
    marker: { color: "#ffcc00", size: 14, symbol: "star" },
    name: "Source",
  });

  // image markers
  const colors = { minimum: "#00ccff", saddle: "#ff4466", maximum: "#88ff44" };
  images.forEach((im, i) => {
    traces.push({
      x: [im.x], y: [im.y],
      mode: "markers+text", type: "scatter",
      marker: { color: colors[im.type] || "#ffffff", size: 10, symbol: "circle" },
      text: [`${im.type} (μ=${im.magnification.toFixed(1)})`],
      textposition: "top center",
      textfont: { color: "#e0e6ff", size: 10 },
      name: `Image ${i + 1}`,
    });
  });

  const layout = {
    title: `Image Positions (${images.length} images found)`,
    xaxis: { title: "θ_x (arcsec)", scaleanchor: "y" },
    yaxis: { title: "θ_y (arcsec)" },
    paper_bgcolor: "#0a0e27",
    plot_bgcolor: "#0a0e27",
    font: { color: "#e0e6ff" },
    showlegend: true,
    legend: { x: 1, xanchor: "right", y: 1, bgcolor: "rgba(10,14,39,0.8)" },
    margin: { t: 50, b: 50, l: 60, r: 30 },
  };

  Plotly.newPlot("lns-images-plot", traces, layout, { responsive: true });
}

/* ── Image table ────────────────────────────────────────────────────── */

function renderImageTable(images) {
  if (!images || images.length === 0) return "<p>No images found.</p>";
  const rows = images.map((im, i) => `
    <tr>
      <td>${i + 1}</td>
      <td>${im.x.toFixed(4)}</td>
      <td>${im.y.toFixed(4)}</td>
      <td>${im.magnification.toFixed(3)}</td>
      <td><span class="badge badge-${im.type}">${im.type}</span></td>
      <td>${im.parity > 0 ? "+" : "−"}</td>
    </tr>`).join("");

  return `<table class="data-table">
    <thead><tr><th>#</th><th>θ_x (″)</th><th>θ_y (″)</th><th>μ</th><th>Type</th><th>Parity</th></tr></thead>
    <tbody>${rows}</tbody>
  </table>`;
}

/* ── main render ────────────────────────────────────────────────────── */

let _ccData = null;

export function render(container) {
  container.innerHTML = `
    <h1>Lensing Analysis</h1>
    <p class="subtitle">Critical curves, caustics, magnification maps & image position solver — Schneider (1992)</p>

    <div class="card">
      <h2>NFW Lens Parameters</h2>
      ${paramBlock()}
      <button id="lns-run-cc" class="btn btn-primary" style="margin-top:12px">
        Compute Critical Curves & Magnification
      </button>
      <div id="lns-cc-status" style="margin-top:8px"></div>
    </div>

    <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:16px">
      <div class="card"><div id="lns-mag-plot" style="height:420px"></div></div>
      <div class="card"><div id="lns-caustic-plot" style="height:420px"></div></div>
    </div>

    <div class="card" style="margin-top:16px">
      <h2>Image Position Solver</h2>
      <p>Find multiple images of a background source lensed by this NFW halo.</p>
      ${sourceBlock()}
      <button id="lns-run-img" class="btn btn-primary" style="margin-top:12px">
        Solve Lens Equation
      </button>
      <div id="lns-img-status" style="margin-top:8px"></div>
    </div>

    <div class="card" style="margin-top:16px">
      <div id="lns-images-plot" style="height:420px"></div>
      <div id="lns-images-table" style="margin-top:12px"></div>
    </div>
  `;

  document.getElementById("lns-run-cc").addEventListener("click", async () => {
    const statusEl = document.getElementById("lns-cc-status");
    statusEl.innerHTML = '<span class="loading">Computing…</span>';
    try {
      const p = readParams();
      _ccData = await fetchCriticalCurves(p);
      plotMagnification(_ccData);
      plotCaustics(_ccData);
      const nCrit = _ccData.critical_curves.x.length;
      const nCaus = _ccData.caustics.x.length;
      statusEl.innerHTML = `<span class="success">✓ ${nCrit} critical-curve pts, ${nCaus} caustic pts</span>`;
    } catch (e) {
      statusEl.innerHTML = `<span class="error">Error: ${e.message}</span>`;
    }
  });

  document.getElementById("lns-run-img").addEventListener("click", async () => {
    const statusEl = document.getElementById("lns-img-status");
    statusEl.innerHTML = '<span class="loading">Solving…</span>';
    try {
      const p = readParams();
      p.source_x = parseFloat(document.getElementById("lns-srcx").value);
      p.source_y = parseFloat(document.getElementById("lns-srcy").value);
      const imgData = await fetchImages(p);
      plotImages(imgData, _ccData);
      document.getElementById("lns-images-table").innerHTML = renderImageTable(imgData.images);
      statusEl.innerHTML = `<span class="success">✓ Found ${imgData.n_images} images</span>`;
    } catch (e) {
      statusEl.innerHTML = `<span class="error">Error: ${e.message}</span>`;
    }
  });
}

export function cleanup() {
  _ccData = null;
  ["lns-mag-plot", "lns-caustic-plot", "lns-images-plot"].forEach(id => {
    const el = document.getElementById(id);
    if (el) Plotly.purge(el);
  });
}

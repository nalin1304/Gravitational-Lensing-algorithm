const state = {
  syntheticRequest: null,
  syntheticResponse: null,
  inferenceResponse: null,
};

const controls = {
  profileType: document.getElementById("profileType"),
  mass: document.getElementById("mass"),
  scaleRadius: document.getElementById("scaleRadius"),
  ellipticity: document.getElementById("ellipticity"),
  gridSize: document.getElementById("gridSize"),
  generateBtn: document.getElementById("generateBtn"),
  inferBtn: document.getElementById("inferBtn"),
  exportBtn: document.getElementById("exportBtn"),
  status: document.getElementById("status"),
};

const metrics = {
  min: document.getElementById("metricMin"),
  mean: document.getElementById("metricMean"),
  max: document.getElementById("metricMax"),
  std: document.getElementById("metricStd"),
};

const output = {
  inference: document.getElementById("inferenceOutput"),
  methods: document.getElementById("methodsOutput"),
  history: document.getElementById("historyList"),
};

function setStatus(message) {
  controls.status.textContent = message;
}

function parseMap(matrix) {
  const flat = matrix.flat().filter((v) => Number.isFinite(v));
  const n = flat.length;
  const mean = flat.reduce((a, b) => a + b, 0) / Math.max(n, 1);
  const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / Math.max(n, 1);
  const std = Math.sqrt(variance);
  return {
    min: Math.min(...flat),
    max: Math.max(...flat),
    mean,
    std,
  };
}

function radialProfile(matrix) {
  const h = matrix.length;
  const w = matrix[0].length;
  const cx = (w - 1) / 2;
  const cy = (h - 1) / 2;
  const maxR = Math.floor(Math.min(cx, cy));

  const sums = new Array(maxR + 1).fill(0);
  const counts = new Array(maxR + 1).fill(0);

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const r = Math.floor(Math.hypot(x - cx, y - cy));
      if (r <= maxR) {
        sums[r] += matrix[y][x];
        counts[r] += 1;
      }
    }
  }

  const radius = [];
  const values = [];
  for (let i = 0; i <= maxR; i += 1) {
    if (counts[i] > 0) {
      radius.push(i);
      values.push(sums[i] / counts[i]);
    }
  }
  return { radius, values };
}

function renderPlots(matrix) {
  const heatmap = {
    z: matrix,
    type: "heatmap",
    colorscale: "Viridis",
  };

  Plotly.newPlot("mapPlot", [heatmap], {
    paper_bgcolor: "#0A0E1A",
    plot_bgcolor: "#0A0E1A",
    margin: { l: 60, r: 20, t: 30, b: 50 },
    font: { color: "#EAF4FF" },
    xaxis: { title: "x pixel" },
    yaxis: { title: "y pixel" },
  }, { responsive: true, displaylogo: false });

  const profile = radialProfile(matrix);
  const radialTrace = {
    x: profile.radius,
    y: profile.values,
    type: "scatter",
    mode: "lines",
    line: { color: "#00D4FF", width: 2 },
    name: "Mean kappa",
  };

  Plotly.newPlot("radialPlot", [radialTrace], {
    paper_bgcolor: "#0A0E1A",
    plot_bgcolor: "#0A0E1A",
    margin: { l: 60, r: 20, t: 30, b: 50 },
    font: { color: "#EAF4FF" },
    xaxis: { title: "Radius (pixels)" },
    yaxis: { title: "Mean kappa" },
  }, { responsive: true, displaylogo: false });
}

function updateMetrics(stats) {
  metrics.min.textContent = stats.min.toExponential(3);
  metrics.mean.textContent = stats.mean.toExponential(3);
  metrics.max.textContent = stats.max.toExponential(3);
  metrics.std.textContent = stats.std.toExponential(3);
}

function makeMethodsText() {
  if (!state.syntheticRequest || !state.syntheticResponse) {
    return "Generate a run to populate this section.";
  }
  const cfg = state.syntheticRequest;
  const meta = state.syntheticResponse.metadata;
  return [
    "Methods (Auto-generated)",
    "",
    "We generated synthetic convergence maps using the API thin-lens pipeline.",
    `Profile: ${cfg.profile_type}`,
    `Mass: ${Number(cfg.mass).toExponential(3)} M_sun`,
    `Scale radius: ${cfg.scale_radius} kpc`,
    `Ellipticity: ${cfg.ellipticity}`,
    `Grid size: ${cfg.grid_size} x ${cfg.grid_size}`,
    `Convergence range: [${meta.min_value.toExponential(3)}, ${meta.max_value.toExponential(3)}]`,
  ].join("\n");
}

function saveHistory(entry) {
  const key = "lensing_workbench_runs";
  const current = JSON.parse(localStorage.getItem(key) || "[]");
  current.unshift(entry);
  const trimmed = current.slice(0, 20);
  localStorage.setItem(key, JSON.stringify(trimmed));
  renderHistory();
}

function renderHistory() {
  const items = JSON.parse(localStorage.getItem("lensing_workbench_runs") || "[]");
  if (!items.length) {
    output.history.textContent = "No runs recorded yet.";
    return;
  }
  output.history.innerHTML = items
    .map((item) => {
      return `<div class="history-item"><strong>${item.kind}</strong> | ${item.profile} | ${item.grid}x${item.grid} | ${item.ts}</div>`;
    })
    .join("");
}

async function apiCall(path, payload) {
  const response = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload),
  });

  if (!response.ok) {
    const data = await response.json().catch(() => ({}));
    const detail = data.detail || data.error || response.statusText;
    throw new Error(`${response.status}: ${detail}`);
  }

  return response.json();
}

function readSyntheticPayload() {
  return {
    profile_type: controls.profileType.value,
    mass: Number(controls.mass.value),
    scale_radius: Number(controls.scaleRadius.value),
    ellipticity: Number(controls.ellipticity.value),
    grid_size: Number(controls.gridSize.value),
  };
}

function downloadText(filename, content, mime = "text/plain") {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = filename;
  link.click();
  URL.revokeObjectURL(url);
}

controls.generateBtn.addEventListener("click", async () => {
  try {
    const payload = readSyntheticPayload();
    state.syntheticRequest = payload;
    setStatus("Generating synthetic convergence map...");

    const data = await apiCall("/api/v1/synthetic", payload);
    state.syntheticResponse = data;
    state.inferenceResponse = null;

    const map = data.convergence_map;
    const stats = parseMap(map);
    updateMetrics(stats);
    renderPlots(map);

    output.methods.textContent = makeMethodsText();
    output.inference.textContent = "Inference not run yet.";

    controls.inferBtn.disabled = false;
    controls.exportBtn.disabled = false;

    saveHistory({
      kind: "Synthetic",
      profile: payload.profile_type,
      grid: payload.grid_size,
      ts: new Date().toISOString(),
    });

    setStatus(`Generated job ${data.job_id}.`);
  } catch (err) {
    setStatus(`Generation failed: ${err.message}`);
  }
});

controls.inferBtn.addEventListener("click", async () => {
  if (!state.syntheticResponse) {
    setStatus("Generate a map first.");
    return;
  }

  try {
    setStatus("Running inference...");

    const payload = {
      convergence_map: state.syntheticResponse.convergence_map,
      target_size: 64,
      mc_samples: 32,
    };

    const data = await apiCall("/api/v1/inference", payload);
    state.inferenceResponse = data;

    output.inference.textContent = JSON.stringify(data, null, 2);
    setStatus(`Inference complete: job ${data.job_id}.`);
  } catch (err) {
    output.inference.textContent = `Inference failed: ${err.message}`;
    setStatus("Inference failed. See details.");
  }
});

controls.exportBtn.addEventListener("click", () => {
  if (!state.syntheticResponse) {
    setStatus("Nothing to export yet.");
    return;
  }

  const report = {
    generated_at: new Date().toISOString(),
    synthetic_request: state.syntheticRequest,
    synthetic_response: state.syntheticResponse,
    inference_response: state.inferenceResponse,
    methods_text: makeMethodsText(),
  };

  downloadText("lensing_run_package.json", JSON.stringify(report, null, 2), "application/json");
  setStatus("Run package downloaded.");
});

renderHistory();

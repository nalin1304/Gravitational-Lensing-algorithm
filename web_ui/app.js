const state = {
  syntheticRequest: null,
  syntheticResponse: null,
  inferenceResponse: null,
  isBusy: false,
};

const presets = {
  einstein_cross: {
    label: "Einstein Cross",
    profile_type: "Elliptical NFW",
    mass: 1.6e12,
    scale_radius: 160,
    ellipticity: 0.22,
    grid_size: 64,
  },
  twin_quasar: {
    label: "Twin Quasar",
    profile_type: "NFW",
    mass: 2.4e13,
    scale_radius: 290,
    ellipticity: 0.08,
    grid_size: 64,
  },
  jwst_cluster: {
    label: "JWST Cluster",
    profile_type: "Elliptical NFW",
    mass: 7.8e13,
    scale_radius: 410,
    ellipticity: 0.30,
    grid_size: 128,
  },
  generic_demo: {
    label: "Generic Demo",
    profile_type: "NFW",
    mass: 2e12,
    scale_radius: 200,
    ellipticity: 0.2,
    grid_size: 64,
  },
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
  presetButtons: Array.from(document.querySelectorAll(".preset-chip")),
};

const hero = {
  demoCount: document.getElementById("heroDemoCount"),
  speed: document.getElementById("heroSpeed"),
  mode: document.getElementById("heroMode"),
  health: document.getElementById("heroHealth"),
  sync: document.getElementById("heroSync"),
};

const loading = {
  overlay: document.getElementById("loadingOverlay"),
  text: document.getElementById("loadingText"),
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

const plotTheme = {
  paper_bgcolor: "#0A0E1A",
  plot_bgcolor: "#0A0E1A",
  margin: { l: 60, r: 18, t: 32, b: 52 },
  font: { color: "#EAF4FF", family: "Inter, Segoe UI, sans-serif" },
};

function formatScientific(value, digits = 3) {
  return Number.isFinite(value) ? Number(value).toExponential(digits) : "n/a";
}

function setStatus(message, tone = "neutral") {
  controls.status.textContent = message;
  controls.status.classList.remove("status-success", "status-warning", "status-error");
  if (tone === "success") controls.status.classList.add("status-success");
  if (tone === "warning") controls.status.classList.add("status-warning");
  if (tone === "error") controls.status.classList.add("status-error");
}

function updateSyncTime() {
  hero.sync.textContent = new Date().toLocaleTimeString();
}

function setBusy(isBusy, message) {
  state.isBusy = isBusy;
  if (loading.overlay) {
    loading.overlay.classList.toggle("hidden", !isBusy);
  }
  if (loading.text && message) {
    loading.text.textContent = message;
  }
  syncActionButtons();
}

function syncActionButtons() {
  controls.generateBtn.disabled = state.isBusy;
  controls.inferBtn.disabled = state.isBusy || !state.syntheticResponse;
  controls.exportBtn.disabled = state.isBusy || !state.syntheticResponse;
}

function parseMap(matrix) {
  const flat = matrix.flat().filter((v) => Number.isFinite(v));
  if (!flat.length) {
    return { min: 0, max: 0, mean: 0, std: 0 };
  }
  const n = flat.length;
  const mean = flat.reduce((a, b) => a + b, 0) / n;
  const variance = flat.reduce((a, b) => a + (b - mean) ** 2, 0) / n;
  return {
    min: Math.min(...flat),
    max: Math.max(...flat),
    mean,
    std: Math.sqrt(variance),
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
      const radius = Math.floor(Math.hypot(x - cx, y - cy));
      if (radius <= maxR) {
        sums[radius] += matrix[y][x];
        counts[radius] += 1;
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

function computeDeflectionProxy(matrix) {
  const h = matrix.length;
  const w = matrix[0].length;
  const gradX = Array.from({ length: h }, () => new Array(w).fill(0));
  const gradY = Array.from({ length: h }, () => new Array(w).fill(0));
  const magnitude = Array.from({ length: h }, () => new Array(w).fill(0));

  for (let y = 0; y < h; y += 1) {
    for (let x = 0; x < w; x += 1) {
      const left = matrix[y][Math.max(0, x - 1)];
      const right = matrix[y][Math.min(w - 1, x + 1)];
      const up = matrix[Math.max(0, y - 1)][x];
      const down = matrix[Math.min(h - 1, y + 1)][x];
      const gx = 0.5 * (right - left);
      const gy = 0.5 * (down - up);
      gradX[y][x] = gx;
      gradY[y][x] = gy;
      magnitude[y][x] = Math.hypot(gx, gy);
    }
  }

  const stride = Math.max(4, Math.floor(Math.min(h, w) / 14));
  const xLines = [];
  const yLines = [];
  for (let y = stride; y < h - stride; y += stride) {
    for (let x = stride; x < w - stride; x += stride) {
      const dx = -gradX[y][x];
      const dy = -gradY[y][x];
      const norm = Math.hypot(dx, dy);
      if (norm < 1e-10) continue;
      const scale = (1.9 * stride) / norm;
      xLines.push(x, x + dx * scale, null);
      yLines.push(y, y + dy * scale, null);
    }
  }

  return { magnitude, xLines, yLines };
}

function renderPlots(matrix) {
  if (typeof Plotly === "undefined") {
    setStatus("Plot library not available in this browser session.", "error");
    return;
  }

  const heatmap = {
    z: matrix,
    type: "heatmap",
    colorscale: "Viridis",
    colorbar: { title: "κ", titlefont: { color: "#EAF4FF" } },
  };

  Plotly.react(
    "mapPlot",
    [heatmap],
    {
      ...plotTheme,
      xaxis: { title: "x pixel" },
      yaxis: { title: "y pixel" },
    },
    { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] },
  );

  const proxy = computeDeflectionProxy(matrix);
  const proxyHeat = {
    z: proxy.magnitude,
    type: "heatmap",
    colorscale: "Cividis",
    opacity: 0.9,
    colorbar: { title: "|∇κ|", titlefont: { color: "#EAF4FF" } },
  };
  const arrows = {
    x: proxy.xLines,
    y: proxy.yLines,
    mode: "lines",
    type: "scattergl",
    line: { color: "rgba(0, 212, 255, 0.75)", width: 1.3 },
    hoverinfo: "skip",
    name: "Deflection proxy vectors",
  };

  Plotly.react(
    "deflectionPlot",
    [proxyHeat, arrows],
    {
      ...plotTheme,
      xaxis: { title: "x pixel" },
      yaxis: { title: "y pixel", scaleanchor: "x" },
    },
    { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] },
  );

  const profile = radialProfile(matrix);
  const radialTrace = {
    x: profile.radius,
    y: profile.values,
    type: "scatter",
    mode: "lines",
    fill: "tozeroy",
    line: { color: "#00D4FF", width: 2.2 },
    fillcolor: "rgba(0, 212, 255, 0.15)",
    name: "Mean κ(r)",
  };

  Plotly.react(
    "radialPlot",
    [radialTrace],
    {
      ...plotTheme,
      xaxis: { title: "Radius (pixels)" },
      yaxis: { title: "Mean κ" },
    },
    { responsive: true, displaylogo: false, modeBarButtonsToRemove: ["lasso2d", "select2d"] },
  );
}

function updateMetrics(stats) {
  metrics.min.textContent = formatScientific(stats.min);
  metrics.mean.textContent = formatScientific(stats.mean);
  metrics.max.textContent = formatScientific(stats.max);
  metrics.std.textContent = formatScientific(stats.std);
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
    "Synthetic convergence maps were generated with the API thin-lens pipeline.",
    `Profile: ${cfg.profile_type}`,
    `Virial mass: ${formatScientific(Number(cfg.mass))} M_sun`,
    `Scale radius: ${cfg.scale_radius} kpc`,
    `Ellipticity: ${cfg.ellipticity}`,
    `Grid size: ${cfg.grid_size} x ${cfg.grid_size}`,
    `Convergence range: [${formatScientific(meta.min_value)}, ${formatScientific(meta.max_value)}]`,
  ].join("\n");
}

function renderInferenceText(response) {
  if (!response) {
    return "No inference run yet.";
  }
  const predictionLines = Object.entries(response.predictions || {}).map(
    ([name, value]) => `  - ${name}: ${formatScientific(value)}`,
  );
  const uncertaintyLines = response.uncertainties
    ? Object.entries(response.uncertainties).map(([name, value]) => `  - ${name}: ${formatScientific(value)}`)
    : ["  - none (single-sample inference)"];
  const classLines = Object.entries(response.classification || {})
    .sort((a, b) => b[1] - a[1])
    .map(([name, value]) => `  - ${name}: ${(100 * value).toFixed(2)}%`);

  return [
    `Inference mode: ${response.inference_mode || "pinn"}`,
    `Job ID: ${response.job_id}`,
    "",
    "Predictions:",
    ...predictionLines,
    "",
    "Uncertainties:",
    ...uncertaintyLines,
    "",
    "Classification:",
    ...classLines,
    "",
    `Entropy: ${Number(response.entropy).toFixed(4)}`,
  ].join("\n");
}

function saveHistory(entry) {
  const key = "lensing_workbench_runs";
  const current = JSON.parse(localStorage.getItem(key) || "[]");
  current.unshift(entry);
  const trimmed = current.slice(0, 24);
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
      return (
        `<div class="history-item"><strong>${item.kind}</strong>` +
        ` | ${item.profile} | ${item.grid}x${item.grid}` +
        ` | mode=${item.mode || "pending"} | ${item.ts}</div>`
      );
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

async function fetchBackendHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json();
    const statusLabel = data.status || "unknown";
    const gpu = data.gpu_available ? "GPU" : "CPU";
    hero.health.textContent = `${statusLabel.toUpperCase()} • ${gpu}`;
    updateSyncTime();
  } catch {
    hero.health.textContent = "Health check unavailable";
  }
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

function applyPreset(presetKey) {
  const preset = presets[presetKey];
  if (!preset) return;
  controls.profileType.value = preset.profile_type;
  controls.mass.value = String(preset.mass);
  controls.scaleRadius.value = String(preset.scale_radius);
  controls.ellipticity.value = String(preset.ellipticity);
  controls.gridSize.value = String(preset.grid_size);
  setStatus(`Preset loaded: ${preset.label}.`, "success");
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

controls.presetButtons.forEach((button) => {
  button.addEventListener("click", () => {
    const presetKey = button.dataset.preset;
    applyPreset(presetKey);
  });
});

controls.generateBtn.addEventListener("click", async () => {
  try {
    const payload = readSyntheticPayload();
    state.syntheticRequest = payload;
    state.inferenceResponse = null;
    hero.mode.textContent = "Synthetic ready";
    setBusy(true, "Generating synthetic convergence map...");
    setStatus("Generating synthetic convergence map...", "warning");

    const data = await apiCall("/api/v1/synthetic", payload);
    state.syntheticResponse = data;

    const map = data.convergence_map;
    const stats = parseMap(map);
    updateMetrics(stats);
    renderPlots(map);

    output.methods.textContent = makeMethodsText();
    output.inference.textContent = "Inference not run yet.";

    saveHistory({
      kind: "Synthetic",
      profile: payload.profile_type,
      grid: payload.grid_size,
      mode: "synthetic_only",
      ts: new Date().toISOString(),
    });

    setStatus(`Generated job ${data.job_id}.`, "success");
    updateSyncTime();
  } catch (error) {
    setStatus(`Generation failed: ${error.message}`, "error");
  } finally {
    setBusy(false, "");
  }
});

controls.inferBtn.addEventListener("click", async () => {
  if (!state.syntheticResponse) {
    setStatus("Generate a map first.", "warning");
    return;
  }

  try {
    setBusy(true, "Running inference...");
    setStatus("Running inference...", "warning");

    const payload = {
      convergence_map: state.syntheticResponse.convergence_map,
      target_size: 64,
      mc_samples: 32,
    };

    const data = await apiCall("/api/v1/inference", payload);
    state.inferenceResponse = data;

    output.inference.textContent = renderInferenceText(data);
    hero.mode.textContent = data.inference_mode || "pinn";

    const modeTone = data.inference_mode === "physics_fallback" ? "warning" : "success";
    const modeMessage =
      data.inference_mode === "physics_fallback"
        ? "Inference complete via physics fallback estimator."
        : `Inference complete: job ${data.job_id}.`;
    setStatus(modeMessage, modeTone);

    saveHistory({
      kind: "Inference",
      profile: state.syntheticRequest?.profile_type || "unknown",
      grid: state.syntheticRequest?.grid_size || 0,
      mode: data.inference_mode || "pinn",
      ts: new Date().toISOString(),
    });
    updateSyncTime();
  } catch (error) {
    output.inference.textContent = `Inference failed: ${error.message}`;
    hero.mode.textContent = "Failure";
    setStatus("Inference failed. See details.", "error");
  } finally {
    setBusy(false, "");
  }
});

controls.exportBtn.addEventListener("click", () => {
  if (!state.syntheticResponse) {
    setStatus("Nothing to export yet.", "warning");
    return;
  }

  const report = {
    generated_at: new Date().toISOString(),
    ui_profile: "journal_workbench_v2",
    synthetic_request: state.syntheticRequest,
    synthetic_response: state.syntheticResponse,
    inference_response: state.inferenceResponse,
    methods_text: makeMethodsText(),
  };

  downloadText("lensing_run_package.json", JSON.stringify(report, null, 2), "application/json");
  setStatus("Run package downloaded.", "success");
});

hero.demoCount.textContent = String(Object.keys(presets).length - 1);
hero.speed.textContent = "134.6 img/s";
syncActionButtons();
renderHistory();
fetchBackendHealth();
setInterval(fetchBackendHealth, 60000);

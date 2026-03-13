/**
 * Inference Page — NUTS-HMC Differentiable Inference Engine
 *
 * Provides:
 * 1. Forward model panel with NFW parameter controls → simulate observation
 * 2. NUTS-HMC posterior sampling with corner/trace plots
 * 3. Fisher information matrix computation
 * 4. Method comparison reference card
 *
 * API endpoints used:
 *   POST /api/v1/nuts/simulate
 *   POST /api/v1/nuts/posterior
 *   POST /api/v1/nuts/fisher
 */

/* global Plotly */
const P = () => window.LensPINN;

const plotTheme = {
  paper_bgcolor: 'rgba(0,0,0,0)',
  plot_bgcolor: 'rgba(0,0,0,0)',
  font: { color: '#c9d1e0', family: 'Inter' },
};

let lastSimResult = null;

export function render() {
  return `
<div class="page-content">

  <!-- Header -->
  <div class="alert alert-info" style="margin-bottom:1.5rem">
    <strong>⚛️ Differentiable Inference Engine</strong>
    NUTS-HMC posterior sampling with automatic differentiation.
    Gradient-based No U-Turn Sampler for efficient exploration of lens model parameter space.
  </div>

  <div class="grid-2col mb-16">

    <!-- LEFT: Forward Model Panel -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">Forward Model</span>
        <span class="badge badge-info">NFW</span>
      </div>

      <div class="form-group">
        <label class="form-label">log₁₀(M<sub>vir</sub> / M☉) <span id="infMvirVal" class="badge">14.0</span></label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="range" id="infLogMvir" min="10" max="15" step="0.1" value="14.0" class="slider" style="flex:1"
                 oninput="document.getElementById('infMvirVal').textContent=this.value;document.getElementById('infMvirNum').value=this.value" />
          <input type="number" id="infMvirNum" class="form-input" style="width:80px" min="10" max="15" step="0.1" value="14.0"
                 oninput="document.getElementById('infMvirVal').textContent=this.value;document.getElementById('infLogMvir').value=this.value" />
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">Concentration <span id="infConcVal" class="badge">5.0</span></label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="range" id="infConc" min="1" max="20" step="0.5" value="5.0" class="slider" style="flex:1"
                 oninput="document.getElementById('infConcVal').textContent=this.value;document.getElementById('infConcNum').value=this.value" />
          <input type="number" id="infConcNum" class="form-input" style="width:80px" min="1" max="20" step="0.5" value="5.0"
                 oninput="document.getElementById('infConcVal').textContent=this.value;document.getElementById('infConc').value=this.value" />
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">z<sub>lens</sub> <span id="infZlVal" class="badge">0.30</span></label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="range" id="infZl" min="0.05" max="2.0" step="0.05" value="0.30" class="slider" style="flex:1"
                 oninput="document.getElementById('infZlVal').textContent=parseFloat(this.value).toFixed(2);document.getElementById('infZlNum').value=this.value" />
          <input type="number" id="infZlNum" class="form-input" style="width:80px" min="0.05" max="2.0" step="0.05" value="0.30"
                 oninput="document.getElementById('infZlVal').textContent=parseFloat(this.value).toFixed(2);document.getElementById('infZl').value=this.value" />
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">z<sub>source</sub> <span id="infZsVal" class="badge">1.50</span></label>
        <div style="display:flex;gap:8px;align-items:center">
          <input type="range" id="infZs" min="0.1" max="5.0" step="0.1" value="1.50" class="slider" style="flex:1"
                 oninput="document.getElementById('infZsVal').textContent=parseFloat(this.value).toFixed(2);document.getElementById('infZsNum').value=this.value" />
          <input type="number" id="infZsNum" class="form-input" style="width:80px" min="0.1" max="5.0" step="0.1" value="1.50"
                 oninput="document.getElementById('infZsVal').textContent=parseFloat(this.value).toFixed(2);document.getElementById('infZs').value=this.value" />
        </div>
      </div>

      <div class="form-group">
        <label class="form-label">Grid Size</label>
        <select id="infGrid" class="form-select">
          <option value="32">32×32</option>
          <option value="64" selected>64×64</option>
          <option value="128">128×128</option>
        </select>
      </div>

      <button id="infSimBtn" class="btn btn-primary" style="width:100%">▶ Simulate</button>
    </div>

    <!-- RIGHT: Posterior Sampling Panel -->
    <div class="card">
      <div class="card-header">
        <span class="card-title">NUTS-HMC Posterior Sampling</span>
        <span class="badge badge-purple">HMC</span>
      </div>

      <div class="form-group">
        <label class="form-label">Number of Samples</label>
        <input type="number" id="infNSamples" class="form-input" min="50" max="2000" step="50" value="500" />
      </div>

      <div class="form-group">
        <label class="form-label">Warmup Steps</label>
        <input type="number" id="infWarmup" class="form-input" min="50" max="500" step="50" value="200" />
      </div>

      <div class="form-group">
        <label class="form-label">Noise σ</label>
        <input type="number" id="infNoiseSigma" class="form-input" min="0.001" max="0.1" step="0.001" value="0.01" />
      </div>

      <button id="infRunNuts" class="btn btn-primary" style="width:100%;margin-bottom:12px" disabled>🔬 Run NUTS-HMC</button>

      <div id="infSamplerStats" style="display:none">
        <div class="metric-row">
          <span class="metric-key">Acceptance Rate</span>
          <span class="metric-val" id="infAcceptRate">—</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Wall Time</span>
          <span class="metric-val" id="infWallTime">—</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Effective Samples</span>
          <span class="metric-val" id="infEffSamples">—</span>
        </div>
      </div>
    </div>

  </div>

  <!-- Simulation output plots -->
  <div class="grid-2 mb-16" id="infSimPlots" style="display:none">
    <div class="card">
      <div class="card-header"><span class="card-title">Convergence Map κ(θ)</span></div>
      <div id="infKappaPlot" style="height:320px"></div>
    </div>
    <div class="card">
      <div class="card-header"><span class="card-title">Lensed Image</span></div>
      <div id="infLensedPlot" style="height:320px"></div>
    </div>
  </div>

  <!-- Posterior plots -->
  <div id="infPosteriorSection" style="display:none">
    <div class="grid-2 mb-16">
      <div class="card">
        <div class="card-header"><span class="card-title">Corner Plot: M<sub>vir</sub> vs Concentration</span></div>
        <div id="infCornerPlot" style="height:350px"></div>
      </div>
      <div class="card">
        <div class="card-header"><span class="card-title">Trace Plots</span></div>
        <div id="infTracePlot" style="height:350px"></div>
      </div>
    </div>
  </div>

  <!-- Fisher Information Panel -->
  <div class="card mb-16">
    <div class="card-header">
      <span class="card-title">Fisher Information Matrix</span>
      <button id="infFisherBtn" class="btn btn-secondary btn-sm" disabled>Compute Fisher Matrix</button>
    </div>
    <div id="infFisherContent">
      <p class="dim small">Run a simulation first, then compute the Fisher information matrix.</p>
    </div>
    <div class="grid-2" id="infFisherPlots" style="display:none">
      <div>
        <table class="data-table" id="infFisherTable"></table>
      </div>
      <div id="infFisherHeatmap" style="height:300px"></div>
    </div>
  </div>

  <!-- Method Comparison Card -->
  <div class="card">
    <div class="card-header">
      <span class="card-title">Method Overview</span>
      <span class="badge badge-info">Differentiable</span>
    </div>
    <div class="metric-row">
      <span class="metric-key">Algorithm</span>
      <span class="metric-val" style="font-family:var(--font);font-weight:500">NUTS-HMC: Gradient-based sampling with No U-Turn Stopping</span>
    </div>
    <div class="metric-row">
      <span class="metric-key">Key Advantage</span>
      <span class="metric-val" style="font-family:var(--font);font-weight:400;color:var(--text-secondary)">
        Automatic differentiation through the forward model enables efficient high-dimensional posterior exploration
      </span>
    </div>
    <div class="metric-row">
      <span class="metric-key">References</span>
      <span class="metric-val small" style="font-family:var(--font);font-weight:400;color:var(--text-muted)">
        Hoffman &amp; Gelman (2014) JMLR 15, 1593–1623 · Galan et al. (2022) A&amp;A 668, A155
      </span>
    </div>
  </div>

</div>
`;
}

export async function init() {
  document.getElementById('infSimBtn').addEventListener('click', runSimulation);
  document.getElementById('infRunNuts').addEventListener('click', runNuts);
  document.getElementById('infFisherBtn').addEventListener('click', runFisher);
}

function readSimParams() {
  return {
    log10_M_vir: parseFloat(document.getElementById('infLogMvir').value),
    concentration: parseFloat(document.getElementById('infConc').value),
    z_lens: parseFloat(document.getElementById('infZl').value),
    z_source: parseFloat(document.getElementById('infZs').value),
    grid_size: parseInt(document.getElementById('infGrid').value, 10),
  };
}

async function runSimulation() {
  const btn = document.getElementById('infSimBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Simulating…';

  try {
    P().showLoading('Running forward model…');
    const body = readSimParams();
    const resp = await P().api('/api/v1/nuts/simulate', { method: 'POST', body });
    if (!resp) { P().toast('Simulation returned empty', 'error'); return; }

    lastSimResult = resp;

    // Show plots section
    document.getElementById('infSimPlots').style.display = '';

    // Convergence heatmap
    if (window.Plotly && resp.convergence_map) {
      Plotly.newPlot('infKappaPlot', [{
        z: resp.convergence_map, type: 'heatmap', colorscale: 'Viridis',
        showscale: true, colorbar: { title: 'κ', thickness: 14 },
      }], {
        ...plotTheme,
        margin: { t: 10, b: 40, l: 40, r: 10 },
        xaxis: { title: 'θ_x (pixels)', color: '#8899aa' },
        yaxis: { title: 'θ_y (pixels)', color: '#8899aa' },
      }, { responsive: true, displayModeBar: false });
    }

    // Lensed image heatmap
    if (window.Plotly && resp.lensed_image) {
      Plotly.newPlot('infLensedPlot', [{
        z: resp.lensed_image, type: 'heatmap', colorscale: 'Plasma',
        showscale: true, colorbar: { title: 'I', thickness: 14 },
      }], {
        ...plotTheme,
        margin: { t: 10, b: 40, l: 40, r: 10 },
        xaxis: { title: 'θ_x (pixels)', color: '#8899aa' },
        yaxis: { title: 'θ_y (pixels)', color: '#8899aa' },
      }, { responsive: true, displayModeBar: false });
    }

    // Enable posterior + fisher buttons
    document.getElementById('infRunNuts').disabled = false;
    document.getElementById('infFisherBtn').disabled = false;
    P().toast('Forward model simulation complete', 'success');
  } catch (e) {
    P().toast(`Simulation failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Simulate';
    P().hideLoading();
  }
}

async function runNuts() {
  if (!lastSimResult) { P().toast('Run a simulation first', 'warning'); return; }

  const btn = document.getElementById('infRunNuts');
  btn.disabled = true;
  btn.textContent = '⏳ Sampling…';

  try {
    P().showLoading('Running NUTS-HMC posterior sampling…');
    const body = {
      ...readSimParams(),
      observation: lastSimResult.convergence_map,
      n_samples: parseInt(document.getElementById('infNSamples').value, 10),
      warmup: parseInt(document.getElementById('infWarmup').value, 10),
      noise_std: parseFloat(document.getElementById('infNoiseSigma').value),
    };
    const resp = await P().api('/api/v1/nuts/posterior', { method: 'POST', body });
    if (!resp) { P().toast('Posterior returned empty', 'error'); return; }

    // Show stats
    const statsEl = document.getElementById('infSamplerStats');
    statsEl.style.display = '';
    document.getElementById('infAcceptRate').textContent =
      resp.acceptance_rate != null ? (resp.acceptance_rate * 100).toFixed(1) + '%' : '—';
    document.getElementById('infWallTime').textContent =
      resp.wall_time_s != null ? resp.wall_time_s.toFixed(2) + ' s' : '—';
    document.getElementById('infEffSamples').textContent =
      resp.n_effective != null ? resp.n_effective : '—';

    // Show posterior section
    document.getElementById('infPosteriorSection').style.display = '';

    const samples_mvir = resp.samples?.log10_M_vir || resp.samples?.param_0 || [];
    const samples_conc = resp.samples?.concentration || resp.samples?.param_1 || [];

    // Corner plot (2D scatter)
    if (window.Plotly && samples_mvir.length && samples_conc.length) {
      Plotly.newPlot('infCornerPlot', [{
        x: samples_mvir, y: samples_conc, mode: 'markers',
        type: 'scattergl',
        marker: { size: 3, color: '#38bdf8', opacity: 0.4 },
        name: 'Posterior samples',
      }], {
        ...plotTheme,
        margin: { t: 10, b: 50, l: 60, r: 20 },
        xaxis: { title: 'log₁₀(M_vir)', color: '#8899aa', gridcolor: 'rgba(255,255,255,0.07)' },
        yaxis: { title: 'Concentration', color: '#8899aa', gridcolor: 'rgba(255,255,255,0.07)' },
      }, { responsive: true, displayModeBar: false });
    }

    // Trace plots
    if (window.Plotly && samples_mvir.length) {
      const idx = samples_mvir.map((_, i) => i);
      Plotly.newPlot('infTracePlot', [
        { x: idx, y: samples_mvir, mode: 'lines', name: 'log₁₀(M_vir)',
          line: { color: '#38bdf8', width: 1 } },
        { x: idx, y: samples_conc, mode: 'lines', name: 'Concentration',
          line: { color: '#a78bfa', width: 1 }, yaxis: 'y2' },
      ], {
        ...plotTheme,
        margin: { t: 10, b: 50, l: 60, r: 60 },
        xaxis: { title: 'Sample index', color: '#8899aa', gridcolor: 'rgba(255,255,255,0.07)' },
        yaxis: { title: 'log₁₀(M_vir)', color: '#38bdf8', gridcolor: 'rgba(255,255,255,0.07)' },
        yaxis2: { title: 'Concentration', color: '#a78bfa', overlaying: 'y', side: 'right',
                  gridcolor: 'rgba(255,255,255,0.07)' },
        legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(0,0,0,0)' },
      }, { responsive: true, displayModeBar: false });
    }

    P().toast('NUTS-HMC posterior sampling complete', 'success');
  } catch (e) {
    P().toast(`NUTS-HMC failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '🔬 Run NUTS-HMC';
    P().hideLoading();
  }
}

async function runFisher() {
  if (!lastSimResult) { P().toast('Run a simulation first', 'warning'); return; }

  const btn = document.getElementById('infFisherBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Computing…';

  try {
    P().showLoading('Computing Fisher information matrix…');
    const body = readSimParams();
    const resp = await P().api('/api/v1/nuts/fisher', { method: 'POST', body });
    if (!resp) { P().toast('Fisher returned empty', 'error'); return; }

    const paramNames = resp.param_names || ['log₁₀(M_vir)', 'Concentration', 'z_lens', 'z_source'];
    const marginalErrors = resp.marginal_errors || [];
    const corrMatrix = resp.correlation_matrix || [];

    // Fisher table
    const tableEl = document.getElementById('infFisherTable');
    tableEl.innerHTML = `
      <thead><tr><th>Parameter</th><th>Marginal Error (1σ)</th></tr></thead>
      <tbody>
        ${paramNames.map((name, i) => `
          <tr>
            <td>${P().esc(name)}</td>
            <td>${marginalErrors[i] != null ? marginalErrors[i].toExponential(3) : '—'}</td>
          </tr>`).join('')}
      </tbody>
    `;

    // Show fisher plots section
    const plotsEl = document.getElementById('infFisherPlots');
    plotsEl.style.display = '';
    document.getElementById('infFisherContent').querySelector('p')?.remove();

    // Correlation heatmap
    if (window.Plotly && corrMatrix.length) {
      Plotly.newPlot('infFisherHeatmap', [{
        z: corrMatrix, x: paramNames, y: paramNames,
        type: 'heatmap', colorscale: 'RdBu', zmin: -1, zmax: 1,
        showscale: true, colorbar: { title: 'ρ', thickness: 14 },
      }], {
        ...plotTheme,
        margin: { t: 10, b: 80, l: 100, r: 10 },
        xaxis: { color: '#8899aa', tickangle: -45 },
        yaxis: { color: '#8899aa' },
      }, { responsive: true, displayModeBar: false });
    }

    P().toast('Fisher matrix computed', 'success');
  } catch (e) {
    P().toast(`Fisher computation failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = 'Compute Fisher Matrix';
    P().hideLoading();
  }
}

export function cleanup() {
  lastSimResult = null;
  ['infKappaPlot', 'infLensedPlot', 'infCornerPlot', 'infTracePlot', 'infFisherHeatmap'].forEach(id => {
    const el = document.getElementById(id);
    if (el && window.Plotly) { try { Plotly.purge(el); } catch (_) { /* noop */ } }
  });
}

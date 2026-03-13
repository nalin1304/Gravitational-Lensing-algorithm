/**
 * PI-SBI Page — Physics-Informed Multi-Messenger Posterior Estimation
 *
 * Provides:
 * 1. Parameter slider panel → simulate observation (κ map + GW spectrum)
 * 2. Visualize output (heatmap + spectrum chart)
 * 3. Run posterior inference → show parameter posterior with uncertainties
 * 4. Status panel showing model availability
 *
 * API endpoints used:
 *   GET  /api/v1/pi-sbi/status
 *   POST /api/v1/pi-sbi/simulate
 *   POST /api/v1/pi-sbi/posterior
 */

/* global Plotly */
const P = () => window.LensPINN;

export function render() {
  return `
<div class="page-content">

  <!-- Hero banner -->
  <div class="alert alert-info" style="margin-bottom:1.5rem">
    <strong>🌌 PI-SBI: Physics-Informed Multi-Messenger Inference</strong>
    Joint optical (Einstein ring) + gravitational wave posterior estimation.
    ~10,000× faster than MCMC · Physics-constrained summary network (∇²ψ = 2κ).
    <em>First joint EM+GW amortized posterior for strong gravitational lensing.</em>
  </div>

  <div class="grid-2col">

    <!-- LEFT: Parameter controls -->
    <div class="card">
      <h3 class="card-title">Lens Parameters</h3>
      <p class="dim" style="margin-bottom:1rem">
        Drawn from SLACS-calibrated prior (Bolton et al. 2006; Auger et al. 2009).
      </p>
      <div class="form-group">
        <label>log₁₀(M<sub>vir</sub> / M☉) <span id="mvirVal" class="badge">12.0</span></label>
        <input type="range" id="logMvir" min="10" max="14" step="0.1" value="12.0" class="slider"
               oninput="document.getElementById('mvirVal').textContent=this.value" />
      </div>
      <div class="form-group">
        <label>log₁₀(r<sub>s</sub> / arcsec) <span id="rsVal" class="badge">0.3</span></label>
        <input type="range" id="logRs" min="-0.5" max="1.5" step="0.05" value="0.3" class="slider"
               oninput="document.getElementById('rsVal').textContent=parseFloat(this.value).toFixed(2)" />
      </div>
      <div class="form-group">
        <label>Lens redshift z<sub>l</sub> <span id="zlVal" class="badge">0.30</span></label>
        <input type="range" id="zl" min="0.06" max="0.50" step="0.01" value="0.30" class="slider"
               oninput="document.getElementById('zlVal').textContent=parseFloat(this.value).toFixed(2)" />
      </div>
      <div class="form-group">
        <label>Source redshift z<sub>s</sub> <span id="zsVal" class="badge">1.00</span></label>
        <input type="range" id="zs" min="0.5" max="2.5" step="0.05" value="1.00" class="slider"
               oninput="document.getElementById('zsVal').textContent=parseFloat(this.value).toFixed(2)" />
      </div>
      <div class="form-group">
        <label>Source offset β<sub>x</sub> <span id="bxVal" class="badge">0.00</span> arcsec</label>
        <input type="range" id="betaX" min="-0.3" max="0.3" step="0.01" value="0.0" class="slider"
               oninput="document.getElementById('bxVal').textContent=parseFloat(this.value).toFixed(2)" />
      </div>
      <div class="form-group">
        <label>Source offset β<sub>y</sub> <span id="byVal" class="badge">0.00</span> arcsec</label>
        <input type="range" id="betaY" min="-0.3" max="0.3" step="0.01" value="0.0" class="slider"
               oninput="document.getElementById('byVal').textContent=parseFloat(this.value).toFixed(2)" />
      </div>
      <button class="btn btn-primary" id="simBtn">▶ Simulate Observation</button>
    </div>

    <!-- RIGHT: Model status -->
    <div class="card" id="statusCard">
      <h3 class="card-title">Model Status</h3>
      <div id="statusContent"><p class="dim">Loading…</p></div>
    </div>

  </div>

  <!-- Simulation outputs -->
  <div class="grid-2col" id="simOutputRow" style="display:none;margin-top:1.5rem">
    <div class="card">
      <h3 class="card-title">κ Map (Convergence)</h3>
      <div id="kappaPlot" style="height:300px"></div>
      <p class="dim small" style="margin-top:.5rem">
        NFW projected convergence — Wright &amp; Brainerd (2000), ApJ 534, 34
      </p>
    </div>
    <div class="card">
      <h3 class="card-title">GW Spectrum |F(ω)|²</h3>
      <div id="gwPlot" style="height:300px"></div>
      <p class="dim small" style="margin-top:.5rem">
        Wave-optics amplification — Nakamura &amp; Deguchi (1999), PTPS 133
      </p>
    </div>
  </div>

  <!-- Posterior panel -->
  <div class="card" id="posteriorCard" style="display:none;margin-top:1.5rem">
    <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:1rem">
      <h3 class="card-title" style="margin:0">Posterior p(θ | d<sub>EM</sub>, d<sub>GW</sub>)</h3>
      <button class="btn btn-secondary" id="inferBtn">🧠 Run PI-SBI Posterior</button>
    </div>
    <div id="posteriorContent"><p class="dim">Click "Run PI-SBI Posterior" after simulating.</p></div>
    <div id="posteriorPlot" style="height:350px;display:none"></div>
  </div>

</div>
`;
}

export async function init() {
  // Load status with error handling
  try {
    await loadStatus();
  } catch (e) {
    const el = document.getElementById('statusContent');
    if (el) el.innerHTML = `<p class="dim">Could not reach PI-SBI service.</p>`;
  }

  document.getElementById('simBtn').addEventListener('click', runSimulation);
  document.getElementById('inferBtn').addEventListener('click', runPosterior);
}

async function loadStatus() {
  const resp = await P().api('/api/v1/pi-sbi/status');
  if (!resp) return;

  const el = document.getElementById('statusContent');
  const ckpt = resp.checkpoint_available;
  el.innerHTML = `
    <div class="metric-row">
      <span class="metric-key">Checkpoint</span>
      <span class="metric-val ${ckpt ? 'text-success' : 'text-warning'}">
        ${ckpt ? '✓ Ready' : '⚠ Not trained yet'}
      </span>
    </div>
    <div class="metric-row">
      <span class="metric-key">Architecture</span>
      <span class="metric-val small dim">${resp.architecture || '—'}</span>
    </div>
    <div class="metric-row">
      <span class="metric-key">Prior source</span>
      <span class="metric-val small dim">${resp.prior_source || '—'}</span>
    </div>
    <div class="metric-row">
      <span class="metric-key">GW noise model</span>
      <span class="metric-val small dim">${resp.gw_model || '—'}</span>
    </div>
    ${resp.training_n_sims ? `
    <div class="metric-row">
      <span class="metric-key">Training simulations</span>
      <span class="metric-val">${resp.training_n_sims.toLocaleString()}</span>
    </div>
    <div class="metric-row">
      <span class="metric-key">Final NLL</span>
      <span class="metric-val">${resp.final_nll?.toFixed(4) ?? '—'}</span>
    </div>` : ''}
    ${!ckpt ? `<p class="dim small" style="margin-top:1rem">
      To train: <code>python3 scripts/train_pi_sbi.py --n-sims 10000 --epochs 50</code>
    </p>` : ''}
    <p class="dim small" style="margin-top:.5rem">${resp.novel_claim || ''}</p>
  `;
}

// Store last simulation for posterior call
let lastSim = null;

async function runSimulation() {
  const btn = document.getElementById('simBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Simulating…';

  const body = {
    log10_M_vir: parseFloat(document.getElementById('logMvir').value),
    log10_r_s: parseFloat(document.getElementById('logRs').value),
    z_l: parseFloat(document.getElementById('zl').value),
    z_s: parseFloat(document.getElementById('zs').value),
    beta_x: parseFloat(document.getElementById('betaX').value),
    beta_y: parseFloat(document.getElementById('betaY').value),
    grid_size: 64,
    n_omega: 32,
    seed: 42,
  };

  try {
    const resp = await P().api('/api/v1/pi-sbi/simulate', { method: 'POST', body });
    if (!resp) { P().toast('Simulation returned empty', 'error'); return; }

    lastSim = resp;

    // Show output row
    document.getElementById('simOutputRow').style.display = '';
    document.getElementById('posteriorCard').style.display = '';

    // Plot κ map
    if (window.Plotly && resp.kappa_map) {
      Plotly.newPlot('kappaPlot', [{
        z: resp.kappa_map,
        type: 'heatmap',
        colorscale: 'Plasma',
        showscale: true,
        colorbar: { title: 'κ', thickness: 14 },
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#c9d1e0' },
        margin: { t: 10, b: 40, l: 40, r: 10 },
        xaxis: { title: 'θ_x (pixels)', color: '#8899aa' },
        yaxis: { title: 'θ_y (pixels)', color: '#8899aa' },
      }, { responsive: true, displayModeBar: false });
    }

    // Plot GW spectrum
    if (window.Plotly && resp.gw_spectrum) {
      Plotly.newPlot('gwPlot', [{
        x: resp.omega_dimensionless,
        y: resp.gw_spectrum,
        mode: 'lines+markers',
        line: { color: '#4cc9f0', width: 2 },
        marker: { size: 4 },
        name: '|F(ω)|²',
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#c9d1e0' },
        margin: { t: 10, b: 50, l: 60, r: 10 },
        xaxis: { title: 'ω (dimensionless)', type: 'log', color: '#8899aa',
                 gridcolor: 'rgba(255,255,255,0.07)' },
        yaxis: { title: '|F(ω)|²', color: '#8899aa',
                 gridcolor: 'rgba(255,255,255,0.07)' },
      }, { responsive: true, displayModeBar: false });
    }

    P().toast('Simulation complete', 'success');
  } catch (e) {
    P().toast(`Simulation failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '▶ Simulate Observation';
  }
}

async function runPosterior() {
  if (!lastSim) { P().toast('Simulate first', 'warning'); return; }

  const btn = document.getElementById('inferBtn');
  btn.disabled = true;
  btn.textContent = '⏳ Estimating…';

  const body = {
    kappa_map: lastSim.kappa_map,
    gw_spectrum: lastSim.gw_spectrum,
    n_samples: 500,
  };

  try {
    const resp = await P().api('/api/v1/pi-sbi/posterior', { method: 'POST', body });
    if (!resp) { P().toast('Posterior returned empty', 'error'); return; }

    const el = document.getElementById('posteriorContent');

    if (resp.status === 'checkpoint_missing') {
      el.innerHTML = `<div class="alert alert-warning">
        <strong>Checkpoint not found.</strong> ${P().esc(resp.message || '')}
      </div>`;
      return;
    }

    const names = resp.param_names || [];
    const mean = resp.posterior_mean || [];
    const std = resp.posterior_std || [];

    const labels = ['log₁₀(M<sub>vir</sub>)', 'log₁₀(r<sub>s</sub>)',
                    'z<sub>l</sub>', 'z<sub>s</sub>', 'β<sub>x</sub>', 'β<sub>y</sub>'];
    el.innerHTML = `
      <table class="data-table" style="margin-bottom:1rem">
        <thead><tr><th>Parameter</th><th>Mean</th><th>Std</th><th>95% CI</th></tr></thead>
        <tbody>
          ${mean.map((m, i) => `
            <tr>
              <td>${labels[i] || names[i]}</td>
              <td>${m.toFixed(4)}</td>
              <td>${(std[i] || 0).toFixed(4)}</td>
              <td>[${(m - 2*(std[i]||0)).toFixed(3)}, ${(m + 2*(std[i]||0)).toFixed(3)}]</td>
            </tr>`).join('')}
        </tbody>
      </table>
      <p class="dim small">
        ${resp.n_samples} posterior samples · ${resp.inference_mode || ''} ·
        Physics: <em>${resp.physics_constraint || ''}</em> ·
        Ref: ${resp.reference || ''}
      </p>
    `;

    // Posterior bar chart (mean ± 1σ)
    if (window.Plotly && mean.length) {
      document.getElementById('posteriorPlot').style.display = '';
      Plotly.newPlot('posteriorPlot', [{
        type: 'bar',
        x: labels.map((l, i) => names[i] || l),
        y: mean,
        error_y: { type: 'data', array: std, visible: true, color: '#f72585' },
        marker: { color: '#4361ee' },
      }], {
        paper_bgcolor: 'rgba(0,0,0,0)',
        plot_bgcolor: 'rgba(0,0,0,0)',
        font: { color: '#c9d1e0' },
        margin: { t: 20, b: 60, l: 60, r: 20 },
        xaxis: { color: '#8899aa' },
        yaxis: { title: 'Posterior mean', color: '#8899aa',
                 gridcolor: 'rgba(255,255,255,0.07)' },
      }, { responsive: true, displayModeBar: false });
    }

    P().toast('Posterior complete', 'success');
  } catch (e) {
    P().toast(`Posterior failed: ${e.message}`, 'error');
  } finally {
    btn.disabled = false;
    btn.textContent = '🧠 Run PI-SBI Posterior';
  }
}

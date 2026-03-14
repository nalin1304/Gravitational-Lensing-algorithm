const L = () => window.LensPINN;

export function render() {
    return `
    <div class="page-content">
      <!-- Page Header -->
      <div style="margin-bottom:24px">
        <h2 style="font-size:1.5rem;font-weight:700;color:var(--text-primary);margin:0 0 6px 0">Next-Gen Statistical Rigor</h2>
        <p class="section-desc">Stage IV Survey modules: SBI NPE, Starlets, SED Multiband, Environmental Linker, Scientific Gate.</p>
      </div>

      <!-- Tab Navigation -->
      <div class="tabs mb-20" id="rigorTabs">
        <button class="tab active" data-rtab="sbi">📊 Likelihood-Free (SBI)</button>
        <button class="tab" data-rtab="starlet">✨ Starlets</button>
        <button class="tab" data-rtab="sed">🌈 SED Multiband</button>
        <button class="tab" data-rtab="env">🌌 Env Linker</button>
        <button class="tab" data-rtab="gate">🛡 Gate Validator</button>
      </div>

      <!-- SBI Panel -->
      <div id="rtab-sbi" class="rigor-panel">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Neural Posterior Estimation (NPE)</span>
            <span class="badge badge-purple">JAX</span>
          </div>
          <p class="section-desc mb-16">Run JAX/Haiku distrax normalizing flow inference on convergence maps from Workbench.</p>
          <button id="sbiRunBtn" class="btn btn-primary">Run SBI Inference</button>
          <div id="sbiResults" class="metric-grid" style="margin-top:16px"></div>
        </div>
      </div>

      <!-- Starlet Panel -->
      <div id="rtab-starlet" class="rigor-panel" style="display:none">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Starlet FISTA Reconstruction</span>
            <span class="badge badge-info">Sparse</span>
          </div>
          <p class="section-desc mb-16">Sparse morphological reconstruction using starlet wavelet transform with FISTA optimization.</p>
          <button id="starletRunBtn" class="btn btn-primary">Run Sparse Morphological Recon</button>
          <div id="starletResults" class="metric-grid" style="margin-top:16px"></div>
        </div>
      </div>

      <!-- SED Panel -->
      <div id="rtab-sed" class="rigor-panel" style="display:none">
        <div class="card">
          <div class="card-header">
            <span class="card-title">SED-Locked Morphologies</span>
            <span class="badge badge-success">Multiband</span>
          </div>
          <p class="section-desc mb-16">Solve linear ML SED amplitudes from observed fluxes (nanomaggies).</p>
          
          <div class="grid-3 gap-12 mb-16">
            <div class="form-group">
              <label class="form-label">g-band Flux (nMgy)</label>
              <input id="sedFluxG" type="number" value="12.0" step="0.1" class="form-input" placeholder="12.0" />
            </div>
            <div class="form-group">
              <label class="form-label">r-band Flux (nMgy)</label>
              <input id="sedFluxR" type="number" value="18.0" step="0.1" class="form-input" placeholder="18.0" />
            </div>
            <div class="form-group">
              <label class="form-label">i-band Flux (nMgy)</label>
              <input id="sedFluxI" type="number" value="22.0" step="0.1" class="form-input" placeholder="22.0" />
            </div>
          </div>
          
          <button id="sedRunBtn" class="btn btn-primary">Run Linear ML SED Amplitudes</button>
          <div id="sedResults" class="metric-grid" style="margin-top:16px"></div>
        </div>
      </div>

      <!-- Env Panel -->
      <div id="rtab-env" class="rigor-panel" style="display:none">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Environmental κ<sub>ext</sub> Linker</span>
            <span class="badge badge-warning">LOS</span>
          </div>
          <p class="section-desc mb-16">Compute external convergence from line-of-sight galaxy catalog.</p>
          
          <div class="grid-3 gap-12 mb-16">
            <div class="form-group">
              <label class="form-label">N Galaxies</label>
              <input id="envNgal" type="number" value="150" class="form-input" placeholder="150" />
            </div>
            <div class="form-group">
              <label class="form-label">z<sub>lens</sub></label>
              <input id="envZlens" type="number" value="0.4" step="0.01" class="form-input" placeholder="0.4" />
            </div>
            <div class="form-group">
              <label class="form-label">z<sub>source</sub></label>
              <input id="envZsource" type="number" value="1.2" step="0.01" class="form-input" placeholder="1.2" />
            </div>
          </div>
          
          <button id="envRunBtn" class="btn btn-primary">Compute κ<sub>ext</sub> from External Catalog</button>
          <div id="envResults" class="metric-grid" style="margin-top:16px"></div>
        </div>
      </div>

      <!-- Sci Gate Panel -->
      <div id="rtab-gate" class="rigor-panel" style="display:none">
        <div class="card">
          <div class="card-header">
            <span class="card-title">Scientific Consistency Gate</span>
            <span class="badge badge-danger">Validator</span>
          </div>
          <p class="section-desc mb-16">Validate lens model parameters against M/L scaling relations and physical bounds.</p>
          
          <div class="grid-3 gap-12 mb-16">
            <div class="form-group">
              <label class="form-label">M<sub>vir</sub> (M☉)</label>
              <input id="gateMvir" type="number" value="1e13" class="form-input" placeholder="1e13" />
            </div>
            <div class="form-group">
              <label class="form-label">r<sub>s</sub> (kpc)</label>
              <input id="gateRs" type="number" value="150" class="form-input" placeholder="150" />
            </div>
            <div class="form-group">
              <label class="form-label">Ellipticity ε</label>
              <input id="gateEll" type="number" value="0.1" class="form-input" placeholder="0.1" />
            </div>
          </div>
          
          <button id="gateRunBtn" class="btn btn-primary">Validate Against M/L Laws</button>
          <div id="gateResults" class="metric-grid" style="margin-top:16px"></div>
        </div>
      </div>
    </div>
  `;
}

export async function init() {
    const P = L();

    // Tabs
    document.querySelectorAll("#rigorTabs .tab").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll("#rigorTabs .tab").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".rigor-panel").forEach(p => p.style.display = "none");
            btn.classList.add("active");
            document.getElementById(`rtab-${btn.dataset.rtab}`).style.display = "";
        });
    });

    // SBI NPE
    document.getElementById("sbiRunBtn").addEventListener("click", async () => {
        const storedMap = sessionStorage.getItem('lastConvergenceMap');
        if (!storedMap) {
            P.toast("Generate a convergence map in the Workbench first", "error");
            document.getElementById("sbiResults").innerHTML = `<p class="section-desc text-warning">⚠️ No convergence map available. Go to <strong>Workbench → Generate</strong> to create one, then return here.</p>`;
            return;
        }
        document.getElementById("sbiResults").innerHTML = `<p class="section-desc">Running NPE flow on workbench convergence map...</p>`;
        try {
            const arr = JSON.parse(storedMap);
            const resp = await P.api("/api/v1/rigor/sbi", { method: "POST", body: { convergence_map: arr, n_samples: 500 }, auth: false });
            document.getElementById("sbiResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">M<sub>vir</sub> Median</span><span class="metric-val">${resp.M_vir.median.toExponential(3)}</span></div>
        <div class="metric-row"><span class="metric-key">Concentration Median</span><span class="metric-val">${resp.ratio.median.toFixed(3)}</span></div>
        <div class="metric-row"><span class="metric-key">M/L Median</span><span class="metric-val">${resp.eff.median.toFixed(3)}</span></div>
      `;
            P.toast("NPE Success", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // Starlets
    document.getElementById("starletRunBtn").addEventListener("click", async () => {
        const storedMap = sessionStorage.getItem('lastConvergenceMap');
        if (!storedMap) {
            P.toast("Generate a convergence map in the Workbench first", "error");
            document.getElementById("starletResults").innerHTML = `<p class="section-desc text-warning">⚠️ No convergence map available. Go to <strong>Workbench → Generate</strong> to create one, then return here.</p>`;
            return;
        }
        document.getElementById("starletResults").innerHTML = `<p class="section-desc">Running FISTA on workbench convergence map...</p>`;
        try {
            const arr = JSON.parse(storedMap);
            const resp = await P.api("/api/v1/rigor/starlet", { method: "POST", body: { image: arr, n_scales: 3, lambda_reg: 0.05, max_iter: 5 }, auth: false });
            document.getElementById("starletResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">Sparsity Fraction</span><span class="metric-val badge badge-success">${(resp.sparsity_fraction * 100).toFixed(1)}%</span></div>
        <div style="margin-top:12px">
          <img src="data:image/png;base64,${resp.image_b64}" style="width:100%;max-width:300px;border-radius:var(--radius-md);border:1px solid var(--border)" />
        </div>
      `;
            P.toast("Starlets run complete", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // SED
    document.getElementById("sedRunBtn").addEventListener("click", async () => {
        const flux_g = parseFloat(document.getElementById("sedFluxG").value);
        const flux_r = parseFloat(document.getElementById("sedFluxR").value);
        const flux_i = parseFloat(document.getElementById("sedFluxI").value);
        document.getElementById("sedResults").innerHTML = `<p class="section-desc">Solving linear ML estimator...</p>`;
        try {
            const resp = await P.api("/api/v1/rigor/sed", { method: "POST", body: { flux_g, flux_r, flux_i }, auth: false });
            document.getElementById("sedResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">Bands Solved</span><span class="metric-val badge badge-success">${resp.amplitudes.length}</span></div>
        <div class="metric-row"><span class="metric-key">Method</span><span class="metric-val">Linear Maximum Likelihood</span></div>
      `;
            P.toast("SED Amplitudes found", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // Env
    document.getElementById("envRunBtn").addEventListener("click", async () => {
        const n_galaxies = parseInt(document.getElementById("envNgal").value);
        const z_lens = parseFloat(document.getElementById("envZlens").value);
        const z_source = parseFloat(document.getElementById("envZsource").value);
        document.getElementById("envResults").innerHTML = `<p class="section-desc">Processing FOV catalog...</p>`;
        try {
            const resp = await P.api("/api/v1/rigor/env_linker", { method: "POST", body: { n_galaxies, z_lens, z_source }, auth: false });
            document.getElementById("envResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">κ<sub>ext</sub> Total</span><span class="metric-val badge badge-info">${resp.kappa_ext.toFixed(4)}</span></div>
        <div class="metric-row"><span class="metric-key">Galaxies Included</span><span class="metric-val">${resp.n_included} (z &lt; ${z_source})</span></div>
      `;
            P.toast("Convergence sum complete", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // Gate
    document.getElementById("gateRunBtn").addEventListener("click", async () => {
        const M_vir = parseFloat(document.getElementById("gateMvir").value);
        const r_s = parseFloat(document.getElementById("gateRs").value);
        const ellipticity = parseFloat(document.getElementById("gateEll").value);
        document.getElementById("gateResults").innerHTML = `<p class="section-desc">Checking physics bounds...</p>`;
        try {
            const resp = await P.api("/api/v1/rigor/consistency_gate", { method: "POST", body: { M_vir, r_s, ellipticity }, auth: false });
            const badgeClass = resp.is_valid ? "badge-success" : "badge-danger";
            document.getElementById("gateResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">Validation Status</span><span class="metric-val"><span class="badge ${badgeClass}">${resp.is_valid ? '✓ Valid' : '✗ Rejected'}</span></span></div>
        <div class="metric-row"><span class="metric-key">Reason</span><span class="metric-val">${P.esc(resp.reason)}</span></div>
        <div class="metric-row"><span class="metric-key">M/L Ratio</span><span class="metric-val">${resp.M_L_ratio.toFixed(2)} M<sub>☉</sub> / L<sub>☉</sub></span></div>
      `;
            P.toast(resp.is_valid ? "Valid Model!" : "Rejected Model!", resp.is_valid ? "success" : "error");
        } catch (e) { P.toast(e.message, "error"); }
    });
}

const L = () => window.LensPINN;

export function render() {
    return `
    <div style="margin-bottom:24px">
      <h2 style="font-size:1.5rem;font-weight:700;color:var(--text-primary);margin:0 0 6px 0">Next-Gen Rigor</h2>
      <p class="section-desc">Stage IV Survey modules: SBI NPE, Starlets, SED Multiband, Env Linker, Sci Gate.</p>
    </div>

    <!-- Tab bar -->
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
        <div class="card-header"><span class="card-title">Neural Posterior Estimation (NPE)</span></div>
        <p class="section-desc">Run JAX/Haiku distrax normalizing flow inference.</p>
        <button id="sbiRunBtn" class="btn btn-primary">Run SBI Inference</button>
        <div id="sbiResults" style="margin-top:12px"></div>
      </div>
    </div>

    <!-- Starlet Panel -->
    <div id="rtab-starlet" class="rigor-panel" style="display:none">
      <div class="card">
        <div class="card-header"><span class="card-title">Starlet FISTA Reconstruction</span></div>
        <button id="starletRunBtn" class="btn btn-primary">Run Sparse Morphological Recon</button>
        <div id="starletResults" style="margin-top:12px"></div>
      </div>
    </div>

    <!-- SED Panel -->
    <div id="rtab-sed" class="rigor-panel" style="display:none">
      <div class="card">
        <div class="card-header"><span class="card-title">SED-Locked Morphologies</span></div>
        <button id="sedRunBtn" class="btn btn-primary">Run Linear ML SED Amplitudes</button>
        <div id="sedResults" style="margin-top:12px"></div>
      </div>
    </div>

    <!-- Env Panel -->
    <div id="rtab-env" class="rigor-panel" style="display:none">
      <div class="card">
        <div class="card-header"><span class="card-title">Environmental κ_ext Linker</span></div>
        <button id="envRunBtn" class="btn btn-primary">Compute κ_ext from external catalog</button>
        <div id="envResults" style="margin-top:12px"></div>
      </div>
    </div>

    <!-- Sci Gate Panel -->
    <div id="rtab-gate" class="rigor-panel" style="display:none">
      <div class="card">
        <div class="card-header"><span class="card-title">Scientific Consistency Gate</span></div>
        <div class="grid-3 gap-12" style="margin-bottom:12px">
            <input id="gateMvir" type="number" value="1e13" class="form-input" placeholder="M_vir" />
            <input id="gateRs" type="number" value="150" class="form-input" placeholder="r_s" />
            <input id="gateEll" type="number" value="0.1" class="form-input" placeholder="ellipticity" />
        </div>
        <button id="gateRunBtn" class="btn btn-primary">Validate against M/L laws</button>
        <div id="gateResults" style="margin-top:12px"></div>
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
        document.getElementById("sbiResults").innerHTML = "Running NPE flow...";
        try {
            // Try to get real convergence data from session, fall back to synthetic demo map
            const storedMap = sessionStorage.getItem('lastConvergenceMap');
            let arr;
            if (storedMap) {
                arr = JSON.parse(storedMap);
            } else {
                // Generate a simple NFW-like convergence map for demo
                arr = Array.from({length: 10}, (_, i) =>
                    Array.from({length: 10}, (_, j) => {
                        const r = Math.sqrt((i-4.5)**2 + (j-4.5)**2) + 0.1;
                        return Math.max(0, 0.5 / (r * (1 + r)**2));  // NFW-like profile
                    })
                );
            }
            const resp = await P.api("/api/v1/rigor/sbi", { method: "POST", body: { convergence_map: arr, n_samples: 500 }, auth: false });
            document.getElementById("sbiResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">M_vir Median</span><span class="metric-val">${resp.M_vir.median.toExponential(3)}</span></div>
        <div class="metric-row"><span class="metric-key">c Median</span><span class="metric-val">${resp.ratio.median.toFixed(3)}</span></div>
        <div class="metric-row"><span class="metric-key">M/L Median</span><span class="metric-val">${resp.eff.median.toFixed(3)}</span></div>
      `;
            P.toast("NPE Success", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // Starlets
    document.getElementById("starletRunBtn").addEventListener("click", async () => {
        document.getElementById("starletResults").innerHTML = "Running FISTA...";
        try {
            const arr = Array(32).fill().map(() => Array(32).fill(1.0));
            const resp = await P.api("/api/v1/rigor/starlet", { method: "POST", body: { image: arr, n_scales: 3, lambda_reg: 0.05, max_iter: 5 }, auth: false });
            document.getElementById("starletResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">Sparsity Fraction</span><span class="metric-val">${(resp.sparsity_fraction * 100).toFixed(1)}%</span></div>
        <img src="data:image/png;base64,${resp.image_b64}" style="width:100px;margin-top:10px;border-radius:4px" />
      `;
            P.toast("Starlets run complete", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // SED
    document.getElementById("sedRunBtn").addEventListener("click", async () => {
        document.getElementById("sedResults").innerHTML = "Solving linear ML estimator...";
        try {
            // Demo values: typical lensed galaxy SED (AB magnitudes → linear fluxes)
            // flux in nanomaggies: g=12, r=18, i=22 (typical lensed system at z~0.5)
            const resp = await P.api("/api/v1/rigor/sed", { method: "POST", body: { flux_g: 12.0, flux_r: 18.0, flux_i: 22.0 }, auth: false });
            document.getElementById("sedResults").innerHTML = `<div class="metric-row"><span class="metric-key">Amplitudes solved</span><span class="metric-val">${resp.amplitudes.length} bands</span></div>`;
            P.toast("SED Amplitudes found", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // Env
    document.getElementById("envRunBtn").addEventListener("click", async () => {
        document.getElementById("envResults").innerHTML = "Processing FOV catalog...";
        try {
            const resp = await P.api("/api/v1/rigor/env_linker", { method: "POST", body: { n_galaxies: 150, z_lens: 0.4, z_source: 1.2 }, auth: false });
            document.getElementById("envResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">κ_ext Total</span><span class="metric-val">${resp.kappa_ext.toFixed(4)}</span></div>
        <div class="metric-row"><span class="metric-key">Galaxies included (z < 1.2)</span><span class="metric-val">${resp.n_included}</span></div>
      `;
            P.toast("Convergence sum complete", "success");
        } catch (e) { P.toast(e.message, "error"); }
    });

    // Gate
    document.getElementById("gateRunBtn").addEventListener("click", async () => {
        const M_vir = parseFloat(document.getElementById("gateMvir").value);
        const r_s = parseFloat(document.getElementById("gateRs").value);
        const ellipticity = parseFloat(document.getElementById("gateEll").value);
        document.getElementById("gateResults").innerHTML = "Checking physics bounds...";
        try {
            const resp = await P.api("/api/v1/rigor/consistency_gate", { method: "POST", body: { M_vir, r_s, ellipticity }, auth: false });
            const col = resp.is_valid ? "var(--success)" : "var(--danger)";
            document.getElementById("gateResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">Validity</span><span class="metric-val" style="color:${col}">${resp.reason}</span></div>
        <div class="metric-row"><span class="metric-key">M/L Ratio</span><span class="metric-val">${resp.M_L_ratio.toFixed(2)} M_sun / L_sun</span></div>
      `;
            P.toast(resp.is_valid ? "Valid Model!" : "Rejected Model!", resp.is_valid ? "success" : "error");
        } catch (e) { P.toast(e.message, "error"); }
    });
}

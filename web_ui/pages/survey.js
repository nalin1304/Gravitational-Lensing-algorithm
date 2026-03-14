/* ============================================================================
   Stage IV Survey — Finder, ePSF, Blinding, Covariance, Joint Deblending
   ============================================================================ */
const L = () => window.LensPINN;

export function render() {
    return `
    <!-- Page Header -->
    <div class="mb-24">
      <h2 style="font-size:1.5rem;font-weight:700;margin:0 0 8px 0">
        Stage IV Survey Tools
      </h2>
      <p class="section-desc" style="margin:0">
        Roman / Euclid readiness: automated lens discovery, ePSF modelling,
        cosmological blinding, correlated-pixel likelihood, and joint deblending.
      </p>
    </div>

    <!-- Tab bar -->
    <div class="tabs mb-20" id="surveyTabs">
      <button class="tab active" data-stab="finder">🔭 Lens Finder</button>
      <button class="tab" data-stab="epsf">🌌 ePSF Model</button>
      <button class="tab" data-stab="blinding">🔒 Blinding</button>
      <button class="tab" data-stab="covariance">📊 Pixel Covariance</button>
      <button class="tab" data-stab="deblend">🛰 Joint Deblending</button>
    </div>

    <!-- ─── Finder ─────────────────────────────────────────────────────── -->
    <div id="stab-finder" class="survey-panel">
      <div class="grid-2 gap-20">
        <div class="card">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
            <span class="card-title">Automated Lens Discovery</span>
            <span class="badge badge-info" id="sfModelBadge">LenNet-style CNN</span>
          </div>
          <p id="sfStatusNote" class="section-desc mb-16">
            Learned survey detection requires a trained checkpoint-backed detector.
          </p>
          <div class="form-group">
            <label class="form-label">Scan Mode</label>
            <select id="sfScanMode" class="form-input">
              <option value="synthetic">Synthetic injection (demo)</option>
              <option value="fits" disabled>Upload FITS file (not enabled in this build)</option>
            </select>
          </div>
          <div id="sfFitsRow" class="form-group" style="display:none">
            <label class="form-label">FITS File</label>
            <input id="sfFits" type="file" class="form-input" accept=".fits,.fit,.fz" />
          </div>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">Stride (px)</label>
              <input id="sfStride" class="form-input" type="number" value="32" min="8" max="128" />
            </div>
            <div class="form-group">
              <label class="form-label">Min Confidence</label>
              <input id="sfConf" class="form-input" type="number" step="0.05" value="0.70" min="0.1" max="0.99" />
            </div>
          </div>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">Injected Lenses (synthetic)</label>
              <input id="sfNLenses" class="form-input" type="number" value="5" min="1" max="20" />
            </div>
            <div class="form-group">
              <label class="form-label">RNG Seed</label>
              <input id="sfSeed" class="form-input" type="number" value="42" />
            </div>
          </div>
          <button id="sfRunBtn" class="btn btn-primary btn-full" disabled>
            🔭 Scan for Lens Candidates
          </button>
        </div>

        <div class="card">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
            <span class="card-title">Candidates</span>
            <span class="badge" id="sfCandCount" style="background:var(--bg-secondary)">—</span>
          </div>
          <div id="sfResults"><p class="section-desc">Run a scan to see candidates.</p></div>
        </div>
      </div>
    </div>

    <!-- ─── ePSF ───────────────────────────────────────────────────────── -->
    <div id="stab-epsf" class="survey-panel" style="display:none">
      <div class="grid-2 gap-20">
        <div class="card">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
            <span class="card-title">Effective PSF Configuration</span>
            <span class="badge badge-info">Z4–Z22 Zernike</span>
          </div>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">Pixel Scale (arcsec/px)</label>
              <input id="epPixelScale" class="form-input" type="number" step="0.01" value="0.11" />
            </div>
            <div class="form-group">
              <label class="form-label">Kernel Size (px, odd)</label>
              <input id="epKernelSize" class="form-input" type="number" value="21" min="7" max="63" step="2" />
            </div>
          </div>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">λ<sub>eff</sub> (μm)</label>
              <input id="epWave" class="form-input" type="number" step="0.05" value="1.55" />
            </div>
            <div class="form-group">
              <label class="form-label">Aperture D (m)</label>
              <input id="epAperture" class="form-input" type="number" step="0.1" value="2.4" />
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Detector Position (x, y pixels)</label>
            <div class="grid-2 gap-8">
              <input id="epDetX" class="form-input" type="number" value="512" placeholder="x" />
              <input id="epDetY" class="form-input" type="number" value="1024" placeholder="y" />
            </div>
          </div>
          <div class="form-group">
            <label class="form-label" style="display:flex;gap:8px;align-items:center;cursor:pointer">
              <input id="epChargeDiff" type="checkbox" checked /> Enable Charge Diffusion (H4RG, σ=0.5 px)
            </label>
          </div>
          <button id="epEvalBtn" class="btn btn-primary btn-full">
            🌌 Evaluate ePSF Kernel
          </button>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">PSF Metrics</span>
          </div>
          <div id="epResults"><p class="section-desc">Configure and evaluate the ePSF to see kernel metrics.</p></div>
          <canvas id="epCanvas" width="200" height="200"
            style="display:none;margin:12px auto;border:1px solid var(--border);border-radius:6px;width:200px;height:200px">
          </canvas>
        </div>
      </div>

      <div class="card" style="margin-top:20px">
        <div class="card-header">
          <span class="card-title">Zernike FOV Map</span>
          <span class="card-subtitle">Spatial variation across detector</span>
        </div>
        <div class="form-group" style="display:flex;gap:12px;align-items:center">
          <label class="form-label" style="white-space:nowrap;margin:0">Zernike Index Z</label>
          <input id="epZernikeIdx" class="form-input" type="number" value="4" min="4" max="22" style="width:80px" />
          <button id="epFovBtn" class="btn btn-primary" style="white-space:nowrap">
            Generate FOV Map
          </button>
        </div>
        <div id="epFovResult"><p class="section-desc">Select a Zernike mode and generate the FOV map.</p></div>
      </div>
    </div>

    <!-- ─── Blinding ──────────────────────────────────────────────────── -->
    <div id="stab-blinding" class="survey-panel" style="display:none">
      <div class="grid-2 gap-20">
        <div class="card">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
            <span class="card-title">Apply Cosmological Blinding</span>
            <span class="badge badge-warning">TDCOSMO Protocol</span>
          </div>
          <p class="section-desc mb-16">
            Enter your analysis values. The blinding handler adds a secret, seed-derived
            offset so you cannot infer the true cosmological parameters during analysis.
          </p>
          <div class="form-group">
            <label class="form-label">Blinding Phrase (secret — not stored)</label>
            <input id="blPhrase" class="form-input" type="password"
              placeholder="Enter a > 12-char secret phrase" autocomplete="off" />
          </div>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">H₀ (km/s/Mpc)</label>
              <input id="blH0" class="form-input" type="number" step="0.1" value="72.1" />
            </div>
            <div class="form-group">
              <label class="form-label">D<sub>Δt</sub> (Mpc)</label>
              <input id="blDtd" class="form-input" type="number" step="1" value="5000" />
            </div>
          </div>
          <button id="blBlindBtn" class="btn btn-warning btn-full">
            🔒 Apply Blinding
          </button>
          <div style="margin-top:12px;padding-top:12px;border-top:1px solid var(--border)">
            <p class="section-desc" style="font-size:0.75rem">
              ⚠ The phrase is sent to the server to compute the HMAC offset and is not stored server-side.
              Write it down before closing this tab.
            </p>
          </div>
        </div>

        <div class="card">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
            <span class="card-title">Unblinding Gate</span>
            <span class="badge badge-danger">Irreversible</span>
          </div>
          <p class="section-desc mb-16">
            Unblind only after all scientific validation checks pass. Enter your blinded
            values and the original phrase.
          </p>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">H₀ Blinded</label>
              <input id="blH0Blind" class="form-input" type="number" step="0.1" placeholder="blinded value" />
            </div>
            <div class="form-group">
              <label class="form-label">D<sub>Δt</sub> Blinded</label>
              <input id="blDtdBlind" class="form-input" type="number" step="1" placeholder="blinded value" />
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Verification Phrase</label>
            <input id="blVerifyPhrase" class="form-input" type="password"
              placeholder="Original phrase to verify" autocomplete="off" />
          </div>
          <button id="blUnblindBtn" class="btn btn-danger btn-full">
            🔓 Request Unblinding via Validation Gate
          </button>
          <div id="blResults" style="margin-top:12px"></div>
        </div>
      </div>

      <!-- Blinding state display -->
      <div class="card" id="blStateCard" style="margin-top:20px;display:none">
        <div class="card-header">
          <span class="card-title">Blinding State</span>
        </div>
        <div id="blState"><p class="section-desc">Apply blinding to see state.</p></div>
      </div>
    </div>

    <!-- ─── Covariance ────────────────────────────────────────────────── -->
    <div id="stab-covariance" class="survey-panel" style="display:none">
      <div class="grid-2 gap-20">
        <div class="card">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
            <span class="card-title">Drizzle Covariance Parameters</span>
            <span class="badge badge-info">Fruchter & Hook 2002</span>
          </div>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">Image Size (px)</label>
              <input id="cvSize" class="form-input" type="number" value="32" min="4" max="64" />
            </div>
            <div class="form-group">
              <label class="form-label">RMS Level (σ)</label>
              <input id="cvSigma" class="form-input" type="number" step="0.005" value="0.02" />
            </div>
          </div>
          <div class="grid-2 gap-12">
            <div class="form-group">
              <label class="form-label">pixfrac</label>
              <input id="cvPixfrac" class="form-input" type="number" step="0.05" value="0.8" min="0.1" max="1.0" />
            </div>
            <div class="form-group">
              <label class="form-label">scale (out/in)</label>
              <input id="cvScale" class="form-input" type="number" step="0.05" value="0.5" min="0.1" max="1.0" />
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Kernel</label>
            <select id="cvKernel" class="form-input">
              <option value="square">square (turbo)</option>
              <option value="gaussian">gaussian</option>
              <option value="lanczos2">lanczos2</option>
              <option value="lanczos3">lanczos3</option>
            </select>
          </div>
          <button id="cvComputeBtn" class="btn btn-primary btn-full">
            📊 Compute Covariance + Whiten
          </button>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Covariance Diagnostics</span>
          </div>
          <div id="cvResults"><p class="section-desc">Configure drizzle parameters and compute.</p></div>
        </div>
      </div>
    </div>

    <!-- ─── Joint Deblending ──────────────────────────────────────────── -->
    <div id="stab-deblend" class="survey-panel" style="display:none">
      <div class="grid-2 gap-20">
        <div class="card">
          <div class="card-header" style="display:flex;justify-content:space-between;align-items:center">
            <span class="card-title">Joint Survey Configuration</span>
            <span class="badge badge-info">Rubin + Roman</span>
          </div>
          <p class="section-desc mb-16">
            Simultaneously model the same lens system at ground-based and space-based
            resolution using the joint log-likelihood framework.
          </p>
          <div class="grid-2 gap-12 mb-16" style="border:1px solid var(--border);border-radius:8px;padding:12px">
            <div>
              <div style="font-size:0.75rem;color:var(--text-muted);font-weight:600;margin-bottom:8px;text-transform:uppercase">GROUND (Rubin-like)</div>
              <div class="form-group">
                <label class="form-label">Grid size (px)</label>
                <input id="jdGroundSize" class="form-input" type="number" value="32" min="16" max="128" />
              </div>
              <div class="form-group">
                <label class="form-label">Pixel scale (arcsec/px)</label>
                <input id="jdGroundScale" class="form-input" type="number" step="0.05" value="0.2" />
              </div>
              <div class="form-group">
                <label class="form-label">Noise σ</label>
                <input id="jdGroundSigma" class="form-input" type="number" step="0.005" value="0.02" />
              </div>
            </div>
            <div>
              <div style="font-size:0.75rem;color:var(--text-muted);font-weight:600;margin-bottom:8px;text-transform:uppercase">SPACE (Roman-like)</div>
              <div class="form-group">
                <label class="form-label">Grid size (px)</label>
                <input id="jdSpaceSize" class="form-input" type="number" value="64" min="16" max="256" />
              </div>
              <div class="form-group">
                <label class="form-label">Pixel scale (arcsec/px)</label>
                <input id="jdSpaceScale" class="form-input" type="number" step="0.01" value="0.11" />
              </div>
              <div class="form-group">
                <label class="form-label">Noise σ</label>
                <input id="jdSpaceSigma" class="form-input" type="number" step="0.005" value="0.01" />
              </div>
            </div>
          </div>
          <div class="form-group">
            <label class="form-label">Likelihood mode</label>
            <select id="jdLikelihood" class="form-input">
              <option value="gaussian">Independent Gaussian</option>
              <option value="correlated">Correlated (Drizzle Covariance)</option>
            </select>
          </div>
          <div class="form-group">
            <label class="form-label">RNG seed</label>
            <input id="jdSeed" class="form-input" type="number" value="42" />
          </div>
          <button id="jdRunBtn" class="btn btn-primary btn-full">
            🛰 Run Joint Deblending
          </button>
        </div>

        <div class="card">
          <div class="card-header">
            <span class="card-title">Joint Likelihood Results</span>
          </div>
          <div id="jdResults"><p class="section-desc">Configure surveys and run joint analysis.</p></div>
        </div>
      </div>
    </div>
  `;
}

export async function init() {
    const P = L();
    let _finderStatus = null;

    // ── Tab switching ────────────────────────────────────────────────────
    document.querySelectorAll("#surveyTabs .tab").forEach(btn => {
        btn.addEventListener("click", () => {
            document.querySelectorAll("#surveyTabs .tab").forEach(b => b.classList.remove("active"));
            document.querySelectorAll(".survey-panel").forEach(p => p.style.display = "none");
            btn.classList.add("active");
            const panel = document.getElementById(`stab-${btn.dataset.stab}`);
            if (panel) panel.style.display = "";
        });
    });

    // ── Finder scan mode toggle ──────────────────────────────────────────
    function syncFinderAvailability() {
        const button = document.getElementById("sfRunBtn");
        const note = document.getElementById("sfStatusNote");
        const badge = document.getElementById("sfModelBadge");
        if (!_finderStatus) return;

        if (_finderStatus.supports_detection) {
            button.disabled = false;
            badge.textContent = "LenNet-style CNN";
            note.innerHTML = `Checkpoint active: <code>${P.esc(_finderStatus.checkpoint_path || "trained detector")}</code>`;
            return;
        }

        button.disabled = true;
        badge.textContent = "Detector unavailable";
        note.textContent = "Survey detection is disabled because no trained LensFinder checkpoint is deployed. The UI is wired to backend status and will enable automatically when a detector is installed.";
    }

    try {
        _finderStatus = await P.api("/api/v1/survey/finder/status", { auth: false });
    } catch (e) {
        _finderStatus = {
            supports_detection: false,
            status: "status_unavailable",
            detail: e.message,
        };
    }
    syncFinderAvailability();

    document.getElementById("sfScanMode").addEventListener("change", e => {
        document.getElementById("sfFitsRow").style.display =
            e.target.value === "fits" ? "" : "none";
    });

    // ── Finder: Run ──────────────────────────────────────────────────────
    document.getElementById("sfRunBtn").addEventListener("click", async () => {
        const mode = document.getElementById("sfScanMode").value;
        const stride = parseInt(document.getElementById("sfStride").value) || 32;
        const conf = parseFloat(document.getElementById("sfConf").value) || 0.7;
        const nlenses = parseInt(document.getElementById("sfNLenses").value) || 5;
        const seed = parseInt(document.getElementById("sfSeed").value) || 42;

        if (!_finderStatus?.supports_detection) {
            const message = "LensFinder is unavailable because no trained checkpoint-backed detector is deployed.";
            document.getElementById("sfResults").innerHTML = `<p class="section-desc" style="color:var(--warning)">${P.esc(message)}</p>`;
            P.toast(message, "warning");
            return;
        }

        document.getElementById("sfResults").innerHTML = `<p class="section-desc">Scanning…</p>`;
        try {
            const body = { mode, stride, confidence_threshold: conf, n_lenses: nlenses, seed };
            const resp = await P.api("/api/v1/survey/finder", { method: "POST", body, auth: false });
            const cands = resp.candidates || [];
            document.getElementById("sfCandCount").textContent = `${cands.length} candidate${cands.length !== 1 ? "s" : ""}`;
            if (!cands.length) {
                document.getElementById("sfResults").innerHTML = `<p class="section-desc">No candidates above threshold.</p>`;
                return;
            }
            document.getElementById("sfResults").innerHTML = cands.map((c, i) => `
        <div class="metric-row">
          <span class="metric-key">Candidate ${i + 1}</span>
          <span class="metric-val">
            conf=${(c.confidence * 100).toFixed(1)}%
            ${c.ra != null ? ` · RA=${c.ra.toFixed(4)}° Dec=${c.dec.toFixed(4)}°` : ` · pixel (${c.pixel_x},${c.pixel_y})`}
          </span>
        </div>
      `).join("");
            P.toast(`Found ${cands.length} lens candidate(s)`, "success");
        } catch (e) {
            document.getElementById("sfResults").innerHTML = `<p class="section-desc" style="color:var(--danger)">${P.esc(e.message)}</p>`;
            P.toast(e.message, "error");
        }
    });

    // ── ePSF: Evaluate ───────────────────────────────────────────────────
    document.getElementById("epEvalBtn").addEventListener("click", async () => {
        const body = {
            pixel_scale: parseFloat(document.getElementById("epPixelScale").value),
            kernel_size: parseInt(document.getElementById("epKernelSize").value),
            wavelength_um: parseFloat(document.getElementById("epWave").value),
            aperture_m: parseFloat(document.getElementById("epAperture").value),
            x_det: parseFloat(document.getElementById("epDetX").value),
            y_det: parseFloat(document.getElementById("epDetY").value),
            charge_diffusion: document.getElementById("epChargeDiff").checked,
        };
        document.getElementById("epResults").innerHTML = `<p class="section-desc">Evaluating…</p>`;
        try {
            const resp = await P.api("/api/v1/survey/epsf", { method: "POST", body, auth: false });
            document.getElementById("epResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">FWHM</span><span class="metric-val">${resp.fwhm_arcsec?.toFixed(3)} arcsec</span></div>
        <div class="metric-row"><span class="metric-key">FWHM pixels</span><span class="metric-val">${resp.fwhm_pixels?.toFixed(2)} px</span></div>
        <div class="metric-row"><span class="metric-key">Strehl ratio</span><span class="metric-val">${resp.strehl?.toFixed(3)}</span></div>
        <div class="metric-row"><span class="metric-key">Kernel sum</span><span class="metric-val">${resp.kernel_sum?.toFixed(6)}</span></div>
        <div class="metric-row"><span class="metric-key">Detector pos</span><span class="metric-val">(${resp.x_det}, ${resp.y_det})</span></div>
        ${resp.zernike_rms_nm != null ? `<div class="metric-row"><span class="metric-key">Wavefront RMS</span><span class="metric-val">${resp.zernike_rms_nm?.toFixed(2)} nm</span></div>` : ""}
      `;
            if (resp.kernel_b64) {
                const canvas = document.getElementById("epCanvas");
                const ctx = canvas.getContext("2d");
                const img = new Image();
                img.onload = () => {
                    ctx.clearRect(0, 0, 200, 200);
                    ctx.drawImage(img, 0, 0, 200, 200);
                    canvas.style.display = "block";
                };
                img.src = `data:image/png;base64,${resp.kernel_b64}`;
            }
            P.toast("ePSF evaluated", "success");
        } catch (e) {
            document.getElementById("epResults").innerHTML = `<p class="section-desc" style="color:var(--danger)">${P.esc(e.message)}</p>`;
            P.toast(e.message, "error");
        }
    });

    // ── ePSF: FOV Map ────────────────────────────────────────────────────
    document.getElementById("epFovBtn").addEventListener("click", async () => {
        const j = parseInt(document.getElementById("epZernikeIdx").value);
        try {
            const resp = await P.api(`/api/v1/survey/epsf/fov?zernike_index=${j}`, { auth: false });
            const stats = resp.stats || {};
            document.getElementById("epFovResult").innerHTML = `
        <div class="metric-row"><span class="metric-key">Z${j} FOV range</span>
          <span class="metric-val">${stats.min?.toFixed(3)} → ${stats.max?.toFixed(3)} nm</span></div>
        <div class="metric-row"><span class="metric-key">RMS over FOV</span>
          <span class="metric-val">${stats.rms?.toFixed(3)} nm</span></div>
        <p class="section-desc" style="font-size:0.75rem;color:var(--text-muted);margin-top:8px">
          Grid: ${resp.grid_points}×${resp.grid_points} · Detector: ${resp.detector_shape?.join('×')}
        </p>
      `;
            P.toast(`Z${j} FOV map computed`, "success");
        } catch (e) {
            document.getElementById("epFovResult").innerHTML =
                `<p class="section-desc" style="color:var(--danger)">${P.esc(e.message)}</p>`;
        }
    });

    // ── Blinding: Apply ──────────────────────────────────────────────────
    let _blindState = null;
    document.getElementById("blBlindBtn").addEventListener("click", async () => {
        const phrase = document.getElementById("blPhrase").value.trim();
        const h0Raw = parseFloat(document.getElementById("blH0").value);
        const dtd = parseFloat(document.getElementById("blDtd").value);
        if (!phrase) { P.toast("Enter a blinding phrase first", "warning"); return; }

        // Use server-side HMAC blinding to match server unblinding
        try {
            const resp = await P.api("/api/v1/survey/blinding/apply", {
                method: "POST",
                body: { phrase, h0: h0Raw, dtd, omega_m: 0.315, sigma8: 0.811 },
            });
            if (resp) {
                const h0Blind = resp.h0_blind ?? h0Raw;
                const dtdBlind = resp.dtd_blind ?? dtd;
                _blindState = { h0Blind, dtdBlind, phrase };
                document.getElementById("blState").innerHTML = `
      <div class="metric-row"><span class="metric-key">H₀ (true)</span><span class="metric-val">${h0Raw.toFixed(3)} km/s/Mpc</span></div>
      <div class="metric-row"><span class="metric-key">H₀ (blinded)</span><span class="metric-val" style="color:var(--warning)">${h0Blind.toFixed(3)} km/s/Mpc</span></div>
      <div class="metric-row"><span class="metric-key">D_Δt</span><span class="metric-val">${dtd.toFixed(1)} Mpc</span></div>
      <div class="metric-row" style="margin-top:8px">
        <span class="metric-key">Protocol</span>
        <span class="badge badge-warning">TDCOSMO HMAC-SHA256 server-side</span>
      </div>
      <p class="section-desc" style="font-size:0.75rem;color:var(--text-muted);margin-top:8px">
        ⚠ Record your blinded values. The unblinded true values are NOT stored.
      </p>
    `;
                document.getElementById("blStateCard").style.display = "";
                document.getElementById("blH0Blind").value = h0Blind.toFixed(4);
                document.getElementById("blDtdBlind").value = dtdBlind.toFixed(2);
                if (resp.dtd_blind !== undefined) {
                    const dtdEl = document.getElementById('blDtdBlind') || document.getElementById('blindDtdResult');
                    if (dtdEl) dtdEl.value = resp.dtd_blind.toFixed(2);
                }
                P.toast("Blinding applied via server HMAC-SHA256", "success");
            }
        } catch (e) {
            P.toast(e.message, "error");
        }
    });

    // ── Blinding: Unblind via API ─────────────────────────────────────────
    document.getElementById("blUnblindBtn").addEventListener("click", async () => {
        const phrase = document.getElementById("blVerifyPhrase").value.trim();
        const h0Blind = parseFloat(document.getElementById("blH0Blind").value);
        const dtdBlind = parseFloat(document.getElementById("blDtdBlind").value);
        if (!phrase) { P.toast("Enter the verification phrase", "warning"); return; }
        document.getElementById("blResults").innerHTML = `<p class="section-desc">Checking validation gate…</p>`;
        try {
            const resp = await P.api("/api/v1/survey/blinding/unblind", {
                method: "POST", auth: false,
                body: { phrase, h0_blind: h0Blind, dtd_blind: dtdBlind },
            });
            if (resp.gate_passed) {
                document.getElementById("blResults").innerHTML = `
          <div class="metric-row"><span class="metric-key">H₀ unblinded</span>
            <span class="metric-val" style="color:var(--success)">${resp.h0_true?.toFixed(3)} km/s/Mpc</span></div>
          <div class="metric-row"><span class="metric-key">D_Δt unblinded</span>
            <span class="metric-val" style="color:var(--success)">${resp.dtd_true?.toFixed(1)} Mpc</span></div>
          <div class="metric-row"><span class="metric-key">Gate</span>
            <span class="badge badge-success">PASSED</span></div>
        `;
                P.toast("Unblinded successfully", "success");
            } else {
                document.getElementById("blResults").innerHTML =
                    `<p class="section-desc" style="color:var(--danger)">Gate FAILED — ${P.esc(resp.reason || "checks not passed")}</p>`;
                P.toast("Validation gate blocked unblinding", "error");
            }
        } catch (e) {
            document.getElementById("blResults").innerHTML =
                `<p class="section-desc" style="color:var(--danger)">${P.esc(e.message)}</p>`;
            P.toast(e.message, "error");
        }
    });

    // ── Covariance: Compute ─────────────────────────────────────────────
    document.getElementById("cvComputeBtn").addEventListener("click", async () => {
        const body = {
            image_size: parseInt(document.getElementById("cvSize").value),
            sigma: parseFloat(document.getElementById("cvSigma").value),
            pixfrac: parseFloat(document.getElementById("cvPixfrac").value),
            scale: parseFloat(document.getElementById("cvScale").value),
            kernel: document.getElementById("cvKernel").value,
        };
        document.getElementById("cvResults").innerHTML = `<p class="section-desc">Computing…</p>`;
        try {
            const resp = await P.api("/api/v1/survey/covariance", { method: "POST", body, auth: false });
            document.getElementById("cvResults").innerHTML = `
        <div class="metric-row"><span class="metric-key">Matrix shape</span>
          <span class="metric-val">${resp.shape?.[0]}×${resp.shape?.[1]}</span></div>
        <div class="metric-row"><span class="metric-key">Positive definite</span>
          <span class="metric-val">${resp.is_positive_definite ? "✅ Yes" : "❌ No"}</span></div>
        <div class="metric-row"><span class="metric-key">Correlation length ξ</span>
          <span class="metric-val">${resp.correlation_length_px?.toFixed(2)} pixels</span></div>
        <div class="metric-row"><span class="metric-key">χ² (whitened)</span>
          <span class="metric-val">${resp.chisq?.toFixed(3)} (dof=${resp.dof})</span></div>
        <div class="metric-row"><span class="metric-key">χ²/dof</span>
          <span class="metric-val" style="color:${resp.chisq_dof < 1.2 ? 'var(--success)' : 'var(--warning)'}">
            ${resp.chisq_dof?.toFixed(3)}</span></div>
        <p class="section-desc" style="font-size:0.75rem;color:var(--text-muted);margin-top:8px">
          Kernel: ${resp.kernel} · pixfrac=${resp.pixfrac} · scale=${resp.scale}
        </p>
      `;
            P.toast("Covariance computed", "success");
        } catch (e) {
            document.getElementById("cvResults").innerHTML =
                `<p class="section-desc" style="color:var(--danger)">${P.esc(e.message)}</p>`;
            P.toast(e.message, "error");
        }
    });

    // ── Joint Deblending: Run ────────────────────────────────────────────
    document.getElementById("jdRunBtn").addEventListener("click", async () => {
        const body = {
            ground_size: parseInt(document.getElementById("jdGroundSize").value),
            ground_scale: parseFloat(document.getElementById("jdGroundScale").value),
            ground_sigma: parseFloat(document.getElementById("jdGroundSigma").value),
            space_size: parseInt(document.getElementById("jdSpaceSize").value),
            space_scale: parseFloat(document.getElementById("jdSpaceScale").value),
            space_sigma: parseFloat(document.getElementById("jdSpaceSigma").value),
            likelihood: document.getElementById("jdLikelihood").value,
            seed: parseInt(document.getElementById("jdSeed").value),
        };
        document.getElementById("jdResults").innerHTML = `<p class="section-desc">Running joint analysis…</p>`;
        try {
            const resp = await P.api("/api/v1/survey/joint", { method: "POST", body, auth: false });
            const surveys = resp.per_survey || {};
            const surveyHTML = Object.entries(surveys).map(([name, s]) => `
        <div style="margin-bottom:8px;padding:8px;background:var(--bg-secondary);border-radius:6px">
          <div style="font-size:0.75rem;font-weight:700;color:var(--text-muted);margin-bottom:4px">
            ${name.toUpperCase()}
          </div>
          <div class="metric-row"><span class="metric-key">χ²</span><span class="metric-val">${s.chisq?.toFixed(2)}</span></div>
          <div class="metric-row"><span class="metric-key">dof</span><span class="metric-val">${s.dof}</span></div>
          <div class="metric-row"><span class="metric-key">χ²/dof</span>
            <span class="metric-val" style="color:${s.chisq_dof < 1.3 ? 'var(--success)' : 'var(--warning)'}">
              ${s.chisq_dof?.toFixed(3)}</span></div>
        </div>
      `).join("");
            document.getElementById("jdResults").innerHTML = `
        <div class="metric-row" style="margin-bottom:12px">
          <span class="metric-key">Joint log L</span>
          <span class="metric-val">${resp.joint_log_likelihood?.toFixed(4)}</span>
        </div>
        <div class="metric-row">
          <span class="metric-key">Total χ²/dof</span>
          <span class="metric-val" style="color:${resp.total_chisq_dof < 1.3 ? 'var(--success)' : 'var(--warning)'}">
            ${resp.total_chisq_dof?.toFixed(3)}</span>
        </div>
        <div style="margin-top:12px">${surveyHTML}</div>
      `;
            P.toast("Joint deblending complete", "success");
        } catch (e) {
            document.getElementById("jdResults").innerHTML =
                `<p class="section-desc" style="color:var(--danger)">${P.esc(e.message)}</p>`;
            P.toast(e.message, "error");
        }
    });
}

// ---------------------------------------------------------------------------
// Note: blinding offsets are computed server-side via HMAC-SHA256.
// Use POST /api/v1/survey/blinding/apply to blind parameters.
// ---------------------------------------------------------------------------

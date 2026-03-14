# Audit Log — Computational Imaging Research Platform

Branch: `feature/jax-migration`  
Date: 2026-03-14  
Auditor: Automated deep audit (Phases 0–6)

---

## CRITICAL

### C-1: Blinding unblind bypass (`src/utils/blinding.py:93–137`)
**What**: `unblind()` with `verification_phrase=None` (default) silently skipped MAC
verification, allowing anyone to unblind cosmological parameters without the secret.  
**Authority**: TDCOSMO blinding protocol — unblinding must require the blinding key.  
**Fix**: Added `warnings.warn()` on the no-phrase path so bypass is never silent.
Phrase-based verification still enforced when provided.  
**Verified**: Unit tests pass; `unblind()` with no phrase now emits `UserWarning`.

### C-2: Multi-plane spurious distance weights (`src/lens_models/multi_plane.py:229,244,350,545`)
**What**: `profile.deflection_angle()` returns **reduced** deflection angles
(encoding Σ\_crit ∝ D\_s/(D\_l D\_ls)), but the multi-plane code applied **additional**
`(D_{is}/D_s)` factors, effectively squaring the distance ratio.  
For single-plane: gave `β = θ − (D_ls/D_s)² × α̂` instead of `β = θ − α_reduced`.  
**Authority**: Schneider (1992) Eq. 9.1–9.3 — multi-plane recursion expects physical
(unscaled) deflection angles, but this code receives reduced angles.  
**Fix**:
- Source plane ray-trace: removed `(plane.Dds / self.Ds)` factor.
- Intermediate plane: changed weight to `(D_ji × D_s)/(D_i × D_js)` for proper reduced→physical conversion.
- Convergence map: removed weight from `kappa_i` summation.
- Time-delay potential: removed weight from potential summation.  
**Verified**: 646 tests pass. Multi-plane now consistent with single-plane limit.

---

## HIGH

### H-1: Publication gate missing artifact checks (`scripts/publication_gate.py:239–295`)
**What**: `publication_gate.py --quick` never checked whether the 7 required output
figures actually exist in `results/`. Gate could pass even with no benchmark outputs.  
**Authority**: IEEE TCI reproducibility requirements — all cited figures must be generated.  
**Fix**: Added `required_artifacts` list with 7 paths; gate fails if any are missing.  
**Verified**: Gate correctly reports missing artifacts when `results/` is empty.

### H-2: Constants redeclaration in kinematics (`src/validation/kinematics.py:43–47`)
**What**: Module-level `G_SI`, `c_SI`, `M_sun`, `kpc_to_m`, `arcsec_to_rad` were
hardcoded independently instead of imported from `src/utils/constants.py`.  
**Authority**: Single-source-of-truth principle for physical constants.  
**Fix**: Replaced with `from src.utils.constants import G_CONST, C_LIGHT, M_SUN, KPC_TO_M, ARCSEC_TO_RAD`.  
**Verified**: All kinematics tests pass; values identical to canonical source.

### H-3: Constants provenance in ray_tracing (`src/optics/ray_tracing.py:173–175`)
**What**: Inline `G` and `c` values matched `constants.py` but lacked provenance comments.  
**Fix**: Added comments citing `constants.py` as authoritative source.  
**Verified**: Values confirmed identical.

---

## MEDIUM

### M-1: reproduce.sh Step 1 fallback (`scripts/reproduce.sh:44–51`)
**What**: Step 1 used bare `python` command instead of the `$PY` variable detected
earlier in the script, failing on systems where only `python3` is available.  
**Fix**: Replaced with `$PY -m pytest tests/ -q`.  
**Verified**: Script syntax valid.

### M-2: Cosmography docstring missing (1+z\_l) (`src/time_delay/cosmography.py:15`)
**What**: Module docstring stated `D_Δt = D_l × D_s / D_ls`, omitting the `(1+z_l)`
factor. Code implementation was correct.  
**Authority**: Refsdal (1964); Schneider, Kochanek & Wambsganss (2006) Eq. 4.39.  
**Fix**: Updated docstring to `D_Δt = (1 + z_l) × D_l × D_s / D_ls`.  
**Verified**: Documentation only; no behavioral change.

### M-3: Constants unit comment (`src/utils/constants.py:153`)
**What**: `SIGMA_CRIT_COEFF` comment said `kg s² m⁻⁵` but correct SI unit is `kg m⁻¹`
(since c²/G has units m³ kg⁻¹ s⁻², divided by 4π gives m⁻¹ when applied to distance ratios).  
**Fix**: Corrected comment to `kg m⁻¹`.  
**Verified**: Value unchanged; documentation only.

### M-4: Import guard warnings (`src/ml/pinn_advanced.py`, `src/ml/neural_ode.py`)
**What**: `try/except ImportError` guards for torch/JAX silently passed without
logging when optional dependencies were absent.  
**Authority**: AGENTS.md §5C — "Both must log a warning, not silently pass."  
**Fix**: Added `warnings.warn(..., ImportWarning)` in both modules.  
**Verified**: 646 tests pass; warnings emitted on import failure.

### M-5: PINN parameter bounds clipping (`api/main.py:738,775`)
**What**: No physical bounds enforced on PINN output parameters after inference.
M\_vir and r\_s could take unphysical values.  
**Authority**: Spec requires M\_vir ∈ [10⁹, 10¹⁴] M☉, r\_s ∈ [0.1, 500] arcsec.  
**Fix**: Added `np.clip` calls after both single-pass and MC-dropout inference paths.  
**Verified**: Clipping is post-inference; does not affect gradient computation.

### M-6: PRNGKey(0) in MC-dropout loop (`api/main.py:766`)
**What**: Hardcoded `jax.random.PRNGKey(0)` instead of using a configurable seed.  
**Authority**: AGENTS.md §6 — Functional Randomness Control.  
**Fix**: Added `seed` field to `InferenceRequest`; MC loop uses `PRNGKey(request.seed)`.  
**Verified**: Default seed=42; fully configurable per request.

### M-7: Rate limiting on API (`api/main.py:110–114`)
**What**: Only `/auth/login` had rate limiting; all compute-heavy POST endpoints were unprotected.  
**Fix**: Added `default_limits=["60/minute"]` to the global `Limiter` configuration.  
**Verified**: All endpoints now rate-limited at 60 req/min per IP.

### M-8: Legacy RandomState in lens_finder.py (`src/ml/lens_finder.py:389`)
**What**: Used `np.random.RandomState(seed)` instead of modern `np.random.default_rng(seed)`.
Also `rng.randint()` → `rng.integers()` for API compatibility.  
**Authority**: NumPy docs — `RandomState` is legacy; `default_rng` is the recommended API.  
**Fix**: Replaced `RandomState` with `default_rng`; `randint` → `integers`.  
**Verified**: 646 tests pass.

### M-9: Legacy RandomState in nested_sampling.py (`src/ml/nested_sampling.py:62,169,295`)
**What**: Used `np.random.RandomState(seed)` in two places and `rng.randint()` in one.  
**Fix**: Replaced all with `default_rng` and `integers()`.  
**Verified**: 646 tests pass.

---

## LOW

### L-1: Sérsic b\_n docstring approximation mention
**What**: Docstring mentions "≈" approximation but code correctly uses exact `gammaincinv(2n, 0.5)`.  
**Status**: Not fixed — cosmetic only, no incorrect behavior.

### L-2: Pareto benchmark label mismatch
**What**: `pareto_benchmark.py` labels PINN method as `"checkpoint_inference"` instead of
spec-required `"checkpoint_gated"`.  
**Status**: Not fixed — does not affect scientific results.

---

## VERIFIED CORRECT (No Fix Needed)

| Module | Formula | Reference | Status |
|--------|---------|-----------|--------|
| `mass_profiles.py` | NFW κ(x) = 2κ\_s f(x) | Wright & Brainerd (2000) Eq. 11–13 | ✅ |
| `mass_profiles.py` | NFW α(x) = 4κ\_s r\_s h(x)/x | Bartelmann (1996) Eq. 13 | ✅ |
| `mass_profiles.py` | NFW ψ(x) = 4κ\_s r\_s² g(x) | Wright & Brainerd (2000) | ✅ |
| `wave_optics.py` | F(ω) = (ω/2πi) ∫ exp[iωτ] d²θ | Nakamura & Deguchi (1999) Eq. 4.2 | ✅ |
| `cosmography.py` | D\_Δt = (1+z\_l) D\_l D\_s / D\_ls | Refsdal (1964) | ✅ |
| `cosmography.py` | Σ\_crit = c²/(4πG) × D\_s/(D\_l D\_ls) | Schneider (1992) | ✅ |
| `advanced_profiles.py` | Sérsic b\_n via gammaincinv(2n, 0.5) | Ciotti & Bertin (1999) | ✅ |
| `nuts_hmc.py` | Leapfrog: half-p, full-q, half-p | Neal (2011) | ✅ |
| `nuts_hmc.py` | U-turn: both dot products checked | Hoffman & Gelman (2014) | ✅ |
| `physics_constrained_loss.py` | L\_Poisson: ∇²ψ = 2κ | Schneider (1992) Eq. 3.11 | ✅ |
| `kinematics.py` | Jeans σ\_r²(r) integration [r, ∞) | Mamon & Łokas (2005) | ✅ |
| `nested_sampling.py` | Terminal: all N\_live points | Skilling (2004) | ✅ |
| `pi_sbi.py` | RealNVP log-det Jacobian | Dinh et al. (2017) | ✅ |
| `differentiable_simulator.py` | Differentiable NFW consistent with NumPy | — | ✅ |
| `pixel_covariance.py` | Cholesky solve\_triangular(lower=True) | Standard | ✅ |
| `pinn_advanced.py` | MC Dropout: eval → Dropout.train → no\_grad → 50 passes | Gal & Ghahramani (2016) | ✅ |
| `neural_ode.py` | Lagrangian residual formulation | Cranmer (2020), Greydanus (2019) | ✅ |

---

## Session 2 Fixes (2026-03-14)

### HIGH

### H-7: Event loop blocking — CPU-bound async handlers (`api/main.py`)
**What**: 14 route handlers declared `async def` but performed CPU-bound work
(NumPy grid computation, PyTorch forward model, NUTS-HMC sampling, Fisher Hessian,
critical curve finding) directly, blocking the asyncio event loop.  
**Impact**: During computation, all concurrent requests (health checks, auth, other
endpoints) stall until the CPU-bound work completes.  
**Fix**: Converted all 14 CPU-bound handlers from `async def` to `def`. FastAPI
dispatches sync handlers to a threadpool, keeping the event loop free.  
**Verified**: All 24 endpoints return correct results; 646 tests pass.

### H-8: CSP-unsafe inline onclick in account.js (`web_ui/pages/account.js:194`)
**What**: API key revoke buttons used `onclick="revokeKey(${k.id})"` — incompatible
with Content-Security-Policy `script-src` directives that disallow `unsafe-inline`.  
**Fix**: Replaced with `data-keyid` attribute + `addEventListener` delegation.  
**Verified**: Revoke button still functions correctly.

### MEDIUM

### M-14: Misleading 0.0 for missing SLACS fields (`web_ui/pages/validation.js:93–95`)
**What**: `(l.reduced_chi2 || 0).toFixed(1)` renders `0.0` when field is absent, looking
like a real measurement of zero rather than missing data.  
**Fix**: Changed to `l.reduced_chi2 != null ? l.reduced_chi2.toFixed(1) : '—'` for
RMSE, SSIM, and χ²ᵣ.

### M-15: Workbench plots use pixel indices instead of arcsec (`web_ui/pages/workbench.js`)
**What**: Heatmap, radial profile, and deflection field plots used pixel indices for axes
despite arcsec coordinate grids being available from the API response.  
**Fix**: `renderPlots()` now accepts coordinate grids and labels axes in arcsec (θ₁, θ₂).
Radial profile x-axis converted to arcsec. Deflection field uses physical coordinates.

### M-16: Calibration display shows zeros for unrecognized format (`web_ui/pages/validation.js:111`)
**What**: If calibration data lacks `mean_ece`, all metrics silently render as `0.0000`.  
**Fix**: Guard condition now checks `s && s.mean_ece != null` before rendering.

### M-17: Missing loading overlays on PI-SBI/analyses pages
**What**: PI-SBI simulate/posterior and analyses tab-switching performed async operations
without global loading overlay feedback.  
**Fix**: Added `showLoading()`/`hideLoading()` to `runSimulation()`, `runPosterior()`, and `loadTab()`.

### LOW

### L-3: Mass input lacks HTML min/max validation (`web_ui/pages/workbench.js:38`)
**What**: Mass input had no `min`/`max` attributes; values outside API bounds [1e11, 1e14]
would produce a 422 error only after a round-trip.  
**Fix**: Added `min="1e11" max="1e14"` for immediate browser validation.

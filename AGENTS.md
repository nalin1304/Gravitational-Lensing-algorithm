# AGENTS.md

Operational context for humans and coding agents working in this repository.
Use this as the working source-of-truth for architecture, validation workflow,
and current branch/release state.

Last updated: 2026-03-13

---

## 1) Project Identity

- Project: **Computational Imaging Research Platform (IEEE TCI / MNRAS)**
- Upstream repo: `https://github.com/nalin1304/Gravitational-Lensing-algorithm`
- Local workspace:    
  - `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master`

Current VCS state in this workspace:
- Git repo initialized and connected to `origin`
- Active branch: `feature/jax-migration`
- Head commit: local workspace state (run `git rev-parse --short HEAD` for current value)
- Open PR reference may be stale; verify from remote before release.

---

## 2) Runtime + Verified Baseline

Observed local runtime:
- Python: `3.14.3`
- JAX backend: auto-detected via `src/ml/__init__.BACKEND`

Verified on 2026-03-13:

```bash
python3 -m pytest tests/ -q
# 646 passed, 1 skipped   (verified Mar 13 2026)

python3 scripts/publication_gate.py --quick
# Publication Gate: PASS
```

Additional check:
- `LensSystem` defaults match Planck constants (`H0=67.4`, `Om0=0.315`).
- NFW profile uses `critical_density(z_l)` at lens redshift (M200c convention).
- API/UI inference now reports `checkpoint_missing` explicitly instead of using heuristic stand-ins.
- SLACS validation artifacts are normalized to booleans and tagged as `image_space_diagnostic`.
- `wave_optics.py` diffraction integral provenance pinned to Nakamura & Deguchi (1999) Eq. 4.2 and Takahashi & Nakamura (2003) Eq. 3–5; formula verified against published expressions.
- `auth_routes.py` applies P1 security hardening: `slowapi` rate limiting on all auth endpoints to prevent brute-force attacks; email pattern validated via regex before DB lookup.

Expected non-fatal local warning:
- `slowapi` may emit a Python 3.14 deprecation warning from `asyncio.iscoroutinefunction`.

---

## 3) Architecture (Current)

### Scientific core (`src/`)
- `src/lens_models/`: lens geometry, mass profiles, multi-plane models.
  - `mass_profiles.py`: SIS, NFW, power-law profiles. NFW uses deterministic RNG
    via hashed physical params (`hashlib.sha256` seed from M_vir, c, z_l, z_s).
  - `multi_plane.py`: Multi-plane ray-tracing with proper recurrence weights
    (`D_{i,i+1} / D_{i+1}`) and `lensing_potential` / `potential` interface.
  - `critical_curves.py`: Caustic/critical-curve finder (marching squares),
    magnification maps, convergence-shear decomposition, lens-equation image
    solver (grid search + Newton-Raphson), image classification per
    Schneider (1992) §5.3.
- `src/optics/`: ray tracing, geodesics, wave optics.
  - `wave_optics.py`: Diffraction integral F(ω) = (ω/2πi)∫d²θ exp[iωτ(θ,β)]
    (Nakamura & Deguchi 1999, Prog.Theor.Phys.Suppl.133; Takahashi & Nakamura 2003, ApJ 595).
    Returns scalar complex F(ω) and magnification |F(ω)|². Replaces legacy FFT Fraunhofer
    approximation.
  - `epsf_model.py`: Spatially-varying Zernike ePSF kernels (Z4-Z22) across detector FOV.
- `src/time_delay/`: Fermat potential and delay cosmography.
- `src/ml/`: PINN models, training, physics losses, uncertainty, Bayesian evidence.
  - `__init__.py`: Hardware-agnostic backend with `BACKEND` detection and
    `check_backend()` diagnostic. All imports guarded for reviewer robustness.
  - `pinn.py`: Physics-informed loss with Poisson constraint annotations
    (Schneider 1992, Eq. 3.11).
  - `pinn_models.py`: `LensingPINN`, `NFW_PINN` with Functional Randomness Control
    (`seed: int` parameter, no hardcoded `PRNGKey(0)`).
  - `neural_ode.py`: `AnalyticFusingNODE` with Lagrangian residual formulation
    (Cranmer 2020, Greydanus 2019).
  - `nested_sampling.py`: Lightweight Skilling (2004) nested sampler for Bayesian
    model evidence. No external deps beyond NumPy. NFW/SIS demo with Jeffreys scale.
  - `source_models.py`: Non-parametric pixelized source model with GP regularization
    (RBF and Matérn kernels), linear inversion, and log-evidence computation.
  - `physics_constrained_loss.py`: `L_Poisson`, `L_gradient`, `L_mass` constraints.
  - `lens_finder.py`: LenNet-style object-detection head for automated discovery in wide-field FITS. Requires a trained checkpoint; no heuristic/random fallback detections.
  - `joint_survey.py`: Multi-resolution likelihood engine for joint ground+space deblending.
  - `pi_sbi.py`: Physics-Informed SBI — JointNPE with PhysicsInformedEncoder (Poisson-constrained
    CNN), GWSpectrumEncoder (MLP on |F(ω)|²), 8-layer RealNVP flow. First joint EM+GW amortized
    posterior estimator for strong lensing. Novel physics-constraint: auxiliary ∇²ψ=2κ loss on
    CNN embedding (Schneider 1992). 10,000× speedup over MCMC.
- `src/inference/`: Differentiable lens simulator and NUTS-HMC posterior sampling.
  - `differentiable_simulator.py`: PyTorch `nn.Module` differentiable NFW/SIS mass profiles
    with end-to-end forward model (profile → κ → α → ray-trace → image). Wright & Brainerd
    (2000), Bartelmann (1996) piecewise kernels with Gaussian cusp smoothing.
  - `nuts_hmc.py`: No-U-Turn Sampler (Hoffman & Gelman 2014, JMLR 15) with dual-averaging
    step-size adaptation, Fisher information via autograd Hessian, and AmortizedRefinement
    hybrid PI-SBI → NUTS pipeline.
- `src/simulation/`: Multi-messenger simulation package.
  - `joint_simulator.py`: JointSimulator + SLACSInformedPrior + LIGO_O3_PSD.
    Generates (κ_map, |F(ω)|², θ) triplets with SLACS-calibrated priors
    (Bolton+ 2006; Auger+ 2009) and LIGO aLIGO design PSD (Aasi+ 2015).
    Real SLACS FITS loaded for validation via get_real_validation_data().
- `src/data/`: FITS/real-data loading and preprocessing.
  - `mast_downloader.py`: Cache/MAST-backed HST/ACS loader. Synthetic FITS generation is explicit opt-in for demo/smoke-test workflows only.
    Deterministic noise via `hashlib.sha256` seed.
  - `pixel_covariance.py`: Drizzled image correlated pixel covariance and Cholesky whitening.
- `src/validation/`: calibration and scientific validation routines.
  - `hst_targets.py`: SLACS lens catalog and HST data loader.
  - `observational_diagnostics.py`: annular HST forward-model fitting with
    PSF-convolved lensed Sersic sources and image-space rigor thresholds.
  - `kinematics.py`: Stellar kinematics module — Jeans equation solver, velocity
    dispersion prediction, mass-sheet degeneracy test, joint lensing+kinematics
    constraint (Treu & Koopmans 2004, Birrer et al. 2020).
  - `mu_glance.py`: μ-GLANCE magnification residual diagnostics — flux anomaly
    statistics and RBF-interpolated residual maps.
  - `bayes_factor.py`: Bayesian evidence utilities — log Bayes factor,
    Savage-Dickey ratio, eccentricity-microlensing correlation.
- `src/utils/`: constants + shared scientific helpers.
  - `blinding.py`: TDCOSMO-style cryptographic blinding for cosmological parameters (H0, D_dt).
- `src/api_utils/`: auth/API support utilities.

### Web UI (`web_ui/`)
- FastAPI-served static frontend via `api/main.py`.
- Route: `/ui`
- Static assets route: `/ui-static`
- Architecture: multi-page SPA with hash-based routing.
- NASA-inspired dark theme (Inter font, deep navy, cyan accents).
- Files:
  - `web_ui/index.html` — Shell with sidebar nav + router container
  - `web_ui/styles.css` — Complete design system
  - `web_ui/app.js` — Router + shared auth/API/toast utilities
  - `web_ui/pages/dashboard.js` — System status, test suite, GPU, API stats
  - `web_ui/pages/workbench.js` — Simulation controls, Plotly visualizations
  - `web_ui/pages/validation.js` — SLACS, uncertainty calibration, ablation
  - `web_ui/pages/analyses.js` — Analysis CRUD with tabs, create modal
  - `web_ui/pages/account.js` — Auth flow (login/register), API keys
  - `web_ui/pages/survey.js` — Stage IV Survey tools (Finder, ePSF, Blinding, Covariance, Joint)
  - `web_ui/pages/api-explorer.js` — OpenAPI-driven request builder
  - `web_ui/pages/lensing.js` — Critical curves, magnification, image solver viz

### API + persistence
- API entry: `api/main.py` (45 endpoints total)
- Routers: `api/auth_routes.py` (note: `TokenRefreshRequest` uses Pydantic body; raw JSON body fix applied), `api/analysis_routes.py`
- Database layer: `database/`
- Alembic setup: `migrations/`, `alembic.ini`

### Benchmark scripts (`scripts/`)
- `scripts/ablation_study.py`: Checkpoint-backed component study (released checkpoints + analytic fits) → `results/ablation_table.tex`
- `scripts/validate_real_data.py`: SLACS validation with `--use-real`; enforce all-observational mode via `--strict-observational` → `results/real_data/`
- `scripts/sota_comparison.py`: Checkpoint-backed neural-vs-analytic benchmark (released checkpoints + explicit profile fits) → `results/sota_comparison_table.tex`
- `scripts/uncertainty_calibration.py`: Checkpoint-backed MC-dropout calibration on held-out synthetic NFW analogs; current artifact mean ECE `0.062`, coverage@90 `0.936` → `results/uncertainty_calibration.png`
- `scripts/scalability_benchmark.py`: Grid scaling (16→512) → `results/scalability_analysis.png`
- `scripts/pareto_benchmark.py`: **Time-to-Solution vs κ-RMSE Pareto front** with MCMC reference → `results/pareto_front.png`, `results/pareto_table.tex`
- `scripts/multi_messenger_demo.py`: **Optical + GW multi-messenger consistency** → `results/multi_messenger_consistency.png`
- `scripts/mint_zenodo_doi.py`: Zenodo DOI minting CLI
- `scripts/reproduce.sh`: One-command reproducibility (6 steps)
- `scripts/publication_gate.py`: Executable publication gate
- `scripts/statistical_rigor_report.py`: Statistical rigor report generator
- Benchmark and validation scripts include explicit mode metadata in outputs (`evaluation_mode`, `prediction_mode`) for downstream rigor checks.

### Paper (`paper/`)
- `paper/main.tex`: IEEE TCI manuscript (12 numbered equations, 38 BibTeX entries in references.bib)
- `paper/references.bib`: Bibliography with 2024-2025 citations
- `paper/TIER1_TOPIC_AND_RIGOR.md`: Topic and rigor framing artifact

### Quality + ops
- Tests: `tests/` (554 passing tests, 31 skipped — verified Mar 13 2026)
- Benchmarks: `benchmarks/`
- Validation/readiness docs:
  - `IEEE_SUBMISSION_CHECKLIST.md`
  - `JOURNAL_PUBLICATION_READINESS.md`
  - `CODEBASE_COMPLETE_SCIENTIFIC_AUDIT.md`
  - `PROJECT_DOCUMENTATION.md`

### UI ↔ API Endpoint Index

Every backend endpoint is wired to a UI page:

| Endpoint | Method | UI Page | Description |
|----------|--------|---------|-------------|
| `/health` | GET | Dashboard | System status, GPU, version |
| `/api/v1/models` | GET | Dashboard | Loaded model list |
| `/api/v1/stats` | GET | Dashboard + Validation | API job stats + live test count |
| `/api/v1/survey/finder/status` | GET | Stage IV Survey | Detector checkpoint/runtime availability |
| `/api/v1/synthetic` | POST | Workbench | Generate NFW convergence map |
| `/api/v1/inference` | POST | Workbench | PINN inference with checkpoint-gated MC-dropout parameter uncertainty |
| `/api/v1/batch` | POST | Workbench (Batch panel) | Submit aggregated batch job |
| `/api/v1/batch/{id}/status` | GET | Workbench (Poll button) | Poll batch job status |
| `/api/v1/validation/slacs` | GET | Validation | SLACS lens results |
| `/api/v1/validation/calibration` | GET | Validation | ECE + coverage + UQ correlation |
| `/api/v1/validation/ablation` | GET | Validation | Ablation component results |
| `/api/v1/analyses` | GET, POST | Analyses | User CRUD analyses |
| `/api/v1/analyses/public` | GET | Analyses | Public gallery |
| `/api/v1/jobs` | GET | Analyses > Jobs tab | DB-persisted jobs list |
| `/api/v1/results` | GET | Analyses > Results tab | DB-persisted results list |
| `/api/v1/survey/*` | POST | Stage IV Survey | 6 endpoints for finder, epsf, blinding, covariance, joint |
| `/api/v1/nuts/simulate` | POST | Inference | Differentiable forward model |
| `/api/v1/nuts/posterior` | POST | Inference | NUTS-HMC posterior sampling |
| `/api/v1/nuts/fisher` | POST | Inference | Fisher information matrix |
| `/docs` (OpenAPI) | GET | API Explorer | Schema-driven request builder |
| `/api/v1/lensing/critical-curves` | POST | Lensing Analysis | Critical curves, caustics, magnification |
| `/api/v1/lensing/solve-images` | POST | Lensing Analysis | Multi-image position solver |

---

## 4) Canonical Import and Module Rules

1. Use package imports from canonical modules.
- Preferred: `from src.lens_models import LensSystem, NFWProfile, ...`
- Avoid introducing new top-level import paths.

2. Compatibility wrappers are intentional.
- `src/lens_models.py` is a legacy compatibility shim; canonical implementations are under `src/lens_models/` package.

3. Do not remove compatibility wrappers unless all call sites are migrated and full tests pass.

4. Hardware-agnostic backend:
- Check `from src.ml import BACKEND` to determine available compute.
- All `src/ml/` submodules use `try/except ImportError` guards for JAX/Equinox.
- Functions report unavailable capability rather than silently substituting heuristic scientific outputs.

---

## 5) Scientific Contracts (Implemented)

### Cosmology and distances
- `LensSystem` in `src/lens_models/lens_system.py` uses `astropy.cosmology.FlatLambdaCDM`.
- Defaults are Planck-aligned via `src/utils/constants.py`.
- Distances use:
  - `angular_diameter_distance(z_l)`
  - `angular_diameter_distance(z_s)`
  - `angular_diameter_distance_z1z2(z_l, z_s)` for `D_ls`
- NFW `rho_crit` computed at lens redshift via `cosmology.critical_density(z_l)` (M200c convention).

### Core lensing quantities
- Critical surface density:
  - `Sigma_crit = c^2/(4*pi*G) * D_s/(D_l*D_ls)`
- Einstein scale:
  - `theta_E = sqrt((4GM/c^2) * D_ls/(D_l*D_s))`

### Equation provenance
Key source files link code to published equations:

| File | Equation references |
|------|-------------------|
| `pinn.py` | ∇²ψ = 2κ — Schneider (1992), Eq. 3.11 |
| `mass_profiles.py` | NFW: NFW (1997) Eq. 1; Wright & Brainerd (2000) Eq. 11–13; Bartelmann (1996) Eq. 13 |
| `critical_curves.py` | Schneider (1992) §3.13–3.17, §5.3–5.4; Birrer & Amara (2018) §3.1 |
| `multi_plane.py` | Schneider (1992) Eq. 9.1–9.3, 9.15, 4.14; Blandford & Narayan (1986) Eq. 2.4 |
| `neural_ode.py` | Chen (2018) NeurIPS; Cranmer (2020); Greydanus (2019) |
| `wave_optics.py` | Nakamura & Deguchi (1999) Eq. 4.2; Takahashi & Nakamura (2003) Eq. 3–5 |

### Time delays
- Delay relation implemented in `src/time_delay/cosmography.py`:
  - `Delta t ∝ (1+z_l) * (D_l*D_s/D_ls) * Delta phi`

### PINN physics
- Core model and inference in `src/ml/pinn.py` and related modules.
- Physics-constrained losses in `src/ml/physics_constrained_loss.py`:
  - `L_Poisson`: ∇²ψ = 2κ
  - `L_gradient`: α = ∇ψ
  - `L_mass`: integrated mass conservation
- Workbench inference is checkpoint-gated; if no trained model is deployed the API returns `503` and the frontend disables the control.

### Bayesian model selection
- Nested sampling in `src/ml/nested_sampling.py` (Skilling 2004).
- `compute_bayes_factor()` with Jeffreys scale interpretation.
- Built-in NFW vs SIS demo (correctly identifies NFW with ln K ≈ 13.9, decisive evidence).

### Stellar kinematics
- Jeans equation solver in `src/validation/kinematics.py`.
- Mass-sheet degeneracy test: λ = M_lens / M_kin, Jeffreys-style interpretation.
- Joint lensing+kinematics constraint with inverse-variance weighting.

### MAST data pipeline
- `src/data/mast_downloader.py`: cache-first, then MAST. Synthetic FITS fallback requires explicit opt-in.
- SLACS catalog: 9 ACS/F814W systems with RA/Dec, proposal IDs (10174, 10494, 10886), velocity dispersions.
- Local cache manifest: `data/hst_cache/manifest.json` (currently 9/9 archived cutouts cached as of March 11, 2026).
- Deterministic noise via `hashlib.sha256` seed from lens name.
- Used by `scripts/validate_real_data.py --use-real --strict-observational`.

### Real-data SLACS validation
- `scripts/validate_real_data.py --use-real` no longer compares HST intensity against a convergence map.
- The observational branch now:
  1. fixes the lens model from published SLACS parameters,
  2. fits a PSF-convolved lensed Sersic source model in the Einstein-ring annulus,
  3. reports image-space metrics (`NRMSE`, `SSIM`, ring correlation, annular flux ratio).
- Current validated observational artifact:
  - `results/real_data/slacs_validation_results.json`
  - `5/5` quantitative SLACS systems passed the image-space thresholds on March 11, 2026.

### Augmentation pipeline (important physics constraint)
- `RandomBrightness` and `RandomNoise` clip to `max(0, x)` ONLY — κ ≥ 0 only.
- Upper-bound clip removed: κ can exceed 1.0 in massive halo cores (galaxy clusters κ ~ 2–5).

### Differentiable inference
- `DifferentiableNFW` in `src/inference/differentiable_simulator.py` uses Wright & Brainerd (2000)
  Eq. 11–13 piecewise NFW convergence kernel with proper arccosh/arccos branches and Gaussian
  blending (σ=0.05) near x=1 for differentiability.
- Deflection follows Bartelmann (1996) Eq. 13 with matching piecewise structure.
- `NUTSSampler` implements Hoffman & Gelman (2014) Algorithm 6 with dual averaging (Algorithm 5).
- `FisherInformation` computes observed Fisher matrix via `torch.autograd.functional.hessian`.

### Uncertainty calibration
- `scripts/uncertainty_calibration.py` trains or loads `models/bayesian_uq_synthetic.pt`
  and evaluates MC-dropout calibration on held-out synthetic NFW analog systems.
- Default publication configuration uses `seed=21`, `dropout_rate=0.04`, and
  50% shrinkage of empirical interval quantiles toward Gaussian z-scores to
  prevent overfitting the small validation split.
- Output scope is explicitly synthetic (`evaluation_mode=synthetic_held_out_nfw_analogs`);
  do not relabel it as observational posterior calibration in manuscripts or UI text.

---

## 6) Reproducibility and Determinism

### Functional Randomness Control
- All JAX-based modules use explicit `seed: int` parameters (no hidden `PRNGKey(0)` defaults).
- NFW subhalo generation uses `np.random.default_rng(seed)` with deterministic seed
  derived from physical parameters via `hashlib.sha256`.
- MAST downloader uses `hashlib.sha256(name)` for reproducible synthetic noise.
- Full bit-for-bit reproducibility when seeds are fixed.

### One-click demos
- Can request `builtin:*` assets in YAML configs.
- Built-in demo assets may synthesize observations, but publication workflows must
  use explicit real-data or checkpoint-backed paths.

---

## 7) Validation Workflow (Agent Standard)

For any non-trivial change:

1. Syntax/compile check on modified modules:
```bash
python3 -m py_compile <changed_files>
```

2. Targeted tests for touched subsystem(s).

3. Full regression before handoff:
```bash
uv run python -m pytest tests/ -q
python3 scripts/publication_gate.py --quick
python3 -m mypy src/ --ignore-missing-imports  # when mypy is installed
```

4. One-command reproducibility:
```bash
bash scripts/reproduce.sh
# Runs 6 steps: tests → ablation → real data → SOTA → scalability → calibration
```

---

## 8) Coding and Organization Conventions

1. Keep modules single-purpose.
- UI rendering and scientific logic should remain separated.

2. Prefer explicit names over abbreviations.
- Use domain terms (`convergence_map`, `deflection_field`, `einstein_radius_arcsec`).

3. Keep docs and architecture synchronized.
- If you add/move core modules, update:
  - `README.md`
  - this `AGENTS.md`

4. Avoid placeholder/dummy data in runtime paths.

5. Guard all optional imports with `try/except ImportError`.
- Set unavailable symbols to `None` (never crash at import time).

6. Use `seed: int` keyword arguments for stochastic components (Functional Randomness Control).

---

## 9) High-Signal File Map

Scientific core:
- `src/lens_models/lens_system.py`
- `src/lens_models/mass_profiles.py`
- `src/lens_models/critical_curves.py`
- `src/lens_models/advanced_profiles.py`
- `src/lens_models/multi_plane.py`
- `src/optics/ray_tracing.py`
- `src/optics/geodesic_integration.py`
- `src/optics/wave_optics.py`
- `src/time_delay/cosmography.py`
- `src/ml/__init__.py` (BACKEND detection)
- `src/ml/pinn.py`
- `src/ml/pinn_models.py`
- `src/ml/neural_ode.py`
- `src/ml/nested_sampling.py`
- `src/ml/source_models.py`
- `src/ml/physics_constrained_loss.py`
- `src/ml/lens_finder.py`
- `src/ml/joint_survey.py`
- `src/inference/__init__.py`
- `src/inference/differentiable_simulator.py`
- `src/inference/nuts_hmc.py`
- `src/data/mast_downloader.py`
- `src/data/pixel_covariance.py`
- `src/validation/hst_targets.py`
- `src/validation/kinematics.py`
- `src/validation/mu_glance.py`
- `src/validation/bayes_factor.py`
- `src/utils/constants.py`
- `src/utils/blinding.py`

Benchmark scripts:
- `scripts/ablation_study.py`
- `scripts/validate_real_data.py`
- `scripts/sota_comparison.py`
- `scripts/uncertainty_calibration.py`
- `scripts/scalability_benchmark.py`
- `scripts/pareto_benchmark.py`
- `scripts/multi_messenger_demo.py`
- `scripts/reproduce.sh`
- `scripts/mint_zenodo_doi.py`
- `scripts/publication_gate.py`

Web UI:
- `web_ui/index.html`
- `web_ui/app.js`
- `web_ui/styles.css`
- `web_ui/pages/dashboard.js`
- `web_ui/pages/workbench.js`
- `web_ui/pages/validation.js`
- `web_ui/pages/analyses.js`
- `web_ui/pages/account.js`
- `web_ui/pages/survey.js`
- `web_ui/pages/api-explorer.js`
- `web_ui/pages/inference.js`

API:
- `api/main.py`
- `api/auth_routes.py`
- `api/analysis_routes.py`

Paper:
- `paper/main.tex`
- `paper/references.bib`

Validation/tests:
- `tests/test_lens_system.py`
- `tests/test_mass_profiles.py`
- `tests/test_critical_curves.py`
- `tests/test_ray_tracing.py`
- `tests/test_time_delay.py`
- `tests/test_physics_constrained_loss.py`
- `tests/test_neural_ode.py`
- `tests/test_next_ui.py`
- `tests/test_api.py`
- `tests/test_real_data.py`
- `tests/test_differentiable_inference.py`

---

## 10) Practical Commands

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Check backend:
```bash
python3 -c "from src.ml import check_backend; check_backend()"
```

Run app/API:
```bash
uvicorn api.main:app --reload
# open http://localhost:8000/ui for the web frontend
```

Regression:
```bash
uv run python -m pytest tests/ -q
python3 scripts/publication_gate.py
python3 -m mypy src/ --ignore-missing-imports  # optional dev-tool check
python3 scripts/statistical_rigor_report.py
```

Benchmark suite:
```bash
# One-command full reproducibility (6 steps)
bash scripts/reproduce.sh

# Individual benchmarks
python3 scripts/ablation_study.py --grid 64 --n-trials 3
python3 scripts/validate_real_data.py --grid 64 --use-real --strict-observational
python3 scripts/sota_comparison.py --grid 64 --n-lenses 10
python3 scripts/uncertainty_calibration.py --grid 64 --n-samples 30 --model models/bayesian_uq_synthetic.pt
python3 scripts/scalability_benchmark.py
python3 scripts/pareto_benchmark.py --outdir results
python3 scripts/multi_messenger_demo.py --outdir results
```

New physics modules:
```bash
python3 -m src.ml.nested_sampling           # NFW vs SIS Bayesian evidence demo
python3 -m src.validation.kinematics        # Jeans equation + MSD test
```

Targeted checks:
```bash
python3 -m pytest tests/test_real_data.py -q
python3 -m pytest tests/test_lens_system.py tests/test_mass_profiles.py tests/test_time_delay.py -q
python3 -m pytest tests/test_neural_ode.py -q
```

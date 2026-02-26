# AGENTS.md

Operational context for humans and coding agents working in this repository.
Use this as the working source-of-truth for architecture, validation workflow,
and current branch/release state.

Last updated: 2026-02-26

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
- Python: `3.9.6`
- JAX backend: auto-detected via `src/ml/__init__.BACKEND`

Verified on 2026-02-26:

```bash
python3 -m pytest tests/ -q
# 481 passed, 35 skipped   (run Feb 26 2026)

python3 -m mypy src/ --ignore-missing-imports
# Success: no issues found in 50+ source files
```

Additional check:
- `LensSystem` defaults match Planck constants (`H0=67.4`, `Om0=0.315`).
- NFW profile uses `critical_density(z_l)` at lens redshift (M200c convention).

Expected non-fatal local warning:
- `urllib3` warns about LibreSSL/OpenSSL mismatch in this Python build.

---

## 3) Architecture (Current)

### Scientific core (`src/`)
- `src/lens_models/`: lens geometry, mass profiles, multi-plane models.
  - `mass_profiles.py`: SIS, NFW, power-law profiles. NFW uses deterministic RNG
    via hashed physical params (`hashlib.sha256` seed from M_vir, c, z_l, z_s).
  - `multi_plane.py`: Multi-plane ray-tracing with proper recurrence weights
    (`D_{i,i+1} / D_{i+1}`) and `lensing_potential` / `potential` interface.
- `src/optics/`: ray tracing, geodesics, wave optics.
  - `wave_optics.py`: Diffraction integral F(ω) with Nakamura & Deguchi (1999)
    and Takahashi & Nakamura (2003) equation provenance.
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
  - `lens_finder.py`: LenNet-style object-detection head for automated discovery in wide-field FITS.
  - `joint_survey.py`: Multi-resolution likelihood engine for joint ground+space deblending.
- `src/data/`: FITS/real-data loading and preprocessing.
  - `mast_downloader.py`: Three-tier HST/ACS download (cache → MAST → synthetic FITS).
    Deterministic noise via `hashlib.sha256` seed.
  - `pixel_covariance.py`: Drizzled image correlated pixel covariance and Cholesky whitening.
- `src/validation/`: calibration and scientific validation routines.
  - `hst_targets.py`: SLACS lens catalog and HST data loader.
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

### API + persistence
- API entry: `api/main.py`
- Routers: `api/auth_routes.py`, `api/analysis_routes.py`
- Database layer: `database/`
- Alembic setup: `migrations/`, `alembic.ini`

### Benchmark scripts (`scripts/`)
- `scripts/ablation_study.py`: Proxy sensitivity ablation (6 configs; controlled perturbation surrogates) → `results/ablation_table.tex`
- `scripts/validate_real_data.py`: SLACS validation with `--use-real`; enforce all-observational mode via `--strict-observational` → `results/real_data/`
- `scripts/sota_comparison.py`: Proxy SOTA sensitivity benchmark (4 surrogate methods) → `results/sota_comparison_table.tex`
- `scripts/uncertainty_calibration.py`: Reliability diagram, ECE, coverage → `results/uncertainty_calibration.png`
- `scripts/scalability_benchmark.py`: Grid scaling (16→512) → `results/scalability_analysis.png`
- `scripts/pareto_benchmark.py`: **Time-to-Solution vs κ-RMSE Pareto front** with MCMC reference → `results/pareto_front.png`, `results/pareto_table.tex`
- `scripts/multi_messenger_demo.py`: **Optical + GW multi-messenger consistency** → `results/multi_messenger_consistency.png`
- `scripts/mint_zenodo_doi.py`: Zenodo DOI minting CLI
- `scripts/reproduce.sh`: One-command reproducibility (6 steps)
- `scripts/publication_gate.py`: Executable publication gate
- `scripts/statistical_rigor_report.py`: Statistical rigor report generator
- Proxy scripts include explicit mode metadata in outputs (`evaluation_mode`, `prediction_mode`) for downstream rigor checks.

### Paper (`paper/`)
- `paper/main.tex`: IEEE TCI manuscript (15 numbered equations, 23 BibTeX refs)
- `paper/references.bib`: Bibliography with 2024-2025 citations
- `paper/TIER1_TOPIC_AND_RIGOR.md`: Topic and rigor framing artifact

### Quality + ops
- Tests: `tests/` (481 passing tests, 35 skipped — verified Feb 26 2026)
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
| `/api/v1/synthetic` | POST | Workbench | Generate NFW convergence map |
| `/api/v1/inference` | POST | Workbench | PINN inference + MC Dropout UQ |
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
| `/docs` (OpenAPI) | GET | API Explorer | Schema-driven request builder |

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
- Functions fall back to `None` when JAX is unavailable (not `ImportError` at import time).

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
- MC Dropout uncertainty: T=30 forward passes, p=0.1 dropout rate.

### Bayesian model selection
- Nested sampling in `src/ml/nested_sampling.py` (Skilling 2004).
- `compute_bayes_factor()` with Jeffreys scale interpretation.
- Built-in NFW vs SIS demo (correctly identifies NFW with ln K ≈ 13.9, decisive evidence).

### Stellar kinematics
- Jeans equation solver in `src/validation/kinematics.py`.
- Mass-sheet degeneracy test: λ = M_lens / M_kin, Jeffreys-style interpretation.
- Joint lensing+kinematics constraint with inverse-variance weighting.

### MAST data pipeline
- `src/data/mast_downloader.py`: three-tier strategy (cache → MAST → synthetic FITS).
- SLACS catalog: 5 lenses with RA/Dec, proposal IDs (10886, 10494), velocity dispersions.
- Deterministic noise via `hashlib.sha256` seed from lens name.
- Used by `scripts/validate_real_data.py --use-real`.

### Augmentation pipeline (important physics constraint)
- `RandomBrightness` and `RandomNoise` clip to `max(0, x)` ONLY — κ ≥ 0 only.
- Upper-bound clip removed: κ can exceed 1.0 in massive halo cores (galaxy clusters κ ~ 2–5).

### Uncertainty calibration
- `scripts/uncertainty_calibration.py` now requires `--model <checkpoint.pt>` for
  publication-valid MC Dropout results.  Without `--model`, it runs in smoke-test mode
  with a `UserWarning` that explicitly marks results as invalid for publication.

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
- If built-in demo assets are missing from `assets/demos/`, app fallback generates
  physics-based synthetic observation from config instead of hard-failing.

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
python3 -m mypy src/ --ignore-missing-imports
python3 scripts/publication_gate.py --quick
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
- `scripts/validation_gate.py`

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
- `tests/test_ray_tracing.py`
- `tests/test_time_delay.py`
- `tests/test_physics_constrained_loss.py`
- `tests/test_neural_ode.py`
- `tests/test_next_ui.py`
- `tests/test_api.py`
- `tests/test_real_data.py`

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
python3 -m mypy src/ --ignore-missing-imports
python3 scripts/publication_gate.py
python3 scripts/statistical_rigor_report.py
```

Benchmark suite:
```bash
# One-command full reproducibility (6 steps)
bash scripts/reproduce.sh

# Individual benchmarks
python3 scripts/ablation_study.py --grid 64 --n-trials 3
python3 scripts/validate_real_data.py --grid 64 --use-real
python3 scripts/sota_comparison.py --grid 64 --n-lenses 10
python3 scripts/uncertainty_calibration.py --grid 64 --n-samples 30
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

# AGENTS.md

This file is the operational context for humans/agents working in this repository.
Use this as source-of-truth for architecture, runtime, validation workflow, and known caveats.

Last updated: 2026-02-23 (local workspace snapshot)

---

## 1) Project Identity

- Project: **Gravitational Lensing Toolkit (ISEF 2025)**
- Primary repo URL: `https://github.com/nalin1304/Gravitational-Lensing-algorithm`
- Local workspace path:
  - `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master`
- Runtime observed in this workspace:
  - `Python 3.9.6`

Important workspace note:
- This local copy currently has **no `.git` metadata** (commands like `git status` fail). Treat it as a filesystem working copy unless re-cloned as a git repo.

---

## 2) Top-Level Architecture

### Core scientific library
- `src/lens_models/`
  - Cosmology/lens geometry, mass profiles (PointMass/NFW/etc), multi-plane lensing.
- `src/optics/`
  - Ray tracing, geodesic integration, wave optics.
- `src/time_delay/`
  - Fermat potential + time-delay cosmography workflows.
- `src/ml/`
  - PINN models, physics-constrained losses, synthetic dataset generation, uncertainty tools.
- `src/data/`
  - FITS/real-data loading and preprocessing support.
- `src/validation/`
  - Scientific validation, calibration, known-system checks.
- `src/utils/`
  - Constants and shared scientific utilities.
- `src/api_utils/`
  - Shared API/auth helper logic.

### Web UI (Streamlit)
- Entry: `app/Home.py`
- Pages: `app/pages/*.py`
- Shared UI/helpers/state:
  - `app/utils/ui.py`
  - `app/utils/plotting.py`
  - `app/utils/helpers.py`
  - `app/utils/session_state.py`
  - `app/utils/demo_helpers.py`

### REST API (FastAPI)
- Entry: `api/main.py`
- Additional routers:
  - `api/auth_routes.py`
  - `api/analysis_routes.py`
- Monitoring/security helper modules in `api/`.

### Persistence / infra
- `database/`: models, CRUD, auth, DB session logic.
- `migrations/` + `alembic.ini`: Alembic migration setup.
- `docker-compose.yml`: local stack orchestration.

### Tests / Benchmarks / Scripts
- `tests/`: broad scientific + API + UI/backend utility coverage.
- `benchmarks/`: performance baselines and reporting (incl. `benchmarks/pinn_results.json`).
- `scripts/`: db init, checks, quick demos, validator/integration scripts.
- `notebooks/`: phase demos and exploratory notebooks.

---

## 3) Canonical Entry Points

### Streamlit UI
```bash
streamlit run app/Home.py
```

Notes:
- `app/main.py` is a deprecated entrypoint stub. Use `app/Home.py`.
- Current page set includes:
  - `app/pages/02_Simple_Lensing.py`
  - `app/pages/03_PINN_Inference.py`
  - `app/pages/03_Results.py`
  - `app/pages/04_Multi_Plane.py`
  - `app/pages/05_Real_Data.py`
  - `app/pages/06_Training.py`
  - `app/pages/07_Validation.py`
  - `app/pages/08_Bayesian_UQ.py`
  - `app/pages/09_Settings.py`

### FastAPI
```bash
uvicorn api.main:app --reload
```

Primary endpoints in `api/main.py`:
- `GET /`
- `GET /health`
- `POST /api/v1/synthetic`
- `POST /api/v1/inference`
- `POST /api/v1/batch`
- `GET /api/v1/batch/{batch_id}/status`
- `GET /api/v1/models`
- `GET /api/v1/stats`

Additional routers:
- Auth: `/api/v1/auth/*` in `api/auth_routes.py`
- Analysis/job CRUD: `/api/v1/*` in `api/analysis_routes.py`

### Docker stack
```bash
docker-compose up -d
```

---

## 4) Scientific Contracts Implemented in Code

These are the key physics contracts currently encoded in core modules and tests.

### Cosmology + distances
- `LensSystem` in `src/lens_models/lens_system.py` uses `astropy.cosmology.FlatLambdaCDM`.
- Distances are computed via:
  - `angular_diameter_distance(z_l)`
  - `angular_diameter_distance(z_s)`
  - `angular_diameter_distance_z1z2(z_l, z_s)` for `D_ls`.
- Critical surface density:
  - `Sigma_crit = c^2/(4*pi*G) * D_s/(D_l*D_ls)` (implemented in `critical_surface_density()`).

### Einstein radius
- Point-mass Einstein scale in `LensSystem.einstein_radius_scale()`:
  - `theta_E = sqrt((4GM/c^2) * D_ls/(D_l*D_s))`.

### NFW + related profiles
- Main implementation in `src/lens_models/mass_profiles.py`.
- Elliptical extension in `src/lens_models/advanced_profiles.py`.
- Deflection/convergence kernels are implemented with dedicated NFW helper functions and projected-profile terms.

### Time-delay cosmography
- Implemented in `src/time_delay/cosmography.py`.
- Uses Fermat-potential decomposition and:
  - `Delta t ∝ (1+z_l) * (D_l*D_s/D_ls) * Delta phi`.

### PINN / physics losses
- Core model in `src/ml/pinn.py` and related models in `src/ml/pinn_models.py` / `src/ml/pinn_advanced.py`.
- Physics constraints in `src/ml/physics_constrained_loss.py`.
- Unit-safe helper routines in `src/ml/physics_unit_safe.py`.

### Constants
- Central constants in `src/utils/constants.py`:
  - CODATA 2018 + Planck 2018 values and conversion helpers.

---

## 5) Known Compatibility Layers (Do Not Remove Blindly)

- `app/utils.py` is legacy; `app/utils/__init__.py` bridges legacy symbols for compatibility.
- `src/lens_models.py` exists alongside package `src/lens_models/`; most active code/tests use the package path (`src.lens_models.*`).
- API and app code include fallback import paths to support multiple launch contexts.

When refactoring:
- Prefer canonical package imports (`src.lens_models`, `app.utils.*`).
- Preserve backward-compatible re-exports unless all call sites are migrated and tests pass.

---

## 6) Current Validation Baseline (This Workspace)

Executed in this workspace on 2026-02-23:

```bash
python3 -m pytest tests/ -q
```

Result:
- `551 passed, 22 skipped`

Also validated:

```bash
python3 -m mypy src/ --ignore-missing-imports
```

Result:
- `Success: no issues found in 42 source files`

Common non-fatal warnings seen:
- `urllib3` OpenSSL/LibreSSL warning in local Python build.
- Some expected deprecation/user warnings in selected tests.

---

## 7) Known Caveats / Drift to Watch

1. README and docs drift
- Some README claims (e.g., historical test counts, some page names, some links/text artifacts) may not match current code state.
- Use code + tests as primary source of truth.

2. Demo assets dependency
- `app/utils/demo_helpers.py` expects built-in assets under:
  - `assets/demos/*.npy`
- In this workspace snapshot, `assets/` is absent.
- One-click demo flows using `builtin:*` source images can fail until assets are added.

3. Cosmology defaults are not globally unified
- `src/utils/constants.py` defines Planck 2018 constants (`H0_PLANCK=67.4`, `OMEGA_M_PLANCK=0.315`).
- `LensSystem` defaults are currently `H0=70.0`, `Om0=0.3`.
- Be explicit about cosmology values in scientific comparisons.

4. API synthetic request compatibility
- `scale_radius` remains in API/data-generation signatures for compatibility even where current NFW constructors derive scale from mass+concentration.

5. Python version considerations
- Project requires `>=3.9` in `pyproject.toml`.
- Avoid Python-only syntax that requires newer runtimes unless guarded or backported.

---

## 8) Agent Workflow Guidance

When making changes:

1. Prefer code reality over docs
- Verify behavior from source + tests.

2. Keep imports/package boundaries stable
- Avoid introducing top-level import ambiguity (especially lens model modules).

3. Preserve scientific consistency
- Use existing constants/utilities where available.
- Keep units explicit and convert intentionally.

4. Validate incrementally
- Minimum:
  - `python3 -m py_compile <changed_files>`
  - targeted pytest modules for touched subsystems
- Before handoff:
  - `python3 -m pytest tests/ -q`
  - `python3 -m mypy src/ --ignore-missing-imports` for core changes

5. UI changes
- Confirm Streamlit page routes exist before switching.
- Avoid debug artifacts (`st.write("Debug: ...")`) in production pages.

6. Data realism
- Do not introduce synthetic stand-in data for scientific outputs.
- If synthetic generation is required, keep deterministic seeds and document assumptions.

---

## 9) High-Signal File Map (Start Here)

- Scientific core:
  - `src/lens_models/lens_system.py`
  - `src/lens_models/mass_profiles.py`
  - `src/lens_models/advanced_profiles.py`
  - `src/optics/ray_tracing.py`
  - `src/time_delay/cosmography.py`
  - `src/ml/pinn.py`
  - `src/ml/physics_constrained_loss.py`
  - `src/utils/constants.py`

- App/UI:
  - `app/Home.py`
  - `app/pages/02_Simple_Lensing.py`
  - `app/pages/03_PINN_Inference.py`
  - `app/pages/03_Results.py`
  - `app/pages/05_Real_Data.py`
  - `app/utils/demo_helpers.py`

- API:
  - `api/main.py`
  - `api/auth_routes.py`
  - `api/analysis_routes.py`

- Validation/test coverage:
  - `tests/test_lens_system.py`
  - `tests/test_mass_profiles.py`
  - `tests/test_ray_tracing.py`
  - `tests/test_time_delay.py`
  - `tests/test_physics_constrained_loss.py`
  - `tests/test_web_interface.py`
  - `tests/test_api.py`
  - `tests/test_real_data.py`

---

## 10) Practical Command Reference

Setup:
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -r requirements-dev.txt
```

Run:
```bash
streamlit run app/Home.py
uvicorn api.main:app --reload
```

Quality checks:
```bash
python3 -m py_compile app/Home.py
python3 -m pytest tests/ -q
python3 -m mypy src/ --ignore-missing-imports
```

Targeted checks:
```bash
python3 -m pytest tests/test_web_interface.py tests/test_real_data.py -q
python3 -m pytest tests/test_lens_system.py tests/test_mass_profiles.py tests/test_time_delay.py -q
```

---

If this file becomes stale, refresh it from code + test execution logs, not from documentation claims.

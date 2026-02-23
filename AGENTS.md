# AGENTS.md

Operational context for humans and coding agents working in this repository.
Use this as the working source-of-truth for architecture, validation workflow,
and current branch/release state.

Last updated: 2026-02-23

---

## 1) Project Identity

- Project: **Gravitational Lensing Toolkit (ISEF 2025)**
- Upstream repo: `https://github.com/nalin1304/Gravitational-Lensing-algorithm`
- Local workspace:
  - `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master`

Current VCS state in this workspace:
- Git repo initialized and connected to `origin`
- Active branch: `codex/publication-ready-r2`
- Head commit: `a040d40`
- Open PR: `https://github.com/nalin1304/Gravitational-Lensing-algorithm/pull/3`

---

## 2) Runtime + Verified Baseline

Observed local runtime:
- Python: `3.9.6`

Verified on 2026-02-23:

```bash
python3 -m pytest tests/ -q
# 551 passed, 22 skipped

python3 -m mypy src/ --ignore-missing-imports
# Success: no issues found in 42 source files
```

Additional check:
- `LensSystem` defaults match Planck constants (`H0=67.4`, `Om0=0.315`).

Expected non-fatal local warning:
- `urllib3` warns about LibreSSL/OpenSSL mismatch in this Python build.

---

## 3) Architecture (Current)

### Scientific core (`src/`)
- `src/lens_models/`: lens geometry, mass profiles, multi-plane models.
- `src/optics/`: ray tracing, geodesics, wave optics.
- `src/time_delay/`: Fermat potential and delay cosmography.
- `src/ml/`: PINN models, training, physics losses, uncertainty.
- `src/data/`: FITS/real-data loading and preprocessing.
- `src/validation/`: calibration and scientific validation routines.
- `src/utils/`: constants + shared scientific helpers.
- `src/api_utils/`: auth/API support utilities.

### Streamlit app (`app/`)
- Entry: `app/Home.py`
- Deprecated redirect entrypoint: `app/main.py`
- Feature pages: `app/pages/*.py`
- Reusable non-UI app logic:
  - `app/core/landing.py`
  - `app/core/web_utils.py`
- Shared UI/session/demo logic:
  - `app/utils/ui.py`
  - `app/utils/session_state.py`
  - `app/utils/demo_helpers.py`
  - `app/utils/plotting.py`
  - `app/utils/helpers.py`

### API + persistence
- API entry: `api/main.py`
- Routers: `api/auth_routes.py`, `api/analysis_routes.py`
- Database layer: `database/`
- Alembic setup: `migrations/`, `alembic.ini`

### Quality + ops
- Tests: `tests/`
- Benchmarks: `benchmarks/`
- Utility scripts: `scripts/`
- Validation/readiness docs:
  - `JOURNAL_PUBLICATION_READINESS.md`
  - `PROJECT_DOCUMENTATION.md`

---

## 4) Canonical Import and Module Rules

1. Use package imports from canonical modules.
- Preferred: `from src.lens_models import LensSystem, NFWProfile, ...`
- Avoid introducing new top-level import paths.

2. Treat `app/core/` as the location for reusable, testable app logic.
- Business/scientific helper logic belongs in `app/core/*`.
- Streamlit orchestration should remain in `app/pages/*` and `app/Home.py`.

3. Compatibility wrappers are intentional.
- `app/utils.py` re-exports from `app/core/web_utils.py`.
- `app/styles.py` re-exports shared UI helpers from `app/utils/ui.py`.
- `src/lens_models.py` is a legacy compatibility shim; canonical implementations are under `src/lens_models/` package.

4. Do not remove compatibility wrappers unless all call sites are migrated and full tests pass.

---

## 5) Scientific Contracts (Implemented)

### Cosmology and distances
- `LensSystem` in `src/lens_models/lens_system.py` uses `astropy.cosmology.FlatLambdaCDM`.
- Defaults are Planck-aligned via `src/utils/constants.py`.
- Distances use:
  - `angular_diameter_distance(z_l)`
  - `angular_diameter_distance(z_s)`
  - `angular_diameter_distance_z1z2(z_l, z_s)` for `D_ls`

### Core lensing quantities
- Critical surface density:
  - `Sigma_crit = c^2/(4*pi*G) * D_s/(D_l*D_ls)`
- Einstein scale:
  - `theta_E = sqrt((4GM/c^2) * D_ls/(D_l*D_s))`

### Time delays
- Delay relation implemented in `src/time_delay/cosmography.py`:
  - `Delta t ∝ (1+z_l) * (D_l*D_s/D_ls) * Delta phi`

### PINN physics
- Core model and inference in `src/ml/pinn.py` and related modules.
- Physics-constrained losses in `src/ml/physics_constrained_loss.py`.

---

## 6) Demo/Data Behavior

- One-click demos can request `builtin:*` assets in YAML configs.
- If built-in demo assets are missing from `assets/demos/`, app fallback now generates a physics-based synthetic observation from config instead of hard-failing.
- Keep demo generation deterministic when randomness is used (seeded RNG).

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
python3 -m pytest tests/ -q
python3 -m mypy src/ --ignore-missing-imports
```

For UI-focused work, also run:
```bash
python3 -m pytest tests/test_web_interface.py -q
```

---

## 8) Coding and Organization Conventions

1. Keep modules single-purpose.
- UI rendering and scientific logic should remain separated.

2. Prefer explicit names over abbreviations.
- Use domain terms (`convergence_map`, `deflection_field`, `einstein_radius_arcsec`) where practical.

3. Keep docs and architecture synchronized.
- If you add/move core modules, update:
  - `README.md`
  - `app/README.md`
  - this `AGENTS.md`

4. Avoid placeholder/dummy data in runtime paths.
- Placeholder terms may exist only in explanatory documentation of scans/results.

---

## 9) High-Signal File Map

Scientific core:
- `src/lens_models/lens_system.py`
- `src/lens_models/mass_profiles.py`
- `src/lens_models/advanced_profiles.py`
- `src/lens_models/multi_plane_recursive.py`
- `src/optics/ray_tracing.py`
- `src/time_delay/cosmography.py`
- `src/ml/pinn.py`
- `src/ml/physics_constrained_loss.py`
- `src/utils/constants.py`

App core/UI:
- `app/Home.py`
- `app/core/landing.py`
- `app/core/web_utils.py`
- `app/utils/ui.py`
- `app/utils/demo_helpers.py`
- `app/pages/03_Results.py`
- `app/pages/05_Real_Data.py`

API:
- `api/main.py`
- `api/auth_routes.py`
- `api/analysis_routes.py`

Validation/tests:
- `tests/test_lens_system.py`
- `tests/test_mass_profiles.py`
- `tests/test_ray_tracing.py`
- `tests/test_time_delay.py`
- `tests/test_physics_constrained_loss.py`
- `tests/test_web_interface.py`
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

Run app/API:
```bash
streamlit run app/Home.py
uvicorn api.main:app --reload
```

Regression:
```bash
python3 -m pytest tests/ -q
python3 -m mypy src/ --ignore-missing-imports
```

Targeted checks:
```bash
python3 -m pytest tests/test_web_interface.py tests/test_real_data.py -q
python3 -m pytest tests/test_lens_system.py tests/test_mass_profiles.py tests/test_time_delay.py -q
```

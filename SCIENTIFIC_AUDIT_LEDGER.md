# Scientific Code Audit Ledger (No Security Findings)

Scope: all `.py` + notebook code cells under project root.

Method evidence: `python3 -m py_compile`, full/per-file pytest runs, import-matrix checks, static line inspection, notebook-cell scans.

## Findings (Ordered by Severity)

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/auth_routes.py:43`
  - Issue: Imports `ACCESS_TOKEN_EXPIRE_MINUTES` from `database`, but `database/__init__.py` does not export it.
  - Evidence: `python3 - <<... import api.auth_routes ...>>` raises `ImportError` at this import.
  - Minimal fix direction: Export the constant from `database/__init__.py` or import directly from `database.auth`.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/main.py:39`
  - Issue: Missing symbol import: `generate_synthetic_convergence` is imported from `src.ml.generate_dataset` but not defined there.
  - Evidence: `python3 -m pytest -q tests/test_api.py` fails during collection with `ImportError` at this line.
  - Minimal fix direction: Import from the actual provider (or implement/export the function in `src/ml/generate_dataset.py`) and align return contract.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils.py:19`
  - Issue: Defines core utility functions that are unreachable through `import app.utils` due to package/module name collision (`app/utils/` wins).
  - Evidence: Benchmarks importing from `app.utils` fail to resolve these functions.
  - Minimal fix direction: Consolidate to one import path: either keep package and move this file into it, or rename this module.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/__init__.py:7`
  - Issue: Namespace collision impact: package `app.utils` exports only session state helpers, but callers expect `generate_synthetic_convergence`, `load_pretrained_model`, and `prepare_model_input`.
  - Evidence: `import app.utils` resolves to `app/utils/__init__.py`; required symbols are absent (`False False False` in runtime check).
  - Minimal fix direction: Either re-export the utility functions here or remove `app/utils.py` module-vs-package ambiguity.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/demo_helpers.py:253`
  - Issue: Imports nonexistent `RayTracingBackend` from ray-tracing backends module.
  - Evidence: Calling `full_analysis_pipeline()` raises `ImportError` at this line.
  - Minimal fix direction: Use available API (`ray_trace`/dual-mode functions) or implement backend class.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/demo_helpers.py:254`
  - Issue: Imports nonexistent `PINN` class from `src.ml.pinn`.
  - Evidence: `src/ml/pinn.py` defines `PhysicsInformedNN`, not `PINN`.
  - Minimal fix direction: Use `PhysicsInformedNN` (or real intended model class) and align constructor usage.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/comparisons.py:22`
  - Issue: Imports non-exported symbols from `app.utils`, causing module import failure.
  - Evidence: `python3 - <<... import benchmarks.comparisons ...>>` raises `ImportError`.
  - Minimal fix direction: Fix import source after utilities namespace is canonicalized.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/mass_profiles.py:1252`
  - Issue: Uses `np.trapezoid`, unavailable under declared NumPy baseline (`>=1.24`).
  - Evidence: Runtime check: NumPy `1.26.2`, `hasattr(np,"trapezoid") == False`; failing tests in `tests/test_alternative_dm.py`.
  - Minimal fix direction: Use `np.trapz` for NumPy 1.x compatibility or bump required NumPy major version and enforce it.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/time_delay/cosmography.py:338`
  - Issue: Posterior normalization uses `np.trapezoid` (NumPy 2 API) but requirements allow NumPy 1.x.
  - Evidence: `tests/test_time_delay.py` fails repeatedly at this line with `AttributeError`.
  - Minimal fix direction: Use `np.trapz` or version-gated integration helper.

- **Critical | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/time_delay/cosmography.py:481`
  - Issue: Outlier-cleaning mask can produce an empty sample (`h0_std_robust==0` gives strict `<` false everywhere), leading to percentile crash.
  - Evidence: Fail trace in Monte Carlo tests: `IndexError` at percentile on empty array.
  - Minimal fix direction: Add non-empty guard and non-strict fallback mask (`<=`) or bypass clipping when robust std is zero.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/demo_helpers.py:282`
  - Issue: Constructs `LensSystem` with unsupported keyword signature (`mass`, `lens_model`, etc.).
  - Evidence: `src/lens_models/lens_system.py` constructor expects only `(z_lens, z_source, H0, Om0)`.
  - Minimal fix direction: Instantiate `LensSystem` correctly and move mass/profile settings to proper profile objects.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/demo_helpers.py:304`
  - Issue: Calls `lens_system.deflection`/`convergence` methods that do not exist on `LensSystem`.
  - Evidence: `LensSystem` class exposes cosmology/distance utilities, not lens field methods.
  - Minimal fix direction: Route through mass-profile or ray-tracing API to compute lensing observables.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/comparisons.py:91`
  - Issue: Return-shape contract mismatch: expects `(map, coords)` but generator returns `(map, X, Y)`.
  - Evidence: `generate_synthetic_convergence` in `app/utils.py` returns 3 values; this call unpacks 2 and treats second as dict.
  - Minimal fix direction: Standardize one return schema and update all callers.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/comparisons.py:287`
  - Issue: Second return-shape mismatch repeats (`test_map, _ = ...`) and will break once import path is fixed.
  - Evidence: Static review of callsite vs provider signature.
  - Minimal fix direction: Align unpacking with provider return values and naming.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/database/__init__.py:105`
  - Issue: Public export surface is incomplete for downstream API usage (missing `ACCESS_TOKEN_EXPIRE_MINUTES`; no `Session` alias expected by API imports).
  - Evidence: Static import check reports missing symbols consumed by `api/auth_routes.py` and `api/main.py`.
  - Minimal fix direction: Export required symbols or adjust all import sites to use exported names only.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase3_dm_comparison.ipynb:38`
  - Issue: Imports nonexistent module path `from src.ray_tracing import RayTracer`.
  - Evidence: No `src/ray_tracing.py` exists; ray tracing lives under `src/optics`.
  - Minimal fix direction: Update notebook imports to current package structure and validate notebook execution end-to-end.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/generate_dataset.py:1`
  - Issue: Module lacks `generate_synthetic_convergence` despite being imported by API as provider.
  - Evidence: Runtime introspection: only `generate_convergence_map*` functions exist.
  - Minimal fix direction: Add canonical generator function or change API to import from the correct module.

- **High | Confidence: Medium**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/pinn.py:303`
  - Issue: Physics residual uses small-z distance approximation `D ≈ cz/H0` even with source redshift up to ~2, introducing cosmology bias in constraint term.
  - Evidence: Distance equations are simplified comments/code; no FLRW distance integral.
  - Minimal fix direction: Use astropy/consistent FLRW distances or precomputed lookup for differentiable approximation with quantified error.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/pinn.py:516`
  - Issue: Loss reporting contract mismatch: `total` includes regularization via `physics_loss`, but reported decomposition names only `physics_residual`; this breaks metric accounting and tests.
  - Evidence: `tests/test_ml.py::test_loss_components_contribution` fails (difference equals unaccounted regularization term).
  - Minimal fix direction: Expose full decomposition (`physics_residual`, `regularization`, `physics_total`) and ensure docs/tests match formula.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/time_delay/cosmography.py:269`
  - Issue: Model cloning for H0 scan uses `type(lens_model)(M,c,lens_sys)` then mutates WDM/SIDM attrs post-init; profile-specific physics is not re-derived for sampled cosmology.
  - Evidence: Code sets `temp_lens.m_wdm` / `temp_lens.sigma_SIDM` after constructor instead of constructing with those params.
  - Minimal fix direction: Instantiate profile with full parameter signature or add dedicated clone/copy constructor.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/utils/visualization.py:357`
  - Issue: Uses top-level import `from optics.ray_tracing ...`, which fails in package context where module is under `src.optics`.
  - Evidence: Static import matrix reports `ModuleNotFoundError: No module named optics`.
  - Minimal fix direction: Use relative import (`from ..optics...`) or absolute `from src.optics...` consistently.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/calibration.py:280`
  - Issue: Einstein-radius estimator uses first radius where `kappa < 1`; for SIS (`kappa = θE/(2r)`) this yields `θE/2`, a systematic underestimation.
  - Evidence: `tests/test_calibration.py` recovers `0.75` for true `1.5` and fails tolerance.
  - Minimal fix direction: Estimate θE from physically consistent criterion (e.g., tangential critical curve / enclosed mass relation) instead of κ=1 crossing heuristic.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/calibration.py:317`
  - Issue: Mass estimate uses `mean_kappa` over `convergence > 0.1` across map, biasing mass by threshold and FoV.
  - Evidence: Calibration factors exceed expected range in tests (`radius_calibration_factor` > 2).
  - Minimal fix direction: Compute enclosed mass inside explicit aperture tied to θE with robust masking and background treatment.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_multi_plane_recursive.py:28`
  - Issue: Imports `lens_models.multi_plane_recursive` as top-level package, inconsistent with `src` namespace.
  - Evidence: Collection error when test is run under standard project root.
  - Minimal fix direction: Use `src.lens_models.multi_plane_recursive` import path.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_physics_constrained_loss.py:23`
  - Issue: Imports `ml.physics_constrained_loss` as top-level package; incompatible with project package layout (`src.ml`).
  - Evidence: Pytest collection error: `ModuleNotFoundError: No module named ml`.
  - Minimal fix direction: Use `from src.ml...` imports consistently in tests.

- **High | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_ray_tracing_backends.py:20`
  - Issue: Imports `optics.*` and `lens_models.*` as top-level packages; incompatible with `src.*` layout.
  - Evidence: Collection error due attempted relative import beyond top-level module resolution.
  - Minimal fix direction: Normalize test imports to `src.optics...` and `src.lens_models...`.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/runner.py:21`
  - Issue: Transitively non-importable because it imports `benchmarks.comparisons` which currently fails import.
  - Evidence: Static import matrix shows runner import failure chain.
  - Minimal fix direction: Fix upstream `benchmarks/comparisons.py` import contract.

- **Medium | Confidence: Medium**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/visualization.py:1`
  - Issue: Transitively blocked by benchmark package import failure; visualization module cannot be reliably used in benchmark workflows.
  - Evidence: `benchmarks.__init__` import chain fails before visualization consumers run.
  - Minimal fix direction: Decouple visualization imports from failing benchmark package init or fix package init imports.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/test_validator.py:255`
  - Issue: Imports nonexistent `calculate_ssim` from `benchmarks.metrics`.
  - Evidence: `benchmarks/metrics.py` exposes `calculate_structural_similarity` instead.
  - Minimal fix direction: Update import to the real function name and adapt callsites.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/dark_matter/substructure.py:182`
  - Issue: `_smooth_model_prediction` is a stand-in returning ones, so anomaly features are not physically anchored to lens model predictions.
  - Evidence: Method docstring marks stand-in; output is constant ones.
  - Minimal fix direction: Integrate actual smooth lens prediction model or flag output as synthetic-only.

- **Medium | Confidence: Medium**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models.py:1`
  - Issue: Legacy standalone module duplicates the `src/lens_models/` package and contains stale interfaces, creating ambiguity and drift risk.
  - Evidence: Repository has both `src/lens_models.py` and `src/lens_models/` package.
  - Minimal fix direction: Remove or archive legacy module and keep a single authoritative lens-model implementation.

- **Medium | Confidence: Medium**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/mass_profiles.py:1241`
  - Issue: `validate_mass_conservation` integrates up to `r_max = 3*r_s` (angular units) while expected-mass comparison may be virial/full-mass, creating model-dependent bias in conservation check.
  - Evidence: Code compares integrated surface-density mass to `enclosed_mass(r_max)`/`mass` with mixed profile semantics.
  - Minimal fix direction: Explicitly harmonize integrated aperture mass definition and expected mass definition per profile.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/generate_dataset.py:19`
  - Issue: Path mutation (`sys.path.append("..")`) plus relative imports creates environment-dependent import behavior and clashes with test imports (`ml.*`).
  - Evidence: Collection error in `tests/test_physics_constrained_loss.py`: relative import beyond top-level package when imported as `ml`.
  - Minimal fix direction: Remove path hacks; enforce one package import convention (`src.*`) and update tests accordingly.

- **Medium | Confidence: Medium**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/pinn.py:367`
  - Issue: Deflection model hardcodes concentration `c_nfw = 10.0`, decoupling inferred `(M_vir, r_s)` from profile shape in physics residual.
  - Evidence: Code does not infer/derive concentration from predicted params.
  - Minimal fix direction: Propagate concentration consistently from predicted parameters or add it as modeled variable.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/time_delay/cosmography.py:330`
  - Issue: Bare `except:` in uncertainty-fit fallback hides numerical/logic errors and silently substitutes heuristic uncertainty.
  - Evidence: Static scan found bare exception; pathway affects scientific uncertainty reporting.
  - Minimal fix direction: Catch explicit exception types and log failure metadata for reproducibility.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/hst_targets.py:171`
  - Issue: Core data loader path is stand-in simulation, not archive-backed ingestion; can invalidate claims of real-data validation if treated as observational truth.
  - Evidence: `download_hst_data` warns and generates synthetic surrogates.
  - Minimal fix direction: Gate stand-in mode explicitly and require real-data path for validation benchmarks.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/scientific_validator.py:792`
  - Issue: Bare `except:` in power-law fitting returns `0.0`, silently converting fit failures into physically meaningful-looking outputs.
  - Evidence: Found at both log-space and direct fit branches.
  - Minimal fix direction: Catch specific numerical exceptions and propagate fit-failure status in metrics instead of silent zeros.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_api_security.py:19`
  - Issue: Test suite depends on externally running server at `http://localhost:8000`, making it non-self-contained and non-reproducible in CI/local defaults.
  - Evidence: Multiple failures show `ConnectionRefusedError` without server fixture startup.
  - Minimal fix direction: Use FastAPI `TestClient`/fixtures or launch app in test fixture and teardown deterministically.

- **Medium | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_time_delay.py:220`
  - Issue: Uses `np.trapezoid`, incompatible with NumPy 1.x baseline used by project.
  - Evidence: Will fail under NumPy 1.26 similarly to production code path.
  - Minimal fix direction: Use `np.trapz` or shared compatibility helper.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/main.py:236`
  - Issue: Version contract inconsistency: root endpoint reports `1.0.0` while app metadata uses `2.0.0`.
  - Evidence: `api/main.py` sets FastAPI version `2.0.0` but root response hardcodes `1.0.0`.
  - Minimal fix direction: Use one canonical version source for all endpoints.

- **Low | Confidence: Medium**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/03_PINN_Inference.py:1`
  - Issue: UTF-8 BOM present, which can break strict parser/tooling pipelines.
  - Evidence: File starts with bytes `EF BB BF`.
  - Minimal fix direction: Save file as UTF-8 without BOM for toolchain compatibility.

- **Low | Confidence: Medium**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/04_Multi_Plane.py:1`
  - Issue: UTF-8 BOM present, which can break strict parser/tooling pipelines.
  - Evidence: File starts with bytes `EF BB BF`.
  - Minimal fix direction: Save file as UTF-8 without BOM for toolchain compatibility.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase1_demo.ipynb:30`
  - Issue: Uses `sys.path.append("..")`, making execution dependent on notebook launch location.
  - Evidence: Path mutation observed in code cell import setup.
  - Minimal fix direction: Use installable package or explicit project-root resolver helper.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase4_time_delay_demo.ipynb:51`
  - Issue: Uses `sys.path.append("..")`, reducing reproducibility across environments.
  - Evidence: Path mutation observed in code cell import setup.
  - Minimal fix direction: Adopt stable package import strategy without runtime path surgery.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase5a_generate_data.ipynb:385`
  - Issue: Uses `sys.path.append("..")`, introducing environment-dependent imports.
  - Evidence: Path mutation observed in notebook cell.
  - Minimal fix direction: Replace with package install/import convention.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase5c_evaluate.ipynb:413`
  - Issue: Uses `sys.path.append("..")`, introducing non-portable execution behavior.
  - Evidence: Path mutation observed in notebook cell.
  - Minimal fix direction: Replace with package install/import convention.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase5d_advanced_training.ipynb:407`
  - Issue: Uses `sys.path.append("..")`, introducing non-portable execution behavior.
  - Evidence: Path mutation observed in notebook cell.
  - Minimal fix direction: Replace with package install/import convention.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/check_imports.py:1`
  - Issue: File is empty; advertised import-check utility is non-functional.
  - Evidence: `wc -l` returns 0 lines.
  - Minimal fix direction: Implement script logic or remove dead script from workflow/documentation.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_api.py:270`
  - Issue: Assertion accepts both success and internal error (`status in [200,500]`), masking regressions.
  - Evidence: Test passes even when endpoint crashes.
  - Minimal fix direction: Tighten expected status and assert structured error behavior separately.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_ray_tracing.py:139`
  - Issue: Tautological assertion `len(images) >= 0` provides no verification.
  - Evidence: This condition is always true.
  - Minimal fix direction: Assert scientifically meaningful bounds or image-count expectations by regime.

- **Low | Confidence: High**
  - File: `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_scientific_validation.py:255`
  - Issue: `assert True` stand-in does not validate profiler behavior.
  - Evidence: No behavioral check beyond non-crash.
  - Minimal fix direction: Replace with explicit assertions on measured outputs/invariants.

## Per-File Coverage Ledger

### Files With Findings

- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/auth_routes.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/main.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/03_PINN_Inference.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/04_Multi_Plane.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/demo_helpers.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/comparisons.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/runner.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/visualization.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/database/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase1_demo.ipynb`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase3_dm_comparison.ipynb`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase4_time_delay_demo.ipynb`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase5a_generate_data.ipynb`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase5c_evaluate.ipynb`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase5d_advanced_training.ipynb`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/check_imports.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/test_validator.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/dark_matter/substructure.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/mass_profiles.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/generate_dataset.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/pinn.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/time_delay/cosmography.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/utils/visualization.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/calibration.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/hst_targets.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/scientific_validator.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_api.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_api_security.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_multi_plane_recursive.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_physics_constrained_loss.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_ray_tracing.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_ray_tracing_backends.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_scientific_validation.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_time_delay.py`

### Reviewed: No Scientific/Rigor Defects Found

- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/analysis_routes.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/monitoring.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/secure_logging.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/api/security_utils.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/Home.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/error_handler.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/main.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/02_Simple_Lensing.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/03_Results.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/05_Real_Data.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/06_Training.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/07_Validation.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/08_Bayesian_UQ.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/pages/09_Settings.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/styles.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/helpers.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/plotting.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/session_state.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/app/utils/ui.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/metrics.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/pinn_inference.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/benchmarks/profiler.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/database/auth.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/database/crud.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/database/database.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/database/models.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/demo_wave_optics.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/migrations/env.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/notebooks/phase2_wave_demo.ipynb`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/benchmark_phase7.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/check_config.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/init_db.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/integrate_validator.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/quick_demo.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/test_bayesian_uq.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/scripts/test_real_data.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/api_utils/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/api_utils/auth.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/data/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/data/real_data_loader.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/advanced_profiles.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/lens_system.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/multi_plane.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/lens_models/multi_plane_recursive.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/augmentation.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/evaluate.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/performance.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/physics_constrained_loss.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/physics_unit_safe.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/pinn_advanced.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/pinn_models.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/tensorboard_logger.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/train_pinn.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/transfer_learning.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/uncertainty/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/ml/uncertainty/bayesian_uq.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/optics/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/optics/geodesic_integration.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/optics/ray_tracing.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/optics/ray_tracing_backends.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/optics/wave_optics.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/time_delay/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/utils/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/utils/common.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/utils/constants.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/src/validation/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/__init__.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_advanced_profiles.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_alternative_dm.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_api_security_integration.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_calibration.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_database_crud.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_lens_system.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_mass_profiles.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_ml.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_performance.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_pinn_adaptive.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_pinn_physics.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_ray_tracing_modes.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_real_data.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_transfer_learning.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_wave_optics.py`
- `/Users/nalinaggarwal/Downloads/Gravitational-Lensing-algorithm-master/tests/test_web_interface.py`

## Reproduction Commands Used

- `python3 -m py_compile $(rg --files -g "*.py")`
- `python3 -m pytest -q`
- `python3 -m pytest -q tests/test_time_delay.py -vv`
- `python3 -m pytest -q tests/test_alternative_dm.py -vv`
- `python3 -m pytest -q tests/test_calibration.py -vv`
- `python3 -m pytest -q tests/test_ml.py -vv`
- `python3 -m pytest -q tests/test_api_security.py -vv`
- `python3 -m pytest -q tests/test_api.py -vv`
- `python3 - <<...import matrix checks...>>`
- `python3 - <<...notebook cell scan...>>`

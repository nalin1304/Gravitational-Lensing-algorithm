# IEEE Submission Checklist (Evidence-Driven)

Last updated: 2026-03-11

This checklist is designed for Tier-1 journal expectations (e.g., IEEE TCI, MNRAS):
scientific correctness, reproducibility, traceability, and artifact integrity.

## 1. Mandatory Reproducibility Gate

Run the publication gate and archive the report:

```bash
python3 scripts/publication_gate.py
```

Required result:
- `Publication Gate: PASS`
- Report written to `results/publication_gate_report.json`

This gate verifies:
1. Required artifact files exist.
2. OpenAPI schema is available and contains required routes.
3. UI/API smoke tests pass.
4. Static analysis for `src/` passes (`mypy` when installed; otherwise syntax sweep recorded by the publication gate).
5. Known-system validation script passes.
6. Statistical rigor report is generated.
7. Full regression suite passes (current verified baseline: 513 passed, 38 skipped).
8. Calibration quality bounds are satisfied:
   - max raw Einstein-radius error <= 10%
   - max radius calibration factor <= 1.5
   - max mass calibration factor <= 2.0

## 2. Scientific Correctness Gate

Confirm core physics relations are implemented and tested:
1. Thin-lens equation: `beta = theta - alpha(theta)`
2. Critical surface density: `Sigma_crit = c^2/(4*pi*G) * D_s/(D_l*D_ls)`
3. Einstein scale: `theta_E = sqrt((4GM/c^2) * D_ls/(D_l*D_s))`
4. Time delay relation includes `(1+z_l)` factor.
5. Cosmological distances use `angular_diameter_distance_z1z2` for `D_ls`.
6. NFW `rho_crit` uses `critical_density(z_l)` at lens redshift (M200c convention).
7. Multi-plane ray-tracing uses proper recurrence: `theta_{i+1} = theta_i - (D_{i,i+1}/D_{i+1}) alpha_i`.

Equation provenance verification:
- `pinn.py`: Poisson constraint ∇²ψ = 2κ — Schneider (1992), Eq. 3.11
- `mass_profiles.py`: NFW(1997) Eq. 1, Wright & Brainerd (2000) Eq. 11–13
- `multi_plane.py`: Schneider (1992) Eq. 9.1–9.3, 9.15, 4.14
- `neural_ode.py`: Chen (2018), Cranmer (2020), Greydanus (2019)
- `wave_optics.py`: Nakamura & Deguchi (1999) Eq. 4.2, Takahashi & Nakamura (2003) Eq. 3–5

Execution checks:

```bash
python3 -m pytest tests/test_lens_system.py tests/test_mass_profiles.py tests/test_time_delay.py -q
python3 -m pytest tests/test_ray_tracing_backends.py tests/test_multi_plane_recursive.py -q
python3 -m pytest tests/test_real_data.py -q
python3 -m pytest tests/test_neural_ode.py -q
python3 scripts/validate_known_systems.py
python3 scripts/statistical_rigor_report.py
```

Benchmark validation (generates manuscript tables/figures):

```bash
bash scripts/reproduce.sh
# Or individually:
python3 scripts/ablation_study.py --grid 64 --n-trials 3 --n-calibration 6 --systems-per-trial 8
python3 scripts/validate_real_data.py --grid 64 --use-real --strict-observational
python3 scripts/sota_comparison.py --grid 64 --n-lenses 10 --n-calibration 6
python3 scripts/uncertainty_calibration.py --grid 64 --n-samples 30 --model models/bayesian_uq_synthetic.pt --seed 21 --dropout-rate 0.04
python3 scripts/scalability_benchmark.py
python3 scripts/pareto_benchmark.py --outdir results
python3 scripts/multi_messenger_demo.py --outdir results
```

## 3. Software Quality Gate

```bash
uv run python -m pytest tests/ -q
python3 -c "from src.ml import check_backend; check_backend()"
python3 -m mypy src/ --ignore-missing-imports  # when installed
```

Required result:
- `pytest`: 513 passed, 38 skipped
- `mypy`: success when the tool is installed; otherwise the publication gate must record a clean `py_compile` syntax sweep
- `check_backend()`: reports detected backend

Hardware-agnostic verification:
- All `src/ml/` imports are guarded with `try/except ImportError`
- `BACKEND` in `{jax, numpy, unavailable}` — no import-time crashes

## 4. Determinism and Reproducibility Gate

Functional Randomness Control verification:
- All JAX models use explicit `seed: int` parameters (no hidden `PRNGKey(0)`)
- NFW subhalo generation uses `np.random.default_rng(hashlib.sha256(params))`
- MAST downloader uses `hashlib.sha256(name)` for noise generation
- Bit-for-bit reproducibility when seeds are fixed

```bash
# Verify determinism (same output twice)
python3 scripts/pareto_benchmark.py --outdir /tmp/test1 --seed 42
python3 scripts/pareto_benchmark.py --outdir /tmp/test2 --seed 42
diff /tmp/test1/pareto_data.json /tmp/test2/pareto_data.json  # Should be identical
```

## 5. Artifact Traceability Gate

Before manuscript submission, ensure each figure/table points to:
1. Exact script/notebook path used to generate it.
2. Exact commit hash.
3. Exact environment specification (`requirements*.txt`, lock file if used).
4. Output artifact path under `results/` or `output/`.

Minimum required metadata files:
- `README.md`
- `AGENTS.md`
- `CITATION.cff`
- `JOURNAL_PUBLICATION_READINESS.md`
- `paper/TIER1_TOPIC_AND_RIGOR.md`

## 6. Required Disclosure Items in Manuscript

Include explicit statements for:
1. Cosmology defaults (`H0=67.4`, `Om0=0.315`) and where configurable.
2. Regime assumptions (thin-lens vs strong-field geodesic usage).
3. Validation protocol (known systems + synthetic controls).
4. Determinism controls (seed policy, stochastic components).
5. Hardware/software environment for reported performance.
6. Availability gating behavior for optional backends and checkpoints (unsupported features must report unavailable state rather than substitute heuristic outputs).
7. Mass-sheet degeneracy handling (kinematics constraint, λ parameter).

## 7. Bayesian Model Selection Evidence

New capability for formal model comparison:
1. Nested sampling engine: `src/ml/nested_sampling.py`
2. Bayes factor computation with Jeffreys scale
3. Built-in NFW vs SIS demo (ln K ≈ 13.9, decisive)
4. Manuscript should cite Skilling (2004) and include ln Z ± σ(ln Z) values

## 8. Current High-Impact Risks (to disclose or close)

1. Mass recovery still relies on calibration for best agreement on all systems.
   - Keep manuscript language explicit about raw vs calibrated metrics.
2. Throughput claims are hardware-dependent.
   - Report benchmark command and hardware profile with each number.
3. Optional acceleration backends may be unavailable in minimal CPU setups.
   - Document: `from src.ml import BACKEND; print(BACKEND)` and the exact capabilities enabled in the archived environment.
4. Uncertainty calibration is now checkpoint-backed, but its validated scope is held-out synthetic NFW analogs rather than observational posterior calibration.
5. Multi-plane convergence uses weak-coupling approximation (not fully coupled Jacobian).
   - Disclose limitation in manuscript.

## 9. Recommended Pre-Submission Freeze Sequence

1. Run: `python3 scripts/publication_gate.py`
2. Run: `bash scripts/reproduce.sh` (generates all benchmark outputs)
3. Run: `python3 scripts/pareto_benchmark.py` (Pareto front figure + LaTeX table)
4. Run: `python3 scripts/multi_messenger_demo.py` (multi-messenger consistency figure)
5. Archive: `results/publication_gate_report.json`
6. Verify all results artifacts exist:
   - `results/uncertainty_calibration.png`
   - `results/real_data/radial_profile_comparison.png`
   - `results/pareto_front.png`
   - `results/pareto_table.tex`
   - `results/multi_messenger_consistency.png`
   - `results/multi_messenger_data.json`
7. Regenerate manuscript figures from clean environment.
8. Verify manuscript values against current outputs.
9. Tag release and create archival snapshot (e.g., Zenodo integration with `scripts/mint_zenodo_doi.py`).
10. Submit with citation metadata (`CITATION.cff`) and reproducibility appendix.

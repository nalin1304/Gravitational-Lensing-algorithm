"""Checkpoint-backed component study for convergence-map reconstruction.

This script replaces the earlier proxy ablation with evidence-backed rows:

1. Full Pipeline: released full checkpoint with affine decoder calibration.
2. Vanilla PINN (No Physics/UQ): released ablated checkpoint.
3. No Calibration Layer: full checkpoint decoder used without affine
   calibration, isolating the importance of the calibration step.
4. Parametric NFW Refit: analytic NFW fit to the same map.
5. SIE Approximation: analytic SIE-like fit with profile mismatch.

Every row is produced by direct execution of a checkpoint or explicit analytic
fit. No controlled-noise surrogate is used.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

mpl_cache_dir = Path(tempfile.gettempdir()) / "gravitational_lensing_matplotlib"
mpl_cache_dir.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache_dir))

from src.lens_models.lens_system import LensSystem
from src.lens_models.mass_profiles import NFWProfile
from src.ml.checkpoint_benchmarks import (
    DEFAULT_ABLATION_CHECKPOINTS,
    fit_affine_decoder_calibration,
    fit_parametric_nfw_profile,
    fit_sie_like_profile,
    generate_sie_like_convergence,
    infer_decoder_output,
    load_ablation_checkpoint_model,
    predict_calibrated_decoder_map,
)
from src.ml.generate_dataset import generate_convergence_map_vectorized
from src.validation import ScientificValidator, ValidationLevel


@dataclass(frozen=True)
class BenchmarkCase:
    """One synthetic evaluation system."""

    system_id: int
    mass_msun: float
    concentration: float
    z_lens: float
    z_source: float
    extent_arcsec: float
    ground_truth: np.ndarray


@dataclass(frozen=True)
class AblationConfig:
    """One evidence-backed comparison row."""

    name: str
    description: str


@dataclass
class AblationResult:
    """Aggregate metrics for one config within one trial."""

    config_name: str
    rmse: float
    mae: float
    ssim: float
    psnr: float
    mass_conservation: float
    gradient_error: float
    elapsed_s: float
    passed: bool
    evaluation_mode: str
    inference_backend: str


ABLATION_CONFIGS = [
    AblationConfig("Full Pipeline", "Released checkpoint with affine decoder calibration"),
    AblationConfig("Vanilla PINN (No Physics/UQ)", "Released ablated checkpoint with the same calibration protocol"),
    AblationConfig("No Calibration Layer", "Full checkpoint decoder interpreted without affine calibration"),
    AblationConfig("Parametric NFW Refit", "Analytic NFW profile fit with known redshifts"),
    AblationConfig("SIE Approximation", "SIE-like isothermal baseline with profile mismatch"),
]


PASS_THRESHOLDS = {
    "rmse_max": 0.08,
    "ssim_min": 0.60,
    "mass_conservation_tolerance": 0.10,
}


def generate_benchmark_cases(
    n_cases: int,
    grid_size: int,
    rng: np.random.Generator,
) -> list[BenchmarkCase]:
    """Generate deterministic NFW benchmark cases."""
    cases: list[BenchmarkCase] = []
    for system_id in range(n_cases):
        mass_msun = float(10.0 ** rng.uniform(11.5, 13.4))
        concentration = float(rng.uniform(5.0, 14.0))
        z_lens = float(rng.uniform(0.15, 0.6))
        z_source = float(min(z_lens + rng.uniform(0.35, 1.6), 3.0))
        lens_system = LensSystem(z_lens=z_lens, z_source=z_source)
        extent_arcsec = float(max(2.0, 2.2 * lens_system.einstein_radius_scale(mass_msun)))
        lens_model = NFWProfile(
            M_vir=mass_msun,
            concentration=concentration,
            lens_system=lens_system,
        )
        ground_truth = generate_convergence_map_vectorized(
            lens_model=lens_model,
            grid_size=grid_size,
            extent=extent_arcsec,
        ).astype(np.float64)
        cases.append(
            BenchmarkCase(
                system_id=system_id,
                mass_msun=mass_msun,
                concentration=concentration,
                z_lens=z_lens,
                z_source=z_source,
                extent_arcsec=extent_arcsec,
                ground_truth=ground_truth,
            )
        )
    return cases


def _evaluate_trial_prediction(
    validator: ScientificValidator,
    predicted: np.ndarray,
    ground_truth: np.ndarray,
) -> dict[str, float]:
    result = validator.validate_convergence_map(
        predicted=predicted,
        ground_truth=ground_truth,
        profile_type="NFW",
        verbose=False,
    )
    return {
        "rmse": float(result.metrics.get("rmse", float("nan"))),
        "mae": float(result.metrics.get("mae", float("nan"))),
        "ssim": float(result.metrics.get("ssim", float("nan"))),
        "psnr": float(result.metrics.get("psnr", float("nan"))),
        "mass_conservation": float(result.metrics.get("mass_conservation_ratio", float("nan"))),
        "gradient_error": float(result.metrics.get("gradient_error", float("nan"))),
    }


def _passes_publication_thresholds(metrics: dict[str, float]) -> bool:
    return bool(
        metrics["rmse"] <= PASS_THRESHOLDS["rmse_max"]
        and metrics["ssim"] >= PASS_THRESHOLDS["ssim_min"]
        and abs(metrics["mass_conservation"] - 1.0) <= PASS_THRESHOLDS["mass_conservation_tolerance"]
    )


def _build_trial_methods(
    calibration_cases: list[BenchmarkCase],
) -> dict[str, Callable[[BenchmarkCase], tuple[np.ndarray, float, str, str]]]:
    full_model, full_device = load_ablation_checkpoint_model(DEFAULT_ABLATION_CHECKPOINTS["full"])
    vanilla_model, vanilla_device = load_ablation_checkpoint_model(DEFAULT_ABLATION_CHECKPOINTS["vanilla"])
    full_calibration = fit_affine_decoder_calibration(
        model=full_model,
        calibration_maps=[case.ground_truth for case in calibration_cases],
        device=full_device,
    )
    vanilla_calibration = fit_affine_decoder_calibration(
        model=vanilla_model,
        calibration_maps=[case.ground_truth for case in calibration_cases],
        device=vanilla_device,
    )

    def _full(case: BenchmarkCase) -> tuple[np.ndarray, float, str, str]:
        start = time.perf_counter()
        predicted = predict_calibrated_decoder_map(
            model=full_model,
            convergence_map=case.ground_truth,
            calibration=full_calibration,
            device=full_device,
        )
        elapsed = time.perf_counter() - start
        return predicted, elapsed, "checkpoint_backed_affine_decoder", str(DEFAULT_ABLATION_CHECKPOINTS["full"])

    def _vanilla(case: BenchmarkCase) -> tuple[np.ndarray, float, str, str]:
        start = time.perf_counter()
        predicted = predict_calibrated_decoder_map(
            model=vanilla_model,
            convergence_map=case.ground_truth,
            calibration=vanilla_calibration,
            device=vanilla_device,
        )
        elapsed = time.perf_counter() - start
        return predicted, elapsed, "checkpoint_backed_affine_decoder", str(DEFAULT_ABLATION_CHECKPOINTS["vanilla"])

    def _no_calibration(case: BenchmarkCase) -> tuple[np.ndarray, float, str, str]:
        start = time.perf_counter()
        raw_decoder = infer_decoder_output(
            model=full_model,
            convergence_map=case.ground_truth,
            device=full_device,
        )
        elapsed = time.perf_counter() - start
        return np.maximum(raw_decoder, 0.0), elapsed, "checkpoint_raw_decoder", str(DEFAULT_ABLATION_CHECKPOINTS["full"])

    def _nfw_refit(case: BenchmarkCase) -> tuple[np.ndarray, float, str, str]:
        start = time.perf_counter()
        fit_result = fit_parametric_nfw_profile(
            convergence_map=case.ground_truth,
            z_lens=case.z_lens,
            z_source=case.z_source,
            extent_arcsec=case.extent_arcsec,
        )
        lens_system = LensSystem(z_lens=case.z_lens, z_source=case.z_source)
        fitted_lens = NFWProfile(
            M_vir=fit_result.mass_msun,
            concentration=fit_result.concentration,
            lens_system=lens_system,
        )
        predicted = generate_convergence_map_vectorized(
            lens_model=fitted_lens,
            grid_size=case.ground_truth.shape[0],
            extent=case.extent_arcsec,
        )
        elapsed = time.perf_counter() - start
        return predicted, elapsed, "analytic_profile_refit", "coarse_to_fine_nfw_grid_search"

    def _sie(case: BenchmarkCase) -> tuple[np.ndarray, float, str, str]:
        start = time.perf_counter()
        fit_result = fit_sie_like_profile(
            convergence_map=case.ground_truth,
            extent_arcsec=case.extent_arcsec,
        )
        predicted = generate_sie_like_convergence(
            grid_size=case.ground_truth.shape[0],
            extent_arcsec=case.extent_arcsec,
            einstein_radius_arcsec=fit_result.einstein_radius_arcsec,
            axis_ratio=fit_result.axis_ratio,
            position_angle_deg=fit_result.position_angle_deg,
        )
        elapsed = time.perf_counter() - start
        return predicted, elapsed, "analytic_profile_refit", "sie_like_moment_fit"

    return {
        "Full Pipeline": _full,
        "Vanilla PINN (No Physics/UQ)": _vanilla,
        "No Calibration Layer": _no_calibration,
        "Parametric NFW Refit": _nfw_refit,
        "SIE Approximation": _sie,
    }


def run_single_ablation(
    config: AblationConfig,
    method: Callable[[BenchmarkCase], tuple[np.ndarray, float, str, str]],
    test_cases: list[BenchmarkCase],
) -> AblationResult:
    """Evaluate one evidence-backed config on one trial."""
    validator = ScientificValidator(level=ValidationLevel.RIGOROUS)
    metric_store: dict[str, list[float]] = {
        "rmse": [],
        "mae": [],
        "ssim": [],
        "psnr": [],
        "mass_conservation": [],
        "gradient_error": [],
    }
    elapsed_values: list[float] = []
    evaluation_mode = getattr(method, 'evaluation_mode', 'unknown')
    inference_backend = getattr(method, 'inference_backend', 'unknown')

    for case in test_cases:
        predicted, elapsed, evaluation_mode, inference_backend = method(case)
        metrics = _evaluate_trial_prediction(
            validator=validator,
            predicted=predicted,
            ground_truth=case.ground_truth,
        )
        for metric_name, metric_value in metrics.items():
            metric_store[metric_name].append(metric_value)
        elapsed_values.append(float(elapsed))

    mean_metrics = {name: float(np.mean(values)) for name, values in metric_store.items()}
    return AblationResult(
        config_name=config.name,
        rmse=mean_metrics["rmse"],
        mae=mean_metrics["mae"],
        ssim=mean_metrics["ssim"],
        psnr=mean_metrics["psnr"],
        mass_conservation=mean_metrics["mass_conservation"],
        gradient_error=mean_metrics["gradient_error"],
        elapsed_s=float(np.mean(elapsed_values)),
        passed=_passes_publication_thresholds(mean_metrics),
        evaluation_mode=evaluation_mode,
        inference_backend=inference_backend,
    )


def run_ablation_suite(
    grid_size: int = 64,
    n_trials: int = 5,
    seed: int = 42,
    n_calibration: int = 6,
    n_systems_per_trial: int = 8,
) -> dict[str, list[AblationResult]]:
    """Run the evidence-backed component study."""
    results: dict[str, list[AblationResult]] = {config.name: [] for config in ABLATION_CONFIGS}
    master_rng = np.random.default_rng(seed)

    print("\n" + "=" * 80)
    print("  ABLATION STUDY — Checkpoint-backed Component Comparison")
    print("=" * 80)
    print("  Mode: released checkpoints + analytic baselines")
    print(f"  Grid size: {grid_size}×{grid_size}")
    print(f"  Trials per config: {n_trials}")
    print(f"  Calibration cases per trial: {n_calibration}")
    print(f"  Test systems per trial: {n_systems_per_trial}")
    print("=" * 80)

    for trial_index in range(n_trials):
        trial_seed = int(master_rng.integers(0, 2**31 - 1))
        trial_rng = np.random.default_rng(trial_seed)
        calibration_cases = generate_benchmark_cases(n_calibration, grid_size, trial_rng)
        test_cases = generate_benchmark_cases(n_systems_per_trial, grid_size, trial_rng)
        methods = _build_trial_methods(calibration_cases)

        print(f"\nTrial {trial_index + 1}/{n_trials}")
        for config in ABLATION_CONFIGS:
            print(f"▶ {config.name} — {config.description}")
            result = run_single_ablation(config, methods[config.name], test_cases)
            results[config.name].append(result)
            status = "PASS" if result.passed else "FAIL"
            print(
                f"  RMSE={result.rmse:.6f} SSIM={result.ssim:.4f} "
                f"PSNR={result.psnr:.2f}dB Pass={status}"
            )

    return results


def compute_summary_stats(results: dict[str, list[AblationResult]]) -> list[dict[str, object]]:
    """Compute mean and standard deviation for each evidence-backed row."""
    summary: list[dict[str, object]] = []
    for config_name, trials in results.items():
        row: dict[str, object] = {"config": config_name}
        for metric_name in ("rmse", "mae", "ssim", "psnr", "mass_conservation", "gradient_error", "elapsed_s"):
            values = [float(getattr(trial, metric_name)) for trial in trials]
            row[f"{metric_name}_mean"] = float(np.mean(values))
            row[f"{metric_name}_std"] = float(np.std(values))
        row["pass_rate"] = float(np.mean([1.0 if trial.passed else 0.0 for trial in trials]))
        row["evaluation_mode"] = trials[0].evaluation_mode
        row["inference_backend"] = trials[0].inference_backend
        summary.append(row)
    return summary


def generate_latex_table(summary: list[dict[str, object]], output_path: Path) -> None:
    """Generate a publication-ready LaTeX table."""
    lines = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Checkpoint-backed component study for convergence-map reconstruction}",
        r"\label{tab:ablation}",
        r"\small",
        r"\begin{tabular}{l|cccc|c}",
        r"\toprule",
        r"\textbf{Configuration} & \textbf{RMSE}$\downarrow$ & \textbf{MAE}$\downarrow$ "
        r"& \textbf{SSIM}$\uparrow$ & \textbf{PSNR}$\uparrow$ & \textbf{Pass} \\",
        r"\midrule",
    ]
    for row in summary:
        config_name = str(row["config"])
        prefix = r"\textbf{" if config_name == "Full Pipeline" else ""
        suffix = "}" if config_name == "Full Pipeline" else ""
        lines.append(
            f"  {prefix}{config_name}{suffix} & "
            f"{row['rmse_mean']:.4f} ± {row['rmse_std']:.4f} & "
            f"{row['mae_mean']:.4f} ± {row['mae_std']:.4f} & "
            f"{row['ssim_mean']:.4f} ± {row['ssim_std']:.4f} & "
            f"{row['psnr_mean']:.1f} ± {row['psnr_std']:.1f} & "
            f"{row['pass_rate']:.0%} \\\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{1mm}",
            r"\parbox{\columnwidth}{\footnotesize "
            r"All rows are executed directly. The neural rows use released checkpoints, while the physics baselines "
            r"use explicit analytic profile fitting with known redshifts.}",
            r"\end{table}",
        ]
    )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"\nLaTeX table written to: {output_path}")


def generate_console_table(summary: list[dict[str, object]]) -> None:
    """Print a compact console summary."""
    print("\n" + "=" * 104)
    print(
        f"{'Configuration':<30} {'RMSE':>12} {'MAE':>12} {'SSIM':>12} "
        f"{'PSNR (dB)':>12} {'Pass Rate':>10}"
    )
    print("-" * 104)
    for row in summary:
        marker = "★" if row["config"] == "Full Pipeline" else " "
        print(
            f"{marker} {row['config']:<28} "
            f"{row['rmse_mean']:.4f}±{row['rmse_std']:.4f}  "
            f"{row['mae_mean']:.4f}±{row['mae_std']:.4f}  "
            f"{row['ssim_mean']:.4f}±{row['ssim_std']:.4f}  "
            f"{row['psnr_mean']:.1f}±{row['psnr_std']:.1f}  "
            f"{row['pass_rate']:.0%}"
        )
    print("=" * 104)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=64, help="Grid size.")
    parser.add_argument("--n-trials", type=int, default=5, help="Independent trial count.")
    parser.add_argument("--n-calibration", type=int, default=6, help="Calibration systems per trial.")
    parser.add_argument("--systems-per-trial", type=int, default=8, help="Evaluation systems per trial.")
    parser.add_argument("--outdir", type=str, default="results", help="Output directory.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    if args.n_trials < 1:
        parser.error("--n-trials must be >= 1")
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    results = run_ablation_suite(
        grid_size=args.grid,
        n_trials=args.n_trials,
        seed=args.seed,
        n_calibration=args.n_calibration,
        n_systems_per_trial=args.systems_per_trial,
    )
    summary = compute_summary_stats(results)

    generate_console_table(summary)
    generate_latex_table(summary, outdir / "ablation_table.tex")
    json_path = outdir / "ablation_results.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"JSON results written to: {json_path}")

    print("\n" + "=" * 80)
    print("  KEY FINDINGS")
    print("=" * 80)
    full_row = next(row for row in summary if row["config"] == "Full Pipeline")
    full_rmse = float(full_row["rmse_mean"])
    for row in summary:
        if row["config"] == "Full Pipeline":
            continue
        rmse_delta = float(row["rmse_mean"]) - full_rmse
        direction = "↑" if rmse_delta > 0 else "↓"
        percent = (rmse_delta / full_rmse) * 100.0 if full_rmse > 0 else float("nan")
        print(f"  {row['config']:<30} RMSE Δ = {rmse_delta:+.6f} ({direction}{abs(percent):.1f}%)")
    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

